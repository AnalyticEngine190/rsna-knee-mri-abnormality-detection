#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSNA 4-family OOF ensemble: strict audit -> full-fold calibration -> multi-res comparison
-> diversity audit -> nested LOFO weight selection -> atomic export for Kaggle 2xT4 + Gen-2.

Pure numpy / pandas / scipy.  NO torch, NO GPU, NO model loading.

v2.7 changes (on top of v2.6 hardening)
---------------------------------------
FIXES REQUESTED
 1. SyntaxError fixed: the NPZ block of load_phase3_gold_oof() now lives inside `for zname in zips:`
    / `with zipfile.ZipFile(...)`; the local-CSV fallback stays at function scope.
 2. REQUIRE_PATIENT_ID is now fail-closed by default (env RSNA_REQUIRE_PATIENT_ID=0 to override).
 3. Embedded-label aliases now include "targets"/"target" (with a guard so an array of target
    *names* stored under that key is not mistaken for a label matrix).
 4. STRICTLY NESTED CALIBRATION: z-score statistics, ECDF references, performance weights and the
    co-occurrence prior are re-fit from the *training folds only* inside every inner/outer split;
    the held-out fold is only transformed with those references.  Deployment calibration (full OOF
    cohort) is kept separate.  Scheme E's P3 logits are now calibrated with the same P2 reference
    as the rest of the ensemble (previously fold-local stats + gold-only ranks).
 5. Gold-count reconciliation is enforced: supply RSNA_GOLD_UID_FILE (authoritative list) or
    explicitly acknowledge the count with RSNA_ACK_GOLD_COUNT=57.  N_GOLD is NOT changed; missing
    studies are listed with their fold (or "not in fold map").

EXTRA IDEAS ADDED (each marked "[v2.7 ADD]" in the code)
 - Study-level bootstrap CIs for the nested score + paired comparison against uniform averaging.
 - Label-permutation canary: re-runs the whole nested pipeline on shuffled gold labels; the result
   should be ~0.5.  Materially higher => leakage somewhere in the pipeline.
 - Gold label-support table (positives per fold x target) so undefined fold AUCs are visible.
 - SHA-256 provenance of every input file written into ensemble_config.json.
 - Hash-verified copies to Drive + timestamped backup of any file that is overwritten.
 - NPZ is loaded with allow_pickle=False first; pickle only if the archive needs it.
 - Cross-stream label consistency check inside the NPZ parser; dead helper functions removed;
   a missing f-string prefix in an error message fixed.

This script can validate file alignment and CV logic, but it cannot prove upstream model checkpoints were
trained without a study unless the upstream OOF-generation provenance is trustworthy.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import shutil
import time
import warnings
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.special import logit as sp_logit
from scipy.stats import rankdata


def env_flag(name, default=True):
    """Boolean env var; '0/false/no/off' => False, anything else => True."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# =====================================================================================
# 0. CONFIG
# =====================================================================================
SCRIPT_VERSION = "2.7.0-nested-calibration"

P2_DIR = os.environ.get("RSNA_P2_DIR", "/content/drive/MyDrive/RSNA_Phase2_Backups")
P3_DIR = os.environ.get("RSNA_P3_DIR", "/content/drive/MyDrive/RSNA_Phase3_Backups")
LOCAL_P2_MODELS_DIR = "/content/phase2_models"
LOCAL_P3_MODELS_DIR = "/content/phase3_models"
LOCAL_OUT_DIR = os.environ.get("RSNA_LOCAL_OUT", "/content/ensemble_artifacts")

FOLDS_CSV_CANDIDATES = [
    "/content/phase2_folds.csv",
    os.path.join(P2_DIR, "phase2_folds.csv"),
    os.path.join(P3_DIR, "phase2_folds.csv"),
]
FUSED_MASTER_CANDIDATES = [
    "/content/phase2_fused_master.csv",
    os.path.join(P2_DIR, "phase2_fused_master.csv"),
    os.path.join(P3_DIR, "phase2_fused_master.csv"),
]

BACKBONES = [
    dict(name="convnext_small", short="convnext", family="CNN", p2_res=320, p3_res=384),
    dict(name="resnet50d", short="resnet50d", family="CNN", p2_res=320, p3_res=384),
    dict(name="eca_nfnet_l0", short="nfnet_l0", family="CNN", p2_res=320, p3_res=384),
    dict(name="vit_small_patch14_dinov2.lvd142m", short="dinov2", family="ViT", p2_res=336, p3_res=392),
]
for _b in BACKBONES:
    _b["stem"] = _b["name"].split(".")[0]

TARGET_COLS = globals().get("TARGET_COLS", [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
])
_EXT_GOLD_MASK = globals().get("gold_row_mask", None)

N_P2_ROWS = 4405
N_GOLD = 57  # Must match the gold UID cohort in the Phase-3 OOF archives. See reconcile_gold_cohort().
N_FOLDS = 5
EXPECTED_GOLD_FOLD_COUNTS = [14, 12, 13, 12, 6]  # Sum must equal N_GOLD.
GOLD_WEIGHT_THRESHOLD = 5.0
EPS_Z = 1e-6
LOGIT_CLIP = 1e-6

# One score scale across folds. With v2.7 the *reference statistics* for that global scale are
# re-fit from training folds inside each validation split (see Ensembler.cal), so "global" no
# longer means "fit on every OOF row including the held-out fold" during validation.
CALIBRATION_SCOPE = "global"

STRICT_P2_GOLD_ORDER = True
REQUIRE_EXPLICIT_P3_FOLD = True
# [v2.7 FIX 2] Patient-level validation fails closed by default, as the comments always claimed.
# Set RSNA_REQUIRE_PATIENT_ID=0 only after independently verifying patient-grouped folds.
REQUIRE_PATIENT_ID = env_flag("RSNA_REQUIRE_PATIENT_ID", True)
DEPLOYMENT_SCHEMES = ("A", "B", "C", "D")  # Scheme E only has P3 predictions for the gold studies.
STANDARDIZE_P3 = "p2_fold"       # Part-3 *diagnostics* only: "p2_fold" | "gold_pooled"

ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
GAMMA_GRID = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
TAU_GRID = [5.0, 10.0, 20.0, 40.0]
ETA_GRID = [0.0, 0.05, 0.10, 0.15]
BETA_P3_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
W_CO_RIDGE_REL = 0.10
W_CO_RESPONSE = "pred"           # "pred" | "pseudo_label"
PROMOTION_MARGIN = 0.001

GEN2_OLD_WEIGHT, GEN2_NEW_WEIGHT = 0.30, 0.70
CKPT_EXT = (".pth", ".pt", ".ckpt", ".safetensors", ".bin")

# [v2.7 ADD] uncertainty / canary / export-safety knobs
BOOTSTRAP_N = int(os.environ.get("RSNA_BOOTSTRAP_N", "500"))
BOOTSTRAP_SEED = 20260617
PERM_CANARY_N = int(os.environ.get("RSNA_PERM_CANARY", "3"))   # 0 disables
PERM_CANARY_WARN = 0.56          # nested macro AUC on shuffled labels above this => warn
KEEP_BACKUPS = env_flag("RSNA_KEEP_BACKUPS", True)


# =====================================================================================
# 1. SMALL UTILITIES
# =====================================================================================
class AuditError(RuntimeError):
    pass


def require(cond, msg):
    if not cond:
        raise AuditError("AUDIT FAILED: " + msg)


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def fmt(df, nd=4):
    return df.round(nd).to_string()


def macro_nanmean(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


def auc_cols(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    require(X.ndim >= 2, f"AUC scores must have at least 2 dimensions; got {X.shape}")
    require(Y.ndim == 2 and Y.shape[-1] == X.shape[-1] and Y.shape[0] == X.shape[-2],
            f"AUC shape mismatch: scores {X.shape}, labels {Y.shape}")
    require(np.isfinite(X).all(), "AUC scores contain NaN/inf")
    require(np.isfinite(Y).all() and np.isin(Y, [0.0, 1.0]).all(), "AUC labels must be finite binary 0/1")
    n = X.shape[-2]
    R = rankdata(X, axis=-2)
    npos = Y.sum(axis=0)
    nneg = n - npos
    num = (R * Y).sum(axis=-2) - npos * (npos + 1.0) / 2.0
    den = npos * nneg
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def fold_mean_auc_cols(scores, labels, fold_ids):
    """Per-target mean AUC across folds; avoids cross-fold score-scale comparisons."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    fold_ids = np.asarray(fold_ids, dtype=int)
    require(scores.ndim == 2 and labels.shape == scores.shape, "fold_mean_auc_cols shape mismatch")
    require(fold_ids.shape == (len(scores),), "fold_mean_auc_cols fold shape mismatch")
    per_fold = []
    for fold_id in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold_id
        if mask.any():
            per_fold.append(auc_cols(scores[mask], labels[mask]))
    require(bool(per_fold), "fold_mean_auc_cols received no folds")
    matrix = np.vstack(per_fold)
    result = np.full(matrix.shape[1], np.nan, dtype=float)
    for col in range(matrix.shape[1]):
        valid = matrix[:, col][np.isfinite(matrix[:, col])]
        if valid.size:
            result[col] = float(valid.mean())
    return result


def fold_mean_macro_auc(scores, labels, fold_ids):
    """Unweighted mean of per-fold macro AUCs (primary validation metric)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    fold_ids = np.asarray(fold_ids, dtype=int)
    require(scores.ndim == 2 and labels.shape == scores.shape, "fold_mean_macro_auc shape mismatch")
    require(fold_ids.shape == (len(scores),), "fold_mean_macro_auc fold shape mismatch")
    per_fold = []
    for fold_id in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold_id
        if mask.any():
            per_fold.append(macro_nanmean(auc_cols(scores[mask], labels[mask])))
    return macro_nanmean(per_fold)


def ecdf_from_sorted_reference(X, sorted_reference):
    X = np.asarray(X, dtype=float)
    ref = np.asarray(sorted_reference, dtype=float)
    require(X.ndim == 2 and ref.ndim == 2 and X.shape[1] == ref.shape[1],
            f"ECDF shape mismatch: X={X.shape}, reference={ref.shape}")
    require(np.isfinite(X).all() and np.isfinite(ref).all(), "ECDF inputs contain NaN/inf")
    out = np.empty_like(X, dtype=float)
    n = ref.shape[0]
    require(n > 0, "ECDF reference is empty")
    for c in range(X.shape[1]):
        rc = ref[:, c]
        require(np.all(rc[:-1] <= rc[1:]), f"ECDF reference column {c} is not sorted")
        left = np.searchsorted(rc, X[:, c], side="left")
        right = np.searchsorted(rc, X[:, c], side="right")
        avg_rank = ((left + 1.0) + right) / 2.0
        out[:, c] = avg_rank / (n + 1.0)
    return out


def _selftest_auc():
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return
    rng = np.random.default_rng(0)
    X = np.round(rng.normal(size=(60, 4)), 1)
    Y = (rng.random((60, 4)) < 0.3).astype(float)
    mine = auc_cols(X, Y)
    ref = np.array([roc_auc_score(Y[:, c], X[:, c]) for c in range(4)])
    assert np.allclose(mine, ref, atol=1e-12), "fast AUC disagrees with sklearn"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# [v2.7 ADD] SHA-256 helper: lets the exported config pin the exact inputs a run consumed.
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_csv(path, df):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_copy(src, dst, keep_backup=None):
    """Copy src -> dst atomically.

    [v2.7 ADD] (a) the temp copy is hash-verified against the source before it replaces dst, so a
    truncated Drive write cannot silently become the "final" file; (b) an existing dst is first
    preserved as dst.bak_<UTC timestamp>, because Gen-2 pseudo-label masters are expensive to
    regenerate.  Disable backups with RSNA_KEEP_BACKUPS=0.
    """
    keep_backup = KEEP_BACKUPS if keep_backup is None else keep_backup
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    src_hash, tmp_hash = sha256_file(src), sha256_file(tmp)
    if src_hash != tmp_hash:
        os.remove(tmp)
        raise AuditError(f"AUDIT FAILED: hash mismatch copying {src} -> {dst} (copy corrupted; nothing replaced)")
    if keep_backup and os.path.exists(dst):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copyfile(dst, f"{dst}.bak_{stamp}")
    os.replace(tmp, dst)
    return src_hash


def parse_bool(s, what="is_gold"):
    if pd.api.types.is_bool_dtype(s.dtype):
        require(s.notna().all(), f"column '{what}' contains missing boolean values")
        return s.astype(bool).values
    m = {"true": True, "false": False, "1": True, "0": False,
         "1.0": True, "0.0": False, "yes": True, "no": False}
    out = s.astype(str).str.strip().str.lower().map(m)
    require(out.notna().all(), f"column '{what}' contains values that are not boolean-like")
    return out.astype(bool).values


# =====================================================================================
# 2. DRIVE, LISTING, DISCOVERY
# =====================================================================================
def ensure_drive():
    if os.path.isdir(P2_DIR) and os.path.isdir(P3_DIR):
        return
    try:
        from google.colab import drive  # type: ignore
        if not os.path.isdir("/content/drive/MyDrive"):
            drive.mount("/content/drive")
    except ImportError:
        pass
    for d in (P2_DIR, P3_DIR):
        require(os.path.isdir(d), f"directory not found: {d}  (mount Drive or set RSNA_P2_DIR / RSNA_P3_DIR)")


def list_dir(d):
    print(f"\n[dir] {d}")
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isdir(p):
            print(f"        <dir>      {name}/")
            continue
        extra = ""
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(p) as zf:
                    extra = f"   (zip, {len(zf.namelist())} members)"
            except Exception as e:
                extra = f"   (UNREADABLE zip: {e})"
        print(f"   {os.path.getsize(p) / 1e6:10.2f} MB  {name}{extra}")


def first_existing(cands, what):
    for p in cands:
        if os.path.isfile(p):
            return p
    raise AuditError(f"AUDIT FAILED: could not find {what}. Looked in: {cands}")


def scan_ckpts(d, files, stem):
    found = []
    for f in sorted(files):
        if stem not in f:
            continue
        if f.lower().endswith(CKPT_EXT):
            found.append(f)
        elif f.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(os.path.join(d, f)) as zf:
                    found += [f"{f}::{m}" for m in zf.namelist() if m.lower().endswith(CKPT_EXT)]
            except Exception:
                pass
    return found


def candidate_p2_files(info, search_dirs):
    cands = []
    seen_names = set()
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        files = sorted(os.listdir(d))
        exact = f"oof_{info['name']}.csv"
        if exact in files and exact not in seen_names:
            cands.append(os.path.join(d, exact))
            seen_names.add(exact)
        pref = [
            f for f in files
            if f.startswith(f"oof_{info['stem']}") and f.lower().endswith(".csv")
            and f not in seen_names
            and not re.search(r"ensemble|master|4family|gen2|phase3|_meta", f)
        ]
        pref.sort(key=lambda f: (len(f), f))
        for f in pref:
            cands.append(os.path.join(d, f))
            seen_names.add(f)
    return cands


# =====================================================================================
# 3. LABEL / GOLD-MASK RESOLUTION
# =====================================================================================
def find_label_matrix(df, targets):
    prefixes = ["", "y_", "y__", "label_", "label__", "true_", "true__", "target_", "target__", "gt_", "gt__"]
    suffixes = ["", "_true", "_label", "_target", "_gt", "_y"]
    cols = []
    for c in targets:
        hit = None
        for p in prefixes:
            for s in suffixes:
                name = f"{p}{c}{s}"
                if name in df.columns:
                    hit = name
                    break
            if hit:
                break
        if hit is None:
            return None
        cols.append(hit)
    return np.column_stack([pd.to_numeric(df[c], errors="coerce").values.astype(float) for c in cols])


def resolve_gold_mask(df, label, expected_n=N_GOLD):
    """Resolve gold rows from an explicit boolean/status/weight column.

    Important: status parsing is exact, not substring-based. In particular,
    "non_gold_pseudolabel" must never be treated as gold merely because it
    contains the word "gold".
    """
    if _EXT_GOLD_MASK is not None:
        try:
            raw_mask = np.asarray(_EXT_GOLD_MASK(df))
            if raw_mask.ndim == 1 and len(raw_mask) == len(df):
                m = parse_bool(pd.Series(raw_mask), "gold_row_mask")
                if expected_n is None or int(m.sum()) == int(expected_n):
                    print(f"   [{label}] gold mask via notebook gold_row_mask(df): {int(m.sum())} rows")
                    return m
        except Exception as e:
            print(f"   [{label}] notebook gold_row_mask(df) rejected ({e}); checking explicit columns")

    col_map = {str(c).strip().lower(): c for c in df.columns}
    status_aliases = {
        "gold", "gold_human", "human_gold", "gold_label", "gold_labels",
        "expert_gold", "gold_expert", "human_expert_gold", "gold_human_label",
        "expert_gold_label", "gold_expert_label", "human_gold_label",
    }
    for low_name in ("is_gold", "gold", "is_gold_human"):
        if low_name in col_map:
            real_col = col_map[low_name]
            if df[real_col].notna().all():
                try:
                    m = parse_bool(df[real_col], real_col)
                except AuditError:
                    status_values = df[real_col].astype(str).str.strip().str.lower().str.replace(r"[ -]+", "_", regex=True)
                    m = status_values.isin(status_aliases).to_numpy(dtype=bool)
                if int(m.sum()) > 0 and (expected_n is None or int(m.sum()) == int(expected_n)):
                    print(f"   [{label}] gold mask via explicit column '{real_col}': {int(m.sum())} rows")
                    return m
                print(f"   [{label}] column '{real_col}' identified {int(m.sum())} gold rows; expected {expected_n}")
    for low_name in ("label_type", "source", "label_source", "src", "origin", "kind", "type", "label_origin"):
        if low_name in col_map:
            real_col = col_map[low_name]
            values = df[real_col].astype(str).str.strip().str.lower().str.replace(r"[ -]+", "_", regex=True)
            m = values.isin(status_aliases).to_numpy(dtype=bool)
            if int(m.sum()) > 0 and (expected_n is None or int(m.sum()) == int(expected_n)):
                print(f"   [{label}] gold mask via exact status values in '{real_col}': {int(m.sum())} rows")
                return m

    # Weights are a last-resort heuristic only when the expected gold count is known.
    # Never infer additional full-master gold rows from weights alone.
    if expected_n is not None and "sample_weight" in col_map:
        real_col = col_map["sample_weight"]
        w = pd.to_numeric(df[real_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        m = w >= GOLD_WEIGHT_THRESHOLD
        if int(m.sum()) > 0 and int(m.sum()) == int(expected_n):
            print(f"   [{label}] gold mask via '{real_col} >= {GOLD_WEIGHT_THRESHOLD}': {int(m.sum())} rows")
            return m

    raise AuditError(
        f"AUDIT FAILED: [{label}] cannot identify exactly {expected_n} gold rows from explicit metadata. "
        f"Columns: {list(df.columns)[:50]}"
    )


# -------------------------------------------------------------------------------------
# [v2.7 FIX 5] Gold-cohort reconciliation
# -------------------------------------------------------------------------------------
def load_authoritative_gold_uids(path):
    """Read an authoritative gold-UID list: CSV with a StudyInstanceUID-like column, or a text
    file with one UID per line ('#' comments allowed)."""
    require(os.path.isfile(path), f"RSNA_GOLD_UID_FILE not found: {path}")
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        col = next((c for c in ("StudyInstanceUID", "study_id", "study_uid", "uid") if c in df.columns), None)
        require(col is not None, f"{path}: needs a StudyInstanceUID (or study_id/study_uid/uid) column")
        raw = df[col].dropna().astype(str).tolist()
    else:
        with open(path, encoding="utf-8") as f:
            raw = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    uids = [u.strip() for u in raw if u.strip()]
    require(len(uids) > 0, f"{path}: no UIDs found")
    require(len(set(uids)) == len(uids), f"{path}: duplicate UIDs in authoritative gold list")
    return set(uids)


def resolve_gold_reference():
    """Fail closed unless the gold cohort size has been reconciled by the operator.

    Option 1 (preferred): RSNA_GOLD_UID_FILE=/path/to/gold_uids.(csv|txt)
    Option 2: RSNA_ACK_GOLD_COUNT=57  -- an explicit statement that you reconciled 57-vs-58 manually.
    """
    path = os.environ.get("RSNA_GOLD_UID_FILE", "").strip()
    if path:
        uids = load_authoritative_gold_uids(path)
        print(f"authoritative gold UID list: {path} ({len(uids)} UIDs)")
        return uids, path
    ack = os.environ.get("RSNA_ACK_GOLD_COUNT", "").strip()
    require(
        ack == str(N_GOLD),
        f"N_GOLD={N_GOLD} has not been reconciled against an authoritative gold-UID list (earlier context "
        "mentioned 58 expert-labelled studies). Set RSNA_GOLD_UID_FILE to the authoritative list, or, after "
        f"manually identifying the missing study and its fold, set RSNA_ACK_GOLD_COUNT={N_GOLD}.",
    )
    print(f"WARNING: gold count {N_GOLD} acknowledged via RSNA_ACK_GOLD_COUNT; no independent UID list was checked.")
    return None, None


def reconcile_gold_cohort(oof_gold_uids, folds_map, fused_raw, authoritative=None):
    """Compare the gold UIDs used for CV with an independent reference; name any missing study.

    Reference = authoritative UID list if supplied, else the gold flags of the FULL fused master
    (which includes rows later dropped from the 4,405-row OOF cohort).
    Fatal: a reference-gold study that IS in the OOF fold map but is not flagged gold (it would be
    treated as a pseudo-label row and overwritten in Gen-2), or any CV 'gold' study the reference
    does not know about.
    Non-fatal (loudly reported): reference-gold studies that were never part of the OOF cohort.
    """
    oof_gold = set(oof_gold_uids)
    master_gold = None
    try:
        m = resolve_gold_mask(fused_raw, "fused_master_full", expected_n=None)
        master_gold = set(fused_raw.loc[m, "StudyInstanceUID"])
    except AuditError as exc:
        print(f"   note: could not derive gold flags from the full fused master ({exc})")

    reference = authoritative if authoritative is not None else master_gold
    ref_name = "authoritative UID list" if authoritative is not None else "full fused-master gold flags"
    report = dict(reference=ref_name if reference is not None else None, n_cv_gold=len(oof_gold),
                  n_reference=None if reference is None else len(reference), missing=[], extra=[])
    if reference is None:
        print("WARNING: no independent gold reference available; cohort size not reconciled.")
        return report

    master_uids = set(fused_raw["StudyInstanceUID"])
    missing = sorted(reference - oof_gold)
    extra = sorted(oof_gold - reference)
    print(f"Gold reconciliation vs {ref_name}: CV gold={len(oof_gold)}  reference={len(reference)}  "
          f"missing from CV={len(missing)}  unknown to reference={len(extra)}")
    require(not extra, f"CV gold studies not present in the {ref_name}: {extra[:5]}")

    detail = []
    for u in missing:
        in_folds = u in folds_map.index
        fold = int(folds_map.loc[u]) if in_folds else None
        detail.append(dict(uid=u, in_fused_master=u in master_uids, in_fold_map=in_folds, fold=fold))
        where = f"fold {fold}" if in_folds else "NOT in phase2_folds.csv (excluded from the OOF cohort; no fold)"
        print(f"   MISSING gold study {u}: in fused master={u in master_uids}; {where}")
    leaked = [d["uid"] for d in detail if d["in_fold_map"]]
    require(not leaked,
            f"gold studies are inside the OOF fold map but not flagged gold (they would be overwritten as "
            f"pseudo-labels in Gen-2 and dropped from validation): {leaked[:5]}")

    if authoritative is not None and master_gold is not None and authoritative != master_gold:
        print(f"   WARNING: authoritative list and fused-master gold flags disagree: "
              f"only-in-list={sorted(authoritative - master_gold)[:5]}  "
              f"only-in-master={sorted(master_gold - authoritative)[:5]}")
    if missing:
        print(f"   WARNING: {len(missing)} gold study(ies) are not evaluated in CV. N_GOLD and the fold counts "
              "are left unchanged; update them only once the missing study's true fold is established.")
    present_folds = [int(folds_map.loc[u]) for u in (reference & set(folds_map.index))]
    print("   reference-gold per fold (for comparison with EXPECTED_GOLD_FOLD_COUNTS):",
          [present_folds.count(f) for f in range(N_FOLDS)])
    report.update(missing=detail, extra=extra)
    return report


# =====================================================================================
# 4. PHASE-2 OOF LOADING / AUDIT
# =====================================================================================
def infer_target_cols(df):
    t = [c[:-6] for c in df.columns if c.endswith("_logit")]
    if not t:
        t = [c[:-5] for c in df.columns if c.endswith("_pred")]
    return t


def read_phase2_oof(path, targets):
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    require("StudyInstanceUID" in raw.columns, f"{os.path.basename(path)} has no StudyInstanceUID column")
    require("fold" in raw.columns, f"{os.path.basename(path)} has no fold column")
    out = pd.DataFrame({
        "StudyInstanceUID": raw["StudyInstanceUID"].astype(str).str.strip(),
        "fold": pd.to_numeric(raw["fold"], errors="coerce"),
    })
    recovered = []
    for c in targets:
        lc, pc = f"{c}_logit", f"{c}_pred"
        if lc in raw.columns:
            logit = pd.to_numeric(raw[lc], errors="coerce").values.astype(float)
        elif pc in raw.columns:
            raw_prob = pd.to_numeric(raw[pc], errors="coerce").values.astype(float)
            require(np.isfinite(raw_prob).all(), f"{os.path.basename(path)}: {pc} has NaN/inf")
            require(((raw_prob >= 0.0) & (raw_prob <= 1.0)).all(),
                    f"{os.path.basename(path)}: {pc} values outside [0,1]")
            logit = sp_logit(np.clip(raw_prob, LOGIT_CLIP, 1 - LOGIT_CLIP))
            recovered.append(c)
        else:
            raise AuditError(f"AUDIT FAILED: {os.path.basename(path)} has neither {lc} nor {pc}")
        pred = (pd.to_numeric(raw[pc], errors="coerce").values.astype(float)
                if pc in raw.columns and lc in raw.columns else expit(logit))
        out[lc] = logit
        out[pc] = pred
    if "is_gold" in raw.columns and raw["is_gold"].notna().all():
        out["is_gold"] = parse_bool(raw["is_gold"])
    else:
        try:
            out["is_gold"] = resolve_gold_mask(raw, os.path.basename(path))
        except Exception:
            out["is_gold"] = np.nan
    lab = find_label_matrix(raw, targets)
    lab = None if lab is None else pd.DataFrame(lab, index=out["StudyInstanceUID"].values, columns=targets)
    return dict(df=out, labels=lab, path=path, recovered=recovered, n_raw=len(raw))


def audit_phase2_frame(name, df, targets, folds_map):
    tag = f"[{name}]"
    require(len(df) == N_P2_ROWS, f"{tag} expected {N_P2_ROWS} rows, got {len(df)}")
    require(not df["StudyInstanceUID"].duplicated().any(), f"{tag} duplicated StudyInstanceUID")
    cols = ["fold"] + [f"{c}_{s}" for c in targets for s in ("logit", "pred")]
    require(not df[cols].isna().any().any(), f"{tag} NaN in UID/fold/logit/pred columns")
    require(not df["StudyInstanceUID"].str.strip().str.lower().isin(["", "nan", "none", "null"]).any(),
            f"{tag} empty/invalid StudyInstanceUID")
    fold_values = pd.to_numeric(df["fold"], errors="coerce").to_numpy(dtype=float)
    require(np.isfinite(fold_values).all(), f"{tag} non-finite fold IDs")
    require(np.equal(fold_values, np.floor(fold_values)).all(), f"{tag} non-integer fold IDs")
    require(set(np.unique(fold_values.astype(int))) == set(range(N_FOLDS)), f"{tag} expected all fold IDs 0..{N_FOLDS-1}")
    logits = df[[f"{c}_logit" for c in targets]].to_numpy(dtype=float)
    preds = df[[f"{c}_pred" for c in targets]].to_numpy(dtype=float)
    require(np.isfinite(logits).all(), f"{tag} non-finite logits")
    require(np.isfinite(preds).all(), f"{tag} non-finite predictions")
    require(((preds >= 0.0) & (preds <= 1.0)).all(), f"{tag} predictions outside [0,1]")
    require(set(df["StudyInstanceUID"]) == set(folds_map.index), f"{tag} UID set differs from phase2_folds.csv")
    expected = folds_map.reindex(df["StudyInstanceUID"]).to_numpy(dtype=int)
    require(np.array_equal(fold_values.astype(int), expected), f"{tag} fold assignments differ from phase2_folds.csv")
    expected_preds = expit(logits)
    if not np.allclose(preds, expected_preds, rtol=1e-3, atol=1e-4):
        max_diff = float(np.max(np.abs(preds - expected_preds)))
        raise AuditError(f"AUDIT FAILED: {tag} supplied probability columns disagree with sigmoid(logits); max abs diff={max_diff:.6g}")


# =====================================================================================
# 5. PHASE-3 GOLD OOF LOADING
# =====================================================================================
def _match_p3_col(df_cols, kind, stream, target):
    if stream is not None:
        candidates = [
            f"{kind}_{stream}__{target}",
            f"{kind}_{stream}_{target}",
            f"{target}_{kind}_{stream}",
            f"{target}__{kind}_{stream}",
            f"{stream}_{kind}__{target}",
            f"{stream}_{kind}_{target}",
        ]
    else:
        candidates = [
            f"{kind}__{target}",
            f"{kind}_{target}",
            f"{target}__{kind}",
            f"{target}_{kind}",
        ]
    for cand in candidates:
        if cand in df_cols:
            return cand
    return None


def _normalize_p3(df, targets):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    uid_col = next((c for c in ("StudyInstanceUID", "study_id", "study_uid", "uid", "study_ids") if c in df.columns), None)
    require(uid_col is not None, f"Phase-3 gold OOF has no StudyInstanceUID column; found {list(df.columns)[:20]}")
    uids = df[uid_col].astype(str).str.strip().to_numpy()
    require(np.all(uids != "") and not pd.Index(uids).duplicated().any(), "Phase-3 CSV has empty or duplicate study IDs")
    out = {"StudyInstanceUID": uids}
    fold_col = next((c for c in ("fold", "fold_id", "fold_ids") if c in df.columns), None)
    if fold_col is not None:
        out["fold"] = pd.to_numeric(df[fold_col], errors="coerce").to_numpy(dtype=float)

    streams = []
    col_set = set(df.columns)
    for stream in ("base", "best", "last"):
        cols, ok = {}, True
        for target in targets:
            lc = _match_p3_col(col_set, "logit", stream, target)
            pc = _match_p3_col(col_set, "pred", stream, target)
            if lc is not None:
                cols[f"logit_{stream}__{target}"] = pd.to_numeric(df[lc], errors="coerce").to_numpy(dtype=float)
            elif pc is not None:
                pred = pd.to_numeric(df[pc], errors="coerce").to_numpy(dtype=float)
                require(np.isfinite(pred).all(), f"Phase-3 CSV has non-finite probabilities in {pc}")
                require(((pred >= 0.0) & (pred <= 1.0)).all(), f"Phase-3 CSV probabilities outside [0,1] in {pc}")
                cols[f"logit_{stream}__{target}"] = sp_logit(np.clip(pred, LOGIT_CLIP, 1 - LOGIT_CLIP))
            else:
                ok = False
                break
        if ok:
            out.update(cols)
            streams.append(stream)

    # Legacy non-streamed CSV is allowed, but its two aliases are explicitly
    # tagged as duplicates and must not be interpreted as independent streams.
    if not streams:
        cols_single, ok_single = {}, True
        for target in targets:
            lc = _match_p3_col(col_set, "logit", None, target)
            pc = _match_p3_col(col_set, "pred", None, target)
            if lc is not None:
                val = pd.to_numeric(df[lc], errors="coerce").to_numpy(dtype=float)
            elif pc is not None:
                pred = pd.to_numeric(df[pc], errors="coerce").to_numpy(dtype=float)
                require(np.isfinite(pred).all(), f"Legacy Phase-3 CSV has non-finite probabilities for {target}")
                require(((pred >= 0.0) & (pred <= 1.0)).all(), f"Legacy Phase-3 CSV probabilities outside [0,1] for {target}")
                val = sp_logit(np.clip(pred, LOGIT_CLIP, 1 - LOGIT_CLIP))
            else:
                ok_single = False
                break
            cols_single[target] = val
        if ok_single:
            for target, val in cols_single.items():
                out[f"logit_last__{target}"] = val
                out[f"logit_best__{target}"] = val
            streams = ["last", "best", "single_stream_v1"]

    require("last" in streams, f"Phase-3 CSV must contain a complete last stream; found streams={streams}")
    label_matrix = find_label_matrix(df, targets)
    labels = None if label_matrix is None else pd.DataFrame(label_matrix, index=uids, columns=targets)
    return pd.DataFrame(out), streams, labels


def _p3_npz_find_key(keys, aliases):
    """Find an exact NPZ key, case-insensitively, without guessing ambiguously."""
    lookup = {}
    for key in keys:
        lookup.setdefault(str(key).strip().lower(), []).append(key)

    for alias in aliases:
        hits = lookup.get(alias.lower(), [])
        if len(hits) > 1:
            raise AuditError(f"Ambiguous NPZ keys for {alias!r}: {hits}")
        if hits:
            return hits[0]
    return None


def _p3_npz_find_label_key(arrays, n_rows, n_targets):
    """[v2.7 FIX 3] Locate an embedded label matrix.

    'labels','label','y_true','ground_truth','y' are unambiguous: if present they must be numeric and
    are shape-checked later (a wrong shape is an error, not a silent skip).
    'targets'/'target' are ambiguous names: they are accepted only when numeric AND shaped exactly
    (n_rows, n_targets); an array of target *names* (strings) or an index vector is ignored so a
    legitimate archive is not rejected.
    """
    keys = list(arrays)
    for alias in ("targets", "target"):
        key = _p3_npz_find_key(keys, [alias])
        if key is None:
            continue
        arr = np.asarray(arrays[key])
        if arr.dtype.kind in "biuf" and arr.shape == (n_rows, n_targets):
            return key
    for alias in ("labels", "label", "y_true", "ground_truth", "y"):
        key = _p3_npz_find_key(keys, [alias])
        if key is None:
            continue
        if np.asarray(arrays[key]).dtype.kind in "biuf":
            return key
    return None


def _p3_npz_uids(values, context):
    """Convert UID data into clean strings; reject invalid or duplicate IDs."""
    arr = np.asarray(values)
    require(arr.ndim > 0, f"{context}: UID array is scalar")
    out = []
    for value in arr.reshape(-1):
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8", errors="strict")
        uid = str(value).strip()
        require(uid.lower() not in ("", "nan", "none", "null"), f"{context}: empty or invalid UID")
        out.append(uid)

    result = np.asarray(out, dtype=str)
    require(len(result) > 0, f"{context}: no study IDs")
    require(len(set(result.tolist())) == len(result), f"{context}: duplicate study IDs")
    return result


def _p3_npz_matrix(values, n_rows, n_targets, context):
    """Require a finite prediction matrix with exactly (n_studies, n_targets)."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{context}: cannot convert array to numeric") from exc

    if arr.ndim == 1 and n_rows == 1 and arr.size == n_targets:
        arr = arr.reshape(1, n_targets)
    elif arr.ndim == 1 and n_targets == 1 and arr.size == n_rows:
        arr = arr.reshape(n_rows, 1)

    require(arr.shape == (n_rows, n_targets),
            f"{context}: expected shape {(n_rows, n_targets)}, got {arr.shape}")
    require(np.isfinite(arr).all(), f"{context}: NaN or infinity detected")
    return arr


def _p3_npz_fold_map(folds_map, expected_folds):
    """Convert a pandas Series, DataFrame, or dict to UID -> canonical fold."""
    if folds_map is None:
        return None

    if isinstance(folds_map, pd.DataFrame):
        require({"StudyInstanceUID", "fold"} <= set(folds_map.columns),
                "folds_map DataFrame needs StudyInstanceUID and fold")
        require(folds_map["StudyInstanceUID"].is_unique, "Canonical fold map has duplicate UIDs")
        mapping = pd.Series(folds_map["fold"].to_numpy(),
                            index=folds_map["StudyInstanceUID"].astype(str).str.strip())
    elif isinstance(folds_map, pd.Series):
        require(folds_map.index.is_unique, "Canonical fold map has duplicate UID index entries")
        mapping = pd.Series(folds_map.to_numpy(), index=pd.Index(folds_map.index).astype(str).str.strip())
    elif isinstance(folds_map, dict):
        mapping = pd.Series(folds_map)
        mapping.index = pd.Index(mapping.index).astype(str).str.strip()
    else:
        raise AuditError("folds_map must be a pandas Series, pandas DataFrame, dict, or None")

    require(mapping.index.is_unique, "Canonical fold map has duplicate UIDs after whitespace cleanup")

    vals = pd.to_numeric(mapping.to_numpy(), errors="coerce").astype(float)
    require(np.isfinite(vals).all() and np.equal(vals, np.floor(vals)).all(),
            "Canonical fold map contains invalid fold values")
    mapping = pd.Series(vals.astype(int), index=mapping.index)
    require(set(mapping.unique()) == set(expected_folds),
            f"Canonical fold map must contain exactly {sorted(expected_folds)}")
    return mapping.to_dict()


def _load_npz_arrays(zf, member):
    """[v2.7 ADD] Load an NPZ member with allow_pickle=False first; fall back to pickle only if the
    archive genuinely holds object arrays (old string-UID format).  Trusted backups only."""
    raw = zf.read(member)
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    except ValueError:
        print(f"   note: {os.path.basename(member)} contains object arrays; loading with allow_pickle=True "
              "(trusted archives only)")
        with np.load(io.BytesIO(raw), allow_pickle=True) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}


def _p3_from_npz(zf, members, targets, folds_map=None, allow_partial_folds=True):
    """Safely assemble fold-level Phase-3 OOF NPZ predictions.

    Supports fold_<k>_phase3_oof_{base,best,last}.npz and legacy fold_<k>_phase3_oof.npz (= last).

    Fold-number handling:
      - With folds_map: each archive's canonical fold is inferred from its UIDs (zero-based,
        one-based and arbitrary numbering all work).
      - Without folds_map: only a complete conventional 0- or 1-based set is accepted.
    Missing folds: with allow_partial_folds=True the available folds are returned and recorded in
    df.attrs for diagnostics; the caller must still reject incomplete cohorts.
    """
    targets = list(targets)
    require(len(targets) > 0, "Target list is empty")
    require(len(set(targets)) == len(targets), "Duplicate target names")

    expected_folds = set(range(N_FOLDS))
    canonical_map = _p3_npz_fold_map(folds_map, expected_folds)

    # ---- A. Discover and read all available fold/stream files -------------------------
    filename_pattern = re.compile(r"fold_(\d+)_phase3_oof_(base|best|last)\.npz$", flags=re.IGNORECASE)
    legacy_pattern = re.compile(r"fold_(\d+)_phase3_oof\.npz$", flags=re.IGNORECASE)

    by_fold = {}

    for member in sorted(members):
        base_name = os.path.basename(member)

        match = filename_pattern.search(base_name)
        if match:
            raw_fold = int(match.group(1))
            stream = match.group(2).lower()
        else:
            match = legacy_pattern.search(base_name)
            if not match:
                continue
            raw_fold = int(match.group(1))
            stream = "last"
            print(f"   note: legacy NPZ {base_name} interpreted as stream='last'")

        try:
            arrays = _load_npz_arrays(zf, member)
        except Exception as exc:
            raise AuditError(f"Cannot read Phase-3 NPZ member {member}: {exc}") from exc

        keys = list(arrays)

        uid_key = _p3_npz_find_key(
            keys, ["study_ids", "StudyInstanceUID", "study_id", "study_uid", "uids", "uid"])
        require(uid_key is not None, f"{member}: no recognized study-ID array; keys={keys}")
        uids = _p3_npz_uids(arrays[uid_key], member)

        n_rows, n_targets = len(uids), len(targets)

        logits_key = _p3_npz_find_key(keys, ["logits", "logit", "oof_logits"])
        pred_key = _p3_npz_find_key(
            keys, ["predictions", "preds", "probabilities", "probs", "oof_predictions"])
        require(logits_key is not None or pred_key is not None, f"{member}: neither logits nor predictions found")

        logits = (_p3_npz_matrix(arrays[logits_key], n_rows, n_targets, f"{member} logits")
                  if logits_key is not None else None)
        preds = (_p3_npz_matrix(arrays[pred_key], n_rows, n_targets, f"{member} predictions")
                 if pred_key is not None else None)

        if preds is not None:
            require(((preds >= 0.0) & (preds <= 1.0)).all(), f"{member}: probabilities outside [0, 1]")

        if logits is not None and preds is not None:
            require(np.allclose(expit(logits), preds, rtol=1e-3, atol=1e-4),
                    f"{member}: sigmoid(logits) disagrees with predictions")

        if logits is None:
            logits = sp_logit(np.clip(preds, LOGIT_CLIP, 1.0 - LOGIT_CLIP))

        # Optional embedded labels. Canonical labels remain authoritative.
        labels_key = _p3_npz_find_label_key(arrays, n_rows, n_targets)   # [v2.7 FIX 3]
        labels = None
        if labels_key is not None:
            labels = _p3_npz_matrix(arrays[labels_key], n_rows, n_targets, f"{member} labels")
            require(np.isin(labels, [0.0, 1.0]).all(), f"{member}: embedded labels are not binary")

        # Optional embedded fold IDs (numbering may differ from the filename numbering).
        fold_key = _p3_npz_find_key(keys, ["fold_ids", "fold_id", "fold"])
        embedded_fold = None
        if fold_key is not None:
            embedded = np.asarray(arrays[fold_key]).reshape(-1)
            if embedded.size == 1:
                embedded = np.repeat(embedded, n_rows)
            require(embedded.size == n_rows, f"{member}: embedded fold ID count does not match study IDs")
            try:
                embedded_num = pd.to_numeric(pd.Series(embedded), errors="coerce").to_numpy(dtype=float)
            except Exception as exc:
                raise AuditError(f"{member}: invalid embedded fold IDs") from exc
            require(np.isfinite(embedded_num).all() and np.equal(embedded_num, np.floor(embedded_num)).all(),
                    f"{member}: non-integer embedded fold IDs")
            unique_embedded = np.unique(embedded_num.astype(int))
            require(len(unique_embedded) == 1,
                    f"{member}: a fold-specific NPZ contains multiple embedded fold IDs: {unique_embedded.tolist()}")
            embedded_fold = int(unique_embedded[0])

        fold_dict = by_fold.setdefault(raw_fold, {})
        require(stream not in fold_dict, f"Duplicate stream '{stream}' for raw fold {raw_fold}: {member}")
        fold_dict[stream] = {"uids": uids, "logits": logits, "labels": labels,
                             "embedded_fold": embedded_fold, "member": member}

    raw_folds = sorted(by_fold.keys())
    require(raw_folds, "No recognized fold-level Phase-3 NPZ files found")
    print(f"   discovered filename fold IDs: {raw_folds}")

    # ---- B. Resolve filename IDs to canonical fold IDs --------------------------------
    raw_to_canonical = {}
    embedded_to_canonical = {}

    if canonical_map is not None:
        for raw_fold, stream_dict in by_fold.items():
            canonical_values_for_raw_fold = set()

            for stream, rec in stream_dict.items():
                missing = [uid for uid in rec["uids"] if uid not in canonical_map]
                # [v2.7] fixed: the second line of this message previously lacked the f prefix, so
                # the example UIDs were printed as a literal "{missing[:5]}".
                require(not missing,
                        f"{rec['member']}: {len(missing)} UIDs are absent from the canonical fold map; "
                        f"examples={missing[:5]}")

                uid_folds = {int(canonical_map[uid]) for uid in rec["uids"]}
                require(len(uid_folds) == 1,
                        f"{rec['member']}: studies span multiple canonical folds {sorted(uid_folds)}")

                canonical_fold = next(iter(uid_folds))
                canonical_values_for_raw_fold.add(canonical_fold)

                embedded_fold = rec["embedded_fold"]
                if embedded_fold is not None:
                    previous = embedded_to_canonical.get(embedded_fold)
                    require(previous is None or previous == canonical_fold,
                            f"Embedded fold ID {embedded_fold} maps to multiple canonical folds")
                    embedded_to_canonical[embedded_fold] = canonical_fold

            require(len(canonical_values_for_raw_fold) == 1,
                    f"Filename fold {raw_fold} maps inconsistently to canonical folds "
                    f"{sorted(canonical_values_for_raw_fold)}")
            raw_to_canonical[raw_fold] = next(iter(canonical_values_for_raw_fold))

        canonical_values = list(raw_to_canonical.values())
        require(len(set(canonical_values)) == len(canonical_values),
                f"Multiple filename fold IDs map to the same canonical fold: {raw_to_canonical}")
        embedded_values = list(embedded_to_canonical.values())
        require(len(set(embedded_values)) == len(embedded_values),
                f"Multiple embedded fold IDs map to the same canonical fold: {embedded_to_canonical}")

        for raw_fold, canonical_fold in raw_to_canonical.items():
            print(f"   fold mapping: archive ID {raw_fold} -> canonical fold {canonical_fold}")

    else:
        zero_based = set(range(N_FOLDS))
        one_based = set(range(1, N_FOLDS + 1))
        observed = set(raw_folds)

        if observed == zero_based:
            raw_to_canonical = {f: f for f in raw_folds}
            print("   fold numbering inferred as conventional 0-based")
        elif observed == one_based:
            raw_to_canonical = {f: f - 1 for f in raw_folds}
            print("   fold numbering inferred as conventional 1-based")
        else:
            raise AuditError(
                "Cannot safely infer fold numbering without folds_map. "
                f"Found raw folds {raw_folds}; expected a complete 0..{N_FOLDS - 1} or 1..{N_FOLDS} sequence. "
                "Pass the canonical UID-to-fold mapping to resolve numbering or identify missing folds safely."
            )

        for raw_fold, stream_dict in by_fold.items():
            canonical_fold = raw_to_canonical[raw_fold]
            for rec in stream_dict.values():
                embedded_fold = rec["embedded_fold"]
                if embedded_fold is None:
                    continue
                previous = embedded_to_canonical.get(embedded_fold)
                require(previous is None or previous == canonical_fold,
                        f"Embedded fold ID {embedded_fold} maps inconsistently")
                embedded_to_canonical[embedded_fold] = canonical_fold

    available_canonical_folds = set(raw_to_canonical.values())
    missing_folds = expected_folds - available_canonical_folds
    unexpected_folds = available_canonical_folds - expected_folds
    require(not unexpected_folds, f"Resolved unexpected canonical folds: {sorted(unexpected_folds)}")
    print(f"   available canonical folds: {sorted(available_canonical_folds)}")

    if missing_folds:
        message = f"Missing canonical folds: {sorted(missing_folds)}; the archive is incomplete."
        if allow_partial_folds:
            print(f"   WARNING: {message} Parsing available folds for diagnosis only.")
        else:
            raise AuditError(message)

    # ---- C. Merge each fold by UID (never assume identical row order) -----------------
    canonical_records = {}
    for raw_fold, stream_dict in by_fold.items():
        canonical_fold = raw_to_canonical[raw_fold]
        require(canonical_fold not in canonical_records,
                f"Duplicate source fold for canonical fold {canonical_fold}")
        canonical_records[canonical_fold] = stream_dict

    for canonical_fold, stream_dict in canonical_records.items():
        require("last" in stream_dict,
                f"Canonical fold {canonical_fold} has no actual 'last' stream. "
                "Do not substitute another stream for it.")

    complete_streams = []
    for stream in ("base", "best", "last"):
        is_complete = True
        for canonical_fold, stream_dict in canonical_records.items():
            if stream not in stream_dict:
                is_complete = False
                break
            reference_uids = set(stream_dict["last"]["uids"].tolist())
            stream_uids = set(stream_dict[stream]["uids"].tolist())
            if stream_uids != reference_uids:
                print(f"   WARNING: omitting stream '{stream}': UID coverage differs in canonical fold {canonical_fold}")
                is_complete = False
                break
        if is_complete:
            complete_streams.append(stream)
        else:
            print(f"   WARNING: stream '{stream}' is incomplete; it will not be included in the assembled dataframe.")

    require("last" in complete_streams, "A complete 'last' stream is required across all available folds")

    frames = []
    label_frames = []
    labels_complete = True

    for canonical_fold in sorted(canonical_records):
        stream_dict = canonical_records[canonical_fold]
        primary = stream_dict["last"]
        reference_uids = primary["uids"]

        frame = pd.DataFrame({
            "StudyInstanceUID": reference_uids,
            "fold": np.full(len(reference_uids), canonical_fold, dtype=int),
        })

        for stream in complete_streams:
            rec = stream_dict[stream]
            index = pd.Index(rec["uids"])
            positions = index.get_indexer(reference_uids)
            require(np.all(positions >= 0) and len(positions) == len(reference_uids),
                    f"UID alignment failed for fold {canonical_fold}, stream {stream}")

            aligned_logits = rec["logits"][positions]
            require(np.isfinite(aligned_logits).all(),
                    f"Non-finite predictions in fold {canonical_fold}, stream {stream}")

            # [v2.7 ADD] if several streams embed labels they must agree after UID alignment.
            if rec["labels"] is not None and primary["labels"] is not None:
                require(np.array_equal(rec["labels"][positions], primary["labels"]),
                        f"Embedded labels differ between streams in canonical fold {canonical_fold}")

            for j, target in enumerate(targets):
                frame[f"logit_{stream}__{target}"] = aligned_logits[:, j]

        frames.append(frame)

        if primary["labels"] is None:
            labels_complete = False
        else:
            label_frames.append(pd.DataFrame(primary["labels"], index=primary["uids"], columns=targets))

    df = pd.concat(frames, ignore_index=True)

    require(not df["StudyInstanceUID"].duplicated().any(), "Duplicate study IDs across assembled folds")
    require(df["fold"].notna().all() and df["fold"].isin(expected_folds).all(),
            "Invalid canonical fold IDs in assembled dataframe")

    prediction_columns = [c for c in df.columns if c.startswith("logit_")]
    require(len(prediction_columns) == len(complete_streams) * len(targets),
            "Unexpected number of assembled stream/target columns")
    require(np.isfinite(df[prediction_columns].to_numpy(dtype=float)).all(),
            "Assembled predictions contain NaN or infinity")

    labels_df = None
    if labels_complete and len(label_frames) == len(frames):
        labels_df = pd.concat(label_frames, axis=0)
        labels_df = labels_df.loc[df["StudyInstanceUID"].tolist(), targets]
        labels_df.index.name = "StudyInstanceUID"

    df.attrs["p3_streams"] = list(complete_streams)
    df.attrs["available_folds"] = sorted(available_canonical_folds)
    df.attrs["missing_folds"] = sorted(missing_folds)
    df.attrs["raw_to_canonical_fold"] = {int(k): int(v) for k, v in raw_to_canonical.items()}
    df.attrs["embedded_to_canonical_fold"] = {int(k): int(v) for k, v in embedded_to_canonical.items()}
    df.attrs["is_complete_fold_set"] = not missing_folds

    print(f"   assembled rows: {len(df)}")
    print(f"   complete streams: {complete_streams}")
    print(f"   canonical fold coverage complete: {not missing_folds}")

    return df, labels_df


def load_phase3_gold_oof(backbone, targets=None, folds_map=None):
    targets = targets or TARGET_COLS
    info = next((b for b in BACKBONES if b["name"] == backbone or b["stem"] == backbone), None)
    stem = info["stem"] if info else backbone.split(".")[0]
    files = sorted(os.listdir(P3_DIR)) if os.path.isdir(P3_DIR) else []
    rejections = []   # [v2.7] collected so the final error explains *why* every candidate failed

    # 1. Loose CSVs in Drive P3_DIR
    loose = [f for f in files if f == f"{backbone}_phase3_gold_oof_predictions.csv"]
    loose += [f for f in files if f.startswith(stem) and f.endswith("gold_oof_predictions.csv") and f not in loose]
    for fname in loose:
        path = os.path.join(P3_DIR, fname)
        try:
            df, streams, lab = _normalize_p3(pd.read_csv(path), targets)
            if len(df) == N_GOLD and ("fold" in df.columns or not REQUIRE_EXPLICIT_P3_FOLD):
                return dict(df=df, streams=streams, labels=lab, source=f"loose CSV: {fname}", path=path)
            if len(df) == N_GOLD and "fold" not in df.columns:
                print(f"   note: skipping {fname}; strict mode requires explicit Phase-3 fold IDs")
                rejections.append(f"{fname}: no fold column")
        except Exception as e:
            print(f"   note: skipping loose CSV {fname} ({e})")
            rejections.append(f"{fname}: {e}")

    # 2. Inside *_phase3.zip in Drive P3_DIR (CSV inside the zip, or fold-level NPZs)
    zips = [f for f in files if f == f"{backbone}_phase3.zip"]
    zips += [f for f in files if f.startswith(stem) and f.endswith("_phase3.zip") and f not in zips]
    zips += [f for f in files if f.startswith(stem) and "phase3" in f and f.endswith(".zip") and f not in zips]
    for zname in zips:
        zpath = os.path.join(P3_DIR, zname)
        # [v2.7 FIX 1] Everything below is inside both `for zname` and `with ZipFile`, so `zf`,
        # `names` and `zname` are in scope and `return` is legal (it previously sat at column 0).
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            csvs = sorted([n for n in names if os.path.basename(n) == "phase3_gold_oof_predictions.csv"], key=len)
            csvs += sorted([n for n in names if n.endswith("gold_oof_predictions.csv") and n not in csvs], key=len)
            if csvs:
                try:
                    df, streams, lab = _normalize_p3(pd.read_csv(io.BytesIO(zf.read(csvs[0]))), targets)
                    if len(df) == N_GOLD and ("fold" in df.columns or not REQUIRE_EXPLICIT_P3_FOLD):
                        return dict(df=df, streams=streams, labels=lab,
                                    source=f"zip CSV: {zname}::{csvs[0]}", path=zpath)
                    if len(df) == N_GOLD and "fold" not in df.columns:
                        print(f"   note: skipping {zname}::{csvs[0]}; strict mode requires explicit Phase-3 fold IDs")
                        rejections.append(f"{zname}::{csvs[0]}: no fold column")
                except AuditError as exc:
                    print(f"   note: skipping {zname}::{csvs[0]} ({exc})")
                    rejections.append(f"{zname}::{csvs[0]}: {exc}")

            npz_members = [
                name for name in names
                if re.search(r"fold_\d+_phase3_oof(?:_(?:base|best|last))?\.npz$",
                             os.path.basename(name), flags=re.IGNORECASE)
            ]
            if not npz_members:
                continue

            try:
                df, labels_df = _p3_from_npz(
                    zf, npz_members, targets, folds_map=folds_map,
                    allow_partial_folds=True,  # diagnostic parsing, not validation approval
                )
            except AuditError as exc:
                print(f"   WARNING: NPZ parse of {zname} rejected: {exc}. Continuing archive discovery.")
                rejections.append(f"{zname} (NPZ): {exc}")
                continue

            actual_streams = df.attrs.get("p3_streams", [])
            complete_fold_set = df.attrs.get("is_complete_fold_set", False)

            if complete_fold_set and len(df) == N_GOLD:
                require(df["StudyInstanceUID"].is_unique, f"{zname}: duplicate study IDs in assembled NPZ predictions")
                require("last" in actual_streams, f"{zname}: no complete last stream")
                return dict(df=df, streams=actual_streams, labels=labels_df,
                            source=f"zip NPZ x{len(npz_members)}: {zname}", path=zpath)

            msg = (f"{zname}: rows={len(df)}/{N_GOLD}, available_folds={df.attrs.get('available_folds')}, "
                   f"missing_folds={df.attrs.get('missing_folds')}")
            print(f"   WARNING: not accepting {msg}. Continuing archive discovery.")
            rejections.append(msg)

    # 3. Local /content/phase3_models/{backbone}/phase3_gold_oof_predictions.csv
    if os.path.isdir(LOCAL_P3_MODELS_DIR):
        for sub in sorted(os.listdir(LOCAL_P3_MODELS_DIR)):
            if sub == backbone or sub.startswith(stem):
                local_csv = os.path.join(LOCAL_P3_MODELS_DIR, sub, "phase3_gold_oof_predictions.csv")
                if os.path.isfile(local_csv):
                    df, streams, lab = _normalize_p3(pd.read_csv(local_csv), targets)
                    if len(df) == N_GOLD and ("fold" in df.columns or not REQUIRE_EXPLICIT_P3_FOLD):
                        return dict(df=df, streams=streams, labels=lab, source=f"local CSV: {local_csv}", path=local_csv)
                    if len(df) == N_GOLD and "fold" not in df.columns:
                        print(f"   note: skipping {local_csv}; strict mode requires explicit Phase-3 fold IDs")
                        rejections.append(f"{local_csv}: no fold column")

    raise AuditError(f"AUDIT FAILED: no Phase-3 gold OOF of length {N_GOLD} found for '{backbone}' in {P3_DIR}. "
                     f"Rejected candidates: {rejections if rejections else 'none found'}")


# =====================================================================================
# 6. CALIBRATION (deployment / diagnostics; the nested engine has its own per-split version)
# =====================================================================================
def fold_calibrate(L, fold, scope=CALIBRATION_SCOPE):
    """Unsupervised calibration fit on the *complete* OOF cohort.

    Used for (a) the deployment calibration exported to ensemble_config.json and applied to the
    non-gold pseudo-label rows, and (b) descriptive diagnostics.  Validation estimates do NOT use
    this directly any more: Ensembler.cal() re-fits the same transform on training folds only.
    """
    L = np.asarray(L, dtype=float)
    fold = np.asarray(fold, dtype=int)
    require(L.ndim == 2 and L.shape[0] == len(fold), "fold_calibrate shape mismatch")
    require(np.isfinite(L).all(), "fold_calibrate received non-finite logits")
    N, C = L.shape
    mu_f, sd_f = np.zeros((N_FOLDS, C)), np.zeros((N_FOLDS, C))
    Z_fold = np.empty_like(L, dtype=float)
    R_fold = np.empty_like(L, dtype=float)
    sorted_ref_by_fold = {}

    for f in range(N_FOLDS):
        idx = np.where(fold == f)[0]
        require(len(idx) > 0, f"Fold {f} has no OOF rows")
        x = L[idx]
        mu_f[f], sd_f[f] = x.mean(0), x.std(0)
        Z_fold[idx] = (x - mu_f[f]) / (sd_f[f] + EPS_Z)
        R_fold[idx] = rankdata(x, axis=0) / (len(idx) + 1.0)
        s_ref = np.sort(x, axis=0)
        sorted_ref_by_fold[str(f)] = s_ref
        require(np.allclose(R_fold[idx], ecdf_from_sorted_reference(x, s_ref), rtol=0, atol=1e-12),
                f"Per-fold ECDF self-check failed on fold {f}")

    mu_g, sd_g = L.mean(0), L.std(0)
    Z_glob = (L - mu_g) / (sd_g + EPS_Z)
    R_glob = rankdata(L, axis=0) / (N + 1.0)
    sorted_ref = np.sort(L, axis=0)
    require(np.allclose(R_glob, ecdf_from_sorted_reference(L, sorted_ref), rtol=0, atol=1e-12),
            "Global ECDF calibration self-check failed")

    Z = Z_fold if scope == "per_fold" else Z_glob
    R = R_fold if scope == "per_fold" else R_glob
    n_f = np.bincount(fold, minlength=N_FOLDS)
    return dict(
        Z=Z, R=R, Z_fold=Z_fold, R_fold=R_fold, Z_glob=Z_glob, R_glob=R_glob,
        mu_f=mu_f, sd_f=sd_f, mu_g=mu_g, sd_g=sd_g, n_f=n_f,
        sorted_ref=sorted_ref, sorted_ref_by_fold=sorted_ref_by_fold,
    )


# =====================================================================================
# 7. ENSEMBLE ENGINE  (v2.7: strictly nested calibration)
# =====================================================================================
class Ensembler:
    """All schemes operate on hybrid scores s^(m)(alpha) = (1-alpha) sigmoid(z) + alpha * ecdf_rank.

    [v2.7 FIX 4] The engine now takes RAW logits and re-derives everything that is data-dependent
    from `fit_folds` only:
        * z-score mean/std and ECDF reference  -> rows of the fit folds (all rows, labels unused)
        * performance weights (B/C/D/E)         -> gold AUC on the fit folds
        * co-occurrence ridge prior (D)         -> non-gold rows of the fit folds
    Held-out rows are merely *transformed* with those references.  `fit_folds` is therefore the single
    knob that defines "what the model was allowed to see": 3 folds for inner CV, 4 folds for the
    outer fold, all 5 folds for deployment.
    """

    def __init__(self, L, fold_all, gold_idx, Yg, fold_g, L3_gold=None, w_co_target=None):
        self.L = np.asarray(L, dtype=float)                    # (M, N, C) raw Phase-2 logits
        self.M, self.N, self.C = self.L.shape
        self.fold_all = np.asarray(fold_all, dtype=int)
        require(self.fold_all.shape == (self.N,), "Ensembler: fold_all length mismatch")
        self.gi = np.asarray(gold_idx)
        self.ng = np.setdiff1d(np.arange(self.N), self.gi)
        self.y, self.fg = Yg, np.asarray(fold_g)
        require(np.array_equal(self.fold_all[self.gi], self.fg), "Ensembler: gold fold IDs inconsistent with fold_all")
        self.L3_gold = None if L3_gold is None else np.asarray(L3_gold, dtype=float)   # (M, G, C) raw P3-last
        if self.L3_gold is not None:
            require(self.L3_gold.shape == (self.M, len(self.gi), self.C), "Ensembler: L3_gold shape mismatch")
        self.w_co_target = w_co_target
        self._cal, self._S, self._Sg, self._auc, self._wco = {}, {}, {}, {}, {}

    # ---- cache keys -------------------------------------------------------------------
    @staticmethod
    def _fkey(folds):
        key = tuple(sorted(int(f) for f in folds))
        require(len(key) > 0, "Ensembler: empty fit-fold set")
        return key

    # ---- calibration fit on `fit_folds`, applied to every row --------------------------
    def cal(self, fit_folds):
        key = self._fkey(fit_folds)
        if key not in self._cal:
            ref_rows = np.where(np.isin(self.fold_all, key))[0]
            require(ref_rows.size > 0, f"Ensembler: no OOF rows in fit folds {key}")
            Z = np.empty_like(self.L)
            R = np.empty_like(self.L)
            Zp3 = Rp3 = None
            if self.L3_gold is not None:
                Zp3, Rp3 = np.empty_like(self.L3_gold), np.empty_like(self.L3_gold)
            for m in range(self.M):
                ref = self.L[m, ref_rows, :]
                mu, sd = ref.mean(0), ref.std(0)
                s_ref = np.sort(ref, axis=0)
                Z[m] = (self.L[m] - mu) / (sd + EPS_Z)
                R[m] = ecdf_from_sorted_reference(self.L[m], s_ref)
                if Zp3 is not None:
                    # [v2.7] Scheme-E P3 logits use the SAME P2 reference as everything else (previously:
                    # fold-local P2 stats + ranks computed inside the 57 gold rows, i.e. a different scale).
                    Zp3[m] = (self.L3_gold[m] - mu) / (sd + EPS_Z)
                    Rp3[m] = ecdf_from_sorted_reference(self.L3_gold[m], s_ref)
            self._cal[key] = (Z, R, Zp3, Rp3)
        return self._cal[key]

    def S(self, a, fit_folds):
        key = (round(float(a), 6), self._fkey(fit_folds))
        if key not in self._S:
            Z, R, _, _ = self.cal(fit_folds)
            self._S[key] = (1.0 - a) * expit(Z) + a * R
        return self._S[key]

    def Sg(self, a, fit_folds, beta_p3=0.0):
        """Hybrid scores for the gold rows only, optionally mixing in Phase-3 (Scheme E)."""
        if beta_p3 <= 0.0 or self.L3_gold is None:
            return self.S(a, fit_folds)[:, self.gi, :]
        key = (round(float(a), 6), self._fkey(fit_folds), round(float(beta_p3), 6))
        if key not in self._Sg:
            Z, R, Zp3, Rp3 = self.cal(fit_folds)
            z_mix = (1.0 - beta_p3) * Z[:, self.gi, :] + beta_p3 * Zp3
            r_mix = (1.0 - beta_p3) * R[:, self.gi, :] + beta_p3 * Rp3
            self._Sg[key] = (1.0 - a) * expit(z_mix) + a * r_mix
        return self._Sg[key]

    def auc_mat(self, a, fit_folds, beta_p3=0.0):
        """(M, C) gold AUC on the fit folds, scored with a calibration fit on those same folds."""
        key = (round(float(a), 6), round(float(beta_p3), 6), self._fkey(fit_folds))
        if key not in self._auc:
            rows = np.where(np.isin(self.fg, list(key[2])))[0]
            self._auc[key] = auc_cols(self.Sg(a, fit_folds, beta_p3)[:, rows, :], self.y[rows])
        return self._auc[key]

    def weights(self, scheme, p, A):
        M, C = A.shape
        if scheme == "A":
            return np.full((M, C), 1.0 / M)
        if scheme == "B":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mA = np.nanmean(A, axis=1)
            mA = np.where(np.isnan(mA), 0.5, mA)
            e = np.exp(p["tau"] * (mA - mA.max()))
            return np.repeat((e / e.sum())[:, None], C, axis=1)
        if scheme in ("C", "D", "E"):
            gam = p.get("gamma", 0.0)
            if gam <= 0.0:
                return np.full((M, C), 1.0 / M)
            A0 = np.where(np.isnan(A), 0.5, A)
            e = np.exp(p["tau"] * (A0 - A0.max(axis=0, keepdims=True)))
            soft = e / e.sum(axis=0, keepdims=True)
            return (1.0 - gam) / M + gam * soft
        raise ValueError(scheme)

    def fit_weights(self, scheme, p, fit_folds):
        a = p["alpha"]
        beta_p3 = float(p.get("beta_p3", 0.0)) if scheme == "E" else 0.0
        if scheme in ("B", "C", "D", "E"):
            A = self.auc_mat(a, fit_folds, beta_p3=beta_p3)
        else:
            A = np.zeros((self.M, self.C))
        return self.weights(scheme, p, A)

    def wco(self, a, fit_folds):
        """Co-occurrence ridge prior (Scheme D), fit on NON-GOLD rows of the fit folds only."""
        key = (round(float(a), 6), self._fkey(fit_folds))
        if key not in self._wco:
            S = self.S(a, fit_folds)
            ng_fit = self.ng[np.isin(self.fold_all[self.ng], key[1])]
            require(ng_fit.size > 0, "wco: no non-gold rows in fit folds")
            U = S.mean(0)
            X = U[ng_fit]
            mu, sd = X.mean(0), X.std(0) + 1e-12
            Xs = (X - mu) / sd
            if W_CO_RESPONSE == "pseudo_label" and self.w_co_target is not None:
                Yr = self.w_co_target[ng_fit]
                ysd = Yr.std(0) + 1e-12
                Ys = (Yr - Yr.mean(0)) / ysd
            else:
                Ys, ysd = Xs, sd
            n, C = Xs.shape
            Gxx, Gxy = Xs.T @ Xs, Xs.T @ Ys
            lam = W_CO_RIDGE_REL * n
            Wstd = np.zeros((C, C))
            for c in range(C):
                idx = [j for j in range(C) if j != c]
                Wstd[c, idx] = np.linalg.solve(Gxx[np.ix_(idx, idx)] + lam * np.eye(C - 1), Gxy[idx, c])
            W = Wstd * (ysd[:, None] / sd[None, :])
            np.fill_diagonal(W, 0.0)
            self._wco[key] = (W, mu)
        return self._wco[key]

    def _scores(self, a, fit_folds, beta_p3, gold_only):
        if gold_only:
            return self.Sg(a, fit_folds, beta_p3)
        S_all = self.S(a, fit_folds).copy()
        if beta_p3 > 0.0 and self.L3_gold is not None:
            S_all[:, self.gi, :] = self.Sg(a, fit_folds, beta_p3)
        return S_all

    def predict_all(self, scheme, p, fit_folds, gold_only=False):
        """Raw ensemble scores using ONLY fit-fold information.

        gold_only=True returns (G, C) for the gold rows (cheap; used by every CV loop);
        gold_only=False returns (N, C) for all rows (used for the deployment fit).
        """
        a = p["alpha"]
        beta_p3 = float(p.get("beta_p3", 0.0)) if scheme == "E" else 0.0
        W = self.fit_weights(scheme, p, fit_folds)
        S = self._scores(a, fit_folds, beta_p3, gold_only)
        P = np.einsum("mc,mnc->nc", W, S)
        if scheme == "D" and p.get("eta", 0.0) > 0:
            Wco, mu = self.wco(a, fit_folds)
            P = P + p["eta"] * (P - mu) @ Wco.T
        n_expected = len(self.gi) if gold_only else self.N
        require(P.shape == (n_expected, self.C) and np.isfinite(P).all(),
                f"Non-finite/incorrect raw prediction array for scheme {scheme}")
        return P

    def predict(self, scheme, p, fit_folds, rows):
        """Raw scores for held-out gold rows (`rows` index the gold array); no fold-local rescaling."""
        return self.predict_all(scheme, p, fit_folds, gold_only=True)[np.asarray(rows, dtype=int)]

    def grid(self, scheme):
        if scheme == "A":
            return [dict(alpha=a) for a in ALPHA_GRID]
        if scheme == "B":
            return [dict(tau=t, alpha=a) for t in TAU_GRID for a in ALPHA_GRID]
        if scheme == "C":
            g = []
            for gam in GAMMA_GRID:
                for t in (TAU_GRID if gam > 0 else TAU_GRID[:1]):
                    for a in ALPHA_GRID:
                        g.append(dict(gamma=gam, tau=t, alpha=a))
            return g
        if scheme == "D":
            g = []
            for eta in ETA_GRID:
                for gam in (0.0, 0.30, 0.60):
                    for t in ([10.0, 20.0] if gam > 0 else [10.0]):
                        for a in (0.0, 0.25, 0.50):
                            g.append(dict(eta=eta, gamma=gam, tau=t, alpha=a))
            return g
        if scheme == "E":
            g = []
            for b3 in BETA_P3_GRID:
                for gam in (0.0, 0.30, 0.60):
                    for t in ([10.0, 20.0] if gam > 0 else [10.0]):
                        for a in (0.0, 0.25, 0.50):
                            g.append(dict(beta_p3=b3, gamma=gam, tau=t, alpha=a))
            return g
        raise ValueError(scheme)

    def inner_score(self, scheme, p, train_folds):
        """Mean macro AUC over inner validation folds; each inner fold is calibrated and weighted
        using only the remaining training folds."""
        fold_scores = []
        for heldout_fold in train_folds:
            rows = np.where(self.fg == heldout_fold)[0]
            if rows.size == 0:
                continue
            fit_folds = [f for f in train_folds if f != heldout_fold]
            if not fit_folds:
                continue
            pred = self.predict(scheme, p, fit_folds, rows)
            score = macro_nanmean(auc_cols(pred, self.y[rows]))
            if np.isfinite(score):
                fold_scores.append(float(score))
        return macro_nanmean(fold_scores)

    def select(self, scheme, train_folds):
        best_p, best_s = None, -np.inf
        for p in self.grid(scheme):
            s = self.inner_score(scheme, p, train_folds)
            if np.isfinite(s) and s > best_s + 1e-12:
                best_p, best_s = p, s
        return best_p, best_s

    def lofo(self, scheme):
        G = len(self.gi)
        P = np.full((G, self.C), np.nan, dtype=float)
        chosen = []
        fold_macros = []
        for heldout_fold in range(N_FOLDS):
            train_folds = [f for f in range(N_FOLDS) if f != heldout_fold]
            params, inner_score = self.select(scheme, train_folds)
            rows = np.where(self.fg == heldout_fold)[0]
            require(rows.size > 0, f"Outer fold {heldout_fold} has no gold rows")
            P[rows] = self.predict(scheme, params, train_folds, rows)
            fold_auc = macro_nanmean(auc_cols(P[rows], self.y[rows]))
            fold_macros.append(float(fold_auc))
            chosen.append(dict(fold=int(heldout_fold), params=params,
                               inner_score=float(inner_score), heldout_macro_auc=float(fold_auc)))
        require(np.isfinite(P).all(), f"LOFO for scheme {scheme} left unfilled/non-finite scores")
        # Pooled AUC mixes scores from differently fitted ensembles -> diagnostic only.
        pooled_per_target = auc_cols(P, self.y)
        fold_mean = macro_nanmean(fold_macros)
        return dict(
            cv_macro=fold_mean,
            pooled_macro=fold_mean,  # compatibility alias: means mean fold macro-AUC
            pooled_auc_diagnostic=macro_nanmean(pooled_per_target),
            pooled_per_target_diagnostic=pooled_per_target,
            fold_macros=fold_macros, fold_mean=fold_mean, chosen=chosen, P=P,
        )

    def choose_scheme(self, schemes, train_folds, promotion_margin=PROMOTION_MARGIN):
        candidate = {}
        for scheme in schemes:
            p, score = self.select(scheme, train_folds)
            candidate[scheme] = {"params": p, "inner_score": float(score)}
        valid = {k: v["inner_score"] for k, v in candidate.items() if np.isfinite(v["inner_score"])}
        require(bool(valid), f"No finite inner-CV score for schemes {list(schemes)}")
        best_score = max(valid.values())
        winner = next((s for s in schemes if s in valid and best_score - valid[s] < promotion_margin), None)
        require(winner is not None, "Could not select a finite ensemble scheme")
        return winner, candidate, best_score

    def nested_scheme_lofo(self, schemes, promotion_margin=PROMOTION_MARGIN):
        """Nested scheme+hyperparameter selection.  For outer fold h: everything (calibration,
        weights, co-occurrence prior, scheme choice, hyperparameters) is fit on the other 4 folds;
        fold h is only transformed.  Primary metric = unweighted mean of per-fold macro AUCs."""
        G = len(self.gi)
        P = np.full((G, self.C), np.nan, dtype=float)
        chosen = []
        for heldout_fold in range(N_FOLDS):
            train_folds = [f for f in range(N_FOLDS) if f != heldout_fold]
            winner, candidates, best_inner = self.choose_scheme(schemes, train_folds, promotion_margin)
            params = candidates[winner]["params"]
            rows = np.where(self.fg == heldout_fold)[0]
            require(rows.size > 0, f"Outer fold {heldout_fold} has no gold rows")
            P[rows] = self.predict_all(winner, params, train_folds, gold_only=True)[rows]
            fold_auc = macro_nanmean(auc_cols(P[rows], self.y[rows]))
            chosen.append(dict(
                fold=int(heldout_fold), scheme=winner, params=params,
                inner_score=float(candidates[winner]["inner_score"]),
                best_inner_score=float(best_inner), heldout_macro_auc=float(fold_auc),
                candidate_scores={name: float(value["inner_score"]) for name, value in candidates.items()},
            ))
        require(np.isfinite(P).all(), "Nested LOFO left unfilled/non-finite gold scores")
        per_target_diagnostic = auc_cols(P, self.y)
        fold_macros = [item["heldout_macro_auc"] for item in chosen]
        fold_mean = macro_nanmean(fold_macros)
        return dict(
            cv_macro=fold_mean,
            pooled_macro=fold_mean,  # compatibility alias; not pooled AUC
            pooled_auc_diagnostic=macro_nanmean(per_target_diagnostic),
            pooled_per_target_diagnostic=per_target_diagnostic,
            fold_macros=fold_macros, fold_mean=fold_mean, chosen=chosen, P=P,
        )

    def with_labels(self, Yg_new):
        """[v2.7 ADD] Shallow clone with different gold labels (for the permutation canary).
        Calibration / score caches are label-free and shared; the AUC cache is label-dependent
        and therefore reset."""
        clone = copy.copy(self)
        clone.y = np.asarray(Yg_new, dtype=float)
        clone._auc = {}
        return clone


def describe_params(p):
    return ", ".join(f"{k}={v:g}" for k, v in p.items())


# -------------------------------------------------------------------------------------
# [v2.7 ADD] Uncertainty + leakage canary
# -------------------------------------------------------------------------------------
def bootstrap_fold_mean_macro_auc(P_list, Y, fold_ids, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Paired, fold-stratified study bootstrap of the primary metric.

    Each replicate resamples gold studies WITH replacement inside each fold and scores every score
    matrix in P_list on the same resample, so differences between methods are paired.
    Caveat: the nested predictions are treated as fixed, so selection variance is not re-simulated;
    the CI is optimistic but still shows how wide the uncertainty is with ~57 studies.
    """
    rng = np.random.default_rng(seed)
    fold_ids = np.asarray(fold_ids, dtype=int)
    out = np.full((n_boot, len(P_list)), np.nan)
    fold_rows = [np.where(fold_ids == f)[0] for f in np.unique(fold_ids)]
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(rows, size=rows.size, replace=True) for rows in fold_rows])
        for k, P in enumerate(P_list):
            out[b, k] = fold_mean_macro_auc(P[idx], Y[idx], fold_ids[idx])
    return out


def permutation_canary(eng, schemes, n_perm, seed=BOOTSTRAP_SEED + 1):
    """Re-run the ENTIRE nested pipeline on gold labels shuffled within each fold.
    A leak-free pipeline must land near 0.5; a clearly higher value means information about the
    held-out labels reaches the fit somewhere."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_perm):
        Yp = eng.y.copy()
        for f in range(N_FOLDS):
            rows = np.where(eng.fg == f)[0]
            Yp[rows] = eng.y[rng.permutation(rows)]
        res = eng.with_labels(Yp).nested_scheme_lofo(list(schemes), PROMOTION_MARGIN)
        scores.append(float(res["fold_mean"]))
    return scores


def gold_label_support(Yg, fold_g, targets):
    """[v2.7 ADD] Positives per fold x target.  Cells with 0 positives or 0 negatives make that
    fold's AUC undefined; such cells are silently dropped by the nan-mean, so we expose them."""
    rows, undefined = [], []
    for f in range(N_FOLDS):
        m = fold_g == f
        pos = Yg[m].sum(0)
        neg = m.sum() - pos
        rows.append(pd.Series(pos.astype(int), index=targets, name=f"fold{f} (n={int(m.sum())})"))
        undefined += [(f, targets[j]) for j in range(len(targets)) if pos[j] == 0 or neg[j] == 0]
    print(pd.DataFrame(rows).to_string())
    n_cells = N_FOLDS * len(targets)
    if undefined:
        print(f"   WARNING: {len(undefined)}/{n_cells} fold x target AUC cells are undefined and are dropped "
              f"from the fold means (first few: {undefined[:6]}).")
    return dict(n_undefined_cells=len(undefined), n_cells=n_cells, undefined=undefined)


# =====================================================================================
# 8. MAIN
# =====================================================================================
def main():
    global TARGET_COLS
    t0 = time.time()
    require(N_GOLD == sum(EXPECTED_GOLD_FOLD_COUNTS),
            f"N_GOLD={N_GOLD} does not match EXPECTED_GOLD_FOLD_COUNTS sum={sum(EXPECTED_GOLD_FOLD_COUNTS)}")
    require(CALIBRATION_SCOPE in ("per_fold", "global"), f"Invalid CALIBRATION_SCOPE={CALIBRATION_SCOPE!r}")
    # [v2.7] The nested engine re-fits a *global* reference on training folds. A per-fold scope would
    # normalise each held-out fold with its own distribution, which is a different (transductive,
    # fold-local) transform, so it is not supported for validation.
    require(CALIBRATION_SCOPE == "global",
            "CALIBRATION_SCOPE='per_fold' is not supported by the strictly nested engine; use 'global'")
    require(STANDARDIZE_P3 in ("p2_fold", "gold_pooled"), f"Invalid STANDARDIZE_P3={STANDARDIZE_P3!r}")
    require(W_CO_RESPONSE in ("pred", "pseudo_label"), f"Invalid W_CO_RESPONSE={W_CO_RESPONSE!r}")
    _selftest_auc()

    # Fail fast on the operator-level gate before any heavy work (FIX 5).
    auth_gold_uids, auth_gold_path = resolve_gold_reference()

    # -------------------------------------------------------------------------------
    section("PART 1 | FILE DISCOVERY, AUTO-UNZIP & STRICT ALIGNMENT AUDIT")
    # -------------------------------------------------------------------------------
    ensure_drive()
    os.makedirs(LOCAL_OUT_DIR, exist_ok=True)
    list_dir(P2_DIR)
    list_dir(P3_DIR)
    files2, files3 = sorted(os.listdir(P2_DIR)), sorted(os.listdir(P3_DIR))

    folds_path = first_existing(FOLDS_CSV_CANDIDATES, "phase2_folds.csv")
    fdf = pd.read_csv(folds_path)
    fdf.columns = [str(c).strip() for c in fdf.columns]
    require({"StudyInstanceUID", "fold"} <= set(fdf.columns), f"{folds_path} needs StudyInstanceUID and fold columns")
    fdf["StudyInstanceUID"] = fdf["StudyInstanceUID"].astype(str).str.strip()
    require(len(fdf) == N_P2_ROWS and fdf["StudyInstanceUID"].is_unique and fdf[["StudyInstanceUID", "fold"]].notna().all().all(),
            f"{folds_path}: need {N_P2_ROWS} unique, NaN-free rows")
    fold_num = pd.to_numeric(fdf["fold"], errors="coerce").to_numpy(dtype=float)
    require(np.isfinite(fold_num).all() and np.equal(fold_num, np.floor(fold_num)).all(),
            f"{folds_path}: fold values must be finite integers")
    require(set(np.unique(fold_num.astype(int))) == set(range(N_FOLDS)),
            f"{folds_path}: expected fold IDs 0..{N_FOLDS-1}")
    fdf["fold"] = fold_num.astype(int)
    folds_map = pd.Series(fdf["fold"].values, index=fdf["StudyInstanceUID"].values)
    print(f"\nfolds file : {folds_path}  ({len(fdf)} rows, fold sizes {folds_map.value_counts().sort_index().tolist()})")

    fused_path = first_existing(FUSED_MASTER_CANDIDATES, "phase2_fused_master.csv")
    fused_raw = pd.read_csv(fused_path)
    fused_raw.columns = [str(c).strip() for c in fused_raw.columns]
    require("StudyInstanceUID" in fused_raw.columns, "phase2_fused_master.csv has no StudyInstanceUID")
    fused_raw["StudyInstanceUID"] = fused_raw["StudyInstanceUID"].astype(str).str.strip()
    require(fused_raw["StudyInstanceUID"].is_unique, "phase2_fused_master.csv has duplicate StudyInstanceUIDs")
    require(set(folds_map.index) <= set(fused_raw["StudyInstanceUID"]),
            "phase2_folds.csv contains StudyInstanceUIDs absent from phase2_fused_master.csv")
    n_dropped_no_planes = len(fused_raw) - N_P2_ROWS
    if n_dropped_no_planes > 0:
        print(f"note: phase2_fused_master.csv has {len(fused_raw)} rows ({n_dropped_no_planes} zero-plane rows "
              f"excluded from the {N_P2_ROWS}-row OOF set, preserved untouched for Gen-2 export).")
    fused = fused_raw.loc[fused_raw["StudyInstanceUID"].isin(folds_map.index)].copy().reset_index(drop=True)
    require(len(fused) == N_P2_ROWS, f"Aligned fused master has {len(fused)} rows, expected {N_P2_ROWS}")

    patient_candidates = ("PatientID", "patient_id", "patientid", "Patient_ID", "patient")
    fused_patient_col = next((c for c in patient_candidates if c in fused.columns), None)
    folds_patient_col = next((c for c in patient_candidates if c in fdf.columns), None)
    if fused_patient_col is not None:
        patient_source = fused[["StudyInstanceUID", fused_patient_col]].rename(columns={fused_patient_col: "__PatientID"})
        patient_col = fused_patient_col
    elif folds_patient_col is not None:
        patient_source = fdf[["StudyInstanceUID", folds_patient_col]].rename(columns={folds_patient_col: "__PatientID"})
        patient_col = folds_patient_col
    else:
        patient_source = None
        patient_col = None
    patient_audit_status = "skipped"
    if patient_source is None:
        require(not REQUIRE_PATIENT_ID,
                "No PatientID column found in fused master or fold map. Patient-level leakage cannot be audited. "
                "Add a canonical PatientID mapping, or explicitly set RSNA_REQUIRE_PATIENT_ID=0 only after verifying grouping.")
        print("WARNING: Patient-level grouping audit SKIPPED by explicit RSNA_REQUIRE_PATIENT_ID=0 override.")
    else:
        patient_audit = patient_source.merge(
            fdf[["StudyInstanceUID", "fold"]], on="StudyInstanceUID", how="left", validate="one_to_one")
        require(patient_audit["__PatientID"].notna().all(), f"'{patient_col}' contains missing patient IDs")
        patient_fold_n = patient_audit.groupby("__PatientID")["fold"].nunique()
        require((patient_fold_n == 1).all(), "Patient leakage across folds detected!")
        n_patients = int(patient_audit["__PatientID"].nunique())
        print(f"Patient grouping audit passed using '{patient_col}' ({n_patients} patients / {len(patient_audit)} studies).")
        patient_audit_status = "passed"
        # [v2.7 ADD] A one-patient-per-study column makes the grouping check vacuous (it can pass
        # trivially if the column is really a study-level proxy).
        if n_patients == len(patient_audit):
            print("   WARNING: every study has a unique patient ID, so this audit cannot detect cross-fold "
                  "leakage. Confirm the column is a true patient identifier.")

    print(f"fused master: {fused_path}  ({len(fused)} aligned OOF rows / {len(fused_raw)} total CSV rows)")
    fused_gold = resolve_gold_mask(fused, "fused_master_aligned")
    fused_gold_uids = set(fused.loc[fused_gold, "StudyInstanceUID"])
    gold_recon = reconcile_gold_cohort(fused_gold_uids, folds_map, fused_raw, auth_gold_uids)   # FIX 5

    # --- discover Phase-2 OOF files ------------------------------------------------------
    print("\nPhase-2 OOF discovery:")
    cand_map = {}
    for b in BACKBONES:
        cands = candidate_p2_files(b, [P2_DIR, "/content"])
        require(cands, f"no oof_{b['stem']}*.csv found in {P2_DIR} or /content")
        cand_map[b["name"]] = cands
        print(f"   {b['name']:<36} candidates: {[os.path.basename(x) for x in cands]}")

    if TARGET_COLS is None:
        first = pd.read_csv(cand_map[BACKBONES[0]["name"]][0], nrows=5)
        first.columns = [str(c).strip() for c in first.columns]
        TARGET_COLS = infer_target_cols(first)
    TARGET_COLS = list(TARGET_COLS)
    T, C = TARGET_COLS, len(TARGET_COLS)
    require(C == 12, f"expected 12 targets, got {C}: {T}")
    missing_fused = [c for c in T if c not in fused.columns]
    require(not missing_fused, f"targets missing from phase2_fused_master.csv: {missing_fused}")

    loaded = {}
    for b in BACKBONES:
        valid_candidates = []
        for fpath in cand_map[b["name"]]:
            try:
                o = read_phase2_oof(fpath, T)
            except (AuditError, ValueError, KeyError) as e:
                print(f"   skip {os.path.basename(fpath)}: {e}")
                continue
            if o["n_raw"] == N_P2_ROWS:
                valid_candidates.append(o)
        require(valid_candidates, f"no valid Phase-2 OOF file for {b['name']}")
        chosen = valid_candidates[0]
        b_fname = os.path.basename(chosen["path"])
        b["timm_name"] = b_fname[4:-4] if (b_fname.startswith("oof_") and b_fname.endswith(".csv")) else b["name"]
        loaded[b["name"]] = chosen
        note = f"  (logits RECOVERED from preds for {len(chosen['recovered'])} targets)" if chosen["recovered"] else ""
        print(f"   {b['name']:<36} -> {b_fname} (timm_name={b['timm_name']}){note}")

    # --- Phase-2 audit ---------------------------------------------------------------------
    print("\nPhase-2 audit (rows / duplicates / NaNs / folds vs phase2_folds.csv / gold):")
    for b in BACKBONES:
        o = loaded[b["name"]]
        df = o["df"]
        if df["is_gold"].isna().any():
            df["is_gold"] = df["StudyInstanceUID"].isin(fused_gold_uids)
        else:
            df["is_gold"] = df["is_gold"].astype(bool)
        model_gold_uids = set(df.loc[df["is_gold"], "StudyInstanceUID"])
        require(model_gold_uids == fused_gold_uids,
                f"[{b['name']}] gold mask UID set disagrees with canonical fused-master gold UID set")
        audit_phase2_frame(b["name"], df, T, folds_map)
        print(f"   OK  {b['name']:<36} rows={len(df)}  dups=0  NaN=0  fold-map identical  gold={int(df['is_gold'].sum())}")

    ref_name = BACKBONES[0]["name"]
    ref_df = loaded[ref_name]["df"]
    uids = ref_df["StudyInstanceUID"].tolist()
    for b in BACKBONES[1:]:
        d = loaded[b["name"]]["df"]
        if d["StudyInstanceUID"].tolist() != uids:
            print(f"   note: [{b['name']}] row order re-aligned by StudyInstanceUID to match {ref_name}")
            loaded[b["name"]]["df"] = d.set_index("StudyInstanceUID").loc[uids].reset_index()

    ref_gold_order = ref_df.loc[ref_df["is_gold"], "StudyInstanceUID"].tolist()
    require(len(ref_gold_order) == N_GOLD and set(ref_gold_order) == fused_gold_uids,
            "Gold UID set in OOF != gold UID set in phase2_fused_master.csv")
    print(f"   OK  gold UID order aligned across all 4 Phase-2 OOF files (n={N_GOLD})")

    FOLD = folds_map.reindex(uids).values.astype(int)
    GOLD = ref_df["is_gold"].values.astype(bool)
    GI = np.where(GOLD)[0]
    NG = np.where(~GOLD)[0]
    fold_g = FOLD[GI]
    gold_uids = [uids[i] for i in GI]
    gold_counts = [int((fold_g == f).sum()) for f in range(N_FOLDS)]
    require(gold_counts == EXPECTED_GOLD_FOLD_COUNTS, f"gold fold counts {gold_counts} != expected {EXPECTED_GOLD_FOLD_COUNTS}")
    print(f"   OK  gold per-fold counts {gold_counts}   (non-gold rows: {len(NG)})")

    L2 = np.stack([np.column_stack([loaded[b["name"]]["df"][f"{c}_logit"].values for c in T]) for b in BACKBONES])
    P2raw = np.stack([np.column_stack([loaded[b["name"]]["df"][f"{c}_pred"].values for c in T]) for b in BACKBONES])
    M = len(BACKBONES)

    # --- gold labels: fused master is canonical; cross-check every embedded copy ----------------
    fused_idx = fused.set_index("StudyInstanceUID")
    Yg_raw = fused_idx.loc[gold_uids, T].apply(pd.to_numeric, errors="coerce").values.astype(float)
    require(np.isfinite(Yg_raw).all() and np.isin(Yg_raw, [0.0, 1.0]).all(),
            "Gold labels in phase2_fused_master.csv must be finite binary 0/1")
    Yg = Yg_raw.astype(float)
    positives = Yg.sum(0)
    require(((positives > 0) & (positives < len(Yg))).all(), "Every target needs both classes in the pooled gold set")
    print(f"   gold positives per target: {dict(zip(T, Yg.sum(0).astype(int).tolist()))}")
    print("\nGold label support (positives per fold):")
    label_support = gold_label_support(Yg, fold_g, T)       # [v2.7 ADD]

    for b in BACKBONES:
        lab = loaded[b["name"]]["labels"]
        if lab is not None:
            lab_g = lab.loc[gold_uids, T].values
            require(np.allclose(lab_g, Yg_raw, atol=1e-5, equal_nan=False),
                    f"[{b['name']}] gold labels in Phase-2 OOF differ from phase2_fused_master.csv")
            print(f"   OK  labels: [{b['name']}] Phase-2 gold labels identical to fused master")

    # --- Phase 3 gold OOF ------------------------------------------------------------------
    print("\nPhase-3 gold OOF discovery:")
    P3 = {}
    for b in BACKBONES:
        r = load_phase3_gold_oof(b["name"], T, folds_map=folds_map)
        d = r["df"]
        require(len(d) == N_GOLD and d["StudyInstanceUID"].is_unique,
                f"[{b['name']}] Phase-3 gold OOF must have {N_GOLD} unique rows")
        require(set(d["StudyInstanceUID"]) == set(gold_uids),
                f"[{b['name']}] Phase-3 gold UID set != Phase-2 gold UID set")
        pos = pd.Series(np.arange(N_GOLD), index=d["StudyInstanceUID"].values).loc[gold_uids].values
        d = d.iloc[pos].reset_index(drop=True)
        expected_p3_fold = folds_map.reindex(d["StudyInstanceUID"]).to_numpy(dtype=int)
        if "fold" in d.columns:
            actual_p3_fold = pd.to_numeric(d["fold"], errors="coerce").to_numpy(dtype=float)
            require(np.isfinite(actual_p3_fold).all() and np.equal(actual_p3_fold, np.floor(actual_p3_fold)).all(),
                    f"[{b['name']}] Phase-3 fold values are missing/non-integer")
            require(np.array_equal(actual_p3_fold.astype(int), expected_p3_fold),
                    f"[{b['name']}] Phase-3 fold IDs disagree with phase2_folds.csv")
            d["fold"] = actual_p3_fold.astype(int)
        else:
            # We can align labels to the canonical map, but this does NOT prove the
            # upstream checkpoint excluded each row. Source provenance must be verified.
            require(not REQUIRE_EXPLICIT_P3_FOLD,
                    f"[{b['name']}] Phase-3 file has no fold column; strict fold verification is enabled")
            d["fold"] = expected_p3_fold
            print(f"   WARNING: [{b['name']}] no stored Phase-3 fold column; assigned canonical fold by UID only. "
                  "Upstream OOF exclusion is not independently proven.")
        cols = [c for c in d.columns if c.startswith("logit_")]
        require(cols and np.isfinite(d[cols].values).all(), f"[{b['name']}] missing or non-finite Phase-3 logits")
        if r["labels"] is not None:
            p3_labels = r["labels"]   # always a UID-indexed DataFrame from the loaders
            require(set(gold_uids) <= set(p3_labels.index.astype(str)),
                    f"[{b['name']}] embedded Phase-3 labels do not cover the canonical gold UIDs")
            lab_g = p3_labels.loc[gold_uids, T].to_numpy(dtype=float)
            require(np.isfinite(lab_g).all() and np.isin(lab_g, [0.0, 1.0]).all() and
                    np.array_equal(lab_g, Yg_raw),
                    f"[{b['name']}] embedded Phase-3 labels disagree with canonical phase2_fused_master.csv labels")
            print(f"   OK  labels: [{b['name']}] embedded Phase-3 labels identical to fused master")
        else:
            print(f"   note: [{b['name']}] Phase-3 source carries no embedded labels; integrity check relies on the fused master")
        P3[b["name"]] = dict(df=d, streams=r["streams"], source=r["source"], path=r.get("path"))
        print(f"   OK  {b['name']:<36} {r['source']}  streams={r['streams']}  fold-map checked")
    print("   OK  all audits passed -- continuing.")

    # -------------------------------------------------------------------------------
    section(f"PART 2 | DEPLOYMENT CALIBRATION ON THE FULL OOF COHORT (scope={CALIBRATION_SCOPE}, labels never touched)")
    # -------------------------------------------------------------------------------
    # NOTE: this full-cohort calibration is used for deployment export and descriptive diagnostics only.
    # Every validation number in Part 5 re-fits calibration on training folds (Ensembler.cal).
    CAL = {b["name"]: fold_calibrate(L2[i], FOLD, scope=CALIBRATION_SCOPE) for i, b in enumerate(BACKBONES)}
    Z = np.stack([CAL[b["name"]]["Z"] for b in BACKBONES])
    R = np.stack([CAL[b["name"]]["R"] for b in BACKBONES])
    print(f"Population per fold (all OOF, ignoring labels): {CAL[ref_name]['n_f'].tolist()}  "
          f"vs gold-only fold sizes {gold_counts}")
    rows = []
    for b in BACKBONES:
        c = CAL[b["name"]]
        rows.append(dict(
            backbone=b["short"],
            z_mean_absmax=float(np.abs(c["Z"].mean(0)).max()),
            z_std_mean=float(c["Z"].std(0).mean()),
            rank_min=float(c["R"].min()),
            rank_max=float(c["R"].max()),
            fold_mu_spread_median=float(np.median(c["mu_f"].std(0) / (c["sd_g"] + 1e-9))),
        ))
    print(fmt(pd.DataFrame(rows).set_index("backbone"), 4))

    # -------------------------------------------------------------------------------
    section("PART 3 | MULTI-RESOLUTION STREAM COMPARISON (mean fold macro AUC; descriptive, no fitting)")
    # -------------------------------------------------------------------------------
    Z2g = Z[:, GI, :]
    Z2g_fold = np.stack([CAL[b["name"]]["Z_fold"][GI] for b in BACKBONES])
    mr_rows, mr_store, drift_rows = [], {}, []
    P3Z_last_list, L3_last_list = [], []
    for i, b in enumerate(BACKBONES):
        d, cal = P3[b["name"]]["df"], CAL[b["name"]]
        zs = {}
        for s in ("base", "best", "last"):
            if s not in P3[b["name"]]["streams"]:
                continue
            L3 = np.column_stack([d[f"logit_{s}__{c}"].values for c in T])
            if STANDARDIZE_P3 == "p2_fold":
                zs[s] = (L3 - cal["mu_f"][fold_g]) / (cal["sd_f"][fold_g] + EPS_Z)
            else:
                zs[s] = (L3 - L3.mean(0)) / (L3.std(0) + EPS_Z)
            if s == "last":
                L2g = L2[i][GI]
                L3_last_list.append(L3)
                drift_rows.append(dict(
                    backbone=b["short"],
                    std_ratio_P3last_over_P2_median=float(np.median(L3.std(0) / (L2g.std(0) + 1e-9))),
                    mean_shift_in_std_median=float(np.median(np.abs(L3.mean(0) - L2g.mean(0)) / (L2g.std(0) + 1e-9))),
                ))
        P3Z_last_list.append(zs["last"])
        is_single_v1 = "single_stream_v1" in P3[b["name"]]["streams"]
        r = dict(backbone=b["short"], res=f"{b['p2_res']}/{b['p3_res']}", p3_schema="v1_single" if is_single_v1 else "v2_multi")
        r["P2_raw_fold_mean"] = fold_mean_macro_auc(L2[i][GI], Yg, fold_g)
        r["P2_fold_calib"] = fold_mean_macro_auc(Z2g_fold[i], Yg, fold_g)
        r["P3_last"] = fold_mean_macro_auc(zs["last"], Yg, fold_g)
        r["P3_base"] = fold_mean_macro_auc(zs["base"], Yg, fold_g) if "base" in zs else np.nan
        r["P3_best [BIAS]"] = np.nan if (is_single_v1 or "best" not in zs) else fold_mean_macro_auc(zs["best"], Yg, fold_g)
        r["MultiRes_50_50"] = fold_mean_macro_auc(0.5 * Z2g_fold[i] + 0.5 * zs["last"], Yg, fold_g)
        r["MultiRes_gain"] = r["MultiRes_50_50"] - max(r["P2_fold_calib"], r["P3_last"])
        mr_rows.append(r)
        mr_store[b["name"]] = {k: v for k, v in r.items() if k not in ("backbone", "res")}
    Z_p3_gold = np.stack(P3Z_last_list)      # diagnostics only (Part 4); the engine takes raw logits instead
    L3_gold_raw = np.stack(L3_last_list)     # (M, G, C) raw P3-last logits for the nested engine (Scheme E)
    mr = pd.DataFrame(mr_rows).set_index("backbone")
    print(fmt(mr, 4))
    print("\nP3-vs-P2 logit scale drift on the gold studies:")
    print(fmt(pd.DataFrame(drift_rows).set_index("backbone"), 3))

    # -------------------------------------------------------------------------------
    section("PART 4 | DIVERSITY & ERROR-CORRELATION AUDIT")
    # -------------------------------------------------------------------------------
    names_short = [b["short"] for b in BACKBONES]
    cnn = [i for i, b in enumerate(BACKBONES) if b["family"] == "CNN"]
    vit = [i for i, b in enumerate(BACKBONES) if b["family"] != "CNN"]

    def mean_corr(Zs, kind):
        acc, cnt = np.zeros((M, M)), 0
        for c in range(Zs.shape[2]):
            X = Zs[:, :, c].T
            if kind == "spearman":
                X = rankdata(X, axis=0)
            cm = np.corrcoef(X, rowvar=False)
            if np.isfinite(cm).all():
                acc += cm
                cnt += 1
        return acc / max(cnt, 1)

    div_store = {}
    for label, Zs in (("ALL 4,405 STUDIES", Z), ("GOLD STUDIES", Z2g)):
        for kind in ("pearson", "spearman"):
            cm = mean_corr(Zs, kind)
            print(f"\n{kind.capitalize()} correlation on standardized logits, mean over {C} targets -- {label}")
            print(fmt(pd.DataFrame(cm, index=names_short, columns=names_short), 3))
            cc = np.mean([cm[i, j] for ii, i in enumerate(cnn) for j in cnn[ii + 1:]])
            cv = np.mean([cm[i, j] for i in cnn for j in vit])
            ok = cv < cc - 0.03
            print(f"   mean corr(CNN_i, CNN_j) = {cc:.3f}   mean corr(CNN, DINOv2) = {cv:.3f}   "
                  f"{'VERIFIED: DINOv2 is substantially more diverse' if ok else 'NOTE: similar correlation'}")
            div_store[f"{label}|{kind}"] = dict(cnn_cnn=float(cc), cnn_dino=float(cv), verified=bool(ok))

    # Descriptive baselines: no fitted weights; the global calibration is unsupervised.
    # (The ORACLE column picks the best model per target on the same labels -> optimistic by design.)
    S0 = expit(Z2g)
    auc_tab = np.column_stack([fold_mean_auc_cols(Z2g[i], Yg, fold_g) for i in range(M)])
    eq_auc = fold_mean_auc_cols(S0.mean(0), Yg, fold_g)
    eq_mr_auc = fold_mean_auc_cols(0.5 * S0.mean(0) + 0.5 * expit(Z_p3_gold).mean(0), Yg, fold_g)
    best_i = np.nanargmax(np.where(np.isnan(auc_tab), -1, auc_tab), axis=1)
    oracle = auc_tab[np.arange(C), best_i]
    tab = pd.DataFrame(auc_tab, index=T, columns=names_short)
    tab["ORACLE_best*"] = oracle
    tab["oracle_pick"] = [names_short[i] for i in best_i]
    tab["Equal_4fam_P2"] = eq_auc
    tab["Equal_4fam_P2+P3"] = eq_mr_auc
    macro = {n: macro_nanmean(tab[n].values) for n in names_short + ["ORACLE_best*", "Equal_4fam_P2", "Equal_4fam_P2+P3"]}
    print("\nPer-target GOLD OOF AUC (mean fold AUC; pooled scores are not the primary metric):")
    out = tab.copy()
    out.loc["MACRO"] = [macro.get(c, np.nan) if c != "oracle_pick" else "" for c in out.columns]
    print(out.to_string(float_format=lambda v: f"{v:.4f}"))

    # -------------------------------------------------------------------------------
    section("PART 5 | STRICTLY NESTED LOFO: CALIBRATION + SCHEME + HYPERPARAMETER SELECTION")
    # -------------------------------------------------------------------------------
    pseudo_mat = None
    if W_CO_RESPONSE == "pseudo_label":
        pseudo_mat = fused_idx.loc[uids, T].apply(pd.to_numeric, errors="coerce").values.astype(float)
    eng = Ensembler(L2, FOLD, GI, Yg, fold_g, L3_gold=L3_gold_raw, w_co_target=pseudo_mat)
    all_folds = list(range(N_FOLDS))

    # Consistency check: with every fold as reference the engine must reproduce the Part-2
    # deployment calibration exactly (guards against drift between the two code paths).
    Zc, Rc, _, _ = eng.cal(all_folds)
    require(np.allclose(Zc, Z, atol=1e-10) and np.allclose(Rc, R, atol=1e-10),
            "Nested-engine full-cohort calibration differs from the Part-2 deployment calibration")
    print("   OK  engine calibration with all folds == deployment calibration (nested path verified)")

    schemes = {
        "A": "Uniform 4-family (P2)",
        "B": "Global performance-weighted simplex (P2)",
        "C": "L2-shrunk per-target softmax (P2)",
        "D": "Shrunk per-target + Co-occurrence prior (P2)",
        "E": "Multi-Res P2+P3_last shrunk per-target softmax",
    }

    fixed_scheme_results = {scheme: eng.lofo(scheme) for scheme in schemes}
    diag_rows = []
    for scheme, name in schemes.items():
        diag_rows.append(dict(
            scheme=f"{scheme}: {name}",
            mean_fold_macro_AUC=fixed_scheme_results[scheme]["fold_mean"],
            pooled_AUC_diagnostic_only=fixed_scheme_results[scheme]["pooled_auc_diagnostic"],
        ))
    print("\nPer-scheme outer LOFO diagnostics (primary metric = mean of per-fold macro AUCs):")
    print(fmt(pd.DataFrame(diag_rows).set_index("scheme"), 4))

    # Research comparison may include E, but the exported non-gold prediction set only
    # has P2 OOF predictions. Thus Scheme E is not eligible for final pseudo-label fit.
    nested_all_result = eng.nested_scheme_lofo(list(schemes.keys()), PROMOTION_MARGIN)
    nested_all_score = nested_all_result["fold_mean"]
    nested_result = eng.nested_scheme_lofo(list(DEPLOYMENT_SCHEMES), PROMOTION_MARGIN)
    nested_score = nested_result["fold_mean"]
    print(f"\nNested all-scheme LOFO diagnostic (includes gold-only P3 Scheme E): "
          f"mean-fold macro AUC = {nested_all_score:.4f}")
    print(f"Nested deployable-scheme LOFO (P2 predictions available for non-gold rows): "
          f"mean-fold macro AUC = {nested_score:.4f}")
    for item in nested_result["chosen"]:
        candidates = ", ".join(f"{name}:{value:.4f}" for name, value in item["candidate_scores"].items())
        print(f"  fold {item['fold']}: selected {item['scheme']} [{describe_params(item['params'])}], "
              f"heldout={item['heldout_macro_auc']:.4f}; inner {{{candidates}}}")

    # [v2.7 ADD] How much of the nested score is signal? Paired bootstrap vs uniform averaging.
    boot = bootstrap_fold_mean_macro_auc([nested_result["P"], fixed_scheme_results["A"]["P"]], Yg, fold_g)
    nb, ub = boot[:, 0], boot[:, 1]
    diff = nb - ub
    ci = lambda x: [float(v) for v in np.nanpercentile(x, [2.5, 97.5])]  # noqa: E731
    boot_summary = dict(
        n_boot=BOOTSTRAP_N,
        nested_deployable_mean=float(np.nanmean(nb)), nested_deployable_ci95=ci(nb),
        uniform_A_mean=float(np.nanmean(ub)), uniform_A_ci95=ci(ub),
        diff_nested_minus_uniform_mean=float(np.nanmean(diff)), diff_ci95=ci(diff),
        p_nested_beats_uniform=float(np.nanmean(diff > 0)),
    )
    print(f"\nStudy-level bootstrap ({BOOTSTRAP_N} reps, paired, fold-stratified):")
    print(f"   nested deployable : {boot_summary['nested_deployable_mean']:.4f}  95% CI {boot_summary['nested_deployable_ci95'][0]:.4f}..{boot_summary['nested_deployable_ci95'][1]:.4f}")
    print(f"   uniform scheme A  : {boot_summary['uniform_A_mean']:.4f}  95% CI {boot_summary['uniform_A_ci95'][0]:.4f}..{boot_summary['uniform_A_ci95'][1]:.4f}")
    print(f"   nested - uniform  : {boot_summary['diff_nested_minus_uniform_mean']:+.4f}  95% CI {boot_summary['diff_ci95'][0]:+.4f}..{boot_summary['diff_ci95'][1]:+.4f}  "
          f"P(nested>uniform)={boot_summary['p_nested_beats_uniform']:.2f}")
    if boot_summary["diff_ci95"][0] <= 0.0:
        print("   NOTE: the CI for (nested - uniform) includes 0; the extra selection machinery is not clearly "
              "better than plain averaging with this few gold studies.")

    # [v2.7 ADD] Label-permutation canary on the full nested pipeline.
    canary = None
    if PERM_CANARY_N > 0:
        t_c = time.time()
        perm_scores = permutation_canary(eng, DEPLOYMENT_SCHEMES, PERM_CANARY_N)
        perm_mean = float(np.mean(perm_scores))
        canary = dict(n_perm=PERM_CANARY_N, scores=perm_scores, mean=perm_mean, warn_threshold=PERM_CANARY_WARN)
        print(f"\nPermutation canary ({PERM_CANARY_N} within-fold label shuffles, {time.time() - t_c:.0f}s): "
              f"nested macro AUC = {perm_mean:.4f}  {[round(s, 4) for s in perm_scores]}  (expected ~0.50)")
        if perm_mean > PERM_CANARY_WARN:
            print(f"   WARNING: shuffled-label score {perm_mean:.3f} > {PERM_CANARY_WARN}; the pipeline may be "
                  "leaking information. Investigate before trusting the nested estimate.")

    final_scheme, final_candidates, _ = eng.choose_scheme(list(DEPLOYMENT_SCHEMES), all_folds, PROMOTION_MARGIN)
    params = final_candidates[final_scheme]["params"]
    alpha = float(params["alpha"])
    eta = float(params.get("eta", 0.0)) if final_scheme == "D" else 0.0
    beta_p3 = float(params.get("beta_p3", 0.0)) if final_scheme == "E" else 0.0
    # Deployment fit: calibration, weights and prior use the COMPLETE OOF cohort (kept separate
    # from every validation fit above).
    W = eng.fit_weights(final_scheme, params, all_folds)
    require(W.shape == (M, C) and np.isfinite(W).all(), "Final ensemble weights have invalid shape or values")
    require((W >= -1e-10).all() and np.allclose(W.sum(axis=0), 1.0, atol=1e-8),
            "Final ensemble weights must be non-negative and sum to one per target")
    final_inner_score = float(final_candidates[final_scheme]["inner_score"])

    p_spec, p_spec_score = eng.select("E", all_folds)
    require(p_spec is not None and np.isfinite(p_spec_score),
            "Scheme E diagnostic weights could not be selected from finite inner-CV scores")
    W_spec = eng.fit_weights("E", p_spec, all_folds)

    print(f"\nFINAL DEPLOYMENT CHOICE: Scheme {final_scheme} ({schemes[final_scheme]}), "
          f"params [{describe_params(params)}], inner-CV={final_inner_score:.4f}")
    print("Winning 4x12 weights w[m][c]:")
    print(fmt(pd.DataFrame(W, index=names_short, columns=T), 3))
    if final_scheme == "A":
        print(f"\nReference Target-Specialized Weights (Scheme E: [{describe_params(p_spec)}]):")
        print(fmt(pd.DataFrame(W_spec, index=names_short, columns=T), 3))

    blend_final = eng.predict_all(final_scheme, params, all_folds, gold_only=False)
    # Gold rows get their nested outer-fold scores (each produced with a calibration/weights fit that
    # never saw that fold); non-gold rows get the deployment fit.  No per-fold rescaling is applied.
    blend_export = blend_final.copy()
    blend_export[GI] = nested_result["P"]
    blend_export_raw = blend_export.copy()

    # Scheme A-C scores are naturally in [0,1]. Scheme D's residual correction can
    # exceed that range; clip only for the probability-like pseudo-label column.
    # These are pseudo-label scores, NOT probabilities proven to be calibrated.
    ens_prob = np.clip(blend_export, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    ens_logits = sp_logit(ens_prob)

    # -------------------------------------------------------------------------------
    section("PART 6 | EXPORT: CONFIG, ROLE-LABELED PREDICTIONS, GOLD OOF, GEN-2")
    # -------------------------------------------------------------------------------
    bb_cfg = []
    for i, b in enumerate(BACKBONES):
        c2 = scan_ckpts(P2_DIR, files2, b["stem"])
        c3 = scan_ckpts(P3_DIR, files3, b["stem"])
        use = lambda lst: [x for x in lst if "last" in x.lower()] or lst  # noqa: E731
        bb_cfg.append(dict(
            name=b["name"], timm_name=b.get("timm_name", b["name"]), short=b["short"], family=b["family"],
            phase2_resolution=b["p2_res"], phase3_resolution=b["p3_res"],
            phase2_oof_csv=os.path.basename(loaded[b["name"]]["path"]),
            phase3_gold_oof_source=P3[b["name"]]["source"], phase3_streams=P3[b["name"]]["streams"],
            phase2_checkpoints_discovered=c2, phase2_checkpoints_use=use(c2),
            phase3_checkpoints_discovered=c3, phase3_checkpoints_use=use(c3),
        ))

    # [v2.7 ADD] input provenance: pins the exact bytes this run consumed.
    def _prov(path):
        return dict(path=path, sha256=sha256_file(path)) if path and os.path.isfile(path) else dict(path=path, sha256=None)

    provenance = dict(
        phase2_folds_csv=_prov(folds_path),
        phase2_fused_master=_prov(fused_path),
        phase2_oof={b["name"]: _prov(loaded[b["name"]]["path"]) for b in BACKBONES},
        phase3_gold_oof={b["name"]: dict(source=P3[b["name"]]["source"], **_prov(P3[b["name"]]["path"])) for b in BACKBONES},
        authoritative_gold_uid_file=_prov(auth_gold_path) if auth_gold_path else None,
        patient_audit=patient_audit_status,
    )

    calib = dict(
        calibration_scope=CALIBRATION_SCOPE,
        standardization="global: z=(logit-global_mean)/(global_std+eps); global ECDF reference is used for ranking",
        deployment_fit="full 4,405-study OOF cohort (labels unused)",
        validation_fit="re-fit on training folds only inside every inner/outer split (Ensembler.cal)",
        eps=EPS_Z, alpha=alpha, beta_p3=beta_p3,
        global_z_stats={b["name"]: dict(mean=dict(zip(T, CAL[b["name"]]["mu_g"])),
                                        std=dict(zip(T, CAL[b["name"]]["sd_g"]))) for b in BACKBONES},
        fold_z_stats={b["name"]: dict(mean=CAL[b["name"]]["mu_f"], std=CAL[b["name"]]["sd_f"])
                      for b in BACKBONES},
        ecdf_reference=dict(
            n_reference=N_P2_ROWS,
            sorted_reference_logits_global={b["name"]: {c: CAL[b["name"]]["sorted_ref"][:, j]
                                                        for j, c in enumerate(T)} for b in BACKBONES},
            sorted_reference_logits_by_fold={b["name"]: {
                str(fold): {c: CAL[b["name"]]["sorted_ref_by_fold"][str(fold)][:, j]
                            for j, c in enumerate(T)}
                for fold in range(N_FOLDS)
            } for b in BACKBONES},
        ),
        pseudo_label_note="ensemble pred columns are clipped blend scores for pseudo-label use; not calibrated probabilities",
    )

    Wco, mu_co = eng.wco(alpha, all_folds)
    co_cfg = dict(eta=eta, W_co=Wco if eta > 0 else None, center=mu_co if eta > 0 else None, target_order=T)

    cfg = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), script_version=SCRIPT_VERSION,
        backbones=bb_cfg, target_cols=T, calibration=calib,
        input_provenance=provenance,
        gold_reconciliation=gold_recon,
        gold_label_support=label_support,
        weights={b["name"]: dict(zip(T, W[i])) for i, b in enumerate(BACKBONES)},
        specialized_weights_scheme_E={b["name"]: dict(zip(T, W_spec[i])) for i, b in enumerate(BACKBONES)},
        specialized_weights_scheme_E_params=p_spec,
        specialized_weights_scheme_E_inner_cv_macro_auc=float(p_spec_score),
        selection=dict(
            scheme=final_scheme, scheme_name=schemes[final_scheme], params=params,
            candidate_inner_scores={s: v["inner_score"] for s, v in final_candidates.items()},
            nested_selection_lofo_mean_fold_macro_auc=nested_score,
            nested_all_schemes_mean_fold_macro_auc=nested_all_score,
            pooled_auc_is_diagnostic_only=True,
            strictly_nested_calibration=True,
            deployment_scheme_candidates=list(DEPLOYMENT_SCHEMES),
            scheme_E_is_gold_only_diagnostic=True,
            nested_deployment_outer_fold_choices=nested_result["chosen"],
        ),
        uncertainty=dict(bootstrap=boot_summary, permutation_canary=canary),
        co_occurrence_matrix=co_cfg,
        oof_metrics=dict(
            gold_macro_auc_individual_p2={b["name"]: macro[b["short"]] for b in BACKBONES},
            equal_weight_4family_p2_macro_auc=macro["Equal_4fam_P2"],
            equal_weight_4family_multires_macro_auc=macro["Equal_4fam_P2+P3"],
            oracle_best_single_macro_auc=macro["ORACLE_best*"],
            per_scheme_lofo_mean_fold_macro_auc={s: fixed_scheme_results[s]["fold_mean"] for s in schemes},
            per_scheme_pooled_auc_diagnostic_only={s: fixed_scheme_results[s]["pooled_auc_diagnostic"] for s in schemes},
            nested_deployable_scheme_lofo_mean_fold_macro_auc=nested_score,
            nested_all_scheme_lofo_mean_fold_macro_auc=nested_all_score,
            multires_stream_comparison=mr_store,
            diversity=div_store,
        ),
    )

    role = np.full(N_P2_ROWS, "non_gold_final_fit_pseudolabel", dtype=object)
    role[GI] = "gold_nested_outer_oof"
    ens_df = pd.DataFrame({"StudyInstanceUID": uids, "fold": FOLD, "is_gold": GOLD, "prediction_role": role})
    for j, c in enumerate(T):
        ens_df[f"{c}_logit"] = ens_logits[:, j]
        ens_df[f"{c}_pred"] = ens_prob[:, j]
        ens_df[f"{c}_blend_score"] = blend_export[:, j]
        ens_df[f"{c}_blend_score_raw"] = blend_export_raw[:, j]
    gold_oof_df = ens_df.loc[ens_df["is_gold"]].copy().reset_index(drop=True)

    gen2 = fused_raw.copy()
    # Canonical UID membership avoids status-substring false positives such as
    # "non_gold_pseudolabel". Gold rows are never overwritten.
    gen2_uids = gen2["StudyInstanceUID"].astype(str).str.strip()
    require(gen2_uids.is_unique, "Gen-2 master has duplicate StudyInstanceUIDs")
    missing_gold_from_gen2 = fused_gold_uids - set(gen2_uids)
    require(not missing_gold_from_gen2, f"Gen-2 master is missing canonical gold UIDs: {list(missing_gold_from_gen2)[:5]}")
    canonical_gold_mask = gen2_uids.isin(fused_gold_uids).to_numpy(dtype=bool)
    require(int(canonical_gold_mask.sum()) == N_GOLD,
            f"Canonical OOF gold UID set matched {int(canonical_gold_mask.sum())} Gen-2 rows; expected {N_GOLD}")
    # Also preserve additional explicitly marked gold studies that may not be part
    # of the 4,405-row OOF subset (for example, a study omitted for missing planes).
    try:
        metadata_gold_mask = resolve_gold_mask(gen2, "gen2_full", expected_n=None)
    except AuditError as exc:
        print(f"   note: no extra Gen-2 gold-status metadata usable ({exc}); canonical OOF gold UID set remains protected")
        metadata_gold_mask = np.zeros(len(gen2), dtype=bool)
    # [v2.7] Studies on the authoritative gold list are protected too, even if the master's own
    # metadata fails to flag them.
    authoritative_mask = (gen2_uids.isin(auth_gold_uids).to_numpy(dtype=bool)
                          if auth_gold_uids is not None else np.zeros(len(gen2), dtype=bool))
    gmask_raw = canonical_gold_mask | metadata_gold_mask | authoritative_mask
    require(np.all(gmask_raw[canonical_gold_mask]), "Canonical OOF gold UID set is not fully protected")
    in_oof_mask = gen2_uids.isin(set(uids)).values
    upd_mask = in_oof_mask & (~gmask_raw)
    uid_to_ens_idx = pd.Series(np.arange(N_P2_ROWS), index=uids)
    ens_rows_for_upd = uid_to_ens_idx.loc[gen2.loc[upd_mask, "StudyInstanceUID"].values].values

    gold_values_before = gen2.loc[gmask_raw, T].copy(deep=True)
    p_old_upd = gen2.loc[upd_mask, T].apply(pd.to_numeric, errors="coerce").values.astype(float)
    require(np.isfinite(p_old_upd).all(), "Existing non-gold Gen-2 label scores contain NaN/inf")
    require(((p_old_upd >= 0.0) & (p_old_upd <= 1.0)).all(),
            "Existing non-gold Gen-2 label scores must be in [0,1]")
    p_new_upd = GEN2_OLD_WEIGHT * p_old_upd + GEN2_NEW_WEIGHT * ens_prob[ens_rows_for_upd]
    require(np.isfinite(p_new_upd).all() and ((p_new_upd >= 0.0) & (p_new_upd <= 1.0)).all(),
            "Blended Gen-2 pseudo-labels are non-finite or outside [0,1]")
    # Assign only the non-gold cells; do not round-trip the whole column through
    # float conversion, which could alter representation of untouched gold cells.
    update_index = gen2.index[upd_mask]
    for j, c in enumerate(T):
        gen2.loc[update_index, c] = p_new_upd[:, j]

    pd.testing.assert_frame_equal(
        gen2.loc[gmask_raw, T].reset_index(drop=True),
        gold_values_before.reset_index(drop=True),
        check_dtype=False, check_exact=True,
    )
    print(f"Gen-2 check: {int(gmask_raw.sum())} detected gold rows bit-identical; "
          f"{int(upd_mask.sum())} non-gold OOF rows updated.")

    targets_out = [
        ("ensemble_config.json", os.path.join(P3_DIR, "ensemble_config.json"),
         lambda p: atomic_write_text(p, json.dumps(jsonable(cfg), indent=2, allow_nan=False))),
        ("oof_4family_ensemble_master.csv", os.path.join(P2_DIR, "oof_4family_ensemble_master.csv"),
         lambda p: atomic_write_csv(p, ens_df)),
        ("oof_4family_ensemble_gold_only.csv", os.path.join(P2_DIR, "oof_4family_ensemble_gold_only.csv"),
         lambda p: atomic_write_csv(p, gold_oof_df)),
        ("phase2_fused_master_gen2.csv", os.path.join(P2_DIR, "phase2_fused_master_gen2.csv"),
         lambda p: atomic_write_csv(p, gen2)),
    ]
    for fname, drive_path, writer in targets_out:
        local = os.path.join(LOCAL_OUT_DIR, fname)
        writer(local)
        digest = atomic_copy(local, drive_path)     # hash-verified, previous file backed up
        print(f"   wrote {fname:<38} -> {drive_path} ({os.path.getsize(drive_path)/1e6:.2f} MB)  sha256={digest[:12]}")

    section("SUMMARY")
    print(f"Final deployment scheme: {final_scheme} ({schemes[final_scheme]}) [{describe_params(params)}]")
    print(f"Equal-weight 4-Family Gold macro AUC -> P2: {macro['Equal_4fam_P2']:.4f} | Multi-Res P2+P3 diagnostic: {macro['Equal_4fam_P2+P3']:.4f}")
    print(f"Nested deployable-scheme LOFO mean-fold gold macro AUC (strictly nested calibration): {nested_score:.4f}  "
          f"95% CI {boot_summary['nested_deployable_ci95'][0]:.4f}..{boot_summary['nested_deployable_ci95'][1]:.4f}")
    if canary is not None:
        print(f"Permutation canary: {canary['mean']:.4f} (expect ~0.50)")
    print("WARNING: exported ensemble pred scores are pseudo-label scores, not calibrated clinical probabilities.")
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()