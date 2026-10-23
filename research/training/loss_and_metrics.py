"""Research snapshot of the ResNet50D study-weighted training loss and metrics.

Extracted from the project's five-fold training script. Requires torch, pandas,
numpy, and scikit-learn. Not a standalone training program.
"""
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
TARGET_COLS = (
"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
"Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)

def compute_weighted_loss(outputs, targets, sample_weights, pos_weight=None):
    """Study-level weighted BCE.

    Each study still casts one vote of size ``sample_weight`` (so a sparsely
    labeled silver row cannot outweigh a full gold study just by having one
    label, and a fully labeled study is not counted 12 times). ``pos_weight``
    rebalances labels *inside* that vote. It is None on validation so the
    reported BCE does not change scale every time the cap or the fold changes.
    """
    logits = outputs.float()
    targets = targets.to(device=logits.device, dtype=torch.float32)
    sample_weights = sample_weights.to(device=logits.device, dtype=torch.float32)
    valid = torch.isfinite(targets)
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    # CHANGED: BCE is outside autocast, in fp32.
    # WHY: an explicit .float() inside an open autocast region is not a
    # guarantee; fp16 BCE overflows and returns NaN, and NaN * 0 stays NaN
    # so the valid-label mask cannot save it.
    loss_device_type = logits.device.type
    autocast_off = (
        torch.amp.autocast(device_type=loss_device_type, enabled=False)
        if loss_device_type == "cuda"
        else nullcontext()
    )
    pw = None if pos_weight is None else pos_weight.to(device=logits.device, dtype=torch.float32)
    with autocast_off:
        element_loss = F.binary_cross_entropy_with_logits(
            logits,
            safe_targets,
            pos_weight=pw,
            reduction="none",
        )
    valid_float = valid.to(dtype=element_loss.dtype)
    per_sample_loss = (element_loss * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1.0)
    denominator = sample_weights.sum()
    numerator = (per_sample_loss * sample_weights).sum()
    if float(denominator.detach()) <= 0:
        raise RuntimeError("Sample weights in this batch sum to <= 0; loss would be NaN.")
    loss = numerator / denominator
    return loss, denominator.detach(), numerator.detach()


def compute_pos_weight(df: pd.DataFrame, cap: float) -> torch.Tensor:
    """neg/pos on the training fold only. Soft labels count as fractional mass.

    A class with no positives or no negatives gets weight 1 (there is nothing
    to rebalance). Weights are capped because a 1-in-1000 finding would
    otherwise dominate every other gradient.
    """
    values = []
    for col in TARGET_COLS:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        y = y[np.isfinite(y)]
        if y.size == 0:
            values.append(1.0)
            continue
        clipped = np.clip(y, 0.0, 1.0)
        pos = float(clipped.sum())
        neg = float((1.0 - clipped).sum())
        if pos <= 0.0 or neg <= 0.0:
            values.append(1.0)
        else:
            values.append(float(min(neg / pos, cap)))
    return torch.tensor(values, dtype=torch.float32)


def compute_macro_auc(logits: np.ndarray, targets: np.ndarray):
    per_target = {}
    skipped = []
    for col_idx, col_name in enumerate(TARGET_COLS):
        valid = np.isfinite(targets[:, col_idx])
        y = targets[valid, col_idx]
        scores = logits[valid, col_idx]
        # roc_auc is rank-based, so raw logits match probabilities.
        # Soft labels that are not {0, 1} are not a valid AUC target.
        if y.size == 0 or np.unique(y).size < 2:
            skipped.append(col_name)
            continue
        try:
            per_target[col_name] = float(roc_auc_score(y, scores))
        except ValueError:
            skipped.append(col_name)
    macro_auc = float(np.mean(list(per_target.values()))) if per_target else float("nan")
    return macro_auc, per_target, skipped


def format_metric(value, digits: int = 6) -> str:
    # CHANGED: helper used by every report line.
    # WHY: ``f"{a}" f"{b}" if cond else "undefined"`` binds the conditional to
    # the whole concatenated string, so a NaN AUC used to print only "undefined".
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "undefined"
    if not np.isfinite(value):
        return "undefined"
    return f"{value:.{digits}f}"
