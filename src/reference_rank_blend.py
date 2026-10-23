"""Illustrative, validated rank blend for already-computed study predictions.

Educational example only: not the original RSNA inference driver, and not
a byte-identical recreation of the competition prediction pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ID = "StudyInstanceUID"
TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)
WEIGHTS = {"v7": 0.60, "s6": 0.25, "v5": 0.15}


def _validated(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if list(df.columns) != [ID, *TARGETS]:
        raise ValueError(f"{label}: columns must exactly match the 13-column schema/order")
    if df[ID].isna().any() or df[ID].astype(str).str.strip().eq("").any():
        raise ValueError(f"{label}: missing/blank study IDs")
    if df[ID].duplicated().any():
        raise ValueError(f"{label}: duplicate study IDs")
    arr = df.loc[:, TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{label}: NaN or infinite prediction")
    if (arr < 0).any() or (arr > 1).any():
        raise ValueError(f"{label}: predictions must lie in [0,1]")
    return df.copy()


def blend(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average per-target ranks with fixed 60/25/15 weights.

    All experts must score exactly the same studies. Output is for ranking,
    not a calibrated medical probability.
    """
    if set(frames) != set(WEIGHTS):
        raise ValueError("Required experts: v7, s6, v5")
    frames = {name: _validated(df, name) for name, df in frames.items()}
    anchor = frames["v7"][ID].tolist()
    if not anchor:
        raise ValueError("Cannot blend an empty set of studies")
    ordered = {}
    for name, frame in frames.items():
        if set(frame[ID]) != set(anchor):
            raise ValueError(f"{name}: exact study-ID sets do not match")
        ordered[name] = frame.set_index(ID).loc[anchor, list(TARGETS)]
    out = pd.DataFrame({ID: anchor})
    n = len(anchor)
    for target in TARGETS:
        score = np.zeros(n, dtype=np.float64)
        for expert, weight in WEIGHTS.items():
            ranks = ordered[expert][target].rank(method="average").to_numpy() / n
            score += weight * ranks
        out[target] = np.clip(score, 0, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for expert in WEIGHTS:
        parser.add_argument(f"--{expert}", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    frames = {name: pd.read_csv(getattr(args, name)) for name in WEIGHTS}
    result = blend(frames)
    result.to_csv(args.out, index=False)
    print(f"Reference blend written: {args.out} ({len(result)} studies)")


if __name__ == "__main__":
    main()
