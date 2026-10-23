"""Study-wise loader for permitted, preprocessed RSNA 16x288x288 NPZ volumes.

This loader never downloads, modifies or redistributes the underlying studies.
Manifest columns: StudyInstanceUID, patient_id, fold, sample_weight,
axial_path, coronal_path, sagittal_path, and 12 targets in TARGETS order.
Relative cache paths are resolved under the explicit cache root.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from src.models.common import PLANES, TARGETS, SLICES


def load_plane(path: Path | None, size: int) -> tuple[torch.Tensor, float]:
    if not path or not path.is_file():
        return torch.zeros(SLICES, size, size, dtype=torch.float32), 0.
    with np.load(path, allow_pickle=False) as item:
        img = item['img']
    if img.shape != (SLICES, 288, 288) or img.dtype != np.uint8:
        raise ValueError(f'invalid cached plane {path}: {img.shape}/{img.dtype}')
    volume = img.astype(np.float32) / 255.
    foreground = volume > 0
    if foreground.any():
        values = volume[foreground]
        mean, std = float(values.mean()), max(float(values.std()), 1e-3)
        volume[foreground] = np.clip((values - mean) / std, -5, 5)
    x = torch.from_numpy(volume).unsqueeze(1)
    x = interpolate(x, (size, size), mode='bilinear', align_corners=False,
                    antialias=(size < 288)).squeeze(1)
    x[~torch.isfinite(x)] = 0.
    return x, 1.


class CachedStudyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_root: Path, *, image_size: int,
                 target_cols=TARGETS):
        required = {'StudyInstanceUID', 'patient_id', 'fold', 'sample_weight',
                    *(f'{p}_path' for p in PLANES), *target_cols}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'manifest is missing columns: {sorted(missing)}')
        if frame['StudyInstanceUID'].duplicated().any():
            raise ValueError('duplicate studies in the manifest')
        if frame['patient_id'].isna().any() or frame['patient_id'].astype(str).str.strip().eq('').any():
            raise ValueError('patient identifiers are required to validate folds')
        if image_size < 14 or image_size % 14:
            raise ValueError('image_size must be a positive multiple of 14')
        self.frame = frame.reset_index(drop=True).copy()
        self.root = Path(cache_root).resolve()
        self.image_size = int(image_size)
        self.targets = tuple(target_cols)

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve(self, value) -> Path | None:
        if pd.isna(value) or not str(value).strip():
            return None
        raw = Path(str(value))
        path = (self.root / raw).resolve() if not raw.is_absolute() else raw.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('cache path escapes the configured cache root')
        if not path.is_file():
            raise FileNotFoundError(path)  # Missing data cannot be mistaken for an absent plane.
        return path

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        tensors, mask = [], []
        for p in PLANES:
            vol, found = load_plane(self._resolve(row[f'{p}_path']), self.image_size)
            tensors.append(vol)
            mask.append(found)
        targets = torch.tensor(pd.to_numeric(row.loc[list(self.targets)], errors='coerce')
                               .to_numpy(dtype=np.float32), dtype=torch.float32)
        weight = float(row['sample_weight'])
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('sample_weight must be positive and finite')
        return (*tensors, torch.tensor(mask, dtype=torch.float32),
                targets, torch.tensor(weight, dtype=torch.float32))


def validate_patient_fold_map(frame: pd.DataFrame) -> None:
    """Fail-closed group validation; no StudyUID-as-patient fallback."""
    needed = {'StudyInstanceUID', 'patient_id', 'fold'}
    if needed - set(frame.columns):
        raise ValueError('manifest must include StudyInstanceUID, patient_id, and fold')
    if frame['patient_id'].isna().any():
        raise ValueError('patient_id cannot be null')
    if frame.groupby('patient_id')['fold'].nunique(dropna=False).max() > 1:
        raise ValueError('patient appears in multiple folds')
    if not pd.api.types.is_integer_dtype(frame['fold']):
        raise ValueError('fold must contain integer identifiers')
    if (frame['fold'] < 0).any():
        raise ValueError('fold must be nonnegative')
