"""Pure, testable DICOM position helpers extracted from RSNA V5 preprocessing logic.

These routines handle slice geometry only. They do not decode pixels, resize,
handle arbitrary oblique resampling, or serve as a complete DICOM converter.
"""
from __future__ import annotations

import numpy as np

PLANE_NORMALS = {
    'axial': np.array([0., 0., 1.]),
    'coronal': np.array([0., 1., 0.]),
    'sagittal': np.array([1., 0., 0.]),
}


def canonical_slice_normal(iop, plane: str, cos_tol: float = .707) -> np.ndarray:
    """Compute canonical slice normal from 6 DICOM orientation cosines.

    Accept axis-aligned / mildly oblique planes; fail rather than silently
    reshaping strongly oblique anatomy without affine resampling.
    """
    vec = np.asarray(iop, dtype=float)
    if vec.shape != (6,) or not np.isfinite(vec).all():
        raise ValueError('IOP must contain 6 finite direction cosines')
    if plane not in PLANE_NORMALS:
        raise ValueError('unknown anatomical plane')
    row, col = vec[:3], vec[3:]
    if not (np.isclose(np.linalg.norm(row), 1., atol=1e-3)
            and np.isclose(np.linalg.norm(col), 1., atol=1e-3)
            and np.isclose(row @ col, 0., atol=1e-3)):
        raise ValueError('IOP must have orthonormal row and column vectors')
    normal = np.cross(row, col)
    if abs(normal @ PLANE_NORMALS[plane]) < cos_tol:
        raise ValueError('series is too oblique for transpose-and-flip canonicalization')
    if normal @ PLANE_NORMALS[plane] < 0:
        normal = -normal
    return normal


def order_and_sample_slices(ipps, iop, plane: str, count: int = 16):
    """Deduplicate by 0.01mm, order positions, greedily select unique physical slices.

    Returns (source_indices, sampled_projected_mm), both ascending by location.
    """
    if count < 1:
        raise ValueError('count must be positive')
    pos = np.asarray(ipps, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3 or not np.isfinite(pos).all():
        raise ValueError('IPP must have shape [slices,3] and be finite')
    norm = canonical_slice_normal(iop, plane)
    z = pos @ norm
    rounded = np.round(z, 2)
    indices = np.unique(rounded, return_index=True)[1]
    indices = indices[np.argsort(z[indices], kind='stable')]
    coord = z[indices]
    if len(coord) < count:
        raise ValueError(f'{len(coord)} distinct slice locations; need {count}')
    if np.any(np.diff(coord) <= 1e-4):
        raise ValueError('duplicate or unordered physical slice positions')
    targets = np.linspace(coord[0], coord[-1], count)
    remaining = list(range(len(indices)))
    selected = []
    for target in targets:
        nearest = min(remaining, key=lambda k: (abs(coord[k] - target), k))
        selected.append(nearest)
        remaining.remove(nearest)
    selected = sorted(selected)
    return indices[selected].astype(int), coord[selected]
