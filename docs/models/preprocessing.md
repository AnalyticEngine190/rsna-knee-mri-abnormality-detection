# Canonical DICOM geometry, V5

The V5 DICOM preprocessing experiments aligned axial, coronal, and sagittal series in **patient physical coordinates**, then cached standardized per-plane images for 2.5D training. The original full preprocessing script was a Kaggle-oriented working notebook with environment-specific paths. Instead of publishing it as a claim of a ready-to-run converter, this repo includes a small and independently tested geometry component: [`src/data/geometry.py`](../../src/data/geometry.py).

## Important design decisions

1. Read DICOM `ImageOrientationPatient` and `ImagePositionPatient`, validating finite values and orthogonality.
2. Compute the cross-product slice normal and canonicalize its sign against the expected anatomical axis.
3. Reject strongly oblique scans that would require proper affine resampling rather than naive axis flips.
4. Deduplicate physical slice positions at 0.01 mm, sort in millimeters, and sample 16 physical anchors without reusing source slices.
5. Raise an error for fewer than 16 distinct locations. This fixes a known 10–15-slice failure mode in one historical generator.
6. The full original pipeline additionally performed modality LUT, MONOCHROME1 inversion, whole-series P1/P99 windowing, physical aspect-ratio preserving resize and zero padding to 288×288; **those pixel transformations are not implemented in the pure geometry module**.

## Caveats

No patient scans or DICOM studies are included. This pure geometry helper must not be mistaken for a clinically validated image converter or drop-in replacement for the exact inference-generation code. Unit tests validate coordinate invariants, not an actual scanner study. The original `triplanar_v5_strict_canonical.py` upload contained a malformed import (`mport os`) and a dataset path that needs verification, which are reasons not to publish the raw file unchanged.
