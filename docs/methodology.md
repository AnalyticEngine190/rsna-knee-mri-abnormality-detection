# Technical methodology

## Scope and labels
The V5 work used 4,407 unique training studies and 12 abnormality targets. A tri-planar cache covered 4,405 studies, while 17 studies lacked at least one view. Qwen2.5-7B-Instruct parsed multilingual radiology reports into structured targets, combined with another report-derived source. Soft targets in [0,1] were used for training. The 58 expert-labeled studies were marked `gold_human`, weighted 10× versus 1× for report-derived rows, and used for gold-only validation.

Gold-only validation with 58 studies is a severe statistical limitation, particularly for infrequent abnormalities. Five-fold GroupKFold splits used patient identifiers when present; otherwise study UID was used as the fallback grouping key. This fallback should not be mistaken for verified patient-level independence.

## DICOM geometry and caching
- Deterministic plane and MRI sequence selection, preferring fluid-sensitive series and resolving ties with series metadata.
- Check ImageOrientationPatient, ImagePositionPatient, PixelSpacing, slice shapes, and consistent series orientation.
- Derive slice normal from DICOM row/column direction vectors; sort and deduplicate slices by physical position.
- Canonical orientation correction by axis transpose and sign-based flips, with strict geometry acceptance rather than unsupported full oblique resampling.
- Apply modality LUT, MONOCHROME1 inversion, whole-series percentile windowing, 16 physical-position samples, aspect-ratio-preserving resize, and zero padding.
- Store (16,288,288) uint8 arrays for axial, coronal and sagittal planes, plus geometry metadata and missing-plane masks.

## Original V5 architecture
Three CNN families (`convnext_small`, `resnet50d`, `eca_nfnet_l0`) used native one-channel slices, spatial mean or learnable GeM pooling, slice transformers, and fused three-plane embeddings. The fourth family used DINOv2 ViT-S/14 with three-channel replicated MRI inputs, patch-token pooling, slice attention, and anatomy-informed plane priors. Five fold checkpoints per family yielded a target ensemble of 20 checkpoints.

CNN Phase 3 input size was 384px; DINOv2 checkpoints were normally 392px with an observed 336px fold-specific exception. The inference loader had to respect checkpoint metadata rather than assume every model shared the same size.

## Inference and fusion
V5 inference required training-time image-processing parity, not merely successful checkpoint loading. External V7/R7 and S6 models produced additional study predictions, with optional V8 specialist inference. Final candidate A: fixed 60/25/15 V7/S6/V5 blending; candidate B: dynamically gated V8 consensus plus mild DINOv2 TTA. Both are described as model-integration experiments rather than authorship claims for external checkpoints.

AUC is ranking-based. The example in `src/reference_rank_blend.py` shows rank blending with strong schema checks, but is intentionally **not** the original production inference code or an exact competition-result reproduction.
