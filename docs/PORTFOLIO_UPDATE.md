# Training architecture portfolio update

The repository contains cleaned, importable research implementations of the **DINOv2 ViT-S/14** and **ECA-NFNet-L0** tri-planar architectures, shared model utilities, a manifest-driven cached MRI loader, and masked BCE and AUC evaluation helpers.

The original notebooks were research notebooks with Colab/Google Drive and Kaggle environment dependencies. These modules are independently importable, do not trigger automatic downloads on import, and support synthetic CPU tests. They are **research reinterpretations of the architecture**, not necessarily byte-for-byte equivalent to previous `.pth` checkpoints.

Run the smoke test (on a machine with PyTorch, NumPy, Pandas, SciKit-Learn):

```bash
python -m unittest discover -s tests -v
```

To instantiate a real backbone, install `timm` and supply pretrained/checkpoint weights separately. **Do not train from randomly initialized DINOv2 or NFNet assuming equivalence with the original experiments.** Real MRI training requires the approved cached data and verified patient-level fold mapping.

## Supported experiments

* `src/models/triplanar_nfnet.py` — native one-channel CNN backbone, GeM pooling, slice Transformer, fusion head.
* `src/models/triplanar_dinov2.py` — RGB-replicated MRI, ViT-S/14 tokens, CLS/mean/focal pooling, gated slice attention, per-pathology anatomical plane priors.
* `src/data/cached.py` — per-plane NPZ loader (16×288×288), foreground standardization, plane presence masking, explicit cache root, patient fold checks.
* `src/training/engine.py` — masked sample-weighted BCE, macro AUC with per-target support rules, small training/evaluation helpers.
* `configs/` — human-readable example Phase 3 settings, not verified archival checkpoint manifests.
* `docs/models/` — design intent and known limitations.

## Security, licensing and reproducibility

The uploads contained research paths and Colab-specific commands. We did not include originals wholesale, private training CSVs, Kaggle data, third-party code bundles, model weights, or any radiology report. Each module is authored as cleaned source and makes no claim of full hidden-test reproducibility.

The user-provided final leaderboard screenshot is a historical competition result, not a medical study image.
