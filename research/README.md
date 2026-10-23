# Research code archive

These files come from the author's original competition work. They complement the documented final V7/S6/V5 and V8 ensemble integrations.

| Path | Source | What it shows |
| --- | --- | --- |
| [`architecture/tri_planar_cnn.py`](architecture/tri_planar_cnn.py) | Extracted from high-resolution ConvNeXt training code | Three-plane encoders, slice transformer, missing-plane fusion, mean/GeM pooling, grayscale TIMM stem |
| [`training/loss_and_metrics.py`](training/loss_and_metrics.py) | Extracted from five-fold ResNet50D training | FP32 masked BCE, sample weighting, class imbalance weights, macro ROC-AUC |
| [`oof/rsna_4family_ensemble_v2_7.py`](oof/rsna_4family_ensemble_v2_7.py) | Original strict OOF script | Fold/UID alignment, nested calibration, gold-label guards, bootstrap comparisons, provenance checks |
| [`experiments/four_expert_pathology_weights.csv`](experiments/four_expert_pathology_weights.csv) | Original 12-target experimental weight table | Per-pathology weight hypotheses for raptor, r7, v8spec, and v5 |

The extraction preserves key model/training functions, but **these extracts are not a standalone retraining program**. You need the full original training environment, private cache, label-generation pipeline, weights, and approved image data. We did not run a 5-fold training job to validate these modules.

## Code verification
- `tri_planar_cnn.py`, `loss_and_metrics.py`, and the v2.7 OOF file were checked for Python syntax. This is **not** a performance or end-to-end execution verification.
- The v2.7 OOF file is an archived advanced research script, not a drop-in Kaggle notebook.
- The optional `four_expert_pathology_weights.csv` is a separate experiment; **do not interpret it as the final fixed 60/25/15 blend**.

## Excluded from publication
- `phase2_fused_master.csv`: contains study identifiers and individual training targets, so this is not shared.
- `rsna-knee-abnormality-detection-privateleaderboard-*.csv`: contains other teams' identifiers and usernames; only aggregate metrics are reported in [leaderboard analysis](../docs/leaderboard-snapshot.md).
- Colab exports containing temporary signed links, notebook-specific commands, and duplicated code have not been published unchanged. The repository prioritizes curated technical examples over a large, non-reproducible dump.

## Source history
Additional private Colab research files cover NFNet, DINOv2, high-resolution ConvNeXt/ResNet50D training, fold-4 recovery, and OOF consolidation. The v2.7 OOF script is the more recent strict-audit version; its earlier v2.6 predecessor remains an archived local experiment.

This is a research/portfolio code sample, not a clinical diagnostic product.
