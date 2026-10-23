# Resume and interview entry

## Resume (three-bullet version)

**RSNA Knee Abnormality Detection — Independent ML Engineering Project** | Oct 2026 (25-day sprint)

**Kaggle final rank: 717 | Final/private macro ROC-AUC: 0.942 | Best public macro ROC-AUC: 0.957**

- Built geometry-aware DICOM preprocessing and a tri-planar MRI classifier for 12 abnormalities, including series selection, physical slice ordering, multi-view masks, and standardized imaging inputs.
- Trained and evaluated a four-family PyTorch/TIMM ensemble (ConvNeXt, ResNet50D, ECA-NFNet, DINOv2) across five folds, using Qwen2.5-assisted multilingual report labeling and soft-target supervision.
- Integrated independently published V7/S6/V8 experts, tested TTA and rank fusion, and audited offline GPU inference; improved documented public macro AUC from 0.823 to 0.957 and finished 717th overall.

## 30-second interview explanation

“I participated in an RSNA knee-MRI machine learning competition during its last 25 days. I built a DICOM canonicalization and multi-plane training pipeline, trained four architecture families across five folds, and used language models to extract weak labels from radiology reports. I integrated external competition experts and debugged checkpoint compatibility and inference preprocessing issues. The best public AUC reached 0.957, and my final private score was 0.942 at rank 717. A major learning was that a small gold-labeled evaluation set and repeated leaderboard tuning can make seemingly strong changes fail to generalize.”

## Accuracy note
Always distinguish public versus private AUC, your own V5 training from the external expert weights, and your team-level final ranking from any unverified per-finalist private result.
