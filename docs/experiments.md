# Selected experiments, failures, and provenance

| Experiment | Public macro AUC | Interpretation |
|---|---:|---|
| Early DINOv2-only effective submission | 0.823 | 15 CNN checkpoints skipped due to incorrect input channel count |
| First full four-family attempt | 0.879 | Legacy raw-DICOM processing did not match V5 cached training volumes |
| Four-family update | 0.889 | Progressive debugging and inference improvements |
| S6 standalone (external) | 0.950 | Independent expert prediction source |
| V7 standalone (external) | 0.956 | Strong independently published base |
| Fixed V7/S6/V5 60/25/15 | 0.957 | Best documented fixed blend |
| Pathology-specific weights, no TTA | 0.957 | Separate per-target heuristic; not the fixed blend |
| CoAt Top-3 5% + V5 10% + V7 60% + S6 25% | 0.956 | Extra image family at low weight |
| V8 Consensus + Mild DINOv2 TTA | 0.956 | Dynamic rank-disagreement gate; additional model diversity |
| Final selected submissions | 0.942 private, rank 717 | Team-level final result, per-submission private score not proven by screenshot |

## Failure-driven engineering
1. Early checkpoint loader used 3-channel CNN stems against grayscale one-channel checkpoints. Fixing `in_chans=1` and strict checkpoint restoration prevented silent branch loss.
2. The cached V5 preprocessing used 3D geometry and whole-series normalization; a raw-DICOM fallback used mismatched per-slice normalization and slice orientation. The difference materially affected performance.
3. The training cache was not a hidden-test cache. Test DICOMs were decoded on demand, without fuzzy UID matching that risked study collisions.
4. Some MRI studies lacked planes. Missing-view masks were used rather than crashing or inventing images.
5. Data-loader worker deadlocks and resource limits required a safer multiprocessing setup.
6. Fold-specific ViT input resolution needed to be loaded from checkpoint metadata.
7. Unverified aggressive TTA and extrapolated pathology-specific blends were kept separate from the known-scoring configurations.

## Fixed blend versus dynamic consensus
The fixed submission applies V7 60%, S6 25%, V5 15%. The V8 consensus experiment builds on the same expert models and gates V8's contribution in roughly the 10–50% range, redistributing remaining contribution in a V7:S6:V5 ratio of 60:25:15. A three-study public execution reported mean V8 share 24.3%, max 50%, and successful branch completion, **not** proof of better hidden-set performance.

The experiment with `rsna_pathology_steady_60_25_15` in its filename actually used class-specific weight overrides. It is not a literal implementation of the fixed baseline.

## Interpretation
Repeated 0.956–0.957 public scores suggested a plateau in postprocessing, not demonstrated immunity to shared errors. No experiment established an independently validated confident-error detector. The final private score was lower than public despite rank improvement.
