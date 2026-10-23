# Data, provenance, and attribution

## Original work
This portfolio documents the author's V5 four-family training and DICOM canonicalization experiments, Qwen-based report-target extraction, patient/study group splitting, debugging, GPU inference integration, and heuristic blend experiments.

## External models
V7/R7, S6, and V8 specialist model packages were developed and published by third-party contributors. Their original code and weights are **not** in this repository. The project author integrated them and experimented with their prediction fusion but does not claim their original training. Any later redistribution requires explicit verification of each upstream license.

## Not redistributed
- Competition MRI images, DICOMs, reports, identifiers, or training labels
- Checkpoints, pretrained weights, original third-party inference packages
- Generated pseudo-label datasets, public/private test predictions, or unreviewed notebook outputs

## Scope and reproducibility
The stand-alone example illustrates schema-safe fusion of already computed predictions. It cannot reproduce the 0.957/0.942 leaderboard scores without the external engines, their assets, original inference settings, and permitted data.

The portfolio records documented experiments; it is not a clinically validated diagnostic device. Verify Kaggle and upstream data licensing before adding more artifacts.
