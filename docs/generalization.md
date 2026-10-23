# Public-to-private generalization: what the results support

## Observed outcomes
The best recorded **public macro AUC was 0.957**. The final result screenshot showed **0.942 private macro AUC, 717th place, and +1,230 positions**. The private score was therefore 0.015 lower than the best public score, even while the relative position improved.

Kaggle scores public and private partitions separately. Different patient samples, prevalence mixtures, acquisitions, and model-selection effects can reorder teams. A public-to-private score decline is not proof of a specific bug or overfitting.

## Why overfitting is plausible
The working training set had 4,407 cases but only 58 with expert reference labels. Pseudo-labels derived from reports may be noisy or incomplete, making strong-looking local validation scores fragile. Rare target positives make label-specific AUC estimates noisy. Repeated tuning against the same public leaderboard can inadvertently select random public-set improvements that do not generalize.

## What cannot be concluded
We **cannot conclude that other teams overfitted** from a +1,230-place movement, or infer which teams did so. This would require at least paired public/private score distributions, and ideally model development and per-case data from those teams. Our own drop to 0.942 is reason to avoid claiming superior absolute generalization. The screenshot also does not reveal which of our two finalists individually achieved the team-level private score.

## Lessons for future studies
- Isolate an untouched expert-labeled evaluation set or use nested cross-validation.
- Group repeated studies by patient ID, not UID alone where patients can recur.
- Audit LLM target quality and preserve uncertainty instead of forcing unmentioned findings to negative.
- Measure paired, per-target AUC with confidence intervals and between-expert error correlation.
- Limit public leaderboard exploration to preserve independent evaluation.
