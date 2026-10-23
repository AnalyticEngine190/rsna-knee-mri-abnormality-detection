# Final private-leaderboard snapshot — 23 October 2026

The author provided a CSV export named `rsna-knee-abnormality-detection-privateleaderboard-2026-10-23T08_18_33.csv` and a team-rank screenshot. The full CSV is not redistributed.

| Statistic | Value |
| --- | ---: |
| Teams listed in exported snapshot | 6,425 |
| Our rank | **717** |
| Our rounded private macro ROC-AUC | **0.942** |
| Approximate share of teams ranked ahead or equal to our place | **11.2%** |
| Rank 1 private AUC | 0.960 |
| Rank 10 private AUC | 0.957 |
| Rank 50 private AUC | 0.951 |
| Rank 100 private AUC | 0.948 |
| Median listed private AUC | 0.925 |

Leaderboard AUC values in this CSV are rounded to three decimals. Different ranks may therefore display the same AUC, including the very broad cluster near 0.942. The *rank* is more precise than the displayed score.

A screenshot of the team's ranking displayed **↑1,230** places after the leaderboard change. Our best public score had been **0.957**, versus final reported **0.942**. This result shows **relative rank improvement**, not an absolute private-score increase. It does not establish that specific competing teams overfitted: we lack paired public/private scores, experiments, and per-case predictions for other teams.

The event illustrates the importance of representative expert validation, caution with report-derived labels, and limiting repeated public-leaderboard tuning.
