# Project Progress

Short AI handoff note. Read this after `concept.md` to understand the current
state without scanning the whole repository.
progress.md should remain concise. Remove outdated status notes instead of accumulating history.

## Current Status

- Working pipeline exists in `src/`: feature generation, validation, training,
  and prediction.
- Main model is LightGBM regression, trained separately for horizons 1 to 5.
- Weather-only and score-history variants are implemented through
  `--no-score-features`.
- Current Kaggle public score reported by the team is about `0.88`.
- A weather-only submission exists at `submissions/submission_no_score.csv`.
- Debug validation outputs exist, but they are not enough to judge final model
  quality because they use a small region subset.

## Current Priority

1. Verify local time-based validation correctness and avoid leakage.
2. Profile and speed up feature extraction before running many experiments.
3. Run feature importance and ablation to remove or revise weak features.
4. Compare weather-only, corrected score-history, and persistence baselines.
5. Tune LightGBM or try CatBoost/XGBoost after validation and features stabilize.

## Known Concerns

- Feature extraction is slow because many rolling-window summaries are recomputed
  in Python loops over regions and target windows.
- Final feature matrix caching is not yet ideal because the feature set is still
  changing. Prefer profiling, deduplication, parallelization, and lightweight
  intermediate caches first.
- Kaggle public leaderboard should not be the only validation signal. Local
  validation is needed for fast feature selection and leakage checks.
