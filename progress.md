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
- Feature extraction has been optimized by combining percentile calculations,
  avoiding repeated column lookups, and reusing repeated per-region cutoff
  features.
- `--n-jobs` supports region-level parallel feature generation. Workers receive
  per-region arrays rather than the full training DataFrame. `--n-jobs 0` uses a
  conservative auto mode capped at 4 workers.
- `src/train.py`, `src/validate.py`, and `src/analyze_features.py` build one
  horizon at a time to keep peak memory bounded.
- `src/analyze_features.py` exists for model importance, permutation importance,
  and feature-group contribution analysis.
- Local validation has been run for the 80k weather-only setup; results are in
  `validation/validation_mae.csv`.
- Fast 30k feature analysis outputs exist for both score-feature and
  weather-only setups in `feature_analysis_fast_30k/` and
  `feature_analysis_fast_30k_weather_only/`.
- 150k model variants have been trained in `models_no_score_150k/` and
  `models_score_150k/`.
- Current Kaggle public score reported by the team is about `0.88`.
- Weather-only and score-feature submissions exist under `submissions/`,
  including 150k variants.

## Current Priority

1. Compare local validation for weather-only 80k vs 150k and `stride 4` vs
   `stride 2` before relying on Kaggle public leaderboard differences.
2. Measure actual peak memory with `/usr/bin/time -v` when increasing the
   example cap, lowering stride, or increasing worker count.
3. Use feature importance and ablation results to remove or revise weak feature
   groups such as unstable humidity/dewpoint/wet-bulb/surface-pressure signals.
4. Compare weather-only, corrected score-history, and persistence baselines.
5. Tune LightGBM regularization/sampling or try CatBoost/XGBoost after
   validation and features stabilize.

## Known Concerns

- Feature extraction has been optimized, but full-data runtime and memory should
  still be measured before large experiment sweeps.
- Large validation runs can still be memory-heavy because feature rows are first
  accumulated as Python dictionaries before becoming DataFrames.
- Final feature matrix caching is not yet ideal because the feature set is still
  changing. Prefer profiling, deduplication, parallelization, and lightweight
  intermediate caches first.
- Kaggle public leaderboard should not be the only validation signal. Local
  validation is needed for fast feature selection and leakage checks.
- Increasing `--max-train-examples-per-horizon` from 80k to 150k changes the
  final per-region sample density even with the same `--stride`; more examples
  are not guaranteed to improve Kaggle score.
