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
- Feature generation now includes oldest-to-newest week 1 through week 12
  weather-bin summaries, 13-week trends, and leakage-safe region monthly
  climatology anomalies over 28 and 91 days. Week 13 remains in trend
  calculations but its summaries are omitted because they duplicate existing
  recent 7-day features. Models trained before this change are a separate
  feature version and should remain available for ensemble comparisons.
- Naming note: use `filtered` for the fixed feature-set/model version that
  removes known duplicate or inference-constant columns. Reserve `ablation` for
  controlled experiments that compare feature/model variants.
- The post-`0.7799` filtered feature set removes inference-constant
  `region_seasonal_coverage_w28`, `region_seasonal_coverage_w91`, and
  `day_index_mod_365`, plus week-13 bin summaries that exactly duplicated
  existing 7-day summaries. It also removes low-gain raw summaries for
  `wind`, `wind_max`, `wind_min`, and `wind_range`, while retaining their
  region seasonal anomalies, weekly bins, 13-week trends, and
  drought-interaction features.
  Prediction retains a metadata-driven legacy path so pre-filtered models can
  still rebuild their original test features.
- `--n-jobs` supports region-level parallel feature generation. Workers receive
  per-region arrays rather than the full training DataFrame. `--n-jobs 0` uses a
  conservative auto mode capped at 4 workers.
- `src/train.py` now supports
  `--model-kind lightgbm|xgboost|catboost|random_forest` for model backend
  comparison. XGBoost and CatBoost dependencies are installed in the project
  venv and listed in `requirements.txt`; Random Forest uses the existing
  scikit-learn dependency.
- `src/train.py` supports per-horizon feature caches through
  `--feature-cache-dir`. Cache files are named
  `feature_horizon_1.joblib` through `feature_horizon_5.joblib` and contain
  `X`, `y`, and `meta`, not trained models. Use `--feature-cache-only` to build
  caches without training, or omit it to build missing caches and then train.
- Training metadata now records `train_command`, `train_runtime_seconds`, and
  `train_runtime_minutes`.
- `src/train.py` supports `--feature-version filtered_641|aggressive_620`.
  The default is `filtered_641`, which is the report setup. `aggressive_620`
  removes 21 additional low-gain raw summary/event features from the 641-feature
  setup. The feature version is recorded in model metadata and feature-cache
  metadata.
- Prediction detects model metadata and rebuilds the removed low-gain columns
  only when the saved model expects them, so both 641-feature and 620-feature
  model directories remain usable.
- Feature workers return compact `float32` NumPy chunks instead of retaining
  large Python dictionaries in the parent process. This keeps the expanded
  871-feature weather-only setup within memory limits at 150k rows per horizon.
- `src/train.py`, `src/validate.py`, and `src/analyze_features.py` build one
  horizon at a time to keep peak memory bounded.
- `src/analyze_features.py` exists for model importance, permutation importance,
  and feature-group contribution analysis.
- Local validation has been run for the older 80k weather-only setup; results
  are in `validation/validation_mae.csv`. It does not yet validate the expanded
  weekly-seasonal feature version.
- Fast 30k feature analysis outputs exist for both score-feature and
  weather-only setups in `feature_analysis_fast_30k/` and
  `feature_analysis_fast_30k_weather_only/`.
- 150k model variants have been trained in `models_no_score_150k/` and
  `models_score_150k/`.
- Current best reported Kaggle public MAE is `0.7663` from CatBoost on the
  641-feature filtered setup.
- A `0.75 * old 100k stride-8 weather-only + 0.25 * new 150k stride-8
  weather-only` submission reached a reported Kaggle public MAE of `0.8703`.
- The expanded 150k stride-8 weather-only model has been trained under
  `models_exp_150k_s8_no_score_weekly_seasonal/`, with its submission under
  `submissions/submission_150k_s8_no_score_weekly_seasonal.csv`.
- The expanded weekly-seasonal submission reached a reported Kaggle public MAE
  of `0.7799`, improving substantially over the previous `0.8703` blend
  baseline. Preserve this submission and model for comparison.
- The 684-feature filtered model has been trained under
  `models_exp_150k_s8_no_score_weekly_seasonal_ablation/`, with its submission
  under
  `submissions/submission_150k_s8_no_score_weekly_seasonal_ablation.csv`.
  It reached a reported Kaggle public MAE of `0.7754`, improving the expanded
  model by `0.0045`.
- Preserve the existing 684-feature filtered baseline model directory. Generate its
  submission with:
  `.venv/bin/python src/predict.py --model-dir models_exp_150k_s8_no_score_weekly_seasonal_ablation --output submissions/submission_150k_s8_no_score_weekly_seasonal_ablation.csv`.
- The filtered model's train gain importance is dominated by
  `hot_dry_w28`, 91-day region seasonal precipitation anomaly, 91-day region
  seasonal temperature-range anomaly, and several other region seasonal
  anomaly features. Across all horizons, weekly bins account for about `25.1%`
  of gain and region seasonal anomalies account for about `20.1%`. The only
  remaining feature unused by every horizon is `prec_w56_min`.
- Experiment A is implemented as the current feature-generation default. It
  removes `prec_w56_min` and 42 raw `wb_tmp` summaries while retaining wet-bulb
  weekly bins, the 13-week trend, and region seasonal anomalies. This reduces
  the weather-only feature set from 684 to 641 columns. Prediction detects
  whether older model metadata requires the removed columns and rebuilds them
  when needed.
- Aggressive filtered experiment is available through
  `--feature-version aggressive_620` on top of Experiment A. Removed features:
  `prec_w91_min`, `prec_w28_min`, `prec_w91_q10`, `prec_w56_q10`,
  `prec_w28_q10`, `rainy_days_w7`, `rainy_days_w14`, `dry_fraction_w91`,
  `tmp_min_last7_mean`, `tmp_last7_mean`, `tmp_max_last7_mean`,
  `surf_tmp_last7_mean`, `dp_tmp_last7_mean`, `prec_last7_mean`,
  `surf_pre_last7_mean`, `surf_pre_w91_mean`, `surf_pre_w56_mean`,
  `surf_pre_w28_mean`, `surf_pre_w91_median`, `surf_tmp_w14_mean`, and
  `tmp_min_w56_mean`.
- Weather-only and score-feature submissions exist under `submissions/`,
  including 150k variants.

## Current Priority

1. Preserve the filtered `0.7754` model and submission as the
   current baseline. Keep the `0.7799` pre-ablation model for comparison.
2. Use `--feature-version filtered_641` for report-aligned retraining. This is
   also the default if the flag is omitted.
3. Use `--feature-version aggressive_620` only for the optional 620-feature
   experiment, with a distinct cache directory, e.g.
   `feature_cache_150k_s8_no_score_weekly_seasonal_aggressive_620/`.
4. Train XGBoost and CatBoost model-backend comparisons using the same
   `150k`, `stride=8`, `--no-score-features`, `filtered_641` feature setup and
   cache unless intentionally testing the 620-feature variant.
5. Test dewpoint raw-summary removal as a separate ablation from the `0.7754`
   baseline while retaining its weekly and region seasonal features.
6. Keep precipitation-event features for now. `dry_days_w91` has measured gain,
   so do not remove that group as part of an unrelated cleanup.
7. Build multi-cutoff validation before broad model tuning or larger experiment
   sweeps.
8. Use permutation importance and group ablation before broader feature
   removal.

## Known Concerns

- Feature extraction has been optimized, but it remains the runtime bottleneck.
  A full 150k-row weather-only horizon with 871 features and `--n-jobs 2`
  measured about 7 minutes and 6 seconds with peak RSS around 3.52 GiB and no
  swap.
- The test set has complete monthly climatology coverage for all 2,248 regions,
  so the removed `region_seasonal_coverage_*` features were constant at
  inference while varying in training. Removed `day_index_mod_365` was also
  constant at inference because every test region has a 91-day window.
- Region seasonal anomalies are leakage-safe: training climatology snapshots
  stop before each 91-day input window starts. Prediction uses each region's
  full official `train.csv` weather history as climatology and never adds test
  weather or hidden targets to that historical baseline.
- Feature caches can become stale when the feature-generation defaults change.
  Keep one cache directory per feature configuration; the cache metadata guard
  checks run settings but cannot know the semantic intent of a reused directory
  name.
- Do not reuse a 641 cache for a 620 experiment or vice versa. Use distinct
  cache directory names; `feature_version` in cache metadata now guards against
  accidental reuse.
- Kaggle public leaderboard should not be the only validation signal. Local
  validation is needed for fast feature selection and leakage checks.
- Increasing `--max-train-examples-per-horizon` from 80k to 150k changes the
  final per-region sample density even with the same `--stride`; more examples
  are not guaranteed to improve Kaggle score.
