# Drought Severity Forecasting Pipeline

This repository contains a reproducible Python script pipeline for the Kaggle final project described in `concept.md`.

The task is supervised regression: for each `region_id`, summarize the previous 91 days of meteorological observations and predict the next five weekly drought severity `score` values. Predictions are clipped to `[0, 5]` and are not rounded by default because Kaggle evaluates MAE.

## AI Entry Point

For future AI-assisted work, read these files first:

- `concept.md`: project task, modeling assumptions, validation strategy, and current implementation direction.
- `progress.md`: short current-state handoff with priorities, completed pieces, and known concerns. progress.md should remain concise. Remove outdated status notes instead of accumulating history.

Use `progress.md` for quick context before making changes. Use `concept.md` when deciding whether a change matches the project direction.

## Environment

Create or reuse the project virtual environment and install the pinned
dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

All commands below assume they are run from the repository root. The expected
input files are:

- `data/train.csv`: training rows with `region_id`, `date`, weather columns, and
  weekly `score` labels.
- `data/test.csv`: 91 daily weather rows for each test `region_id`.
- `sample_submission.csv`: Kaggle submission template.

The main scripts are plain Python entry points and can be inspected with
`--help`:

```bash
.venv/bin/python src/train.py --help
.venv/bin/python src/predict.py --help
```

## Files

- `src/make_features.py`: shared data loading and 91-day window feature generation.
- `src/validate.py`: optional time-based validation/sanity check.
- `src/train.py`: trains one model per forecast horizon and saves models under a model directory.
- `src/predict.py`: loads saved models and writes a Kaggle-format submission.

## Feature Engineering

For every 91-day window, the pipeline computes meteorological summaries from:

`prec`, `surf_pre`, `humidity`, `tmp`, `dp_tmp`, `wb_tmp`, `tmp_max`, `tmp_min`, `tmp_range`, `surf_tmp`, `wind`, `wind_max`, `wind_min`, `wind_range`.

Features include:

- Mean, standard deviation, min, max, median, and percentiles over 7, 14, 28, 56, and 91-day windows.
- Last value, last 3-day mean, last 7-day mean, and recent trend.
- Precipitation sum, dry day count, rainy day count, and maximum consecutive dry days.
- Recent-vs-91-day anomaly features.
- Non-overlapping weekly bins for weeks 1 through 12, ordered from oldest to
  newest, plus 13-week linear trends. Week 13 summaries are omitted because
  they duplicate the existing 7-day summaries. These features preserve recent
  weather sequence shape that a single 91-day summary would discard.
- Region seasonal anomaly features over 28 and 91 days. Each weather summary is
  compared with that region's month-weighted historical climatology. For
  example, `prec_region_seasonal_anom_w91` is the recent 91-day daily mean
  precipitation minus the expected daily mean precipitation for the same
  region and mix of calendar months. Training windows only use weather history
  before the 91-day input window starts; test features use climatology built
  from that region's full history in `train.csv`.
- Hot/dry and evaporation-risk proxy features using precipitation, `tmp_max`, wind, humidity, and surface temperature.
- Relative seasonal features from anonymized date strings.
- Optional historical score persistence features. When enabled, training examples only use scores before the 91-day weather window starts, matching the fact that test windows contain weather but no score labels. Use `--no-score-features` to train or validate weather-only models.

## Quick Reproduction

The report uses the weather-only 641-feature filtered setup with CatBoost as the
best model backend. The full experiment uses stride 8 and at most 150,000
training examples per horizon.

Build the reusable feature cache once:

```bash
.venv/bin/python src/train.py \
  --feature-cache-only \
  --model-dir models_feature_cache_only_641 \
  --feature-cache-dir feature_cache_150k_s8_no_score_weekly_seasonal_filtered \
  --feature-version filtered_641 \
  --max-train-examples-per-horizon 150000 \
  --stride 8 \
  --n-jobs 2 \
  --no-score-features
```

If this cache directory already exists and its `cache_metadata.json` matches the
command settings, this step can be skipped. The later training command will load
the existing `feature_horizon_*.joblib` files automatically.

Train the CatBoost model from that cache:

```bash
.venv/bin/python src/train.py \
  --model-kind catboost \
  --model-dir models_exp_150k_s8_no_score_weekly_seasonal_filtered_catboost \
  --feature-cache-dir feature_cache_150k_s8_no_score_weekly_seasonal_filtered \
  --feature-version filtered_641 \
  --max-train-examples-per-horizon 150000 \
  --stride 8 \
  --n-jobs 2 \
  --model-n-jobs 2 \
  --rf-n-estimators 80 \
  --rf-max-depth 24 \
  --rf-max-samples 0.5 \
  --rf-min-samples-leaf 10 \
  --rf-max-features sqrt \
  --no-score-features
```

Generate the Kaggle submission:

```bash
.venv/bin/python src/predict.py \
  --model-dir models_exp_150k_s8_no_score_weekly_seasonal_filtered_catboost \
  --output submissions/submission_150k_s8_no_score_weekly_seasonal_filtered_catboost.csv
```

The resulting CSV keeps the exact row order and columns from
`sample_submission.csv`. Prediction reads `data/train.csv` even for
weather-only models because regional seasonal anomaly features need historical
weather climatology.

## Training Options

`src/train.py` trains five independent models, one for each forecast horizon.
It writes:

- `horizon_1.joblib` through `horizon_5.joblib`: trained models.
- `metadata.json`: feature columns, training settings, runtime, model kind, and
  feature version.
- `feature_importance.csv`: per-horizon feature importance when the backend
  exposes it.

The most important training arguments are:

- `--model-kind lightgbm|xgboost|catboost|random_forest`: model backend.
- `--feature-version filtered_641|aggressive_620`: feature set version.
  `filtered_641` is the default and the report setup. `aggressive_620` removes
  21 additional low-gain raw summary/event features.
- `--feature-cache-dir DIR`: read/write per-horizon supervised feature matrices.
- `--feature-cache-only`: build feature caches and exit before model training.
- `--no-score-features`: disable historical score features. This is used for
  the final weather-only submissions.
- `--stride N`: use every Nth weekly labeled row before the final random cap.
- `--max-train-examples-per-horizon N`: cap each horizon after feature
  generation. Use `0` for no cap.
- `--n-jobs N`: number of region-level feature-generation workers. Use `0` for
  conservative auto mode.

Feature cache files are named `feature_horizon_1.joblib` through
`feature_horizon_5.joblib`. They contain `X`, `y`, and `meta`, not trained
models. Cache metadata records the feature version and training settings; if a
setting differs, training stops with a cache mismatch error. Use a separate
cache directory for 641-feature and 620-feature experiments.

For a quick smoke test:

```bash
.venv/bin/python src/train.py --debug --no-score-features
```

## Model Ablation

To compare model backbones fairly, reuse the same `filtered_641` feature cache
and change only `--model-kind`.

```bash
# LightGBM
.venv/bin/python src/train.py \
  --model-kind lightgbm \
  --model-dir models_exp_150k_s8_no_score_weekly_seasonal_filtered_lgbm \
  --feature-cache-dir feature_cache_150k_s8_no_score_weekly_seasonal_filtered \
  --feature-version filtered_641 \
  --max-train-examples-per-horizon 150000 \
  --stride 8 \
  --n-jobs 2 \
  --no-score-features

# XGBoost
.venv/bin/python src/train.py \
  --model-kind xgboost \
  --model-dir models_exp_150k_s8_no_score_weekly_seasonal_filtered_xgboost \
  --feature-cache-dir feature_cache_150k_s8_no_score_weekly_seasonal_filtered \
  --feature-version filtered_641 \
  --max-train-examples-per-horizon 150000 \
  --stride 8 \
  --n-jobs 2 \
  --no-score-features

# CatBoost
.venv/bin/python src/train.py \
  --model-kind catboost \
  --model-dir models_exp_150k_s8_no_score_weekly_seasonal_filtered_catboost \
  --feature-cache-dir feature_cache_150k_s8_no_score_weekly_seasonal_filtered \
  --feature-version filtered_641 \
  --max-train-examples-per-horizon 150000 \
  --stride 8 \
  --n-jobs 2 \
  --no-score-features

# Random Forest
.venv/bin/python src/train.py \
  --model-kind random_forest \
  --model-dir models_exp_150k_s8_no_score_weekly_seasonal_filtered_random_forest \
  --feature-cache-dir feature_cache_150k_s8_no_score_weekly_seasonal_filtered \
  --feature-version filtered_641 \
  --max-train-examples-per-horizon 150000 \
  --stride 8 \
  --n-jobs 2 \
  --model-n-jobs 2 \
  --rf-n-estimators 80 \
  --rf-max-depth 24 \
  --rf-max-samples 0.5 \
  --rf-min-samples-leaf 10 \
  --rf-max-features sqrt \
  --no-score-features
```

## Feature-Version Experiments

The default feature version is `filtered_641`. To test the smaller 620-feature
variant, use a different cache directory:

```bash
.venv/bin/python src/train.py \
  --feature-version aggressive_620 \
  --model-kind catboost \
  --model-dir models_exp_150k_s8_no_score_weekly_seasonal_aggressive_620_catboost \
  --feature-cache-dir feature_cache_150k_s8_no_score_weekly_seasonal_aggressive_620 \
  --max-train-examples-per-horizon 150000 \
  --stride 8 \
  --n-jobs 2 \
  --no-score-features
```

Do not reuse a `filtered_641` cache for `aggressive_620`, or the reverse.

## Prediction Modes

Default prediction mode loads trained models:

```bash
.venv/bin/python src/predict.py \
  --model-dir models_exp_150k_s8_no_score_weekly_seasonal_filtered_catboost \
  --output submissions/submission.csv
```

Optional baseline/debug modes are also available:

```bash
.venv/bin/python src/predict.py --mode zero --output submissions/zero.csv
.venv/bin/python src/predict.py --mode latest-score --output submissions/latest_score.csv
```

Blend modes exist for calibration experiments but are not used for the reported
weather-only CatBoost result:

```bash
.venv/bin/python src/predict.py \
  --model-dir models_exp_150k_s8_no_score_weekly_seasonal_filtered_catboost \
  --mode blend-zero \
  --zero-weight 0.75 \
  --output submissions/blend_zero.csv
```

## Validation

`src/validate.py` remains available for quick pipeline sanity checks:

```bash
.venv/bin/python src/validate.py --debug --no-score-features
```

Earlier local validation numbers are not reported as final results because they
used nearby score information and were not representative of the final Kaggle
test input, where recent score labels are unavailable. The report therefore
uses Kaggle public MAE for the submitted model and ablation comparisons.
