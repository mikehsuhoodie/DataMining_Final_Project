# Drought Severity Forecasting Pipeline

This repository contains a reproducible Python script pipeline for the Kaggle final project described in `concept.md`.

The task is supervised regression: for each `region_id`, summarize the previous 91 days of meteorological observations and predict the next five weekly drought severity `score` values. Predictions are clipped to `[0, 5]` and are not rounded by default because Kaggle evaluates MAE.

## AI Entry Point

For future AI-assisted work, read these files first:

- `concept.md`: project task, modeling assumptions, validation strategy, and current implementation direction.
- `progress.md`: short current-state handoff with priorities, completed pieces, and known concerns. progress.md should remain concise. Remove outdated status notes instead of accumulating history.

Use `progress.md` for quick context before making changes. Use `concept.md` when deciding whether a change matches the project direction.

## Environment

Use the project virtual environment:

```bash
.venv/bin/python src/validate.py --debug
.venv/bin/python src/train.py
.venv/bin/python src/predict.py --output submissions/submission.csv
```

The scripts prefer LightGBM when installed. If LightGBM is unavailable, they fall back to scikit-learn regressors.

## Files

- `src/make_features.py`: shared data loading and 91-day window feature generation.
- `src/validate.py`: time-based validation with baselines and horizon MAE.
- `src/train.py`: trains one model per forecast horizon and saves models under `models/`.
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

## Validation

Validation is time-based, not random. For each region, `src/validate.py` holds out the latest five weekly labels and builds one validation forecast block with horizons 1 through 5. It reports:

- Global mean score baseline.
- Region historical mean baseline.
- Recent region score mean baseline.
- First model result.
- Overall MAE and MAE by horizon.

For memory control, validation builds and trains one horizon at a time instead
of materializing all five horizon feature matrices at once.

Quick check:

```bash
.venv/bin/python src/validate.py --debug
```

Fuller validation can be run by increasing regions/examples or disabling caps:

```bash
.venv/bin/python src/validate.py --stride 4 --max-train-examples-per-horizon 250000
```

Feature generation can be parallelized across independent regions:

```bash
.venv/bin/python src/validate.py --n-jobs 4
```

Use `--n-jobs 0` for conservative auto mode, currently capped at 4 workers.
Keep this lower if memory pressure or CPU contention with model training becomes
a problem.

Parallel workers receive only one region's `values`, `dates`, and `scores`
arrays at a time, not the full training DataFrame. Each worker also builds a
small expanding monthly weather climatology for leakage-safe region seasonal
anomaly features. Completed region features are returned as compact `float32`
NumPy chunks to limit parent-process memory usage.

Weather-only validation:

```bash
.venv/bin/python src/validate.py --no-score-features
```

## Training

Default training uses every fourth weekly target and caps each horizon at 250,000 examples to keep runtime practical on a large CSV. Use `--stride 1 --max-train-examples-per-horizon 0` for exhaustive feature generation if hardware allows.

```bash
.venv/bin/python src/train.py
```

Parallel feature generation during training:

```bash
.venv/bin/python src/train.py --n-jobs 4
```

Training still builds and fits one horizon at a time, so `--n-jobs` speeds up
per-horizon feature generation without materializing all five horizon matrices
at once. Each training run also writes `feature_importance.csv` under its model
directory with per-horizon LightGBM split and gain importance. Use
`src/analyze_features.py` separately when validation-based permutation
importance is needed. Train gain importance is useful for selecting ablation
candidates, but it is not enough by itself to prove that a feature should be
deleted.

Weather-only training:

```bash
.venv/bin/python src/train.py --no-score-features --model-dir models_no_score
```

Current expanded weather-only experiment:

```bash
.venv/bin/python src/train.py \
  --model-dir models_exp_150k_s8_no_score_weekly_seasonal \
  --max-train-examples-per-horizon 150000 \
  --stride 8 \
  --n-jobs 2 \
  --no-score-features
```

This expanded weekly-seasonal model has a reported Kaggle public MAE of
`0.7799`. Keep its model directory and submission for comparison when running
feature ablations.

The current controlled ablation removes inference-constant coverage/day-index
features, duplicate week-13 summaries, and low-gain raw wind summaries. Wind
seasonal anomalies, weekly bins, weekly trends, and drought interactions remain
available to the model. Prediction detects older metadata and rebuilds removed
features when loading pre-ablation models.

The trained 684-feature controlled ablation model under
`models_exp_150k_s8_no_score_weekly_seasonal_ablation/` has a reported Kaggle
public MAE of `0.7754` and is the current baseline. Preserve that model
directory. Regenerate its submission with:

```bash
.venv/bin/python src/predict.py \
  --model-dir models_exp_150k_s8_no_score_weekly_seasonal_ablation \
  --output submissions/submission_150k_s8_no_score_weekly_seasonal_ablation.csv
```

The next controlled experiment is implemented as the current
feature-generation default. It removes the unused `prec_w56_min` feature and
raw wet-bulb summaries while retaining wet-bulb weekly bins, the 13-week trend,
and region seasonal anomalies. Train it into a separate model directory so the
`0.7754` baseline remains available for comparison.

Debug training:

```bash
.venv/bin/python src/train.py --debug
```

## Prediction

After training, generate a submission:

```bash
.venv/bin/python src/predict.py --output submissions/submission.csv
```

The output preserves the exact row order and columns from `sample_submission.csv`.
Prediction reads `train.csv` even for weather-only models because region
seasonal anomaly features require historical weather climatology.
