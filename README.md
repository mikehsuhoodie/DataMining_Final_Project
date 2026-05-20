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

Quick check:

```bash
.venv/bin/python src/validate.py --debug
```

Fuller validation can be run by increasing regions/examples or disabling caps:

```bash
.venv/bin/python src/validate.py --stride 4 --max-train-examples-per-horizon 250000
```

Weather-only validation:

```bash
.venv/bin/python src/validate.py --no-score-features
```

## Training

Default training uses every fourth weekly target and caps each horizon at 250,000 examples to keep runtime practical on a large CSV. Use `--stride 1 --max-train-examples-per-horizon 0` for exhaustive feature generation if hardware allows.

```bash
.venv/bin/python src/train.py
```

Weather-only training:

```bash
.venv/bin/python src/train.py --no-score-features --model-dir models_no_score
```

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
