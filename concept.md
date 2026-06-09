# Natural Disaster Severity Prediction: Implementation Concept

## Current Project Direction

This project is currently focused on building a reliable tabular forecasting
pipeline before spending effort on model tuning. The most important principle is
to avoid optimizing against an untrusted validation setup.

Current implementation direction:

1. Verify local time-based validation correctness.
2. Profile and reduce feature extraction cost, especially repeated 91-day
   window computation.
3. Use feature importance and ablation to decide which engineered features are
   useful.
4. Improve the feature set based on validation and Kaggle feedback.
5. Tune LightGBM or compare CatBoost/XGBoost only after validation and features
   are stable.
6. Try simple persistence baselines and blends because drought severity is
   usually temporally persistent.

Do not treat Kaggle public leaderboard as the only validation signal. It is
useful external feedback, but local validation is needed for fast iteration,
feature selection, leakage checks, horizon-level errors, and model comparison.

Feature caching should be introduced carefully. Since the feature set is still
changing, caching final feature matrices too early may waste time. Prefer first
profiling feature extraction, removing repeated computation, and adding reusable
intermediate or lightweight caches. Cache final feature matrices once the feature
set becomes more stable.

## Task Summary

The final project is a Kaggle forecasting competition. For each `region_id`, we are given 91 consecutive days of meteorological observations in `test.csv`, and we must predict the drought severity `score` for the next five consecutive weeks.

The target `score` ranges from 0 to 5. In `train.csv`, scores are weekly labels and the other six days usually have missing target values. Kaggle evaluates predictions using MAE, so the implementation should optimize ordered numeric accuracy rather than only classification accuracy.

This should be treated primarily as supervised regression. The labels are ordinal drought severity levels, but the leaderboard metric is mean absolute error, and Kaggle accepts floating-point predictions. Lower public leaderboard scores are better because the score is an error value: smaller MAE means predictions are closer to the hidden true scores.

## Data Snapshot

- Train rows: 12,319,040
- Test rows: 204,568
- Regions: 2,248 in train and test
- Train length per region: 5,480 daily rows
- Test length per region: 91 daily rows
- Test horizon: five weekly scores per region
- Meteorological features:
  - `prec`
  - `surf_pre`
  - `humidity`
  - `tmp`
  - `dp_tmp`
  - `wb_tmp`
  - `tmp_max`
  - `tmp_min`
  - `tmp_range`
  - `surf_tmp`
  - `wind`
  - `wind_max`
  - `wind_min`
  - `wind_range`
- Non-null train labels: 1,757,936
- Score distribution:
  - `0`: 1,048,333
  - `1`: 303,432
  - `2`: 186,279
  - `3`: 118,496
  - `4`: 69,422
  - `5`: 31,974

The labels are imbalanced toward low severity, so a strong model should avoid overpredicting rare severe weeks while still learning warning patterns for scores 3 to 5.

The dates appear to be anonymized or shifted, for example years like `3004` and `3020`. Do not interpret these as real calendar years. They still preserve temporal order, so they are useful for sorting, building lag features, rolling windows, and extracting relative seasonal features such as day-of-year or week-of-year.

The dataset appears to be derived from Kaggle's public "Predict Droughts using Weather & Soil Data" dataset. The column names have been simplified:

- `prec` corresponds to precipitation.
- `surf_pre` corresponds to surface pressure.
- `humidity` corresponds to specific humidity.
- `tmp`, `tmp_max`, `tmp_min`, and `tmp_range` correspond to temperature variables.
- `dp_tmp` corresponds to dew/frost point temperature.
- `wb_tmp` corresponds to wet bulb temperature.
- `surf_tmp` corresponds to earth/skin surface temperature.
- `wind`, `wind_max`, `wind_min`, and `wind_range` correspond to wind speed variables.

## Recommended Implementation

Build a supervised tabular forecasting pipeline with sliding 91-day windows.

For every region in `train.csv`, create training examples like this:

1. Choose a weekly labeled row as the prediction target.
2. Use only the previous 91 days of meteorological features as input.
3. Create one row of engineered features for that window.
4. Train separate targets for horizons 1 to 5 weeks ahead, or train one model with `horizon` as a categorical/numeric feature.

Recommended main model:

- LightGBM regression, one model per horizon.
- Objective: regression with MAE-style evaluation.
- Prediction post-processing: clip predictions to `[0, 5]`.
- Optional: round only if leaderboard validation shows integer predictions help. Since Kaggle allows floats and evaluates MAE, floats should be the default.

This is a practical fit because the dataset is large, tabular, and mostly meteorological. Gradient boosting should train faster and be easier to explain than sequence neural networks, while still capturing nonlinear weather interactions.

## Feature Engineering Plan

For each 91-day input window, compute features from the 14 meteorological columns.

Window summary features:

- Mean, standard deviation, minimum, maximum, median.
- Recent value: last day, last 3-day mean, last 7-day mean.
- Trend: difference between recent 7-day mean and older 7-day mean.
- Distribution: 10th, 25th, 75th, and 90th percentiles.

Multi-scale history features:

- Last 7 days.
- Last 14 days.
- Last 28 days.
- Last 56 days.
- Full 91 days.
- Twelve non-overlapping weekly-bin summaries, ordered from oldest to newest,
  plus per-variable 13-week trends to retain sequence shape. The thirteenth
  bin is still used by the trends but is not emitted separately because it
  duplicates the existing recent 7-day summaries.
- Region seasonal anomalies over 28 and 91 days, using each region's historical
  monthly weather climatology. Each anomaly subtracts a month-weighted expected
  daily mean from the recent window mean. Training examples must only use
  climatology available before their 91-day input window.

Drought-oriented derived features:

- Dry day count: number of days where `prec == 0`.
- Rainy day count: number of days where `prec > 0`.
- Total precipitation over each window.
- Maximum consecutive dry days in the 91-day window.
- Temperature stress: mean and max of `tmp_max`, `surf_tmp`, and `tmp_range`.
- Humidity stress: low-percentile humidity and recent humidity trend.
- Wind stress: mean/max `wind`, `wind_max`, and `wind_range`.
- Hot-dry interaction features, such as high recent `tmp_max` combined with low recent precipitation.
- Evaporation-risk proxies, such as high `tmp_max`, high `wind`, and low `humidity`.
- Recent-vs-long-term anomaly features, such as last 14-day precipitation minus last 91-day daily average precipitation.

Persistence-oriented features:

- Last known weekly `score` before the prediction window.
- Mean of the last 2, 4, and 8 known weekly scores.
- Trend between the latest known score and older known scores.
- Region-level historical mean, median, and severe-score frequency.

These can be strong because drought severity usually changes gradually. However, test data has no `score` column inside the provided 91-day window. Therefore, any score-lag feature used at inference must come only from historical train data available before the test period. To keep train and test consistent, the first version can use only meteorological features, then add historical score/persistence features as a controlled experiment.

Useful heuristic logic should be implemented as features or post-processing rather than as a fully manual rule system. Good heuristics for this problem are:

- Low precipitation over many days increases drought risk.
- High temperature, high surface temperature, high wind, and low humidity can increase dryness stress.
- Severity is persistent, so predictions should not jump aggressively unless recent weather strongly supports it.
- Final predictions should be clipped to `[0, 5]`.

## Validation Strategy

Use time-based validation that matches the Kaggle setup.

For each region:

1. Use earlier years/windows for training.
2. Hold out the latest part of `train.csv`.
3. From each validation cutoff, use 91 days of weather history.
4. Predict the next five weekly scores.
5. Measure MAE across all regions and horizons.

Avoid random row splitting because adjacent windows from the same region overlap heavily. Random validation would overestimate performance.

Suggested validation sets:

- Baseline validation: last 5 weekly labels per region.
- Robust validation: several rolling cutoffs near the end of the training period.
- Report metric: overall MAE plus MAE by horizon week 1 to week 5.

## Baselines

Implement these before the main model:

1. Global constant baseline
   - Predict the global mean or median score from train.

2. Region historical baseline
   - Predict each region's historical mean score.
   - Fallback to global mean for missing cases.

3. Recent region baseline
   - Predict the mean of the latest available weekly scores from the same region in train.

4. Weather-feature LightGBM baseline
   - Use only simple 91-day means and recent 7-day means.

These baselines are important for the report and for detecting leakage or validation mistakes.

## Experiments For The Report

Recommended ablations:

- Baseline comparison: global mean vs. region mean vs. LightGBM.
- Feature window comparison: 7/14/28 days only vs. full 91-day features.
- Derived drought features: with vs. without dry-day and precipitation features.
- Model structure: one model per horizon vs. one shared model with a `horizon` feature.
- Prediction post-processing: raw clipped floats vs. rounded integers.

Useful analysis:

- Score imbalance and why MAE encourages conservative predictions.
- MAE by horizon, because week 5 should usually be harder than week 1.
- MAE by severity group, especially whether severe drought scores are underpredicted.
- Case studies for regions where the model improves over region mean.

## Implementation Files To Build

Suggested repository structure:

```text
.
├── concept.md
├── README.md
├── data/
│   ├── train.csv
│   └── test.csv
├── sample_submission.csv
├── src/
│   ├── make_features.py
│   ├── train.py
│   ├── validate.py
│   └── predict.py
├── configs/
│   └── lightgbm.yaml
├── models/
└── submissions/
```

Core scripts:

- `make_features.py`: convert daily region sequences into supervised 91-day feature rows.
- `train.py`: train horizon-specific LightGBM models.
- `validate.py`: run rolling time validation and save MAE tables.
- `predict.py`: generate `submission.csv` in the sample submission format.

## First Milestone

For the May 21 progress check, aim to show:

- Data statistics and target distribution.
- A working validation split.
- At least two baselines.
- First LightGBM result with feature importance.
- One Kaggle submission generated by reproducible code.

## Final Recommendation

Start with a meteorological feature engineering plus LightGBM regression system. It is strong enough for leaderboard performance, simple enough to reproduce for TA verification, and easy to explain in the IEEE-style report. The current priority is not changing the backbone model first; it is validating the split, reducing feature extraction cost, and identifying which features actually help. After the baseline and validation are stable, improve with feature ablation, rolling validation, drought-specific features, simple persistence blends, and careful horizon-wise model tuning.
