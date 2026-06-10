from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from make_features import (
    HORIZONS,
    LOW_GAIN_FILTERED_FEATURES,
    PRED_COLS,
    align_feature_columns,
    build_test_features,
    configure_logging,
    read_test_csv,
    read_train_csv,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission predictions.")
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument("--train-csv", default="data/train.csv", help="Used for region weather climatology and optional score history.")
    parser.add_argument("--sample-submission", default="sample_submission.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output", default="submissions/submission.csv")
    parser.add_argument(
        "--mode",
        choices=["model", "zero", "latest-score", "blend-zero", "blend-latest"],
        default="model",
        help="Prediction mode. zero is a sanity baseline; blend modes calibrate model predictions.",
    )
    parser.add_argument(
        "--zero-weight",
        type=float,
        default=0.5,
        help="For --mode blend-zero, fraction of the model prediction to keep. 0.25 means 75%% shrink toward zero.",
    )
    parser.add_argument(
        "--latest-weight",
        type=float,
        default=0.5,
        help="For --mode blend-latest, fraction of the latest-region-score prediction to use.",
    )
    parser.add_argument("--max-regions", type=int, default=None, help="Debug feature generation on a subset.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(args.sample_submission)

    if args.mode == "zero":
        submission = sample.copy()
        submission[PRED_COLS] = 0.0
        write_submission(submission, args.output)
        return

    if args.mode == "latest-score":
        LOGGER.info("Reading train labels from %s for latest-score baseline", args.train_csv)
        train_df = read_train_csv(args.train_csv, max_regions=None)
        submission = build_latest_score_submission(sample, train_df)
        write_submission(submission, args.output)
        return

    model_dir = Path(args.model_dir)
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}. Run python src/train.py first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    include_score_features = bool(metadata.get("include_score_features", True))
    feature_columns_by_horizon = metadata["feature_columns_by_horizon"]
    legacy_ablation_features = {
        "region_seasonal_coverage_w28",
        "region_seasonal_coverage_w91",
        "day_index_mod_365",
        "prec_week13_sum",
    }
    include_legacy_ablation_features = any(
        legacy_ablation_features.intersection(feature_columns)
        for feature_columns in feature_columns_by_horizon.values()
    )
    include_raw_wind_features = any(
        "wind_w7_mean" in feature_columns
        for feature_columns in feature_columns_by_horizon.values()
    )
    include_prec_w56_min = any(
        "prec_w56_min" in feature_columns
        for feature_columns in feature_columns_by_horizon.values()
    )
    include_raw_wb_tmp_features = any(
        "wb_tmp_w7_mean" in feature_columns
        for feature_columns in feature_columns_by_horizon.values()
    )
    include_low_gain_filtered_features = any(
        LOW_GAIN_FILTERED_FEATURES.intersection(feature_columns)
        for feature_columns in feature_columns_by_horizon.values()
    )
    LOGGER.info("Reading train weather history from %s", args.train_csv)
    train_df = read_train_csv(args.train_csv, max_regions=None)

    LOGGER.info("Reading test data from %s", args.test_csv)
    test_df = read_test_csv(args.test_csv, max_regions=args.max_regions)
    LOGGER.info("Building test features")
    X_test, meta = build_test_features(
        test_df,
        train_df,
        include_score_features=include_score_features,
        include_legacy_ablation_features=include_legacy_ablation_features,
        include_raw_wind_features=include_raw_wind_features,
        include_prec_w56_min=include_prec_w56_min,
        include_raw_wb_tmp_features=include_raw_wb_tmp_features,
        include_low_gain_filtered_features=include_low_gain_filtered_features,
    )

    pred_by_horizon = {}
    for h in HORIZONS:
        model_path = model_dir / f"horizon_{h}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}. Run python src/train.py first.")
        model = joblib.load(model_path)
        feature_columns = feature_columns_by_horizon[str(h)]
        Xh = align_feature_columns(X_test, feature_columns)
        pred_by_horizon[h] = np.clip(model.predict(Xh), 0.0, 5.0)
        LOGGER.info("Predicted horizon %d", h)

    pred_df = pd.DataFrame({"region_id": meta["region_id"].astype(str)})
    for h, col in zip(HORIZONS, PRED_COLS):
        pred_df[col] = pred_by_horizon[h]

    submission = sample[["region_id"]].merge(pred_df, on="region_id", how="left")
    missing = submission[PRED_COLS].isna().any(axis=1)
    if missing.any():
        fallback = float(metadata.get("target_mean", 0.0))
        LOGGER.warning("Filling %d missing sample regions with target mean %.4f", int(missing.sum()), fallback)
        submission.loc[missing, PRED_COLS] = fallback
    submission = submission[sample.columns]
    submission[PRED_COLS] = submission[PRED_COLS].astype(float).clip(0.0, 5.0)
    if args.mode == "blend-zero":
        keep_weight = min(max(args.zero_weight, 0.0), 1.0)
        LOGGER.info("Shrinking model predictions toward zero with keep weight %.4f", keep_weight)
        submission[PRED_COLS] = submission[PRED_COLS] * keep_weight
    elif args.mode == "blend-latest":
        latest_weight = min(max(args.latest_weight, 0.0), 1.0)
        LOGGER.info("Blending model predictions with latest score using latest weight %.4f", latest_weight)
        latest_submission = build_latest_score_submission(sample, train_df)
        submission[PRED_COLS] = (
            submission[PRED_COLS] * (1.0 - latest_weight)
            + latest_submission[PRED_COLS] * latest_weight
        ).clip(0.0, 5.0)

    write_submission(submission, args.output)


def build_latest_score_submission(sample: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    labeled = train_df.dropna(subset=["score"])
    latest = labeled.groupby("region_id", sort=False)["score"].last()
    fallback = float(labeled["score"].mean())
    submission = sample.copy()
    values = sample["region_id"].astype(str).map(latest).fillna(fallback).astype(float)
    for col in PRED_COLS:
        submission[col] = values
    submission[PRED_COLS] = submission[PRED_COLS].clip(0.0, 5.0)
    return submission


def write_submission(submission: pd.DataFrame, output: str) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    LOGGER.info("Wrote %s with shape %s", out_path, submission.shape)


if __name__ == "__main__":
    main()
