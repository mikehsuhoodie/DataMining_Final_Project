from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from make_features import (
    HORIZONS,
    align_feature_columns,
    build_training_set_for_horizon,
    configure_logging,
    read_train_csv,
)


LOGGER = logging.getLogger(__name__)


def make_model(random_state: int = 42):
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="mae",
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            min_child_samples=80,
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
        ), "lightgbm"
    except Exception as exc:
        LOGGER.warning("LightGBM unavailable (%s); using sklearn fallback", exc)

    try:
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=350,
            learning_rate=0.05,
            l2_regularization=0.05,
            random_state=random_state,
        ), "hist_gradient_boosting"
    except Exception as exc:
        LOGGER.warning("HistGradientBoosting unavailable (%s); using ExtraTrees", exc)
        return ExtraTreesRegressor(
            n_estimators=250,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        ), "extra_trees"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train drought forecasting models.")
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--stride", type=int, default=4, help="Use every Nth weekly target for training.")
    parser.add_argument("--max-regions", type=int, default=None, help="Limit regions for quick checks.")
    parser.add_argument(
        "--max-train-examples-per-horizon",
        type=int,
        default=250_000,
        help="Subsample each horizon after feature generation. Use 0 for no limit.",
    )
    parser.add_argument("--debug", action="store_true", help="Fast run using 20 regions and fewer examples.")
    parser.add_argument(
        "--no-score-features",
        action="store_true",
        help="Train weather-only models without historical score persistence features.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    max_regions = args.max_regions
    max_examples = args.max_train_examples_per_horizon or None
    stride = max(1, args.stride)
    if args.debug:
        max_regions = max_regions or 20
        max_examples = min(max_examples or 10_000, 10_000)
        stride = max(stride, 12)

    LOGGER.info("Reading training data from %s", args.train_csv)
    train_df = read_train_csv(args.train_csv, max_regions=max_regions)
    metadata = {
        "model_kind_by_horizon": {},
        "feature_columns_by_horizon": {},
        "stride": stride,
        "max_regions": max_regions,
        "max_train_examples_per_horizon": max_examples,
        "include_score_features": not args.no_score_features,
        "score_history_timing": "window_start" if not args.no_score_features else "disabled",
    }

    for h in HORIZONS:
        LOGGER.info("Building supervised features for horizon %d", h)
        X, y, _meta = build_training_set_for_horizon(
            train_df,
            horizon=h,
            stride=stride,
            max_train_examples=max_examples,
            include_score_features=not args.no_score_features,
        )
        if len(X) == 0:
            raise RuntimeError(f"No training examples were built for horizon {h}.")
        feature_columns = list(X.columns)
        X = align_feature_columns(X, feature_columns)

        model, model_kind = make_model(random_state=42 + h)
        LOGGER.info("Training horizon %d %s model on %d rows x %d features", h, model_kind, len(X), X.shape[1])
        model.fit(X, y)
        model_path = model_dir / f"horizon_{h}.joblib"
        joblib.dump(model, model_path)
        metadata["model_kind_by_horizon"][str(h)] = model_kind
        metadata["feature_columns_by_horizon"][str(h)] = feature_columns
        LOGGER.info("Saved %s", model_path)
        del X, y, model
        gc.collect()

    metadata["target_mean"] = float(np.nanmean(train_df["score"].to_numpy(dtype=np.float32)))
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s", metadata_path)


if __name__ == "__main__":
    main()
