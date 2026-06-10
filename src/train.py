from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import shlex
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from make_features import (
    HORIZONS,
    align_feature_columns,
    build_training_set_for_horizon,
    configure_logging,
    read_train_csv,
)


LOGGER = logging.getLogger(__name__)
FEATURE_VERSION_CHOICES = ("filtered_641", "aggressive_620")
DEFAULT_FEATURE_VERSION = "filtered_641"


def include_low_gain_filtered_features(feature_version: str) -> bool:
    if feature_version == "filtered_641":
        return True
    if feature_version == "aggressive_620":
        return False
    raise ValueError(f"Unknown feature version: {feature_version}")


def make_model(
    model_kind: str = "lightgbm",
    random_state: int = 42,
    model_n_jobs: int = -1,
    rf_n_estimators: int = 80,
    rf_max_depth: int | None = 24,
    rf_max_samples: float = 0.5,
    rf_min_samples_leaf: int = 10,
    rf_max_features: str | float = "sqrt",
    rf_verbose: int = 1,
):
    if model_kind == "lightgbm":
        try:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise RuntimeError("LightGBM is not available. Install lightgbm in the project venv.") from exc

        return LGBMRegressor(
            objective="mae",
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            min_child_samples=80,
            n_jobs=model_n_jobs,
            random_state=random_state,
            verbosity=-1,
        ), "lightgbm"

    if model_kind == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise RuntimeError("XGBoost is not available. Install xgboost in the project venv.") from exc

        return XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            n_estimators=450,
            learning_rate=0.035,
            max_leaves=63,
            grow_policy="lossguide",
            tree_method="hist",
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            min_child_weight=80,
            n_jobs=model_n_jobs,
            random_state=random_state,
            verbosity=0,
        ), "xgboost"

    if model_kind == "catboost":
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:
            raise RuntimeError("CatBoost is not available. Install catboost in the project venv.") from exc

        return CatBoostRegressor(
            loss_function="MAE",
            iterations=450,
            learning_rate=0.035,
            depth=8,
            l2_leaf_reg=2.0,
            random_seed=random_state,
            thread_count=model_n_jobs,
            verbose=False,
            allow_writing_files=False,
        ), "catboost"

    if model_kind == "random_forest":
        try:
            from sklearn.ensemble import RandomForestRegressor
        except Exception as exc:
            raise RuntimeError("scikit-learn is not available. Install scikit-learn in the project venv.") from exc

        return RandomForestRegressor(
            n_estimators=rf_n_estimators,
            criterion="squared_error",
            max_depth=rf_max_depth,
            max_features=rf_max_features,
            min_samples_leaf=rf_min_samples_leaf,
            bootstrap=True,
            max_samples=rf_max_samples,
            n_jobs=model_n_jobs,
            random_state=random_state,
            verbose=rf_verbose,
        ), "random_forest"

    raise ValueError(f"Unknown model kind: {model_kind}")


def model_importance_rows(model, model_kind: str, horizon: int, feature_columns: list[str]) -> list[dict]:
    booster = getattr(model, "booster_", None)
    if booster is not None:
        split_importance = np.asarray(booster.feature_importance(importance_type="split"))
        gain_importance = np.asarray(booster.feature_importance(importance_type="gain"))
    else:
        split_importance = np.asarray(getattr(model, "feature_importances_", np.zeros(len(feature_columns))))
        gain_importance = split_importance

    order = np.argsort(-gain_importance, kind="stable")
    return [
        {
            "horizon": horizon,
            "model_kind": model_kind,
            "gain_rank": rank,
            "feature": feature_columns[idx],
            "split_importance": float(split_importance[idx]),
            "gain_importance": float(gain_importance[idx]),
        }
        for rank, idx in enumerate(order, start=1)
    ]


def write_model_importance(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cache_config(
    train_csv: str,
    stride: int,
    max_regions: int | None,
    max_examples: int | None,
    include_score_features: bool,
    feature_version: str,
    n_jobs: int,
) -> dict:
    return {
        "train_csv": train_csv,
        "stride": stride,
        "max_regions": max_regions,
        "max_train_examples_per_horizon": max_examples,
        "include_score_features": include_score_features,
        "score_history_timing": "window_start" if include_score_features else "disabled",
        "feature_n_jobs": n_jobs,
        "feature_matrix_dtype": "float32",
        "feature_version": feature_version,
        "feature_filter_version": feature_version,
    }


def cache_path(cache_dir: Path, horizon: int) -> Path:
    return cache_dir / f"feature_horizon_{horizon}.joblib"


def load_or_build_training_set(
    train_df,
    horizon: int,
    args: argparse.Namespace,
    stride: int,
    max_examples: int | None,
    cache_dir: Path | None,
) -> tuple:
    path = cache_path(cache_dir, horizon) if cache_dir else None
    if path and path.exists():
        LOGGER.info("Loading cached supervised features for horizon %d from %s", horizon, path)
        payload = joblib.load(path)
        return payload["X"], payload["y"], payload["meta"]

    LOGGER.info("Building supervised features for horizon %d", horizon)
    X, y, meta = build_training_set_for_horizon(
        train_df,
        horizon=horizon,
        stride=stride,
        max_train_examples=max_examples,
        include_score_features=not args.no_score_features,
        include_low_gain_filtered_features=include_low_gain_filtered_features(args.feature_version),
        n_jobs=args.n_jobs,
    )
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving supervised feature cache for horizon %d to %s", horizon, path)
        joblib.dump({"X": X, "y": y, "meta": meta}, path, compress=0)
    return X, y, meta


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
        "--n-jobs",
        type=int,
        default=1,
        help="Number of worker processes for feature generation. Use 0 for conservative auto mode.",
    )
    parser.add_argument(
        "--no-score-features",
        action="store_true",
        help="Train weather-only models without historical score persistence features.",
    )
    parser.add_argument(
        "--model-kind",
        choices=["lightgbm", "xgboost", "catboost", "random_forest"],
        default="lightgbm",
        help="Model backend to train.",
    )
    parser.add_argument(
        "--model-n-jobs",
        type=int,
        default=-1,
        help="Thread/process count passed to the model backend. This is separate from feature-generation --n-jobs.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=80,
        help="RandomForest only: number of trees. The old value 300 is very slow for 150k x 641.",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=24,
        help="RandomForest only: maximum tree depth. Use 0 for unlimited depth.",
    )
    parser.add_argument(
        "--rf-max-samples",
        type=float,
        default=0.5,
        help="RandomForest only: row fraction sampled per tree when bootstrap=True.",
    )
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=10,
        help="RandomForest only: minimum samples per leaf.",
    )
    parser.add_argument(
        "--rf-max-features",
        default="sqrt",
        help="RandomForest only: max_features. Use sqrt, log2, or a float such as 0.7.",
    )
    parser.add_argument(
        "--rf-verbose",
        type=int,
        default=1,
        help="RandomForest only: sklearn verbose level, useful because RF training can be long.",
    )
    parser.add_argument(
        "--feature-version",
        choices=FEATURE_VERSION_CHOICES,
        default=DEFAULT_FEATURE_VERSION,
        help="Feature set version. filtered_641 is the report/default setup; aggressive_620 removes extra low-gain features.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        default=None,
        help="Directory for per-horizon feature caches. Existing horizon caches are reused automatically.",
    )
    parser.add_argument(
        "--feature-cache-only",
        action="store_true",
        help="Build or refresh feature caches, then exit before model training.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--drop-feature", type=str, action="append", help="Feature name to drop from training (can specify multiple times).")
    return parser.parse_args()


def main() -> None:
    run_started_at = time.perf_counter()
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
    rf_max_depth = None if args.rf_max_depth == 0 else args.rf_max_depth
    try:
        rf_max_features: str | float = float(args.rf_max_features)
    except ValueError:
        rf_max_features = args.rf_max_features

    LOGGER.info("Reading training data from %s", args.train_csv)
    train_df = read_train_csv(args.train_csv, max_regions=max_regions)
    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None
    feature_cache_config = cache_config(
        args.train_csv,
        stride,
        max_regions,
        max_examples,
        include_score_features=not args.no_score_features,
        feature_version=args.feature_version,
        n_jobs=args.n_jobs,
    )
    if feature_cache_dir:
        feature_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_metadata_path = feature_cache_dir / "cache_metadata.json"
        if cache_metadata_path.exists():
            existing_cache_config = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
            if existing_cache_config != feature_cache_config:
                raise RuntimeError(
                    f"Feature cache config mismatch in {cache_metadata_path}. "
                    "Use a different --feature-cache-dir or remove the old cache."
                )
        else:
            cache_metadata_path.write_text(json.dumps(feature_cache_config, indent=2), encoding="utf-8")

    metadata = {
        "model_kind_by_horizon": {},
        "stride": stride,
        "max_regions": max_regions,
        "max_train_examples_per_horizon": max_examples,
        "include_score_features": not args.no_score_features,
        "score_history_timing": "window_start" if not args.no_score_features else "disabled",
        "feature_n_jobs": args.n_jobs,
        "feature_matrix_dtype": "float32",
        "feature_version": args.feature_version,
        "feature_filter_version": args.feature_version,
        "include_low_gain_filtered_features": include_low_gain_filtered_features(args.feature_version),
        "requested_model_kind": args.model_kind,
        "model_n_jobs": args.model_n_jobs,
        "random_forest_params": {
            "n_estimators": args.rf_n_estimators,
            "max_depth": rf_max_depth,
            "max_samples": args.rf_max_samples,
            "min_samples_leaf": args.rf_min_samples_leaf,
            "max_features": rf_max_features,
            "verbose": args.rf_verbose,
        },
        "train_command": shlex.join([sys.executable, *sys.argv]),
        "feature_cache_dir": str(feature_cache_dir) if feature_cache_dir else None,
    }
    feature_columns_by_horizon = {}
    importance_rows = []

    for h in HORIZONS:
        X, y, _meta = load_or_build_training_set(
            train_df,
            h,
            args,
            stride,
            max_examples,
            feature_cache_dir,
        )
        if len(X) == 0:
            raise RuntimeError(f"No training examples were built for horizon {h}.")
        feature_columns = list(X.columns)
        
        # Feature ablation
        if args.drop_feature:
            X = X.drop(columns=[c for c in args.drop_feature if c in X.columns])
            feature_columns = list(X.columns)  # 更新
        
        X = align_feature_columns(X, feature_columns)
        feature_columns_by_horizon[str(h)] = feature_columns

        if args.feature_cache_only:
            LOGGER.info("Skipping training for horizon %d because --feature-cache-only is set", h)
            del X, y
            gc.collect()
            continue

        model, model_kind = make_model(
            args.model_kind,
            random_state=42 + h,
            model_n_jobs=args.model_n_jobs,
            rf_n_estimators=args.rf_n_estimators,
            rf_max_depth=rf_max_depth,
            rf_max_samples=args.rf_max_samples,
            rf_min_samples_leaf=args.rf_min_samples_leaf,
            rf_max_features=rf_max_features,
            rf_verbose=args.rf_verbose,
        )
        LOGGER.info("Training horizon %d %s model on %d rows x %d features", h, model_kind, len(X), X.shape[1])
        model.fit(X, y)
        importance_rows.extend(model_importance_rows(model, model_kind, h, feature_columns))
        model_path = model_dir / f"horizon_{h}.joblib"
        joblib.dump(model, model_path)
        metadata["model_kind_by_horizon"][str(h)] = model_kind
        LOGGER.info("Saved %s", model_path)
        del X, y, model
        gc.collect()

    if args.feature_cache_only:
        metadata["feature_columns_by_horizon"] = feature_columns_by_horizon
        metadata["target_mean"] = float(np.nanmean(train_df["score"].to_numpy(dtype=np.float32)))
        metadata["train_runtime_seconds"] = round(time.perf_counter() - run_started_at, 3)
        metadata["train_runtime_minutes"] = round(metadata["train_runtime_seconds"] / 60.0, 3)
        metadata["feature_cache_only"] = True
        metadata_path = model_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.info("Saved %s", metadata_path)
        LOGGER.info("Feature cache build complete; no models were trained.")
        return

    metadata["feature_columns_by_horizon"] = feature_columns_by_horizon
    metadata["target_mean"] = float(np.nanmean(train_df["score"].to_numpy(dtype=np.float32)))
    metadata["train_runtime_seconds"] = round(time.perf_counter() - run_started_at, 3)
    metadata["train_runtime_minutes"] = round(metadata["train_runtime_seconds"] / 60.0, 3)
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s", metadata_path)
    importance_path = model_dir / "feature_importance.csv"
    write_model_importance(importance_rows, importance_path)
    LOGGER.info("Saved %s", importance_path)


if __name__ == "__main__":
    main()
