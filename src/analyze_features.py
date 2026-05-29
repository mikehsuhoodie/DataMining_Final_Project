from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from make_features import HORIZONS, build_validation_set_for_horizon, configure_logging, read_train_csv
from train import make_model


LOGGER = logging.getLogger(__name__)


def make_analysis_model(random_state: int = 42):
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="mae",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_samples=120,
            max_bin=127,
            importance_type="gain",
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
        ), "lightgbm_fast_analysis"
    except Exception as exc:
        LOGGER.warning("Fast LightGBM analysis model unavailable (%s); using default model", exc)
        return make_model(random_state=random_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze feature contribution on the time-based validation split.")
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--output-dir", default="feature_analysis")
    parser.add_argument("--val-weeks", type=int, default=5)
    parser.add_argument("--stride", type=int, default=8, help="Use every Nth weekly target for validation training.")
    parser.add_argument("--max-regions", type=int, default=None)
    parser.add_argument(
        "--max-train-examples-per-horizon",
        type=int,
        default=150_000,
        help="Subsample each horizon after feature generation. Use 0 for no limit.",
    )
    parser.add_argument(
        "--top-k-permutation",
        type=int,
        default=120,
        help="Run permutation only on the top K model-importance features. Use 0 for all features.",
    )
    parser.add_argument("--n-repeats", type=int, default=3)
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
        help="Analyze weather-only models without historical score persistence features.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_regions = args.max_regions
    max_examples = args.max_train_examples_per_horizon or None
    stride = max(1, args.stride)
    if args.debug:
        max_regions = max_regions or 20
        max_examples = min(max_examples or 8_000, 8_000)
        stride = max(stride, 12)

    LOGGER.info("Reading training data from %s", args.train_csv)
    train_df = read_train_csv(args.train_csv, max_regions=max_regions)
    importance_rows = []
    permutation_rows = []
    summary_rows = []

    for h in HORIZONS:
        LOGGER.info("Building validation split for horizon %d", h)
        X_train, y_train, _train_meta, X_val, y_val, _val_meta = build_validation_set_for_horizon(
            train_df,
            horizon=h,
            val_weeks=args.val_weeks,
            stride=stride,
            max_train_examples=max_examples,
            include_score_features=not args.no_score_features,
            n_jobs=args.n_jobs,
        )
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)
        if len(X_train) == 0 or len(X_val) == 0:
            LOGGER.warning("Skipping horizon %d because train or validation data is empty", h)
            continue

        model, model_kind = make_analysis_model(random_state=42 + h)
        LOGGER.info("Training horizon %d %s on %d rows", h, model_kind, len(X_train))
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_val), 0.0, 5.0)
        model_mae = mean_absolute_error(y_val, pred)

        importances = _model_importances(model, X_train.columns)
        for rank, (feature, value) in enumerate(importances, start=1):
            importance_rows.append(
                {
                    "horizon": h,
                    "feature": feature,
                    "model_importance_rank": rank,
                    "model_importance": value,
                    "feature_group": _feature_group(feature),
                }
            )

        top_k = args.top_k_permutation if args.top_k_permutation > 0 else len(importances)
        selected_features = [feature for feature, _value in importances[:top_k]]
        LOGGER.info(
            "Running permutation for horizon %d on %d features x %d repeats",
            h,
            len(selected_features),
            args.n_repeats,
        )
        permuted = _permutation_mae_increase(model, X_val, y_val, selected_features, args.n_repeats, 42 + h)
        for feature, mean_increase, std_increase in permuted:
            permutation_rows.append(
                {
                    "horizon": h,
                    "feature": feature,
                    "mae_increase_mean": mean_increase,
                    "mae_increase_std": std_increase,
                    "feature_group": _feature_group(feature),
                }
            )

        summary_rows.append(
            {
                "horizon": h,
                "model_kind": model_kind,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "model_mae": model_mae,
                "permuted_features": len(selected_features),
            }
        )
        del X_train, y_train, X_val, y_val, model
        gc.collect()

    importance = pd.DataFrame(importance_rows)
    permutation = pd.DataFrame(permutation_rows)
    summary = pd.DataFrame(summary_rows)

    if not permutation.empty:
        permutation = permutation.sort_values(["horizon", "mae_increase_mean"], ascending=[True, False])
        group_summary = (
            permutation.groupby(["horizon", "feature_group"], as_index=False)
            .agg(mae_increase_mean=("mae_increase_mean", "sum"), n_features=("feature", "count"))
            .sort_values(["horizon", "mae_increase_mean"], ascending=[True, False])
        )
        group_summary.to_csv(output_dir / "feature_group_permutation.csv", index=False)

    importance.to_csv(output_dir / "model_importance.csv", index=False)
    permutation.to_csv(output_dir / "permutation_importance.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    LOGGER.info("Saved feature analysis outputs under %s", output_dir)


def _model_importances(model, columns: pd.Index) -> list[tuple[str, float]]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        values = np.zeros(len(columns), dtype=np.float32)
    pairs = [(feature, float(value)) for feature, value in zip(columns, values)]
    return sorted(pairs, key=lambda item: item[1], reverse=True)


def _permutation_mae_increase(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    features: list[str],
    n_repeats: int,
    random_state: int,
) -> list[tuple[str, float, float]]:
    rng = np.random.default_rng(random_state)
    baseline_pred = np.clip(model.predict(X_val), 0.0, 5.0)
    baseline_mae = mean_absolute_error(y_val, baseline_pred)
    X_work = X_val.copy()
    out = []

    for feature in features:
        original = X_work[feature].to_numpy(copy=True)
        increases = []
        for _ in range(max(1, n_repeats)):
            X_work[feature] = rng.permutation(original)
            pred = np.clip(model.predict(X_work), 0.0, 5.0)
            increases.append(mean_absolute_error(y_val, pred) - baseline_mae)
        X_work[feature] = original
        out.append((feature, float(np.mean(increases)), float(np.std(increases))))

    return sorted(out, key=lambda item: item[1], reverse=True)


def _feature_group(feature: str) -> str:
    if feature.startswith("score_"):
        return "score_history"
    if feature in {"month", "day_of_year", "week_of_year", "day_index_mod_365", "doy_sin", "doy_cos"}:
        return "seasonality"
    if feature.startswith("dry_days") or feature.startswith("rainy_days") or feature.startswith("max_consecutive_dry"):
        return "precipitation_event"
    if feature.startswith("hot_dry") or feature.startswith("evap_risk") or feature.startswith("surface_heat"):
        return "drought_interaction"
    if feature.startswith("prec_"):
        return "precipitation"
    return feature.split("_", 1)[0]


if __name__ == "__main__":
    main()
