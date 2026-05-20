from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from make_features import HORIZONS, build_validation_sets, configure_logging, read_train_csv
from train import make_model


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-based validation for drought forecasting.")
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--output-dir", default="validation")
    parser.add_argument("--val-weeks", type=int, default=5)
    parser.add_argument("--stride", type=int, default=8, help="Use every Nth weekly target for validation training.")
    parser.add_argument("--max-regions", type=int, default=None)
    parser.add_argument(
        "--max-train-examples-per-horizon",
        type=int,
        default=150_000,
        help="Subsample each horizon after feature generation. Use 0 for no limit.",
    )
    parser.add_argument("--debug", action="store_true", help="Fast run using 20 regions and fewer examples.")
    parser.add_argument(
        "--no-score-features",
        action="store_true",
        help="Validate weather-only models without historical score persistence features.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def baseline_scores(train_df: pd.DataFrame, val_meta: pd.DataFrame, y_true: np.ndarray) -> dict[str, float]:
    global_mean = float(np.nanmean(train_df["score"].to_numpy(dtype=np.float32)))
    labeled = train_df.dropna(subset=["score"])
    region_mean = labeled.groupby("region_id", sort=False)["score"].mean().to_dict()
    region_recent = labeled.groupby("region_id", sort=False)["score"].apply(lambda s: s.tail(8).mean()).to_dict()

    regions = val_meta["region_id"].astype(str).to_numpy()
    pred_global = np.full(len(y_true), global_mean, dtype=np.float32)
    pred_region = np.array([region_mean.get(r, global_mean) for r in regions], dtype=np.float32)
    pred_recent = np.array([region_recent.get(r, region_mean.get(r, global_mean)) for r in regions], dtype=np.float32)

    return {
        "global_mean": mean_absolute_error(y_true, pred_global),
        "region_mean": mean_absolute_error(y_true, pred_region),
        "recent_region_mean": mean_absolute_error(y_true, pred_recent),
    }


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
    LOGGER.info("Building validation split")
    train_data, val_data = build_validation_sets(
        train_df,
        val_weeks=args.val_weeks,
        stride=stride,
        max_train_examples_per_horizon=max_examples,
        include_score_features=not args.no_score_features,
    )

    rows = []
    all_y = []
    all_model_pred = []
    all_global = []
    all_region = []
    all_recent = []

    for h in HORIZONS:
        X_train = train_data.X_by_horizon[h]
        y_train = train_data.y_by_horizon[h]
        X_val = val_data.X_by_horizon[h].reindex(columns=X_train.columns, fill_value=0.0)
        y_val = val_data.y_by_horizon[h]
        if len(X_train) == 0 or len(X_val) == 0:
            LOGGER.warning("Skipping horizon %d because train or validation data is empty", h)
            continue

        base = baseline_scores(train_df, val_data.meta_by_horizon[h], y_val)
        model, model_kind = make_model(random_state=42 + h)
        LOGGER.info("Training validation horizon %d %s on %d rows", h, model_kind, len(X_train))
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_val), 0.0, 5.0)
        model_mae = mean_absolute_error(y_val, pred)

        rows.append(
            {
                "horizon": h,
                "n_train": len(X_train),
                "n_val": len(X_val),
                **base,
                "model": model_mae,
                "model_kind": model_kind,
            }
        )

        all_y.append(y_val)
        all_model_pred.append(pred)
        all_global.append(np.full(len(y_val), np.nanmean(train_df["score"]), dtype=np.float32))
        labeled = train_df.dropna(subset=["score"])
        region_mean = labeled.groupby("region_id", sort=False)["score"].mean().to_dict()
        region_recent = labeled.groupby("region_id", sort=False)["score"].apply(lambda s: s.tail(8).mean()).to_dict()
        regions = val_data.meta_by_horizon[h]["region_id"].astype(str).to_numpy()
        all_region.append(np.array([region_mean.get(r, np.nanmean(train_df["score"])) for r in regions], dtype=np.float32))
        all_recent.append(np.array([region_recent.get(r, region_mean.get(r, np.nanmean(train_df["score"]))) for r in regions], dtype=np.float32))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        y_all = np.concatenate(all_y)
        model_all = np.concatenate(all_model_pred)
        global_all = np.concatenate(all_global)
        region_all = np.concatenate(all_region)
        recent_all = np.concatenate(all_recent)
        overall = pd.DataFrame(
            [
                {
                    "horizon": "overall",
                    "n_train": sum(int(r["n_train"]) for r in rows),
                    "n_val": len(y_all),
                    "global_mean": mean_absolute_error(y_all, global_all),
                    "region_mean": mean_absolute_error(y_all, region_all),
                    "recent_region_mean": mean_absolute_error(y_all, recent_all),
                    "model": mean_absolute_error(y_all, model_all),
                    "model_kind": ",".join(sorted(set(summary["model_kind"]))),
                }
            ]
        )
        summary = pd.concat([summary, overall], ignore_index=True)

    out_path = output_dir / "validation_mae.csv"
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    LOGGER.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
