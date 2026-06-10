"""Feature generation for the drought severity forecasting pipeline.

The public test set contains one 91-day weather block per region.  Training
examples are built to match that shape: choose a cutoff day, summarize only the
91 days before that cutoff, then predict labels 1..5 weeks after the cutoff.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

WEATHER_COLS = [
    "prec",
    "surf_pre",
    "humidity",
    "tmp",
    "dp_tmp",
    "wb_tmp",
    "tmp_max",
    "tmp_min",
    "tmp_range",
    "surf_tmp",
    "wind",
    "wind_max",
    "wind_min",
    "wind_range",
]

PRED_COLS = [f"pred_week{i}" for i in range(1, 6)]
WINDOW_DAYS = 91
HORIZONS = [1, 2, 3, 4, 5]
ROLLING_WINDOWS = [7, 14, 28, 56, 91]
PERCENTILES = [10, 50, 90]
WEEKLY_BIN_DAYS = 7
WEEKLY_BIN_COUNT = WINDOW_DAYS // WEEKLY_BIN_DAYS
REGION_SEASONAL_WINDOWS = [28, 91]
LOW_GAIN_FILTERED_FEATURES = {
    "prec_w91_min",
    "prec_w28_min",
    "prec_w91_q10",
    "prec_w56_q10",
    "prec_w28_q10",
    "rainy_days_w7",
    "rainy_days_w14",
    "dry_fraction_w91",
    "tmp_min_last7_mean",
    "tmp_last7_mean",
    "tmp_max_last7_mean",
    "surf_tmp_last7_mean",
    "dp_tmp_last7_mean",
    "prec_last7_mean",
    "surf_pre_last7_mean",
    "surf_pre_w91_mean",
    "surf_pre_w56_mean",
    "surf_pre_w28_mean",
    "surf_pre_w91_median",
    "surf_tmp_w14_mean",
    "tmp_min_w56_mean",
}

COL_IDX = {c: i for i, c in enumerate(WEATHER_COLS)}
PREC_IDX = COL_IDX["prec"]
TMP_MAX_IDX = COL_IDX["tmp_max"]
HUMIDITY_IDX = COL_IDX["humidity"]
WIND_IDX = COL_IDX["wind"]
WIND_MAX_IDX = COL_IDX["wind_max"]
SURF_TMP_IDX = COL_IDX["surf_tmp"]

FLOAT_DTYPE = {c: "float32" for c in WEATHER_COLS}
TRAIN_DTYPES = {"region_id": "string", "score": "float32", **FLOAT_DTYPE}
TEST_DTYPES = {"region_id": "string", **FLOAT_DTYPE}

MONTH_START_DOY = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


@dataclass(frozen=True)
class FeatureBuildResult:
    X_by_horizon: dict[int, pd.DataFrame]
    y_by_horizon: dict[int, np.ndarray]
    meta_by_horizon: dict[int, pd.DataFrame]


@dataclass(frozen=True)
class RegionSeasonalContext:
    """Monthly weather climatology, optionally stored as expanding prefixes."""

    sums: np.ndarray | tuple[np.ndarray, ...]
    counts: np.ndarray | tuple[np.ndarray, ...]
    row_indices: tuple[np.ndarray, ...] | None = None


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def read_train_csv(path: str | Path, max_regions: int | None = None) -> pd.DataFrame:
    usecols = ["region_id", "date", *WEATHER_COLS, "score"]
    df = pd.read_csv(path, usecols=usecols, dtype=TRAIN_DTYPES)
    return _limit_regions(df, max_regions)


def read_test_csv(path: str | Path, max_regions: int | None = None) -> pd.DataFrame:
    usecols = ["region_id", "date", *WEATHER_COLS]
    df = pd.read_csv(path, usecols=usecols, dtype=TEST_DTYPES)
    return _limit_regions(df, max_regions)


def _limit_regions(df: pd.DataFrame, max_regions: int | None) -> pd.DataFrame:
    if not max_regions:
        return df
    regions = pd.unique(df["region_id"])[:max_regions]
    LOGGER.info("Limiting to %d regions", len(regions))
    return df[df["region_id"].isin(regions)].copy()


def build_validation_sets(
    train_df: pd.DataFrame,
    val_weeks: int = 5,
    stride: int = 4,
    max_train_examples_per_horizon: int | None = None,
    include_score_features: bool = True,
    include_low_gain_filtered_features: bool = True,
    n_jobs: int = 1,
) -> tuple[FeatureBuildResult, FeatureBuildResult]:
    """Build time-based train/validation examples.

    For each region, validation uses the latest five weekly labels as the
    future horizon.  Training examples use only earlier target labels.
    """

    n_regions = train_df["region_id"].nunique()
    per_region_budget = _per_region_budget(max_train_examples_per_horizon, n_regions)
    train_rows = {h: [] for h in HORIZONS}
    train_y = {h: [] for h in HORIZONS}
    train_meta = {h: [] for h in HORIZONS}
    val_rows = {h: [] for h in HORIZONS}
    val_y = {h: [] for h in HORIZONS}
    val_meta = {h: [] for h in HORIZONS}

    worker_args = (
        (*payload, val_weeks, stride, per_region_budget, include_score_features, include_low_gain_filtered_features)
        for payload in _iter_region_arrays(train_df)
    )
    for region_result in _run_region_workers(_build_validation_region, worker_args, n_jobs):
        _extend_horizon_dicts(train_rows, train_y, train_meta, *region_result["train"])
        _extend_horizon_dicts(val_rows, val_y, val_meta, *region_result["val"])

    feature_columns = _feature_columns(include_score_features, include_low_gain_filtered_features)
    train_result = _to_result(train_rows, train_y, train_meta, max_train_examples_per_horizon, feature_columns)
    val_result = _to_result(val_rows, val_y, val_meta, None, feature_columns)
    return train_result, val_result


def build_validation_set_for_horizon(
    train_df: pd.DataFrame,
    horizon: int,
    val_weeks: int = 5,
    stride: int = 4,
    max_train_examples: int | None = None,
    include_score_features: bool = True,
    include_low_gain_filtered_features: bool = True,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build one validation horizon at a time to keep peak memory bounded."""

    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")

    n_regions = train_df["region_id"].nunique()
    per_region_budget = _per_region_budget(max_train_examples, n_regions)
    train_rows = []
    train_y = []
    train_meta = []
    val_rows = []
    val_y = []
    val_meta = []

    worker_args = (
        (*payload, horizon, val_weeks, stride, per_region_budget, include_score_features, include_low_gain_filtered_features)
        for payload in _iter_region_arrays(train_df)
    )
    for region_train_rows, region_train_y, region_train_meta, region_val_rows, region_val_y, region_val_meta in (
        _run_region_workers(_build_validation_region_for_horizon, worker_args, n_jobs)
    ):
        _append_feature_chunk(train_rows, region_train_rows)
        train_y.extend(region_train_y)
        train_meta.extend(region_train_meta)
        _append_feature_chunk(val_rows, region_val_rows)
        val_y.extend(region_val_y)
        val_meta.extend(region_val_meta)

    feature_columns = _feature_columns(include_score_features, include_low_gain_filtered_features)
    X_train, y_train, meta_train = _to_frame(train_rows, train_y, train_meta, max_train_examples, horizon, feature_columns)
    X_val, y_val, meta_val = _to_frame(val_rows, val_y, val_meta, None, horizon, feature_columns)
    return X_train, y_train, meta_train, X_val, y_val, meta_val


def build_training_sets(
    train_df: pd.DataFrame,
    stride: int = 4,
    max_train_examples_per_horizon: int | None = None,
    include_score_features: bool = True,
    include_low_gain_filtered_features: bool = True,
    n_jobs: int = 1,
) -> FeatureBuildResult:
    n_regions = train_df["region_id"].nunique()
    per_region_budget = _per_region_budget(max_train_examples_per_horizon, n_regions)
    rows = {h: [] for h in HORIZONS}
    y = {h: [] for h in HORIZONS}
    meta = {h: [] for h in HORIZONS}

    worker_args = (
        (*payload, stride, per_region_budget, include_score_features, include_low_gain_filtered_features)
        for payload in _iter_region_arrays(train_df)
    )
    for region_rows, region_y, region_meta in _run_region_workers(_build_training_region, worker_args, n_jobs):
        _extend_horizon_dicts(rows, y, meta, region_rows, region_y, region_meta)

    return _to_result(
        rows,
        y,
        meta,
        max_train_examples_per_horizon,
        _feature_columns(include_score_features, include_low_gain_filtered_features),
    )


def build_training_set_for_horizon(
    train_df: pd.DataFrame,
    horizon: int,
    stride: int = 4,
    max_train_examples: int | None = None,
    include_score_features: bool = True,
    include_low_gain_filtered_features: bool = True,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build one horizon at a time to keep peak memory bounded."""

    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")

    n_regions = train_df["region_id"].nunique()
    per_region_budget = _per_region_budget(max_train_examples, n_regions)
    rows = []
    y = []
    meta = []

    worker_args = (
        (*payload, horizon, stride, per_region_budget, include_score_features, include_low_gain_filtered_features)
        for payload in _iter_region_arrays(train_df)
    )
    for region_rows, region_y, region_meta in _run_region_workers(
        _build_training_region_for_horizon,
        worker_args,
        n_jobs,
    ):
        _append_feature_chunk(rows, region_rows)
        y.extend(region_y)
        meta.extend(region_meta)

    X, yy, mm = _to_frame(
        rows,
        y,
        meta,
        max_train_examples,
        horizon,
        _feature_columns(include_score_features, include_low_gain_filtered_features),
    )
    return X, yy, mm


def build_test_features(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    include_score_features: bool = True,
    include_legacy_ablation_features: bool = False,
    include_raw_wind_features: bool = False,
    include_prec_w56_min: bool = False,
    include_raw_wb_tmp_features: bool = False,
    include_low_gain_filtered_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta = []
    history_by_region = _latest_score_history_by_region(train_df) if include_score_features and train_df is not None else {}
    seasonal_by_region = _latest_region_seasonal_context_by_region(train_df)

    for region_id, g in _iter_regions(test_df):
        values = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        cutoff = len(g)
        if cutoff < WINDOW_DAYS:
            raise ValueError(f"Region {region_id} has only {cutoff} test days; expected at least {WINDOW_DAYS}.")
        rows.append(
            compute_window_features(
                values,
                dates,
                cutoff,
                history_by_region.get(region_id),
                include_score_features,
                seasonal_context=seasonal_by_region.get(region_id),
                include_legacy_ablation_features=include_legacy_ablation_features,
                include_raw_wind_features=include_raw_wind_features,
                include_prec_w56_min=include_prec_w56_min,
                include_raw_wb_tmp_features=include_raw_wb_tmp_features,
                include_low_gain_filtered_features=include_low_gain_filtered_features,
            )
        )
        meta.append({"region_id": region_id})

    return pd.DataFrame(rows).fillna(0.0), pd.DataFrame(meta)


def compute_window_features(
    values: np.ndarray,
    dates: np.ndarray,
    cutoff: int,
    score_history: dict[str, float] | None = None,
    include_score_features: bool = True,
    seasonal_context: RegionSeasonalContext | None = None,
    seasonal_history_end: int | None = None,
    include_legacy_ablation_features: bool = False,
    include_raw_wind_features: bool = False,
    include_prec_w56_min: bool = False,
    include_raw_wb_tmp_features: bool = False,
    include_low_gain_filtered_features: bool = True,
) -> dict[str, float]:
    window = values[cutoff - WINDOW_DAYS : cutoff]
    if len(window) != WINDOW_DAYS:
        raise ValueError(f"Expected {WINDOW_DAYS} rows before cutoff {cutoff}, got {len(window)}")

    feats: dict[str, float] = {}
    for size in ROLLING_WINDOWS:
        sub = window[-size:]
        prefix = f"w{size}"
        means = np.nanmean(sub, axis=0)
        stds = np.nanstd(sub, axis=0)
        mins = np.nanmin(sub, axis=0)
        maxs = np.nanmax(sub, axis=0)
        q10, medians, q90 = np.nanpercentile(sub, PERCENTILES, axis=0)
        for i, col in enumerate(WEATHER_COLS):
            feats[f"{col}_{prefix}_mean"] = float(means[i])
            feats[f"{col}_{prefix}_std"] = float(stds[i])
            feats[f"{col}_{prefix}_min"] = float(mins[i])
            feats[f"{col}_{prefix}_max"] = float(maxs[i])
            feats[f"{col}_{prefix}_median"] = float(medians[i])
            feats[f"{col}_{prefix}_q10"] = float(q10[i])
            feats[f"{col}_{prefix}_q90"] = float(q90[i])

        prec = sub[:, PREC_IDX]
        feats[f"prec_{prefix}_sum"] = float(np.nansum(prec))
        feats[f"dry_days_{prefix}"] = float(np.sum(prec == 0))
        feats[f"rainy_days_{prefix}"] = float(np.sum(prec > 0))

    last = window[-1]
    last3 = np.nanmean(window[-3:], axis=0)
    last7 = np.nanmean(window[-7:], axis=0)
    prev7 = np.nanmean(window[-14:-7], axis=0)
    for i, col in enumerate(WEATHER_COLS):
        feats[f"{col}_last"] = float(last[i])
        feats[f"{col}_last3_mean"] = float(last3[i])
        feats[f"{col}_last7_mean"] = float(last7[i])
        feats[f"{col}_trend_7_vs_prev7"] = float(last7[i] - prev7[i])

    long_mean = np.nanmean(window, axis=0)
    for short in [7, 14, 28]:
        short_mean = np.nanmean(window[-short:], axis=0)
        for i, col in enumerate(WEATHER_COLS):
            feats[f"{col}_anom_w{short}_vs_w91"] = float(short_mean[i] - long_mean[i])

    prec91 = window[:, PREC_IDX]
    dry = prec91 == 0
    feats["max_consecutive_dry_days_w91"] = float(_max_consecutive_true(dry))
    feats["prec_recent7_vs_91_daily"] = float(np.nansum(prec91[-7:]) - np.nanmean(prec91) * 7)
    feats["prec_recent14_vs_91_daily"] = float(np.nansum(prec91[-14:]) - np.nanmean(prec91) * 14)

    tmp_max7 = np.nanmean(window[-7:, TMP_MAX_IDX])
    tmp_max28 = np.nanmean(window[-28:, TMP_MAX_IDX])
    prec7 = np.nansum(window[-7:, PREC_IDX])
    prec28 = np.nansum(window[-28:, PREC_IDX])
    humidity7 = np.nanmean(window[-7:, HUMIDITY_IDX])
    wind7 = np.nanmean(window[-7:, WIND_IDX])
    windmax7 = np.nanmean(window[-7:, WIND_MAX_IDX])
    surf_tmp7 = np.nanmean(window[-7:, SURF_TMP_IDX])
    feats["hot_dry_w7"] = float(tmp_max7 / (1.0 + prec7))
    feats["hot_dry_w28"] = float(tmp_max28 / (1.0 + prec28))
    feats["evap_risk_w7"] = float(tmp_max7 * (wind7 + windmax7) / (1.0 + humidity7))
    feats["surface_heat_low_humidity_w7"] = float(surf_tmp7 / (1.0 + humidity7))
    feats["dry_fraction_w91"] = float(np.mean(dry))

    feats.update(_weekly_bin_features(window, include_legacy_ablation_features))
    feats.update(
        _region_seasonal_anomaly_features(
            window,
            dates[cutoff - WINDOW_DAYS : cutoff],
            seasonal_context,
            seasonal_history_end,
            include_legacy_ablation_features,
        )
    )
    feats.update(_date_features(str(dates[cutoff - 1]), cutoff, include_legacy_ablation_features))
    if not include_raw_wind_features:
        feats = _without_raw_wind_features(feats)
    if not include_prec_w56_min:
        feats.pop("prec_w56_min", None)
    if not include_raw_wb_tmp_features:
        feats = _without_raw_weather_summaries(feats, ("wb_tmp_",))
    if not include_low_gain_filtered_features:
        feats = _without_low_gain_filtered_features(feats)
    if not include_score_features:
        return feats
    if score_history:
        feats.update(score_history)
    else:
        feats.update(_empty_score_history())
    return feats


def _without_raw_wind_features(feats: dict[str, float]) -> dict[str, float]:
    wind_prefixes = ("wind_", "wind_max_", "wind_min_", "wind_range_")
    return _without_raw_weather_summaries(feats, wind_prefixes)


def _without_low_gain_filtered_features(feats: dict[str, float]) -> dict[str, float]:
    return {feature: value for feature, value in feats.items() if feature not in LOW_GAIN_FILTERED_FEATURES}


def _without_raw_weather_summaries(
    feats: dict[str, float],
    weather_prefixes: tuple[str, ...],
) -> dict[str, float]:
    keep_markers = ("_region_seasonal_anom_", "_weekly_trend_13w")
    return {
        feature: value
        for feature, value in feats.items()
        if not feature.startswith(weather_prefixes)
        or "_week" in feature
        or any(marker in feature for marker in keep_markers)
    }


def align_feature_columns(X: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    return X.reindex(columns=list(feature_columns), fill_value=0.0)


def _to_result(rows, y, meta, max_examples: int | None, feature_columns: list[str]) -> FeatureBuildResult:
    X_by_horizon = {}
    y_by_horizon = {}
    meta_by_horizon = {}
    for h in HORIZONS:
        X, yy, mm = _to_frame(rows[h], y[h], meta[h], max_examples, h, feature_columns)
        X_by_horizon[h] = X
        y_by_horizon[h] = yy
        meta_by_horizon[h] = mm
    return FeatureBuildResult(X_by_horizon, y_by_horizon, meta_by_horizon)


def _to_frame(
    rows,
    y,
    meta,
    max_examples: int | None,
    horizon: int,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    matrix = np.concatenate(rows, axis=0) if rows else np.empty((0, len(feature_columns)), dtype=np.float32)
    X = pd.DataFrame(matrix, columns=feature_columns).fillna(0.0)
    yy = np.asarray(y, dtype=np.float32)
    mm = pd.DataFrame(meta)
    if max_examples and len(X) > max_examples:
        rng = np.random.default_rng(42 + horizon)
        keep = np.sort(rng.choice(len(X), size=max_examples, replace=False))
        X = X.iloc[keep].reset_index(drop=True)
        yy = yy[keep]
        mm = mm.iloc[keep].reset_index(drop=True)
    LOGGER.info("Horizon %d: built %d examples with %d features", horizon, len(X), X.shape[1])
    return X, yy, mm


def _per_region_budget(max_examples: int | None, n_regions: int) -> int | None:
    if not max_examples or n_regions <= 0:
        return None
    return max(1, int(np.ceil(max_examples / n_regions)))


def _preselect_candidates(candidates: np.ndarray, per_region_budget: int | None) -> np.ndarray:
    """Select candidate target rows before expensive feature computation.

    This keeps the sample spread through time instead of taking only the
    earliest windows for each region.
    """

    if not per_region_budget or len(candidates) <= per_region_budget:
        return candidates
    keep = np.linspace(0, len(candidates) - 1, per_region_budget).round().astype(int)
    return candidates[np.unique(keep)]


def _run_region_workers(worker_fn, worker_args, n_jobs: int):
    workers = _normalize_n_jobs(n_jobs)
    if workers == 1:
        for args in worker_args:
            yield worker_fn(args)
        return

    LOGGER.info("Building region features with %d worker processes", workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from _ordered_bounded_map(executor, worker_fn, worker_args, max_pending=workers)


def _normalize_n_jobs(n_jobs: int) -> int:
    if n_jobs <= 0:
        return max(1, min(4, os.cpu_count() or 1))
    return max(1, n_jobs)


def _ordered_bounded_map(executor: ProcessPoolExecutor, worker_fn, worker_args, max_pending: int):
    args_iter = iter(worker_args)
    pending = {}
    completed = {}
    next_submit = 0
    next_yield = 0

    def submit_one() -> bool:
        nonlocal next_submit
        try:
            args = next(args_iter)
        except StopIteration:
            return False
        pending[executor.submit(worker_fn, args)] = next_submit
        next_submit += 1
        return True

    for _ in range(max_pending):
        if not submit_one():
            break

    while pending:
        done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            idx = pending.pop(future)
            completed[idx] = future.result()
            submit_one()

        while next_yield in completed:
            yield completed.pop(next_yield)
            next_yield += 1


def _extend_horizon_dicts(rows, y, meta, region_rows, region_y, region_meta) -> None:
    for h in HORIZONS:
        _append_feature_chunk(rows[h], region_rows[h])
        y[h].extend(region_y[h])
        meta[h].extend(region_meta[h])


def _append_feature_chunk(chunks: list[np.ndarray], chunk: np.ndarray) -> None:
    if len(chunk):
        chunks.append(chunk)


def _empty_horizon_lists() -> tuple[dict[int, list], dict[int, list], dict[int, list]]:
    return {h: [] for h in HORIZONS}, {h: [] for h in HORIZONS}, {h: [] for h in HORIZONS}


def _iter_regions(df: pd.DataFrame):
    for region_id, g in df.groupby("region_id", sort=False):
        yield str(region_id), g.reset_index(drop=True)


def _iter_region_arrays(df: pd.DataFrame):
    for region_id, g in _iter_regions(df):
        yield (
            region_id,
            g[WEATHER_COLS].to_numpy(dtype=np.float32),
            g["date"].to_numpy(),
            g["score"].to_numpy(dtype=np.float32),
        )


def _build_validation_region(args):
    (
        region_id,
        values,
        dates,
        scores,
        val_weeks,
        stride,
        per_region_budget,
        include_score_features,
        include_low_gain_filtered_features,
    ) = args
    train_rows, train_y, train_meta = _empty_horizon_lists()
    val_rows, val_y, val_meta = _empty_horizon_lists()

    label_idx = np.flatnonzero(~np.isnan(scores))
    if len(label_idx) < val_weeks + max(HORIZONS) + 1:
        return {
            "train": (_pack_feature_rows_by_horizon(train_rows), train_y, train_meta),
            "val": (_pack_feature_rows_by_horizon(val_rows), val_y, val_meta),
        }

    first_val_target = int(label_idx[-val_weeks])
    val_cutoff = first_val_target - 7
    if val_cutoff < WINDOW_DAYS:
        return {
            "train": (_pack_feature_rows_by_horizon(train_rows), train_y, train_meta),
            "val": (_pack_feature_rows_by_horizon(val_rows), val_y, val_meta),
        }

    get_features = _make_region_feature_getter(
        values,
        dates,
        scores,
        include_score_features,
        include_low_gain_filtered_features,
    )

    for h in HORIZONS:
        target_idx = val_cutoff + 7 * h
        if target_idx < len(scores) and not np.isnan(scores[target_idx]):
            val_rows[h].append(get_features(val_cutoff))
            val_y[h].append(float(scores[target_idx]))
            val_meta[h].append({"region_id": region_id, "cutoff_idx": val_cutoff, "target_idx": target_idx})

    cutoff_candidates = label_idx[label_idx < first_val_target]
    cutoff_candidates = cutoff_candidates[cutoff_candidates >= WINDOW_DAYS + 7 * max(HORIZONS)]
    cutoff_candidates = _preselect_candidates(cutoff_candidates[:: max(1, stride)], per_region_budget)
    for target_idx in cutoff_candidates:
        for h in HORIZONS:
            cutoff = int(target_idx - 7 * h)
            if cutoff < WINDOW_DAYS:
                continue
            train_rows[h].append(get_features(cutoff))
            train_y[h].append(float(scores[target_idx]))
            train_meta[h].append({"region_id": region_id, "cutoff_idx": cutoff, "target_idx": int(target_idx)})

    return {
        "train": (_pack_feature_rows_by_horizon(train_rows), train_y, train_meta),
        "val": (_pack_feature_rows_by_horizon(val_rows), val_y, val_meta),
    }


def _build_validation_region_for_horizon(args):
    (
        region_id,
        values,
        dates,
        scores,
        horizon,
        val_weeks,
        stride,
        per_region_budget,
        include_score_features,
        include_low_gain_filtered_features,
    ) = args
    train_rows = []
    train_y = []
    train_meta = []
    val_rows = []
    val_y = []
    val_meta = []

    label_idx = np.flatnonzero(~np.isnan(scores))
    if len(label_idx) < val_weeks + max(HORIZONS) + 1:
        return _pack_feature_rows(train_rows), train_y, train_meta, _pack_feature_rows(val_rows), val_y, val_meta

    first_val_target = int(label_idx[-val_weeks])
    val_cutoff = first_val_target - 7
    if val_cutoff < WINDOW_DAYS:
        return _pack_feature_rows(train_rows), train_y, train_meta, _pack_feature_rows(val_rows), val_y, val_meta

    get_features = _make_region_feature_getter(
        values,
        dates,
        scores,
        include_score_features,
        include_low_gain_filtered_features,
    )

    target_idx = val_cutoff + 7 * horizon
    if target_idx < len(scores) and not np.isnan(scores[target_idx]):
        val_rows.append(get_features(val_cutoff))
        val_y.append(float(scores[target_idx]))
        val_meta.append({"region_id": region_id, "cutoff_idx": val_cutoff, "target_idx": target_idx})

    cutoff_candidates = label_idx[label_idx < first_val_target]
    cutoff_candidates = cutoff_candidates[cutoff_candidates >= WINDOW_DAYS + 7 * max(HORIZONS)]
    cutoff_candidates = _preselect_candidates(cutoff_candidates[:: max(1, stride)], per_region_budget)
    for target_idx in cutoff_candidates:
        cutoff = int(target_idx - 7 * horizon)
        if cutoff < WINDOW_DAYS:
            continue
        train_rows.append(get_features(cutoff))
        train_y.append(float(scores[target_idx]))
        train_meta.append({"region_id": region_id, "cutoff_idx": cutoff, "target_idx": int(target_idx)})

    return _pack_feature_rows(train_rows), train_y, train_meta, _pack_feature_rows(val_rows), val_y, val_meta


def _build_training_region(args):
    (
        region_id,
        values,
        dates,
        scores,
        stride,
        per_region_budget,
        include_score_features,
        include_low_gain_filtered_features,
    ) = args
    rows, y, meta = _empty_horizon_lists()
    label_idx = np.flatnonzero(~np.isnan(scores))
    label_idx = label_idx[label_idx >= WINDOW_DAYS + 7 * max(HORIZONS)]
    label_idx = _preselect_candidates(label_idx[:: max(1, stride)], per_region_budget)
    get_features = _make_region_feature_getter(
        values,
        dates,
        scores,
        include_score_features,
        include_low_gain_filtered_features,
    )

    for target_idx in label_idx:
        for h in HORIZONS:
            cutoff = int(target_idx - 7 * h)
            if cutoff < WINDOW_DAYS:
                continue
            rows[h].append(get_features(cutoff))
            y[h].append(float(scores[target_idx]))
            meta[h].append({"region_id": region_id, "cutoff_idx": cutoff, "target_idx": int(target_idx)})

    return _pack_feature_rows_by_horizon(rows), y, meta


def _build_training_region_for_horizon(args):
    (
        region_id,
        values,
        dates,
        scores,
        horizon,
        stride,
        per_region_budget,
        include_score_features,
        include_low_gain_filtered_features,
    ) = args
    rows = []
    y = []
    meta = []
    label_idx = np.flatnonzero(~np.isnan(scores))
    label_idx = label_idx[label_idx >= WINDOW_DAYS + 7 * max(HORIZONS)]
    label_idx = _preselect_candidates(label_idx[:: max(1, stride)], per_region_budget)
    get_features = _make_region_feature_getter(
        values,
        dates,
        scores,
        include_score_features,
        include_low_gain_filtered_features,
    )

    for target_idx in label_idx:
        cutoff = int(target_idx - 7 * horizon)
        if cutoff < WINDOW_DAYS:
            continue
        rows.append(get_features(cutoff))
        y.append(float(scores[target_idx]))
        meta.append({"region_id": region_id, "cutoff_idx": cutoff, "target_idx": int(target_idx)})

    return _pack_feature_rows(rows), y, meta


def _make_region_feature_getter(
    values: np.ndarray,
    dates: np.ndarray,
    scores: np.ndarray,
    include_score_features: bool,
    include_low_gain_filtered_features: bool = True,
):
    cache: dict[int, dict[str, float]] = {}
    seasonal_context = _expanding_region_seasonal_context(values, dates)

    def get_features(cutoff: int) -> dict[str, float]:
        if cutoff not in cache:
            hist = _score_history_for_window(scores, cutoff, include_score_features)
            cache[cutoff] = compute_window_features(
                values,
                dates,
                cutoff,
                hist,
                include_score_features,
                seasonal_context=seasonal_context,
                seasonal_history_end=max(0, cutoff - WINDOW_DAYS),
                include_low_gain_filtered_features=include_low_gain_filtered_features,
            )
        return cache[cutoff]

    return get_features


def _pack_feature_rows(rows: list[dict[str, float]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray([list(row.values()) for row in rows], dtype=np.float32)


def _pack_feature_rows_by_horizon(rows: dict[int, list[dict[str, float]]]) -> dict[int, np.ndarray]:
    return {h: _pack_feature_rows(rows[h]) for h in HORIZONS}


@lru_cache(maxsize=4)
def _feature_columns(include_score_features: bool, include_low_gain_filtered_features: bool = True) -> list[str]:
    values = np.zeros((WINDOW_DAYS, len(WEATHER_COLS)), dtype=np.float32)
    dates = np.full(WINDOW_DAYS, "2000-01-01")
    return list(
        compute_window_features(
            values,
            dates,
            WINDOW_DAYS,
            include_score_features=include_score_features,
            include_low_gain_filtered_features=include_low_gain_filtered_features,
        )
    )


def _score_history_features(scores: np.ndarray, cutoff: int) -> dict[str, float]:
    past = scores[:cutoff]
    past = past[~np.isnan(past)]
    if len(past) == 0:
        return _empty_score_history()
    feats = {
        "score_last": float(past[-1]),
        "score_mean_all": float(np.mean(past)),
        "score_median_all": float(np.median(past)),
        "score_severe_freq_all": float(np.mean(past >= 3)),
    }
    for n in [2, 4, 8]:
        tail = past[-n:]
        feats[f"score_mean_last{n}"] = float(np.mean(tail))
    feats["score_trend_last_vs_last8"] = float(feats["score_last"] - feats["score_mean_last8"])
    return feats


def _score_history_for_window(scores: np.ndarray, cutoff: int, include_score_features: bool) -> dict[str, float] | None:
    if not include_score_features:
        return None
    return _score_history_features(scores, max(0, cutoff - WINDOW_DAYS))


def _latest_score_history_by_region(train_df: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    if train_df is None:
        return {}
    out = {}
    for region_id, g in _iter_regions(train_df):
        scores = g["score"].to_numpy(dtype=np.float32)
        out[region_id] = _score_history_features(scores, len(scores))
    return out


def _latest_region_seasonal_context_by_region(
    train_df: pd.DataFrame | None,
) -> dict[str, RegionSeasonalContext]:
    if train_df is None:
        return {}
    out = {}
    for region_id, g in _iter_regions(train_df):
        values = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        out[region_id] = _region_seasonal_summary(values, dates)
    return out


def _weekly_bin_features(window: np.ndarray, include_legacy_ablation_features: bool = False) -> dict[str, float]:
    weekly = window.reshape(WEEKLY_BIN_COUNT, WEEKLY_BIN_DAYS, len(WEATHER_COLS))
    means = np.nanmean(weekly, axis=1)
    prec = weekly[:, :, PREC_IDX]
    prec_sums = np.nansum(prec, axis=1)
    dry_days = np.sum(prec == 0, axis=1)
    feats = {}
    # Week 13 duplicates the existing 7-day summaries.
    bin_count = WEEKLY_BIN_COUNT if include_legacy_ablation_features else WEEKLY_BIN_COUNT - 1
    for week_idx in range(bin_count):
        week = week_idx + 1
        for col_idx, col in enumerate(WEATHER_COLS):
            feats[f"{col}_week{week:02d}_mean"] = float(means[week_idx, col_idx])
        feats[f"prec_week{week:02d}_sum"] = float(prec_sums[week_idx])
        feats[f"dry_days_week{week:02d}"] = float(dry_days[week_idx])

    for col_idx, col in enumerate(WEATHER_COLS):
        feats[f"{col}_weekly_trend_13w"] = _linear_trend(means[:, col_idx])
    feats["prec_weekly_sum_trend_13w"] = _linear_trend(prec_sums)
    feats["dry_days_weekly_trend_13w"] = _linear_trend(dry_days)
    return feats


def _linear_trend(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=np.float32)
    mask = ~np.isnan(values)
    if np.sum(mask) < 2:
        return 0.0
    centered = x[mask] - np.mean(x[mask])
    denominator = np.sum(centered * centered)
    if denominator == 0:
        return 0.0
    return float(np.sum(centered * (values[mask] - np.mean(values[mask]))) / denominator)


def _region_seasonal_summary(values: np.ndarray, dates: np.ndarray) -> RegionSeasonalContext:
    months = _months_from_dates(dates)
    sums = np.zeros((12, len(WEATHER_COLS)), dtype=np.float64)
    counts = np.zeros((12, len(WEATHER_COLS)), dtype=np.int32)
    for month in range(1, 13):
        month_values = values[months == month]
        if len(month_values) == 0:
            continue
        sums[month - 1] = np.nansum(month_values, axis=0)
        counts[month - 1] = np.sum(~np.isnan(month_values), axis=0)
    return RegionSeasonalContext(sums=sums, counts=counts)


def _expanding_region_seasonal_context(values: np.ndarray, dates: np.ndarray) -> RegionSeasonalContext:
    months = _months_from_dates(dates)
    monthly_sums = []
    monthly_counts = []
    monthly_indices = []
    for month in range(1, 13):
        indices = np.flatnonzero(months == month)
        month_values = values[indices]
        sums = np.zeros((len(indices) + 1, len(WEATHER_COLS)), dtype=np.float32)
        counts = np.zeros((len(indices) + 1, len(WEATHER_COLS)), dtype=np.int16)
        sums[1:] = np.cumsum(np.nan_to_num(month_values, nan=0.0), axis=0)
        counts[1:] = np.cumsum(~np.isnan(month_values), axis=0)
        monthly_sums.append(sums)
        monthly_counts.append(counts)
        monthly_indices.append(indices)
    return RegionSeasonalContext(
        sums=tuple(monthly_sums),
        counts=tuple(monthly_counts),
        row_indices=tuple(monthly_indices),
    )


def _region_seasonal_anomaly_features(
    window: np.ndarray,
    window_dates: np.ndarray,
    context: RegionSeasonalContext | None,
    history_end: int | None,
    include_legacy_ablation_features: bool = False,
) -> dict[str, float]:
    feats = {}
    if context is None:
        for size in REGION_SEASONAL_WINDOWS:
            for col in WEATHER_COLS:
                feats[f"{col}_region_seasonal_anom_w{size}"] = 0.0
            if include_legacy_ablation_features:
                feats[f"region_seasonal_coverage_w{size}"] = 0.0
        return feats

    sums, counts = _seasonal_snapshot(context, history_end)
    overall_sum = np.sum(sums, axis=0)
    overall_count = np.sum(counts, axis=0)
    overall_mean = np.divide(overall_sum, overall_count, out=np.zeros_like(overall_sum), where=overall_count > 0)
    months = _months_from_dates(window_dates)

    for size in REGION_SEASONAL_WINDOWS:
        sub = window[-size:]
        sub_mean = np.nanmean(sub, axis=0)
        month_weights = np.bincount(months[-size:] - 1, minlength=12).astype(np.float64)
        month_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        month_means = np.where(counts > 0, month_means, overall_mean[None, :])
        baseline = np.sum(month_means * month_weights[:, None], axis=0) / size
        baseline = np.where(overall_count > 0, baseline, sub_mean)
        for col_idx, col in enumerate(WEATHER_COLS):
            feats[f"{col}_region_seasonal_anom_w{size}"] = float(sub_mean[col_idx] - baseline[col_idx])
        if include_legacy_ablation_features:
            covered_days = np.sum(month_weights[:, None] * (counts > 0), axis=0)
            feats[f"region_seasonal_coverage_w{size}"] = float(np.mean(covered_days / size))
    return feats


def _seasonal_snapshot(
    context: RegionSeasonalContext,
    history_end: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if context.row_indices is None:
        return context.sums, context.counts
    end = np.iinfo(np.int64).max if history_end is None else max(0, history_end)
    sums = np.zeros((12, len(WEATHER_COLS)), dtype=np.float64)
    counts = np.zeros((12, len(WEATHER_COLS)), dtype=np.int32)
    for month in range(12):
        prefix_idx = int(np.searchsorted(context.row_indices[month], end, side="left"))
        sums[month] = context.sums[month][prefix_idx]
        counts[month] = context.counts[month][prefix_idx]
    return sums, counts


def _months_from_dates(dates: np.ndarray) -> np.ndarray:
    return np.array([int(str(date).split("-")[1]) for date in dates], dtype=np.int8)


def _empty_score_history() -> dict[str, float]:
    return {
        "score_last": 0.0,
        "score_mean_all": 0.0,
        "score_median_all": 0.0,
        "score_severe_freq_all": 0.0,
        "score_mean_last2": 0.0,
        "score_mean_last4": 0.0,
        "score_mean_last8": 0.0,
        "score_trend_last_vs_last8": 0.0,
    }


def _date_features(date_text: str, cutoff: int, include_legacy_ablation_features: bool = False) -> dict[str, float]:
    parts = date_text.split("-")
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1
    doy = int(MONTH_START_DOY[month] + day)
    angle = 2.0 * np.pi * doy / 365.0
    feats = {
        "month": float(month),
        "day_of_year": float(doy),
        "week_of_year": float((doy - 1) // 7 + 1),
        "doy_sin": float(np.sin(angle)),
        "doy_cos": float(np.cos(angle)),
    }
    if include_legacy_ablation_features:
        feats["day_index_mod_365"] = float(cutoff % 365)
    return feats


def _max_consecutive_true(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best
