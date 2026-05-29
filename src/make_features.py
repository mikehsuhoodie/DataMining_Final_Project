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
        (*payload, val_weeks, stride, per_region_budget, include_score_features)
        for payload in _iter_region_arrays(train_df)
    )
    for region_result in _run_region_workers(_build_validation_region, worker_args, n_jobs):
        _extend_horizon_dicts(train_rows, train_y, train_meta, *region_result["train"])
        _extend_horizon_dicts(val_rows, val_y, val_meta, *region_result["val"])

    train_result = _to_result(train_rows, train_y, train_meta, max_train_examples_per_horizon)
    val_result = _to_result(val_rows, val_y, val_meta, None)
    return train_result, val_result


def build_validation_set_for_horizon(
    train_df: pd.DataFrame,
    horizon: int,
    val_weeks: int = 5,
    stride: int = 4,
    max_train_examples: int | None = None,
    include_score_features: bool = True,
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
        (*payload, horizon, val_weeks, stride, per_region_budget, include_score_features)
        for payload in _iter_region_arrays(train_df)
    )
    for region_train_rows, region_train_y, region_train_meta, region_val_rows, region_val_y, region_val_meta in (
        _run_region_workers(_build_validation_region_for_horizon, worker_args, n_jobs)
    ):
        train_rows.extend(region_train_rows)
        train_y.extend(region_train_y)
        train_meta.extend(region_train_meta)
        val_rows.extend(region_val_rows)
        val_y.extend(region_val_y)
        val_meta.extend(region_val_meta)

    X_train, y_train, meta_train = _to_frame(train_rows, train_y, train_meta, max_train_examples, horizon)
    X_val, y_val, meta_val = _to_frame(val_rows, val_y, val_meta, None, horizon)
    return X_train, y_train, meta_train, X_val, y_val, meta_val


def build_training_sets(
    train_df: pd.DataFrame,
    stride: int = 4,
    max_train_examples_per_horizon: int | None = None,
    include_score_features: bool = True,
    n_jobs: int = 1,
) -> FeatureBuildResult:
    n_regions = train_df["region_id"].nunique()
    per_region_budget = _per_region_budget(max_train_examples_per_horizon, n_regions)
    rows = {h: [] for h in HORIZONS}
    y = {h: [] for h in HORIZONS}
    meta = {h: [] for h in HORIZONS}

    worker_args = (
        (*payload, stride, per_region_budget, include_score_features)
        for payload in _iter_region_arrays(train_df)
    )
    for region_rows, region_y, region_meta in _run_region_workers(_build_training_region, worker_args, n_jobs):
        _extend_horizon_dicts(rows, y, meta, region_rows, region_y, region_meta)

    return _to_result(rows, y, meta, max_train_examples_per_horizon)


def build_training_set_for_horizon(
    train_df: pd.DataFrame,
    horizon: int,
    stride: int = 4,
    max_train_examples: int | None = None,
    include_score_features: bool = True,
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
        (*payload, horizon, stride, per_region_budget, include_score_features)
        for payload in _iter_region_arrays(train_df)
    )
    for region_rows, region_y, region_meta in _run_region_workers(
        _build_training_region_for_horizon,
        worker_args,
        n_jobs,
    ):
        rows.extend(region_rows)
        y.extend(region_y)
        meta.extend(region_meta)

    X, yy, mm = _to_frame(rows, y, meta, max_train_examples, horizon)
    return X, yy, mm


def build_test_features(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    include_score_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta = []
    history_by_region = _latest_score_history_by_region(train_df) if include_score_features and train_df is not None else {}

    for region_id, g in _iter_regions(test_df):
        values = g[WEATHER_COLS].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        cutoff = len(g)
        if cutoff < WINDOW_DAYS:
            raise ValueError(f"Region {region_id} has only {cutoff} test days; expected at least {WINDOW_DAYS}.")
        rows.append(compute_window_features(values, dates, cutoff, history_by_region.get(region_id), include_score_features))
        meta.append({"region_id": region_id})

    return pd.DataFrame(rows).fillna(0.0), pd.DataFrame(meta)


def compute_window_features(
    values: np.ndarray,
    dates: np.ndarray,
    cutoff: int,
    score_history: dict[str, float] | None = None,
    include_score_features: bool = True,
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

    feats.update(_date_features(str(dates[cutoff - 1]), cutoff))
    if not include_score_features:
        return feats
    if score_history:
        feats.update(score_history)
    else:
        feats.update(_empty_score_history())
    return feats


def align_feature_columns(X: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    return X.reindex(columns=list(feature_columns), fill_value=0.0)


def _to_result(rows, y, meta, max_examples: int | None) -> FeatureBuildResult:
    X_by_horizon = {}
    y_by_horizon = {}
    meta_by_horizon = {}
    for h in HORIZONS:
        X, yy, mm = _to_frame(rows[h], y[h], meta[h], max_examples, h)
        X_by_horizon[h] = X
        y_by_horizon[h] = yy
        meta_by_horizon[h] = mm
    return FeatureBuildResult(X_by_horizon, y_by_horizon, meta_by_horizon)


def _to_frame(rows, y, meta, max_examples: int | None, horizon: int) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    X = pd.DataFrame(rows).fillna(0.0)
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
        rows[h].extend(region_rows[h])
        y[h].extend(region_y[h])
        meta[h].extend(region_meta[h])


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
    region_id, values, dates, scores, val_weeks, stride, per_region_budget, include_score_features = args
    train_rows, train_y, train_meta = _empty_horizon_lists()
    val_rows, val_y, val_meta = _empty_horizon_lists()

    label_idx = np.flatnonzero(~np.isnan(scores))
    if len(label_idx) < val_weeks + max(HORIZONS) + 1:
        return {"train": (train_rows, train_y, train_meta), "val": (val_rows, val_y, val_meta)}

    first_val_target = int(label_idx[-val_weeks])
    val_cutoff = first_val_target - 7
    if val_cutoff < WINDOW_DAYS:
        return {"train": (train_rows, train_y, train_meta), "val": (val_rows, val_y, val_meta)}

    get_features = _make_region_feature_getter(values, dates, scores, include_score_features)

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

    return {"train": (train_rows, train_y, train_meta), "val": (val_rows, val_y, val_meta)}


def _build_validation_region_for_horizon(args):
    region_id, values, dates, scores, horizon, val_weeks, stride, per_region_budget, include_score_features = args
    train_rows = []
    train_y = []
    train_meta = []
    val_rows = []
    val_y = []
    val_meta = []

    label_idx = np.flatnonzero(~np.isnan(scores))
    if len(label_idx) < val_weeks + max(HORIZONS) + 1:
        return train_rows, train_y, train_meta, val_rows, val_y, val_meta

    first_val_target = int(label_idx[-val_weeks])
    val_cutoff = first_val_target - 7
    if val_cutoff < WINDOW_DAYS:
        return train_rows, train_y, train_meta, val_rows, val_y, val_meta

    get_features = _make_region_feature_getter(values, dates, scores, include_score_features)

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

    return train_rows, train_y, train_meta, val_rows, val_y, val_meta


def _build_training_region(args):
    region_id, values, dates, scores, stride, per_region_budget, include_score_features = args
    rows, y, meta = _empty_horizon_lists()
    label_idx = np.flatnonzero(~np.isnan(scores))
    label_idx = label_idx[label_idx >= WINDOW_DAYS + 7 * max(HORIZONS)]
    label_idx = _preselect_candidates(label_idx[:: max(1, stride)], per_region_budget)
    get_features = _make_region_feature_getter(values, dates, scores, include_score_features)

    for target_idx in label_idx:
        for h in HORIZONS:
            cutoff = int(target_idx - 7 * h)
            if cutoff < WINDOW_DAYS:
                continue
            rows[h].append(get_features(cutoff))
            y[h].append(float(scores[target_idx]))
            meta[h].append({"region_id": region_id, "cutoff_idx": cutoff, "target_idx": int(target_idx)})

    return rows, y, meta


def _build_training_region_for_horizon(args):
    region_id, values, dates, scores, horizon, stride, per_region_budget, include_score_features = args
    rows = []
    y = []
    meta = []
    label_idx = np.flatnonzero(~np.isnan(scores))
    label_idx = label_idx[label_idx >= WINDOW_DAYS + 7 * max(HORIZONS)]
    label_idx = _preselect_candidates(label_idx[:: max(1, stride)], per_region_budget)
    get_features = _make_region_feature_getter(values, dates, scores, include_score_features)

    for target_idx in label_idx:
        cutoff = int(target_idx - 7 * horizon)
        if cutoff < WINDOW_DAYS:
            continue
        rows.append(get_features(cutoff))
        y.append(float(scores[target_idx]))
        meta.append({"region_id": region_id, "cutoff_idx": cutoff, "target_idx": int(target_idx)})

    return rows, y, meta


def _make_region_feature_getter(
    values: np.ndarray,
    dates: np.ndarray,
    scores: np.ndarray,
    include_score_features: bool,
):
    cache: dict[int, dict[str, float]] = {}

    def get_features(cutoff: int) -> dict[str, float]:
        if cutoff not in cache:
            hist = _score_history_for_window(scores, cutoff, include_score_features)
            cache[cutoff] = compute_window_features(values, dates, cutoff, hist, include_score_features)
        return cache[cutoff]

    return get_features


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


def _date_features(date_text: str, cutoff: int) -> dict[str, float]:
    parts = date_text.split("-")
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1
    doy = int(MONTH_START_DOY[month] + day)
    angle = 2.0 * np.pi * doy / 365.0
    return {
        "month": float(month),
        "day_of_year": float(doy),
        "week_of_year": float((doy - 1) // 7 + 1),
        "day_index_mod_365": float(cutoff % 365),
        "doy_sin": float(np.sin(angle)),
        "doy_cos": float(np.cos(angle)),
    }


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
