from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dco_qwen3_visualize.config import DCOQwen3VisualizeConfig
from dco_qwen3_visualize.progress import format_duration, progress_snapshot

LOGGER = logging.getLogger(__name__)

SAMPLING_COLUMNS = [
    "row_id",
    "source_uri",
    "source_row_number",
    "customer",
    "sales_date",
    "origin",
    "destination",
    "origin_metro",
    "destination_metro",
    "trip_type",
    "cabin",
    "stops",
    "source",
    "carrier",
    "price_inc",
    "advance_purchase",
]

STRING_COLUMNS = [
    "row_id",
    "source_uri",
    "customer",
    "sales_date",
    "origin",
    "destination",
    "origin_metro",
    "destination_metro",
    "trip_type",
    "cabin",
    "stops",
    "source",
    "carrier",
]


def _normalize_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("missing")


def _quantile_codes(series: pd.Series, bins: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(["missing"] * len(series), index=series.index, dtype="string")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(valid.to_numpy(dtype=np.float64), quantiles))
    if len(edges) <= 1:
        labels = pd.Series(["q01"] * len(series), index=series.index, dtype="string")
        labels[numeric.isna()] = "missing"
        return labels
    codes = np.searchsorted(edges[1:-1], numeric.to_numpy(dtype=np.float64, na_value=np.nan), side="right")
    labels = pd.Series([f"q{code + 1:02d}" for code in codes], index=series.index, dtype="string")
    labels[numeric.isna()] = "missing"
    return labels


def _allocate_capped_quotas(
    counts: pd.Series,
    total: int,
    *,
    guarantee_all: bool,
) -> pd.Series:
    counts = counts.astype(int).sort_index()
    quotas = pd.Series(0, index=counts.index, dtype=np.int64)
    if total <= 0 or counts.empty:
        return quotas

    if guarantee_all and total >= len(counts):
        quotas += np.minimum(counts.to_numpy(), 1)
        remaining = int(total - quotas.sum())
    else:
        remaining = int(total)

    while remaining > 0:
        capacity = counts - quotas
        available = capacity[capacity > 0]
        if available.empty:
            break
        weights = np.sqrt(available.to_numpy(dtype=np.float64))
        if weights.sum() <= 0:
            order = available.sort_values(ascending=False).index.tolist()
            for key in order[:remaining]:
                quotas.loc[key] += 1
            break

        exact = weights / weights.sum() * remaining
        floor = np.floor(exact).astype(np.int64)
        floor = np.minimum(floor, available.to_numpy(dtype=np.int64))
        if floor.sum() > 0:
            quotas.loc[available.index] += floor
            remaining = int(total - quotas.sum())
            continue

        fractions = exact - floor
        tie_break = pd.DataFrame(
            {
                "key": available.index.astype(str),
                "fraction": fractions,
                "count": available.to_numpy(dtype=np.int64),
            },
            index=available.index,
        ).sort_values(["fraction", "count", "key"], ascending=[False, False, True])
        for key in tie_break.index[:remaining]:
            quotas.loc[key] += 1
        remaining = int(total - quotas.sum())

    return quotas


def _systematic_positions(count: int, quota: int) -> np.ndarray:
    if quota <= 0 or count <= 0:
        return np.empty((0,), dtype=np.int64)
    if quota >= count:
        return np.arange(count, dtype=np.int64)

    step = count / quota
    seeds = np.floor((np.arange(quota) + 0.5) * step).astype(np.int64)
    used: set[int] = set()
    selected: list[int] = []
    for seed in seeds:
        candidate = int(np.clip(seed, 0, count - 1))
        while candidate in used and candidate < count - 1:
            candidate += 1
        while candidate in used and candidate > 0:
            candidate -= 1
        if candidate in used:
            continue
        used.add(candidate)
        selected.append(candidate)
    if len(selected) < quota:
        for candidate in range(count):
            if candidate not in used:
                selected.append(candidate)
            if len(selected) == quota:
                break
    return np.asarray(sorted(selected), dtype=np.int64)


def _low_fare_first_positions(count: int, quota: int) -> np.ndarray:
    if quota <= 0 or count <= 0:
        return np.empty((0,), dtype=np.int64)
    if quota >= count:
        return np.arange(count, dtype=np.int64)
    if quota == 1:
        return np.asarray([0], dtype=np.int64)

    tail_positions = _systematic_positions(count - 1, quota - 1) + 1
    return np.asarray(sorted({0, *tail_positions.tolist()}), dtype=np.int64)


def _prepare_sampling_frame(parquet_path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(parquet_path, columns=SAMPLING_COLUMNS)
    for column in STRING_COLUMNS:
        frame[column] = _normalize_string(frame[column])

    frame["price_inc"] = pd.to_numeric(frame["price_inc"], errors="coerce")
    frame["advance_purchase"] = pd.to_numeric(frame["advance_purchase"], errors="coerce")
    frame["metro_od"] = frame["origin_metro"] + "->" + frame["destination_metro"]
    frame["airport_od"] = frame["origin"] + "->" + frame["destination"]
    frame["secondary_key"] = (
        frame["trip_type"] + "|" + frame["cabin"] + "|" + _normalize_string(frame["stops"])
    )
    frame["price_bin"] = _quantile_codes(frame["price_inc"], 10)
    frame["ap_bin"] = _quantile_codes(frame["advance_purchase"], 10)
    frame["row_hash"] = pd.util.hash_pandas_object(frame["row_id"], index=False).astype(np.uint64)

    airport_counts = frame["airport_od"].value_counts(dropna=False)
    source_counts = frame["source"].value_counts(dropna=False)
    carrier_counts = frame["carrier"].value_counts(dropna=False)
    frame["airport_count"] = frame["airport_od"].map(airport_counts).astype(np.int32)
    frame["source_count"] = frame["source"].map(source_counts).astype(np.int32)
    frame["carrier_count"] = frame["carrier"].map(carrier_counts).astype(np.int32)

    for column in [
        "origin",
        "destination",
        "origin_metro",
        "destination_metro",
        "trip_type",
        "cabin",
        "stops",
        "source",
        "carrier",
        "metro_od",
        "airport_od",
        "secondary_key",
        "price_bin",
        "ap_bin",
    ]:
        frame[column] = frame[column].astype("category")
    return frame


def _select_representative_row_ids(
    sampling_frame: pd.DataFrame,
    target_rows: int,
) -> list[str]:
    if target_rows >= len(sampling_frame):
        return sampling_frame["row_id"].astype(str).tolist()

    metro_counts = sampling_frame.groupby("metro_od", observed=True).size()
    metro_quotas = _allocate_capped_quotas(metro_counts, target_rows, guarantee_all=True)

    selected_row_ids: list[str] = []
    for _, metro_frame in sampling_frame.groupby("metro_od", observed=True, sort=False):
        metro_key = metro_frame["metro_od"].iloc[0]
        metro_quota = int(metro_quotas.get(metro_key, 0))
        if metro_quota <= 0:
            continue

        secondary_counts = metro_frame.groupby("secondary_key", observed=True).size()
        secondary_quotas = _allocate_capped_quotas(
            secondary_counts,
            metro_quota,
            guarantee_all=metro_quota >= len(secondary_counts),
        )

        for _, secondary_frame in metro_frame.groupby("secondary_key", observed=True, sort=False):
            secondary_key = secondary_frame["secondary_key"].iloc[0]
            secondary_quota = int(secondary_quotas.get(secondary_key, 0))
            if secondary_quota <= 0:
                continue

            ordered = secondary_frame.sort_values(
                by=[
                    "airport_count",
                    "source_count",
                    "carrier_count",
                    "price_bin",
                    "ap_bin",
                    "price_inc",
                    "row_hash",
                ],
                ascending=[True, True, True, True, True, True, True],
                kind="mergesort",
            )
            positions = _low_fare_first_positions(len(ordered), secondary_quota)
            selected_row_ids.extend(ordered.iloc[positions]["row_id"].astype(str).tolist())

    if len(selected_row_ids) > target_rows:
        selected_row_ids = selected_row_ids[:target_rows]
    return selected_row_ids


def _max_abs_distribution_error(
    population: pd.Series,
    sample: pd.Series,
    *,
    top_n: int | None = None,
) -> float:
    pop = population.astype("string").fillna("missing").value_counts(normalize=True)
    if top_n is not None:
        pop = pop.head(top_n)
    sample_counts = sample.astype("string").fillna("missing").value_counts(normalize=True)
    return float(max(abs(pop[value] - sample_counts.get(value, 0.0)) for value in pop.index))


def _build_quality_metrics(
    population_frame: pd.DataFrame,
    sample_frame: pd.DataFrame,
    config: DCOQwen3VisualizeConfig,
) -> dict[str, Any]:
    top_airport_markets = (
        population_frame["airport_od"].astype("string").value_counts().head(config.top_airport_market_coverage_n).index
    )
    sample_airport_markets = set(sample_frame["airport_od"].astype("string").unique())
    airport_coverage_count = int(sum(market in sample_airport_markets for market in top_airport_markets))

    population_price = pd.to_numeric(population_frame["price_inc"], errors="coerce")
    sample_price = pd.to_numeric(sample_frame["price_inc"], errors="coerce")
    population_p10 = float(population_price.quantile(0.10))
    sample_low_share = float((sample_price <= population_p10).mean())
    population_low_share = float((population_price <= population_p10).mean())

    return {
        "rows": int(len(sample_frame)),
        "metro_market_coverage": float(
            sample_frame["metro_od"].nunique(dropna=True) / population_frame["metro_od"].nunique(dropna=True)
        ),
        "metro_market_count": int(sample_frame["metro_od"].nunique(dropna=True)),
        "metro_market_total": int(population_frame["metro_od"].nunique(dropna=True)),
        "top_airport_market_coverage": float(airport_coverage_count / max(len(top_airport_markets), 1)),
        "top_airport_market_coverage_count": airport_coverage_count,
        "top_airport_market_total": int(len(top_airport_markets)),
        "fare_bin_coverage": int(sample_frame["price_bin"].nunique(dropna=True)),
        "fare_bin_total": int(population_frame["price_bin"].nunique(dropna=True)),
        "advance_purchase_bin_coverage": int(sample_frame["ap_bin"].nunique(dropna=True)),
        "advance_purchase_bin_total": int(population_frame["ap_bin"].nunique(dropna=True)),
        "trip_type_abs_error": _max_abs_distribution_error(population_frame["trip_type"], sample_frame["trip_type"]),
        "cabin_abs_error": _max_abs_distribution_error(population_frame["cabin"], sample_frame["cabin"]),
        "stops_abs_error": _max_abs_distribution_error(population_frame["stops"], sample_frame["stops"]),
        "source_top_abs_error": _max_abs_distribution_error(
            population_frame["source"],
            sample_frame["source"],
            top_n=config.top_source_coverage_n,
        ),
        "carrier_top_abs_error": _max_abs_distribution_error(
            population_frame["carrier"],
            sample_frame["carrier"],
            top_n=config.top_carrier_coverage_n,
        ),
        "price_median": float(sample_price.median()),
        "population_price_median": float(population_price.median()),
        "low_price_share": sample_low_share,
        "population_low_price_share": population_low_share,
        "low_price_share_delta": float(sample_low_share - population_low_share),
    }


def _write_selected_frames(
    parquet_path: str | Path,
    *,
    viz_row_ids: list[str],
    train_row_ids: list[str],
    viz_output_path: str | Path,
    train_output_path: str | Path,
    batch_size: int,
) -> dict[str, int]:
    parquet_path = Path(parquet_path)
    viz_output_path = Path(viz_output_path)
    train_output_path = Path(train_output_path)
    viz_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)

    viz_set = set(viz_row_ids)
    train_set = set(train_row_ids)
    viz_writer: pq.ParquetWriter | None = None
    train_writer: pq.ParquetWriter | None = None
    viz_rows = 0
    train_rows = 0

    with parquet_path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        schema = parquet.schema_arrow

    try:
        with parquet_path.open("rb") as handle:
            parquet = pq.ParquetFile(handle)
            started_at = time.perf_counter()
            total_rows = int(parquet.metadata.num_rows)
            processed_rows = 0
            for batch_index, batch in enumerate(parquet.iter_batches(batch_size=batch_size), start=1):
                frame = pa.Table.from_batches([batch]).to_pandas()
                viz_frame = frame[frame["row_id"].astype("string").isin(viz_set)]
                if not viz_frame.empty:
                    viz_table = pa.Table.from_pandas(viz_frame, schema=schema, preserve_index=False, safe=False)
                    if viz_writer is None:
                        viz_writer = pq.ParquetWriter(viz_output_path, viz_table.schema, compression="zstd")
                    viz_writer.write_table(viz_table)
                    viz_rows += len(viz_frame)

                    train_frame = viz_frame[viz_frame["row_id"].astype("string").isin(train_set)]
                    if not train_frame.empty:
                        train_table = pa.Table.from_pandas(train_frame, schema=schema, preserve_index=False, safe=False)
                        if train_writer is None:
                            train_writer = pq.ParquetWriter(train_output_path, train_table.schema, compression="zstd")
                        train_writer.write_table(train_table)
                        train_rows += len(train_frame)

                processed_rows += batch.num_rows
                if batch_index <= 2 or processed_rows == total_rows or batch_index % 5 == 0:
                    snapshot = progress_snapshot(processed_rows, total_rows, started_at)
                    LOGGER.info(
                        "Representative materialization progress: rows=%d/%d pct=%.1f viz_rows=%d train_rows=%d elapsed=%s remaining=%s eta_utc=%s",
                        snapshot.done,
                        snapshot.total,
                        snapshot.percent,
                        viz_rows,
                        train_rows,
                        format_duration(snapshot.elapsed_seconds),
                        format_duration(snapshot.remaining_seconds),
                        snapshot.eta_utc or "unknown",
                    )
    finally:
        if viz_writer is not None:
            viz_writer.close()
        if train_writer is not None:
            train_writer.close()

    return {"viz_rows": viz_rows, "train_rows": train_rows}


def build_representative_samples_from_parquet(
    parquet_path: str | Path,
    *,
    train_rows: int,
    viz_rows: int,
    random_seed: int,
    batch_size: int,
    config: DCOQwen3VisualizeConfig,
    train_output_path: str | Path,
    viz_output_path: str | Path,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    LOGGER.info(
        "Building representative samples from %s target_train_rows=%d target_viz_rows=%d",
        parquet_path,
        train_rows,
        viz_rows,
    )
    sampling_frame = _prepare_sampling_frame(parquet_path)
    LOGGER.info(
        "Loaded representative sampling frame: rows=%d metro_markets=%d airport_markets=%d elapsed=%s",
        len(sampling_frame),
        sampling_frame["metro_od"].nunique(dropna=True),
        sampling_frame["airport_od"].nunique(dropna=True),
        format_duration(time.perf_counter() - started_at),
    )

    viz_row_ids = _select_representative_row_ids(sampling_frame, min(viz_rows, len(sampling_frame)))
    viz_sampling_frame = sampling_frame[sampling_frame["row_id"].astype("string").isin(viz_row_ids)].copy()
    train_row_ids = _select_representative_row_ids(viz_sampling_frame, min(train_rows, len(viz_sampling_frame)))

    write_stats = _write_selected_frames(
        parquet_path,
        viz_row_ids=viz_row_ids,
        train_row_ids=train_row_ids,
        viz_output_path=viz_output_path,
        train_output_path=train_output_path,
        batch_size=batch_size,
    )

    quality = {
        "viz": _build_quality_metrics(sampling_frame, viz_sampling_frame, config),
        "train": _build_quality_metrics(
            sampling_frame,
            viz_sampling_frame[viz_sampling_frame["row_id"].isin(train_row_ids)],
            config,
        ),
    }
    result = {
        "train_rows": write_stats["train_rows"],
        "viz_rows": write_stats["viz_rows"],
        "quality": quality,
        "population": {
            "rows": int(len(sampling_frame)),
            "metro_market_count": int(sampling_frame["metro_od"].nunique(dropna=True)),
            "airport_market_count": int(sampling_frame["airport_od"].nunique(dropna=True)),
        },
    }
    LOGGER.info(
        "Representative sampling complete: train_rows=%d viz_rows=%d elapsed=%s",
        result["train_rows"],
        result["viz_rows"],
        format_duration(time.perf_counter() - started_at),
    )
    return result
