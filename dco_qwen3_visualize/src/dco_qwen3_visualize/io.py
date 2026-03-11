from __future__ import annotations

import json
import http.client
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from dco_qwen3_visualize.progress import format_duration, progress_snapshot

METADATA_COLUMNS = ["row_id", "source_uri", "source_row_number", "customer", "sales_date"]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParquetObject:
    uri: str
    row_count: int
    hour: str


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    without_scheme = uri[5:]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


@lru_cache(maxsize=4)
def _get_s3_filesystem(profile: str | None) -> s3fs.S3FileSystem:
    kwargs: dict[str, Any] = {}
    if profile:
        kwargs["profile"] = profile
    return s3fs.S3FileSystem(**kwargs)


def open_uri(uri: str, mode: str = "rb"):
    if uri.startswith("s3://"):
        fs = _get_s3_filesystem(os.environ.get("AWS_PROFILE"))
        return fs.open(uri, mode)
    return fsspec.open(uri, mode)


def list_s3_parquet_objects(prefix_uri: str) -> list[str]:
    import boto3

    bucket, prefix = parse_s3_uri(prefix_uri)
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")

    LOGGER.info("Listing parquet objects under %s", prefix_uri)
    parquet_uris: list[str] = []
    page_count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        page_count += 1
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".parquet"):
                parquet_uris.append(f"s3://{bucket}/{key}")

    parquet_uris.sort()
    LOGGER.info("Listed %d parquet objects under %s across %d pages", len(parquet_uris), prefix_uri, page_count)
    return parquet_uris


def collect_parquet_metadata(parquet_uris: list[str]) -> list[ParquetObject]:
    LOGGER.info("Collecting parquet metadata for %d files", len(parquet_uris))
    objects: list[ParquetObject] = []
    started_at = time.perf_counter()
    for index, uri in enumerate(parquet_uris, start=1):
        with open_uri(uri, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            objects.append(
                ParquetObject(
                    uri=uri,
                    row_count=parquet.metadata.num_rows,
                    hour=uri.rstrip("/").split("/")[-2],
                )
            )
        if index == len(parquet_uris) or index % 10 == 0:
            snapshot = progress_snapshot(index, len(parquet_uris), started_at)
            LOGGER.info(
                "Metadata progress: files=%d/%d pct=%.1f elapsed=%s rate=%.2f files/s remaining=%s eta_utc=%s",
                snapshot.done,
                snapshot.total,
                snapshot.percent,
                format_duration(snapshot.elapsed_seconds),
                snapshot.rate_per_second,
                format_duration(snapshot.remaining_seconds),
                snapshot.eta_utc or "unknown",
            )
    return objects


def sample_row_indices(total_rows: int, sample_size: int, random_seed: int) -> np.ndarray:
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    target_rows = min(sample_size, total_rows)
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(total_rows, size=target_rows, replace=False))


def decorate_frame_with_metadata(
    frame: pd.DataFrame,
    source_uri: str,
    source_row_numbers: np.ndarray,
    customer: str,
    sales_date: str,
) -> pd.DataFrame:
    enriched = frame.copy()
    source_row_numbers = source_row_numbers.astype(int, copy=False)
    enriched.insert(0, "source_row_number", source_row_numbers)
    enriched.insert(0, "source_uri", source_uri)
    enriched.insert(0, "row_id", [f"{source_uri}#{row_number}" for row_number in source_row_numbers])
    if "customer" not in enriched.columns:
        enriched.insert(3, "customer", customer)
    else:
        enriched["customer"] = enriched["customer"].fillna(customer)
    if "sales_date" not in enriched.columns:
        enriched.insert(4, "sales_date", sales_date)
    else:
        enriched["sales_date"] = enriched["sales_date"].fillna(sales_date)
    return enriched


def _metadata_field(name: str) -> pa.Field:
    if name in {"row_id", "source_uri", "customer", "sales_date"}:
        return pa.field(name, pa.string())
    if name == "source_row_number":
        return pa.field(name, pa.int64())
    raise KeyError(f"Unsupported metadata field: {name}")


def _unified_source_schema(parquet_uris: list[str]) -> pa.Schema:
    schemas: list[pa.Schema] = []
    for uri in parquet_uris:
        with open_uri(uri, "rb") as handle:
            schemas.append(pq.ParquetFile(handle).schema_arrow)
    return pa.unify_schemas(schemas)


def _schema_for_enriched_frame(columns: list[str], source_schema: pa.Schema) -> pa.Schema:
    source_fields = {field.name: field for field in source_schema}
    fields: list[pa.Field] = []
    for column in columns:
        if column in source_fields:
            fields.append(source_fields[column])
        elif column in METADATA_COLUMNS:
            fields.append(_metadata_field(column))
        else:
            raise KeyError(f"Column {column!r} is missing from the unified schema.")
    return pa.schema(fields)


def build_profile(
    sample_frame: pd.DataFrame,
    parquet_objects: list[ParquetObject],
    total_rows: int,
    sales_date: str,
    customer: str,
) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    for column in sample_frame.columns:
        series = sample_frame[column]
        columns.append(
            {
                "name": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "unique": int(series.nunique(dropna=True)),
            }
        )

    return {
        "customer": customer,
        "sales_date": sales_date,
        "source_prefix": parquet_objects[0].uri.rsplit("/", 2)[0] + "/",
        "parquet_file_count": len(parquet_objects),
        "total_rows": total_rows,
        "sample_rows": int(len(sample_frame)),
        "train_rows": int(len(sample_frame)),
        "column_count": int(len(sample_frame.columns)),
        "hours_present": sorted({item.hour for item in parquet_objects}),
        "row_counts_by_file": [{"uri": item.uri, "rows": item.row_count, "hour": item.hour} for item in parquet_objects],
        "columns": columns,
    }


def sample_parquet_files(
    parquet_uris: list[str],
    sample_size: int,
    customer: str,
    sales_date: str,
    random_seed: int,
    batch_size: int,
    parquet_objects: list[ParquetObject] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not parquet_uris:
        raise ValueError("No parquet files found for the requested partition.")

    parquet_objects = parquet_objects or collect_parquet_metadata(parquet_uris)
    total_rows = sum(item.row_count for item in parquet_objects)
    if total_rows == 0:
        raise ValueError("Parquet files were found but contain zero rows.")
    LOGGER.info(
        "Sampling %d rows from %d parquet files for customer=%s sales_date=%s total_rows=%d",
        sample_size,
        len(parquet_objects),
        customer,
        sales_date,
        total_rows,
    )
    started_at = time.perf_counter()

    global_indices = sample_row_indices(total_rows, sample_size, random_seed)
    sample_frames: list[pd.DataFrame] = []
    file_start = 0
    sampled_files = 0
    for index, item in enumerate(parquet_objects, start=1):
        file_end = file_start + item.row_count
        left = int(np.searchsorted(global_indices, file_start, side="left"))
        right = int(np.searchsorted(global_indices, file_end, side="left"))
        item_indices = global_indices[left:right] - file_start

        if len(item_indices) == 0:
            file_start = file_end
            continue

        sampled_files += 1
        cursor = 0
        batch_offset = 0
        with open_uri(item.uri, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for batch in parquet.iter_batches(batch_size=batch_size):
                batch_end = batch_offset + batch.num_rows
                next_cursor = int(np.searchsorted(item_indices, batch_end, side="left"))
                if next_cursor > cursor:
                    batch_local_indices = item_indices[cursor:next_cursor] - batch_offset
                    table = pa.Table.from_batches([batch]).take(pa.array(batch_local_indices, type=pa.int64()))
                    frame = decorate_frame_with_metadata(
                        frame=table.to_pandas(),
                        source_uri=item.uri,
                        source_row_numbers=(batch_offset + batch_local_indices),
                        customer=customer,
                        sales_date=sales_date,
                    )
                    sample_frames.append(frame)
                    cursor = next_cursor
                batch_offset = batch_end
        file_start = file_end
        if index == len(parquet_objects) or index % 10 == 0:
            collected_rows = sum(len(frame) for frame in sample_frames)
            file_snapshot = progress_snapshot(index, len(parquet_objects), started_at)
            row_snapshot = progress_snapshot(collected_rows, min(sample_size, total_rows), started_at)
            LOGGER.info(
                "Sampling progress: files=%d/%d pct=%.1f rows=%d/%d row_pct=%.1f sampled_files=%d elapsed=%s row_rate=%.0f rows/s remaining=%s eta_utc=%s",
                file_snapshot.done,
                file_snapshot.total,
                file_snapshot.percent,
                collected_rows,
                row_snapshot.total,
                row_snapshot.percent,
                sampled_files,
                format_duration(row_snapshot.elapsed_seconds),
                row_snapshot.rate_per_second,
                format_duration(row_snapshot.remaining_seconds),
                row_snapshot.eta_utc or "unknown",
            )

    sample_frame = pd.concat(sample_frames, ignore_index=True)
    profile = build_profile(sample_frame, parquet_objects, total_rows, sales_date, customer)
    LOGGER.info(
        "Completed sampling: sample_rows=%d total_rows=%d parquet_files=%d elapsed=%s",
        len(sample_frame),
        total_rows,
        len(parquet_objects),
        format_duration(time.perf_counter() - started_at),
    )
    return sample_frame, profile


def materialize_parquet_files(
    parquet_uris: list[str],
    customer: str,
    sales_date: str,
    batch_size: int,
    output_path: str | Path,
    parquet_objects: list[ParquetObject] | None = None,
    compression: str = "zstd",
) -> dict[str, Any]:
    if not parquet_uris:
        raise ValueError("No parquet files found for the requested partition.")

    parquet_objects = parquet_objects or collect_parquet_metadata(parquet_uris)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Materializing %d parquet files into %s for customer=%s sales_date=%s",
        len(parquet_objects),
        output_path,
        customer,
        sales_date,
    )
    started_at = time.perf_counter()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    expected_total_rows = sum(item.row_count for item in parquet_objects)
    source_schema = _unified_source_schema(parquet_uris)

    try:
        for index, item in enumerate(parquet_objects, start=1):
            with open_uri(item.uri, "rb") as handle:
                parquet = pq.ParquetFile(handle)
                batch_offset = 0
                for batch in parquet.iter_batches(batch_size=batch_size):
                    table = pa.Table.from_batches([batch])
                    source_row_numbers = np.arange(batch_offset, batch_offset + batch.num_rows, dtype=np.int64)
                    frame = decorate_frame_with_metadata(
                        frame=table.to_pandas(),
                        source_uri=item.uri,
                        source_row_numbers=source_row_numbers,
                        customer=customer,
                        sales_date=sales_date,
                    )
                    target_schema = writer.schema if writer is not None else _schema_for_enriched_frame(
                        list(frame.columns),
                        source_schema=source_schema,
                    )
                    enriched_table = pa.Table.from_pandas(
                        frame,
                        schema=target_schema,
                        preserve_index=False,
                        safe=False,
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, enriched_table.schema, compression=compression)
                    writer.write_table(enriched_table)
                    total_rows += batch.num_rows
                    batch_offset += batch.num_rows
            if index == len(parquet_objects) or index % 10 == 0:
                file_snapshot = progress_snapshot(index, len(parquet_objects), started_at)
                row_snapshot = progress_snapshot(total_rows, expected_total_rows, started_at)
                LOGGER.info(
                    "Materialization progress: files=%d/%d pct=%.1f rows=%d/%d row_pct=%.1f elapsed=%s row_rate=%.0f rows/s remaining=%s eta_utc=%s",
                    file_snapshot.done,
                    file_snapshot.total,
                    file_snapshot.percent,
                    row_snapshot.done,
                    row_snapshot.total,
                    row_snapshot.percent,
                    format_duration(row_snapshot.elapsed_seconds),
                    row_snapshot.rate_per_second,
                    format_duration(row_snapshot.remaining_seconds),
                    row_snapshot.eta_utc or "unknown",
                )
    finally:
        if writer is not None:
            writer.close()
    LOGGER.info(
        "Completed materialization to %s rows=%d elapsed=%s",
        output_path,
        total_rows,
        format_duration(time.perf_counter() - started_at),
    )

    return {
        "output_path": str(output_path),
        "total_rows": int(total_rows),
        "parquet_file_count": len(parquet_objects),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def content_type_for_path(path: str | Path) -> str | None:
    suffix = Path(path).suffix.lower()
    return {
        ".html": "text/html",
        ".json": "application/json",
        ".png": "image/png",
        ".parquet": "application/octet-stream",
        ".joblib": "application/octet-stream",
        ".pt": "application/octet-stream",
        ".tar": "application/x-tar",
        ".gz": "application/gzip",
        ".jsonl": "application/jsonl",
    }.get(suffix)


def artifact_uri(destination_prefix: str, filename: str) -> str:
    return f"{destination_prefix.rstrip('/')}/{filename}"


def generate_presigned_upload_urls(
    destination_prefix: str,
    filenames: list[str],
    profile_name: str | None = None,
    expires_in: int = 12 * 60 * 60,
) -> dict[str, str]:
    import boto3

    session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
    client = session.client("s3")
    bucket, prefix = parse_s3_uri(destination_prefix.rstrip("/") + "/placeholder")
    prefix = prefix.rsplit("/", 1)[0]

    urls: dict[str, str] = {}
    for filename in filenames:
        key = f"{prefix}/{filename}" if prefix else filename
        params = {"Bucket": bucket, "Key": key}
        content_type = content_type_for_path(filename)
        if content_type is not None:
            params["ContentType"] = content_type
        urls[filename] = client.generate_presigned_url(
            ClientMethod="put_object",
            Params=params,
            ExpiresIn=expires_in,
            HttpMethod="PUT",
        )
    return urls


def _put_bytes_with_presigned_url(url: str, payload: bytes, *, content_type: str | None = None) -> None:
    parts = urlsplit(url)
    connection = http.client.HTTPSConnection(parts.netloc)
    path = parts.path
    if parts.query:
        path = f"{path}?{parts.query}"
    headers = {"Content-Length": str(len(payload))}
    if content_type:
        headers["Content-Type"] = content_type
    connection.request("PUT", path, body=payload, headers=headers)
    response = connection.getresponse()
    response.read()
    connection.close()
    if response.status not in {200, 201}:
        raise RuntimeError(f"Failed to upload artifact via presigned URL: HTTP {response.status}")


def upload_artifacts_via_presigned_urls(
    filename_to_local_path: dict[str, str],
    upload_urls: dict[str, str],
) -> None:
    for filename, local_path in filename_to_local_path.items():
        if filename not in upload_urls:
            raise KeyError(f"Missing presigned URL for {filename}")
        payload = Path(local_path).read_bytes()
        _put_bytes_with_presigned_url(
            upload_urls[filename],
            payload,
            content_type=content_type_for_path(filename),
        )
