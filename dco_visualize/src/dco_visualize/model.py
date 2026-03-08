from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import trustworthiness
from torch.utils.data import DataLoader, TensorDataset

from dco_visualize.config import DCOVisualizeConfig
from dco_visualize.io import METADATA_COLUMNS, parquet_row_count, sample_row_indices

TIME_LIKE_PATTERN = re.compile(r"(date|time|timestamp|datetime|observation|depart)", re.IGNORECASE)
MONEY_LIKE_PATTERN = re.compile(r"(price|fare|tax|rate)", re.IGNORECASE)
DURATION_PATTERN = re.compile(r"(duration|gcm)", re.IGNORECASE)
TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "0.0", "false", "f", "no", "n"}
EPSILON = 1e-6

PREFERRED_CONTEXT_COLUMNS = [
    "carrier",
    "source",
    "channel",
    "pos",
    "currency",
    "trip_type",
    "stops",
    "cabin",
    "origin",
    "destination",
    "origin_city",
    "destination_city",
    "origin_metro",
    "destination_metro",
    "origin_country",
    "destination_country",
    "outbound_departure_date",
    "inbound_departure_date",
    "advance_purchase",
    "length_of_stay",
    "price_inc",
    "price_exc",
    "tax",
    "outbound_flight_duration",
    "inbound_flight_duration",
    "outbound_gcm",
]

PREFERRED_HOVER_COLUMNS = [
    "market_token",
    "carrier",
    "source",
    "trip_type",
    "stops",
    "cabin",
    "origin",
    "destination",
    "origin_metro",
    "destination_metro",
    "outbound_departure_date",
    "inbound_departure_date",
    "advance_purchase",
    "length_of_stay",
    "price_inc",
]

FINGERPRINT_FEATURES = [
    "trip_type",
    "stops",
    "cabin",
    "source",
    "carrier",
    "market_token",
]


@dataclass
class SchemaProfile:
    numeric_columns: list[str]
    boolean_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    excluded_columns: dict[str, str]
    retained_columns: list[str]
    hover_columns: list[str]
    money_columns: list[str]
    duration_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectionSpec:
    name: str
    params: dict[str, Any]


@dataclass
class FitResult:
    model: "TabularEmbeddingModel"
    metrics: dict[str, Any]


@dataclass
class AggregateRecord:
    view: str
    key_1: str | None
    key_2: str | None
    key_3: str | None
    segment_id: int | None
    count: int
    mean_log_price: float | None
    mean_price: float | None
    value: float | None


@lru_cache(maxsize=1)
def load_city_lookup() -> pd.DataFrame:
    lookup_path = (
        Path(__file__).resolve().parents[3]
        / "json2vec"
        / "src"
        / "json2vec"
        / "tasks"
        / "metrics"
        / "citylocation.csv"
    )
    frame = pd.read_csv(lookup_path)
    frame = frame.rename(columns={"citycode": "code", "countrycode": "country"})
    frame = frame[frame["latitude"].astype(str) != "(null)"].copy()
    frame = frame[frame["longitude"].astype(str) != "(null)"].copy()
    frame["code"] = frame["code"].astype(str)
    frame["country"] = frame["country"].astype(str)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame = frame.dropna(subset=["latitude", "longitude"])
    return frame[["code", "country", "latitude", "longitude"]].drop_duplicates("code").reset_index(drop=True)


def _coerce_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float32")
    text = series.astype("string").str.lower()
    mapped = text.map({value: 1.0 for value in TRUE_VALUES} | {value: 0.0 for value in FALSE_VALUES})
    return pd.to_numeric(mapped, errors="coerce").astype("float32")


def _series_is_datetime(series: pd.Series, column: str, config: DCOVisualizeConfig) -> bool:
    if not TIME_LIKE_PATTERN.search(column):
        return False
    parsed = pd.to_datetime(series.astype("string"), errors="coerce", utc=True)
    return float(parsed.notna().mean()) >= config.datetime_success_ratio


def infer_schema(frame: pd.DataFrame, config: DCOVisualizeConfig) -> SchemaProfile:
    numeric_columns: list[str] = []
    boolean_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []
    excluded_columns: dict[str, str] = {}

    for column in frame.columns:
        if column in METADATA_COLUMNS:
            continue

        series = frame[column]
        non_null = series.dropna()
        if non_null.empty:
            excluded_columns[column] = "all_null"
            continue

        unique_count = int(non_null.nunique(dropna=True))
        if unique_count <= 1:
            excluded_columns[column] = "constant"
            continue

        if pd.api.types.is_bool_dtype(series):
            boolean_columns.append(column)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_values = set(pd.Series(non_null).astype(str).str.lower().unique())
            if numeric_values.issubset(TRUE_VALUES | FALSE_VALUES | {"1.0"}):
                boolean_columns.append(column)
            else:
                numeric_columns.append(column)
            continue

        if _series_is_datetime(series, column, config):
            datetime_columns.append(column)
            continue

        text = non_null.astype("string")
        average_length = float(text.str.len().mean())
        uniqueness_ratio = float(unique_count / len(non_null))
        if uniqueness_ratio >= config.id_like_uniqueness_ratio and unique_count > 32:
            excluded_columns[column] = "id_like"
            continue
        if average_length > config.long_text_threshold:
            excluded_columns[column] = "long_text"
            continue
        categorical_columns.append(column)

    retained_columns = [column for column in PREFERRED_CONTEXT_COLUMNS if column in frame.columns]
    hover_columns = [column for column in PREFERRED_HOVER_COLUMNS if column in frame.columns]
    money_columns = [column for column in numeric_columns if MONEY_LIKE_PATTERN.search(column)]
    duration_columns = [column for column in numeric_columns if DURATION_PATTERN.search(column)]

    return SchemaProfile(
        numeric_columns=numeric_columns,
        boolean_columns=boolean_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        excluded_columns=excluded_columns,
        retained_columns=retained_columns,
        hover_columns=hover_columns[: config.max_hover_columns],
        money_columns=money_columns,
        duration_columns=duration_columns,
    )


def _clip_series(series: pd.Series, lower: float, upper: float) -> pd.Series:
    clipped = series.copy()
    clipped = clipped.clip(lower=lower, upper=upper)
    return clipped


def _bucketize_numeric(series: pd.Series, edges: list[float]) -> pd.Series:
    labels: list[str] = []
    for left, right in zip(edges[:-1], edges[1:]):
        labels.append(f"{int(left)}-{int(right)}")
    bucketed = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
    return bucketed.astype("string").fillna("missing")


def _great_circle_miles(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    rad = math.pi / 180.0
    lat1r = lat1 * rad
    lon1r = lon1 * rad
    lat2r = lat2 * rad
    lon2r = lon2 * rad
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return pd.Series(3958.7613 * (2.0 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))), index=lat1.index)


class FTTransformerEncoder(nn.Module):
    def __init__(
        self,
        numeric_count: int,
        categorical_cardinalities: list[int],
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.numeric_count = numeric_count
        self.categorical_cardinalities = categorical_cardinalities
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.numeric_weight = nn.Parameter(torch.randn(max(numeric_count, 1), d_model) * 0.02)
        self.numeric_bias = nn.Parameter(torch.zeros(max(numeric_count, 1), d_model))
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in categorical_cardinalities]
        )
        total_tokens = 1 + numeric_count + len(categorical_cardinalities)
        self.feature_embedding = nn.Parameter(torch.randn(1, max(total_tokens, 1), d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.embedding_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.numeric_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(numeric_count)])
        self.categorical_heads = nn.ModuleList(
            [nn.Linear(d_model, cardinality) for cardinality in categorical_cardinalities]
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        batch_size = numeric.shape[0] if numeric.numel() else categorical.shape[0]
        tokens: list[torch.Tensor] = [self.cls_token.expand(batch_size, -1, -1)]

        if self.numeric_count:
            numeric_tokens = numeric.unsqueeze(-1) * self.numeric_weight[: self.numeric_count]
            numeric_tokens = numeric_tokens + self.numeric_bias[: self.numeric_count]
            tokens.append(numeric_tokens)

        if self.categorical_cardinalities:
            categorical_tokens = torch.stack(
                [embedding(categorical[:, index]) for index, embedding in enumerate(self.categorical_embeddings)],
                dim=1,
            )
            tokens.append(categorical_tokens)

        token_tensor = torch.cat(tokens, dim=1)
        token_tensor = token_tensor + self.feature_embedding[:, : token_tensor.shape[1], :]
        encoded = self.output_norm(self.transformer(token_tensor))
        cls_embedding = self.embedding_head(encoded[:, 0, :])

        numeric_reconstruction: list[torch.Tensor] = []
        categorical_reconstruction: list[torch.Tensor] = []
        offset = 1
        if self.numeric_count:
            numeric_tokens = encoded[:, offset : offset + self.numeric_count, :]
            numeric_reconstruction = [
                head(numeric_tokens[:, index, :]).squeeze(-1) for index, head in enumerate(self.numeric_heads)
            ]
            offset += self.numeric_count
        if self.categorical_cardinalities:
            categorical_tokens = encoded[:, offset : offset + len(self.categorical_cardinalities), :]
            categorical_reconstruction = [
                head(categorical_tokens[:, index, :]) for index, head in enumerate(self.categorical_heads)
            ]
        return cls_embedding, numeric_reconstruction, categorical_reconstruction

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection_head(embeddings)


class AggregateCollector:
    def __init__(self) -> None:
        self.flow_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"count": 0, "sum_log_price": 0.0, "sum_price": 0.0})
        self.calendar_stats: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: {"count": 0, "sum_log_price": 0.0, "sum_price": 0.0})
        self.fingerprint_counts: Counter[tuple[int, str, str]] = Counter()
        self.global_feature_counts: Counter[tuple[str, str]] = Counter()
        self.segment_sizes: Counter[int] = Counter()

    def update(self, frame: pd.DataFrame) -> None:
        working = frame.copy()
        if "price_inc" in working.columns:
            price = pd.to_numeric(working["price_inc"], errors="coerce")
        else:
            price = pd.Series(np.nan, index=working.index)
        price = price.fillna(0.0)
        log_price = np.log1p(np.clip(price, 0, None))
        working["__price"] = price
        working["__log_price"] = log_price

        if {"origin_metro", "destination_metro"} <= set(working.columns):
            grouped = (
                working.groupby(["origin_metro", "destination_metro"], dropna=False)
                .agg(count=("row_id", "size"), sum_log_price=("__log_price", "sum"), sum_price=("__price", "sum"))
                .reset_index()
            )
            for row in grouped.itertuples(index=False):
                key = (str(row.origin_metro), str(row.destination_metro))
                stats = self.flow_stats[key]
                stats["count"] += int(row.count)
                stats["sum_log_price"] += float(row.sum_log_price)
                stats["sum_price"] += float(row.sum_price)

        if "outbound_departure_date" in working.columns:
            dep_dates = pd.to_datetime(working["outbound_departure_date"], errors="coerce")
            working["dep_date_key"] = dep_dates.dt.date.astype("string").fillna("missing")
            if "advance_purchase" in working.columns:
                ap = pd.to_numeric(working["advance_purchase"], errors="coerce")
                working["advance_bucket_key"] = _bucketize_numeric(ap, [0, 7, 14, 30, 60, 90, 180, 400])
                grouped = (
                    working.groupby(["dep_date_key", "advance_bucket_key"], dropna=False)
                    .agg(count=("row_id", "size"), sum_log_price=("__log_price", "sum"), sum_price=("__price", "sum"))
                    .reset_index()
                )
                for row in grouped.itertuples(index=False):
                    key = (str(row.dep_date_key), str(row.advance_bucket_key), "advance_purchase")
                    stats = self.calendar_stats[key]
                    stats["count"] += int(row.count)
                    stats["sum_log_price"] += float(row.sum_log_price)
                    stats["sum_price"] += float(row.sum_price)
            if "return_gap_bucket" in working.columns:
                rt = working[working["return_gap_bucket"].notna() & (working["return_gap_bucket"].astype("string") != "missing")].copy()
                if not rt.empty:
                    grouped = (
                        rt.groupby(["dep_date_key", "return_gap_bucket"], dropna=False)
                        .agg(count=("row_id", "size"), sum_log_price=("__log_price", "sum"), sum_price=("__price", "sum"))
                        .reset_index()
                    )
                    for row in grouped.itertuples(index=False):
                        key = (str(row.dep_date_key), str(row.return_gap_bucket), "return_gap")
                        stats = self.calendar_stats[key]
                        stats["count"] += int(row.count)
                        stats["sum_log_price"] += float(row.sum_log_price)
                        stats["sum_price"] += float(row.sum_price)

        if "segment_id" in working.columns:
            segment_counts = working["segment_id"].value_counts(dropna=False).to_dict()
            for segment_id, count in segment_counts.items():
                self.segment_sizes[int(segment_id)] += int(count)

        for feature in FINGERPRINT_FEATURES:
            if feature not in working.columns:
                continue
            normalized = working[feature].astype("string").fillna("missing")
            top_values = normalized.value_counts().head(24).index
            normalized = np.where(normalized.isin(top_values), normalized, "other")
            for value in normalized:
                self.global_feature_counts[(feature, str(value))] += 1
            if "segment_id" not in working.columns:
                continue
            grouped = (
                pd.DataFrame({"segment_id": working["segment_id"].astype(int), feature: normalized})
                .groupby(["segment_id", feature], dropna=False)
                .size()
                .rename("count")
                .reset_index()
            )
            for row in grouped.itertuples(index=False):
                self.fingerprint_counts[(int(row.segment_id), feature, str(getattr(row, feature)))] += int(row.count)

    def to_frame(self) -> pd.DataFrame:
        records: list[AggregateRecord] = []
        for (origin_metro, destination_metro), stats in sorted(self.flow_stats.items()):
            mean_log_price = stats["sum_log_price"] / max(stats["count"], 1)
            mean_price = stats["sum_price"] / max(stats["count"], 1)
            records.append(
                AggregateRecord(
                    view="metro_flow",
                    key_1=origin_metro,
                    key_2=destination_metro,
                    key_3=None,
                    segment_id=None,
                    count=int(stats["count"]),
                    mean_log_price=float(mean_log_price),
                    mean_price=float(mean_price),
                    value=None,
                )
            )

        for (dep_date, bucket, bucket_type), stats in sorted(self.calendar_stats.items()):
            mean_log_price = stats["sum_log_price"] / max(stats["count"], 1)
            mean_price = stats["sum_price"] / max(stats["count"], 1)
            records.append(
                AggregateRecord(
                    view="fare_calendar",
                    key_1=dep_date,
                    key_2=bucket,
                    key_3=bucket_type,
                    segment_id=None,
                    count=int(stats["count"]),
                    mean_log_price=float(mean_log_price),
                    mean_price=float(mean_price),
                    value=None,
                )
            )

        total_rows = max(sum(self.segment_sizes.values()), 1)
        for segment_id, segment_total in sorted(self.segment_sizes.items()):
            records.append(
                AggregateRecord(
                    view="segment_size",
                    key_1=None,
                    key_2=None,
                    key_3=None,
                    segment_id=int(segment_id),
                    count=int(segment_total),
                    mean_log_price=None,
                    mean_price=None,
                    value=None,
                )
            )
            feature_values = {
                (feature, value)
                for label, feature, value in self.fingerprint_counts
                if label == segment_id
            }
            for feature, value in feature_values:
                count = self.fingerprint_counts[(segment_id, feature, value)]
                global_count = self.global_feature_counts[(feature, value)]
                segment_share = count / max(segment_total, 1)
                global_share = global_count / total_rows
                lift = math.log2((segment_share + EPSILON) / (global_share + EPSILON))
                records.append(
                    AggregateRecord(
                        view="segment_fingerprint",
                        key_1=feature,
                        key_2=value,
                        key_3=None,
                        segment_id=int(segment_id),
                        count=int(count),
                        mean_log_price=None,
                        mean_price=None,
                        value=float(lift),
                    )
                )

        return pd.DataFrame([asdict(record) for record in records])


class TabularEmbeddingModel:
    def __init__(self, config: DCOVisualizeConfig) -> None:
        self.config = config
        self.schema: SchemaProfile | None = None
        self.numeric_feature_names: list[str] = []
        self.categorical_feature_names: list[str] = []
        self.numeric_centers: dict[str, float] = {}
        self.numeric_scales: dict[str, float] = {}
        self.raw_winsorization: dict[str, tuple[float, float]] = {}
        self.categorical_vocabularies: dict[str, dict[str, int]] = {}
        self.encoder: FTTransformerEncoder | None = None
        self.encoder_device: str = "cpu"
        self.segmenter: Any | None = None
        self.segmenter_kind: str = self.config.segment_method
        self.projector: Any | None = None
        self.projection_spec: ProjectionSpec | None = None
        self.cluster_count: int | None = None
        self.noise_fraction: float = 0.0
        self.geocode_coverage: dict[str, float] = {}
        self.train_loss_history: list[dict[str, float]] = []

    def fit(self, frame: pd.DataFrame) -> dict[str, Any]:
        self.schema = infer_schema(frame, self.config)
        numeric_frame, categorical_frame, engineered = self._build_feature_frames(frame, fit=True)
        numeric_array, categorical_array = self._fit_preprocessor(numeric_frame, categorical_frame)
        self._fit_encoder(numeric_array, categorical_array)
        embeddings = self._encode_embeddings(numeric_array, categorical_array)
        labels = self._fit_segmenter(embeddings)
        metrics = self._build_metrics(engineered, embeddings, labels)
        return metrics

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        if self.schema is None or self.encoder is None:
            raise ValueError("Model must be fitted before transform().")
        numeric_frame, categorical_frame, engineered = self._build_feature_frames(frame, fit=False)
        numeric_array, categorical_array = self._transform_preprocessor(numeric_frame, categorical_frame)
        embeddings = self._encode_embeddings(numeric_array, categorical_array)
        labels = self._predict_segments(embeddings)
        transformed = self._assemble_output_frame(frame, engineered, embeddings, labels)
        return transformed, embeddings

    def project_visualization_sample(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        if frame.empty:
            raise ValueError("Visualization frame is empty.")
        embedding_columns = [column for column in frame.columns if column.startswith("embedding_")]
        if not embedding_columns:
            raise ValueError("Visualization frame does not contain embedding columns.")
        embeddings = frame[embedding_columns].to_numpy(dtype=np.float32, copy=False)
        neighbors = max(2, min(45, max(10, len(frame) // 200), len(frame) - 1))
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            densmap=True,
            n_neighbors=neighbors,
            min_dist=0.05,
            random_state=self.config.random_seed,
            transform_seed=self.config.random_seed,
        )
        coordinates = reducer.fit_transform(embeddings)
        result = frame.copy()
        result["layout_x"] = coordinates[:, 0]
        result["layout_y"] = coordinates[:, 1]
        result["layout_method"] = "densmap"
        self.projector = reducer
        self.projection_spec = ProjectionSpec(
            name="densmap",
            params={
                "metric": "cosine",
                "n_neighbors": int(neighbors),
                "min_dist": 0.05,
                "densmap": True,
            },
        )
        trust = None
        if len(frame) > 10:
            try:
                trust = float(trustworthiness(embeddings, coordinates, n_neighbors=min(10, len(frame) - 1)))
            except Exception:
                trust = None
        return result, {
            "projection": {"name": "densmap", "metric": "cosine", "n_neighbors": int(neighbors), "min_dist": 0.05},
            "projection_trustworthiness": trust,
        }

    def _build_feature_frames(self, frame: pd.DataFrame, fit: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert self.schema is not None
        working = frame.copy()
        lookup = load_city_lookup()
        code_to_country = lookup.set_index("code")["country"].to_dict()
        code_to_lat = lookup.set_index("code")["latitude"].to_dict()
        code_to_lon = lookup.set_index("code")["longitude"].to_dict()

        if "origin_metro" in working.columns:
            working["origin_latitude"] = working["origin_metro"].astype("string").map(code_to_lat)
            working["origin_longitude"] = working["origin_metro"].astype("string").map(code_to_lon)
        if "destination_metro" in working.columns:
            working["destination_latitude"] = working["destination_metro"].astype("string").map(code_to_lat)
            working["destination_longitude"] = working["destination_metro"].astype("string").map(code_to_lon)

        for column in ("origin_country", "destination_country"):
            if column in working.columns:
                continue
            metro_column = "origin_metro" if column.startswith("origin") else "destination_metro"
            if metro_column in working.columns:
                working[column] = working[metro_column].astype("string").map(code_to_country)

        if "outbound_gcm" in working.columns:
            gcm = pd.to_numeric(working["outbound_gcm"], errors="coerce")
        elif {"origin_latitude", "origin_longitude", "destination_latitude", "destination_longitude"} <= set(working.columns):
            gcm = _great_circle_miles(
                pd.to_numeric(working["origin_latitude"], errors="coerce"),
                pd.to_numeric(working["origin_longitude"], errors="coerce"),
                pd.to_numeric(working["destination_latitude"], errors="coerce"),
                pd.to_numeric(working["destination_longitude"], errors="coerce"),
            )
        else:
            gcm = pd.Series(np.nan, index=working.index)
        working["route_great_circle_miles"] = gcm

        origin_market = None
        destination_market = None
        for preferred in ("origin_metro", "origin_city", "origin"):
            if preferred in working.columns:
                origin_market = working[preferred].astype("string")
                break
        for preferred in ("destination_metro", "destination_city", "destination"):
            if preferred in working.columns:
                destination_market = working[preferred].astype("string")
                break
        if origin_market is not None and destination_market is not None:
            working["market_token"] = origin_market.fillna("missing") + "->" + destination_market.fillna("missing")
        else:
            working["market_token"] = "missing"

        if {"origin_country", "destination_country"} <= set(working.columns):
            working["domestic_flag"] = (
                working["origin_country"].astype("string").fillna("missing")
                == working["destination_country"].astype("string").fillna("missing")
            ).astype("float32")
        else:
            working["domestic_flag"] = np.nan

        outbound_source = (
            working["outbound_departure_date"]
            if "outbound_departure_date" in working.columns
            else pd.Series(pd.NaT, index=working.index)
        )
        inbound_source = (
            working["inbound_departure_date"]
            if "inbound_departure_date" in working.columns
            else pd.Series(pd.NaT, index=working.index)
        )
        outbound_departure = pd.to_datetime(outbound_source, errors="coerce", utc=True)
        inbound_departure = pd.to_datetime(inbound_source, errors="coerce", utc=True)
        sale_timestamp = pd.Timestamp(self.config.sales_date, tz="UTC")
        working["outbound_days_from_sale"] = (outbound_departure - sale_timestamp).dt.total_seconds() / 86_400.0
        working["return_gap_days"] = (inbound_departure - outbound_departure).dt.total_seconds() / 86_400.0
        working["is_round_trip"] = (
            working.get("trip_type", pd.Series("", index=working.index)).astype("string").fillna("") == "RT"
        ).astype("float32")
        working["return_gap_bucket"] = _bucketize_numeric(
            pd.to_numeric(working["return_gap_days"], errors="coerce"),
            [0, 1, 3, 7, 14, 21, 30, 60, 400],
        )

        geocode_fields = {
            "origin_metro": working.get("origin_metro"),
            "destination_metro": working.get("destination_metro"),
            "origin_city": working.get("origin_city"),
            "destination_city": working.get("destination_city"),
        }
        if fit:
            self.geocode_coverage = {}
            for name, series in geocode_fields.items():
                if series is None:
                    continue
                non_null = series.astype("string").dropna()
                if non_null.empty:
                    self.geocode_coverage[name] = 0.0
                else:
                    self.geocode_coverage[name] = float(non_null.isin(lookup["code"]).mean())

        numeric = pd.DataFrame(index=working.index)
        categorical = pd.DataFrame(index=working.index)

        for column in self.schema.numeric_columns:
            series = pd.to_numeric(working[column], errors="coerce")
            if column == "length_of_stay":
                numeric["length_of_stay_days"] = series.mask(series < 0)
                numeric["is_one_way_stay_sentinel"] = series.lt(0).astype("float32")
                continue
            if MONEY_LIKE_PATTERN.search(column):
                if fit:
                    lower = float(series.quantile(0.001))
                    upper = float(series.quantile(0.999))
                    self.raw_winsorization[column] = (lower, upper)
                lower, upper = self.raw_winsorization.get(column, (float(series.min()), float(series.max())))
                clipped = _clip_series(series, lower, upper)
                numeric[f"log1p_{column}"] = np.log1p(np.clip(clipped, 0, None))
                continue
            if DURATION_PATTERN.search(column):
                numeric[f"log1p_{column}"] = np.log1p(np.clip(series, 0, None))
                continue
            numeric[column] = series

        for column in self.schema.boolean_columns:
            numeric[column] = _coerce_boolean(working[column])

        for column in self.schema.datetime_columns:
            parsed = pd.to_datetime(working[column], errors="coerce", utc=True)
            numeric[f"{column}__days_from_sale"] = (parsed - sale_timestamp).dt.total_seconds() / 86_400.0
            day = parsed.dt.dayofweek
            month = parsed.dt.month
            numeric[f"{column}__weekday_sin"] = np.sin((2.0 * math.pi * day) / 7.0)
            numeric[f"{column}__weekday_cos"] = np.cos((2.0 * math.pi * day) / 7.0)
            numeric[f"{column}__month_sin"] = np.sin((2.0 * math.pi * month) / 12.0)
            numeric[f"{column}__month_cos"] = np.cos((2.0 * math.pi * month) / 12.0)

        numeric["outbound_days_from_sale"] = pd.to_numeric(working["outbound_days_from_sale"], errors="coerce")
        numeric["return_gap_days"] = pd.to_numeric(working["return_gap_days"], errors="coerce").mask(
            pd.to_numeric(working["return_gap_days"], errors="coerce") < 0
        )
        numeric["is_round_trip"] = pd.to_numeric(working["is_round_trip"], errors="coerce")
        numeric["domestic_flag"] = pd.to_numeric(working["domestic_flag"], errors="coerce")
        numeric["log1p_route_great_circle_miles"] = np.log1p(np.clip(pd.to_numeric(working["route_great_circle_miles"], errors="coerce"), 0, None))

        for column in self.schema.categorical_columns:
            if column in self.schema.datetime_columns:
                continue
            categorical[column] = working[column].astype("string")
        categorical["market_token"] = working["market_token"].astype("string")
        if "origin_country" in working.columns and "destination_country" in working.columns:
            categorical["country_pair"] = (
                working["origin_country"].astype("string").fillna("missing")
                + "->"
                + working["destination_country"].astype("string").fillna("missing")
            )

        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        categorical = categorical.fillna("missing")
        return numeric, categorical, working

    def _fit_preprocessor(self, numeric: pd.DataFrame, categorical: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if numeric.empty and categorical.empty:
            raise ValueError("No usable engineered features remain after schema inference.")
        self.numeric_feature_names = numeric.columns.tolist()
        self.categorical_feature_names = categorical.columns.tolist()

        numeric_array = np.zeros((len(numeric), len(self.numeric_feature_names)), dtype=np.float32)
        for index, column in enumerate(self.numeric_feature_names):
            series = pd.to_numeric(numeric[column], errors="coerce")
            center = float(series.median()) if series.notna().any() else 0.0
            q1 = float(series.quantile(0.25)) if series.notna().any() else 0.0
            q3 = float(series.quantile(0.75)) if series.notna().any() else 0.0
            scale = q3 - q1
            if not np.isfinite(scale) or abs(scale) < EPSILON:
                scale = float(series.std()) if series.notna().any() else 1.0
            if not np.isfinite(scale) or abs(scale) < EPSILON:
                scale = 1.0
            self.numeric_centers[column] = center
            self.numeric_scales[column] = scale
            filled = series.fillna(center)
            numeric_array[:, index] = ((filled - center) / scale).to_numpy(dtype=np.float32)

        categorical_array = np.zeros((len(categorical), len(self.categorical_feature_names)), dtype=np.int64)
        for index, column in enumerate(self.categorical_feature_names):
            counts = categorical[column].astype("string").value_counts()
            allowed = counts.head(max(self.config.max_categories - 2, 1)).index.tolist()
            vocab = {"<unk>": 0, "<mask>": 1}
            for item in allowed:
                vocab[str(item)] = len(vocab)
            self.categorical_vocabularies[column] = vocab
            categorical_array[:, index] = categorical[column].astype("string").map(vocab).fillna(0).astype(int).to_numpy()
        return numeric_array, categorical_array

    def _transform_preprocessor(self, numeric: pd.DataFrame, categorical: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric_array = np.zeros((len(numeric), len(self.numeric_feature_names)), dtype=np.float32)
        for index, column in enumerate(self.numeric_feature_names):
            if column not in numeric.columns:
                numeric_array[:, index] = 0.0
                continue
            series = pd.to_numeric(numeric[column], errors="coerce").fillna(self.numeric_centers[column])
            numeric_array[:, index] = ((series - self.numeric_centers[column]) / self.numeric_scales[column]).to_numpy(dtype=np.float32)

        categorical_array = np.zeros((len(categorical), len(self.categorical_feature_names)), dtype=np.int64)
        for index, column in enumerate(self.categorical_feature_names):
            vocab = self.categorical_vocabularies[column]
            series = categorical[column].astype("string") if column in categorical.columns else pd.Series("missing", index=categorical.index)
            categorical_array[:, index] = series.map(vocab).fillna(0).astype(int).to_numpy()
        return numeric_array, categorical_array

    def _fit_encoder(self, numeric_array: np.ndarray, categorical_array: np.ndarray) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder_device = device
        cardinalities = [
            max(self.categorical_vocabularies[column].values()) + 1 for column in self.categorical_feature_names
        ]
        self.encoder = FTTransformerEncoder(
            numeric_count=len(self.numeric_feature_names),
            categorical_cardinalities=cardinalities,
            d_model=self.config.transformer_width,
            n_heads=self.config.transformer_heads,
            n_layers=self.config.transformer_layers,
            dropout=self.config.transformer_dropout,
            embedding_dim=self.config.embedding_dims,
        ).to(device)
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        numeric_tensor = torch.from_numpy(numeric_array)
        categorical_tensor = torch.from_numpy(categorical_array if categorical_array.size else np.zeros((len(numeric_array), 0), dtype=np.int64))
        dataset = TensorDataset(numeric_tensor, categorical_tensor)
        batch_size = min(self.config.train_batch_size, max(len(dataset), 1))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train_loss_history = []
        self.encoder.train()
        for _ in range(self.config.pretrain_epochs):
            total_loss = 0.0
            total_examples = 0
            for batch_numeric, batch_categorical in loader:
                batch_numeric = batch_numeric.to(device)
                batch_categorical = batch_categorical.to(device)
                aug_numeric_a, aug_categorical_a = self._corrupt_batch(batch_numeric, batch_categorical)
                aug_numeric_b, aug_categorical_b = self._corrupt_batch(batch_numeric, batch_categorical)

                emb_a, num_rec_a, cat_rec_a = self.encoder(aug_numeric_a, aug_categorical_a)
                emb_b, _, _ = self.encoder(aug_numeric_b, aug_categorical_b)

                proj_a = self.encoder.project(emb_a)
                proj_b = self.encoder.project(emb_b)
                loss = self._contrastive_loss(proj_a, proj_b)

                if self.numeric_feature_names:
                    numeric_loss = 0.0
                    for index, prediction in enumerate(num_rec_a):
                        numeric_loss = numeric_loss + F.smooth_l1_loss(prediction, batch_numeric[:, index])
                    loss = loss + (0.2 * numeric_loss / max(len(num_rec_a), 1))

                if self.categorical_feature_names:
                    categorical_loss = 0.0
                    for index, prediction in enumerate(cat_rec_a):
                        categorical_loss = categorical_loss + F.cross_entropy(prediction, batch_categorical[:, index])
                    loss = loss + (0.2 * categorical_loss / max(len(cat_rec_a), 1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_examples += int(batch_numeric.shape[0])
                total_loss += float(loss.item()) * batch_numeric.shape[0]
            self.train_loss_history.append({"loss": total_loss / max(total_examples, 1)})
        self.encoder.eval()

    def _corrupt_batch(self, numeric: torch.Tensor, categorical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        numeric_out = numeric.clone()
        categorical_out = categorical.clone()
        if numeric_out.numel():
            numeric_mask = torch.rand_like(numeric_out) < self.config.corruption_rate
            perm = torch.randperm(numeric_out.shape[0], device=numeric_out.device) if numeric_out.shape[0] > 1 else None
            replacement = numeric_out[perm] if perm is not None else torch.zeros_like(numeric_out)
            numeric_out = torch.where(numeric_mask, replacement, numeric_out)
        if categorical_out.numel():
            categorical_mask = torch.rand(categorical_out.shape, device=categorical_out.device) < self.config.corruption_rate
            if categorical_out.shape[0] > 1:
                replacement = categorical_out[torch.randperm(categorical_out.shape[0], device=categorical_out.device)]
            else:
                replacement = torch.zeros_like(categorical_out)
            use_mask_token = torch.rand(categorical_out.shape, device=categorical_out.device) < 0.5
            replacement = torch.where(use_mask_token, torch.ones_like(categorical_out), replacement)
            categorical_out = torch.where(categorical_mask, replacement, categorical_out)
        return numeric_out, categorical_out

    def _contrastive_loss(self, proj_a: torch.Tensor, proj_b: torch.Tensor) -> torch.Tensor:
        proj_a = F.normalize(proj_a, dim=-1)
        proj_b = F.normalize(proj_b, dim=-1)
        logits = torch.matmul(proj_a, proj_b.T) / self.config.contrastive_temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

    @torch.inference_mode()
    def _encode_embeddings(self, numeric_array: np.ndarray, categorical_array: np.ndarray) -> np.ndarray:
        assert self.encoder is not None
        numeric_tensor = torch.from_numpy(numeric_array)
        categorical_tensor = torch.from_numpy(categorical_array if categorical_array.size else np.zeros((len(numeric_array), 0), dtype=np.int64))
        dataset = TensorDataset(numeric_tensor, categorical_tensor)
        batch_size = min(self.config.train_batch_size, max(len(dataset), 1))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        embeddings: list[np.ndarray] = []
        self.encoder.eval()
        for batch_numeric, batch_categorical in loader:
            batch_numeric = batch_numeric.to(self.encoder_device)
            batch_categorical = batch_categorical.to(self.encoder_device)
            cls_embedding, _, _ = self.encoder(batch_numeric, batch_categorical)
            embeddings.append(cls_embedding.cpu().numpy())
        return np.concatenate(embeddings, axis=0).astype(np.float32)

    def _fit_segmenter(self, embeddings: np.ndarray) -> np.ndarray:
        min_cluster_size = min(max(self.config.min_segment_size, len(embeddings) // 200), max(len(embeddings) // 10, 2))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(min_cluster_size, 2),
            min_samples=max(min_cluster_size // 2, 2),
            prediction_data=True,
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)
        unique_labels = {int(label) for label in labels if int(label) >= 0}
        if len(unique_labels) < 2:
            fallback_clusters = min(8, max(2, len(embeddings) // 5_000 or 2))
            fallback = MiniBatchKMeans(
                n_clusters=fallback_clusters,
                random_state=self.config.random_seed,
                batch_size=min(self.config.train_batch_size, len(embeddings)),
                n_init="auto",
            )
            labels = fallback.fit_predict(embeddings)
            self.segmenter = fallback
            self.segmenter_kind = "kmeans_fallback"
        else:
            self.segmenter = clusterer
            self.segmenter_kind = "hdbscan"

        self.cluster_count = len({int(label) for label in labels if int(label) >= 0})
        self.noise_fraction = float(np.mean(labels == -1)) if self.segmenter_kind == "hdbscan" else 0.0
        return labels.astype(int)

    def _predict_segments(self, embeddings: np.ndarray) -> np.ndarray:
        if self.segmenter is None:
            raise ValueError("Segmenter must be fitted before transform().")
        if self.segmenter_kind == "hdbscan":
            labels, _ = hdbscan.approximate_predict(self.segmenter, embeddings)
            return labels.astype(int)
        return self.segmenter.predict(embeddings).astype(int)

    def _assemble_output_frame(
        self,
        frame: pd.DataFrame,
        engineered: pd.DataFrame,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        assert self.schema is not None
        output_columns = METADATA_COLUMNS + [column for column in self.schema.retained_columns if column in frame.columns]
        output = frame[output_columns].copy()
        derived_columns = {
            "market_token": engineered["market_token"].astype("string"),
            "domestic_flag": pd.to_numeric(engineered["domestic_flag"], errors="coerce"),
            "is_round_trip": pd.to_numeric(engineered["is_round_trip"], errors="coerce"),
            "return_gap_days": pd.to_numeric(engineered["return_gap_days"], errors="coerce"),
            "return_gap_bucket": engineered["return_gap_bucket"].astype("string"),
        }
        if "price_inc" in frame.columns:
            price = pd.to_numeric(frame["price_inc"], errors="coerce")
            derived_columns["log_price_inc"] = np.log1p(np.clip(price.fillna(0.0), 0, None))
        if "advance_purchase" in frame.columns:
            derived_columns["advance_purchase_bucket"] = _bucketize_numeric(
                pd.to_numeric(frame["advance_purchase"], errors="coerce"),
                [0, 7, 14, 30, 60, 90, 180, 400],
            )
        for name, series in derived_columns.items():
            output[name] = series
        for index in range(embeddings.shape[1]):
            output[f"embedding_{index:03d}"] = embeddings[:, index]
        output["segment_id"] = labels.astype(int)
        return output

    def _build_metrics(self, frame: pd.DataFrame, embeddings: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        assert self.schema is not None
        non_noise = labels[labels >= 0] if self.segmenter_kind == "hdbscan" else labels
        segment_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        dominance = 0.0
        if len(labels):
            dominance = max(segment_sizes.values()) / len(labels)
        return {
            "sample_rows": int(len(frame)),
            "train_rows": int(len(frame)),
            "retained_columns": self.schema.retained_columns,
            "excluded_columns": self.schema.excluded_columns,
            "hover_columns": self.schema.hover_columns,
            "numeric_feature_count": len(self.numeric_feature_names),
            "categorical_feature_count": len(self.categorical_feature_names),
            "embedding_dims": int(embeddings.shape[1]),
            "encoder_backend": self.config.encoder_backend,
            "segment_method": self.segmenter_kind,
            "segment_count": int(len({int(label) for label in non_noise})),
            "segment_sizes": {str(key): int(value) for key, value in segment_sizes.items()},
            "cluster_dominance": float(dominance),
            "noise_fraction": float(self.noise_fraction),
            "schema": self.schema.to_dict(),
            "geocode_coverage": self.geocode_coverage,
            "winsorization_thresholds": {
                key: {"lower": float(value[0]), "upper": float(value[1])}
                for key, value in self.raw_winsorization.items()
            },
            "training": {
                "epochs": self.config.pretrain_epochs,
                "batch_size": self.config.train_batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "loss_history": self.train_loss_history,
            },
            "config": self.config.to_dict(),
        }

    def to_bundle(self) -> dict[str, Any]:
        if self.encoder is None or self.schema is None:
            raise ValueError("Model must be fitted before serialization.")
        return {
            "config": self.config.to_dict(),
            "schema": self.schema.to_dict(),
            "numeric_feature_names": self.numeric_feature_names,
            "categorical_feature_names": self.categorical_feature_names,
            "numeric_centers": self.numeric_centers,
            "numeric_scales": self.numeric_scales,
            "raw_winsorization": self.raw_winsorization,
            "categorical_vocabularies": self.categorical_vocabularies,
            "encoder_device": self.encoder_device,
            "encoder_state_dict": self.encoder.state_dict(),
            "encoder_model_args": {
                "numeric_count": len(self.numeric_feature_names),
                "categorical_cardinalities": [
                    max(self.categorical_vocabularies[column].values()) + 1 for column in self.categorical_feature_names
                ],
                "d_model": self.config.transformer_width,
                "n_heads": self.config.transformer_heads,
                "n_layers": self.config.transformer_layers,
                "dropout": self.config.transformer_dropout,
                "embedding_dim": self.config.embedding_dims,
            },
            "segmenter": self.segmenter,
            "segmenter_kind": self.segmenter_kind,
            "projection_spec": asdict(self.projection_spec) if self.projection_spec else None,
            "projector": self.projector,
            "geocode_coverage": self.geocode_coverage,
        }


def fit_embedding_model(frame: pd.DataFrame, config: DCOVisualizeConfig) -> FitResult:
    model = TabularEmbeddingModel(config)
    metrics = model.fit(frame)
    return FitResult(model=model, metrics=metrics)


def write_embedding_bundle(path: str | Path, model: TabularEmbeddingModel) -> None:
    torch.save(model.to_bundle(), path)


def stratify_visualization_sample(frame: pd.DataFrame, viz_rows: int, random_seed: int) -> pd.DataFrame:
    if len(frame) <= viz_rows:
        return frame.reset_index(drop=True)
    sampled = frame.copy()
    if "log_price_inc" not in sampled.columns:
        if "price_inc" in sampled.columns:
            sampled["log_price_inc"] = np.log1p(np.clip(pd.to_numeric(sampled["price_inc"], errors="coerce").fillna(0.0), 0, None))
        else:
            sampled["log_price_inc"] = 0.0
    quantiles = min(10, max(2, sampled["log_price_inc"].nunique(dropna=True)))
    sampled["fare_bucket"] = pd.qcut(sampled["log_price_inc"].rank(method="first"), q=quantiles, duplicates="drop").astype("string")
    sampled["_stratum"] = (
        sampled.get("segment_id", pd.Series(0, index=sampled.index)).astype("string").fillna("missing")
        + "|"
        + sampled.get("trip_type", pd.Series("missing", index=sampled.index)).astype("string").fillna("missing")
        + "|"
        + sampled.get("stops", pd.Series("missing", index=sampled.index)).astype("string").fillna("missing")
        + "|"
        + sampled["fare_bucket"].astype("string").fillna("missing")
    )

    rng = np.random.default_rng(random_seed)
    counts = sampled["_stratum"].value_counts()
    allocated = {stratum: max(1, int(math.floor(viz_rows * count / len(sampled)))) for stratum, count in counts.items()}
    current = sum(allocated.values())
    if current > viz_rows:
        for stratum in sorted(allocated, key=allocated.get, reverse=True):
            if current <= viz_rows:
                break
            if allocated[stratum] > 1:
                allocated[stratum] -= 1
                current -= 1
    elif current < viz_rows:
        for stratum in counts.index.tolist():
            if current >= viz_rows:
                break
            allocated[stratum] += 1
            current += 1

    selections: list[pd.DataFrame] = []
    for stratum, take in allocated.items():
        group = sampled[sampled["_stratum"] == stratum]
        if group.empty:
            continue
        take = min(take, len(group))
        if take >= len(group):
            selections.append(group)
            continue
        indices = rng.choice(group.index.to_numpy(), size=take, replace=False)
        selections.append(group.loc[indices])
    result = pd.concat(selections, ignore_index=True)
    if len(result) > viz_rows:
        result = result.sample(n=viz_rows, random_state=random_seed)
    return result.drop(columns=["_stratum", "fare_bucket"], errors="ignore").sort_values(["segment_id", "market_token"], kind="stable").reset_index(drop=True)


def transform_parquet_file(
    model: TabularEmbeddingModel,
    parquet_path: str | Path,
    output_path: str | Path,
    viz_rows: int,
    batch_size: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = parquet_row_count(str(parquet_path))
    candidate_rows = min(total_rows, max(viz_rows, viz_rows * model.config.viz_candidate_multiplier))
    sample_indices = sample_row_indices(total_rows, candidate_rows, random_seed)
    sample_cursor = 0
    batch_offset = 0
    embedded_rows = 0
    segment_sizes: dict[int, int] = {}
    candidate_frames: list[pd.DataFrame] = []
    writer: pq.ParquetWriter | None = None
    collector = AggregateCollector()

    try:
        with parquet_path.open("rb") as handle:
            parquet = pq.ParquetFile(handle)
            for batch in parquet.iter_batches(batch_size=batch_size):
                frame = pa.Table.from_batches([batch]).to_pandas()
                transformed, _ = model.transform(frame)
                embedded_rows += len(transformed)
                collector.update(transformed)

                counts = transformed["segment_id"].value_counts().to_dict()
                for segment_id, count in counts.items():
                    segment_sizes[int(segment_id)] = segment_sizes.get(int(segment_id), 0) + int(count)

                transformed_table = pa.Table.from_pandas(transformed, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, transformed_table.schema, compression="zstd")
                writer.write_table(transformed_table)

                batch_end = batch_offset + batch.num_rows
                next_cursor = int(np.searchsorted(sample_indices, batch_end, side="left"))
                if next_cursor > sample_cursor:
                    local_indices = (sample_indices[sample_cursor:next_cursor] - batch_offset).astype(int, copy=False)
                    candidate_frames.append(transformed.iloc[local_indices].reset_index(drop=True))
                    sample_cursor = next_cursor
                batch_offset = batch_end
    finally:
        if writer is not None:
            writer.close()

    if not candidate_frames:
        raise ValueError("No rows were collected for visualization sampling.")

    candidate_frame = pd.concat(candidate_frames, ignore_index=True)
    viz_frame = stratify_visualization_sample(candidate_frame, viz_rows=viz_rows, random_seed=random_seed)
    viz_frame, projection_metrics = model.project_visualization_sample(viz_frame)
    aggregate_frame = collector.to_frame()

    metrics = {
        "embedded_rows": int(embedded_rows),
        "viz_rows": int(len(viz_frame)),
        "segment_sizes": {str(key): int(value) for key, value in sorted(segment_sizes.items())},
        "segment_count": int(len({segment_id for segment_id in segment_sizes if segment_id >= 0})),
        "noise_fraction": float(model.noise_fraction),
        **projection_metrics,
    }
    return viz_frame, aggregate_frame, metrics
