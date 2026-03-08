from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from dco_visualize.config import DCOVisualizeConfig
from dco_visualize.io import METADATA_COLUMNS, sample_row_indices

LOGGER = logging.getLogger(__name__)

PREFERRED_HOVER_COLUMNS = [
    "carrier",
    "source",
    "trip_type",
    "cabin",
    "stops",
    "origin",
    "destination",
    "origin_city",
    "destination_city",
    "origin_metro",
    "destination_metro",
    "outbound_departure_date",
    "inbound_departure_date",
    "price_inc",
]

FINGERPRINT_FEATURES = [
    "trip_type",
    "stops",
    "cabin",
    "source",
    "carrier",
    "origin",
    "destination",
    "origin_city",
    "destination_city",
    "origin_metro",
    "destination_metro",
]


@dataclass
class ProjectionSpec:
    name: str
    params: dict[str, Any]


@dataclass
class FitResult:
    model: "TabPFNEmbeddingModel"
    metrics: dict[str, Any]


@dataclass
class AggregateRecord:
    view: str
    branch: str | None
    key_1: str | None
    key_2: str | None
    key_3: str | None
    segment_id: int | None
    count: int
    mean_price: float | None
    value: float | None


@dataclass
class SegmenterModel:
    kind: str
    model: Any
    n_clusters: int

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            return np.empty((0,), dtype=np.int64)
        if self.kind == "hdbscan":
            labels, _ = hdbscan.prediction.approximate_predict(self.model, embeddings)
            return labels.astype(np.int64, copy=False)
        return self.model.predict(embeddings).astype(np.int64, copy=False)


@dataclass
class BranchState:
    name: str
    model: Any
    embedding_model: Any
    segmenter: SegmenterModel
    embedding_dim: int
    prediction_metrics: dict[str, Any]
    projection: ProjectionSpec | None = None
    projector: Any | None = None
    segment_labels: list[int] | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=np.float32)

    def get_embeddings(self, X: pd.DataFrame, data_source: str) -> np.ndarray:
        raw = self.embedding_model.get_embeddings(X, data_source=data_source)
        return collapse_embeddings(raw)


@dataclass
class TabPFNEmbeddingModel:
    config: DCOVisualizeConfig
    target_column: str
    feature_columns: list[str]
    feature_kinds: dict[str, str]
    categorical_feature_indices: list[int]
    excluded_columns: dict[str, str]
    retained_columns: list[str]
    hover_columns: list[str]
    pretrained: BranchState
    finetuned: BranchState
    route_source_column: str
    route_destination_column: str
    departure_date_column: str
    advance_purchase_column: str
    return_date_column: str

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        return transform_frame(self, frame)


@dataclass
class AggregateAccumulator:
    route_stats: dict[tuple[str, str], list[float]]
    fare_calendar: dict[tuple[str, str], list[float]]
    agreement_counts: Counter[tuple[int, int]]
    branch_segment_counts: Counter[tuple[str, int]]
    overall_feature_counts: Counter[tuple[str, str]]
    branch_segment_feature_counts: Counter[tuple[str, int, str, str]]

    @classmethod
    def create(cls) -> "AggregateAccumulator":
        return cls(
            route_stats=defaultdict(lambda: [0.0, 0.0]),
            fare_calendar=defaultdict(lambda: [0.0, 0.0]),
            agreement_counts=Counter(),
            branch_segment_counts=Counter(),
            overall_feature_counts=Counter(),
            branch_segment_feature_counts=Counter(),
        )


def _runtime_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_tabpfn_runtime() -> None:
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    token = os.environ.get("HF_TOKEN")
    if not token:
        LOGGER.warning("HF_TOKEN is not set; TabPFN gated model download may fail")
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False, skip_if_logged_in=True)
        LOGGER.info("Authenticated to Hugging Face for TabPFN runtime")
    except TypeError:
        login(token=token, add_to_git_credential=False)
        LOGGER.info("Authenticated to Hugging Face for TabPFN runtime")


def _tabpfn_regressor_model_path() -> str:
    from tabpfn.model_loading import ModelSource, prepend_cache_path

    return str(prepend_cache_path(ModelSource.get_regressor_v2_5().default_filename))


def collapse_embeddings(raw: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(raw)
    if embeddings.ndim == 3:
        return embeddings.mean(axis=0).astype(np.float32, copy=False)
    if embeddings.ndim == 2:
        return embeddings.astype(np.float32, copy=False)
    raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")


def _normalize_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def infer_feature_contract(
    frame: pd.DataFrame, config: DCOVisualizeConfig
) -> tuple[list[str], dict[str, str], dict[str, str], list[str]]:
    feature_columns: list[str] = []
    feature_kinds: dict[str, str] = {}
    excluded_columns: dict[str, str] = {}

    for column in frame.columns:
        if column in METADATA_COLUMNS or column == config.target_column:
            continue
        series = frame[column]
        if series.dropna().empty:
            excluded_columns[column] = "all_null"
            continue
        feature_columns.append(column)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            feature_kinds[column] = "numeric"
        else:
            feature_kinds[column] = "categorical"

    hover_columns = [column for column in PREFERRED_HOVER_COLUMNS if column in frame.columns]
    retained_columns = feature_columns + ([config.target_column] if config.target_column in frame.columns else [])
    return feature_columns, feature_kinds, excluded_columns, hover_columns[: config.max_hover_columns]


def prepare_feature_frame(
    frame: pd.DataFrame, feature_columns: list[str], feature_kinds: dict[str, str]
) -> pd.DataFrame:
    normalized = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        if column not in frame.columns:
            if feature_kinds.get(column) == "numeric":
                normalized[column] = pd.Series(np.nan, index=frame.index, dtype="float64")
            else:
                normalized[column] = pd.Series(pd.NA, index=frame.index, dtype="string")
            continue

        series = frame[column]
        if feature_kinds.get(column) == "numeric":
            normalized[column] = pd.to_numeric(series, errors="coerce")
        else:
            normalized[column] = series.map(_normalize_scalar).astype("string")
    return normalized


def prepare_target(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _category_indices(feature_columns: list[str], feature_kinds: dict[str, str]) -> list[int]:
    return [index for index, column in enumerate(feature_columns) if feature_kinds[column] != "numeric"]


def _validation_split(
    X: pd.DataFrame, y: pd.Series, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.Series, pd.Series | None]:
    if len(X) < 64:
        return X, None, y, None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=min(0.1, 5000 / max(len(X), 1)),
        random_state=random_seed,
    )
    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
    )


def _prediction_metrics(
    y_true: pd.Series | None, y_pred: np.ndarray | None
) -> dict[str, float | None]:
    if y_true is None or y_pred is None or len(y_true) == 0:
        return {"rmse": None, "mae": None}
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae)}


def _fit_segmenter(embeddings: np.ndarray, config: DCOVisualizeConfig) -> SegmenterModel:
    min_cluster_size = min(
        max(16, len(embeddings) // 200),
        max(config.hdbscan_min_cluster_size, 16),
    )
    min_cluster_size = max(16, min_cluster_size)
    min_samples = min(config.hdbscan_min_samples, max(8, min_cluster_size // 4))

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(embeddings)
        valid_labels = sorted(label for label in np.unique(labels) if label >= 0)
        if len(valid_labels) >= 2:
            return SegmenterModel(kind="hdbscan", model=clusterer, n_clusters=len(valid_labels))
    except Exception:
        pass

    kmeans_clusters = min(8, max(2, len(embeddings) // 4_000))
    clusterer = MiniBatchKMeans(n_clusters=kmeans_clusters, random_state=config.random_seed)
    clusterer.fit(embeddings)
    return SegmenterModel(kind="kmeans", model=clusterer, n_clusters=kmeans_clusters)


def _fit_layout(
    embeddings: np.ndarray, config: DCOVisualizeConfig
) -> tuple[Any, np.ndarray, ProjectionSpec, float]:
    if len(embeddings) < 3:
        coords = np.zeros((len(embeddings), 2), dtype=np.float32)
        projection = ProjectionSpec(name="umap", params={"n_neighbors": 2, "min_dist": config.umap_min_dist})
        return None, coords, projection, 1.0

    n_neighbors = min(config.umap_neighbors, max(2, len(embeddings) - 1))
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=config.umap_min_dist,
        densmap=True,
        random_state=config.random_seed,
    )
    coords = reducer.fit_transform(embeddings).astype(np.float32, copy=False)
    trust_neighbors = max(1, min(10, (len(embeddings) - 1) // 2))
    score = float(trustworthiness(embeddings, coords, n_neighbors=trust_neighbors))
    projection = ProjectionSpec(
        name="densmap",
        params={"n_neighbors": n_neighbors, "min_dist": config.umap_min_dist, "metric": "cosine"},
    )
    return reducer, coords, projection, score


def _fit_pretrained_branch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    categorical_feature_indices: list[int],
    config: DCOVisualizeConfig,
) -> BranchState:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    device = _runtime_device()
    LOGGER.info(
        "Fitting pretrained TabPFN branch on device=%s train_rows=%d val_rows=%d categorical_features=%d",
        device,
        len(X_train),
        0 if X_val is None else len(X_val),
        len(categorical_feature_indices),
    )
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_5,
        categorical_features_indices=categorical_feature_indices,
        device=device,
        fit_mode=config.pretrained_fit_mode,
        n_estimators=config.pretrained_n_estimators,
        random_state=config.random_seed,
        ignore_pretraining_limits=True,
    )
    model.fit(X_train, y_train)
    y_pred = None if X_val is None else model.predict(X_val)
    train_embeddings = model.get_embeddings(X_train, data_source="train")
    collapsed = collapse_embeddings(train_embeddings)
    LOGGER.info(
        "Pretrained branch fit complete: embedding_dim=%d rmse=%s mae=%s",
        collapsed.shape[1],
        _prediction_metrics(y_val, y_pred)["rmse"],
        _prediction_metrics(y_val, y_pred)["mae"],
    )
    metrics = _prediction_metrics(y_val, y_pred)
    return BranchState(
        name="pretrained",
        model=model,
        embedding_model=model,
        segmenter=_fit_segmenter(collapsed, config),
        embedding_dim=int(collapsed.shape[1]),
        prediction_metrics={
            "device": device,
            "n_estimators": config.pretrained_n_estimators,
            "version": config.tabpfn_version,
            **metrics,
        },
    )


def _fit_finetuned_branch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    categorical_feature_indices: list[int],
    config: DCOVisualizeConfig,
) -> BranchState:
    from tabpfn.finetuning import FinetunedTabPFNRegressor

    device = _runtime_device()
    LOGGER.info(
        "Fitting fine-tuned TabPFN branch on device=%s train_rows=%d val_rows=%d epochs=%d",
        device,
        len(X_train),
        0 if X_val is None else len(X_val),
        config.finetune_epochs,
    )
    model = FinetunedTabPFNRegressor(
        device=device,
        epochs=config.finetune_epochs,
        learning_rate=config.finetune_learning_rate,
        weight_decay=config.finetune_weight_decay,
        validation_split_ratio=config.finetune_validation_split_ratio,
        n_finetune_ctx_plus_query_samples=config.finetune_ctx_plus_query_samples,
        finetune_ctx_query_split_ratio=config.finetune_ctx_query_split_ratio,
        n_inference_subsample_samples=config.finetune_inference_subsample_samples,
        random_state=config.random_seed,
        early_stopping_patience=config.finetune_early_stopping_patience,
        min_delta=config.finetune_min_delta,
        n_estimators_final_inference=config.finetuned_n_estimators,
        save_checkpoint_interval=None,
        extra_regressor_kwargs={
            "model_path": _tabpfn_regressor_model_path(),
            "categorical_features_indices": categorical_feature_indices,
        },
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    embedding_model = model.finetuned_inference_regressor_
    y_pred = None if X_val is None else model.predict(X_val)
    train_embeddings = embedding_model.get_embeddings(X_train, data_source="train")
    collapsed = collapse_embeddings(train_embeddings)
    LOGGER.info(
        "Fine-tuned branch fit complete: embedding_dim=%d rmse=%s mae=%s",
        collapsed.shape[1],
        _prediction_metrics(y_val, y_pred)["rmse"],
        _prediction_metrics(y_val, y_pred)["mae"],
    )
    metrics = _prediction_metrics(y_val, y_pred)
    return BranchState(
        name="finetuned",
        model=model,
        embedding_model=embedding_model,
        segmenter=_fit_segmenter(collapsed, config),
        embedding_dim=int(collapsed.shape[1]),
        prediction_metrics={
            "device": device,
            "epochs": config.finetune_epochs,
            "n_estimators_final_inference": config.finetuned_n_estimators,
            "version": config.tabpfn_version,
            **metrics,
        },
    )


def fit_embedding_model(frame: pd.DataFrame, config: DCOVisualizeConfig) -> FitResult:
    ensure_tabpfn_runtime()
    feature_columns, feature_kinds, excluded_columns, hover_columns = infer_feature_contract(frame, config)
    LOGGER.info(
        "Preparing TabPFN fit: input_rows=%d features=%d excluded=%d target=%s",
        len(frame),
        len(feature_columns),
        len(excluded_columns),
        config.target_column,
    )
    if config.target_column not in frame.columns:
        raise ValueError(f"Target column {config.target_column!r} is missing from the training frame.")

    trainable = frame.dropna(subset=[config.target_column]).reset_index(drop=True)
    if trainable.empty:
        raise ValueError(f"Target column {config.target_column!r} has no non-null values.")

    X_full = prepare_feature_frame(trainable, feature_columns, feature_kinds)
    y_full = prepare_target(trainable[config.target_column])
    valid_mask = y_full.notna()
    X_full = X_full.loc[valid_mask].reset_index(drop=True)
    y_full = y_full.loc[valid_mask].reset_index(drop=True)
    if X_full.empty:
        raise ValueError(f"Target column {config.target_column!r} has no numeric values after normalization.")

    X_train, X_val, y_train, y_val = _validation_split(X_full, y_full, config.random_seed)
    categorical_feature_indices = _category_indices(feature_columns, feature_kinds)
    LOGGER.info(
        "Normalized fit data: train_rows=%d val_rows=%d numeric_features=%d categorical_features=%d",
        len(X_train),
        0 if X_val is None else len(X_val),
        sum(1 for kind in feature_kinds.values() if kind == "numeric"),
        len(categorical_feature_indices),
    )

    pretrained = _fit_pretrained_branch(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_feature_indices=categorical_feature_indices,
        config=config,
    )
    finetuned = _fit_finetuned_branch(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_feature_indices=categorical_feature_indices,
        config=config,
    )

    route_source_column = next(
        (column for column in [config.route_source_column, "origin_city", "origin"] if column in frame.columns),
        "origin",
    )
    route_destination_column = next(
        (column for column in [config.route_destination_column, "destination_city", "destination"] if column in frame.columns),
        "destination",
    )
    departure_date_column = next(
        (column for column in [config.departure_date_column, "departure_date"] if column in frame.columns),
        config.departure_date_column,
    )
    advance_purchase_column = (
        config.advance_purchase_column if config.advance_purchase_column in frame.columns else config.advance_purchase_column
    )
    return_date_column = (
        config.return_date_column if config.return_date_column in frame.columns else config.return_date_column
    )

    model = TabPFNEmbeddingModel(
        config=config,
        target_column=config.target_column,
        feature_columns=feature_columns,
        feature_kinds=feature_kinds,
        categorical_feature_indices=categorical_feature_indices,
        excluded_columns=excluded_columns,
        retained_columns=feature_columns + [config.target_column],
        hover_columns=hover_columns,
        pretrained=pretrained,
        finetuned=finetuned,
        route_source_column=route_source_column,
        route_destination_column=route_destination_column,
        departure_date_column=departure_date_column,
        advance_purchase_column=advance_purchase_column,
        return_date_column=return_date_column,
    )
    metrics = {
        "encoder_backend": "tabpfn_2_5",
        "embedding_extraction": "direct_get_embeddings",
        "target_column": config.target_column,
        "tabpfn_version": config.tabpfn_version,
        "pretrained_embedding_dim": pretrained.embedding_dim,
        "finetuned_embedding_dim": finetuned.embedding_dim,
        "feature_columns": feature_columns,
        "excluded_columns": excluded_columns,
        "retained_columns": model.retained_columns,
        "hover_columns": model.hover_columns,
        "pretrained": pretrained.prediction_metrics,
        "finetuned": finetuned.prediction_metrics,
    }
    LOGGER.info(
        "Completed TabPFN fit: pretrained_dim=%d finetuned_dim=%d",
        pretrained.embedding_dim,
        finetuned.embedding_dim,
    )
    return FitResult(model=model, metrics=metrics)


def _append_embedding_columns(frame: pd.DataFrame, prefix: str, embeddings: np.ndarray) -> None:
    for index in range(embeddings.shape[1]):
        frame[f"{prefix}_{index:03d}"] = embeddings[:, index]


def _route_key(row: pd.Series, source_column: str, destination_column: str) -> tuple[str, str]:
    source = str(row.get(source_column) or row.get("origin") or "unknown")
    destination = str(row.get(destination_column) or row.get("destination") or "unknown")
    return source, destination


def _advance_purchase_bucket(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "missing"
    if numeric <= 3:
        return "0-3"
    if numeric <= 7:
        return "4-7"
    if numeric <= 14:
        return "8-14"
    if numeric <= 30:
        return "15-30"
    if numeric <= 60:
        return "31-60"
    if numeric <= 90:
        return "61-90"
    return "91+"


def _prepare_departure_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    return str(_normalize_scalar(value))


def transform_frame(model: TabPFNEmbeddingModel, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = prepare_feature_frame(frame, model.feature_columns, model.feature_kinds)
    transformed = frame.copy()

    pretrained_embeddings = model.pretrained.get_embeddings(features, data_source="test")
    finetuned_embeddings = model.finetuned.get_embeddings(features, data_source="test")

    _append_embedding_columns(transformed, "pretrained_emb", pretrained_embeddings)
    _append_embedding_columns(transformed, "finetuned_emb", finetuned_embeddings)

    transformed["pretrained_segment_id"] = model.pretrained.segmenter.predict(pretrained_embeddings)
    transformed["finetuned_segment_id"] = model.finetuned.segmenter.predict(finetuned_embeddings)
    transformed["pretrained_price_pred"] = model.pretrained.predict(features)
    transformed["finetuned_price_pred"] = model.finetuned.predict(features)

    if model.target_column in transformed.columns:
        actual = prepare_target(transformed[model.target_column])
        transformed["pretrained_abs_error"] = (transformed["pretrained_price_pred"] - actual).abs()
        transformed["finetuned_abs_error"] = (transformed["finetuned_price_pred"] - actual).abs()
    else:
        transformed["pretrained_abs_error"] = np.nan
        transformed["finetuned_abs_error"] = np.nan
    transformed["price_prediction_gap"] = (
        transformed["pretrained_price_pred"] - transformed["finetuned_price_pred"]
    ).abs()

    metrics = {
        "rows": int(len(transformed)),
        "pretrained_embedding_dim": int(pretrained_embeddings.shape[1]),
        "finetuned_embedding_dim": int(finetuned_embeddings.shape[1]),
    }
    return transformed, metrics


def _update_aggregates(
    accumulator: AggregateAccumulator, frame: pd.DataFrame, model: TabPFNEmbeddingModel
) -> None:
    price_series = prepare_target(frame[model.target_column]) if model.target_column in frame.columns else pd.Series(
        np.nan, index=frame.index
    )
    route_source = model.route_source_column if model.route_source_column in frame.columns else "origin"
    route_destination = model.route_destination_column if model.route_destination_column in frame.columns else "destination"
    departure_column = (
        model.departure_date_column if model.departure_date_column in frame.columns else model.departure_date_column
    )

    working = frame.copy()
    working["_route_source"] = working[route_source].astype("string").fillna("missing")
    working["_route_destination"] = working[route_destination].astype("string").fillna("missing")
    working["_departure_value"] = working[departure_column].map(_prepare_departure_value)
    working["_advance_bucket"] = (
        working[model.advance_purchase_column].map(_advance_purchase_bucket)
        if model.advance_purchase_column in working.columns
        else "missing"
    )
    working["_price_value"] = price_series

    route_group = (
        working.groupby(["_route_source", "_route_destination"], dropna=False)["_price_value"]
        .agg(["count", "sum"])
        .reset_index()
    )
    for source, destination, count, total_price in route_group.itertuples(index=False, name=None):
        stats = accumulator.route_stats[(str(source), str(destination))]
        stats[0] += float(count)
        stats[1] += float(total_price) if not pd.isna(total_price) else 0.0

    calendar_group = (
        working.groupby(["_departure_value", "_advance_bucket"], dropna=False)["_price_value"]
        .agg(["count", "sum"])
        .reset_index()
    )
    for departure_value, advance_bucket, count, total_price in calendar_group.itertuples(index=False, name=None):
        stats = accumulator.fare_calendar[(str(departure_value), str(advance_bucket))]
        stats[0] += float(count)
        stats[1] += float(total_price) if not pd.isna(total_price) else 0.0

    agreement_group = (
        working.groupby(["pretrained_segment_id", "finetuned_segment_id"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    for row in agreement_group.itertuples(index=False):
        accumulator.agreement_counts[(int(row.pretrained_segment_id), int(row.finetuned_segment_id))] += int(row.count)

    for branch, segment_column in [("pretrained", "pretrained_segment_id"), ("finetuned", "finetuned_segment_id")]:
        segment_group = working.groupby(segment_column, dropna=False).size().reset_index(name="count")
        for row in segment_group.itertuples(index=False):
            accumulator.branch_segment_counts[(branch, int(getattr(row, segment_column)))] += int(row.count)

        for feature in FINGERPRINT_FEATURES:
            if feature not in working.columns:
                continue
            values = working[feature].astype("string").fillna("missing")
            overall_group = values.value_counts(dropna=False)
            for value, count in overall_group.items():
                accumulator.overall_feature_counts[(feature, str(value))] += int(count)

            fingerprint_group = (
                pd.DataFrame({"segment": working[segment_column], "value": values})
                .groupby(["segment", "value"], dropna=False)
                .size()
                .reset_index(name="count")
            )
            for row in fingerprint_group.itertuples(index=False):
                accumulator.branch_segment_feature_counts[
                    (branch, int(row.segment), feature, str(row.value))
                ] += int(row.count)


def _collect_viz_rows(
    frame: pd.DataFrame,
    global_indices: np.ndarray,
    batch_start: int,
    batch_end: int,
) -> pd.DataFrame:
    if len(global_indices) == 0:
        return frame.iloc[0:0].copy()
    left = int(np.searchsorted(global_indices, batch_start, side="left"))
    right = int(np.searchsorted(global_indices, batch_end, side="left"))
    if right <= left:
        return frame.iloc[0:0].copy()
    local_indices = global_indices[left:right] - batch_start
    return frame.iloc[local_indices].copy().reset_index(drop=True)


def _finalize_aggregate_frame(
    accumulator: AggregateAccumulator, model: TabPFNEmbeddingModel, total_rows: int
) -> pd.DataFrame:
    records: list[AggregateRecord] = []

    for (source, destination), (count, total_price) in accumulator.route_stats.items():
        mean_price = (total_price / count) if count else None
        records.append(
            AggregateRecord(
                view="route_network",
                branch=None,
                key_1=source,
                key_2=destination,
                key_3=None,
                segment_id=None,
                count=int(count),
                mean_price=float(mean_price) if mean_price is not None else None,
                value=None,
            )
        )
        records.append(
            AggregateRecord(
                view="market_matrix",
                branch=None,
                key_1=source,
                key_2=destination,
                key_3=None,
                segment_id=None,
                count=int(count),
                mean_price=float(mean_price) if mean_price is not None else None,
                value=None,
            )
        )

    for (departure_value, advance_bucket), (count, total_price) in accumulator.fare_calendar.items():
        mean_price = (total_price / count) if count else None
        records.append(
            AggregateRecord(
                view="fare_calendar",
                branch=None,
                key_1=departure_value,
                key_2=advance_bucket,
                key_3=None,
                segment_id=None,
                count=int(count),
                mean_price=float(mean_price) if mean_price is not None else None,
                value=None,
            )
        )

    for (branch, segment_id), count in accumulator.branch_segment_counts.items():
        records.append(
            AggregateRecord(
                view="segment_size",
                branch=branch,
                key_1=None,
                key_2=None,
                key_3=None,
                segment_id=int(segment_id),
                count=int(count),
                mean_price=None,
                value=None,
            )
        )

    for (pretrained_segment, finetuned_segment), count in accumulator.agreement_counts.items():
        records.append(
            AggregateRecord(
                view="segment_agreement",
                branch=None,
                key_1=str(pretrained_segment),
                key_2=str(finetuned_segment),
                key_3=None,
                segment_id=None,
                count=int(count),
                mean_price=None,
                value=None,
            )
        )

    top_values: dict[str, set[str]] = {}
    for feature in FINGERPRINT_FEATURES:
        ranked = [
            (value, count)
            for (candidate_feature, value), count in accumulator.overall_feature_counts.items()
            if candidate_feature == feature
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        top_values[feature] = {
            value for value, _ in ranked[: model.config.max_fingerprint_values_per_feature]
        }

    for (branch, segment_id, feature, value), count in accumulator.branch_segment_feature_counts.items():
        if value not in top_values.get(feature, set()):
            continue
        segment_total = accumulator.branch_segment_counts.get((branch, segment_id), 0)
        overall_count = accumulator.overall_feature_counts.get((feature, value), 0)
        if segment_total == 0 or overall_count == 0 or total_rows == 0:
            continue
        lift = math.log((count / segment_total + 1e-9) / (overall_count / total_rows + 1e-9))
        records.append(
            AggregateRecord(
                view="segment_fingerprint",
                branch=branch,
                key_1=feature,
                key_2=value,
                key_3=None,
                segment_id=int(segment_id),
                count=int(count),
                mean_price=None,
                value=float(lift),
            )
        )

    aggregate_frame = pd.DataFrame([asdict(record) for record in records])
    if aggregate_frame.empty:
        return pd.DataFrame(
            columns=["view", "branch", "key_1", "key_2", "key_3", "segment_id", "count", "mean_price", "value"]
        )
    return aggregate_frame.sort_values(["view", "branch", "segment_id", "count"], ascending=[True, True, True, False]).reset_index(
        drop=True
    )


def transform_parquet_file(
    model: TabPFNEmbeddingModel,
    parquet_path: str | Path,
    output_path: str | Path,
    viz_rows: int,
    batch_size: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with parquet_path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        total_rows = int(parquet.metadata.num_rows)
    LOGGER.info(
        "Starting parquet transform for %s rows=%d viz_rows=%d batch_size=%d output=%s",
        parquet_path,
        total_rows,
        viz_rows,
        batch_size,
        output_path,
    )

    viz_indices = sample_row_indices(total_rows, viz_rows, random_seed)
    viz_frames: list[pd.DataFrame] = []
    accumulator = AggregateAccumulator.create()
    writer: pq.ParquetWriter | None = None
    batch_start = 0

    try:
        with parquet_path.open("rb") as handle:
            parquet = pq.ParquetFile(handle)
            for batch_index, batch in enumerate(parquet.iter_batches(batch_size=batch_size), start=1):
                batch_frame = pa.Table.from_batches([batch]).to_pandas()
                transformed_batch, _ = transform_frame(model, batch_frame)
                _update_aggregates(accumulator, transformed_batch, model)
                viz_rows_frame = _collect_viz_rows(
                    transformed_batch,
                    global_indices=viz_indices,
                    batch_start=batch_start,
                    batch_end=batch_start + len(transformed_batch),
                )
                if not viz_rows_frame.empty:
                    viz_frames.append(viz_rows_frame)

                table = pa.Table.from_pandas(transformed_batch, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                batch_start += len(transformed_batch)
                if batch_index == 1 or batch_start == total_rows or batch_index % 10 == 0:
                    LOGGER.info(
                        "Transform progress: batches=%d rows=%d/%d viz_rows_collected=%d",
                        batch_index,
                        batch_start,
                        total_rows,
                        sum(len(frame) for frame in viz_frames),
                    )
    finally:
        if writer is not None:
            writer.close()

    viz_frame = pd.concat(viz_frames, ignore_index=True) if viz_frames else pd.DataFrame()
    if not viz_frame.empty:
        pretrained_columns = [column for column in viz_frame.columns if column.startswith("pretrained_emb_")]
        finetuned_columns = [column for column in viz_frame.columns if column.startswith("finetuned_emb_")]

        pretrained_projector, pretrained_coords, pretrained_projection, pretrained_trust = _fit_layout(
            viz_frame[pretrained_columns].to_numpy(dtype=np.float32, copy=False),
            model.config,
        )
        finetuned_projector, finetuned_coords, finetuned_projection, finetuned_trust = _fit_layout(
            viz_frame[finetuned_columns].to_numpy(dtype=np.float32, copy=False),
            model.config,
        )

        viz_frame["pretrained_layout_x"] = pretrained_coords[:, 0]
        viz_frame["pretrained_layout_y"] = pretrained_coords[:, 1]
        viz_frame["finetuned_layout_x"] = finetuned_coords[:, 0]
        viz_frame["finetuned_layout_y"] = finetuned_coords[:, 1]
        viz_frame["layout_method"] = "densmap"

        model.pretrained.projector = pretrained_projector
        model.pretrained.projection = pretrained_projection
        model.finetuned.projector = finetuned_projector
        model.finetuned.projection = finetuned_projection
    else:
        pretrained_trust = 1.0
        finetuned_trust = 1.0
        viz_frame["layout_method"] = []

    aggregate_frame = _finalize_aggregate_frame(accumulator, model, total_rows)
    metrics = {
        "embedded_rows": total_rows,
        "viz_rows": int(len(viz_frame)),
        "pretrained_embedding_dim": model.pretrained.embedding_dim,
        "finetuned_embedding_dim": model.finetuned.embedding_dim,
        "pretrained_projection": asdict(model.pretrained.projection) if model.pretrained.projection else None,
        "finetuned_projection": asdict(model.finetuned.projection) if model.finetuned.projection else None,
        "pretrained_projection_trustworthiness": float(pretrained_trust),
        "finetuned_projection_trustworthiness": float(finetuned_trust),
        "pretrained_segment_count": model.pretrained.segmenter.n_clusters,
        "finetuned_segment_count": model.finetuned.segmenter.n_clusters,
    }
    LOGGER.info(
        "Completed parquet transform: embedded_rows=%d viz_rows=%d aggregate_rows=%d pretrained_segments=%d finetuned_segments=%d",
        total_rows,
        len(viz_frame),
        len(aggregate_frame),
        model.pretrained.segmenter.n_clusters,
        model.finetuned.segmenter.n_clusters,
    )
    return viz_frame, aggregate_frame, metrics


def write_embedding_bundle(path: str | Path, model: TabPFNEmbeddingModel) -> None:
    bundle = {
        "config": model.config.to_dict(),
        "target_column": model.target_column,
        "feature_columns": model.feature_columns,
        "feature_kinds": model.feature_kinds,
        "categorical_feature_indices": model.categorical_feature_indices,
        "excluded_columns": model.excluded_columns,
        "hover_columns": model.hover_columns,
        "route_source_column": model.route_source_column,
        "route_destination_column": model.route_destination_column,
        "departure_date_column": model.departure_date_column,
        "advance_purchase_column": model.advance_purchase_column,
        "return_date_column": model.return_date_column,
        "pretrained": model.pretrained,
        "finetuned": model.finetuned,
    }
    torch.save(bundle, Path(path))
    LOGGER.info("Wrote embedding bundle to %s", path)
