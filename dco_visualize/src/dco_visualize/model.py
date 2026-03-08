from __future__ import annotations

import json
import logging
import math
import os
import time
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
from dco_visualize.progress import format_duration, progress_snapshot

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
        if self.kind == "constant":
            return np.zeros((len(embeddings),), dtype=np.int64)
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
        embeddings = collapse_embeddings(raw)
        if len(embeddings) != len(X):
            raise ValueError(
                f"{self.name} branch returned {len(embeddings)} {data_source} embeddings for {len(X)} rows."
            )
        return embeddings


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
    LOGGER.info("Using HF_TOKEN from environment for TabPFN runtime")


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


def _extract_training_embeddings(
    embedding_model: Any,
    X: pd.DataFrame,
    *,
    branch_name: str,
) -> tuple[np.ndarray, str]:
    last_error: Exception | None = None
    for data_source in ("train", "test"):
        try:
            collapsed = collapse_embeddings(embedding_model.get_embeddings(X, data_source=data_source))
            if len(collapsed) == 0:
                LOGGER.warning(
                    "%s branch returned zero %s embeddings for %d rows; trying next source",
                    branch_name,
                    data_source,
                    len(X),
                )
                continue
            if len(collapsed) != len(X):
                raise ValueError(
                    f"{branch_name} branch returned {len(collapsed)} {data_source} embeddings for {len(X)} rows."
                )
            if data_source != "train":
                LOGGER.warning(
                    "Using %s embeddings for %s branch clustering because train embeddings were unavailable",
                    data_source,
                    branch_name,
                )
            return collapsed, data_source
        except Exception as error:
            last_error = error
            LOGGER.warning(
                "Embedding extraction failed for %s branch data_source=%s rows=%d error=%s",
                branch_name,
                data_source,
                len(X),
                error,
            )
    raise RuntimeError(f"Unable to extract non-empty embeddings for {branch_name} branch") from last_error


def _slice_rows(X: Any, n_rows: int) -> Any:
    if hasattr(X, "iloc"):
        return X.iloc[:n_rows]
    return X[:n_rows]


def _smoke_validate_inference_regressor(
    regressor: Any,
    X_reference: Any,
    *,
    branch_name: str,
) -> None:
    sample_size = min(len(X_reference), 8)
    if sample_size <= 0:
        return
    X_probe = _slice_rows(X_reference, sample_size)
    regressor.predict(X_probe)
    probe_embeddings = collapse_embeddings(regressor.get_embeddings(X_probe, data_source="test"))
    if len(probe_embeddings) != sample_size:
        raise ValueError(
            f"{branch_name} smoke validation returned {len(probe_embeddings)} embeddings for {sample_size} rows."
        )


def _fit_mode_fallbacks(primary_fit_mode: str) -> list[str]:
    fallbacks = [primary_fit_mode]
    if primary_fit_mode != "fit_preprocessors":
        fallbacks.append("fit_preprocessors")
    if primary_fit_mode != "low_memory":
        fallbacks.append("low_memory")
    seen: list[str] = []
    for fit_mode in fallbacks:
        if fit_mode not in seen:
            seen.append(fit_mode)
    return seen


def _fit_regressor_with_fallback(
    factory: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    branch_name: str,
    requested_fit_mode: str,
    requested_memory_saving_mode: bool | str,
) -> tuple[Any, str, bool | str]:
    last_error: Exception | None = None
    for fit_mode in _fit_mode_fallbacks(requested_fit_mode):
        memory_saving_mode = requested_memory_saving_mode if fit_mode == requested_fit_mode else "auto"
        try:
            model = factory(fit_mode, memory_saving_mode)
            model.fit(X_train, y_train)
            if fit_mode != requested_fit_mode:
                LOGGER.warning(
                    "Using fallback fit mode for %s branch: requested=%s actual=%s memory_saving_mode=%s",
                    branch_name,
                    requested_fit_mode,
                    fit_mode,
                    memory_saving_mode,
                )
            return model, fit_mode, memory_saving_mode
        except Exception as error:
            last_error = error
            LOGGER.warning(
                "TabPFN fit mode failed for %s branch: fit_mode=%s memory_saving_mode=%s error=%s",
                branch_name,
                fit_mode,
                memory_saving_mode,
                error,
            )
    raise RuntimeError(f"Unable to fit {branch_name} branch with any supported fit mode") from last_error


def _model_branches(
    model: TabPFNEmbeddingModel,
    branches: tuple[str, ...],
) -> list[tuple[str, BranchState]]:
    branch_lookup = {
        "pretrained": model.pretrained,
        "finetuned": model.finetuned,
    }
    unknown = [branch for branch in branches if branch not in branch_lookup]
    if unknown:
        raise ValueError(f"Unknown branches requested: {unknown}")
    return [(branch, branch_lookup[branch]) for branch in branches]


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


def _numeric_target_array(series: pd.Series | np.ndarray | None) -> np.ndarray:
    if series is None:
        return np.asarray([], dtype=np.float64)
    array = pd.to_numeric(pd.Series(series), errors="coerce").dropna().to_numpy(dtype=np.float64)
    return array


def summarize_target_distribution(series: pd.Series | np.ndarray | None) -> dict[str, float | int | None]:
    numeric = _numeric_target_array(series)
    if numeric.size == 0:
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "p99": None,
            "p995": None,
            "p999": None,
            "max": None,
            "rows_gt_100k": 0,
            "rows_gt_1m": 0,
            "rows_gt_10m": 0,
        }
    return {
        "count": int(numeric.size),
        "median": float(np.quantile(numeric, 0.5)),
        "p95": float(np.quantile(numeric, 0.95)),
        "p99": float(np.quantile(numeric, 0.99)),
        "p995": float(np.quantile(numeric, 0.995)),
        "p999": float(np.quantile(numeric, 0.999)),
        "max": float(np.max(numeric)),
        "rows_gt_100k": int(np.sum(numeric > 100_000.0)),
        "rows_gt_1m": int(np.sum(numeric > 1_000_000.0)),
        "rows_gt_10m": int(np.sum(numeric > 10_000_000.0)),
    }


def _winsor_bounds(
    series: pd.Series | np.ndarray | None,
    upper_quantile: float,
) -> tuple[float | None, float | None]:
    numeric = _numeric_target_array(series)
    if numeric.size == 0:
        return None, None
    if upper_quantile >= 1.0:
        return float(np.min(numeric)), float(np.max(numeric))
    lower_quantile = max(0.0, 1.0 - upper_quantile)
    lower = float(np.quantile(numeric, lower_quantile))
    upper = float(np.quantile(numeric, upper_quantile))
    return lower, upper


def _winsorized_prediction_metrics(
    y_true: pd.Series | np.ndarray | None,
    y_pred: np.ndarray | None,
    *,
    lower: float | None,
    upper: float | None,
) -> dict[str, float | None]:
    if y_true is None or y_pred is None:
        return {"winsorized_rmse": None, "winsorized_mae": None}
    y_true_array = _numeric_target_array(y_true)
    y_pred_array = np.asarray(y_pred, dtype=np.float64)
    if y_true_array.size == 0 or y_true_array.size != y_pred_array.size:
        return {"winsorized_rmse": None, "winsorized_mae": None}
    if lower is None or upper is None:
        clipped_true = y_true_array
        clipped_pred = y_pred_array
    else:
        clipped_true = np.clip(y_true_array, lower, upper)
        clipped_pred = np.clip(y_pred_array, lower, upper)
    rmse = math.sqrt(mean_squared_error(clipped_true, clipped_pred))
    mae = mean_absolute_error(clipped_true, clipped_pred)
    return {"winsorized_rmse": float(rmse), "winsorized_mae": float(mae)}


def _log_target_distribution(series: pd.Series | np.ndarray | None, label: str) -> dict[str, float | int | None]:
    stats = summarize_target_distribution(series)
    LOGGER.info(
        "%s target distribution: count=%s median=%s p95=%s p99=%s p99.5=%s p99.9=%s max=%s rows_gt_100k=%s rows_gt_1m=%s rows_gt_10m=%s",
        label,
        stats["count"],
        stats["median"],
        stats["p95"],
        stats["p99"],
        stats["p995"],
        stats["p999"],
        stats["max"],
        stats["rows_gt_100k"],
        stats["rows_gt_1m"],
        stats["rows_gt_10m"],
    )
    return stats


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


def _build_branch_metrics(
    *,
    base_metrics: dict[str, float | None],
    winsorized_metrics: dict[str, float | None],
    winsor_lower: float | None,
    winsor_upper: float | None,
    target_stats: dict[str, float | int | None],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        **extra,
        **base_metrics,
        **winsorized_metrics,
        "eval_winsor_lower": winsor_lower,
        "eval_winsor_upper": winsor_upper,
        "target_distribution": target_stats,
    }


def _fit_segmenter(embeddings: np.ndarray, config: DCOVisualizeConfig) -> SegmenterModel:
    if len(embeddings) <= 1:
        LOGGER.warning(
            "Using constant segmenter because only %d embedding rows are available",
            len(embeddings),
        )
        return SegmenterModel(kind="constant", model=None, n_clusters=1)

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
    winsor_lower: float | None,
    winsor_upper: float | None,
    target_stats: dict[str, float | int | None],
    config: DCOVisualizeConfig,
) -> BranchState:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    device = _runtime_device()
    started_at = time.perf_counter()
    LOGGER.info(
        "Fitting pretrained TabPFN branch on device=%s train_rows=%d val_rows=%d categorical_features=%d requested_fit_mode=%s n_estimators=%d n_preprocessing_jobs=%d",
        device,
        len(X_train),
        0 if X_val is None else len(X_val),
        len(categorical_feature_indices),
        config.pretrained_fit_mode,
        config.pretrained_n_estimators,
        config.n_preprocessing_jobs,
    )
    model, fit_mode_used, memory_saving_mode_used = _fit_regressor_with_fallback(
        lambda fit_mode, memory_saving_mode: TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5,
            categorical_features_indices=categorical_feature_indices,
            device=device,
            fit_mode=fit_mode,
            memory_saving_mode=memory_saving_mode,
            n_estimators=config.pretrained_n_estimators,
            n_preprocessing_jobs=config.n_preprocessing_jobs,
            random_state=config.random_seed,
            ignore_pretraining_limits=True,
        ),
        X_train,
        y_train,
        branch_name="pretrained",
        requested_fit_mode=config.pretrained_fit_mode,
        requested_memory_saving_mode=config.pretrained_memory_saving_mode,
    )
    y_pred = None if X_val is None else model.predict(X_val)
    collapsed, embedding_source = _extract_training_embeddings(
        model,
        X_train,
        branch_name="pretrained",
    )
    base_metrics = _prediction_metrics(y_val, y_pred)
    winsorized_metrics = _winsorized_prediction_metrics(
        y_val,
        y_pred,
        lower=winsor_lower,
        upper=winsor_upper,
    )
    LOGGER.info(
        "Pretrained branch fit complete: embedding_dim=%d rmse=%s mae=%s winsorized_rmse=%s winsorized_mae=%s fit_mode=%s memory_saving_mode=%s clustering_source=%s elapsed=%s",
        collapsed.shape[1],
        base_metrics["rmse"],
        base_metrics["mae"],
        winsorized_metrics["winsorized_rmse"],
        winsorized_metrics["winsorized_mae"],
        fit_mode_used,
        memory_saving_mode_used,
        embedding_source,
        format_duration(time.perf_counter() - started_at),
    )
    return BranchState(
        name="pretrained",
        model=model,
        embedding_model=model,
        segmenter=_fit_segmenter(collapsed, config),
        embedding_dim=int(collapsed.shape[1]),
        prediction_metrics=_build_branch_metrics(
            base_metrics=base_metrics,
            winsorized_metrics=winsorized_metrics,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            target_stats=target_stats,
            extra={
                "device": device,
                "n_estimators": config.pretrained_n_estimators,
                "fit_mode": fit_mode_used,
                "memory_saving_mode": memory_saving_mode_used,
                "clustering_embedding_source": embedding_source,
                "version": config.tabpfn_version,
            },
        ),
    )


def _fit_finetuned_branch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    categorical_feature_indices: list[int],
    winsor_lower: float | None,
    winsor_upper: float | None,
    target_stats: dict[str, float | int | None],
    config: DCOVisualizeConfig,
) -> BranchState:
    from tabpfn import TabPFNRegressor
    from tabpfn.finetuning import FinetunedTabPFNRegressor
    from tabpfn.finetuning.finetuned_base import EvalResult
    from tabpfn.finetuning.train_util import clone_model_for_evaluation

    final_inference_requested_fit_mode = config.finetune_inference_fit_mode
    final_inference_requested_memory_saving = config.finetune_inference_memory_saving_mode

    class DCORobustFinetunedTabPFNRegressor(FinetunedTabPFNRegressor):
        def __init__(self, *args: Any, eval_winsor_quantile: float, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.eval_winsor_quantile = eval_winsor_quantile
            self.final_inference_fit_mode_used_: str | None = None
            self.final_inference_memory_saving_mode_used_: bool | str | None = None

        @property
        def _metric_name(self) -> str:  # type: ignore[override]
            return "winsorized_MSE"

        def _evaluate_model(  # type: ignore[override]
            self,
            eval_config: dict[str, Any],
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
        ) -> EvalResult:
            eval_regressor = clone_model_for_evaluation(
                self.finetuned_estimator_,
                eval_config,
                TabPFNRegressor,
            )
            eval_regressor.fit(X_train, y_train)

            try:
                predictions = eval_regressor.predict(X_val)  # type: ignore[assignment]
                raw_mse = mean_squared_error(y_val, predictions)
                raw_mae = mean_absolute_error(y_val, predictions)
                lower, upper = _winsor_bounds(y_train, self.eval_winsor_quantile)
                clipped_true = np.clip(np.asarray(y_val, dtype=np.float64), lower, upper)
                clipped_pred = np.clip(np.asarray(predictions, dtype=np.float64), lower, upper)
                winsorized_mse = mean_squared_error(clipped_true, clipped_pred)
            except (ValueError, RuntimeError, AttributeError) as error:
                LOGGER.warning("An error occurred during fine-tuned evaluation: %s", error)
                raw_mse = np.nan
                raw_mae = np.nan
                lower = np.nan
                upper = np.nan
                winsorized_mse = np.nan

            return EvalResult(
                primary=winsorized_mse,
                secondary={
                    "raw_mse": raw_mse,
                    "raw_mae": raw_mae,
                    "eval_winsor_lower": lower,
                    "eval_winsor_upper": upper,
                },
            )

        def _get_checkpoint_metrics(self, eval_result: EvalResult) -> dict[str, float]:  # type: ignore[override]
            return {
                "winsorized_mse": float(eval_result.primary),
                **{key: float(value) for key, value in eval_result.secondary.items()},
            }

        def _log_epoch_evaluation(  # type: ignore[override]
            self,
            epoch: int,
            eval_result: EvalResult,
            mean_train_loss: float | None,
        ) -> None:
            mean_train_loss_label = "N/A" if mean_train_loss is None else f"{mean_train_loss:.4f}"
            LOGGER.info(
                "Epoch %d Evaluation | Val winsorized MSE: %.4f | Raw MSE: %.4f | Raw MAE: %.4f | Train Loss: %s",
                epoch + 1,
                eval_result.primary,
                eval_result.secondary.get("raw_mse", np.nan),
                eval_result.secondary.get("raw_mae", np.nan),
                mean_train_loss_label,
            )

        def _setup_inference_model(  # type: ignore[override]
            self, final_inference_eval_config: dict[str, Any]
        ) -> None:
            last_error: Exception | None = None
            for fit_mode in _fit_mode_fallbacks(final_inference_requested_fit_mode):
                memory_saving_mode = (
                    final_inference_requested_memory_saving
                    if fit_mode == final_inference_requested_fit_mode
                    else "auto"
                )
                try:
                    finetuned_inference_regressor = clone_model_for_evaluation(
                        self.finetuned_estimator_,
                        final_inference_eval_config,
                        TabPFNRegressor,
                    )
                    self.finetuned_inference_regressor_ = finetuned_inference_regressor
                    self.finetuned_inference_regressor_.fit_mode = fit_mode  # type: ignore[attr-defined]
                    self.finetuned_inference_regressor_.memory_saving_mode = memory_saving_mode  # type: ignore[attr-defined]
                    self.finetuned_inference_regressor_.n_preprocessing_jobs = config.n_preprocessing_jobs  # type: ignore[attr-defined]
                    self.finetuned_inference_regressor_.fit(self.X_, self.y_)  # type: ignore[arg-type]
                    _smoke_validate_inference_regressor(
                        self.finetuned_inference_regressor_,
                        self.X_,
                        branch_name="finetuned",
                    )
                    self.final_inference_fit_mode_used_ = fit_mode
                    self.final_inference_memory_saving_mode_used_ = memory_saving_mode
                    if fit_mode != final_inference_requested_fit_mode:
                        LOGGER.warning(
                            "Using fallback fit mode for fine-tuned inference regressor: requested=%s actual=%s memory_saving_mode=%s",
                            final_inference_requested_fit_mode,
                            fit_mode,
                            memory_saving_mode,
                        )
                    return
                except Exception as error:
                    last_error = error
                    LOGGER.warning(
                        "Fine-tuned inference regressor setup failed: fit_mode=%s memory_saving_mode=%s error=%s",
                        fit_mode,
                        memory_saving_mode,
                        error,
                    )
            raise RuntimeError("Unable to build fine-tuned inference regressor") from last_error

    device = _runtime_device()
    started_at = time.perf_counter()
    LOGGER.info(
        "Fitting fine-tuned TabPFN branch on device=%s train_rows=%d val_rows=%d epochs=%d final_inference_fit_mode=%s n_estimators_finetune=%d n_estimators_validation=%d n_estimators_final_inference=%d",
        device,
        len(X_train),
        0 if X_val is None else len(X_val),
        config.finetune_epochs,
        config.finetune_inference_fit_mode,
        config.finetune_n_estimators_finetune,
        config.finetune_n_estimators_validation,
        config.finetuned_n_estimators,
    )
    model = DCORobustFinetunedTabPFNRegressor(
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
        n_estimators_finetune=config.finetune_n_estimators_finetune,
        n_estimators_validation=config.finetune_n_estimators_validation,
        n_estimators_final_inference=config.finetuned_n_estimators,
        save_checkpoint_interval=None,
        eval_winsor_quantile=config.finetune_eval_winsor_quantile,
        extra_regressor_kwargs={
            "model_path": _tabpfn_regressor_model_path(),
            "categorical_features_indices": categorical_feature_indices,
        },
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    embedding_model = model.finetuned_inference_regressor_
    y_pred = None if X_val is None else model.predict(X_val)
    collapsed, embedding_source = _extract_training_embeddings(
        embedding_model,
        X_train,
        branch_name="finetuned",
    )
    base_metrics = _prediction_metrics(y_val, y_pred)
    winsorized_metrics = _winsorized_prediction_metrics(
        y_val,
        y_pred,
        lower=winsor_lower,
        upper=winsor_upper,
    )
    LOGGER.info(
        "Fine-tuned branch fit complete: embedding_dim=%d rmse=%s mae=%s winsorized_rmse=%s winsorized_mae=%s inference_fit_mode=%s inference_memory_saving_mode=%s clustering_source=%s elapsed=%s",
        collapsed.shape[1],
        base_metrics["rmse"],
        base_metrics["mae"],
        winsorized_metrics["winsorized_rmse"],
        winsorized_metrics["winsorized_mae"],
        model.final_inference_fit_mode_used_,
        model.final_inference_memory_saving_mode_used_,
        embedding_source,
        format_duration(time.perf_counter() - started_at),
    )
    return BranchState(
        name="finetuned",
        model=model,
        embedding_model=embedding_model,
        segmenter=_fit_segmenter(collapsed, config),
        embedding_dim=int(collapsed.shape[1]),
        prediction_metrics=_build_branch_metrics(
            base_metrics=base_metrics,
            winsorized_metrics=winsorized_metrics,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            target_stats=target_stats,
            extra={
                "device": device,
                "epochs": config.finetune_epochs,
                "n_estimators_finetune": config.finetune_n_estimators_finetune,
                "n_estimators_validation": config.finetune_n_estimators_validation,
                "n_estimators_final_inference": config.finetuned_n_estimators,
                "version": config.tabpfn_version,
                "eval_metric": "winsorized_mse",
                "fit_mode": model.final_inference_fit_mode_used_,
                "memory_saving_mode": model.final_inference_memory_saving_mode_used_,
                "clustering_embedding_source": embedding_source,
            },
        ),
    )


def fit_embedding_model(frame: pd.DataFrame, config: DCOVisualizeConfig) -> FitResult:
    started_at = time.perf_counter()
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
    target_stats = _log_target_distribution(y_train, "Training")
    winsor_lower, winsor_upper = _winsor_bounds(y_train, config.finetune_eval_winsor_quantile)
    LOGGER.info(
        "Normalized fit data: train_rows=%d val_rows=%d numeric_features=%d categorical_features=%d eval_winsor_bounds=(%s, %s)",
        len(X_train),
        0 if X_val is None else len(X_val),
        sum(1 for kind in feature_kinds.values() if kind == "numeric"),
        len(categorical_feature_indices),
        winsor_lower,
        winsor_upper,
    )

    pretrained = _fit_pretrained_branch(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_feature_indices=categorical_feature_indices,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        target_stats=target_stats,
        config=config,
    )
    finetuned = _fit_finetuned_branch(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_feature_indices=categorical_feature_indices,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        target_stats=target_stats,
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
        "Completed TabPFN fit: pretrained_dim=%d finetuned_dim=%d elapsed=%s",
        pretrained.embedding_dim,
        finetuned.embedding_dim,
        format_duration(time.perf_counter() - started_at),
    )
    return FitResult(model=model, metrics=metrics)


def _embedding_frame(prefix: str, embeddings: np.ndarray, index: pd.Index) -> pd.DataFrame:
    columns = [f"{prefix}_{position:03d}" for position in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=columns, index=index)


def _derived_output_field(column: str, series: pd.Series) -> pa.Field:
    if column.startswith(("pretrained_emb_", "finetuned_emb_")):
        return pa.field(column, pa.float32())
    if column.endswith("_segment_id"):
        return pa.field(column, pa.int64())
    if column in {"pretrained_price_pred", "finetuned_price_pred", "price_prediction_gap"}:
        return pa.field(column, pa.float32())
    if column.endswith("_abs_error"):
        return pa.field(column, pa.float64())
    if pd.api.types.is_bool_dtype(series):
        return pa.field(column, pa.bool_())
    if pd.api.types.is_integer_dtype(series):
        return pa.field(column, pa.int64())
    if pd.api.types.is_float_dtype(series):
        return pa.field(column, pa.float64())
    return pa.field(column, pa.string())


def _transformed_output_schema(frame: pd.DataFrame, source_schema: pa.Schema) -> pa.Schema:
    source_fields = {field.name: field for field in source_schema}
    fields: list[pa.Field] = []
    for column in frame.columns:
        if column in source_fields:
            fields.append(source_fields[column])
        else:
            fields.append(_derived_output_field(column, frame[column]))
    return pa.schema(fields)


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


def _deduplicate_feature_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if features.empty:
        return features, np.empty((0,), dtype=np.int64)
    row_hashes = pd.util.hash_pandas_object(features, index=False).to_numpy(dtype=np.uint64, copy=False)
    _, first_positions, inverse = np.unique(row_hashes, return_index=True, return_inverse=True)
    order = np.argsort(first_positions)
    ordered_first_positions = first_positions[order]
    reorder = np.empty(len(order), dtype=np.int64)
    reorder[order] = np.arange(len(order), dtype=np.int64)
    inverse_positions = reorder[inverse]
    unique_features = features.iloc[ordered_first_positions].reset_index(drop=True)
    return unique_features, inverse_positions


def transform_frame(
    model: TabPFNEmbeddingModel,
    frame: pd.DataFrame,
    *,
    branches: tuple[str, ...] = ("pretrained", "finetuned"),
    include_predictions: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = prepare_feature_frame(frame, model.feature_columns, model.feature_kinds)
    transformed = frame.copy()
    derived_frames: list[pd.DataFrame] = []
    branch_predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {"rows": int(len(transformed)), "branches": list(branches)}
    unique_features, inverse_positions = _deduplicate_feature_frame(features)
    metrics["unique_feature_rows"] = int(len(unique_features))
    metrics["duplicate_feature_rows"] = int(len(features) - len(unique_features))
    metrics["duplicate_feature_fraction"] = (
        float((len(features) - len(unique_features)) / len(features)) if len(features) else 0.0
    )

    for branch_name, branch_state in _model_branches(model, branches):
        branch_embeddings_unique = branch_state.get_embeddings(unique_features, data_source="test")
        branch_embeddings = branch_embeddings_unique[inverse_positions]
        derived_frames.append(_embedding_frame(f"{branch_name}_emb", branch_embeddings, transformed.index))
        derived_frames.append(
            pd.DataFrame(
                {f"{branch_name}_segment_id": branch_state.segmenter.predict(branch_embeddings)},
                index=transformed.index,
            )
        )
        metrics[f"{branch_name}_embedding_dim"] = int(branch_embeddings.shape[1])
        if include_predictions:
            branch_predictions_unique = branch_state.predict(unique_features)
            branch_predictions[branch_name] = branch_predictions_unique[inverse_positions]
            derived_frames.append(
                pd.DataFrame(
                    {f"{branch_name}_price_pred": branch_predictions[branch_name]},
                    index=transformed.index,
                )
            )

    if derived_frames:
        transformed = pd.concat([transformed, pd.concat(derived_frames, axis=1)], axis=1)

    if include_predictions:
        if model.target_column in transformed.columns:
            actual = prepare_target(transformed[model.target_column])
            for branch_name in branch_predictions:
                transformed[f"{branch_name}_abs_error"] = (
                    transformed[f"{branch_name}_price_pred"] - actual
                ).abs()
        else:
            for branch_name in branch_predictions:
                transformed[f"{branch_name}_abs_error"] = np.nan
        if {"pretrained", "finetuned"} <= set(branch_predictions):
            transformed["price_prediction_gap"] = (
                transformed["pretrained_price_pred"] - transformed["finetuned_price_pred"]
            ).abs()
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

    if {"pretrained_segment_id", "finetuned_segment_id"} <= set(working.columns):
        agreement_group = (
            working.groupby(["pretrained_segment_id", "finetuned_segment_id"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        for row in agreement_group.itertuples(index=False):
            accumulator.agreement_counts[(int(row.pretrained_segment_id), int(row.finetuned_segment_id))] += int(row.count)

    for branch, segment_column in [("pretrained", "pretrained_segment_id"), ("finetuned", "finetuned_segment_id")]:
        if segment_column not in working.columns:
            continue
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


def aggregate_parquet_file(
    model: TabPFNEmbeddingModel,
    parquet_path: str | Path,
    batch_size: int,
) -> pd.DataFrame:
    parquet_path = Path(parquet_path)
    started_at = time.perf_counter()
    with parquet_path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        total_rows = int(parquet.metadata.num_rows)

    LOGGER.info(
        "Starting aggregate pass for %s rows=%d batch_size=%d",
        parquet_path,
        total_rows,
        batch_size,
    )
    accumulator = AggregateAccumulator.create()
    processed_rows = 0
    total_batches = max(1, math.ceil(total_rows / batch_size))

    with parquet_path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch_index, batch in enumerate(parquet.iter_batches(batch_size=batch_size), start=1):
            frame = pa.Table.from_batches([batch]).to_pandas()
            _update_aggregates(accumulator, frame, model)
            processed_rows += batch.num_rows
            if (
                batch_index <= 2
                or processed_rows == total_rows
                or batch_index % model.config.progress_log_every_batches == 0
            ):
                row_snapshot = progress_snapshot(processed_rows, total_rows, started_at)
                batch_snapshot = progress_snapshot(batch_index, total_batches, started_at)
                LOGGER.info(
                    "Aggregate progress: batches=%d/%d batch_pct=%.1f rows=%d/%d row_pct=%.1f elapsed=%s row_rate=%.0f rows/s remaining=%s eta_utc=%s",
                    batch_snapshot.done,
                    batch_snapshot.total,
                    batch_snapshot.percent,
                    row_snapshot.done,
                    row_snapshot.total,
                    row_snapshot.percent,
                    format_duration(row_snapshot.elapsed_seconds),
                    row_snapshot.rate_per_second,
                    format_duration(row_snapshot.remaining_seconds),
                    row_snapshot.eta_utc or "unknown",
                )

    aggregate_frame = _finalize_aggregate_frame(accumulator, model, total_rows)
    LOGGER.info(
        "Completed aggregate pass: aggregate_rows=%d elapsed=%s",
        len(aggregate_frame),
        format_duration(time.perf_counter() - started_at),
    )
    return aggregate_frame


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
    started_at = time.perf_counter()

    with parquet_path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        total_rows = int(parquet.metadata.num_rows)
        source_schema = parquet.schema_arrow
    total_batches = max(1, math.ceil(total_rows / batch_size))
    LOGGER.info(
        "Starting parquet transform for %s rows=%d viz_rows=%d batch_size=%d estimated_batches=%d output=%s embedding_branches=%s viz_branches=%s",
        parquet_path,
        total_rows,
        viz_rows,
        batch_size,
        total_batches,
        output_path,
        ["finetuned"],
        ["pretrained", "finetuned"],
    )

    viz_indices = sample_row_indices(total_rows, viz_rows, random_seed)
    viz_raw_frames: list[pd.DataFrame] = []
    accumulator = AggregateAccumulator.create()
    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    batch_start = 0

    cumulative_unique_rows = 0
    cumulative_duplicate_rows = 0

    try:
        with parquet_path.open("rb") as handle:
            parquet = pq.ParquetFile(handle)
            for batch_index, batch in enumerate(parquet.iter_batches(batch_size=batch_size), start=1):
                batch_frame = pa.Table.from_batches([batch]).to_pandas()
                viz_rows_frame = _collect_viz_rows(
                    batch_frame,
                    global_indices=viz_indices,
                    batch_start=batch_start,
                    batch_end=batch_start + len(batch_frame),
                )
                if not viz_rows_frame.empty:
                    viz_raw_frames.append(viz_rows_frame)

                transformed_batch, batch_metrics = transform_frame(
                    model,
                    batch_frame,
                    branches=("finetuned",),
                    include_predictions=False,
                )
                cumulative_unique_rows += int(batch_metrics.get("unique_feature_rows", len(batch_frame)))
                cumulative_duplicate_rows += int(batch_metrics.get("duplicate_feature_rows", 0))
                _update_aggregates(accumulator, transformed_batch, model)

                if output_schema is None:
                    output_schema = _transformed_output_schema(transformed_batch, source_schema)
                aligned_batch = transformed_batch.reindex(columns=output_schema.names)
                table = pa.Table.from_pandas(
                    aligned_batch,
                    schema=output_schema,
                    preserve_index=False,
                    safe=False,
                )
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                batch_start += len(transformed_batch)
                if (
                    batch_index <= 2
                    or batch_start == total_rows
                    or batch_index % model.config.progress_log_every_batches == 0
                ):
                    row_snapshot = progress_snapshot(batch_start, total_rows, started_at)
                    batch_snapshot = progress_snapshot(batch_index, total_batches, started_at)
                    LOGGER.info(
                        "Transform progress: batches=%d/%d batch_pct=%.1f rows=%d/%d row_pct=%.1f viz_rows_collected=%d deduped_rows=%d duplicate_rows=%d elapsed=%s row_rate=%.0f rows/s remaining=%s eta_utc=%s",
                        batch_snapshot.done,
                        batch_snapshot.total,
                        batch_snapshot.percent,
                        row_snapshot.done,
                        row_snapshot.total,
                        row_snapshot.percent,
                        sum(len(frame) for frame in viz_raw_frames),
                        cumulative_unique_rows,
                        cumulative_duplicate_rows,
                        format_duration(row_snapshot.elapsed_seconds),
                        row_snapshot.rate_per_second,
                        format_duration(row_snapshot.remaining_seconds),
                        row_snapshot.eta_utc or "unknown",
                    )
    finally:
        if writer is not None:
            writer.close()

    viz_raw_frame = pd.concat(viz_raw_frames, ignore_index=True) if viz_raw_frames else pd.DataFrame()
    LOGGER.info(
        "Completed embedding parquet stage: embedded_rows=%d viz_rows_buffered=%d elapsed=%s",
        total_rows,
        len(viz_raw_frame),
        format_duration(time.perf_counter() - started_at),
    )
    viz_frame = pd.DataFrame()
    if not viz_raw_frame.empty:
        viz_started_at = time.perf_counter()
        LOGGER.info("Starting viz sample transform and projection for %d rows", len(viz_raw_frame))
        viz_frame, _ = transform_frame(
            model,
            viz_raw_frame,
            branches=("pretrained", "finetuned"),
            include_predictions=False,
        )
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

        viz_accumulator = AggregateAccumulator.create()
        _update_aggregates(viz_accumulator, viz_frame, model)
        viz_aggregate_frame = _finalize_aggregate_frame(viz_accumulator, model, len(viz_frame))
        LOGGER.info(
            "Completed viz sample transform and projection: viz_rows=%d aggregate_rows=%d elapsed=%s",
            len(viz_frame),
            len(viz_aggregate_frame),
            format_duration(time.perf_counter() - viz_started_at),
        )
    else:
        pretrained_trust = 1.0
        finetuned_trust = 1.0
        viz_frame["layout_method"] = []
        viz_aggregate_frame = pd.DataFrame()

    aggregate_frame = _finalize_aggregate_frame(accumulator, model, total_rows)
    if not viz_aggregate_frame.empty:
        viz_only_views = viz_aggregate_frame[
            (viz_aggregate_frame["view"] == "segment_agreement")
            | (
                (viz_aggregate_frame["branch"] == "pretrained")
                & viz_aggregate_frame["view"].isin(["segment_size", "segment_fingerprint"])
            )
        ]
        if not viz_only_views.empty:
            aggregate_frame = pd.concat([aggregate_frame, viz_only_views], ignore_index=True)
    metrics = {
        "embedded_rows": total_rows,
        "viz_rows": int(len(viz_frame)),
        "full_day_unique_feature_rows": int(cumulative_unique_rows),
        "full_day_duplicate_feature_rows": int(cumulative_duplicate_rows),
        "duplicate_feature_fraction": (
            float(cumulative_duplicate_rows / max(cumulative_unique_rows + cumulative_duplicate_rows, 1))
        ),
        "finetuned_embedding_dim": model.finetuned.embedding_dim,
        "pretrained_embedding_dim": model.pretrained.embedding_dim,
        "pretrained_projection": asdict(model.pretrained.projection) if model.pretrained.projection else None,
        "finetuned_projection": asdict(model.finetuned.projection) if model.finetuned.projection else None,
        "pretrained_projection_trustworthiness": float(pretrained_trust),
        "finetuned_projection_trustworthiness": float(finetuned_trust),
        "finetuned_segment_count": model.finetuned.segmenter.n_clusters,
        "pretrained_segment_count": model.pretrained.segmenter.n_clusters,
        "full_day_embedding_branches": ["finetuned"],
        "viz_embedding_branches": ["pretrained", "finetuned"],
    }
    LOGGER.info(
        "Completed parquet transform: embedded_rows=%d viz_rows=%d aggregate_rows=%d pretrained_segments=%d finetuned_segments=%d elapsed=%s",
        total_rows,
        len(viz_frame),
        len(aggregate_frame),
        model.pretrained.segmenter.n_clusters,
        model.finetuned.segmenter.n_clusters,
        format_duration(time.perf_counter() - started_at),
    )
    return viz_frame, aggregate_frame, metrics


def write_embedding_bundle(path: str | Path, model: TabPFNEmbeddingModel) -> None:
    def branch_bundle(branch: BranchState) -> dict[str, Any]:
        return {
            "name": branch.name,
            "embedding_dim": branch.embedding_dim,
            "prediction_metrics": branch.prediction_metrics,
            "projection": asdict(branch.projection) if branch.projection else None,
            "segmenter": branch.segmenter,
            "segment_labels": branch.segment_labels,
        }

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
        "pretrained": branch_bundle(model.pretrained),
        "finetuned": branch_bundle(model.finetuned),
    }
    torch.save(bundle, Path(path))
    LOGGER.info("Wrote embedding bundle to %s", path)
