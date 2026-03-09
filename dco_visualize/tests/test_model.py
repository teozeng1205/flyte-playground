from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import make_dco_frame
from dco_visualize.config import DCOVisualizeConfig
from dco_visualize.model import (
    BranchState,
    FitResult,
    SegmenterModel,
    TabPFNEmbeddingModel,
    _extract_training_embeddings,
    _fit_layout,
    _prediction_metrics,
    _winsor_bounds,
    _winsorized_prediction_metrics,
    aggregate_parquet_file,
    fit_embedding_model,
    summarize_target_distribution,
    transform_parquet_file,
)


class FakePredictor:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = np.arange(len(X), dtype=np.float32)
        return base + self.offset

    def get_embeddings(self, X: pd.DataFrame, data_source: str = "test") -> np.ndarray:
        values = np.arange(len(X), dtype=np.float32) + self.offset
        embeddings = np.stack(
            [
                values,
                values * 0.5 + 1.0,
                values * 0.25 + 2.0,
            ],
            axis=1,
        )
        return np.expand_dims(embeddings, axis=0)


class FakeClusterer:
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return (embeddings[:, 0] >= np.median(embeddings[:, 0])).astype(np.int64)


class EmptyTrainEmbeddingPredictor(FakePredictor):
    def get_embeddings(self, X: pd.DataFrame, data_source: str = "test") -> np.ndarray:
        if data_source == "train":
            return np.zeros((1, 0, 192), dtype=np.float32)
        return super().get_embeddings(X, data_source=data_source)


def make_fake_branch(name: str, offset: float) -> BranchState:
    predictor = FakePredictor(offset=offset)
    return BranchState(
        name=name,
        model=predictor,
        embedding_model=predictor,
        segmenter=SegmenterModel(kind="kmeans", model=FakeClusterer(), n_clusters=2),
        embedding_dim=3,
        prediction_metrics={"device": "cpu", "rmse": 1.23, "mae": 0.98},
    )


def test_fit_embedding_model_produces_dual_branch_contract(monkeypatch) -> None:
    frame = make_dco_frame(rows=18)
    config = DCOVisualizeConfig(sample_rows=len(frame), train_rows=len(frame), viz_rows=10)

    monkeypatch.setattr(
        "dco_visualize.model._fit_pretrained_branch",
        lambda **kwargs: make_fake_branch("pretrained", offset=1.0),
    )
    monkeypatch.setattr(
        "dco_visualize.model._fit_finetuned_branch",
        lambda **kwargs: make_fake_branch("finetuned", offset=2.0),
    )

    result = fit_embedding_model(frame, config)

    assert result.metrics["encoder_backend"] == "tabpfn_2_5"
    assert result.metrics["embedding_extraction"] == "direct_get_embeddings"
    assert result.metrics["target_column"] == "price_inc"
    assert result.metrics["feature_columns"]
    assert result.model.pretrained.embedding_dim == 3
    assert result.model.finetuned.embedding_dim == 3
    assert "origin_metro" in result.model.feature_columns
    assert "price_inc" not in result.model.feature_columns


def test_transform_parquet_file_emits_branch_embeddings_and_views(tmp_path) -> None:
    frame = make_dco_frame(rows=20)
    config = DCOVisualizeConfig(sample_rows=len(frame), train_rows=12, viz_rows=8)
    model = TabPFNEmbeddingModel(
        config=config,
        target_column="price_inc",
        feature_columns=[column for column in frame.columns if column not in {"price_inc", "row_id", "source_uri", "source_row_number", "customer", "sales_date"}],
        feature_kinds={
            column: ("numeric" if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column]) else "categorical")
            for column in frame.columns
            if column not in {"price_inc", "row_id", "source_uri", "source_row_number", "customer", "sales_date"}
        },
        categorical_feature_indices=[],
        excluded_columns={},
        retained_columns=list(frame.columns),
        hover_columns=["carrier", "origin_metro", "destination_metro"],
        pretrained=make_fake_branch("pretrained", offset=1.0),
        finetuned=make_fake_branch("finetuned", offset=2.0),
        route_source_column="origin_metro",
        route_destination_column="destination_metro",
        departure_date_column="outbound_departure_date",
        advance_purchase_column="advance_purchase",
        return_date_column="inbound_departure_date",
    )

    input_path = tmp_path / "sample.parquet"
    output_path = tmp_path / "embeddings.parquet"
    frame.to_parquet(input_path, index=False)

    viz_frame, aggregate_frame, metrics = transform_parquet_file(
        model=model,
        parquet_path=input_path,
        output_path=output_path,
        viz_rows=config.viz_rows,
        batch_size=5,
        random_seed=config.random_seed,
    )

    transformed = pd.read_parquet(output_path)
    assert len(transformed) == len(frame)
    assert {"finetuned_emb_000", "finetuned_segment_id"} <= set(transformed.columns)
    assert "pretrained_emb_000" not in transformed.columns
    assert "pretrained_segment_id" not in transformed.columns
    assert {"pretrained_layout_x", "pretrained_layout_y", "finetuned_layout_x", "finetuned_layout_y", "layout_method"} <= set(
        viz_frame.columns
    )
    assert viz_frame["layout_method"].eq("densmap").all()
    assert metrics["embedded_rows"] == len(frame)
    assert metrics["full_day_embedding_branches"] == ["finetuned"]
    assert metrics["viz_embedding_branches"] == ["pretrained", "finetuned"]
    assert {
        "route_network",
        "market_matrix",
        "fare_calendar",
        "segment_fingerprint",
        "segment_agreement",
        "segment_size",
    } <= set(aggregate_frame["view"].unique())


def test_extract_training_embeddings_falls_back_to_test_source() -> None:
    frame = make_dco_frame(rows=6)
    predictor = EmptyTrainEmbeddingPredictor(offset=1.0)

    embeddings, source = _extract_training_embeddings(
        predictor,
        frame[["advance_purchase", "origin_metro"]],
        branch_name="pretrained",
    )

    assert source == "test"
    assert embeddings.shape == (6, 3)


def test_fit_layout_caps_fit_and_trustworthiness_rows(monkeypatch) -> None:
    calls: dict[str, int | bool] = {}

    class FakeReducer:
        def __init__(self, **kwargs) -> None:
            calls["densmap"] = bool(kwargs["densmap"])
            calls["low_memory"] = bool(kwargs["low_memory"])

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            calls["fit_rows"] = len(X)
            return np.column_stack(
                [
                    np.arange(len(X), dtype=np.float32),
                    np.arange(len(X), dtype=np.float32) * 0.5,
                ]
            )

        def transform(self, X: np.ndarray) -> np.ndarray:
            calls["transform_rows"] = len(X)
            return np.column_stack(
                [
                    np.arange(len(X), dtype=np.float32),
                    np.arange(len(X), dtype=np.float32) * 0.25,
                ]
            )

    def fake_trustworthiness(X: np.ndarray, Y: np.ndarray, n_neighbors: int) -> float:
        calls["trust_rows"] = len(X)
        calls["trust_neighbors"] = n_neighbors
        return 0.87

    monkeypatch.setattr("dco_visualize.model.umap.UMAP", FakeReducer)
    monkeypatch.setattr("dco_visualize.model.trustworthiness", fake_trustworthiness)

    config = DCOVisualizeConfig(layout_fit_rows=10, trustworthiness_rows=7)
    embeddings = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)

    _, coords, projection, score = _fit_layout(embeddings, config)

    assert coords.shape == (20, 2)
    assert projection.name == "umap_transform"
    assert projection.params["fit_rows"] == 10
    assert projection.params["trust_rows"] == 7
    assert calls["fit_rows"] == 10
    assert calls["transform_rows"] == 20
    assert calls["trust_rows"] == 7
    assert calls["densmap"] is False
    assert calls["low_memory"] is True
    assert score == 0.87


def test_aggregate_parquet_file_emits_full_day_views(tmp_path) -> None:
    frame = make_dco_frame(rows=18)
    config = DCOVisualizeConfig(sample_rows=len(frame), train_rows=12, viz_rows=12)
    model = TabPFNEmbeddingModel(
        config=config,
        target_column="price_inc",
        feature_columns=[column for column in frame.columns if column not in {"price_inc", "row_id", "source_uri", "source_row_number", "customer", "sales_date"}],
        feature_kinds={
            column: ("numeric" if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column]) else "categorical")
            for column in frame.columns
            if column not in {"price_inc", "row_id", "source_uri", "source_row_number", "customer", "sales_date"}
        },
        categorical_feature_indices=[],
        excluded_columns={},
        retained_columns=list(frame.columns),
        hover_columns=["carrier", "origin_metro", "destination_metro"],
        pretrained=make_fake_branch("pretrained", offset=1.0),
        finetuned=make_fake_branch("finetuned", offset=2.0),
        route_source_column="origin_metro",
        route_destination_column="destination_metro",
        departure_date_column="outbound_departure_date",
        advance_purchase_column="advance_purchase",
        return_date_column="inbound_departure_date",
    )

    input_path = tmp_path / "full_day.parquet"
    frame.to_parquet(input_path, index=False)

    aggregate_frame = aggregate_parquet_file(model=model, parquet_path=input_path, batch_size=5)

    assert not aggregate_frame.empty
    assert {"route_network", "market_matrix", "fare_calendar"} <= set(aggregate_frame["view"].unique())


def test_transform_parquet_file_handles_all_null_string_batches(tmp_path) -> None:
    frame = make_dco_frame(rows=12)
    for column in [
        "inbound_booking_class",
        "inbound_codeshare",
        "inbound_departure_timeband",
        "inbound_fare_basis",
        "inbound_fare_family",
        "inbound_travel_stop_over",
    ]:
        frame[column] = [None] * 6 + ["X"] * 6

    config = DCOVisualizeConfig(sample_rows=len(frame), train_rows=8, viz_rows=6)
    model = TabPFNEmbeddingModel(
        config=config,
        target_column="price_inc",
        feature_columns=[column for column in frame.columns if column not in {"price_inc", "row_id", "source_uri", "source_row_number", "customer", "sales_date"}],
        feature_kinds={
            column: ("numeric" if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column]) else "categorical")
            for column in frame.columns
            if column not in {"price_inc", "row_id", "source_uri", "source_row_number", "customer", "sales_date"}
        },
        categorical_feature_indices=[],
        excluded_columns={},
        retained_columns=list(frame.columns),
        hover_columns=["carrier", "origin_metro", "destination_metro"],
        pretrained=make_fake_branch("pretrained", offset=1.0),
        finetuned=make_fake_branch("finetuned", offset=2.0),
        route_source_column="origin_metro",
        route_destination_column="destination_metro",
        departure_date_column="outbound_departure_date",
        advance_purchase_column="advance_purchase",
        return_date_column="inbound_departure_date",
    )

    input_path = tmp_path / "sample_null_batches.parquet"
    output_path = tmp_path / "embeddings_null_batches.parquet"
    frame.to_parquet(input_path, index=False)

    viz_frame, _, metrics = transform_parquet_file(
        model=model,
        parquet_path=input_path,
        output_path=output_path,
        viz_rows=config.viz_rows,
        batch_size=3,
        random_seed=config.random_seed,
    )

    transformed = pd.read_parquet(output_path)
    assert len(transformed) == len(frame)
    assert "inbound_booking_class" in transformed.columns
    assert metrics["embedded_rows"] == len(frame)
    assert len(viz_frame) == config.viz_rows


def test_target_tail_metrics_show_outlier_domination() -> None:
    y_true = pd.Series([100.0, 110.0, 120.0, 130.0, 18_000_000.0])
    y_pred = np.array([105.0, 115.0, 125.0, 135.0, 400_000.0], dtype=np.float64)

    raw_metrics = _prediction_metrics(y_true, y_pred)
    lower, upper = _winsor_bounds(y_true, 0.8)
    winsorized_metrics = _winsorized_prediction_metrics(
        y_true,
        y_pred,
        lower=lower,
        upper=upper,
    )
    target_stats = summarize_target_distribution(y_true)

    assert target_stats["rows_gt_1m"] == 1
    assert raw_metrics["rmse"] is not None
    assert winsorized_metrics["winsorized_rmse"] is not None
    assert raw_metrics["rmse"] > winsorized_metrics["winsorized_rmse"]
