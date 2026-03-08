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
    fit_embedding_model,
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
    assert {"pretrained_emb_000", "finetuned_emb_000"} <= set(transformed.columns)
    assert {"pretrained_segment_id", "finetuned_segment_id"} <= set(transformed.columns)
    assert {"pretrained_layout_x", "pretrained_layout_y", "finetuned_layout_x", "finetuned_layout_y", "layout_method"} <= set(
        viz_frame.columns
    )
    assert viz_frame["layout_method"].eq("densmap").all()
    assert metrics["embedded_rows"] == len(frame)
    assert {
        "route_network",
        "market_matrix",
        "fare_calendar",
        "segment_fingerprint",
        "segment_agreement",
        "segment_size",
    } <= set(aggregate_frame["view"].unique())
