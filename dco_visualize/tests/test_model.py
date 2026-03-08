from __future__ import annotations

from pathlib import Path

import pandas as pd

from dco_visualize.config import DCOVisualizeConfig
from dco_visualize.model import fit_embedding_model, transform_parquet_file
from conftest import make_dco_frame


def test_fit_embedding_model_produces_segmented_embeddings() -> None:
    frame = make_dco_frame(rows=18)
    config = DCOVisualizeConfig(
        sample_rows=len(frame),
        train_rows=len(frame),
        viz_rows=10,
        embedding_dims=8,
        pretrain_epochs=1,
        train_batch_size=8,
    )
    result = fit_embedding_model(frame, config)
    transformed, _ = result.model.transform(frame)

    embedding_columns = [column for column in transformed.columns if column.startswith("embedding_")]
    assert embedding_columns
    assert "segment_id" in transformed.columns
    assert "layout_x" not in transformed.columns
    assert result.metrics["embedding_dims"] == len(embedding_columns)
    assert result.metrics["encoder_backend"] == "ft_transformer_contrastive"
    assert "search_class" in result.metrics["excluded_columns"]
    assert "market_token" in transformed.columns


def test_transform_parquet_file_emits_viz_sample_and_aggregate_views(tmp_path: Path) -> None:
    frame = make_dco_frame(rows=20)
    config = DCOVisualizeConfig(
        sample_rows=len(frame),
        train_rows=12,
        viz_rows=8,
        embedding_dims=8,
        pretrain_epochs=1,
        train_batch_size=6,
    )
    fit_result = fit_embedding_model(frame.iloc[:12].reset_index(drop=True), config)
    input_path = tmp_path / "sample.parquet"
    output_path = tmp_path / "embeddings.parquet"
    frame.to_parquet(input_path, index=False)

    viz_frame, aggregate_frame, metrics = transform_parquet_file(
        model=fit_result.model,
        parquet_path=input_path,
        output_path=output_path,
        viz_rows=config.viz_rows,
        batch_size=5,
        random_seed=config.random_seed,
    )

    transformed = pd.read_parquet(output_path)
    assert len(transformed) == len(frame)
    assert {"layout_x", "layout_y", "layout_method"} <= set(viz_frame.columns)
    assert viz_frame["layout_method"].eq("densmap").all()
    assert metrics["embedded_rows"] == len(frame)
    assert {"metro_flow", "fare_calendar", "segment_fingerprint", "segment_size"} <= set(aggregate_frame["view"].unique())
