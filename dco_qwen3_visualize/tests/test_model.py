from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from conftest import make_dco_frame
from dco_qwen3_visualize.config import DCOQwen3VisualizeConfig
from dco_qwen3_visualize.model import (
    ProjectionResult,
    build_similarity_pairs,
    run_qwen3_visualization,
    serialize_dco_row,
    serialize_dco_frame,
    write_training_stats_plot,
)


def test_serialize_dco_row_preserves_core_fields_and_missing() -> None:
    frame = make_dco_frame(rows=1)
    row = frame.iloc[0].copy()
    row["inbound_departure_date"] = None
    text = serialize_dco_row(row, [column for column in frame.columns if column not in {"row_id", "source_uri", "source_row_number", "customer", "sales_date"}])

    assert text.startswith("airfare offer row")
    assert "origin: JFK" in text
    assert "destination: LHR" in text
    assert "price_inc: 220" in text
    assert "inbound_departure_date: missing" in text


def test_build_similarity_pairs_keeps_market_context() -> None:
    frame = make_dco_frame(rows=12)
    texts = serialize_dco_frame(frame, [column for column in frame.columns if column not in {"row_id", "source_uri", "source_row_number", "customer", "sales_date"}])
    config = DCOQwen3VisualizeConfig(train_rows=12, viz_rows=12, finetune_pair_rows=12, finetune_max_negatives=2)

    pairs = build_similarity_pairs(frame, texts, config)

    assert pairs
    first = pairs[0]
    assert "anchor_text" in first
    assert "positive_text" in first
    assert len(first["negative_texts"]) == 2
    assert "metro_od" in first


def test_run_qwen3_visualization_emits_dual_branch_contract(tmp_path: Path, monkeypatch) -> None:
    frame = make_dco_frame(rows=10)
    config = DCOQwen3VisualizeConfig(sample_rows=10, train_rows=8, viz_rows=10, layout_fit_rows=5, trustworthiness_rows=4)

    class FakeEncoder:
        def __init__(self, offset: float) -> None:
            self.offset = offset
            self.device = "cpu"

        def encode(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
            base = np.arange(len(texts), dtype=np.float32) + self.offset
            return np.stack([base, base * 0.5 + 1.0, base * 0.25 + 2.0], axis=1)

    def fake_load(config: DCOQwen3VisualizeConfig, *, adapter_path=None, trainable=False):
        return FakeEncoder(offset=1.0 if adapter_path is None else 2.0)

    def fake_finetune(pair_records, config: DCOQwen3VisualizeConfig, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        return {"status": "succeeded", "pairs": len(pair_records), "epochs": config.finetune_epochs}

    def fake_segments(embeddings: np.ndarray, config: DCOQwen3VisualizeConfig):
        return (np.arange(len(embeddings)) % 2).astype(np.int64), {"kind": "fake", "segment_count": 2}

    def fake_layout(embeddings: np.ndarray, config: DCOQwen3VisualizeConfig):
        coords = np.column_stack(
            [np.arange(len(embeddings), dtype=np.float32), np.arange(len(embeddings), dtype=np.float32) * 0.5]
        )
        return ProjectionResult(
            coordinates=coords,
            trustworthiness=0.91,
            fit_rows=min(config.layout_fit_rows, len(embeddings)),
            trust_rows=min(config.trustworthiness_rows, len(embeddings)),
            method="umap_transform",
        )

    monkeypatch.setattr("dco_qwen3_visualize.model._load_qwen_encoder", fake_load)
    monkeypatch.setattr("dco_qwen3_visualize.model._fine_tune_qwen3_adapter", fake_finetune)
    monkeypatch.setattr("dco_qwen3_visualize.model._fit_segment_ids", fake_segments)
    monkeypatch.setattr("dco_qwen3_visualize.model._fit_layout", fake_layout)

    result = run_qwen3_visualization(frame.iloc[:8].copy(), frame.copy(), config, tmp_path)

    viz_frame = result["viz_frame"]
    assert len(viz_frame) == len(frame)
    assert {"pretrained_layout_x", "pretrained_layout_y", "finetuned_layout_x", "finetuned_layout_y"} <= set(viz_frame.columns)
    assert {"pretrained_emb_000", "finetuned_emb_000"} <= set(viz_frame.columns)
    assert result["metrics"]["encoder_backend"] == "qwen3_embedding"
    assert result["metrics"]["pair_count"] >= 0
    assert Path(result["finetune_pairs_path"]).exists()
    assert Path(result["adapter_tar_path"]).exists()


def test_write_training_stats_plot_renders(tmp_path: Path) -> None:
    output_path = tmp_path / "training_stats.png"
    metrics = {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "train_rows": 50_000,
        "viz_rows": 50_000,
        "pair_count": 19_677,
        "elapsed_seconds": 321.0,
        "pretrained": {
            "embedding_dim": 512,
            "projection_trustworthiness": 0.91,
            "price_neighbor_correlation": 0.42,
            "segment_count": 8,
        },
        "finetuned": {
            "embedding_dim": 512,
            "projection_trustworthiness": 0.96,
            "price_neighbor_correlation": 0.68,
            "segment_count": 10,
            "finetune": {
                "status": "succeeded",
                "steps": 120,
                "mean_loss": 0.41,
                "final_loss": 0.19,
                "epoch_mean_losses": [0.62, 0.35, 0.21],
                "sampled_loss_steps": [1, 20, 40, 60, 80, 100, 120],
                "sampled_loss_values": [0.9, 0.8, 0.65, 0.48, 0.35, 0.24, 0.19],
            },
        },
    }

    write_training_stats_plot(output_path, metrics)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
