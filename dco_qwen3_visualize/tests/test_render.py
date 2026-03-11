from __future__ import annotations

import pandas as pd

from dco_qwen3_visualize.render import build_visualization_frame, render_standalone_dashboard


def _make_viz_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"row-{idx}" for idx in range(6)],
            "source_uri": ["s3://bucket/sample.parquet"] * 6,
            "source_row_number": list(range(6)),
            "customer": ["AA"] * 6,
            "sales_date": ["2026-03-07"] * 6,
            "origin": ["JFK", "BOS", "DFW", "MIA", "JFK", "BOS"],
            "destination": ["LHR", "SFO", "CDG", "LAX", "LHR", "SFO"],
            "origin_metro": ["NYC", "BOS", "DFW", "MIA", "NYC", "BOS"],
            "destination_metro": ["LON", "SFO", "PAR", "LAX", "LON", "SFO"],
            "trip_type": ["OW", "RT", "RT", "OW", "RT", "OW"],
            "stops": [0, 1, 1, 0, 1, 0],
            "carrier": ["AA", "DL", "BA", "UA", "AA", "DL"],
            "source": ["gds", "airline", "meta", "gds", "meta", "gds"],
            "cabin": ["E", "B", "E", "F", "E", "B"],
            "price_inc": [420.0, 650.0, 910.0, 380.0, 700.0, 430.0],
            "advance_purchase": [14, 21, 30, 7, 28, 35],
            "outbound_departure_date": ["2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13", "2026-03-14", "2026-03-15"],
            "pretrained_layout_x": [-8.0, -2.0, 4.0, 8.0, -6.0, 6.0],
            "pretrained_layout_y": [3.0, 7.0, 1.0, -2.0, 5.0, -4.0],
            "finetuned_layout_x": [-6.0, -1.0, 3.0, 7.0, -4.0, 5.0],
            "finetuned_layout_y": [2.0, 6.0, 2.0, -1.0, 4.0, -3.0],
            "layout_method": ["umap_transform"] * 6,
            "pretrained_segment_id": [0, 1, 1, 0, 2, 2],
            "finetuned_segment_id": [1, 1, 0, 0, 2, 2],
            "pretrained_emb_000": [0.1, 0.2, 0.3, 0.0, -0.2, 0.4],
            "pretrained_emb_001": [1.2, 1.4, -0.4, 0.8, 0.2, -1.0],
            "finetuned_emb_000": [0.3, 0.5, 0.1, -0.1, -0.4, 0.6],
            "finetuned_emb_001": [1.0, 1.1, -0.1, 0.7, 0.4, -0.8],
        }
    )


def test_render_standalone_dashboard_generates_embedding_only_document() -> None:
    frame = build_visualization_frame(_make_viz_frame(), viz_rows=4, random_seed=42)
    html = render_standalone_dashboard(
        frame=frame,
        hover_columns=["carrier", "origin_metro", "destination_metro"],
        customer="AA",
        sales_date="2026-03-07",
        profile={"representative_sampling": {"quality": {"viz": {"rows": 4, "metro_market_coverage": 1.0}}}},
        total_rows=10000,
        parquet_file_count=12,
        hours_present=["00", "01", "02"],
        metrics={"model_id": "Qwen/Qwen3-Embedding-0.6B", "train_rows": 50000, "pair_count": 40000},
    )

    assert "DCO Qwen3 Embedding Dashboard" in html
    assert "Color By" in html
    assert "Category Filters" in html
    assert "Pretrained Qwen3" in html
    assert "Fine-tuned Qwen3" in html
    assert "plotly" in html.lower()
