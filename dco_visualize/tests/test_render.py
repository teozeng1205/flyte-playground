from __future__ import annotations

from pathlib import Path

import pandas as pd

from dco_visualize.render import build_visualization_frame, render_standalone_dashboard, save_dashboard_images


def _make_viz_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"row-{idx}" for idx in range(6)],
            "source_uri": ["s3://bucket/sample.parquet"] * 6,
            "source_row_number": list(range(6)),
            "customer": ["AA"] * 6,
            "sales_date": ["2026-03-07"] * 6,
            "origin_metro": ["NYC", "BOS", "DFW", "MIA", "NYC", "BOS"],
            "destination_metro": ["LON", "SFO", "PAR", "LAX", "LON", "SFO"],
            "trip_type": ["OW", "RT", "RT", "OW", "RT", "OW"],
            "stops": [0, 1, 1, 0, 1, 0],
            "carrier": ["AA", "DL", "BA", "UA", "AA", "DL"],
            "price_inc": [420.0, 650.0, 910.0, 380.0, 700.0, 430.0],
            "pretrained_layout_x": [-8.0, -2.0, 4.0, 8.0, -6.0, 6.0],
            "pretrained_layout_y": [3.0, 7.0, 1.0, -2.0, 5.0, -4.0],
            "finetuned_layout_x": [-6.0, -1.0, 3.0, 7.0, -4.0, 5.0],
            "finetuned_layout_y": [2.0, 6.0, 2.0, -1.0, 4.0, -3.0],
            "layout_method": ["densmap"] * 6,
            "pretrained_segment_id": [0, 1, 1, 0, 2, 2],
            "finetuned_segment_id": [1, 1, 0, 0, 2, 2],
            "pretrained_emb_000": [0.1, 0.2, 0.3, 0.0, -0.2, 0.4],
            "pretrained_emb_001": [1.2, 1.4, -0.4, 0.8, 0.2, -1.0],
            "finetuned_emb_000": [0.3, 0.5, 0.1, -0.1, -0.4, 0.6],
            "finetuned_emb_001": [1.0, 1.1, -0.1, 0.7, 0.4, -0.8],
        }
    )


def _make_aggregate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"view": "route_network", "branch": None, "key_1": "NYC", "key_2": "LON", "key_3": None, "segment_id": None, "count": 1200, "mean_price": 510.0, "value": None},
            {"view": "route_network", "branch": None, "key_1": "BOS", "key_2": "SFO", "key_3": None, "segment_id": None, "count": 900, "mean_price": 430.0, "value": None},
            {"view": "market_matrix", "branch": None, "key_1": "NYC", "key_2": "LON", "key_3": None, "segment_id": None, "count": 1200, "mean_price": 510.0, "value": None},
            {"view": "market_matrix", "branch": None, "key_1": "BOS", "key_2": "SFO", "key_3": None, "segment_id": None, "count": 900, "mean_price": 430.0, "value": None},
            {"view": "fare_calendar", "branch": None, "key_1": "2026-03-10", "key_2": "4-7", "key_3": None, "segment_id": None, "count": 400, "mean_price": 450.0, "value": None},
            {"view": "fare_calendar", "branch": None, "key_1": "2026-03-11", "key_2": "8-14", "key_3": None, "segment_id": None, "count": 500, "mean_price": 520.0, "value": None},
            {"view": "segment_size", "branch": "pretrained", "key_1": None, "key_2": None, "key_3": None, "segment_id": 0, "count": 2000, "mean_price": None, "value": None},
            {"view": "segment_size", "branch": "finetuned", "key_1": None, "key_2": None, "key_3": None, "segment_id": 1, "count": 1800, "mean_price": None, "value": None},
            {"view": "segment_fingerprint", "branch": "pretrained", "key_1": "trip_type", "key_2": "OW", "key_3": None, "segment_id": 0, "count": 900, "mean_price": None, "value": 1.4},
            {"view": "segment_fingerprint", "branch": "finetuned", "key_1": "origin_metro", "key_2": "NYC", "key_3": None, "segment_id": 1, "count": 700, "mean_price": None, "value": 1.1},
            {"view": "segment_agreement", "branch": None, "key_1": "0", "key_2": "1", "key_3": None, "segment_id": None, "count": 120, "mean_price": None, "value": None},
            {"view": "segment_agreement", "branch": None, "key_1": "1", "key_2": "1", "key_3": None, "segment_id": None, "count": 240, "mean_price": None, "value": None},
        ]
    )


def test_render_standalone_dashboard_generates_plotly_document(tmp_path: Path) -> None:
    frame = build_visualization_frame(_make_viz_frame(), viz_rows=4, random_seed=42)
    aggregate_frame = _make_aggregate_frame()
    image_paths = save_dashboard_images(frame, aggregate_frame, tmp_path, customer="AA", sales_date="2026-03-07")
    html = render_standalone_dashboard(
        frame=frame,
        aggregate_frame=aggregate_frame,
        hover_columns=["carrier", "origin_metro", "destination_metro"],
        customer="AA",
        sales_date="2026-03-07",
        profile={
            "representative_sampling": {
                "quality": {
                    "train": {
                        "metro_market_coverage": 1.0,
                        "top_airport_market_coverage": 1.0,
                        "trip_type_abs_error": 0.01,
                        "carrier_top_abs_error": 0.02,
                        "low_price_share_delta": 0.03,
                    },
                    "viz": {"metro_market_coverage": 1.0},
                }
            }
        },
        total_points=5000,
        total_rows=10000,
        parquet_file_count=12,
        hours_present=["00", "01", "02"],
        metrics={
            "train_rows": 50000,
            "embedded_rows": 200000,
            "duplicate_feature_fraction": 0.2,
            "pretrained_segment_count": 3,
            "finetuned_segment_count": 3,
            "pretrained_projection_trustworthiness": 0.91,
            "finetuned_projection_trustworthiness": 0.89,
            "pretrained": {"version": "2.5", "device": "cuda", "n_estimators": 4, "rmse": 12.3, "mae": 8.7},
            "finetuned": {"version": "2.5", "device": "cuda", "epochs": 8, "n_estimators_final_inference": 4, "rmse": 11.1, "mae": 7.9},
        },
        image_paths=image_paths,
    )

    assert "DCO TabPFN 2.5 Dashboard" in html
    assert "Embedding Comparison" in html
    assert "Representative Sampling" in html
    assert "plotly" in html.lower()


def test_save_dashboard_images_creates_replacement_pngs(tmp_path: Path) -> None:
    frame = _make_viz_frame()
    aggregate_frame = _make_aggregate_frame()
    images = save_dashboard_images(frame, aggregate_frame, tmp_path, customer="AA", sales_date="2026-03-07")
    assert {
        "pretrained_embedding_density_png",
        "finetuned_embedding_density_png",
        "route_network_png",
        "fare_calendar_png",
        "market_matrix_png",
        "segment_fingerprint_png",
    } <= set(images)
    for path in images.values():
        assert Path(path).exists()
