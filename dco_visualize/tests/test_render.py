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
            "market_token": ["NYC->LON", "BOS->SFO", "DFW->PAR", "MIA->LON", "NYC->LON", "BOS->SFO"],
            "trip_type": ["OW", "RT", "RT", "OW", "RT", "OW"],
            "stops": [0, 1, 1, 0, 1, 0],
            "carrier": ["AA", "DL", "BA", "UA", "AA", "DL"],
            "price_inc": [420.0, 650.0, 910.0, 380.0, 700.0, 430.0],
            "layout_x": [-8.0, -2.0, 4.0, 8.0, -6.0, 6.0],
            "layout_y": [3.0, 7.0, 1.0, -2.0, 5.0, -4.0],
            "layout_method": ["densmap"] * 6,
            "segment_id": [0, 1, 1, 0, 2, 2],
            "embedding_000": [0.1, 0.2, 0.3, 0.0, -0.2, 0.4],
            "embedding_001": [1.2, 1.4, -0.4, 0.8, 0.2, -1.0],
        }
    )


def _make_aggregate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"view": "metro_flow", "key_1": "NYC", "key_2": "LON", "key_3": None, "segment_id": None, "count": 1200, "mean_log_price": 6.2, "mean_price": 510.0, "value": None},
            {"view": "metro_flow", "key_1": "BOS", "key_2": "SFO", "key_3": None, "segment_id": None, "count": 900, "mean_log_price": 6.0, "mean_price": 430.0, "value": None},
            {"view": "fare_calendar", "key_1": "2026-03-10", "key_2": "7-14", "key_3": "advance_purchase", "segment_id": None, "count": 400, "mean_log_price": 6.1, "mean_price": 450.0, "value": None},
            {"view": "fare_calendar", "key_1": "2026-03-11", "key_2": "14-30", "key_3": "advance_purchase", "segment_id": None, "count": 500, "mean_log_price": 6.3, "mean_price": 520.0, "value": None},
            {"view": "fare_calendar", "key_1": "2026-03-11", "key_2": "3-7", "key_3": "return_gap", "segment_id": None, "count": 280, "mean_log_price": 6.5, "mean_price": 680.0, "value": None},
            {"view": "segment_size", "key_1": None, "key_2": None, "key_3": None, "segment_id": 0, "count": 2000, "mean_log_price": None, "mean_price": None, "value": None},
            {"view": "segment_size", "key_1": None, "key_2": None, "key_3": None, "segment_id": 1, "count": 1800, "mean_log_price": None, "mean_price": None, "value": None},
            {"view": "segment_fingerprint", "key_1": "trip_type", "key_2": "OW", "key_3": None, "segment_id": 0, "count": 900, "mean_log_price": None, "mean_price": None, "value": 1.4},
            {"view": "segment_fingerprint", "key_1": "market_token", "key_2": "NYC->LON", "key_3": None, "segment_id": 1, "count": 700, "mean_log_price": None, "mean_price": None, "value": 1.1},
            {"view": "segment_fingerprint", "key_1": "carrier", "key_2": "AA", "key_3": None, "segment_id": 2, "count": 600, "mean_log_price": None, "mean_price": None, "value": 0.8},
        ]
    )


def test_render_standalone_dashboard_generates_plotly_document(tmp_path: Path) -> None:
    frame = build_visualization_frame(_make_viz_frame(), viz_rows=4, random_seed=42)
    aggregate_frame = _make_aggregate_frame()
    image_paths = save_dashboard_images(frame, aggregate_frame, tmp_path, customer="AA", sales_date="2026-03-07")
    html = render_standalone_dashboard(
        frame=frame,
        aggregate_frame=aggregate_frame,
        hover_columns=["carrier", "market_token"],
        customer="AA",
        sales_date="2026-03-07",
        total_points=5000,
        total_rows=10000,
        parquet_file_count=12,
        hours_present=["00", "01", "02"],
        metrics={
            "segment_count": 3,
            "noise_fraction": 0.05,
            "projection": {"name": "densmap", "n_neighbors": 10},
            "projection_trustworthiness": 0.91,
            "encoder_backend": "ft_transformer_contrastive",
            "segment_method": "hdbscan",
        },
        image_paths=image_paths,
    )

    assert "DCO Dashboard" in html
    assert "Fare surfaces, route flows, and segment structure" in html
    assert "plotly" in html.lower()


def test_save_dashboard_images_creates_replacement_pngs(tmp_path: Path) -> None:
    frame = _make_viz_frame()
    aggregate_frame = _make_aggregate_frame()
    images = save_dashboard_images(frame, aggregate_frame, tmp_path, customer="AA", sales_date="2026-03-07")
    assert {"embedding_density_png", "metro_flow_map_png", "fare_calendar_png", "market_matrix_png", "segment_fingerprint_png"} <= set(images)
    for path in images.values():
        assert Path(path).exists()
