from __future__ import annotations

from pathlib import Path

import pandas as pd

from conftest import make_dco_frame
from dco_visualize.config import DCOVisualizeConfig
from dco_visualize.io import sample_parquet_files
from dco_visualize.sampling import build_representative_samples_from_parquet


def test_sample_parquet_files_adds_metadata(tmp_path: Path) -> None:
    left = pd.DataFrame(
        {
            "carrier": ["AA", "AA", "DL", "UA"],
            "price_inc": [100.0, 120.0, 90.0, 140.0],
        }
    )
    right = pd.DataFrame(
        {
            "carrier": ["WN", "B6", "AA"],
            "price_inc": [80.0, 110.0, 130.0],
        }
    )

    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    left.to_parquet(left_path, index=False)
    right.to_parquet(right_path, index=False)

    sample, profile = sample_parquet_files(
        parquet_uris=[str(left_path), str(right_path)],
        sample_size=5,
        customer="AA",
        sales_date="2026-03-07",
        random_seed=42,
        batch_size=2,
    )

    assert len(sample) == 5
    assert {"row_id", "source_uri", "source_row_number", "customer", "sales_date"} <= set(sample.columns)
    assert sample["customer"].eq("AA").all()
    assert sample["sales_date"].eq("2026-03-07").all()
    assert profile["parquet_file_count"] == 2
    assert profile["total_rows"] == 7
    assert profile["train_rows"] == 5


def test_representative_sampler_covers_all_metro_markets_and_prefers_lowest_fare(tmp_path: Path) -> None:
    frame = make_dco_frame(rows=16)
    market_specs = [
        ("NYC", "LON", "JFK", "LHR", [300.0, 100.0, 200.0, 400.0]),
        ("BOS", "SFO", "BOS", "SFO", [500.0, 510.0, 520.0, 530.0]),
        ("DFW", "PAR", "DFW", "CDG", [620.0, 610.0, 630.0, 640.0]),
        ("MIA", "LAX", "MIA", "LAX", [700.0, 690.0, 710.0, 720.0]),
    ]
    for group_index, (origin_metro, destination_metro, origin, destination, prices) in enumerate(market_specs):
        start = group_index * 4
        stop = start + 4
        frame.loc[start:stop - 1, "origin_metro"] = origin_metro
        frame.loc[start:stop - 1, "destination_metro"] = destination_metro
        frame.loc[start:stop - 1, "origin"] = origin
        frame.loc[start:stop - 1, "destination"] = destination
        frame.loc[start:stop - 1, "trip_type"] = "OW"
        frame.loc[start:stop - 1, "cabin"] = "E"
        frame.loc[start:stop - 1, "stops"] = 0
        frame.loc[start:stop - 1, "source"] = "gds"
        frame.loc[start:stop - 1, "carrier"] = "AA"
        frame.loc[start:stop - 1, "price_inc"] = prices
        frame.loc[start:stop - 1, "advance_purchase"] = [7, 14, 21, 28]

    input_path = tmp_path / "representative_input.parquet"
    train_output_path = tmp_path / "representative_train.parquet"
    viz_output_path = tmp_path / "representative_viz.parquet"
    frame.to_parquet(input_path, index=False)

    config = DCOVisualizeConfig(sample_rows=len(frame), train_rows=4, viz_rows=8)
    stats = build_representative_samples_from_parquet(
        input_path,
        train_rows=4,
        viz_rows=8,
        random_seed=config.random_seed,
        batch_size=4,
        config=config,
        train_output_path=train_output_path,
        viz_output_path=viz_output_path,
    )

    train_frame = pd.read_parquet(train_output_path)
    viz_frame = pd.read_parquet(viz_output_path)
    population_markets = set((frame["origin_metro"] + "->" + frame["destination_metro"]).astype(str))
    train_markets = set((train_frame["origin_metro"] + "->" + train_frame["destination_metro"]).astype(str))

    assert stats["train_rows"] == 4
    assert stats["viz_rows"] == 8
    assert train_markets == population_markets
    assert set(train_frame["row_id"]).issubset(set(viz_frame["row_id"]))

    nyc_lhr_price = train_frame.loc[
        (train_frame["origin_metro"] == "NYC") & (train_frame["destination_metro"] == "LON"),
        "price_inc",
    ].iloc[0]
    assert nyc_lhr_price == 100.0
