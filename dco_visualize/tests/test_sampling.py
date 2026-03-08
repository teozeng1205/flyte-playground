from __future__ import annotations

from pathlib import Path

import pandas as pd

from dco_visualize.io import sample_parquet_files


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
