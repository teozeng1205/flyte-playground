from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def make_dco_frame(rows: int = 16) -> pd.DataFrame:
    base_dates = pd.date_range("2026-03-07", periods=rows, freq="D")
    metros = ["NYC", "LON", "BOS", "SFO", "DFW", "MIA", "PAR", "CHI"]
    countries = {
        "NYC": "US",
        "BOS": "US",
        "SFO": "US",
        "DFW": "US",
        "MIA": "US",
        "CHI": "US",
        "LON": "GB",
        "PAR": "FR",
    }
    frame = pd.DataFrame(
        {
            "row_id": [f"row-{idx}" for idx in range(rows)],
            "source_uri": ["file:///tmp/sample.parquet"] * rows,
            "source_row_number": list(range(rows)),
            "customer": ["AA"] * rows,
            "sales_date": ["2026-03-07"] * rows,
            "advance_purchase": [(idx * 7) % 180 for idx in range(rows)],
            "cabin": ["E", "B", "E", "F"] * (rows // 4) + ["E"] * (rows % 4),
            "carrier": ["AA", "DL", "UA", "BA"] * (rows // 4) + ["AA"] * (rows % 4),
            "channel": ["web", "gds", None, "meta"] * (rows // 4) + ["web"] * (rows % 4),
            "currency": ["USD"] * rows,
            "origin": ["JFK", "BOS", "DFW", "MIA"] * (rows // 4) + ["JFK"] * (rows % 4),
            "destination": ["LHR", "SFO", "CDG", "LAX"] * (rows // 4) + ["LHR"] * (rows % 4),
            "origin_city": [metros[idx % len(metros)] for idx in range(rows)],
            "destination_city": [metros[(idx + 1) % len(metros)] for idx in range(rows)],
            "origin_metro": [metros[idx % len(metros)] for idx in range(rows)],
            "destination_metro": [metros[(idx + 1) % len(metros)] for idx in range(rows)],
            "origin_country": [countries[metros[idx % len(metros)]] for idx in range(rows)],
            "destination_country": [countries[metros[(idx + 1) % len(metros)]] for idx in range(rows)],
            "inbound_departure_date": [base_dates[min(idx + 5, rows - 1)].date().isoformat() if idx % 2 else None for idx in range(rows)],
            "length_of_stay": [7 if idx % 2 else -1 for idx in range(rows)],
            "observation_datetime": [f"{base_dates[idx].date().isoformat()} 08:00" for idx in range(rows)],
            "outbound_departure_date": [base_dates[idx].date().isoformat() for idx in range(rows)],
            "outbound_flight_duration": [120 + (idx * 20) for idx in range(rows)],
            "inbound_flight_duration": [140 + (idx * 15) for idx in range(rows)],
            "outbound_gcm": [350 + (idx * 80) for idx in range(rows)],
            "pos": ["US", "US", "GB", "US"] * (rows // 4) + ["US"] * (rows % 4),
            "price_exc": [180.0 + (idx * 23) for idx in range(rows)],
            "price_inc": [220.0 + (idx * 31) for idx in range(rows)],
            "refundable": [idx % 3 == 0 for idx in range(rows)],
            "source": ["gds", "airline", "meta", "gds"] * (rows // 4) + ["gds"] * (rows % 4),
            "stops": [0, 1, 1, 0] * (rows // 4) + [0] * (rows % 4),
            "tax": [40.0 + (idx * 8) for idx in range(rows)],
            "trip_type": ["OW" if idx % 2 == 0 else "RT" for idx in range(rows)],
            "search_class": ["E"] * rows,
        }
    )
    return frame.iloc[:rows].reset_index(drop=True)
