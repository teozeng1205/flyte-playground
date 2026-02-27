import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _first_scalar(value: Any) -> Optional[float]:
    """Extract a scalar float from nested list/array structures."""
    current = value
    for _ in range(4):
        if current is None:
            return None
        if isinstance(current, np.ndarray):
            if current.size == 0:
                return None
            current = current.flat[0]
            continue
        if isinstance(current, (list, tuple)):
            if len(current) == 0:
                return None
            current = current[0]
            continue
        break
    try:
        return float(current)
    except Exception:
        return None


def _clean_pred(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and math.isnan(value):
        return 0.0
    return max(0.0, float(value))


def _extract_preds(pred: Dict[str, Any]) -> Dict[str, float]:
    yq = _first_scalar(
        pred.get("itinerary/tax_yq_amount", {}).get("content") if isinstance(pred, dict) else None
    )
    yr = _first_scalar(
        pred.get("itinerary/tax_yr_amount", {}).get("content") if isinstance(pred, dict) else None
    )
    total = _first_scalar(
        pred.get("itinerary/total_tax", {}).get("content") if isinstance(pred, dict) else None
    )
    return {
        "predicted_YQ_tax": _clean_pred(yq),
        "predicted_YR_tax": _clean_pred(yr),
        "predicted_total_tax": _clean_pred(total),
    }


def read_metrics_dataframe(path: str) -> pd.DataFrame:
    """Read either the legacy flat parquet or the newer nested (inputs/predictions) parquet."""
    schema = pq.read_schema(path)
    if "validatingCarrier" in schema.names:
        return pd.read_parquet(path)

    if "inputs" not in schema.names:
        raise ValueError(f"Unrecognized parquet schema (missing 'inputs'): {path}")

    inputs_raw = pd.read_parquet(path, columns=["inputs"])
    df = pd.json_normalize(inputs_raw["inputs"])

    # Prefer `predictions` struct when present; it's the model output.
    # Some datasets also include `inputs.predicted_*` fields, but they may be defaulted (e.g. 0.0).
    if "predictions" in schema.names:
        preds_raw = pd.read_parquet(path, columns=["predictions"])
        preds = pd.DataFrame(preds_raw["predictions"].apply(_extract_preds).tolist())
        for col in ["predicted_YQ_tax", "predicted_YR_tax", "predicted_total_tax"]:
            if col in df.columns:
                df[col] = preds[col]
            else:
                df[col] = preds[col]
    elif "total_tax" in df.columns and "predicted_total_tax" not in df.columns:
        # Fallback when only inputs are present.
        df["predicted_total_tax"] = df["total_tax"]

    return df
