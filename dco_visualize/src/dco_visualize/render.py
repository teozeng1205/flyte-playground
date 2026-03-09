from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from plotly.offline.offline import get_plotlyjs
from plotly.subplots import make_subplots

PRETRAINED_DENSITY_PNG = "pretrained_embedding_density.png"
FINETUNED_DENSITY_PNG = "finetuned_embedding_density.png"
ROUTE_NETWORK_PNG = "route_network.png"
CALENDAR_PNG = "fare_calendar.png"
MATRIX_PNG = "market_matrix.png"
FINGERPRINT_PNG = "segment_fingerprint.png"
LOGGER = logging.getLogger(__name__)

FIELD_LABELS = {
    "carrier": "Carrier",
    "source": "Source",
    "trip_type": "Trip Type",
    "cabin": "Cabin",
    "stops": "Stops",
    "origin_metro": "Origin Metro",
    "destination_metro": "Destination Metro",
    "outbound_departure_date": "Departure Date",
    "price_inc": "Fare",
    "advance_purchase": "Advance Purchase",
    "pretrained_segment_id": "Pretrained Segment",
    "finetuned_segment_id": "Fine-tuned Segment",
}


def _parse_aggregate_views(aggregate_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    views: dict[str, pd.DataFrame] = {}
    for view, subset in aggregate_frame.groupby("view", dropna=False):
        views[str(view)] = subset.reset_index(drop=True)
    return views


def _figure_to_html(figure: go.Figure, *, include_js: bool) -> str:
    return pio.to_html(
        figure,
        include_plotlyjs=include_js,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )


def _image_to_data_uri(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_visualization_frame(frame: pd.DataFrame, viz_rows: int, random_seed: int) -> pd.DataFrame:
    if len(frame) <= viz_rows:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=viz_rows, random_state=random_seed).reset_index(drop=True)


def _format_metric_value(value: object, *, precision: int = 3, percentage: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "n/a"
        return f"{value * 100:.1f}%" if percentage else f"{value:.{precision}f}"
    if isinstance(value, (np.integer, int)):
        return f"{value:,}"
    return str(value)


def _dashboard_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def _interesting_categorical(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    values = frame[column].astype("string").fillna("missing")
    counts = values.value_counts(dropna=False)
    if len(counts) <= 1:
        return False
    return bool(counts.iloc[0] / max(len(values), 1) < 0.995)


def _interesting_numeric(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return values.nunique() > 1


def _color_option_definitions(frame: pd.DataFrame) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    categorical_candidates = [
        ("carrier", "Carrier", 10),
        ("trip_type", "Trip Type", 8),
        ("cabin", "Cabin", 8),
        ("stops", "Stops", 8),
        ("source", "Source", 10),
        ("origin_metro", "Origin Metro", 12),
        ("destination_metro", "Destination Metro", 12),
        ("pretrained_segment_id", "Pretrained Segment", 8),
        ("finetuned_segment_id", "Fine-tuned Segment", 8),
    ]
    numeric_candidates = [
        ("price_inc", "Fare", "currency"),
        ("advance_purchase", "Advance Purchase", "number"),
    ]

    for column, label, top_k in categorical_candidates:
        if _interesting_categorical(frame, column):
            options.append({"key": column, "label": label, "kind": "categorical", "top_k": top_k})
    for column, label, number_format in numeric_candidates:
        if _interesting_numeric(frame, column):
            options.append({"key": column, "label": label, "kind": "numeric", "format": number_format})
    return options


def _embedding_dashboard_payload(frame: pd.DataFrame, hover_columns: Iterable[str]) -> dict[str, object]:
    option_definitions = _color_option_definitions(frame)
    default_key = next((option["key"] for option in option_definitions if option["key"] == "carrier"), None)
    if default_key is None and option_definitions:
        default_key = option_definitions[0]["key"]

    hover_fields = [
        field
        for field in [
            "carrier",
            "source",
            "trip_type",
            "cabin",
            "stops",
            "origin_metro",
            "destination_metro",
            "outbound_departure_date",
            "price_inc",
        ]
        if field in frame.columns
    ]
    for field in hover_columns:
        if field in frame.columns and field not in hover_fields:
            hover_fields.append(field)
    hover_fields = hover_fields[:10]

    column_keys = {
        *hover_fields,
        *(str(option["key"]) for option in option_definitions),
    }
    columns = {
        key: [_dashboard_value(value) for value in frame[key].tolist()]
        for key in column_keys
        if key in frame.columns
    }
    return {
        "row_count": int(len(frame)),
        "points": {
            "pretrained": {
                "x": [float(value) for value in frame["pretrained_layout_x"].tolist()],
                "y": [float(value) for value in frame["pretrained_layout_y"].tolist()],
            },
            "finetuned": {
                "x": [float(value) for value in frame["finetuned_layout_x"].tolist()],
                "y": [float(value) for value in frame["finetuned_layout_y"].tolist()],
            },
        },
        "hover_fields": hover_fields,
        "field_labels": FIELD_LABELS,
        "columns": columns,
        "color_options": option_definitions,
        "default_color_key": default_key,
    }


def _build_branch_embedding_figure(
    frame: pd.DataFrame, branch: str, hover_columns: Iterable[str]
) -> go.Figure:
    x_column = f"{branch}_layout_x"
    y_column = f"{branch}_layout_y"
    segment_column = f"{branch}_segment_id"
    available_hover = [column for column in hover_columns if column in frame.columns]

    figure = go.Figure()
    for segment_id, subset in frame.groupby(segment_column, dropna=False):
        figure.add_trace(
            go.Scattergl(
                x=subset[x_column],
                y=subset[y_column],
                mode="markers",
                name=f"{branch} · segment {segment_id}",
                customdata=subset[available_hover].astype(str).to_numpy() if available_hover else None,
                marker={"size": 4, "opacity": 0.55},
                hovertemplate=(
                    "<br>".join(
                        [f"segment={segment_id}", "x=%{x:.2f}", "y=%{y:.2f}"]
                        + [f"{column}=%{{customdata[{index}]}}" for index, column in enumerate(available_hover)]
                    )
                    + "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    figure.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 50, "b": 40},
        title=f"{branch.title()} Embedding Density",
        xaxis_title="layout_x",
        yaxis_title="layout_y",
    )
    return figure


def _build_embedding_comparison_figure(frame: pd.DataFrame, hover_columns: Iterable[str]) -> go.Figure:
    available_hover = [column for column in hover_columns if column in frame.columns]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Pretrained TabPFN 2.5", "Fine-tuned TabPFN 2.5"),
        horizontal_spacing=0.08,
    )
    for col_index, branch in enumerate(["pretrained", "finetuned"], start=1):
        x_column = f"{branch}_layout_x"
        y_column = f"{branch}_layout_y"
        segment_column = f"{branch}_segment_id"
        for segment_id, subset in frame.groupby(segment_column, dropna=False):
            figure.add_trace(
                go.Scattergl(
                    x=subset[x_column],
                    y=subset[y_column],
                    mode="markers",
                    marker={"size": 4, "opacity": 0.55},
                    customdata=subset[available_hover].astype(str).to_numpy() if available_hover else None,
                    hovertemplate=(
                        "<br>".join(
                            [f"segment={segment_id}", "x=%{x:.2f}", "y=%{y:.2f}"]
                            + [f"{column}=%{{customdata[{index}]}}" for index, column in enumerate(available_hover)]
                        )
                        + "<extra></extra>"
                    ),
                    name=f"{branch}-{segment_id}",
                    showlegend=False,
                ),
                row=1,
                col=col_index,
            )
        figure.update_xaxes(title_text="layout_x", row=1, col=col_index)
        figure.update_yaxes(title_text="layout_y", row=1, col=col_index)

    figure.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 70, "b": 40},
        title="Embedding Comparison",
        height=560,
    )
    return figure


def _build_route_network_figure(route_frame: pd.DataFrame, customer: str, sales_date: str) -> go.Figure:
    if route_frame.empty:
        return go.Figure().update_layout(title="Route Network", template="plotly_white")

    working = route_frame.sort_values("count", ascending=False).head(30).copy()
    sources = [f"O:{value}" for value in working["key_1"].astype(str)]
    destinations = [f"D:{value}" for value in working["key_2"].astype(str)]
    nodes = list(dict.fromkeys(sources + destinations))
    node_index = {node: idx for idx, node in enumerate(nodes)}
    source_idx = [node_index[node] for node in sources]
    dest_idx = [node_index[node] for node in destinations]
    labels = [node[2:] for node in nodes]

    figure = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 20,
                    "thickness": 18,
                    "line": {"color": "#1f2933", "width": 0.5},
                    "label": labels,
                    "color": ["#264653" if node.startswith("O:") else "#e76f51" for node in nodes],
                },
                link={
                    "source": source_idx,
                    "target": dest_idx,
                    "value": working["count"],
                    "color": "rgba(42,157,143,0.35)",
                    "customdata": working["mean_price"],
                    "hovertemplate": (
                        "%{source.label} → %{target.label}<br>"
                        "rows=%{value:,}<br>"
                        "mean fare=$%{customdata:.0f}<extra></extra>"
                    ),
                },
            )
        ]
    )
    figure.update_layout(
        title=f"Route Network · {customer} · {sales_date}",
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        height=580,
    )
    return figure


def _build_fare_calendar_figure(calendar_frame: pd.DataFrame, customer: str, sales_date: str) -> go.Figure:
    if calendar_frame.empty:
        return go.Figure().update_layout(title="Fare Calendar", template="plotly_white")
    pivot = (
        calendar_frame.pivot_table(index="key_2", columns="key_1", values="mean_price", aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns.astype(str).tolist(),
            y=pivot.index.astype(str).tolist(),
            colorscale="Sunset",
            hovertemplate="departure=%{x}<br>advance purchase=%{y}<br>mean fare=$%{z:.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Fare Calendar · {customer} · {sales_date}",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        xaxis_title="Departure Date",
        yaxis_title="Advance Purchase Bucket",
        height=520,
    )
    return figure


def _build_market_matrix_figure(route_frame: pd.DataFrame, customer: str, sales_date: str) -> go.Figure:
    if route_frame.empty:
        return go.Figure().update_layout(title="Market Matrix", template="plotly_white")

    working = route_frame.copy()
    top_origins = working.groupby("key_1")["count"].sum().sort_values(ascending=False).head(14).index
    top_destinations = working.groupby("key_2")["count"].sum().sort_values(ascending=False).head(14).index
    working = working[working["key_1"].isin(top_origins) & working["key_2"].isin(top_destinations)]
    matrix = working.pivot_table(index="key_1", columns="key_2", values="mean_price", aggfunc="mean").fillna(0.0)
    counts = working.pivot_table(index="key_1", columns="key_2", values="count", aggfunc="sum").fillna(0.0)

    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns.astype(str).tolist(),
            y=matrix.index.astype(str).tolist(),
            customdata=counts.to_numpy(),
            colorscale="YlGnBu",
            hovertemplate=(
                "origin=%{y}<br>destination=%{x}<br>"
                "mean fare=$%{z:.0f}<br>rows=%{customdata:,}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"Market Matrix · {customer} · {sales_date}",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        xaxis_title="Destination",
        yaxis_title="Origin",
        height=560,
    )
    return figure


def _build_segment_fingerprint_figure(fingerprint_frame: pd.DataFrame, customer: str, sales_date: str) -> go.Figure:
    if fingerprint_frame.empty:
        return go.Figure().update_layout(title="Segment Fingerprints", template="plotly_white")

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Pretrained segments", "Fine-tuned segments"),
        horizontal_spacing=0.1,
    )
    for column_index, branch in enumerate(["pretrained", "finetuned"], start=1):
        branch_frame = fingerprint_frame[fingerprint_frame["branch"] == branch].copy()
        if branch_frame.empty:
            continue
        branch_frame["label"] = branch_frame["key_1"].astype(str) + " · " + branch_frame["key_2"].astype(str)
        labels = (
            branch_frame.assign(abs_value=branch_frame["value"].abs())
            .sort_values("abs_value", ascending=False)["label"]
            .drop_duplicates()
            .head(18)
        )
        branch_frame = branch_frame[branch_frame["label"].isin(labels)]
        pivot = branch_frame.pivot_table(
            index="label",
            columns="segment_id",
            values="value",
            aggfunc="mean",
        ).fillna(0.0)
        figure.add_trace(
            go.Heatmap(
                z=pivot.to_numpy(),
                x=[str(value) for value in pivot.columns.tolist()],
                y=pivot.index.astype(str).tolist(),
                colorscale="RdBu",
                zmid=0.0,
                hovertemplate="label=%{y}<br>segment=%{x}<br>lift=%{z:.2f}<extra></extra>",
                showscale=(column_index == 2),
            ),
            row=1,
            col=column_index,
        )
    figure.update_layout(
        title=f"Segment Fingerprints · {customer} · {sales_date}",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        height=720,
    )
    return figure


def _build_segment_agreement_figure(agreement_frame: pd.DataFrame, customer: str, sales_date: str) -> go.Figure:
    if agreement_frame.empty:
        return go.Figure().update_layout(title="Segment Agreement", template="plotly_white")
    pivot = agreement_frame.pivot_table(index="key_1", columns="key_2", values="count", aggfunc="sum").fillna(0.0)
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns.astype(str).tolist(),
            y=pivot.index.astype(str).tolist(),
            colorscale="Viridis",
            hovertemplate=(
                "pretrained segment=%{y}<br>fine-tuned segment=%{x}<br>rows=%{z:,}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"Segment Agreement · {customer} · {sales_date}",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        xaxis_title="Fine-tuned Segment",
        yaxis_title="Pretrained Segment",
        height=520,
    )
    return figure


def _save_branch_density_png(frame: pd.DataFrame, branch: str, output_path: Path) -> None:
    x_column = f"{branch}_layout_x"
    y_column = f"{branch}_layout_y"
    cmap = LinearSegmentedColormap.from_list("dco_density", ["#f7f3e8", "#2a9d8f", "#264653"])
    plt.figure(figsize=(8, 6))
    plt.hexbin(frame[x_column], frame[y_column], gridsize=55, cmap=cmap, mincnt=1)
    plt.title(f"{branch.title()} embedding density")
    plt.xlabel("layout_x")
    plt.ylabel("layout_y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _save_route_network_png(route_frame: pd.DataFrame, output_path: Path) -> None:
    top_routes = route_frame.sort_values("count", ascending=False).head(12).copy()
    top_routes["route"] = top_routes["key_1"].astype(str) + " -> " + top_routes["key_2"].astype(str)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=top_routes, x="count", y="route", hue="mean_price", dodge=False, palette="crest")
    plt.title("Top routes by row count")
    plt.xlabel("Rows")
    plt.ylabel("Route")
    plt.legend(title="Mean fare")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _save_calendar_png(calendar_frame: pd.DataFrame, output_path: Path) -> None:
    pivot = (
        calendar_frame.pivot_table(index="key_2", columns="key_1", values="mean_price", aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="mako")
    plt.title("Fare calendar")
    plt.xlabel("Departure date")
    plt.ylabel("Advance purchase bucket")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _save_market_matrix_png(route_frame: pd.DataFrame, output_path: Path) -> None:
    top_origins = route_frame.groupby("key_1")["count"].sum().sort_values(ascending=False).head(10).index
    top_destinations = route_frame.groupby("key_2")["count"].sum().sort_values(ascending=False).head(10).index
    working = route_frame[route_frame["key_1"].isin(top_origins) & route_frame["key_2"].isin(top_destinations)]
    pivot = working.pivot_table(index="key_1", columns="key_2", values="mean_price", aggfunc="mean").fillna(0.0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlGnBu")
    plt.title("Market matrix")
    plt.xlabel("Destination")
    plt.ylabel("Origin")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _save_fingerprint_png(fingerprint_frame: pd.DataFrame, output_path: Path) -> None:
    branch_frame = fingerprint_frame[fingerprint_frame["branch"] == "finetuned"].copy()
    if branch_frame.empty:
        branch_frame = fingerprint_frame.copy()
    branch_frame["label"] = branch_frame["key_1"].astype(str) + " · " + branch_frame["key_2"].astype(str)
    top_labels = (
        branch_frame.assign(abs_value=branch_frame["value"].abs())
        .sort_values("abs_value", ascending=False)["label"]
        .drop_duplicates()
        .head(18)
    )
    branch_frame = branch_frame[branch_frame["label"].isin(top_labels)]
    pivot = branch_frame.pivot_table(index="label", columns="segment_id", values="value", aggfunc="mean").fillna(0.0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap="RdBu_r", center=0.0)
    plt.title("Fine-tuned segment fingerprint")
    plt.xlabel("Segment")
    plt.ylabel("Feature value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_dashboard_images(
    frame: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    output_dir: str | Path,
    customer: str,
    sales_date: str,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = _parse_aggregate_views(aggregate_frame)
    LOGGER.info(
        "Saving dashboard images to %s for customer=%s sales_date=%s viz_rows=%d aggregate_rows=%d",
        output_dir,
        customer,
        sales_date,
        len(frame),
        len(aggregate_frame),
    )

    pretrained_path = output_dir / PRETRAINED_DENSITY_PNG
    finetuned_path = output_dir / FINETUNED_DENSITY_PNG
    route_path = output_dir / ROUTE_NETWORK_PNG
    calendar_path = output_dir / CALENDAR_PNG
    matrix_path = output_dir / MATRIX_PNG
    fingerprint_path = output_dir / FINGERPRINT_PNG

    _save_branch_density_png(frame, "pretrained", pretrained_path)
    _save_branch_density_png(frame, "finetuned", finetuned_path)
    _save_route_network_png(views.get("route_network", pd.DataFrame()), route_path)
    _save_calendar_png(views.get("fare_calendar", pd.DataFrame()), calendar_path)
    _save_market_matrix_png(views.get("market_matrix", pd.DataFrame()), matrix_path)
    _save_fingerprint_png(views.get("segment_fingerprint", pd.DataFrame()), fingerprint_path)
    LOGGER.info("Saved dashboard image pack to %s", output_dir)

    return {
        "pretrained_embedding_density_png": str(pretrained_path),
        "finetuned_embedding_density_png": str(finetuned_path),
        "route_network_png": str(route_path),
        "fare_calendar_png": str(calendar_path),
        "market_matrix_png": str(matrix_path),
        "segment_fingerprint_png": str(fingerprint_path),
    }


def render_standalone_dashboard(
    frame: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    hover_columns: Iterable[str],
    customer: str,
    sales_date: str,
    profile: dict[str, object] | None,
    total_points: int,
    total_rows: int,
    parquet_file_count: int,
    hours_present: list[str],
    metrics: dict[str, object],
    image_paths: dict[str, str],
) -> str:
    LOGGER.info(
        "Rendering standalone dashboard for customer=%s sales_date=%s viz_rows=%d",
        customer,
        sales_date,
        len(frame),
    )
    payload_json = json.dumps(_embedding_dashboard_payload(frame, hover_columns)).replace("</", "<\\/")
    plotly_js = get_plotlyjs()
    LOGGER.info("Dashboard render complete for customer=%s sales_date=%s", customer, sales_date)
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DCO TabPFN Dashboard · {customer} · {sales_date}</title>
  <style>
    html {{
      height: 100%;
    }}
    :root {{
      --paper: #f4efe7;
      --ink: #1f2933;
      --teal: #2a9d8f;
      --ember: #e76f51;
      --sand: #e9c46a;
      --panel: #fffdf9;
      --line: rgba(31, 41, 51, 0.12);
      --shadow: 0 18px 48px rgba(31, 41, 51, 0.10);
    }}
    body {{
      min-height: 100%;
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(233, 196, 106, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(42, 157, 143, 0.18), transparent 24%),
        linear-gradient(180deg, #f8f3ea 0%, #f4efe7 100%);
    }}
    main {{
      width: 100%;
      min-height: 100vh;
      box-sizing: border-box;
      padding: 16px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 12px;
    }}
    .topbar {{
      padding: 18px 20px;
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) auto;
      gap: 18px;
      align-items: end;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-family: "Iowan Old Style", "Palatino", serif;
      font-size: 38px;
      line-height: 0.98;
    }}
    .hero p {{
      margin: 0;
      max-width: 920px;
      font-size: 15px;
      line-height: 1.5;
      color: rgba(31, 41, 51, 0.78);
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .workspace {{
      min-height: 0;
      min-height: 0;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 12px;
    }}
    .sidebar {{
      padding: 18px;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 16px;
      min-height: 0;
    }}
    .stage {{
      padding: 18px 18px 14px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 12px;
      min-height: 0;
    }}
    .sidebar h2,
    .stage h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .topbar-badges,
    .stage-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      align-items: center;
    }}
    .metric-pill {{
      display: inline-flex;
      flex-direction: column;
      gap: 2px;
      min-width: 96px;
      padding: 10px 12px;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,255,255,0.86));
      border: 1px solid rgba(31, 41, 51, 0.10);
    }}
    .metric-pill .pill-label {{
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(31, 41, 51, 0.52);
    }}
    .metric-pill .pill-value {{
      font-size: 16px;
      font-weight: 700;
      color: var(--ink);
    }}
    .stage-header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: end;
    }}
    .stage-header p {{
      margin: 0;
      font-size: 14px;
      color: rgba(31, 41, 51, 0.72);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(42, 157, 143, 0.10);
      border: 1px solid rgba(42, 157, 143, 0.18);
      font-size: 12px;
      font-weight: 600;
      color: #1c645c;
    }}
    .sidebar-kicker {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(31, 41, 51, 0.48);
      margin-bottom: 8px;
    }}
    .control-stack {{
      display: grid;
      gap: 14px;
    }}
    .control-block label {{
      display: block;
      margin-bottom: 8px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(31, 41, 51, 0.6);
    }}
    .control-block select {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
      font: inherit;
    }}
    .control-block select:focus {{
      outline: 2px solid rgba(42, 157, 143, 0.2);
      border-color: rgba(42, 157, 143, 0.45);
    }}
    .view-switch {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      padding: 4px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      gap: 4px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
    }}
    .view-switch button {{
      border: 0;
      background: transparent;
      color: rgba(31, 41, 51, 0.72);
      font: inherit;
      font-weight: 700;
      padding: 10px 14px;
      border-radius: 10px;
      cursor: pointer;
      transition: background 120ms ease, color 120ms ease, transform 120ms ease;
    }}
    .view-switch button:hover {{
      background: rgba(38, 70, 83, 0.06);
      color: var(--ink);
    }}
    .view-switch button.active {{
      background: linear-gradient(135deg, #264653, #2a9d8f);
      color: white;
      box-shadow: 0 8px 20px rgba(38, 70, 83, 0.18);
    }}
    .sidebar-meta {{
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 14px;
      min-height: 0;
    }}
    .selection-summary {{
      padding: 14px 15px;
      border-radius: 16px;
      background:
        radial-gradient(circle at top right, rgba(42, 157, 143, 0.12), transparent 42%),
        linear-gradient(180deg, rgba(38, 70, 83, 0.05), rgba(38, 70, 83, 0.02));
      border: 1px solid rgba(38, 70, 83, 0.10);
    }}
    .selection-summary .summary-title {{
      margin: 0 0 6px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: rgba(31, 41, 51, 0.52);
    }}
    .selection-summary .summary-primary {{
      margin: 0 0 4px;
      font-size: 24px;
      font-weight: 700;
      line-height: 1;
    }}
    .selection-summary .summary-secondary {{
      margin: 0;
      font-size: 13px;
      line-height: 1.45;
      color: rgba(31, 41, 51, 0.72);
    }}
    .legend-note {{
      margin: 0;
      font-size: 13px;
      line-height: 1.55;
      color: rgba(31, 41, 51, 0.75);
    }}
    .legend-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      align-items: center;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.85);
      border: 1px solid var(--line);
      font-size: 13px;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      flex: 0 0 12px;
      border: 1px solid rgba(31, 41, 51, 0.15);
    }}
    #embedding-comparison-plot {{
      min-height: 0;
      height: 100%;
      border-radius: 20px;
      overflow: hidden;
      background:
        radial-gradient(circle at 20% 20%, rgba(42, 157, 143, 0.06), transparent 24%),
        radial-gradient(circle at 80% 18%, rgba(231, 111, 81, 0.08), transparent 26%),
        #fffdf9;
      border: 1px solid rgba(31, 41, 51, 0.07);
    }}
    .plot-frame {{
      min-height: 0;
    }}
    @media (max-width: 960px) {{
      .topbar {{
        grid-template-columns: 1fr;
      }}
      .hero h1 {{
        font-size: 34px;
      }}
      .workspace {{
        grid-template-columns: 1fr;
      }}
      .stage-header {{
        grid-template-columns: 1fr;
      }}
      .topbar-badges,
      .stage-badges {{
        justify-content: flex-start;
      }}
      main {{
        min-height: auto;
      }}
      .stage {{
        min-height: 720px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="topbar panel">
      <div class="hero">
          <h1>DCO TabPFN 2.5 Dashboard</h1>
          <p>
            Raw DCO fare rows for <strong>{customer}</strong> on <strong>{sales_date}</strong>, encoded twice with the same
            foundation model family: once as a pretrained regressor and once after task-specific fine-tuning on
            <code>price_inc</code>. This view is intentionally focused on one job: compare the two manifolds cleanly.
          </p>
      </div>
      <div class="topbar-badges">
        <div class="metric-pill"><span class="pill-label">Rows</span><span class="pill-value">{total_points:,}</span></div>
        <div class="metric-pill"><span class="pill-label">Files</span><span class="pill-value">{parquet_file_count:,}</span></div>
        <div class="metric-pill"><span class="pill-label">Hours</span><span class="pill-value">{len(hours_present)}</span></div>
      </div>
    </section>

    <section class="workspace">
      <aside class="sidebar panel">
        <div>
          <div class="sidebar-kicker">Controls</div>
          <h2>Embedding Comparison</h2>
          <p style="margin:0 0 12px; font-size:14px; line-height:1.5; color:rgba(31,41,51,0.72);">
            Recolor both manifolds with the same field, then filter to a slice and switch between side-by-side and single-branch inspection.
          </p>
        </div>
        <div class="control-stack">
          <div class="control-block">
            <label for="color-field">Color By</label>
            <select id="color-field"></select>
          </div>
          <div class="control-block">
            <label for="filter-field">Filter By</label>
            <select id="filter-field"></select>
          </div>
          <div class="control-block">
            <label for="filter-value">Filter Value</label>
            <select id="filter-value"></select>
          </div>
          <div class="control-block">
            <label>Branch View</label>
            <div class="view-switch" id="branch-view">
              <button type="button" data-mode="compare" class="active">Compare</button>
              <button type="button" data-mode="pretrained">Pretrained</button>
              <button type="button" data-mode="finetuned">Fine-tuned</button>
            </div>
          </div>
        </div>
        <div class="sidebar-meta">
          <div class="selection-summary" id="selection-summary">
            <p class="summary-title">Current View</p>
            <p class="summary-primary">All rows</p>
            <p class="summary-secondary">Showing the full dashboard sample.</p>
          </div>
          <div id="color-legend" class="legend-list"></div>
          <p id="color-note" class="legend-note"></p>
        </div>
      </aside>

      <section class="stage panel">
        <div class="stage-header">
          <div>
            <h2>Embedding Stage</h2>
            <p>Read structure changes directly: stable color islands mean the model preserved the slice, while warped or separated islands mean fine-tuning changed the geometry.</p>
          </div>
          <div class="stage-badges">
            <span class="badge">{total_points:,} dashboard rows</span>
            <span class="badge">{parquet_file_count:,} parquet files</span>
            <span class="badge">hours {", ".join(hours_present)}</span>
          </div>
        </div>
        <div class="plot-frame">
          <div id="embedding-comparison-plot"></div>
        </div>
      </section>
    </section>
  </main>
  <script>{plotly_js}</script>
  <script id="embedding-dashboard-data" type="application/json">{payload_json}</script>
  <script>
    const embeddingDashboard = JSON.parse(document.getElementById("embedding-dashboard-data").textContent);
    const hoverFields = embeddingDashboard.hover_fields;
    const fieldLabels = embeddingDashboard.field_labels;
    const columns = embeddingDashboard.columns;
    const colorOptions = embeddingDashboard.color_options;
    const defaultColorKey = embeddingDashboard.default_color_key || (colorOptions[0] && colorOptions[0].key);
    const selectElement = document.getElementById("color-field");
    const filterFieldElement = document.getElementById("filter-field");
    const filterValueElement = document.getElementById("filter-value");
    const legendElement = document.getElementById("color-legend");
    const noteElement = document.getElementById("color-note");
    const selectionSummaryElement = document.getElementById("selection-summary");
    const plotElement = document.getElementById("embedding-comparison-plot");
    const branchViewElement = document.getElementById("branch-view");

    const neutralColor = "#d7d2c8";
    const categoricalPalette = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#457b9d", "#f4a261", "#6d597a", "#43aa8b", "#bc4749", "#577590", "#8ab17d", "#9d4edd"];
    let currentBranchMode = "compare";
    let currentFilterKey = "__all__";
    let currentFilterValue = "__all__";

    function fieldLabel(key) {{
      return fieldLabels[key] || key.replace(/_/g, " ");
    }}

    function normalizeCategory(value) {{
      return value === null || value === undefined || value === "" ? "missing" : String(value);
    }}

    function quantile(values, q) {{
      if (!values.length) return 0;
      const sorted = [...values].sort((a, b) => a - b);
      const position = (sorted.length - 1) * q;
      const base = Math.floor(position);
      const rest = position - base;
      if (sorted[base + 1] !== undefined) {{
        return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
      }}
      return sorted[base];
    }}

    function formatValue(key, value) {{
      if (value === null || value === undefined || value === "") return "missing";
      if (key === "price_inc" && Number.isFinite(Number(value))) {{
        return "$" + Number(value).toLocaleString(undefined, {{ maximumFractionDigits: 0 }});
      }}
      if (key === "advance_purchase" && Number.isFinite(Number(value))) {{
        return Number(value).toLocaleString(undefined, {{ maximumFractionDigits: 0 }}) + " days";
      }}
      return String(value);
    }}

    function semanticOrder(key, values) {{
      const orderMaps = {{
        trip_type: ["OW", "RT", "OJ", "MC", "missing", "Other"],
        cabin: ["E", "P", "B", "F", "missing", "Other"],
        stops: ["0", "1", "2", "3", "missing", "Other"],
      }};
      if (key.endsWith("segment_id")) {{
        return [...values].sort((a, b) => Number(a) - Number(b));
      }}
      if (!orderMaps[key]) return values;
      const order = orderMaps[key];
      return [...values].sort((a, b) => {{
        const left = order.indexOf(a);
        const right = order.indexOf(b);
        if (left === -1 && right === -1) return a.localeCompare(b);
        if (left === -1) return 1;
        if (right === -1) return -1;
        return left - right;
      }});
    }}

    function categoricalPaletteMap(key, categories) {{
      const colors = {{}};
      let paletteIndex = 0;
      const ordered = semanticOrder(key, categories);
      for (const category of ordered) {{
        if (category === "-1" || category === "missing" || category === "Other") {{
          colors[category] = category === "Other" ? "#c3beb3" : neutralColor;
          continue;
        }}
        if (key === "trip_type") {{
          colors[category] = {{
            OW: "#2a9d8f",
            RT: "#e76f51",
            OJ: "#264653",
            MC: "#e9c46a",
          }}[category] || categoricalPalette[paletteIndex++ % categoricalPalette.length];
          continue;
        }}
        if (key === "cabin") {{
          colors[category] = {{
            E: "#2a9d8f",
            P: "#3a86ff",
            B: "#e9c46a",
            F: "#1f2933",
          }}[category] || categoricalPalette[paletteIndex++ % categoricalPalette.length];
          continue;
        }}
        if (key === "stops") {{
          colors[category] = {{
            "0": "#2a9d8f",
            "1": "#457b9d",
            "2": "#6d597a",
            "3": "#bc4749",
          }}[category] || categoricalPalette[paletteIndex++ % categoricalPalette.length];
          continue;
        }}
        colors[category] = categoricalPalette[paletteIndex++ % categoricalPalette.length];
      }}
      return colors;
    }}

    function indicesForFilter(filterKey, filterValue) {{
      if (filterKey === "__all__" || filterValue === "__all__") {{
        return Array.from({{ length: embeddingDashboard.row_count }}, (_, index) => index);
      }}
      return columns[filterKey]
        .map((value, index) => [normalizeCategory(value), index])
        .filter(([value]) => value === filterValue)
        .map(([, index]) => index);
    }}

    function filterableOptions() {{
      return colorOptions.filter((option) => option.kind === "categorical");
    }}

    function valuesForIndices(values, indices) {{
      return indices.map((index) => values[index]);
    }}

    function categoricalState(option, indices) {{
      const rawValues = valuesForIndices(columns[option.key], indices).map(normalizeCategory);
      const counts = new Map();
      rawValues.forEach((value) => counts.set(value, (counts.get(value) || 0) + 1));
      let ordered = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([value]) => value);
      ordered = semanticOrder(option.key, ordered);
      const topK = option.top_k || 10;
      const keep = ordered.length > topK ? ordered.slice(0, topK - 1) : ordered;
      const mappedValues = rawValues.map((value) => keep.includes(value) ? value : (ordered.length > topK ? "Other" : value));
      const mappedCounts = new Map();
      mappedValues.forEach((value) => mappedCounts.set(value, (mappedCounts.get(value) || 0) + 1));
      const categories = semanticOrder(option.key, [...mappedCounts.keys()]);
      const palette = categoricalPaletteMap(option.key, categories);
      return {{
        labels: mappedValues.map((value) => value === "-1" ? "noise" : value),
        marker: (showScale) => ({{
          size: 4,
          opacity: 0.72,
          color: mappedValues.map((value) => palette[value]),
          showscale: false,
        }}),
        hoverTitle: option.label,
        legendHtml: categories.map((value) => {{
          const label = value === "-1" ? "noise" : value;
          const count = mappedCounts.get(value) || 0;
          return `<span class="legend-item"><span class="legend-swatch" style="background:${{palette[value]}}"></span>${{label}} · ${{count.toLocaleString()}}</span>`;
        }}).join(""),
        note: ordered.length > topK
          ? `Showing the top ${{topK - 1}} ${{option.label.toLowerCase()}} values in color and grouping the remainder into Other.`
          : `${{option.label}} is colored with a categorical palette that preserves ordering for stops, cabins, and segments.`,
      }};
    }}

    function numericState(option, indices) {{
      const numericValues = valuesForIndices(columns[option.key], indices).map((value) => {{
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
      }});
      const finiteValues = numericValues.filter((value) => value !== null);
      const lower = quantile(finiteValues, 0.01);
      const upper = quantile(finiteValues, 0.99);
      const clipped = numericValues.map((value) => value === null ? null : Math.max(lower, Math.min(upper, value)));
      return {{
        labels: numericValues.map((value) => formatValue(option.key, value)),
        marker: (showScale) => ({{
          size: 4,
          opacity: 0.72,
          color: clipped,
          colorscale: "Turbo",
          cmin: lower,
          cmax: upper,
          showscale: showScale,
          colorbar: {{
            title: option.label,
            len: 0.82,
            thickness: 14,
            x: 1.02,
          }},
        }}),
        hoverTitle: option.label,
        legendHtml: `<span class="legend-item"><span class="legend-swatch" style="background:linear-gradient(90deg,#30123b,#1f9e89,#fde725)"></span>${{formatValue(option.key, lower)}} to ${{formatValue(option.key, upper)}}</span>`,
        note: `${{option.label}} uses a clipped continuous scale between the 1st and 99th percentiles to avoid a few extreme fares dominating the colors.`,
      }};
    }}

    function optionState(option, indices) {{
      return option.kind === "numeric" ? numericState(option, indices) : categoricalState(option, indices);
    }}

    const customData = Array.from({{ length: embeddingDashboard.row_count }}, (_, index) =>
      hoverFields.map((field) => formatValue(field, columns[field][index]))
    );

    function hoverTemplate(optionLabel) {{
      const lines = [
        `${{optionLabel}}=%{{text}}`,
        "x=%{{x:.2f}}",
        "y=%{{y:.2f}}",
        ...hoverFields.map((field, index) => `${{fieldLabel(field)}}=%{{customdata[${{index}}]}}`),
      ];
      return lines.join("<br>") + "<extra></extra>";
    }}

    function computePlotHeight() {{
      const panelRect = plotElement.parentElement.getBoundingClientRect();
      return Math.max(window.innerHeight - panelRect.top - 28, 560);
    }}

    function singleBranchLayout(branch, height) {{
      const title = branch === "pretrained" ? "Pretrained TabPFN 2.5" : "Fine-tuned TabPFN 2.5";
      const axisPrefix = branch === "pretrained" ? "pretrained" : "finetuned";
      return {{
        paper_bgcolor: "#fffdf9",
        plot_bgcolor: "#fffdf9",
        margin: {{ l: 24, r: 24, t: 72, b: 40 }},
        height,
        xaxis: {{ title: `${{axisPrefix}}_layout_x`, zeroline: false }},
        yaxis: {{ title: `${{axisPrefix}}_layout_y`, zeroline: false }},
        annotations: [
          {{
            text: title,
            x: 0.5,
            y: 1.04,
            xref: "paper",
            yref: "paper",
            showarrow: false,
            font: {{ size: 18 }},
          }},
        ],
      }};
    }}

    function comparisonLayout(height) {{
      return {{
        paper_bgcolor: "#fffdf9",
        plot_bgcolor: "#fffdf9",
        margin: {{ l: 24, r: 24, t: 72, b: 40 }},
        height,
        xaxis: {{ domain: [0.0, 0.46], title: "pretrained_layout_x", zeroline: false }},
        yaxis: {{ domain: [0.0, 1.0], title: "pretrained_layout_y", zeroline: false }},
        xaxis2: {{ domain: [0.54, 1.0], title: "finetuned_layout_x", zeroline: false }},
        yaxis2: {{ domain: [0.0, 1.0], title: "finetuned_layout_y", zeroline: false }},
        annotations: [
          {{ text: "Pretrained TabPFN 2.5", x: 0.23, y: 1.05, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 16 }} }},
          {{ text: "Fine-tuned TabPFN 2.5", x: 0.77, y: 1.05, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 16 }} }},
        ],
      }};
    }}

    function tracesFor(option, branchMode, indices) {{
      const state = optionState(option, indices);
      const filteredCustomData = valuesForIndices(customData, indices);
      const filteredPretrainedX = valuesForIndices(embeddingDashboard.points.pretrained.x, indices);
      const filteredPretrainedY = valuesForIndices(embeddingDashboard.points.pretrained.y, indices);
      const filteredFinetunedX = valuesForIndices(embeddingDashboard.points.finetuned.x, indices);
      const filteredFinetunedY = valuesForIndices(embeddingDashboard.points.finetuned.y, indices);
      if (branchMode === "compare") {{
        return {{
          traces: [
            {{
              type: "scattergl",
              mode: "markers",
              x: filteredPretrainedX,
              y: filteredPretrainedY,
              xaxis: "x",
              yaxis: "y",
              text: state.labels,
              customdata: filteredCustomData,
              marker: state.marker(false),
              hovertemplate: hoverTemplate(option.label),
              showlegend: false,
              name: "Pretrained",
            }},
            {{
              type: "scattergl",
              mode: "markers",
              x: filteredFinetunedX,
              y: filteredFinetunedY,
              xaxis: "x2",
              yaxis: "y2",
              text: state.labels,
              customdata: filteredCustomData,
              marker: state.marker(true),
              hovertemplate: hoverTemplate(option.label),
              showlegend: false,
              name: "Fine-tuned",
            }},
          ],
          legendHtml: state.legendHtml,
          note: state.note,
          layout: comparisonLayout(computePlotHeight()),
        }};
      }}
      const branchKey = branchMode === "pretrained" ? "pretrained" : "finetuned";
      return {{
        traces: [
          {{
            type: "scattergl",
            mode: "markers",
            x: branchKey === "pretrained" ? filteredPretrainedX : filteredFinetunedX,
            y: branchKey === "pretrained" ? filteredPretrainedY : filteredFinetunedY,
            text: state.labels,
            customdata: filteredCustomData,
            marker: state.marker(true),
            hovertemplate: hoverTemplate(option.label),
            showlegend: false,
            name: branchKey,
          }},
        ],
        legendHtml: state.legendHtml,
        note: state.note,
        layout: singleBranchLayout(branchKey, computePlotHeight()),
      }};
    }}

    function populateFilterValues(selectedKey) {{
      filterValueElement.innerHTML = "";
      const allOption = document.createElement("option");
      allOption.value = "__all__";
      allOption.textContent = "All values";
      filterValueElement.appendChild(allOption);
      if (selectedKey === "__all__") {{
        filterValueElement.value = "__all__";
        return;
      }}
      const counts = new Map();
      columns[selectedKey].map(normalizeCategory).forEach((value) => counts.set(value, (counts.get(value) || 0) + 1));
      const ordered = semanticOrder(
        selectedKey,
        [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([value]) => value),
      );
      ordered.forEach((value) => {{
        const option = document.createElement("option");
        option.value = value;
        option.textContent = `${{value === "-1" ? "noise" : value}} · ${{(counts.get(value) || 0).toLocaleString()}}`;
        filterValueElement.appendChild(option);
      }});
      if (![...filterValueElement.options].some((option) => option.value === currentFilterValue)) {{
        currentFilterValue = "__all__";
      }}
      filterValueElement.value = currentFilterValue;
    }}

    function renderOption(optionKey) {{
      const option = colorOptions.find((entry) => entry.key === optionKey) || colorOptions[0];
      if (!option) return;
      const indices = indicesForFilter(currentFilterKey, currentFilterValue);
      const state = tracesFor(option, currentBranchMode, indices);
      legendElement.innerHTML = state.legendHtml;
      const branchLabel =
        currentBranchMode === "compare"
          ? "Compare"
          : currentBranchMode === "pretrained"
            ? "Pretrained only"
            : "Fine-tuned only";
      const filterSummary =
        currentFilterKey === "__all__" || currentFilterValue === "__all__"
          ? `Showing all ${{indices.length.toLocaleString()}} rows in the dashboard sample.`
          : `Filtered to ${{fieldLabel(currentFilterKey)}} = ${{currentFilterValue === "-1" ? "noise" : currentFilterValue}} across ${{indices.length.toLocaleString()}} rows.`;
      noteElement.textContent = `${{state.note}} ${{filterSummary}}`;
      selectionSummaryElement.innerHTML = `
        <p class="summary-title">Current View</p>
        <p class="summary-primary">${{indices.length.toLocaleString()}} rows</p>
        <p class="summary-secondary">
          ${{branchLabel}} · colored by ${{option.label}}
          ${{
            currentFilterKey === "__all__" || currentFilterValue === "__all__"
              ? " · no filter applied."
              : ` · filtered on ${{fieldLabel(currentFilterKey)}} = ${{currentFilterValue === "-1" ? "noise" : currentFilterValue}}.`
          }}
        </p>
      `;
      Plotly.react(plotElement, state.traces, state.layout, {{
        displaylogo: false,
        responsive: true,
      }});
    }}

    colorOptions.forEach((option) => {{
      const optionElement = document.createElement("option");
      optionElement.value = option.key;
      optionElement.textContent = option.label;
      selectElement.appendChild(optionElement);
    }});
    const filterFieldAllOption = document.createElement("option");
    filterFieldAllOption.value = "__all__";
    filterFieldAllOption.textContent = "No filter";
    filterFieldElement.appendChild(filterFieldAllOption);
    filterableOptions().forEach((option) => {{
      const optionElement = document.createElement("option");
      optionElement.value = option.key;
      optionElement.textContent = option.label;
      filterFieldElement.appendChild(optionElement);
    }});
    selectElement.value = defaultColorKey;
    selectElement.addEventListener("change", (event) => renderOption(event.target.value));
    filterFieldElement.value = currentFilterKey;
    populateFilterValues(currentFilterKey);
    filterFieldElement.addEventListener("change", (event) => {{
      currentFilterKey = event.target.value;
      currentFilterValue = "__all__";
      populateFilterValues(currentFilterKey);
      renderOption(selectElement.value);
    }});
    filterValueElement.addEventListener("change", (event) => {{
      currentFilterValue = event.target.value;
      renderOption(selectElement.value);
    }});
    branchViewElement.querySelectorAll("button").forEach((button) => {{
      button.addEventListener("click", () => {{
        currentBranchMode = button.dataset.mode;
        branchViewElement.querySelectorAll("button").forEach((item) => item.classList.toggle("active", item === button));
        renderOption(selectElement.value);
      }});
    }});
    window.addEventListener("resize", () => renderOption(selectElement.value));
    renderOption(defaultColorKey);
  </script>
</body>
</html>
"""
