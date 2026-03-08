from __future__ import annotations

import base64
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
from plotly.subplots import make_subplots

PRETRAINED_DENSITY_PNG = "pretrained_embedding_density.png"
FINETUNED_DENSITY_PNG = "finetuned_embedding_density.png"
ROUTE_NETWORK_PNG = "route_network.png"
CALENDAR_PNG = "fare_calendar.png"
MATRIX_PNG = "market_matrix.png"
FINGERPRINT_PNG = "segment_fingerprint.png"
LOGGER = logging.getLogger(__name__)


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
    views = _parse_aggregate_views(aggregate_frame)
    LOGGER.info(
        "Rendering standalone dashboard for customer=%s sales_date=%s viz_rows=%d aggregate_views=%s",
        customer,
        sales_date,
        len(frame),
        sorted(views),
    )
    embedding_html = _figure_to_html(
        _build_embedding_comparison_figure(frame, hover_columns),
        include_js=True,
    )
    route_html = _figure_to_html(
        _build_route_network_figure(views.get("route_network", pd.DataFrame()), customer, sales_date),
        include_js=False,
    )
    calendar_html = _figure_to_html(
        _build_fare_calendar_figure(views.get("fare_calendar", pd.DataFrame()), customer, sales_date),
        include_js=False,
    )
    matrix_html = _figure_to_html(
        _build_market_matrix_figure(views.get("market_matrix", pd.DataFrame()), customer, sales_date),
        include_js=False,
    )
    fingerprint_html = _figure_to_html(
        _build_segment_fingerprint_figure(views.get("segment_fingerprint", pd.DataFrame()), customer, sales_date),
        include_js=False,
    )
    agreement_html = _figure_to_html(
        _build_segment_agreement_figure(views.get("segment_agreement", pd.DataFrame()), customer, sales_date),
        include_js=False,
    )

    gallery = "".join(
        f"""
        <figure class="gallery-card">
          <img src="{_image_to_data_uri(path)}" alt="{name}">
          <figcaption>{name.replace('_', ' ')}</figcaption>
        </figure>
        """
        for name, path in image_paths.items()
    )

    pretrained_metrics = metrics.get("pretrained", {})
    finetuned_metrics = metrics.get("finetuned", {})
    representative = (profile or {}).get("representative_sampling", {}) if profile else {}
    quality = representative.get("quality", {}) if isinstance(representative, dict) else {}
    train_quality = quality.get("train", {}) if isinstance(quality, dict) else {}
    viz_quality = quality.get("viz", {}) if isinstance(quality, dict) else {}
    embedded_rows = int(metrics.get("embedded_rows", total_points) or total_points)
    train_rows = int(metrics.get("train_rows", 0) or 0)
    duplicate_fraction = float(metrics.get("duplicate_feature_fraction", 0.0) or 0.0)
    duplicate_compression = 1.0 / max(1.0 - duplicate_fraction, 1e-9) if duplicate_fraction < 1.0 else float("inf")
    LOGGER.info("Dashboard render complete for customer=%s sales_date=%s", customer, sales_date)
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DCO TabPFN Dashboard · {customer} · {sales_date}</title>
  <style>
    :root {{
      --paper: #f4efe7;
      --ink: #1f2933;
      --teal: #2a9d8f;
      --ember: #e76f51;
      --sand: #e9c46a;
      --panel: #fffdf9;
      --line: rgba(31, 41, 51, 0.12);
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(233, 196, 106, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(42, 157, 143, 0.18), transparent 24%),
        linear-gradient(180deg, #f8f3ea 0%, #f4efe7 100%);
    }}
    main {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
      align-items: end;
    }}
    .hero-copy h1 {{
      margin: 0 0 12px;
      font-family: "Iowan Old Style", "Palatino", serif;
      font-size: 44px;
      line-height: 1;
    }}
    .hero-copy p {{
      margin: 0;
      max-width: 760px;
      font-size: 17px;
      line-height: 1.55;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .stat-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 8px 28px rgba(31, 41, 51, 0.06);
    }}
    .stat-card {{
      padding: 16px 18px;
    }}
    .stat-card .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(31, 41, 51, 0.6);
    }}
    .stat-card .value {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .metrics-panel {{
      padding: 18px;
    }}
    .metrics-panel h2 {{
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .metrics-panel dl {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px 12px;
      margin: 0;
    }}
    .metrics-panel dt {{
      color: rgba(31, 41, 51, 0.7);
    }}
    .metrics-panel dd {{
      margin: 0;
      font-weight: 600;
    }}
    .panel {{
      padding: 16px;
      margin-bottom: 20px;
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
    }}
    .gallery-card {{
      margin: 0;
      background: rgba(255,255,255,0.8);
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid var(--line);
    }}
    .gallery-card img {{
      display: block;
      width: 100%;
      height: 220px;
      object-fit: cover;
      background: #fff;
    }}
    .gallery-card figcaption {{
      padding: 10px 12px;
      font-size: 13px;
      text-transform: capitalize;
    }}
    @media (max-width: 960px) {{
      .hero {{
        grid-template-columns: 1fr;
      }}
      .hero-copy h1 {{
        font-size: 34px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-copy">
        <h1>DCO TabPFN 2.5 Dashboard</h1>
        <p>
          Raw DCO fare rows for <strong>{customer}</strong> on <strong>{sales_date}</strong>,
          encoded twice with the same foundation model family: once as a pretrained regressor
          and once after task-specific fine-tuning on <code>price_inc</code>. The dashboard compares
          both embedding spaces directly and keeps route, fare, and date views grounded in the
          original DCO columns.
        </p>
      </div>
      <div class="panel">
        <h2>Run Context</h2>
        <p style="margin:0 0 10px;"><strong>Hours:</strong> {", ".join(hours_present)}</p>
        <p style="margin:0 0 10px;"><strong>Parquet files:</strong> {parquet_file_count:,}</p>
        <p style="margin:0;"><strong>Visualization rows:</strong> {total_points:,}</p>
      </div>
    </section>

    <section class="card-grid">
      <div class="stat-card"><div class="label">Total Rows</div><div class="value">{total_rows:,}</div></div>
      <div class="stat-card"><div class="label">Train Context</div><div class="value">{train_rows:,}</div></div>
      <div class="stat-card"><div class="label">Embedded Rows</div><div class="value">{embedded_rows:,}</div></div>
      <div class="stat-card"><div class="label">Dashboard Points</div><div class="value">{total_points:,}</div></div>
      <div class="stat-card"><div class="label">Pretrained Segments</div><div class="value">{metrics.get("pretrained_segment_count", 0)}</div></div>
      <div class="stat-card"><div class="label">Fine-tuned Segments</div><div class="value">{metrics.get("finetuned_segment_count", 0)}</div></div>
      <div class="stat-card"><div class="label">Pretrained Trust</div><div class="value">{metrics.get("pretrained_projection_trustworthiness", 0.0):.3f}</div></div>
      <div class="stat-card"><div class="label">Fine-tuned Trust</div><div class="value">{metrics.get("finetuned_projection_trustworthiness", 0.0):.3f}</div></div>
    </section>

    <section class="metrics-grid">
      <div class="panel metrics-panel">
        <h2>Pretrained Branch</h2>
        <dl>
          <dt>Version</dt><dd>{pretrained_metrics.get("version", "2.5")}</dd>
          <dt>Device</dt><dd>{pretrained_metrics.get("device", "unknown")}</dd>
          <dt>Estimators</dt><dd>{pretrained_metrics.get("n_estimators", "n/a")}</dd>
          <dt>Validation RMSE</dt><dd>{pretrained_metrics.get("rmse") or "n/a"}</dd>
          <dt>Validation MAE</dt><dd>{pretrained_metrics.get("mae") or "n/a"}</dd>
        </dl>
      </div>
      <div class="panel metrics-panel">
        <h2>Fine-tuned Branch</h2>
        <dl>
          <dt>Version</dt><dd>{finetuned_metrics.get("version", "2.5")}</dd>
          <dt>Device</dt><dd>{finetuned_metrics.get("device", "unknown")}</dd>
          <dt>Epochs</dt><dd>{finetuned_metrics.get("epochs", "n/a")}</dd>
          <dt>Estimators</dt><dd>{finetuned_metrics.get("n_estimators_final_inference", "n/a")}</dd>
          <dt>Validation RMSE</dt><dd>{finetuned_metrics.get("rmse") or "n/a"}</dd>
          <dt>Validation MAE</dt><dd>{finetuned_metrics.get("mae") or "n/a"}</dd>
        </dl>
      </div>
      <div class="panel metrics-panel">
        <h2>Representative Sampling</h2>
        <dl>
          <dt>Train metro coverage</dt><dd>{_format_metric_value(train_quality.get("metro_market_coverage"), percentage=True)}</dd>
          <dt>Viz metro coverage</dt><dd>{_format_metric_value(viz_quality.get("metro_market_coverage"), percentage=True)}</dd>
          <dt>Top airport coverage</dt><dd>{_format_metric_value(train_quality.get("top_airport_market_coverage"), percentage=True)}</dd>
          <dt>Trip type abs error</dt><dd>{_format_metric_value(train_quality.get("trip_type_abs_error"))}</dd>
          <dt>Carrier abs error</dt><dd>{_format_metric_value(train_quality.get("carrier_top_abs_error"))}</dd>
          <dt>Low-fare share delta</dt><dd>{_format_metric_value(train_quality.get("low_price_share_delta"), percentage=True)}</dd>
          <dt>Dedup compression</dt><dd>{_format_metric_value(duplicate_compression)}</dd>
        </dl>
      </div>
    </section>

    <section class="panel">
      <h2>Embedding Comparison</h2>
      {embedding_html}
    </section>

    <section class="panel">
      <h2>Route Network</h2>
      {route_html}
    </section>

    <section class="panel">
      <h2>Fare Calendar</h2>
      {calendar_html}
    </section>

    <section class="panel">
      <h2>Market Matrix</h2>
      {matrix_html}
    </section>

    <section class="panel">
      <h2>Segment Fingerprints</h2>
      {fingerprint_html}
    </section>

    <section class="panel">
      <h2>Branch Agreement</h2>
      {agreement_html}
    </section>

    <section class="panel">
      <h2>Static Image Pack</h2>
      <div class="gallery">
        {gallery}
      </div>
    </section>
  </main>
</body>
</html>
"""
