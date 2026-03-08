from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Iterable

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots

from dco_visualize.model import load_city_lookup

EMBEDDING_PNG = "embedding_density.png"
FLOW_PNG = "metro_flow_map.png"
CALENDAR_PNG = "fare_calendar.png"
MATRIX_PNG = "market_matrix.png"
FINGERPRINT_PNG = "segment_fingerprint.png"


def _parse_aggregate_views(aggregate_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    views: dict[str, pd.DataFrame] = {}
    for view, subset in aggregate_frame.groupby("view", dropna=False):
        views[str(view)] = subset.reset_index(drop=True)
    return views


def _figure_to_html(figure: go.Figure, *, include_js: bool) -> str:
    return pio.to_html(figure, include_plotlyjs=include_js, full_html=False, config={"displaylogo": False, "responsive": True})


def _image_to_data_uri(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _color_key(labels: Iterable[str]) -> dict[str, str]:
    palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6d597a", "#355070", "#b56576", "#43aa8b", "#577590"]
    return {label: palette[index % len(palette)] for index, label in enumerate(sorted(labels))}


def build_visualization_frame(frame: pd.DataFrame, viz_rows: int, random_seed: int) -> pd.DataFrame:
    if len(frame) <= viz_rows:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=viz_rows, random_state=random_seed).sort_values(["segment_id", "market_token"], kind="stable").reset_index(drop=True)


def _build_embedding_diagnostics_figure(frame: pd.DataFrame, hover_columns: Iterable[str], customer: str, sales_date: str) -> go.Figure:
    hover_columns = [column for column in hover_columns if column in frame.columns]
    custom_data = frame[hover_columns].astype(str).to_numpy() if hover_columns else None
    hover_template = "<br>".join([f"{column}: %{{customdata[{index}]}}" for index, column in enumerate(hover_columns)])
    figure = go.Figure()
    for segment_id, subset in frame.groupby("segment_id", dropna=False):
        figure.add_trace(
            go.Scattergl(
                x=subset["layout_x"],
                y=subset["layout_y"],
                mode="markers",
                name=f"Segment {segment_id}",
                marker={"size": 4, "opacity": 0.55},
                customdata=subset[hover_columns].astype(str).to_numpy() if hover_columns else None,
                hovertemplate=(
                    "segment=%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>" + hover_template + "<extra></extra>"
                    if hover_columns
                    else "segment=%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>"
                ),
                text=[str(segment_id)] * len(subset),
            )
        )
    figure.update_layout(
        title=f"Embedding Diagnostics · {customer} · {sales_date}",
        template="plotly_white",
        legend_title_text="Segment",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        xaxis_title="densMAP 1",
        yaxis_title="densMAP 2",
    )
    return figure


def _build_flow_figure(flow_frame: pd.DataFrame, customer: str, sales_date: str) -> go.Figure:
    if flow_frame.empty:
        return go.Figure().update_layout(title="Metro Flow Map", template="plotly_white")
    lookup = load_city_lookup().set_index("code")
    top_flows = flow_frame.sort_values("count", ascending=False).head(80).copy()
    top_flows = top_flows[top_flows["key_1"].isin(lookup.index) & top_flows["key_2"].isin(lookup.index)].copy()
    if top_flows.empty:
        return go.Figure().update_layout(title="Metro Flow Map", template="plotly_white")

    figure = go.Figure()
    max_count = max(float(top_flows["count"].max()), 1.0)
    for row in top_flows.itertuples(index=False):
        origin = lookup.loc[str(row.key_1)]
        destination = lookup.loc[str(row.key_2)]
        width = 1.0 + (5.0 * math.sqrt(float(row.count) / max_count))
        figure.add_trace(
            go.Scattergeo(
                lon=[float(origin.longitude), float(destination.longitude)],
                lat=[float(origin.latitude), float(destination.latitude)],
                mode="lines",
                line={"width": width, "color": "#d1495b"},
                opacity=0.45,
                hovertemplate=(
                    f"{row.key_1} → {row.key_2}<br>"
                    f"rows={int(row.count):,}<br>"
                    f"mean fare=${float(row.mean_price):,.0f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    nodes = pd.concat(
        [
            top_flows[["key_1", "count"]].rename(columns={"key_1": "code"}),
            top_flows[["key_2", "count"]].rename(columns={"key_2": "code"}),
        ],
        ignore_index=True,
    )
    nodes = nodes.groupby("code", as_index=False)["count"].sum()
    nodes = nodes[nodes["code"].isin(lookup.index)].copy()
    nodes["latitude"] = nodes["code"].map(lookup["latitude"])
    nodes["longitude"] = nodes["code"].map(lookup["longitude"])
    figure.add_trace(
        go.Scattergeo(
            lon=nodes["longitude"],
            lat=nodes["latitude"],
            mode="markers",
            text=nodes["code"],
            marker={
                "size": 6 + 16 * np.sqrt(nodes["count"] / max(float(nodes["count"].max()), 1.0)),
                "color": nodes["count"],
                "colorscale": "YlOrRd",
                "line": {"width": 0.5, "color": "#264653"},
                "showscale": True,
                "colorbar": {"title": "Rows"},
            },
            hovertemplate="%{text}<br>rows=%{marker.color:,}<extra></extra>",
            name="metros",
        )
    )
    figure.update_geos(
        projection_type="natural earth",
        showcountries=True,
        showland=True,
        landcolor="#f7efe5",
        oceancolor="#e8f1f2",
        showocean=True,
        coastlinecolor="#a8c7cc",
        lataxis_showgrid=True,
        lonaxis_showgrid=True,
    )
    figure.update_layout(
        title=f"Metro Flow Map · {customer} · {sales_date}",
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        height=640,
    )
    return figure


def _calendar_pivots(calendar_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    advance = calendar_frame[calendar_frame["key_3"] == "advance_purchase"].copy()
    ret = calendar_frame[calendar_frame["key_3"] == "return_gap"].copy()
    advance_pivot = (
        advance.pivot_table(index="key_2", columns="key_1", values="mean_price", aggfunc="mean")
        .sort_index(axis=1)
        .sort_index(axis=0)
    )
    return_pivot = (
        ret.pivot_table(index="key_2", columns="key_1", values="mean_price", aggfunc="mean")
        .sort_index(axis=1)
        .sort_index(axis=0)
    )
    return advance_pivot, return_pivot


def _build_calendar_figure(calendar_frame: pd.DataFrame, customer: str, sales_date: str) -> go.Figure:
    advance_pivot, return_pivot = _calendar_pivots(calendar_frame)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Departure x Advance Purchase", "Departure x Return Gap"))
    if not advance_pivot.empty:
        figure.add_trace(
            go.Heatmap(
                z=advance_pivot.to_numpy(),
                x=advance_pivot.columns.astype(str).tolist(),
                y=advance_pivot.index.astype(str).tolist(),
                colorscale="YlGnBu",
                colorbar={"title": "Mean Fare"},
                hovertemplate="dep=%{x}<br>bucket=%{y}<br>mean fare=$%{z:.0f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if not return_pivot.empty:
        figure.add_trace(
            go.Heatmap(
                z=return_pivot.to_numpy(),
                x=return_pivot.columns.astype(str).tolist(),
                y=return_pivot.index.astype(str).tolist(),
                colorscale="Magma",
                showscale=False,
                hovertemplate="dep=%{x}<br>gap=%{y}<br>mean fare=$%{z:.0f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    figure.update_layout(
        title=f"Fare Calendar Surfaces · {customer} · {sales_date}",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 70, "b": 40},
        height=520,
    )
    return figure


def _build_market_matrix_figure(flow_frame: pd.DataFrame, customer: str, sales_date: str) -> go.Figure:
    if flow_frame.empty:
        return go.Figure().update_layout(title="Market Matrix", template="plotly_white")
    working = flow_frame.copy()
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
            colorscale="Sunset",
            customdata=counts.to_numpy(),
            hovertemplate="origin=%{y}<br>destination=%{x}<br>mean fare=$%{z:.0f}<br>rows=%{customdata:,}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Market Matrix · {customer} · {sales_date}",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        xaxis_title="Destination Metro",
        yaxis_title="Origin Metro",
    )
    return figure


def _build_segment_fingerprint_figure(fingerprint_frame: pd.DataFrame, customer: str, sales_date: str) -> go.Figure:
    if fingerprint_frame.empty:
        return go.Figure().update_layout(title="Segment Fingerprints", template="plotly_white")
    working = fingerprint_frame.copy()
    working["label"] = working["key_1"].astype(str) + " · " + working["key_2"].astype(str)
    top_labels = (
        working.assign(abs_value=working["value"].abs())
        .sort_values("abs_value", ascending=False)
        .head(24)["label"]
        .tolist()
    )
    working = working[working["label"].isin(top_labels)]
    pivot = (
        working.pivot_table(index="label", columns="segment_id", values="value", aggfunc="mean")
        .sort_index(axis=1)
        .fillna(0.0)
    )
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=[f"Segment {column}" for column in pivot.columns.tolist()],
            y=pivot.index.astype(str).tolist(),
            colorscale="RdBu",
            zmid=0.0,
            hovertemplate="%{y}<br>%{x}<br>lift=%{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Segment Fingerprints · {customer} · {sales_date}",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        xaxis_title="Segment",
        yaxis_title="Feature Lift",
        height=720,
    )
    return figure


def _build_segment_size_figure(segment_sizes: pd.DataFrame) -> go.Figure:
    if segment_sizes.empty:
        return go.Figure().update_layout(title="Segment Sizes", template="plotly_white")
    working = segment_sizes.copy().sort_values("segment_id")
    figure = go.Figure(
        go.Bar(
            x=[f"Segment {segment_id}" for segment_id in working["segment_id"]],
            y=working["count"],
            marker={"color": "#264653"},
            hovertemplate="%{x}<br>rows=%{y:,}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Segment Size Balance",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        xaxis_title="Segment",
        yaxis_title="Rows",
    )
    return figure


def render_embedding_dashboard(
    frame: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    hover_columns: Iterable[str],
    customer: str,
    sales_date: str,
    total_points: int,
    total_rows: int,
    parquet_file_count: int,
    hours_present: list[str],
    metrics: dict[str, object],
    embedded_image_uri: str,
) -> str:
    views = _parse_aggregate_views(aggregate_frame)
    flow_frame = views.get("metro_flow", pd.DataFrame())
    calendar_frame = views.get("fare_calendar", pd.DataFrame())
    fingerprint_frame = views.get("segment_fingerprint", pd.DataFrame())
    segment_sizes = views.get("segment_size", pd.DataFrame())

    figures = [
        _build_flow_figure(flow_frame, customer, sales_date),
        _build_calendar_figure(calendar_frame, customer, sales_date),
        _build_market_matrix_figure(flow_frame, customer, sales_date),
        _build_segment_fingerprint_figure(fingerprint_frame, customer, sales_date),
        _build_embedding_diagnostics_figure(frame, hover_columns, customer, sales_date),
        _build_segment_size_figure(segment_sizes),
    ]
    figure_html = [
        _figure_to_html(figure, include_js=(index == 0)) for index, figure in enumerate(figures)
    ]

    summary_cards = [
        ("Embedded Rows", f"{total_points:,}"),
        ("Viz Sample", f"{len(frame):,}"),
        ("Total Day Rows", f"{total_rows:,}"),
        ("Segments", f"{int(metrics.get('segment_count', 0)):,}"),
        ("Noise", f"{float(metrics.get('noise_fraction', 0.0)):.1%}"),
        ("Projection", str(metrics.get("projection", {}).get("name", "densmap"))),
    ]
    summary_html = "".join(
        f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>'
        for label, value in summary_cards
    )
    hours_html = ", ".join(hours_present)
    projection_details = metrics.get("projection", {})
    trustworthiness_score = metrics.get("projection_trustworthiness")
    trustworthiness_text = "n/a" if trustworthiness_score is None else f"{float(trustworthiness_score):.3f}"

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DCO Dashboard · {customer} · {sales_date}</title>
    <style>
      :root {{
        --bg: #f3ede1;
        --panel: rgba(255, 251, 245, 0.96);
        --ink: #16242c;
        --muted: #5e6f78;
        --accent: #c44536;
        --accent-soft: #f4a261;
        --line: rgba(22, 36, 44, 0.12);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        color: var(--ink);
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        background:
          radial-gradient(circle at 0% 0%, rgba(196,69,54,0.14), transparent 32%),
          radial-gradient(circle at 100% 0%, rgba(42,157,143,0.1), transparent 25%),
          linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
      }}
      .page {{
        max-width: 1540px;
        margin: 0 auto;
        padding: 28px 22px 48px;
      }}
      .hero {{
        display: grid;
        gap: 8px;
        margin-bottom: 22px;
      }}
      .eyebrow {{
        color: var(--accent);
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.76rem;
      }}
      .hero h1 {{
        margin: 0;
        font-size: clamp(2.1rem, 4vw, 4rem);
        line-height: 0.92;
      }}
      .hero p {{
        margin: 0;
        color: var(--muted);
        max-width: 960px;
        font-size: 1rem;
      }}
      .stats {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin-bottom: 18px;
      }}
      .stat-card, .panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: 0 18px 50px rgba(33, 29, 24, 0.08);
      }}
      .stat-card {{
        padding: 16px 18px;
      }}
      .stat-label {{
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
      }}
      .stat-value {{
        margin-top: 6px;
        font-size: 1.5rem;
        font-weight: 700;
      }}
      .meta {{
        display: grid;
        grid-template-columns: 1.1fr 0.9fr;
        gap: 18px;
        margin-bottom: 18px;
      }}
      .panel {{
        padding: 18px;
      }}
      .panel h2 {{
        margin: 0 0 10px 0;
        font-size: 1.15rem;
      }}
      .meta-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px 18px;
        color: var(--muted);
      }}
      .meta-grid strong {{
        color: var(--ink);
      }}
      .hero-image {{
        width: 100%;
        border-radius: 18px;
        border: 1px solid var(--line);
        display: block;
      }}
      .charts {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
        gap: 18px;
      }}
      .chart-card {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: 0 18px 50px rgba(33, 29, 24, 0.08);
        overflow: hidden;
        padding: 8px;
      }}
      @media (max-width: 900px) {{
        .meta {{ grid-template-columns: 1fr; }}
      }}
      @media (max-width: 640px) {{
        .page {{ padding: 18px 12px 32px; }}
        .charts {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <div class="eyebrow">DCO Visualize v2</div>
        <h1>Fare surfaces, route flows, and segment structure for {customer}</h1>
        <p>The dashboard now treats DCO as airfare search inventory instead of a generic tabular point cloud. The density panel is a densMAP view over a stratified visualization sample, while the map, calendar, matrix, and fingerprint panels summarize the full-day embedding output.</p>
      </section>
      <section class="stats">{summary_html}</section>
      <section class="meta">
        <div class="panel">
          <h2>Run Metadata</h2>
          <div class="meta-grid">
            <div><strong>Sales Date:</strong> {sales_date}</div>
            <div><strong>Parquet Files:</strong> {parquet_file_count:,}</div>
            <div><strong>Hours Present:</strong> {len(hours_present)}</div>
            <div><strong>Trustworthiness:</strong> {trustworthiness_text}</div>
            <div><strong>Projection:</strong> {projection_details.get("name", "densmap")}</div>
            <div><strong>Neighbors:</strong> {projection_details.get("n_neighbors", "n/a")}</div>
            <div><strong>Encoder:</strong> {metrics.get("encoder_backend", "ft_transformer_contrastive")}</div>
            <div><strong>Segmenter:</strong> {metrics.get("segment_method", "hdbscan")}</div>
          </div>
          <p style="margin:12px 0 0 0;color:var(--muted);"><strong>Available hours:</strong> {hours_html}</p>
        </div>
        <div class="panel">
          <h2>Embedding Density</h2>
          <img class="hero-image" src="{embedded_image_uri}" alt="Embedding density view" />
        </div>
      </section>
      <section class="charts">
        <div class="chart-card">{figure_html[0]}</div>
        <div class="chart-card">{figure_html[1]}</div>
        <div class="chart-card">{figure_html[2]}</div>
        <div class="chart-card">{figure_html[3]}</div>
        <div class="chart-card">{figure_html[4]}</div>
        <div class="chart-card">{figure_html[5]}</div>
      </section>
    </main>
  </body>
</html>
"""


def render_embedding_html(
    frame: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    hover_columns: Iterable[str],
    customer: str,
    sales_date: str,
    total_points: int,
    metrics: dict[str, object],
) -> str:
    return render_embedding_dashboard(
        frame=frame,
        aggregate_frame=aggregate_frame,
        hover_columns=hover_columns,
        customer=customer,
        sales_date=sales_date,
        total_points=total_points,
        total_rows=total_points,
        parquet_file_count=1,
        hours_present=[],
        metrics=metrics,
        embedded_image_uri="",
    )


def _save_datashader_density(frame: pd.DataFrame, output_path: str | Path) -> None:
    working = frame.copy()
    working["segment_label"] = pd.Categorical(working["segment_id"].astype("string").fillna("missing"))
    canvas = ds.Canvas(plot_width=1200, plot_height=900)
    aggregate = canvas.points(working, "layout_x", "layout_y", ds.count_cat("segment_label"))
    image = tf.shade(aggregate, color_key=_color_key(working["segment_label"].unique().tolist()), how="eq_hist")
    image = tf.set_background(image, "#fcf9f2")
    image.to_pil().save(output_path)


def _save_flow_png(flow_frame: pd.DataFrame, output_path: str | Path) -> None:
    lookup = load_city_lookup().set_index("code")
    top_flows = flow_frame.sort_values("count", ascending=False).head(60).copy()
    top_flows = top_flows[top_flows["key_1"].isin(lookup.index) & top_flows["key_2"].isin(lookup.index)].copy()
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_facecolor("#eef4f5")
    if top_flows.empty:
        ax.text(0.5, 0.5, "No metro flow data", ha="center", va="center")
        ax.axis("off")
    else:
        max_count = max(float(top_flows["count"].max()), 1.0)
        for row in top_flows.itertuples(index=False):
            origin = lookup.loc[str(row.key_1)]
            destination = lookup.loc[str(row.key_2)]
            width = 0.25 + 2.0 * math.sqrt(float(row.count) / max_count)
            ax.plot([origin.longitude, destination.longitude], [origin.latitude, destination.latitude], color="#d1495b", alpha=0.25, linewidth=width)
        ax.scatter(lookup["longitude"], lookup["latitude"], s=3, c="#264653", alpha=0.25)
    ax.set_title("Metro Flow Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_calendar_png(calendar_frame: pd.DataFrame, output_path: str | Path) -> None:
    advance_pivot, return_pivot = _calendar_pivots(calendar_frame)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivots = [(advance_pivot, "Departure x Advance Purchase"), (return_pivot, "Departure x Return Gap")]
    for axis, (pivot, title) in zip(axes, pivots):
        axis.set_title(title)
        if pivot.empty:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
            axis.axis("off")
            continue
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns.astype(str).tolist(), rotation=90, fontsize=7)
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels(pivot.index.astype(str).tolist(), fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_matrix_png(flow_frame: pd.DataFrame, output_path: str | Path) -> None:
    working = flow_frame.copy()
    fig, ax = plt.subplots(figsize=(11, 8))
    if working.empty:
        ax.text(0.5, 0.5, "No market matrix data", ha="center", va="center")
        ax.axis("off")
    else:
        top_origins = working.groupby("key_1")["count"].sum().sort_values(ascending=False).head(12).index
        top_destinations = working.groupby("key_2")["count"].sum().sort_values(ascending=False).head(12).index
        working = working[working["key_1"].isin(top_origins) & working["key_2"].isin(top_destinations)]
        pivot = working.pivot_table(index="key_1", columns="key_2", values="mean_price", aggfunc="mean").fillna(0.0)
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="inferno")
        ax.set_title("Market Matrix")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.astype(str).tolist(), rotation=90)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(str).tolist())
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_fingerprint_png(fingerprint_frame: pd.DataFrame, output_path: str | Path) -> None:
    working = fingerprint_frame.copy()
    cmap = LinearSegmentedColormap.from_list("lift", ["#2f6690", "#f4f1de", "#c44536"])
    fig, ax = plt.subplots(figsize=(10, 9))
    if working.empty:
        ax.text(0.5, 0.5, "No fingerprint data", ha="center", va="center")
        ax.axis("off")
    else:
        working["label"] = working["key_1"].astype(str) + " · " + working["key_2"].astype(str)
        top_labels = (
            working.assign(abs_value=working["value"].abs())
            .sort_values("abs_value", ascending=False)
            .head(24)["label"]
            .tolist()
        )
        working = working[working["label"].isin(top_labels)]
        pivot = working.pivot_table(index="label", columns="segment_id", values="value", aggfunc="mean").fillna(0.0)
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
        ax.set_title("Segment Fingerprints")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"S{column}" for column in pivot.columns.tolist()])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(str).tolist(), fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_dashboard_images(
    frame: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    output_dir: str | Path,
    customer: str,
    sales_date: str,
) -> dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    views = _parse_aggregate_views(aggregate_frame)

    embedding_path = output_root / EMBEDDING_PNG
    _save_datashader_density(frame, embedding_path)

    flow_path = output_root / FLOW_PNG
    _save_flow_png(views.get("metro_flow", pd.DataFrame()), flow_path)

    calendar_path = output_root / CALENDAR_PNG
    _save_calendar_png(views.get("fare_calendar", pd.DataFrame()), calendar_path)

    matrix_path = output_root / MATRIX_PNG
    _save_matrix_png(views.get("metro_flow", pd.DataFrame()), matrix_path)

    fingerprint_path = output_root / FINGERPRINT_PNG
    _save_fingerprint_png(views.get("segment_fingerprint", pd.DataFrame()), fingerprint_path)

    return {
        "embedding_density_png": str(embedding_path),
        "metro_flow_map_png": str(flow_path),
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
    total_points: int,
    total_rows: int,
    parquet_file_count: int,
    hours_present: list[str],
    metrics: dict[str, object],
    image_paths: dict[str, str],
) -> str:
    return render_embedding_dashboard(
        frame=frame,
        aggregate_frame=aggregate_frame,
        hover_columns=hover_columns,
        customer=customer,
        sales_date=sales_date,
        total_points=total_points,
        total_rows=total_rows,
        parquet_file_count=parquet_file_count,
        hours_present=hours_present,
        metrics=metrics,
        embedded_image_uri=_image_to_data_uri(image_paths["embedding_density_png"]),
    )
