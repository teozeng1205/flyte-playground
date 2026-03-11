from __future__ import annotations

import json
import logging
from typing import Iterable

import numpy as np
import pandas as pd
from plotly.offline.offline import get_plotlyjs

LOGGER = logging.getLogger(__name__)

FIELD_LABELS = {
    "carrier": "Carrier",
    "source": "Source",
    "trip_type": "Trip Type",
    "cabin": "Cabin",
    "stops": "Stops",
    "origin_metro": "Origin Metro",
    "destination_metro": "Destination Metro",
    "origin": "Origin Airport",
    "destination": "Destination Airport",
    "outbound_departure_date": "Departure Date",
    "inbound_departure_date": "Return Date",
    "advance_purchase": "Advance Purchase",
    "price_inc": "Fare",
    "pretrained_segment_id": "Pretrained Segment",
    "finetuned_segment_id": "Fine-tuned Segment",
}

CATEGORICAL_FILTER_FIELDS = [
    "carrier",
    "source",
    "trip_type",
    "cabin",
    "stops",
    "origin_metro",
    "destination_metro",
    "origin",
    "destination",
    "pretrained_segment_id",
    "finetuned_segment_id",
]


def build_visualization_frame(frame: pd.DataFrame, viz_rows: int, random_seed: int) -> pd.DataFrame:
    if len(frame) <= viz_rows:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=viz_rows, random_state=random_seed).reset_index(drop=True)


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
        ("carrier", "Carrier", 12),
        ("trip_type", "Trip Type", 8),
        ("cabin", "Cabin", 8),
        ("stops", "Stops", 8),
        ("source", "Source", 12),
        ("origin_metro", "Origin Metro", 16),
        ("destination_metro", "Destination Metro", 16),
        ("pretrained_segment_id", "Pretrained Segment", 8),
        ("finetuned_segment_id", "Fine-tuned Segment", 8),
    ]
    for column, label, top_k in categorical_candidates:
        if _interesting_categorical(frame, column):
            options.append({"key": column, "label": label, "kind": "categorical", "top_k": top_k})
    for column, label, number_format in [("price_inc", "Fare", "currency"), ("advance_purchase", "Advance Purchase", "number")]:
        if _interesting_numeric(frame, column):
            options.append({"key": column, "label": label, "kind": "numeric", "format": number_format})
    return options


def _embedding_dashboard_payload(frame: pd.DataFrame, hover_columns: Iterable[str]) -> dict[str, object]:
    color_options = _color_option_definitions(frame)
    default_color_key = next((option["key"] for option in color_options if option["key"] == "carrier"), None)
    if default_color_key is None and color_options:
        default_color_key = color_options[0]["key"]

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
            "origin",
            "destination",
            "outbound_departure_date",
            "price_inc",
        ]
        if field in frame.columns
    ]
    for field in hover_columns:
        if field in frame.columns and field not in hover_fields:
            hover_fields.append(field)
    hover_fields = hover_fields[:10]

    filter_fields = [field for field in CATEGORICAL_FILTER_FIELDS if _interesting_categorical(frame, field)]
    column_keys = {
        *hover_fields,
        *filter_fields,
        *(str(option["key"]) for option in color_options),
    }
    columns = {
        key: [_dashboard_value(value) for value in frame[key].tolist()]
        for key in column_keys
        if key in frame.columns
    }
    filter_options = {}
    for field in filter_fields:
        counts = frame[field].astype("string").fillna("missing").value_counts(dropna=False)
        filter_options[field] = counts.head(50).index.astype(str).tolist()

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
        "filter_fields": filter_fields,
        "filter_options": filter_options,
        "field_labels": FIELD_LABELS,
        "columns": columns,
        "color_options": color_options,
        "default_color_key": default_color_key,
    }


def render_standalone_dashboard(
    frame: pd.DataFrame,
    hover_columns: Iterable[str],
    customer: str,
    sales_date: str,
    profile: dict[str, object] | None,
    total_rows: int,
    parquet_file_count: int,
    hours_present: list[str],
    metrics: dict[str, object],
) -> str:
    LOGGER.info(
        "Rendering Qwen standalone dashboard for customer=%s sales_date=%s viz_rows=%d",
        customer,
        sales_date,
        len(frame),
    )
    payload_json = json.dumps(_embedding_dashboard_payload(frame, hover_columns)).replace("</", "<\\/")
    plotly_js = get_plotlyjs()
    pair_count = int(metrics.get("pair_count", 0))
    train_rows = int(metrics.get("train_rows", len(frame)))
    hours_label = ", ".join(hours_present[:6]) + ("..." if len(hours_present) > 6 else "")
    quality = (((profile or {}).get("representative_sampling") or {}).get("quality") or {})
    quality_text = ""
    if quality:
        viz_quality = quality.get("viz", {})
        quality_text = (
            f"Representative sampling retained {viz_quality.get('rows', len(frame)):,} viz rows with "
            f"{float(viz_quality.get('metro_market_coverage', 0.0)) * 100:.1f}% metro-market coverage."
        )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DCO Qwen3 Dashboard · {customer} · {sales_date}</title>
  <style>
    html {{ height: 100%; }}
    :root {{
      --paper: #f4efe7;
      --ink: #1f2933;
      --teal: #2a9d8f;
      --ember: #e76f51;
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
        radial-gradient(circle at top left, rgba(231, 111, 81, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(42, 157, 143, 0.16), transparent 24%),
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
    .control-stack {{ display: grid; gap: 14px; }}
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
    .view-switch {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      padding: 4px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      gap: 4px;
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
    }}
    .view-switch button.active {{
      background: linear-gradient(135deg, rgba(42, 157, 143, 0.16), rgba(231, 111, 81, 0.16));
      color: var(--ink);
    }}
    .filters-grid {{
      display: grid;
      gap: 10px;
      max-height: 360px;
      overflow: auto;
      padding-right: 4px;
    }}
    .summary-card {{
      padding: 14px 16px;
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(38, 70, 83, 0.08), rgba(42, 157, 143, 0.08));
      border: 1px solid rgba(38, 70, 83, 0.08);
    }}
    .summary-card dl {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 8px 12px;
      margin: 0;
      font-size: 13px;
    }}
    .summary-card dt {{
      font-weight: 700;
      color: rgba(31, 41, 51, 0.55);
    }}
    .summary-card dd {{ margin: 0; }}
    #plot {{
      width: 100%;
      height: calc(100vh - 170px);
      min-height: 680px;
    }}
    @media (max-width: 1180px) {{
      .workspace {{ grid-template-columns: 1fr; }}
      #plot {{ height: 72vh; min-height: 560px; }}
    }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <main>
    <section class="panel topbar">
      <div class="hero">
        <h1>DCO Qwen3 Embedding Dashboard</h1>
        <p>
          Standalone comparison of pretrained versus LoRA fine-tuned Qwen3 row embeddings for DCO airfare offers.
          Rows are serialized from raw DCO columns, embedded with <code>{metrics.get("model_id", "")}</code>,
          and projected into two manifolds. {quality_text}
        </p>
      </div>
      <div class="topbar-badges">
        <div class="metric-pill"><span class="pill-label">Customer</span><span class="pill-value">{customer}</span></div>
        <div class="metric-pill"><span class="pill-label">Sales Date</span><span class="pill-value">{sales_date}</span></div>
        <div class="metric-pill"><span class="pill-label">Train Rows</span><span class="pill-value">{train_rows:,}</span></div>
        <div class="metric-pill"><span class="pill-label">Pair Count</span><span class="pill-value">{pair_count:,}</span></div>
      </div>
    </section>
    <section class="workspace">
      <aside class="panel sidebar">
        <div class="control-stack">
          <div>
            <div class="sidebar-kicker">View</div>
            <div class="view-switch">
              <button id="compare-button" class="active" data-branch="compare">Compare</button>
              <button id="pretrained-button" data-branch="pretrained">Pretrained</button>
              <button id="finetuned-button" data-branch="finetuned">Fine-tuned</button>
            </div>
          </div>
          <div class="control-block">
            <label for="color-select">Color By</label>
            <select id="color-select"></select>
          </div>
          <div class="summary-card">
            <dl>
              <dt>Visible Rows</dt><dd id="summary-visible">0</dd>
              <dt>Branch View</dt><dd id="summary-branch">Compare</dd>
              <dt>Color Field</dt><dd id="summary-color">Carrier</dd>
              <dt>Filters</dt><dd id="summary-filters">None</dd>
              <dt>Total Rows</dt><dd>{total_rows:,}</dd>
              <dt>Parquet Files</dt><dd>{parquet_file_count:,}</dd>
              <dt>Hours</dt><dd>{hours_label or "n/a"}</dd>
            </dl>
          </div>
        </div>
        <div class="panel" style="padding:14px 16px; border-radius:16px; box-shadow:none;">
          <div class="sidebar-kicker">Category Filters</div>
          <div class="filters-grid" id="filters-grid"></div>
        </div>
      </aside>
      <section class="panel stage">
        <div class="stage-header">
          <div>
            <h2 style="margin:0 0 6px;">Embedding Comparison</h2>
            <p>Inspect the pretrained and fine-tuned Qwen3 manifolds, recolor by travel categories or fare, and filter with intersecting category controls.</p>
          </div>
          <div class="stage-badges">
            <span class="badge">Model: {metrics.get("model_id", "")}</span>
            <span class="badge">Eval Rows: {len(frame):,}</span>
            <span class="badge">Instruction-Aware Embeddings</span>
          </div>
        </div>
        <div id="plot"></div>
      </section>
    </section>
  </main>
  <script>
    const payload = {payload_json};
    const palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#7b6d8d", "#4c956c", "#577590", "#bc4749", "#5f0f40", "#0f4c5c", "#9c6644", "#4361ee", "#2b9348", "#e36414", "#6a4c93"];
    const numericScale = [
      [0.0, "#233d4d"],
      [0.25, "#2a9d8f"],
      [0.5, "#e9c46a"],
      [0.75, "#f4a261"],
      [1.0, "#e76f51"],
    ];
    const state = {{
      branch: "compare",
      colorKey: payload.default_color_key,
      filters: {{}},
    }};

    function labelFor(key) {{
      return payload.field_labels[key] || key;
    }}

    function optionByKey(key) {{
      return payload.color_options.find((option) => option.key === key);
    }}

    function buildHoverText(index) {{
      const lines = [];
      for (const field of payload.hover_fields) {{
        const values = payload.columns[field] || [];
        const value = values[index];
        if (value !== null && value !== undefined && String(value).length > 0) {{
          lines.push(`<b>${{labelFor(field)}}</b>: ${{value}}`);
        }}
      }}
      return lines.join("<br>");
    }}

    function filteredIndices() {{
      const indices = [];
      for (let index = 0; index < payload.row_count; index += 1) {{
        let keep = true;
        for (const [field, selected] of Object.entries(state.filters)) {{
          if (!selected) {{
            continue;
          }}
          const values = payload.columns[field] || [];
          if (String(values[index] ?? "missing") !== selected) {{
            keep = false;
            break;
          }}
        }}
        if (keep) {{
          indices.push(index);
        }}
      }}
      return indices;
    }}

    function categoricalColors(indices, key, topK) {{
      const values = payload.columns[key] || [];
      const counts = new Map();
      for (const index of indices) {{
        const value = String(values[index] ?? "missing");
        counts.set(value, (counts.get(value) || 0) + 1);
      }}
      const ranked = [...counts.entries()].sort((left, right) => right[1] - left[1]).slice(0, topK).map(([value]) => value);
      const colorMap = new Map(ranked.map((value, idx) => [value, palette[idx % palette.length]]));
      return indices.map((index) => colorMap.get(String(values[index] ?? "missing")) || "#c4c4c4");
    }}

    function numericColors(indices, key) {{
      const values = indices.map((index) => Number(payload.columns[key]?.[index]));
      const finite = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
      if (!finite.length) {{
        return values.map(() => 0);
      }}
      const low = finite[Math.floor(finite.length * 0.02)];
      const high = finite[Math.floor(finite.length * 0.98)];
      const denom = high > low ? (high - low) : 1.0;
      return values.map((value) => {{
        if (!Number.isFinite(value)) {{
          return low;
        }}
        return Math.max(low, Math.min(high, value));
      }});
    }}

    function branchTrace(branch, indices, colors, colorOption) {{
      const x = indices.map((index) => payload.points[branch].x[index]);
      const y = indices.map((index) => payload.points[branch].y[index]);
      const texts = indices.map((index) => buildHoverText(index));
      const marker = {{
        size: 6,
        opacity: 0.78,
      }};
      if (colorOption?.kind === "numeric") {{
        marker.color = colors;
        marker.colorscale = numericScale;
        marker.colorbar = state.branch === "compare" ? undefined : {{ title: labelFor(state.colorKey), len: 0.6 }};
      }} else {{
        marker.color = colors;
      }}
      return {{
        type: "scattergl",
        mode: "markers",
        x,
        y,
        text: texts,
        hovertemplate: "%{{text}}<extra></extra>",
        marker,
        showlegend: false,
        name: branch,
      }};
    }}

    function renderPlot() {{
      const indices = filteredIndices();
      const colorOption = optionByKey(state.colorKey);
      const colors = colorOption?.kind === "numeric"
        ? numericColors(indices, state.colorKey)
        : categoricalColors(indices, state.colorKey, colorOption?.top_k || 10);
      const traces = [];
      const annotations = [];
      const layout = {{
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(255,255,255,0.72)",
        margin: {{ l: 18, r: 18, t: 24, b: 16 }},
        hoverlabel: {{ bgcolor: "#fffdf9", font: {{ color: "#1f2933" }} }},
      }};

      if (state.branch === "compare") {{
        const pretrainedTrace = branchTrace("pretrained", indices, colors, colorOption);
        pretrainedTrace.xaxis = "x";
        pretrainedTrace.yaxis = "y";
        const finetunedTrace = branchTrace("finetuned", indices, colors, colorOption);
        finetunedTrace.xaxis = "x2";
        finetunedTrace.yaxis = "y2";
        traces.push(pretrainedTrace, finetunedTrace);
        layout.xaxis = {{ domain: [0.0, 0.46], title: "pretrained_layout_x", zeroline: false }};
        layout.yaxis = {{ domain: [0.0, 1.0], title: "pretrained_layout_y", zeroline: false }};
        layout.xaxis2 = {{ domain: [0.54, 1.0], title: "finetuned_layout_x", zeroline: false }};
        layout.yaxis2 = {{ domain: [0.0, 1.0], title: "finetuned_layout_y", zeroline: false }};
        annotations.push(
          {{ text: "Pretrained Qwen3", x: 0.23, y: 1.04, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 16 }} }},
          {{ text: "Fine-tuned Qwen3", x: 0.77, y: 1.04, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 16 }} }},
        );
      }} else {{
        traces.push(branchTrace(state.branch, indices, colors, colorOption));
        const title = state.branch === "pretrained" ? "Pretrained Qwen3" : "Fine-tuned Qwen3";
        layout.xaxis = {{ title: `${{state.branch}}_layout_x`, zeroline: false }};
        layout.yaxis = {{ title: `${{state.branch}}_layout_y`, zeroline: false }};
        annotations.push({{ text: title, x: 0.5, y: 1.04, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 16 }} }});
      }}

      layout.annotations = annotations;
      Plotly.react("plot", traces, layout, {{ displaylogo: false, responsive: true }});

      document.getElementById("summary-visible").textContent = indices.length.toLocaleString();
      document.getElementById("summary-branch").textContent =
        state.branch === "compare" ? "Compare" : (state.branch === "pretrained" ? "Pretrained" : "Fine-tuned");
      document.getElementById("summary-color").textContent = labelFor(state.colorKey);
      const activeFilters = Object.entries(state.filters).filter(([, value]) => value).map(([field, value]) => `${{labelFor(field)}}=${{value}}`);
      document.getElementById("summary-filters").textContent = activeFilters.length ? activeFilters.join(" · ") : "None";
    }}

    function buildControls() {{
      const colorSelect = document.getElementById("color-select");
      for (const option of payload.color_options) {{
        const element = document.createElement("option");
        element.value = option.key;
        element.textContent = option.label;
        if (option.key === state.colorKey) {{
          element.selected = true;
        }}
        colorSelect.appendChild(element);
      }}
      colorSelect.addEventListener("change", (event) => {{
        state.colorKey = event.target.value;
        renderPlot();
      }});

      const filtersGrid = document.getElementById("filters-grid");
      for (const field of payload.filter_fields) {{
        const wrapper = document.createElement("div");
        wrapper.className = "control-block";
        const label = document.createElement("label");
        label.textContent = labelFor(field);
        label.htmlFor = `filter-${{field}}`;
        const select = document.createElement("select");
        select.id = `filter-${{field}}`;
        const anyOption = document.createElement("option");
        anyOption.value = "";
        anyOption.textContent = "All";
        select.appendChild(anyOption);
        for (const value of payload.filter_options[field] || []) {{
          const option = document.createElement("option");
          option.value = value;
          option.textContent = value;
          select.appendChild(option);
        }}
        select.addEventListener("change", (event) => {{
          state.filters[field] = event.target.value || "";
          renderPlot();
        }});
        wrapper.appendChild(label);
        wrapper.appendChild(select);
        filtersGrid.appendChild(wrapper);
      }}

      for (const button of document.querySelectorAll(".view-switch button")) {{
        button.addEventListener("click", () => {{
          state.branch = button.dataset.branch;
          for (const peer of document.querySelectorAll(".view-switch button")) {{
            peer.classList.toggle("active", peer === button);
          }}
          renderPlot();
        }});
      }}
    }}

    buildControls();
    renderPlot();
  </script>
</body>
</html>
"""
