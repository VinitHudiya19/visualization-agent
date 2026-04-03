"""
LLM-powered Plotly spec generation via Groq API.
Production guarantees:
  - generate_spec() NEVER returns None — always returns a valid Plotly spec
  - 2 LLM attempts, then deterministic fallback spec built from real data
  - compute_data_stats() uses ALL rows (not just sample)
  - _get_client() validated at call time with clear error if key missing
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter

from fastapi import HTTPException
from groq import Groq

from app.config import settings
from app.utils.color_palettes import get_palette, get_background

logger = logging.getLogger("viz-agent")


# ── Client ────────────────────────────────────────────────────────────────────

def _get_client() -> Groq:
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")
    return Groq(api_key=settings.GROQ_API_KEY)


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_data_stats(columns: list[dict], rows: list[dict]) -> dict:
    """
    Compute summary statistics from ALL rows.
    Numeric: min, max, mean, median, std, count
    Categorical: unique_count, top_values (up to 15)
    Datetime: min, max, count
    """
    stats: dict = {}
    if not rows:
        return stats

    for col in columns:
        name = col["name"]
        sem = col.get("semantic", "other")
        values = [r.get(name) for r in rows if r.get(name) is not None]
        if not values:
            continue

        if sem == "numeric":
            try:
                nums = sorted(float(v) for v in values)
                n = len(nums)
                mean = sum(nums) / n
                mid = n // 2
                median = nums[mid] if n % 2 else (nums[mid - 1] + nums[mid]) / 2
                variance = sum((x - mean) ** 2 for x in nums) / n
                std = variance ** 0.5
                stats[name] = {
                    "type": "numeric",
                    "min": round(nums[0], 2),
                    "max": round(nums[-1], 2),
                    "mean": round(mean, 2),
                    "median": round(median, 2),
                    "std": round(std, 2),
                    "count": n,
                    "range": round(nums[-1] - nums[0], 2),
                }
            except (ValueError, TypeError):
                pass

        elif sem == "categorical":
            counts = Counter(str(v) for v in values)
            top = counts.most_common(15)
            stats[name] = {
                "type": "categorical",
                "unique_count": len(counts),
                "top_values": [{"value": v, "count": c} for v, c in top],
            }

        elif sem == "datetime":
            str_vals = sorted(str(v) for v in values)
            stats[name] = {
                "type": "datetime",
                "min": str_vals[0],
                "max": str_vals[-1],
                "count": len(str_vals),
            }

    return stats


# ── Fallback spec builder ─────────────────────────────────────────────────────

def _build_fallback_spec(
    chart_type: str,
    task: str,
    columns: list[dict],
    rows: list[dict],
    color_scheme: str,
) -> dict:
    """
    Deterministic Plotly spec built directly from data — no LLM needed.
    Always produces a valid, renderable chart. Used when LLM fails.
    """
    palette = get_palette(color_scheme)
    bg = get_background(color_scheme)

    numerics = [c["name"] for c in columns if c.get("semantic") == "numeric"]
    cats = [c["name"] for c in columns if c.get("semantic") == "categorical"]
    dts = [c["name"] for c in columns if c.get("semantic") == "datetime"]

    title = task[:80] if task else chart_type.title()
    layout_base: dict = {
        "title": {"text": title, "font": {"size": 18, "color": bg["text"]}},
        "plot_bgcolor": bg["plot_bg"],
        "paper_bgcolor": bg["paper_bg"],
        "font": {"family": "Inter, -apple-system, sans-serif", "color": bg["text"]},
        "margin": {"l": 70, "r": 40, "t": 80, "b": 70},
        "legend": {"orientation": "h", "y": -0.2, "x": 0.5, "xanchor": "center"},
        "xaxis": {"gridcolor": bg["grid"], "zeroline": False},
        "yaxis": {"gridcolor": bg["grid"], "zeroline": False},
    }

    # ── Bar / Line / Area ──────────────────────────────────────────────────────
    if chart_type in ("bar", "line", "area") and rows:
        x_col = dts[0] if dts else (cats[0] if cats else None)
        y_col = numerics[0] if numerics else None

        if x_col and y_col:
            # Aggregate: sum y per x value
            agg: dict = {}
            for row in rows:
                xv = str(row.get(x_col, ""))
                yv = row.get(y_col)
                if xv and yv is not None:
                    try:
                        agg[xv] = agg.get(xv, 0.0) + float(yv)
                    except (ValueError, TypeError):
                        pass
            x_vals = list(agg.keys())
            y_vals = [round(agg[k], 2) for k in x_vals]

            if chart_type == "bar":
                trace = {
                    "type": "bar", "x": x_vals, "y": y_vals,
                    "marker": {"color": palette[0]},
                    "hovertemplate": "<b>%{x}</b><br>" + y_col + ": %{y:,.2f}<extra></extra>",
                }
            elif chart_type == "area":
                trace = {
                    "type": "scatter", "mode": "lines", "fill": "tozeroy",
                    "x": x_vals, "y": y_vals,
                    "line": {"color": palette[0], "width": 2},
                    "fillcolor": palette[0] + "33",
                    "hovertemplate": "<b>%{x}</b><br>" + y_col + ": %{y:,.2f}<extra></extra>",
                }
            else:  # line
                trace = {
                    "type": "scatter", "mode": "lines+markers",
                    "x": x_vals, "y": y_vals,
                    "line": {"color": palette[0], "width": 2},
                    "marker": {"size": 6, "color": palette[0]},
                    "hovertemplate": "<b>%{x}</b><br>" + y_col + ": %{y:,.2f}<extra></extra>",
                }

            layout_base["xaxis"]["title"] = x_col
            layout_base["yaxis"]["title"] = y_col
            return {"data": [trace], "layout": layout_base}

    # ── Pie ───────────────────────────────────────────────────────────────────
    if chart_type == "pie" and rows:
        label_col = cats[0] if cats else None
        val_col = numerics[0] if numerics else None
        if label_col and val_col:
            agg = {}
            for row in rows:
                lv = str(row.get(label_col, ""))
                vv = row.get(val_col)
                if lv and vv is not None:
                    try:
                        agg[lv] = agg.get(lv, 0.0) + float(vv)
                    except (ValueError, TypeError):
                        pass
            labels = list(agg.keys())
            values = [round(agg[k], 2) for k in labels]
            trace = {
                "type": "pie", "labels": labels, "values": values,
                "hole": 0.4,
                "marker": {"colors": palette[:len(labels)]},
                "textinfo": "label+percent",
                "hovertemplate": "<b>%{label}</b><br>%{value:,.2f} (%{percent})<extra></extra>",
            }
            return {"data": [trace], "layout": layout_base}

    # ── Scatter ───────────────────────────────────────────────────────────────
    if chart_type == "scatter" and len(numerics) >= 2 and rows:
        x_col, y_col = numerics[0], numerics[1]
        x_vals, y_vals = [], []
        for row in rows:
            try:
                x_vals.append(float(row[x_col]))
                y_vals.append(float(row[y_col]))
            except (KeyError, TypeError, ValueError):
                pass
        trace = {
            "type": "scatter", "mode": "markers",
            "x": x_vals, "y": y_vals,
            "marker": {"color": palette[0], "size": 8, "opacity": 0.8},
            "hovertemplate": x_col + ": %{x:,.2f}<br>" + y_col + ": %{y:,.2f}<extra></extra>",
        }
        layout_base["xaxis"]["title"] = x_col
        layout_base["yaxis"]["title"] = y_col
        return {"data": [trace], "layout": layout_base}

    # ── Histogram ─────────────────────────────────────────────────────────────
    if chart_type == "histogram" and numerics and rows:
        col = numerics[0]
        vals = []
        for row in rows:
            try:
                vals.append(float(row[col]))
            except (KeyError, TypeError, ValueError):
                pass
        trace = {
            "type": "histogram", "x": vals,
            "marker": {"color": palette[0]},
            "hovertemplate": "Value: %{x}<br>Count: %{y}<extra></extra>",
        }
        layout_base["xaxis"]["title"] = col
        layout_base["yaxis"]["title"] = "Count"
        return {"data": [trace], "layout": layout_base}

    # ── Box ───────────────────────────────────────────────────────────────────
    if chart_type == "box" and numerics and rows:
        traces = []
        group_col = cats[0] if cats else None
        for i, num_col in enumerate(numerics[:4]):
            if group_col:
                groups: dict = {}
                for row in rows:
                    gv = str(row.get(group_col, ""))
                    try:
                        groups.setdefault(gv, []).append(float(row[num_col]))
                    except (KeyError, TypeError, ValueError):
                        pass
                for j, (gname, gvals) in enumerate(groups.items()):
                    traces.append({
                        "type": "box", "y": gvals, "name": gname,
                        "marker": {"color": palette[j % len(palette)]},
                        "boxmean": True,
                    })
            else:
                vals = []
                for row in rows:
                    try:
                        vals.append(float(row[num_col]))
                    except (KeyError, TypeError, ValueError):
                        pass
                traces.append({
                    "type": "box", "y": vals, "name": num_col,
                    "marker": {"color": palette[i % len(palette)]},
                    "boxmean": True,
                })
        if traces:
            layout_base["yaxis"]["title"] = numerics[0]
            return {"data": traces, "layout": layout_base}

    # ── Heatmap ───────────────────────────────────────────────────────────────
    if chart_type == "heatmap" and len(numerics) >= 2 and rows:
        # Correlation matrix
        data_by_col: dict = {n: [] for n in numerics[:6]}
        for row in rows:
            for n in numerics[:6]:
                try:
                    data_by_col[n].append(float(row[n]))
                except (KeyError, TypeError, ValueError):
                    data_by_col[n].append(0.0)

        cols_used = list(data_by_col.keys())
        n = len(cols_used)
        z = []
        for ci in cols_used:
            row_corr = []
            xi = data_by_col[ci]
            mean_i = sum(xi) / len(xi) if xi else 0
            for cj in cols_used:
                xj = data_by_col[cj]
                mean_j = sum(xj) / len(xj) if xj else 0
                num = sum((a - mean_i) * (b - mean_j) for a, b in zip(xi, xj))
                di = (sum((a - mean_i) ** 2 for a in xi) ** 0.5)
                dj = (sum((b - mean_j) ** 2 for b in xj) ** 0.5)
                corr = round(num / (di * dj), 3) if di and dj else 0.0
                row_corr.append(corr)
            z.append(row_corr)

        trace = {
            "type": "heatmap", "z": z, "x": cols_used, "y": cols_used,
            "colorscale": "Blues", "zmin": -1, "zmax": 1,
            "text": [[str(v) for v in row] for row in z],
            "texttemplate": "%{text}",
            "hovertemplate": "%{y} × %{x}: %{z:.3f}<extra></extra>",
        }
        return {"data": [trace], "layout": layout_base}

    # ── Treemap ───────────────────────────────────────────────────────────────
    if chart_type == "treemap" and cats and numerics and rows:
        label_col = cats[0]
        val_col = numerics[0]
        agg = {}
        for row in rows:
            lv = str(row.get(label_col, ""))
            try:
                agg[lv] = agg.get(lv, 0.0) + float(row[val_col])
            except (KeyError, TypeError, ValueError):
                pass
        labels = list(agg.keys())
        values = [round(agg[k], 2) for k in labels]
        trace = {
            "type": "treemap",
            "labels": labels,
            "parents": [""] * len(labels),
            "values": values,
            "marker": {"colorscale": "Blues"},
            "textinfo": "label+value+percent entry",
            "hovertemplate": "<b>%{label}</b><br>%{value:,.2f}<extra></extra>",
        }
        return {"data": [trace], "layout": layout_base}

    # ── Waterfall ─────────────────────────────────────────────────────────────
    if chart_type == "waterfall" and rows:
        x_col = cats[0] if cats else (dts[0] if dts else None)
        y_col = numerics[0] if numerics else None
        if x_col and y_col:
            x_vals, y_vals = [], []
            for row in rows:
                try:
                    x_vals.append(str(row[x_col]))
                    y_vals.append(float(row[y_col]))
                except (KeyError, TypeError, ValueError):
                    pass
            trace = {
                "type": "waterfall",
                "x": x_vals, "y": y_vals,
                "connector": {"line": {"color": bg["grid"]}},
                "increasing": {"marker": {"color": palette[2]}},
                "decreasing": {"marker": {"color": palette[5] if len(palette) > 5 else "#EF4444"}},
                "hovertemplate": "<b>%{x}</b><br>%{y:,.2f}<extra></extra>",
            }
            layout_base["xaxis"]["title"] = x_col
            layout_base["yaxis"]["title"] = y_col
            return {"data": [trace], "layout": layout_base}

    # ── Ultimate fallback: bar chart with whatever data we have ───────────────
    if rows and (cats or dts) and numerics:
        x_col = (cats + dts)[0]
        y_col = numerics[0]
        agg = {}
        for row in rows:
            xv = str(row.get(x_col, ""))
            try:
                agg[xv] = agg.get(xv, 0.0) + float(row.get(y_col, 0))
            except (TypeError, ValueError):
                pass
        x_vals = list(agg.keys())
        y_vals = [round(agg[k], 2) for k in x_vals]
        trace = {
            "type": "bar", "x": x_vals, "y": y_vals,
            "marker": {"color": palette[0]},
            "hovertemplate": "<b>%{x}</b><br>%{y:,.2f}<extra></extra>",
        }
        layout_base["xaxis"]["title"] = x_col
        layout_base["yaxis"]["title"] = y_col
        return {"data": [trace], "layout": layout_base}

    # ── Absolute last resort: empty chart with message ────────────────────────
    return {
        "data": [],
        "layout": {
            **layout_base,
            "annotations": [{
                "text": "No renderable data found",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5, "showarrow": False,
                "font": {"size": 16, "color": bg["text"]},
            }],
        },
    }


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _strip_markdown_fences(raw: str) -> str:
    raw = raw.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)```\s*$", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        inner = parts[1] if len(parts) >= 3 else parts[-1]
        return inner.lstrip("json").strip()
    return raw

def _sanitize_spec(spec: dict) -> dict:
    """
    Validate and clean a Plotly spec for React consumption.
    Fixes common LLM errors:
      - Extra top-level keys (config, frames) → removed
      - data not a list → wrapped
      - Empty traces → removed
      - Missing trace type → defaults to 'scatter'
      - NaN/Infinity in numeric arrays → replaced with None
    """
    # Ensure only valid top-level keys
    clean = {
        "data": spec.get("data", []),
        "layout": spec.get("layout", {}),
    }

    # data must be a list
    if not isinstance(clean["data"], list):
        clean["data"] = [clean["data"]] if clean["data"] else []

    # layout must be a dict
    if not isinstance(clean["layout"], dict):
        clean["layout"] = {}

    # Validate each trace
    valid_traces = []
    for trace in clean["data"]:
        if not isinstance(trace, dict):
            continue
        # Ensure trace has a type
        if "type" not in trace:
            trace["type"] = "scatter"
        # Clean NaN/Infinity from numeric arrays
        for key in ("x", "y", "z", "values", "text"):
            if key in trace and isinstance(trace[key], list):
                trace[key] = [
                    None if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf'))
                    else v
                    for v in trace[key]
                ]
        valid_traces.append(trace)

    clean["data"] = valid_traces
    return clean


def _parse_spec(raw: str) -> dict:
    cleaned = _strip_markdown_fences(raw)
    spec = json.loads(cleaned)
    if "data" not in spec or "layout" not in spec:
        raise ValueError("Spec missing 'data' or 'layout'")
    return _sanitize_spec(spec)


def _parse_insight_array(raw: str) -> list[dict]:
    cleaned = _strip_markdown_fences(raw)
    arr = json.loads(cleaned)
    if not isinstance(arr, list):
        raise ValueError("Expected JSON array")
    return arr

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a world-class data visualization engineer at a top-tier analytics firm.
Your charts appear in executive dashboards, investor reports, and Bloomberg terminals.

ABSOLUTE RULES — violating any of these is a failure:
1. Output ONLY a raw JSON object. No markdown fences, no explanation, no comments.
2. The JSON must have exactly two top-level keys: "data" (array) and "layout" (object).
3. NEVER invent data. Use ONLY the values provided in the data sample.
4. NEVER use placeholder text like "Category 1" or "Value A". Use actual column values.
5. Every trace MUST have a hovertemplate. No exceptions.
6. All numbers in labels/hover must be formatted (commas, units, % as appropriate).
7. The chart must be immediately readable without any explanation."""


_USER_PROMPT = """Generate a premium Plotly JSON figure for this exact dataset.

═══════════════════════════════════════
CHART SPECIFICATION
═══════════════════════════════════════
Chart type   : {chart_type}
User task    : {task}
Color palette: {palette}

THEME SETTINGS (apply exactly):
  plot_bgcolor   = "{plot_bg}"
  paper_bgcolor  = "{paper_bg}"
  grid_color     = "{grid_color}"
  text_color     = "{text_color}"
  font_family    = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

═══════════════════════════════════════
COLUMN DEFINITIONS
═══════════════════════════════════════
{columns}

═══════════════════════════════════════
DATASET STATISTICS (computed from ALL rows)
═══════════════════════════════════════
{stats}

═══════════════════════════════════════
DATA ({n_rows} rows — use ALL of them for traces)
═══════════════════════════════════════
{data_sample}

═══════════════════════════════════════
UNIVERSAL LAYOUT REQUIREMENTS
═══════════════════════════════════════
title:
  - insight-driven, not generic (e.g. "North Region Leads Q1 Revenue by 34%", not "Revenue by Region")
  - add subtitle: title.text = "Main Title<br><sub>Supporting context or date range</sub>"
  - font.size=18, font.color="{text_color}", font.weight="bold", x=0.5, xanchor="center"

axes (apply to all xaxis/yaxis):
  - title.font.size=13, title.font.color="{text_color}"
  - tickfont.size=12, tickfont.color="{text_color}"
  - gridcolor="{grid_color}", gridwidth=1, showgrid=true
  - linecolor="{grid_color}", linewidth=1, showline=true
  - zeroline=false

margins: l=70, r=40, t=100, b=80
legend:
  - orientation="h", x=0.5, xanchor="center", y=-0.18
  - bgcolor="rgba(0,0,0,0)", bordercolor="{grid_color}", borderwidth=1
  - font.size=12, font.color="{text_color}"

hoverlabel:
  - bgcolor="{paper_bg}", bordercolor="{grid_color}"
  - font.size=13, font.color="{text_color}"

═══════════════════════════════════════
CHART-TYPE SPECIFIC REQUIREMENTS
═══════════════════════════════════════

BAR chart:
  - orientation="v" unless >8 categories (then "h" for readability)
  - marker.color = assign palette colors per category or use first palette color
  - marker.line.color = darker shade of bar color, marker.line.width=1.5
  - text = formatted values, textposition="outside", textfont.size=11
  - texttemplate="%{{value:,.0f}}"
  - hovertemplate="<b>%{{x}}</b><br>Value: %{{y:,.0f}}<extra></extra>"
  - If grouped (multiple series): barmode="group", gap between groups
  - If stacked: barmode="stack"

LINE chart:
  - mode="lines+markers" for ≤50 points, mode="lines" for >50 points
  - line.width=2.5, line.shape="spline" (smooth curves)
  - marker.size=7, marker.symbol="circle"
  - One trace per category/series, each with a distinct palette color
  - hovertemplate="<b>%{{x}}</b><br>%{{fullData.name}}: %{{y:,.0f}}<extra></extra>"
  - xaxis.type="date" if column is datetime, tickformat="%b %Y"

AREA chart:
  - filltype="tozeroy" for single series, stackgroup="one" for multi-series
  - line.width=2, opacity=0.7 for fill
  - Same hover/color rules as LINE

PIE / DONUT chart:
  - hole=0.45 for donut (preferred), hole=0 for pie
  - textinfo="label+percent", textposition="outside"
  - textfont.size=12
  - marker.line.color="{paper_bg}", marker.line.width=2
  - hovertemplate="<b>%{{label}}</b><br>Value: %{{value:,.0f}}<br>Share: %{{percent}}<extra></extra>"
  - Pull the largest slice by 0.05 for emphasis

SCATTER chart:
  - marker.size=10, marker.opacity=0.8
  - marker.line.color="white", marker.line.width=1
  - If color dimension exists: use colorscale, add colorbar
  - hovertemplate="<b>Point</b><br>X: %{{x:,.2f}}<br>Y: %{{y:,.2f}}<extra></extra>"
  - Add trendline as a separate scatter trace with mode="lines", dash="dot"

HISTOGRAM chart:
  - bargap=0.05, nbinsx=25
  - marker.color = first palette color, marker.opacity=0.85
  - marker.line.color="{paper_bg}", marker.line.width=0.5
  - hovertemplate="Range: %{{x}}<br>Count: %{{y}}<extra></extra>"
  - Add mean line as a shape annotation (vertical dashed line)

HEATMAP chart:
  - colorscale built from palette (light → dark)
  - showscale=true, colorbar.thickness=15, colorbar.len=0.8
  - annotations: add text annotation on every cell showing the value
  - annotation font.size=11, font.color auto (dark text on light cells, light on dark)
  - hovertemplate="X: %{{x}}<br>Y: %{{y}}<br>Value: %{{z:,.0f}}<extra></extra>"

BOX chart:
  - boxmean="sd" (show mean + std deviation markers)
  - jitter=0.3, pointpos=0, marker.size=4, marker.opacity=0.5
  - One box per category, each with a distinct palette color
  - hovertemplate="<b>%{{x}}</b><br>Median: %{{median:,.0f}}<extra></extra>"

TREEMAP chart:
  - textinfo="label+value+percent entry"
  - textfont.size=13
  - marker.line.color="{paper_bg}", marker.line.width=2
  - hovertemplate="<b>%{{label}}</b><br>Value: %{{value:,.0f}}<br>%{{percentEntry}} of total<extra></extra>"

WATERFALL chart:
  - measure: first row "absolute", rest "relative", last row "total"
  - increasing.marker.color = palette[1] (positive color)
  - decreasing.marker.color = palette[0] (negative color, make it red-ish)
  - totals.marker.color = palette[2]
  - connector.line.color="{grid_color}", connector.line.width=1, connector.line.dash="dot"
  - textposition="outside", texttemplate="%{{value:+,.0f}}"
  - hovertemplate="<b>%{{x}}</b><br>Change: %{{y:+,.0f}}<extra></extra>"

═══════════════════════════════════════
OUTPUT
═══════════════════════════════════════
Return ONLY the raw JSON. First character must be {{ and last must be }}."""


_CORRECTION_PROMPT = """Your previous response was not valid JSON or was missing required keys.

Return ONLY a JSON object. Requirements:
- First character: {
- Last character: }
- Must have "data" key (array of trace objects)
- Must have "layout" key (object)
- No markdown fences (no ```)
- No explanation text before or after
- No comments inside the JSON

If you were generating a {chart_type} chart for task "{task}", try again now.
Output the raw JSON immediately:"""

# ── Main API ──────────────────────────────────────────────────────────────────

def generate_spec(
    chart_type: str,
    task: str,
    columns: list[dict],
    data_sample: list[dict],
    color_scheme: str = "corporate",
) -> dict:
    """
    Generate a Plotly JSON spec via Groq LLM.
    ALWAYS returns a valid spec — falls back to deterministic builder if LLM fails.
    """
    palette = get_palette(color_scheme)
    bg = get_background(color_scheme)
    sample = data_sample[:15]  # slightly more context than before
    stats = compute_data_stats(columns, data_sample)  # full dataset stats

    user_prompt = _USER_PROMPT.format(
        chart_type=chart_type,
        task=task,
        columns=json.dumps(columns, indent=1),
        stats=json.dumps(stats, indent=1),
        palette=json.dumps(palette),
        plot_bg=bg["plot_bg"],
        paper_bg=bg["paper_bg"],
        grid_color=bg["grid"],
        text_color=bg["text"],
        n_rows=len(sample),
        data_sample=json.dumps(sample, indent=1, default=str),
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw = ""

    # ── Attempt 1 ─────────────────────────────────────────────────────────────
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            temperature=0.15,
            messages=messages,
        )
        raw = resp.choices[0].message.content.strip()
        spec = _parse_spec(raw)
        logger.info("generate_spec OK attempt-1 (chart=%s, scheme=%s)", chart_type, color_scheme)
        return spec
    except HTTPException:
        raise  # API key missing — propagate
    except Exception as e1:
        logger.warning("generate_spec attempt-1 failed: %s", e1)

    # ── Attempt 2 (correction prompt) ─────────────────────────────────────────
    try:
        client = _get_client()
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _CORRECTION_PROMPT.format(chart_type=chart_type, task=task)},
        ]
        resp2 = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            temperature=0.0,
            messages=retry_messages,
        )
        raw2 = resp2.choices[0].message.content.strip()
        spec = _parse_spec(raw2)
        logger.info("generate_spec OK attempt-2 (chart=%s)", chart_type)
        return spec
    except HTTPException:
        raise
    except Exception as e2:
        logger.warning("generate_spec attempt-2 failed: %s — using fallback", e2)

    # ── Fallback: deterministic spec from real data ────────────────────────────
    logger.info("generate_spec using deterministic fallback (chart=%s)", chart_type)
    return _sanitize_spec(_build_fallback_spec(chart_type, task, columns, data_sample, color_scheme))


def auto_select_insights(
    columns: list[dict],
    data_sample: list[dict],
    n_insights: int = 3,
    exclude_types: set[str] | None = None,
) -> list[dict]:
    """
    Ask LLM to pick the most impactful visualizations.
    Accepts exclude_types to avoid duplicating chart types already chosen
    by the rule engine.
    Returns [] on failure — caller handles gracefully.
    """
    sample = data_sample[:15]
    stats = compute_data_stats(columns, data_sample)

    exclude_clause = ""
    if exclude_types:
        exclude_clause = (
            f"\nDO NOT use these chart types (already selected): "
            f"{', '.join(sorted(exclude_types))}"
        )

    system = f"""You are a senior data scientist. Given column profiles and statistics, identify the most impactful visualizations that reveal non-obvious insights.
Return ONLY a JSON array. Each item: {{"chart_type": "bar|line|scatter|pie|histogram|heatmap|box|treemap|waterfall|area", "task": "specific insight description"}}
No fences, no explanation.{exclude_clause}"""

    user = f"""Columns: {json.dumps(columns, indent=1)}
Statistics: {json.dumps(stats, indent=1)}
Sample ({len(sample)} rows): {json.dumps(sample, indent=1, default=str)}

Return {n_insights} most impactful visualizations. Each must use a DIFFERENT chart type. Focus on insights that aren't obvious from raw numbers."""

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            temperature=0.25,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        insights = _parse_insight_array(raw)
        logger.info("auto_select_insights -> %d insights", len(insights))
        return insights[:n_insights]
    except Exception as exc:
        logger.warning("auto_select_insights failed: %s", exc)
        return []
