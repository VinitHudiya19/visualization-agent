"""
Smart data aggregation for chart generation.

Chart-type-aware: different charts need different data preparation.
  - bar/pie/treemap/waterfall: group by categorical, sum numerics
  - line/area: sort by datetime, aggregate duplicate dates
  - scatter/histogram/box/heatmap: return raw data
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("viz-agent")

# Keywords that signal categorical aggregation
_CAT_AGG_KEYWORDS = frozenset({
    "compare", "total", "across", "by region", "by category",
    "rank", "top", "breakdown", "per region", "per category",
    "share", "proportion", "percentage", "composition",
    "sum of", "average of", "count by", "revenue by", "sales by",
    "profit by", "cost by", "orders by",
})

# Chart types that always need categorical aggregation
_CAT_AGG_CHARTS = frozenset({"bar", "pie", "treemap", "waterfall"})

# Chart types that need date-based aggregation
_DATE_AGG_CHARTS = frozenset({"line", "area"})

# Chart types that work best with raw data
_RAW_CHARTS = frozenset({"scatter", "histogram", "box", "heatmap"})


def auto_aggregate(
    columns: list[dict],
    rows: list[dict],
    task: str,
    chart_type: Optional[str] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Smart data aggregation based on task keywords + chart type.

    Rules:
      1. Empty data: passthrough
      2. Raw-data chart types (scatter/histogram/box/heatmap): passthrough
      3. Cat-agg charts (bar/pie/treemap/waterfall): ALWAYS group by
         first categorical and sum numerics, even for small datasets
      4. Date-agg charts (line/area): group by first datetime, sum numerics
      5. Keyword match without explicit chart_type: aggregate if >20 rows
      6. Default: passthrough
    """
    if not rows:
        return columns, rows

    # Raw-data chart types: never aggregate
    if chart_type in _RAW_CHARTS:
        return columns, rows

    cat_cols = [c["name"] for c in columns if c.get("semantic") == "categorical"]
    num_cols = [c["name"] for c in columns if c.get("semantic") == "numeric"]
    date_cols = [c["name"] for c in columns if c.get("semantic") == "datetime"]
    task_l = task.lower()

    # Determine aggregation needs
    force_cat_agg = chart_type in _CAT_AGG_CHARTS
    keyword_cat_agg = any(kw in task_l for kw in _CAT_AGG_KEYWORDS)
    force_date_agg = chart_type in _DATE_AGG_CHARTS

    # Small datasets: only aggregate if chart type explicitly requires it
    if len(rows) <= 20 and not force_cat_agg and not force_date_agg:
        return columns, rows

    df = pd.DataFrame(rows)

    # ── Categorical aggregation (bar, pie, treemap, waterfall) ────────────
    if (force_cat_agg or keyword_cat_agg) and cat_cols and num_cols:
        group_col = cat_cols[0]
        if group_col in df.columns:
            agg_dict = {c: "sum" for c in num_cols if c in df.columns}
            if agg_dict:
                df_agg = df.groupby(group_col, as_index=False).agg(agg_dict)
                # Sort by primary numeric descending for better visuals
                if num_cols[0] in df_agg.columns:
                    df_agg = df_agg.sort_values(num_cols[0], ascending=False)

                new_cols = [
                    {"name": group_col, "semantic": "categorical",
                     "unique": int(df_agg[group_col].nunique())}
                ]
                for c in num_cols:
                    if c in df_agg.columns:
                        new_cols.append({"name": c, "semantic": "numeric"})

                logger.info(
                    "auto_aggregate: %d->%d rows (group by %s, chart=%s)",
                    len(rows), len(df_agg), group_col, chart_type or "auto",
                )
                return new_cols, df_agg.to_dict(orient="records")

    # ── Date aggregation (line, area) ─────────────────────────────────────
    if force_date_agg and date_cols and num_cols:
        date_col = date_cols[0]
        if date_col in df.columns:
            agg_dict = {c: "sum" for c in num_cols if c in df.columns}
            if agg_dict:
                df_agg = (
                    df.groupby(date_col, as_index=False)
                    .agg(agg_dict)
                    .sort_values(date_col)
                )
                new_cols = [{"name": date_col, "semantic": "datetime"}]
                for c in num_cols:
                    if c in df_agg.columns:
                        new_cols.append({"name": c, "semantic": "numeric"})

                logger.info(
                    "auto_aggregate: %d->%d rows (group by %s date, chart=%s)",
                    len(rows), len(df_agg), date_col, chart_type or "auto",
                )
                return new_cols, df_agg.to_dict(orient="records")

    return columns, rows