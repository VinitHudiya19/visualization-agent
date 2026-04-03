"""
Rule-based chart recommendation engine.
No LLM calls — pure if/else logic on column semantics and task keywords.

Priority order (highest → lowest):
  1. Task keyword groups (explicit user intent)
  2. Comparison keywords BEFORE datetime rule  ← critical fix
  3. Column-type fallbacks
  4. Safe default

Supports: bar, line, scatter, pie, histogram, heatmap,
          box, treemap, waterfall, area, sunburst
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("viz-agent")


# ── Column Profile Helpers ────────────────────────────────────────────────────

def _count_by_semantic(columns: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"numeric": 0, "categorical": 0, "datetime": 0, "other": 0}
    for c in columns:
        sem = c.get("semantic", "other")
        counts[sem] = counts.get(sem, 0) + 1
    return counts


def _get_names_by_type(columns: list[dict], sem_type: str) -> list[str]:
    return [c["name"] for c in columns if c.get("semantic") == sem_type]


def _max_cardinality(columns: list[dict], sem_type: str = "categorical") -> int:
    return max(
        (c.get("unique", 0) for c in columns if c.get("semantic") == sem_type),
        default=0,
    )


# ── Task Keyword Groups ───────────────────────────────────────────────────────
# Each group is a frozenset so membership checks are O(1).
# Keywords are lowercase substrings — partial match is intentional
# (e.g. "correlat" matches "correlation", "correlated", "correlates").

_BOXPLOT_KW     = frozenset({"box", "quartile", "median", "interquartile",
                              "range of", "variability of", "iqr", "outlier"})

_DISTRIBUTION_KW = frozenset({"distribution", "spread", "histogram", "frequency",
                               "density", "how spread", "variability", "skew"})

_HIERARCHY_KW   = frozenset({"hierarchy", "hierarchical", "tree", "nested",
                              "drill", "breakdown by", "segment by", "treemap",
                              "drilldown", "sub-category", "subcategory"})

_FLOW_KW        = frozenset({"waterfall", "build-up", "bridge", "contribution to",
                              "change from", "net change", "variance bridge",
                              "adds to", "subtracts from"})

_PROPORTION_KW  = frozenset({"proportion", "share", "percent", "percentage",
                              "composition", "make up", "ratio", "mix",
                              "part of", "slice", "donut", "pie"})

_CORRELATION_KW = frozenset({"correlat", "relationship between", "versus",
                              " vs ", "association", "impact of", "dependency",
                              "scatter", "regression", "predict"})

_COMPARISON_KW  = frozenset({"compare", "comparison", "rank", "ranking",
                              "top", "bottom", "best", "worst", "benchmark",
                              "performance of", "across", "by region",
                              "by category", "per region", "per category",
                              "total by", "revenue by", "sales by", "count by",
                              "average by", "breakdown", "contrast"})

_TREND_KW       = frozenset({"trend", "over time", "growth", "decline",
                              "forecast", "time series", "temporal", "timeline",
                              "progress", "evolution", "monthly", "weekly",
                              "daily", "yearly", "over the", "by month",
                              "by week", "by year", "by quarter"})

_CUMULATIVE_KW  = frozenset({"cumulative", "running total", "accumulated",
                              "stacked area", "fill under", "area chart"})

_HEATMAP_KW     = frozenset({"heatmap", "heat map", "matrix", "correlation matrix",
                              "cross tab", "crosstab", "pivot"})


def _task_matches(task: str, keywords: frozenset[str]) -> bool:
    """Case-insensitive substring match against any keyword in the set."""
    task_l = task.lower()
    return any(kw in task_l for kw in keywords)


# ── Main Recommender ──────────────────────────────────────────────────────────

def recommend_chart(columns: list[dict], task: str) -> Optional[str]:
    """
    Pure rule-based chart type selection. No LLM, no network calls.

    Returns a chart type string, or None when rules are ambiguous
    (caller should fall back to LLM).

    Priority order is intentional — do NOT reorder without testing:
      1. Box / outlier analysis
      2. Distribution (histogram)
      3. Hierarchy (treemap)
      4. Flow / waterfall
      5. Proportion (pie)
      6. Explicit heatmap keyword
      7. Correlation / scatter
      8. Cumulative area
      9. COMPARISON → bar  ← must come before datetime rule
     10. Explicit trend keywords → line
     11. Datetime + numeric fallback → line  ← only if no category present
     12. Categorical + numeric fallback → bar
     13. Two+ numerics → scatter
     14. None (ambiguous)
    """
    counts        = _count_by_semantic(columns)
    has_datetime  = counts["datetime"] > 0
    has_numeric   = counts["numeric"] > 0
    numeric_count = counts["numeric"]
    cat_names     = _get_names_by_type(columns, "categorical")
    cat_card      = _max_cardinality(columns, "categorical")

    # ── 1. Box / outlier ──────────────────────────────────────────────────────
    if _task_matches(task, _BOXPLOT_KW) and has_numeric:
        logger.debug("recommend_chart: box (boxplot keywords matched)")
        return "box"

    # ── 2. Distribution / histogram ───────────────────────────────────────────
    if _task_matches(task, _DISTRIBUTION_KW):
        logger.debug("recommend_chart: histogram (distribution keywords matched)")
        return "histogram"

    # ── 3. Hierarchy / treemap ────────────────────────────────────────────────
    if _task_matches(task, _HIERARCHY_KW) and len(cat_names) >= 2 and has_numeric:
        logger.debug("recommend_chart: treemap (hierarchy keywords matched)")
        return "treemap"

    # ── 4. Waterfall / flow ───────────────────────────────────────────────────
    if _task_matches(task, _FLOW_KW) and has_numeric:
        logger.debug("recommend_chart: waterfall (flow keywords matched)")
        return "waterfall"

    # ── 5. Proportion / pie ───────────────────────────────────────────────────
    if _task_matches(task, _PROPORTION_KW):
        if cat_card <= 8:
            logger.debug("recommend_chart: pie (proportion keywords, <=8 categories)")
            return "pie"
        # Too many slices — bar is more readable
        logger.debug("recommend_chart: bar (proportion keywords but >8 categories)")
        return "bar"

    # ── 6. Explicit heatmap ───────────────────────────────────────────────────
    if _task_matches(task, _HEATMAP_KW) and numeric_count >= 2:
        logger.debug("recommend_chart: heatmap (heatmap keywords matched)")
        return "heatmap"

    # ── 7. Correlation / scatter ──────────────────────────────────────────────
    if _task_matches(task, _CORRELATION_KW) and numeric_count >= 2:
        chart = "heatmap" if numeric_count > 5 else "scatter"
        logger.debug("recommend_chart: %s (correlation keywords matched)", chart)
        return chart

    # ── 8. Cumulative area ────────────────────────────────────────────────────
    if _task_matches(task, _CUMULATIVE_KW) and has_datetime and has_numeric:
        logger.debug("recommend_chart: area (cumulative keywords matched)")
        return "area"

    # ── 9. COMPARISON / RANKING → BAR ────────────────────────────────────────
    # CRITICAL: This MUST come before the datetime → line rule.
    # "Compare revenue by region Jan-Mar" has BOTH datetime and categorical.
    # User intent is comparison, not time series. Bar wins.
    if _task_matches(task, _COMPARISON_KW) and cat_names and has_numeric:
        logger.debug("recommend_chart: bar (comparison keywords matched, beats datetime rule)")
        return "bar"

    # ── 10. Explicit trend keywords + datetime → line ─────────────────────────
    if _task_matches(task, _TREND_KW) and has_datetime and has_numeric:
        logger.debug("recommend_chart: line (trend keywords + datetime)")
        return "line"

    # ── 11. Datetime + numeric (no categorical) → line ────────────────────────
    # Only fires when there is NO categorical column present.
    # If categorical IS present, the comparison rule (step 9) already handled it.
    if has_datetime and has_numeric and not cat_names:
        logger.debug("recommend_chart: line (datetime + numeric, no categories)")
        return "line"

    # ── 12. Categorical + numeric fallback → bar ──────────────────────────────
    if cat_names and has_numeric:
        if cat_card <= 15:
            logger.debug("recommend_chart: bar (categorical + numeric fallback)")
            return "bar"
        # High cardinality: LLM should decide (table? treemap? truncated bar?)
        logger.debug("recommend_chart: None (high cardinality categorical — LLM fallback)")
        return None

    # ── 13. Two+ numerics, no categorical → scatter ───────────────────────────
    if numeric_count >= 2:
        logger.debug("recommend_chart: scatter (numeric-only fallback)")
        return "scatter"

    # ── 14. Ambiguous — let LLM decide ───────────────────────────────────────
    logger.debug("recommend_chart: None (ambiguous — LLM fallback)")
    return None


# ── Auto-Insight Suggestion ───────────────────────────────────────────────────

def suggest_best_insights(columns: list[dict], max_insights: int = 5) -> list[dict]:
    """
    Auto-suggest the best visualization tasks purely from column profiles.
    Called by POST /auto-insights when no task is provided by the user.

    Returns a list of {chart_type, task} dicts, deduplicated by chart_type,
    capped at max_insights.

    Insight generation order is priority-ranked:
      trend → comparison → proportion → distribution → correlation
      → hierarchy → heatmap → area
    """
    numerics     = _get_names_by_type(columns, "numeric")
    categoricals = _get_names_by_type(columns, "categorical")
    datetimes    = _get_names_by_type(columns, "datetime")
    cat_card     = _max_cardinality(columns, "categorical")

    candidates: list[dict] = []

    # 1. Time-series trend line (highest value if datetime exists)
    if datetimes and numerics:
        dt  = datetimes[0]
        num = numerics[0]
        cat_clause = f" broken down by {categoricals[0]}" if categoricals else ""
        candidates.append({
            "chart_type": "line",
            "task": f"Reveal the trend of {num} over {dt}{cat_clause}",
        })

    # 2. Comparison bar chart
    if categoricals and numerics:
        candidates.append({
            "chart_type": "bar",
            "task": f"Compare total {numerics[0]} across {categoricals[0]} segments",
        })

    # 3. Proportion pie (only if cardinality is sensible)
    if categoricals and numerics and 2 <= cat_card <= 8:
        candidates.append({
            "chart_type": "pie",
            "task": f"Show the percentage share of {numerics[0]} by {categoricals[0]}",
        })

    # 4a. Box plot (if categorical exists — more informative)
    if numerics and categoricals:
        candidates.append({
            "chart_type": "box",
            "task": f"Analyze the distribution and outliers of {numerics[0]} across {categoricals[0]}",
        })
    # 4b. Histogram (if no categorical)
    elif numerics:
        candidates.append({
            "chart_type": "histogram",
            "task": f"Examine the distribution pattern of {numerics[0]}",
        })

    # 5. Scatter / correlation
    if len(numerics) >= 2:
        candidates.append({
            "chart_type": "scatter",
            "task": f"Identify the relationship between {numerics[0]} and {numerics[1]}",
        })

    # 6. Treemap (needs 2 categoricals)
    if len(categoricals) >= 2 and numerics:
        candidates.append({
            "chart_type": "treemap",
            "task": (
                f"Visualize {numerics[0]} hierarchy"
                f" across {categoricals[0]} and {categoricals[1]}"
            ),
        })

    # 7. Heatmap correlation matrix (needs 4+ numerics)
    if len(numerics) >= 4:
        candidates.append({
            "chart_type": "heatmap",
            "task": f"Reveal correlations between {', '.join(numerics[:5])}",
        })

    # 8. Stacked area (needs datetime + 2 numerics)
    if datetimes and len(numerics) >= 2:
        candidates.append({
            "chart_type": "area",
            "task": (
                f"Show cumulative contribution of"
                f" {numerics[0]} and {numerics[1]} over {datetimes[0]}"
            ),
        })

    # Deduplicate by chart_type, preserve insertion order
    seen:   set[str]    = set()
    unique: list[dict]  = []
    for item in candidates:
        if item["chart_type"] not in seen:
            seen.add(item["chart_type"])
            unique.append(item)
        if len(unique) >= max_insights:
            break

    logger.info(
        "suggest_best_insights -> %d insights from %d columns",
        len(unique), len(columns),
    )
    return unique