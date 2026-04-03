"""
Visualization Agent — FastAPI Microservice v2.3 (Production)

Endpoints:
  GET  /health          → service status
  POST /recommend       → rule-based chart recommendation (zero LLM)
  POST /chart           → data + task → Plotly spec + optional PNG
  POST /chart/render    → existing spec → PNG
  POST /auto-insights   → auto-pick 2 best charts (1 rule + 1 LLM)
  POST /run             → orchestrator unified entry point
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.llm.chart_selector import auto_select_insights, generate_spec
from app.utils.aggregator import auto_aggregate
from app.utils.chart_rules import recommend_chart, suggest_best_insights
from app.utils.color_palettes import SCHEME_NAMES
from app.utils.renderer import render_png, shutdown_renderer

logger = logging.getLogger("viz-agent")

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs(settings.CHART_OUTPUT_PATH, exist_ok=True)

# ── Types ─────────────────────────────────────────────────────────────────────
ColorScheme = Literal[
    "corporate", "executive", "vibrant", "neon", "pastel",
    "ocean", "dark", "midnight", "monochrome", "slate",
]

SUPPORTED_CHART_TYPES = [
    "bar", "line", "scatter", "pie", "histogram",
    "heatmap", "box", "treemap", "waterfall", "area",
]


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Viz-agent starting up — output dir: %s", settings.CHART_OUTPUT_PATH)
    yield
    # Shutdown: close kaleido thread pool gracefully
    shutdown_renderer()
    logger.info("Viz-agent shut down cleanly")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Visualization Agent",
    version="2.2.0",
    description="Chart generation — 10 color schemes, 10 chart types, Groq LLM.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ColumnProfile(BaseModel):
    name: str
    semantic: Literal["numeric", "categorical", "datetime"]
    unique: Optional[int] = None


class DataPayload(BaseModel):
    columns: list[ColumnProfile]
    rows: list[dict] = Field(..., description="Max 1000 rows. Pre-aggregate before sending.")

    @field_validator("rows")
    @classmethod
    def rows_not_empty(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("rows cannot be empty")
        return v


class ChartRequest(BaseModel):
    task: str = Field(..., min_length=3)
    data: DataPayload
    chart_type: Optional[str] = None
    color_scheme: ColorScheme = "corporate"
    # FIX: do NOT use alias="render_png" — it clashes with the field name
    # and causes Pydantic to silently drop it during model_dump(by_alias=True)
    render_png: bool = True
    width: int = Field(1000, ge=200, le=3000)
    height: int = Field(600, ge=200, le=2000)


class RecommendRequest(BaseModel):
    columns: list[ColumnProfile]
    task: str = Field(..., min_length=3)


class RenderRequest(BaseModel):
    # FIX: accept a full spec dict, not split data+layout
    # Orchestrator and frontend both send {"spec": {...}} or the spec directly
    spec: dict
    width: int = Field(1000, ge=200, le=3000)
    height: int = Field(600, ge=200, le=2000)


class AutoInsightRequest(BaseModel):
    data: DataPayload
    color_scheme: ColorScheme = "corporate"
    render_png: bool = True
    max_insights: int = Field(2, ge=1, le=4)
    width: int = Field(1000, ge=200, le=2000)
    height: int = Field(600, ge=200, le=2000)


class RunRequest(BaseModel):
    """
    Unified entry point payload sent by the orchestrator executor.
    Columns + rows can come directly OR via _context from upstream agents.
    """
    task: str = Field("", description="Natural language task description")
    columns: Optional[list[ColumnProfile]] = None
    rows: Optional[list[dict]] = None
    chart_type: Optional[str] = None
    color_scheme: ColorScheme = "corporate"
    render_png: bool = False   # off by default in orchestrator pipeline
    max_insights: int = Field(2, ge=1, le=4)

    model_config = {"extra": "allow"}  # allow _context and other orchestrator fields



# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_rows(rows: list[dict], label: str = "data") -> None:
    if len(rows) == 0:
        raise HTTPException(status_code=400, detail=f"{label}: rows cannot be empty")
    if len(rows) > settings.MAX_DATA_ROWS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label}: {len(rows)} rows exceeds max {settings.MAX_DATA_ROWS}. "
                "Pre-aggregate your data before sending."
            ),
        )


def _to_col_dicts(columns: list[ColumnProfile]) -> list[dict]:
    """Convert Pydantic ColumnProfile list → plain dicts for internal functions."""
    return [c.model_dump() for c in columns]


def _resolve_chart_type(
    cols: list[dict],
    task: str,
    explicit: Optional[str],
) -> str:
    if explicit:
        if explicit not in SUPPORTED_CHART_TYPES:
            logger.warning("Unknown chart_type=%s requested — using bar", explicit)
            return "bar"
        return explicit
    return recommend_chart(cols, task) or "bar"


def _dtype_to_semantic(dtype: str) -> str:
    """Map context-agent dtype strings → viz-agent semantic types."""
    mapping = {
        "number": "numeric", "int": "numeric", "float": "numeric",
        "integer": "numeric", "double": "numeric",
        "text": "categorical", "string": "categorical",
        "varchar": "categorical", "boolean": "categorical",
        "date": "datetime", "datetime": "datetime",
        "timestamp": "datetime", "time": "datetime",
    }
    return mapping.get(dtype.lower(), "categorical")


def _extract_from_context(context: dict) -> tuple[list | None, list | None]:
    """
    Pull columns and rows from orchestrator _context dict.
    Context contains results from upstream agents keyed by task_id.
    Priority: context-agent profiles > SQL-agent rows.
    Returns (columns_as_dicts_or_None, rows_or_None).
    """
    columns: list | None = None
    rows: list | None = None

    for dep_result in context.values():
        if not isinstance(dep_result, dict):
            continue

        # Context agent: provides column_profiles + raw_sample_preview
        if "column_profiles" in dep_result and columns is None:
            try:
                columns = [
                    {
                        "name": p["name"],
                        "semantic": _dtype_to_semantic(p.get("dtype", "text")),
                        "unique": p.get("unique_count"),
                    }
                    for p in dep_result["column_profiles"]
                    if isinstance(p, dict) and "name" in p
                ]
            except Exception as e:
                logger.warning("Failed to parse column_profiles from context: %s", e)

        if "raw_sample_preview" in dep_result and rows is None:
            preview = dep_result["raw_sample_preview"]
            if isinstance(preview, list) and preview:
                rows = preview

        # SQL agent: provides columns list + rows list
        if "columns" in dep_result and columns is None:
            raw_cols = dep_result["columns"]
            if isinstance(raw_cols, list) and raw_cols:
                if isinstance(raw_cols[0], dict) and "semantic" in raw_cols[0]:
                    columns = raw_cols
                elif isinstance(raw_cols[0], str):
                    # SQL agent may return plain column name strings
                    columns = [{"name": c, "semantic": "categorical"} for c in raw_cols]

        if "rows" in dep_result and rows is None:
            r = dep_result["rows"]
            if isinstance(r, list) and r:
                rows = r

    return columns, rows


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health() -> dict:
    return {
        "status": "ok",
        "agent": "viz-agent",
        "version": "2.2.0",
        "model": settings.GROQ_MODEL,
        "port": settings.PORT,
        "color_schemes": SCHEME_NAMES,
        "chart_types": SUPPORTED_CHART_TYPES,
    }


@app.post("/recommend", tags=["Recommendation"])
def recommend(req: RecommendRequest) -> dict:
    """Instant rule-based chart recommendation — zero LLM calls."""
    cols = _to_col_dicts(req.columns)
    chart_type = recommend_chart(cols, req.task) or "bar"
    return {
        "recommended_chart": chart_type,
        "used_llm": False,
        "reason": "Rule-based selection from column semantics and task keywords",
    }


@app.post("/chart", tags=["Chart Generation"])
async def generate_chart(req: ChartRequest) -> dict:
    """
    Full pipeline: data + task → Plotly spec + optional PNG.
    Auto-aggregates data when task implies comparison across categories.
    generate_spec never raises — fallback spec guaranteed.
    """
    _validate_rows(req.data.rows)
    cols = _to_col_dicts(req.data.columns)

    # Resolve chart type from ORIGINAL columns, then aggregate with type-awareness
    chart_type = _resolve_chart_type(cols, req.task, req.chart_type)
    agg_cols, agg_rows = auto_aggregate(cols, req.data.rows, req.task, chart_type=chart_type)

    logger.info(
        "POST /chart chart_type=%s scheme=%s rows_in=%d rows_agg=%d task=%.80s",
        chart_type, req.color_scheme,
        len(req.data.rows), len(agg_rows),
        req.task,
    )

    spec = generate_spec(
        chart_type=chart_type,
        task=req.task,
        columns=agg_cols,
        data_sample=agg_rows,       # always list[dict], never Pydantic model
        color_scheme=req.color_scheme,
    )

    result: dict = {
        "chart_type": chart_type,
        "color_scheme": req.color_scheme,
        "spec": spec,
        "rows_aggregated": len(agg_rows),
        "rows_original": len(req.data.rows),
    }

    if req.render_png:
        png_b64, file_path = await render_png(spec, req.width, req.height)
        result["png_base64"] = png_b64
        result["file_path"] = file_path

    return result


@app.post("/chart/render", tags=["Chart Generation"])
async def render_chart(req: RenderRequest) -> dict:
    """
    Render an existing Plotly spec dict to PNG.
    Accepts: {"spec": {"data": [...], "layout": {...}}, "width": 1000, "height": 600}
    """
    if "data" not in req.spec or "layout" not in req.spec:
        raise HTTPException(400, "spec must contain 'data' and 'layout' keys")

    png_b64, file_path = await render_png(req.spec, req.width, req.height)
    if not png_b64:
        raise HTTPException(500, "PNG render failed — run: python -m kaleido_get_chrome")
    return {"png_base64": png_b64, "file_path": file_path}


@app.post("/auto-insights", tags=["Auto Insights"])
async def auto_insights(req: AutoInsightRequest) -> dict:
    """
    Generate exactly N (default 2) best charts for a dataset.
    Strategy: 1 rule-based (instant) + 1 LLM-selected (complementary).
    Sequential generation avoids Groq rate-limits and timeouts.
    Chart-type-aware aggregation ensures accurate, clean visuals.
    """
    _validate_rows(req.data.rows)
    cols = _to_col_dicts(req.data.columns)

    logger.info(
        "POST /auto-insights cols=%d rows=%d scheme=%s max=%d",
        len(cols), len(req.data.rows), req.color_scheme, req.max_insights,
    )

    # ── Step 1: Best rule-based insight (instant, no LLM call) ────────────
    insights: list[dict] = suggest_best_insights(cols, max_insights=1)

    # ── Step 2: LLM picks a complementary insight (different chart type) ──
    if len(insights) < req.max_insights:
        existing_types = {i["chart_type"] for i in insights}
        try:
            llm_suggestions = auto_select_insights(
                cols, req.data.rows,
                n_insights=3,
                exclude_types=existing_types,
            )
            for li in llm_suggestions:
                ct = li.get("chart_type")
                if ct and ct not in existing_types:
                    insights.append({"chart_type": ct, "task": li.get("task", ct)})
                    existing_types.add(ct)
                if len(insights) >= req.max_insights:
                    break
        except Exception as e:
            logger.warning("LLM insight selection failed (rule results still used): %s", e)

    # If LLM failed and we still need more, fill from rules
    if len(insights) < req.max_insights:
        extra_rules = suggest_best_insights(cols, max_insights=req.max_insights)
        existing_types = {i["chart_type"] for i in insights}
        for er in extra_rules:
            if er["chart_type"] not in existing_types:
                insights.append(er)
                existing_types.add(er["chart_type"])
            if len(insights) >= req.max_insights:
                break

    insights = insights[: req.max_insights]
    logger.info("Generating %d insight charts sequentially", len(insights))

    # ── Step 3: Generate charts sequentially (avoids Groq rate limits) ────
    charts: list[dict] = []
    for idx, insight in enumerate(insights):
        ct   = insight["chart_type"]
        task = insight["task"]
        try:
            agg_cols, agg_rows = auto_aggregate(cols, req.data.rows, task, chart_type=ct)
            spec = generate_spec(
                chart_type=ct,
                task=task,
                columns=agg_cols,
                data_sample=agg_rows,
                color_scheme=req.color_scheme,
            )
            out: dict = {
                "index": idx,
                "chart_type": ct,
                "task": task,
                "spec": spec,
                "status": "success",
            }
            if req.render_png:
                png_b64, file_path = await render_png(spec, req.width, req.height)
                out["png_base64"] = png_b64
                out["file_path"] = file_path
            charts.append(out)
        except Exception as exc:
            logger.warning("Insight chart[%d] type=%s failed: %s", idx, ct, exc)
            charts.append({
                "index": idx,
                "chart_type": ct,
                "task": task,
                "spec": None,
                "status": "failed",
                "error": str(exc),
            })

    successful = [c for c in charts if c["status"] == "success"]

    return {
        "total_requested": len(insights),
        "total_generated": len(successful),
        "charts": charts,
    }




# ── Orchestrator unified entry point ──────────────────────────────────────────

@app.post("/run", tags=["Orchestrator"])
async def run(req: RunRequest) -> dict:
    """
    Unified entry point called by the orchestrator executor.
    Routes to /chart or /auto-insights based on task keywords.
    Columns + rows can arrive directly OR via _context from upstream agents.
    """
    # FIX: read extra fields via model_extra, not model_dump
    # model_dump(by_alias=True) loses extra fields in Pydantic v2
    context: dict = req.model_extra.get("_context", {}) or {}

    columns = req.columns
    rows    = req.rows

    # Extract from upstream agent context if not directly provided
    if not columns or not rows:
        ctx_cols, ctx_rows = _extract_from_context(context)
        if not columns and ctx_cols:
            try:
                columns = [ColumnProfile(**c) if isinstance(c, dict) else c for c in ctx_cols]
            except Exception as e:
                logger.warning("Could not parse context columns: %s", e)
        if not rows and ctx_rows:
            rows = ctx_rows

    if not columns or not rows:
        raise HTTPException(
            status_code=400,
            detail=(
                "No data available. Provide columns + rows directly, "
                "or ensure an upstream context/sql agent ran successfully."
            ),
        )

    task_lower = req.task.lower()
    is_auto = any(
        kw in task_lower
        for kw in ("auto", "insight", "best chart", "recommend", "overview", "explore", "suggest")
    )

    if is_auto and not req.chart_type:
        auto_req = AutoInsightRequest(
            data=DataPayload(columns=columns, rows=rows),
            color_scheme=req.color_scheme,
            render_png=req.render_png,
            max_insights=req.max_insights,
        )
        return await auto_insights(auto_req)
    else:
        chart_req = ChartRequest(
            task=req.task or "Visualize this data",
            data=DataPayload(columns=columns, rows=rows),
            chart_type=req.chart_type,
            color_scheme=req.color_scheme,
            render_png=req.render_png,
        )
        return await generate_chart(chart_req)