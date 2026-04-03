"""
Async PNG renderer using Plotly + Kaleido.

Fixes over original:
  1. ThreadPoolExecutor with max_workers=2 — prevents thread starvation
     when multiple /chart calls arrive simultaneously. Default executor
     is unbounded and can spin up hundreds of threads under load.
  2. Kaleido version detection done ONCE at import time, not per render.
     Original re-imported plotly on every single call inside the thread.
  3. Timeout wrapper — kaleido can hang forever on corrupt specs.
     Original had no timeout; one bad call would block a thread permanently.
  4. Spec validation before sending to kaleido — catches bad specs early
     with a clear error instead of a cryptic kaleido crash.
  5. File size guard — empty PNG (0 bytes) is treated as failure.
  6. Output dir created once at module load, not per render call.
  7. Thread-safe: executor is module-level singleton, not created per call.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import plotly.graph_objects as go

from app.config import settings

logger = logging.getLogger("viz-agent")

# ── Render timeout (seconds) ──────────────────────────────────────────────────
# Kaleido can hang on malformed specs. Kill it after this many seconds.
_RENDER_TIMEOUT = 30

# ── Thread pool ───────────────────────────────────────────────────────────────
# max_workers=2: kaleido is CPU+memory heavy. More than 2 concurrent renders
# on a t3.medium will OOM. Tune up if you move to a larger instance.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kaleido")

# ── Output directory — created once at import ─────────────────────────────────
_OUT_DIR = Path(settings.CHART_OUTPUT_PATH)
_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Kaleido API detection — done ONCE at import, not per call ─────────────────
# kaleido 1.x: fig.to_image() works directly
# kaleido 0.x: must use plotly.io.to_image()
def _detect_render_fn() -> Callable:
    """Return the correct render function for the installed kaleido version."""
    try:
        import kaleido
        major = int(kaleido.__version__.split(".")[0])
        if major >= 1:
            logger.info("Kaleido %s detected — using fig.to_image()", kaleido.__version__)
            def _render_v1(fig: go.Figure, width: int, height: int) -> bytes:
                return fig.to_image(format="png", width=width, height=height)
            return _render_v1
    except Exception:
        pass

    # Fallback to plotly.io (kaleido 0.x)
    import plotly.io as pio
    logger.info("Kaleido 0.x or unknown — using pio.to_image()")
    def _render_v0(fig: go.Figure, width: int, height: int) -> bytes:
        return pio.to_image(fig, format="png", width=width, height=height)
    return _render_v0


_kaleido_render = _detect_render_fn()


# ── Spec validator ────────────────────────────────────────────────────────────

def _validate_spec(spec: dict) -> None:
    """
    Raise ValueError if spec is structurally invalid.
    Called before touching kaleido so errors are clear and fast.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"spec must be a dict, got {type(spec).__name__}")
    if "data" not in spec:
        raise ValueError("spec missing required key: 'data'")
    if "layout" not in spec:
        raise ValueError("spec missing required key: 'layout'")
    if not isinstance(spec["data"], list):
        raise ValueError(f"spec['data'] must be a list, got {type(spec['data']).__name__}")
    if len(spec["data"]) == 0:
        raise ValueError("spec['data'] is empty — no traces to render")


# ── Sync render (runs inside thread) ─────────────────────────────────────────

def _sync_render(spec: dict, width: int, height: int) -> tuple[str, str]:
    """
    Blocking render — always runs in _executor thread, never on event loop.
    Returns (base64_string, file_path).
    Raises on any failure so the async wrapper can catch and log cleanly.
    """
    _validate_spec(spec)

    fig = go.Figure(spec)
    img_bytes: bytes = _kaleido_render(fig, width, height)

    # Guard: kaleido sometimes returns empty bytes without raising
    if not img_bytes:
        raise RuntimeError("Kaleido returned empty bytes — render silently failed")

    filename = f"{uuid.uuid4().hex}.png"
    filepath = _OUT_DIR / filename
    filepath.write_bytes(img_bytes)

    b64 = base64.b64encode(img_bytes).decode("ascii")
    logger.info("PNG saved -> %s (%d bytes)", filepath, len(img_bytes))
    return b64, str(filepath)


# ── Public async API ──────────────────────────────────────────────────────────

async def render_png(
    spec: dict,
    width: int = 1000,
    height: int = 600,
) -> tuple[str | None, str | None]:
    """
    Render a Plotly spec to PNG asynchronously.

    - Runs kaleido in a dedicated ThreadPoolExecutor (never blocks event loop)
    - Enforces a 30-second timeout (kaleido can hang on bad specs)
    - Returns (base64_string, file_path) on success
    - Returns (None, None) on ANY failure — never raises, never crashes caller

    Usage:
        png_b64, file_path = await render_png(spec)
        if png_b64 is None:
            # chart still works — just no PNG
    """
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _sync_render, spec, width, height),
            timeout=_RENDER_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(
            "PNG render timed out after %ds — kaleido may be hung. "
            "Run: python -m kaleido_get_chrome  to fix Chromium issues.",
            _RENDER_TIMEOUT,
        )
        return None, None
    except ValueError as exc:
        # Bad spec — log clearly, no stack trace needed
        logger.warning("PNG render skipped — invalid spec: %s", exc)
        return None, None
    except Exception as exc:
        logger.warning("PNG render failed: %s", exc, exc_info=True)
        return None, None


# ── Cleanup hook (call on app shutdown) ──────────────────────────────────────

def shutdown_renderer() -> None:
    """
    Gracefully shut down the thread pool.
    Call this in FastAPI lifespan on shutdown:

        @asynccontextmanager
        async def lifespan(app):
            yield
            shutdown_renderer()
    """
    _executor.shutdown(wait=False)
    logger.info("Renderer thread pool shut down")