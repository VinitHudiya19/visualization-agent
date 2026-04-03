"""
Tests for app.utils.renderer
Mocks plotly.io.to_image so kaleido is not required for testing.
"""

import base64
from unittest.mock import patch, MagicMock

import pytest

from app.utils.renderer import render_png


SAMPLE_SPEC = {
    "data": [{"type": "bar", "x": ["A", "B"], "y": [10, 20]}],
    "layout": {"title": "Test Chart"},
}

FAKE_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # fake PNG header


@pytest.mark.asyncio
class TestRenderPng:

    @patch("app.utils.renderer.uuid")
    @patch("plotly.io.to_image", return_value=FAKE_PNG_BYTES)
    async def test_successful_render_returns_base64_and_path(self, mock_to_image, mock_uuid):
        mock_uuid.uuid4.return_value = MagicMock(hex="abc123deadbeef")

        b64, filepath = await render_png(SAMPLE_SPEC, width=800, height=400)

        assert b64 is not None
        assert isinstance(b64, str)
        # Verify it's valid base64
        decoded = base64.b64decode(b64)
        assert decoded == FAKE_PNG_BYTES
        # Verify file path contains UUID
        assert filepath is not None
        assert "abc123deadbeef" in filepath

    @patch("plotly.io.to_image", side_effect=Exception("Kaleido crashed"))
    async def test_kaleido_failure_returns_none(self, mock_to_image):
        b64, filepath = await render_png(SAMPLE_SPEC)

        assert b64 is None
        assert filepath is None
        # Verify it doesn't crash — reaching this line means success

    @patch("app.utils.renderer.uuid")
    @patch("plotly.io.to_image", return_value=FAKE_PNG_BYTES)
    async def test_custom_dimensions_passed_to_plotly(self, mock_to_image, mock_uuid):
        mock_uuid.uuid4.return_value = MagicMock(hex="dim_test")

        await render_png(SAMPLE_SPEC, width=1200, height=800)

        mock_to_image.assert_called_once()
        call_kwargs = mock_to_image.call_args
        assert call_kwargs.kwargs.get("width") == 1200 or call_kwargs[1].get("width") == 1200
