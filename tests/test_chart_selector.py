"""
Tests for app.llm.chart_selector
Mocks the Groq client so tests run without API key or network.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.llm.chart_selector import (
    _strip_markdown_fences,
    _parse_spec,
    generate_spec,
)


# ── Markdown fence stripping ─────────────────────────────────────────────────


class TestStripMarkdownFences:

    def test_no_fences(self):
        raw = '{"data": [], "layout": {}}'
        assert _strip_markdown_fences(raw) == raw

    def test_json_fence(self):
        raw = '```json\n{"data": [], "layout": {}}\n```'
        assert json.loads(_strip_markdown_fences(raw)) == {"data": [], "layout": {}}

    def test_plain_fence(self):
        raw = '```\n{"data": [], "layout": {}}\n```'
        assert json.loads(_strip_markdown_fences(raw)) == {"data": [], "layout": {}}


# ── Parse spec validation ────────────────────────────────────────────────────


class TestParseSpec:

    def test_valid_spec(self):
        raw = '{"data": [{"type": "bar"}], "layout": {"title": "Test"}}'
        spec = _parse_spec(raw)
        assert "data" in spec
        assert "layout" in spec

    def test_missing_data_key(self):
        raw = '{"layout": {}}'
        with pytest.raises(ValueError, match="missing"):
            _parse_spec(raw)

    def test_missing_layout_key(self):
        raw = '{"data": []}'
        with pytest.raises(ValueError, match="missing"):
            _parse_spec(raw)

    def test_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_spec("not json at all")


# ── generate_spec with mocked Groq ───────────────────────────────────────────

VALID_SPEC_JSON = json.dumps({
    "data": [{"type": "bar", "x": ["A"], "y": [1]}],
    "layout": {"title": "Test"},
})

INVALID_JSON = "this is not json"

MISSING_KEYS_JSON = json.dumps({"foo": "bar"})


def _mock_response(content: str) -> MagicMock:
    """Build a mock Groq ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestGenerateSpec:

    COLUMNS = [{"name": "x", "semantic": "categorical"}]
    SAMPLE = [{"x": "A"}]

    @patch("app.llm.chart_selector._get_client")
    def test_valid_response_first_try(self, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(VALID_SPEC_JSON)
        mock_get_client.return_value = client

        spec = generate_spec("bar", "Compare values", self.COLUMNS, self.SAMPLE)
        assert "data" in spec
        assert "layout" in spec
        assert client.chat.completions.create.call_count == 1

    @patch("app.llm.chart_selector._get_client")
    def test_invalid_json_triggers_retry_then_succeeds(self, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_response(INVALID_JSON),
            _mock_response(VALID_SPEC_JSON),
        ]
        mock_get_client.return_value = client

        spec = generate_spec("bar", "Compare values", self.COLUMNS, self.SAMPLE)
        assert "data" in spec
        assert client.chat.completions.create.call_count == 2

    @patch("app.llm.chart_selector._get_client")
    def test_double_failure_returns_fallback_spec(self, mock_get_client):
        """
        Both LLM attempts fail → deterministic fallback spec returned.
        User always gets a visual — no HTTP 500.
        """
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_response(INVALID_JSON),
            _mock_response(INVALID_JSON),
        ]
        mock_get_client.return_value = client

        spec = generate_spec("bar", "Compare values", self.COLUMNS, self.SAMPLE)
        # Fallback must still return a valid Plotly spec
        assert "data" in spec
        assert "layout" in spec
        assert client.chat.completions.create.call_count == 2

    @patch("app.llm.chart_selector._get_client")
    def test_markdown_fenced_response(self, mock_get_client):
        fenced = f"```json\n{VALID_SPEC_JSON}\n```"
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(fenced)
        mock_get_client.return_value = client

        spec = generate_spec("bar", "Compare values", self.COLUMNS, self.SAMPLE)
        assert "data" in spec
        assert "layout" in spec

    @patch("app.llm.chart_selector._get_client")
    def test_missing_keys_triggers_retry(self, mock_get_client):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _mock_response(MISSING_KEYS_JSON),
            _mock_response(VALID_SPEC_JSON),
        ]
        mock_get_client.return_value = client

        spec = generate_spec("bar", "Compare values", self.COLUMNS, self.SAMPLE)
        assert "data" in spec
        assert client.chat.completions.create.call_count == 2
