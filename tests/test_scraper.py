"""Unit test suite for the web scraper module.

Validates webpage retrieval, DOM sanitization, tag decomposition,
custom timeout handling, and network exception management.
All network interactions are mocked to ensure isolated, deterministic testing.
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest
import requests

# Ensure project root is in sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scraper import fetch_url_content


def test_fetch_url_content_success() -> None:
    """Validate successful text extraction from standard HTML."""
    sample_html = """
    <!DOCTYPE html>
    <html>
        <head><title>Test Article</title></head>
        <body>
            <h1>Engineering Architecture</h1>
            <p>Automated test suites ensure system stability across deployments.</p>
        </body>
    </html>
    """

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = sample_html

    with patch("requests.get", return_value=mock_response) as mock_get:
        result = fetch_url_content("https://example.com/article")

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["timeout"] == 10
        assert "Mozilla/5.0" in call_kwargs["headers"]["User-Agent"]

        assert "Engineering Architecture" in result
        assert "Automated test suites ensure system stability across deployments." in result


def test_fetch_url_content_strips_unwanted_tags() -> None:
    """Validate that boilerplate and non-content tags are stripped from output."""
    sample_html = """
    <html>
        <head>
            <style>body { color: red; }</style>
            <script>console.log("analytics script");</script>
        </head>
        <body>
            <header>Site Header Navigation</header>
            <nav><a href="/">Home</a></nav>
            <main>
                <article>
                    <p>Primary article content that should be preserved.</p>
                </article>
            </main>
            <footer>Copyright 2026 Corporation</footer>
            <noscript>Please enable JavaScript</noscript>
            <svg><path d="M0 0"/></svg>
        </body>
    </html>
    """

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = sample_html

    with patch("requests.get", return_value=mock_response):
        result = fetch_url_content("https://example.com/clean-test")

        assert "Primary article content that should be preserved." in result
        assert "Site Header Navigation" not in result
        assert "analytics script" not in result
        assert "color: red" not in result
        assert "Home" not in result
        assert "Copyright 2026" not in result
        assert "Please enable JavaScript" not in result


def test_fetch_url_content_custom_timeout() -> None:
    """Validate that custom timeout arguments are propagated to requests."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<p>Content with custom timeout.</p>"

    with patch("requests.get", return_value=mock_response) as mock_get:
        result = fetch_url_content("https://example.com/timeout", timeout=25)
        assert "Content with custom timeout." in result
        assert mock_get.call_args.kwargs["timeout"] == 25


def test_fetch_url_content_http_error() -> None:
    """Validate that HTTP client/server errors trigger raise_for_status."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error: Not Found")

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(requests.HTTPError):
            fetch_url_content("https://example.com/missing")


def test_fetch_url_content_network_timeout() -> None:
    """Validate that network timeouts are propagated as requests.Timeout exceptions."""
    with patch("requests.get", side_effect=requests.Timeout("Connection timed out")):
        with pytest.raises(requests.Timeout):
            fetch_url_content("https://example.com/slow")


def test_fetch_url_content_connection_error() -> None:
    """Validate that network connection errors are propagated."""
    with patch("requests.get", side_effect=requests.ConnectionError("DNS failure")):
        with pytest.raises(requests.ConnectionError):
            fetch_url_content("https://invalid-domain.example")


def test_fetch_url_content_empty_html_raises_value_error() -> None:
    """Validate that responses containing no text content raise ValueError."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body>   </body></html>"

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="No readable text found"):
            fetch_url_content("https://example.com/empty")


def test_fetch_url_content_only_scripts_raises_value_error() -> None:
    """Validate that responses containing only decomposed tags raise ValueError."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><script>let x = 1;</script><style>p {}</style></html>"

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(ValueError, match="No readable text found"):
            fetch_url_content("https://example.com/scripts-only")
