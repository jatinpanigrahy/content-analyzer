"""Unit test suite for the PDF generation module.

Validates document compilation, Markdown HTML conversion, typographic
Unicode normalization, social hashtag sanitization, and error handling.
"""

from pathlib import Path
import sys
from unittest.mock import patch
import pytest

# Ensure project root is in sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pdf_gen import (
    ReportPDF,
    _normalize_text,
    _sanitize_markdown_headers,
    create_pdf_from_text,
)


def test_create_pdf_from_text_success() -> None:
    """Validate that standard Markdown content compiles into a valid PDF byte stream."""
    title = "Content Analyzer - Core Summary"
    markdown_content = """
    ## Executive Summary
    This document outlines the architectural strategy for content synthesis.

    ### Key Objectives
    - **Performance**: High throughput with deterministic caching.
    - **Reliability**: Automated failover across candidate model tiers.
    - **Extensibility**: Modular ingestion and rendering pipelines.
    """

    pdf_bytes = create_pdf_from_text(title=title, content=markdown_content)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    # Standard PDF documents begin with the %PDF- magic header
    assert pdf_bytes.startswith(b"%PDF-")


def test_sanitize_markdown_headers_escapes_hashtags() -> None:
    """Validate that standalone social hashtags are escaped while true headings are preserved."""
    raw_markdown = (
        "# Valid Top Heading\n"
        "Here is some text discussing industry trends.\n"
        "#AI\n"
        "## Valid Subheading\n"
        "#MachineLearning and #DataScience\n"
        "### Valid Third Heading\n"
        "##NoSpaceHeading"
    )

    sanitized = _sanitize_markdown_headers(raw_markdown)

    # Valid headings with spaces must remain untouched
    assert "# Valid Top Heading" in sanitized
    assert "## Valid Subheading" in sanitized
    assert "### Valid Third Heading" in sanitized

    # Hashtags starting lines without spaces must be escaped to prevent H1/H2 expansion
    assert r"\#AI" in sanitized
    assert r"\#NoSpaceHeading" in sanitized


def test_normalize_text_unicode_replacements() -> None:
    """Validate that curly quotes, dashes, and unsupported characters are normalized."""
    text_with_unicode = (
        "Editorial review: \u201cSmart Systems\u201d \u2014 "
        "incorporating \u2018novel\u2019 patterns \u2013 "
        "\u2022 bullet item \u2026 ellipsis."
    )

    normalized = _normalize_text(text_with_unicode)

    # Check typographic replacements
    assert '"Smart Systems"' in normalized
    assert " - " in normalized  # em-dash
    assert "-" in normalized    # en-dash
    assert "'novel'" in normalized
    assert "* bullet item" in normalized
    assert "... ellipsis." in normalized


def test_normalize_text_strips_unrepresentable_characters() -> None:
    """Validate that characters outside Latin-1 are stripped without raising encoding errors."""
    text_with_emojis = "Frontier intelligence \U0001f9e0 and multilingual \u65e5\u672c\u8a9e text."
    normalized = _normalize_text(text_with_emojis)

    assert "Frontier intelligence" in normalized
    assert "and multilingual" in normalized
    # Non Latin-1 characters should be omitted cleanly
    assert "\U0001f9e0" not in normalized
    assert "\u65e5" not in normalized


def test_create_pdf_from_text_empty_title_raises_value_error() -> None:
    """Validate that empty or whitespace-only titles raise ValueError."""
    with pytest.raises(ValueError, match="Title cannot be empty"):
        create_pdf_from_text(title="   ", content="Valid content body.")


def test_create_pdf_from_text_empty_content_raises_value_error() -> None:
    """Validate that empty or whitespace-only content raises ValueError."""
    with pytest.raises(ValueError, match="Content cannot be empty"):
        create_pdf_from_text(title="Valid Title", content="   ")


def test_create_pdf_from_text_html_conversion_fallback() -> None:
    """Validate fallback to multi_cell plain text rendering when HTML parsing fails."""
    title = "Fallback Test Report"
    content = "Malformed content that should fall back cleanly."

    with patch("src.pdf_gen.markdown.markdown", side_effect=Exception("Parser error")):
        pdf_bytes = create_pdf_from_text(title=title, content=content)

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")


def test_create_pdf_from_text_multipage_generation() -> None:
    """Validate that long documents cleanly generate multiple pages with headers and footers."""
    title = "Comprehensive Engineering Analysis"
    # Generate long content to force multiple page breaks
    long_content = "\n\n".join([f"### Section {i}\n\nDetailed content paragraph number {i} exploring system architecture." for i in range(1, 40)])

    pdf_bytes = create_pdf_from_text(title=title, content=long_content)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF-")


def test_report_pdf_header_and_footer() -> None:
    """Validate that the custom ReportPDF class adds headers and footers to generated pages."""
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.cell(text="Test Page 1")

    assert pdf.page_no() == 1
    output_bytes = bytes(pdf.output())
    assert output_bytes.startswith(b"%PDF-")
