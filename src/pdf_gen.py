"""PDF document generation module for Content Analyzer.

Transforms unstructured markdown analysis results into structured,
formatted PDF documents using fpdf2 and markdown HTML parsing.
"""

import re
from fpdf import FPDF
from fpdf.html import TextStyle
import markdown

# Character mapping for typographic Unicode symbols to standard equivalents
UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u2014": " - ",   # em-dash
    "\u2013": "-",     # en-dash
    "\u2018": "'",     # left single quotation mark
    "\u2019": "'",     # right single quotation mark
    "\u201c": '"',     # left double quotation mark
    "\u201d": '"',     # right double quotation mark
    "\u2022": "*",     # bullet symbol
    "\u2026": "...",   # horizontal ellipsis
}

# Typography style definitions for HTML tags in PDF output
DOCUMENT_TAG_STYLES: dict[str, TextStyle] = {
    "h1": TextStyle(font_size_pt=14, color="#09090b", font_style="B"),
    "h2": TextStyle(font_size_pt=12, color="#09090b", font_style="B"),
    "h3": TextStyle(font_size_pt=11, color="#18181b", font_style="B"),
    "h4": TextStyle(font_size_pt=10, color="#18181b", font_style="B"),
}


class ReportPDF(FPDF):
    """Custom PDF document layout with header and footer branding."""

    def header(self) -> None:
        """Render top document header."""
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(110, 110, 115)
        self.cell(0, 8, "Content Analyzer | Generated Report", border=0, align="L")
        self.ln(8)

    def footer(self) -> None:
        """Render bottom page numbering."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", border=0, align="C")


def _sanitize_markdown_headers(text: str) -> str:
    """Prevent social hashtags from being misinterpreted as Markdown headings.

    Lines beginning with hashtags without subsequent spaces (e.g. #Productivity)
    are escaped so that Markdown compilers treat them as body text rather than
    top-level heading elements.

    Args:
        text: The source markdown text.

    Returns:
        Sanitized markdown string with escaped leading hashtags.
    """
    return re.sub(r"(^|\n)#+([^\s#])", r"\1\\#\2", text)


def _normalize_text(text: str) -> str:
    """Normalize typography and remove unrepresentable characters for PDF output.

    Args:
        text: Input text containing special unicode characters.

    Returns:
        Latin-1 compatible string safe for standard PDF fonts.
    """
    for unicode_char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, replacement)
    return text.encode("latin-1", "ignore").decode("latin-1")


def create_pdf_from_text(title: str, content: str) -> bytes:
    """Generate a formatted PDF byte stream from markdown analysis content.

    Converts Markdown headings, bullet points, and inline styling into
    native PDF elements via HTML parsing with custom typography styles
    and plain-text fallback.

    Args:
        title: The document title (e.g. analysis mode heading).
        content: The Markdown or raw text content to format.

    Returns:
        Raw bytes representing the compiled PDF document.

    Raises:
        ValueError: If title or content is empty.
    """
    if not title or not title.strip():
        raise ValueError("Title cannot be empty.")
    if not content or not content.strip():
        raise ValueError("Content cannot be empty.")

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Document title block
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(20, 20, 24)
    sanitized_title = _normalize_text(title.strip())
    pdf.multi_cell(0, 8, text=sanitized_title)
    pdf.ln(4)

    # Content body formatted via HTML conversion with tag styling
    sanitized_body = _normalize_text(content.strip())
    sanitized_body = _sanitize_markdown_headers(sanitized_body)

    try:
        html_body = markdown.markdown(sanitized_body)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(45, 45, 50)
        pdf.write_html(html_body, tag_styles=DOCUMENT_TAG_STYLES)
    except Exception:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(45, 45, 50)
        pdf.multi_cell(0, 5, text=sanitized_body)

    return bytes(pdf.output())
