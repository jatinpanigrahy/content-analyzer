"""PDF generation module for Content Analyzer.

Transforms extracted insights and markdown text into clean, professional
downloadable PDF documents using fpdf2.
"""

from fpdf import FPDF

# Mapping of common Unicode typography to standard ASCII equivalents
UNICODE_REPLACEMENTS = {
    "\u2014": " - ",   # em-dash
    "\u2013": "-",     # en-dash
    "\u2018": "'",     # left single quotation mark
    "\u2019": "'",     # right single quotation mark
    "\u201c": '"',     # left double quotation mark
    "\u201d": '"',     # right double quotation mark
    "\u2022": "*",     # bullet symbol
    "\u2026": "...",   # horizontal ellipsis
}


class ReportPDF(FPDF):
    """Custom PDF document with standard header and footer branding."""

    def header(self) -> None:
        """Render standard top header."""
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(110, 110, 115)
        self.cell(0, 8, "Content Analyzer | Generated Report", border=0, align="L")
        self.ln(8)

    def footer(self) -> None:
        """Render standard bottom page numbering."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", border=0, align="C")


def _normalize_text(text: str) -> str:
    """Normalize special Unicode typographic characters for PDF compatibility."""
    for unicode_char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, replacement)
    return text.encode("latin-1", "replace").decode("latin-1")


def create_pdf_from_text(title: str, content: str) -> bytes:
    """Generate a formatted PDF byte stream from analysis content.

    Args:
        title: The header title (e.g. analysis mode).
        content: The text or markdown content to format.

    Returns:
        Raw bytes representing the compiled PDF document.

    Raises:
        ValueError: If content or title is empty.
    """
    if not title or not title.strip():
        raise ValueError("Title cannot be empty.")
    if not content or not content.strip():
        raise ValueError("Content cannot be empty.")

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title block
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(20, 20, 24)
    sanitized_title = _normalize_text(title.strip())
    pdf.multi_cell(0, 8, text=sanitized_title)
    pdf.ln(4)

    # Content body
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(45, 45, 50)
    sanitized_body = _normalize_text(content.strip())
    pdf.multi_cell(0, 5, text=sanitized_body)

    return bytes(pdf.output())
