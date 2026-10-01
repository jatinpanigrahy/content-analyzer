"""Web scraping and content extraction module.

Fetches target URLs and cleans HTML DOM structures to isolate
readable textual content for processing and analysis.
"""

import requests
from bs4 import BeautifulSoup


def fetch_url_content(url: str, timeout: int = 10) -> str:
    """Fetch webpage content and extract clean, readable text.

    Args:
        url: The HTTP/HTTPS target URL.
        timeout: Request timeout in seconds.

    Returns:
        Cleaned plain text extracted from the webpage DOM.

    Raises:
        requests.RequestException: If the network request fails or returns non-200.
        ValueError: If no readable text can be extracted.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements and scripts to retain clean text
    unwanted_tags = ["script", "style", "nav", "footer", "header", "noscript", "svg"]
    for tag in soup(unwanted_tags):
        tag.decompose()

    extracted_text = soup.get_text(separator=" ", strip=True)

    if not extracted_text:
        raise ValueError("No readable text found at target URL.")

    return extracted_text
