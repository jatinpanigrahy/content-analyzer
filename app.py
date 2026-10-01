"""Main application module for Content Analyzer.

Provides an interactive user interface to ingest web URLs or raw text,
route content through Google Gemini analysis modes, and export results
as Markdown, Plain Text, or PDF with one-click clipboard copying.
"""

import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from google.genai import errors

from src.scraper import fetch_url_content
from src.llm import generate_analysis, AVAILABLE_MODES
from src.pdf_gen import create_pdf_from_text


def load_stylesheet(css_path: str = "assets/style.css") -> None:
    """Load external CSS styling into the Streamlit document."""
    path = Path(css_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_copy_button(text: str) -> None:
    """Render a one-click clipboard copy button with transient confirmation."""
    escaped_text = json.dumps(text)
    button_html = f"""
    <div style="display: flex; align-items: center; margin: 0; padding: 0;">
        <button id="copy-btn" onclick="copyContent()" style="
            background-color: #ededed;
            color: #0a0a0c;
            border: none;
            border-radius: 8px;
            padding: 0.45rem 1rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        ">Copy to Clipboard</button>
    </div>
    <script>
    function copyContent() {{
        const text = {escaped_text};
        navigator.clipboard.writeText(text).then(function() {{
            const btn = document.getElementById("copy-btn");
            btn.innerText = "Copied to Clipboard";
            btn.style.backgroundColor = "#22c55e";
            btn.style.color = "#ffffff";
            setTimeout(function() {{
                btn.innerText = "Copy to Clipboard";
                btn.style.backgroundColor = "#ededed";
                btn.style.color = "#0a0a0c";
            }}, 2000);
        }}).catch(function(err) {{
            console.error("Clipboard write error:", err);
        }});
    }}
    </script>
    """
    components.html(button_html, height=45)


# Page setup
st.set_page_config(
    page_title="Content Analyzer",
    layout="wide",
    page_icon="assets/favicon.svg",
    initial_sidebar_state="collapsed",
)

# Apply external stylesheet
load_stylesheet("assets/style.css")

# Initialize session state
if "output_data" not in st.session_state:
    st.session_state.output_data = None
if "model_used" not in st.session_state:
    st.session_state.model_used = None

# Centered workspace layout
_, center_col, _ = st.columns([1, 6, 1])

with center_col:
    st.title("Content Analyzer")
    st.markdown(
        "Transform articles, notes, or web pages into structured insights, summaries, or communication formats."
    )

    # Input card container
    with st.container(border=True):
        utility_mode = st.selectbox(
            "Select Analysis Mode",
            AVAILABLE_MODES,
            help="Choose the analysis format for your content.",
        )

        user_input = st.text_area(
            "Content Input",
            placeholder="Paste raw text or enter a web URL (https://...)",
            height=200,
            label_visibility="collapsed",
        )

        execute_btn = st.button("Analyze Content", type="primary")

    if execute_btn:
        if user_input and user_input.strip():
            api_key = st.secrets.get("GEMINI_API_KEY", "")
            if not api_key:
                st.error("Configuration Error: GEMINI_API_KEY is missing from Streamlit secrets.")
            else:
                target_content = ""
                with st.status("Analyzing content...", expanded=True) as status:
                    is_url = user_input.startswith("http://") or user_input.startswith("https://")
                    if is_url:
                        status.write("Fetching and extracting web content...")
                        try:
                            target_content = fetch_url_content(user_input.strip())
                        except Exception as scrape_err:
                            status.update(label="Scraping failed", state="error", expanded=True)
                            st.error(f"Web extraction error: {scrape_err}")
                    else:
                        target_content = user_input.strip()

                    if target_content:
                        status.write("Processing content with Google Gemini...")
                        try:
                            output_text, model_name = generate_analysis(
                                content=target_content,
                                mode=utility_mode,
                                api_key=api_key,
                            )
                            st.session_state.output_data = output_text
                            st.session_state.model_used = model_name
                            status.update(
                                label="Analysis complete",
                                state="complete",
                                expanded=False,
                            )
                        except errors.APIError as api_err:
                            status.update(label="API communication error", state="error")
                            st.error(f"API communication error: {api_err}")
                        except Exception as sys_err:
                            status.update(label="Processing failed", state="error")
                            st.error(f"System error: {sys_err}")
        else:
            st.warning("Input content is empty. Provide text or a valid URL.")

    st.divider()

    # Results rendering or empty-state feature cards
    if st.session_state.output_data:
        st.subheader("Output")
        if st.session_state.get("model_used"):
            st.caption(f"Generated via `{st.session_state.model_used}`")

        # Output card
        with st.container(border=True):
            st.markdown(st.session_state.output_data)

        st.markdown("### Export & Actions")
        col_copy, col_spacer = st.columns([2, 4])
        with col_copy:
            render_copy_button(st.session_state.output_data)

        col_md, col_txt, col_pdf = st.columns(3)
        with col_md:
            st.download_button(
                label="Download Markdown (.md)",
                data=st.session_state.output_data,
                file_name="analysis.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col_txt:
            st.download_button(
                label="Download Text (.txt)",
                data=st.session_state.output_data,
                file_name="analysis.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_pdf:
            try:
                pdf_bytes = create_pdf_from_text(
                    title=f"Content Analyzer - {utility_mode}",
                    content=st.session_state.output_data,
                )
                st.download_button(
                    label="Download PDF (.pdf)",
                    data=pdf_bytes,
                    file_name="analysis.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as pdf_err:
                st.error(f"Error compiling PDF: {pdf_err}")
    else:
        st.subheader("Features")
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown("### :material/language: Any Source")
                st.markdown(
                    "Process raw text blocks or web URLs via automated extraction."
                )
        with c2:
            with st.container(border=True):
                st.markdown("### :material/memory: Gemini Flash")
                st.markdown(
                    "High-performance inference with automated model discovery and fallback."
                )
        with c3:
            with st.container(border=True):
                st.markdown("### :material/dashboard: 8 Analysis Modes")
                st.markdown(
                    "Generate summaries, action items, outlines, Q&A, and targeted posts."
                )
