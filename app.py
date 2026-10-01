"""Main application module for Content Analyzer.

Provides an interactive user interface to ingest web URLs or raw text,
route content through Google Gemini analysis modes, and export results
as Markdown, Plain Text, or PDF.
"""

import streamlit as st
from google.genai import errors

from src.scraper import fetch_url_content
from src.llm import generate_analysis, AVAILABLE_MODES
from src.pdf_gen import create_pdf_from_text

st.set_page_config(
    page_title="Content Analyzer",
    layout="wide",
    page_icon="assets/favicon.svg",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0a0a0c;
    }
    
    .stTextArea textarea {
        background-color: #121214 !important;
        border: 1px solid #27272a !important;
        color: #ededed !important;
        border-radius: 8px;
        padding: 12px;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: #121214 !important;
        border: 1px solid #27272a !important;
        border-radius: 8px;
    }
    
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
        cursor: pointer !important;
    }
    
    .stButton button {
        background-color: #ededed !important;
        border: none !important;
        border-radius: 8px;
        transition: all 0.2s ease;
        width: 100%;
        padding: 0.5rem 1rem;
    }
    
    .stButton button p, .stButton button div, .stButton button span {
        color: #0a0a0c !important;
        font-weight: 600 !important;
    }
    
    .stButton button:hover {
        background-color: #ffffff !important;
        transform: translateY(-1px);
    }
    
    div[data-testid="stContainer"] {
        background-color: #121214;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    h1, h2, h3, p, span, div {
        color: #ededed;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "output_data" not in st.session_state:
    st.session_state.output_data = None

st.title("Content Analyzer")
st.markdown(
    "Transform articles, notes, or web pages into structured insights, summaries, or communication formats."
)

col_mode, col_input = st.columns([1, 3])

with col_mode:
    utility_mode = st.selectbox(
        "Select Mode",
        AVAILABLE_MODES,
        label_visibility="collapsed",
    )

with col_input:
    user_input = st.text_area(
        "Input",
        placeholder="Enter raw text or a target URL (https://...)",
        height=200,
        label_visibility="collapsed",
    )

execute_btn = st.button("Analyze Content", type="primary")

if execute_btn:
    if user_input:
        target_content = ""
        if user_input.startswith("http://") or user_input.startswith("https://"):
            try:
                target_content = fetch_url_content(user_input)
            except Exception as e:
                st.error(f"Scraping error: {e}")
        else:
            target_content = user_input

        if target_content:
            with st.spinner("Processing content..."):
                try:
                    api_key = st.secrets.get("GEMINI_API_KEY", "")
                    if not api_key:
                        st.error("Configuration Error: GEMINI_API_KEY is missing from Streamlit secrets.")
                    else:
                        st.session_state.output_data = generate_analysis(
                            content=target_content,
                            mode=utility_mode,
                            api_key=api_key,
                        )
                except errors.APIError as e:
                    st.error(f"API communication error: {e}")
                except Exception as e:
                    st.error(f"System exception: {e}")
    else:
        st.warning("Input content is empty. Provide text or a valid URL.")

st.divider()

if st.session_state.output_data:
    st.subheader("Output")
    st.markdown(st.session_state.output_data)

    st.divider()
    st.markdown("### Export Options")
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
        except Exception as e:
            st.error(f"Error preparing PDF: {e}")
else:
    st.subheader("Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("### Multi-Source Input")
            st.markdown(
                "Process raw text blocks or direct web URLs via automated extraction."
            )
    with c2:
        with st.container(border=True):
            st.markdown("### Intelligent Inference")
            st.markdown(
                "Uses Google Gemini Flash models to analyze, extract, and reformat content."
            )
    with c3:
        with st.container(border=True):
            st.markdown("### 8 Analysis Modes")
            st.markdown(
                "Generate summaries, action items, outlines, Q&A, and targeted posts instantly."
            )
