"""
pages/Working_Paper.py — Embedded working paper for Coleman Coalition Analyzer.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Working Paper — Coleman Coalition Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("Working Paper")
st.warning(
    "⚠️ **Unpublished draft — not peer-reviewed.** "
    "This document is a working paper shared for transparency and feedback. "
    "It has not been submitted to or accepted by any journal. "
    "Please do not cite without permission from the author."
)

# The PDF is served as a static asset at app/static/draft.pdf.
# Static serving is enabled via .streamlit/config.toml.
_static_pdf_url = "app/static/draft.pdf"
_local_pdf = Path(__file__).parent.parent / "static" / "draft.pdf"

st.components.v1.iframe(_static_pdf_url, height=900, scrolling=True)

if _local_pdf.exists():
    st.download_button(
        "⬇️  Download draft PDF",
        data=_local_pdf.read_bytes(),
        file_name="coleman_coalitions_draft.pdf",
        mime="application/pdf",
    )
else:
    st.info("Run `make paper` from the project root to compile the PDF from source.")
