"""
pages/Working_Paper.py — Embedded working paper for Coleman Coalition Analyzer.
"""
from __future__ import annotations

import base64
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

_pdf_path = Path(__file__).parent.parent / "assets" / "draft.pdf"
try:
    _pdf_bytes = _pdf_path.read_bytes()
    _b64 = base64.b64encode(_pdf_bytes).decode()
    st.components.v1.html(
        f'<iframe src="data:application/pdf;base64,{_b64}" '
        f'width="100%" height="900px" style="border:none;"></iframe>',
        height=920,
        scrolling=False,
    )
    st.download_button(
        "⬇️  Download draft PDF",
        data=_pdf_bytes,
        file_name="coleman_coalitions_draft.pdf",
        mime="application/pdf",
    )
except FileNotFoundError:
    st.info(
        "Draft PDF not found. Run `make paper` from the project root to compile it from source."
    )
