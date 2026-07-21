"""Morpho Market Optimizer — entry point with multi-page navigation."""

import streamlit as st

from lib.portfolio import init_session_defaults
from lib.ui.theme import inject_theme

st.set_page_config(page_title="Morpho Market Optimizer", layout="wide", page_icon="⚖️")
inject_theme()
init_session_defaults()

pages = [
    st.Page("pages/1_Markets.py", title="Markets", icon="🔍", default=True),
    st.Page("pages/2_Market_Detail.py", title="Market", icon="📊", url_path="market"),
    st.Page("pages/3_Optimize.py", title="Optimize", icon="⚖️", url_path="optimize"),
]

pg = st.navigation(pages)
pg.run()
