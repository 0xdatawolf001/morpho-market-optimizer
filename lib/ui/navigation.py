"""Centralized page navigation helpers for st.navigation."""

import streamlit as st

MARKETS_PAGE = "pages/1_Markets.py"
MARKET_DETAIL_PAGE = "pages/2_Market_Detail.py"
OPTIMIZE_PAGE = "pages/3_Optimize.py"


def go_to_market_detail(chain_id: int, market_id: str):
    # switch_page clears query params — persist market context in session instead
    st.session_state.detail_market = {
        "chain_id": int(chain_id),
        "market_id": market_id,
    }
    st.switch_page(MARKET_DETAIL_PAGE)


def go_to_markets(clear_market_params: bool = True):
    if clear_market_params:
        for key in ("chain_id", "market_id"):
            if key in st.query_params:
                del st.query_params[key]
    st.switch_page(MARKETS_PAGE)


def go_to_optimize():
    st.switch_page(OPTIMIZE_PAGE)
