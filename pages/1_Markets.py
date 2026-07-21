"""Markets discovery — Monarch-inspired live filters and basket."""

import pandas as pd
import streamlit as st

from lib.morpho_api import ensure_market_index, refresh_market_index
from lib.portfolio import init_session_defaults
from lib.ui.filters import apply_market_filters, paginate_df, render_filter_bar
from lib.ui.market_cards import render_market_table, render_row_actions
from lib.ui.theme import inject_theme, render_basket_sidebar, render_page_header, render_summary_cards

inject_theme()
init_session_defaults()

render_page_header("🔍 Markets", "Browse Morpho lending markets, filter live, and add candidates to your optimizer basket.")

try:
    with st.spinner("Loading market index from Morpho GraphQL…"):
        df_all = ensure_market_index()
except RuntimeError as exc:
    st.error(str(exc))
    if st.button("Retry loading markets"):
        st.session_state.pop("market_dict", None)
        refresh_market_index()
        st.rerun()
    st.stop()

if df_all.empty:
    st.error("Could not load markets. Check your network connection and try refreshing.")
    if st.button("Retry"):
        st.session_state.pop("market_dict", None)
        refresh_market_index()
        st.rerun()
    st.stop()

cols_to_num = [
    "Supply APY",
    "Borrow APY",
    "Utilization",
    "Available Liquidity (USD)",
    "Total Supply (USD)",
    "Total Borrow (USD)",
    "LLTV",
]
for c in cols_to_num:
    if c in df_all.columns:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce").fillna(0.0)

render_basket_sidebar(df_all)

filters = render_filter_bar(df_all)
df_filtered = apply_market_filters(df_all, filters)

median_apy = df_filtered["Supply APY"].median() if not df_filtered.empty else 0
total_liq = df_filtered["Available Liquidity (USD)"].sum() if not df_filtered.empty else 0
render_summary_cards(
    [
        ("Markets", f"{len(df_filtered):,}", "In current filter set"),
        ("Median Supply APY", f"{median_apy:.2%}", None),
        ("Total Liquidity", f"${total_liq:,.0f}", "Sum of available liquidity"),
        ("Index Size", f"{len(df_all):,}", "All tracked chains"),
    ]
)

df_page = paginate_df(df_filtered)
selection = render_market_table(df_page)
render_row_actions(df_page, selection)

foot1, foot2 = st.columns([3, 1])
foot1.caption("Data source: Morpho GraphQL API")
if foot2.button("Refresh index", key="refresh_markets"):
    st.session_state.market_dict = refresh_market_index()
    st.toast("Market index refreshed")
    st.rerun()
