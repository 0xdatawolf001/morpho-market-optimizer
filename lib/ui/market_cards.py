"""Market table and card presentation helpers."""

import pandas as pd
import streamlit as st

from lib.config import monarch_link
from lib.portfolio import add_to_basket


def prepare_display_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["Market"] = out.apply(
        lambda r: f"{r['Loan Token']}/{r['Collateral']} ({str(r['Market ID'])[:8]}…)",
        axis=1,
    )
    out["Supply APY %"] = out["Supply APY"]
    out["Borrow APY %"] = out["Borrow APY"]
    out["Util %"] = out["Utilization"]
    out["LLTV %"] = out.get("LLTV", 0)
    return out


def render_market_table(df_page: pd.DataFrame, selection_key: str = "market_table"):
    if df_page.empty:
        st.info("No markets match your filters.")
        return None

    display = prepare_display_table(df_page)
    cols = [
        "Market",
        "Chain",
        "Supply APY %",
        "Borrow APY %",
        "Total Supply (USD)",
        "Available Liquidity (USD)",
        "Util %",
        "LLTV %",
        "Indicators",
        "Market ID",
    ]
    display = display[[c for c in cols if c in display.columns]]

    event = st.dataframe(
        display,
        column_config={
            "Supply APY %": st.column_config.NumberColumn("Supply APY", format="percent"),
            "Borrow APY %": st.column_config.NumberColumn("Borrow APY", format="percent"),
            "Util %": st.column_config.NumberColumn("Util", format="percent"),
            "LLTV %": st.column_config.NumberColumn("LLTV", format="percent"),
            "Total Supply (USD)": st.column_config.NumberColumn(format="dollar"),
            "Available Liquidity (USD)": st.column_config.NumberColumn("Liquidity", format="dollar"),
        },
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=selection_key,
    )
    return event


def render_row_actions(df_page: pd.DataFrame, selection_event):
    if not selection_event or not selection_event.selection.rows:
        return
    idx = selection_event.selection.rows[0]
    if idx >= len(df_page):
        return
    row = df_page.iloc[idx]
    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button("View market", type="primary", key="view_market_btn"):
        st.query_params["chain_id"] = str(int(row["ChainID"]))
        st.query_params["market_id"] = row["Market ID"]
        st.switch_page("pages/2_Market_Detail.py")
    if c2.button("Add to optimizer", key="add_basket_btn"):
        add_to_basket(row["Market ID"], row["ChainID"])
        st.toast(f"Added {row['Loan Token']}/{row['Collateral']} to basket")
    c3.markdown(f"[Open in Monarch]({monarch_link(row['ChainID'], row['Market ID'])})")
