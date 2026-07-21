"""Market table and card presentation helpers."""

import pandas as pd
import streamlit as st

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
        selection_mode="multi-row",
        key=selection_key,
    )
    return event


def render_row_actions(df_page: pd.DataFrame, selection_event):
    if not selection_event or not selection_event.selection.rows:
        return

    selected_indices = [i for i in selection_event.selection.rows if i < len(df_page)]
    if not selected_indices:
        return

    selected_rows = df_page.iloc[selected_indices]
    count = len(selected_rows)
    st.caption(f"{count} market{'s' if count != 1 else ''} selected")

    if st.button(f"Add {count} to optimizer", key="add_basket_btn", type="primary"):
        added = sum(
            1
            for _, row in selected_rows.iterrows()
            if add_to_basket(row["Market ID"], row["ChainID"])
        )
        st.toast(f"Added {added} market{'s' if added != 1 else ''} to optimizer basket")
        st.rerun()
