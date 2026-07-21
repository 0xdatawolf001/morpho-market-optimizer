"""Market table and card presentation helpers."""

import pandas as pd
import streamlit as st

from lib.portfolio import add_to_basket
from lib.ui.navigation import go_to_market_detail


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


def _remember_selection(df_page: pd.DataFrame, selected_indices: list[int]):
    if len(selected_indices) != 1:
        return
    row = df_page.iloc[selected_indices[0]]
    st.session_state.selected_market = {
        "chain_id": int(row["ChainID"]),
        "market_id": row["Market ID"],
        "label": f"{row['Loan Token']}/{row['Collateral']}",
        "chain": row["Chain"],
        "supply_apy": float(row["Supply APY"]),
        "borrow_apy": float(row["Borrow APY"]),
        "utilization": float(row["Utilization"]),
        "liquidity": float(row["Available Liquidity (USD)"]),
    }


def remember_table_selection(df_page: pd.DataFrame, selection_event):
    if not selection_event or not selection_event.selection.rows:
        return
    selected_indices = [i for i in selection_event.selection.rows if i < len(df_page)]
    if len(selected_indices) == 1:
        _remember_selection(df_page, selected_indices)
    elif selected_indices:
        st.session_state.pop("selected_market", None)


def render_selection_toolbar():
    """Toolbar shown above the table when one market is selected (persists across pages)."""
    selected = st.session_state.get("selected_market")
    if not selected:
        return

    label = selected["label"]
    st.info(
        f"**Selected:** {label} · {selected['chain']} · "
        f"Supply {selected['supply_apy']:.2%} · Liquidity ${selected['liquidity']:,.0f}"
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("View details", type="primary", key="toolbar_view_market"):
        go_to_market_detail(selected["chain_id"], selected["market_id"])
    if c2.button("Add to optimizer", key="toolbar_add_basket"):
        if add_to_basket(selected["market_id"], selected["chain_id"]):
            st.toast("Added to optimizer basket")
        else:
            st.toast("Already in optimizer basket")
        st.rerun()
    if c3.button("Clear selection", key="toolbar_clear_selection"):
        st.session_state.pop("selected_market", None)
        st.rerun()


def render_selection_preview(row: pd.Series):
    """Compact preview when exactly one market is selected."""
    label = f"{row['Loan Token']}/{row['Collateral']}"
    st.markdown(
        f"""
        <div class="morpho-card">
            <h3>{label}</h3>
            <p>{row['Chain']} · Supply {row['Supply APY']:.2%} · Borrow {row['Borrow APY']:.2%}
            · Util {row['Utilization']:.1%} · Liquidity ${row['Available Liquidity (USD)']:,.0f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    if len(selected_indices) == 1:
        return

    selected_rows = df_page.iloc[selected_indices]
    count = len(selected_rows)
    st.caption(f"{count} markets selected")

    c1, c2 = st.columns([1, 1])
    c1.button("View market", disabled=True, help="Select a single market to view details")
    if c2.button(f"Add {count} to optimizer", key="add_basket_btn"):
        added = sum(
            1
            for _, row in selected_rows.iterrows()
            if add_to_basket(row["Market ID"], row["ChainID"])
        )
        st.toast(f"Added {added} market{'s' if added != 1 else ''} to optimizer basket")
        st.rerun()
