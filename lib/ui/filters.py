"""Live filter bar helpers for market discovery."""

import pandas as pd
import streamlit as st

from lib.config import PAGE_SIZE


def get_tokens(df: pd.DataFrame, col: str) -> list[str]:
    unique_symbols = df[col].dropna().unique()
    return sorted([str(s) for s in unique_symbols if str(s).lower() != "nan"])


def count_active_filters(filters: dict) -> int:
    count = 0
    if filters.get("search"):
        count += 1
    if filters.get("chains"):
        count += 1
    if filters.get("loans"):
        count += 1
    if filters.get("collaterals"):
        count += 1
    for chip in ("high_apy", "deep_liq", "whitelisted", "low_util"):
        if filters.get(chip):
            count += 1
    if filters.get("hide_warnings"):
        count += 1
    if filters.get("min_apy", 0) > 0 or filters.get("max_apy", 1000) < 1000:
        count += 1
    if filters.get("min_util", 0) > 0 or filters.get("max_util", 100) < 100:
        count += 1
    if filters.get("min_supply", 0) > 0:
        count += 1
    if filters.get("min_liquidity", 0) > 0:
        count += 1
    return count


def apply_market_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    search = (filters.get("search") or "").strip().lower()
    if search:
        mask = (
            out["Market ID"].str.lower().str.contains(search, na=False)
            | out["Loan Token"].astype(str).str.lower().str.contains(search, na=False)
            | out["Collateral"].astype(str).str.lower().str.contains(search, na=False)
            | out["Market Label"].astype(str).str.lower().str.contains(search, na=False)
        )
        out = out[mask]

    if filters.get("chains"):
        out = out[out["Chain"].isin(filters["chains"])]
    if filters.get("loans"):
        out = out[out["Loan Token"].isin(filters["loans"])]
    if filters.get("collaterals"):
        out = out[out["Collateral"].isin(filters["collaterals"])]

    if filters.get("high_apy") and not out.empty:
        threshold = out["Supply APY"].quantile(0.75)
        out = out[out["Supply APY"] >= threshold]

    if filters.get("deep_liq"):
        out = out[out["Available Liquidity (USD)"] >= 1_000_000]

    if filters.get("whitelisted"):
        out = out[out["Whitelisted"]]

    if filters.get("low_util"):
        out = out[out["Utilization"] <= 0.5]

    if filters.get("hide_warnings"):
        out = out[out["Warnings"].apply(lambda w: len(w) == 0 if isinstance(w, list) else True)]

    min_apy = filters.get("min_apy", 0) / 100.0
    max_apy = filters.get("max_apy", 1000) / 100.0
    out = out[(out["Supply APY"] >= min_apy) & (out["Supply APY"] <= max_apy)]

    min_util = filters.get("min_util", 0) / 100.0
    max_util = filters.get("max_util", 100) / 100.0
    out = out[(out["Utilization"] >= min_util) & (out["Utilization"] <= max_util)]

    if filters.get("min_supply", 0) > 0:
        out = out[out["Total Supply (USD)"] >= filters["min_supply"]]
    if filters.get("min_liquidity", 0) > 0:
        out = out[out["Available Liquidity (USD)"] >= filters["min_liquidity"]]

    sort_key = filters.get("sort_key", "Total Supply (USD)")
    ascending = filters.get("sort_asc", False)
    if sort_key in out.columns:
        out = out.sort_values(sort_key, ascending=ascending)

    return out.reset_index(drop=True)


def render_filter_bar(df_all: pd.DataFrame) -> dict:
    loan_symbols = get_tokens(df_all, "Loan Token")
    collateral_symbols = get_tokens(df_all, "Collateral")
    unique_chains = sorted(df_all["Chain"].dropna().unique().tolist())

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        search = st.text_input("Search", placeholder="Market ID, loan, or collateral…", key="mkt_search")
    with c2:
        chain_sel = st.selectbox("Chain", ["All"] + unique_chains, key="mkt_chain")
    with c3:
        sort_key = st.selectbox(
            "Sort by",
            [
                "Total Supply (USD)",
                "Supply APY",
                "Borrow APY",
                "Available Liquidity (USD)",
                "Utilization",
                "LLTV",
            ],
            key="mkt_sort",
        )
    with c4:
        sort_dir = st.selectbox("Order", ["Desc", "Asc"], key="mkt_sort_dir")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        sel_loans = st.multiselect("Loan tokens", options=loan_symbols, key="mkt_loans")
    with r1c2:
        sel_colls = st.multiselect("Collateral tokens", options=collateral_symbols, key="mkt_colls")

    chip_cols = st.columns(4)
    high_apy = chip_cols[0].checkbox("High Supply APY", key="chip_high_apy")
    deep_liq = chip_cols[1].checkbox("Deep Liquidity", key="chip_deep_liq")
    whitelisted = chip_cols[2].checkbox("Whitelisted", key="chip_whitelist")
    low_util = chip_cols[3].checkbox("Low Utilization", key="chip_low_util")

    with st.expander("Advanced filters"):
        a1, a2 = st.columns(2)
        min_apy = a1.number_input("Min APY %", 0.0, 1000.0, 0.0, key="adv_min_apy")
        max_apy = a2.number_input("Max APY %", 0.0, 1000.0, 1000.0, key="adv_max_apy")
        u1, u2 = st.columns(2)
        min_util = u1.number_input("Min Util %", 0.0, 100.0, 0.0, key="adv_min_util")
        max_util = u2.number_input("Max Util %", 0.0, 100.0, 100.0, key="adv_max_util")
        s1, s2 = st.columns(2)
        min_supply = s1.number_input("Min Total Supply (USD)", 0.0, 10_000_000_000.0, 0.0, key="adv_min_supply")
        min_liquidity = s2.number_input("Min Liquidity (USD)", 0.0, 10_000_000_000.0, 0.0, key="adv_min_liq")
        hide_warnings = st.checkbox("Hide markets with warnings", key="adv_hide_warn")

    filters = {
        "search": search,
        "chains": [chain_sel] if chain_sel != "All" else [],
        "loans": sel_loans,
        "collaterals": sel_colls,
        "high_apy": high_apy,
        "deep_liq": deep_liq,
        "whitelisted": whitelisted,
        "low_util": low_util,
        "hide_warnings": hide_warnings,
        "min_apy": min_apy,
        "max_apy": max_apy,
        "min_util": min_util,
        "max_util": max_util,
        "min_supply": min_supply,
        "min_liquidity": min_liquidity,
        "sort_key": sort_key,
        "sort_asc": sort_dir == "Asc",
    }

    active = count_active_filters(filters)
    b1, b2, _ = st.columns([1, 1, 4])
    b1.caption(f"**{active}** active filter{'s' if active != 1 else ''}")
    if b2.button("Clear all filters", key="clear_filters"):
        for k in list(st.session_state.keys()):
            if k.startswith(("mkt_", "chip_", "adv_")):
                del st.session_state[k]
        st.rerun()

    return filters


def paginate_df(df: pd.DataFrame, page_key: str = "mkt_page") -> pd.DataFrame:
    if df.empty:
        return df
    total_pages = max(1, (len(df) - 1) // PAGE_SIZE + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key=page_key)
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    st.caption(f"Showing {start + 1}–{min(end, len(df))} of {len(df)} markets")
    return df.iloc[start:end].copy()
