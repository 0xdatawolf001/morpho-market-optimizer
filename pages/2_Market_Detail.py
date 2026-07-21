"""Market detail view with query-param routing."""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from lib.config import monarch_link
from lib.morpho_api import fetch_market_detail, fetch_market_historical
from lib.portfolio import add_to_basket, init_session_defaults
from lib.ui.charts import build_historical_line_chart, historical_series_to_df
from lib.ui.theme import inject_theme, render_empty_state, render_page_header

inject_theme()
init_session_defaults()

chain_id = st.query_params.get("chain_id")
market_id = st.query_params.get("market_id")

if not chain_id or not market_id:
    render_empty_state(
        "No market selected",
        "Open a market from the Markets page or add ?chain_id=&market_id= to the URL.",
    )
    if st.button("Browse markets"):
        st.switch_page("pages/1_Markets.py")
    st.stop()

try:
    chain_id_int = int(chain_id)
except ValueError:
    st.error("Invalid chain_id in URL.")
    st.stop()

with st.spinner("Loading market details…"):
    detail = fetch_market_detail(chain_id_int, market_id)

if not detail:
    st.error("Market not found. It may be on an unsupported chain or the ID is incorrect.")
    if st.button("Back to Markets"):
        st.switch_page("pages/1_Markets.py")
    st.stop()

label = detail["Market Label"]
st.markdown(f"### {label}")
badges = f"**{detail['Chain']}** · `{market_id[:12]}…`"
if detail.get("Whitelisted"):
    badges += " · ✅ Whitelisted"
st.caption(badges)

h1, h2, h3, h4 = st.columns(4)
h1.metric("Supply APY", f"{detail['Supply APY']:.2%}")
h2.metric("Borrow APY", f"{detail['Borrow APY']:.2%}")
h3.metric("Utilization", f"{detail['Utilization']:.2%}")
h4.metric("Available Liquidity", f"${detail['Available Liquidity (USD)']:,.0f}")

s1, s2, s3, s4, s5, s6 = st.columns(6)
s1.metric("Total Supply", f"${detail['Total Supply (USD)']:,.0f}")
s2.metric("Total Borrow", f"${detail['Total Borrow (USD)']:,.0f}")
s3.metric("LLTV", f"{detail.get('LLTV', 0):.1f}%")
s4.metric("Fee", f"{detail.get('Fee', 0):.4%}")
s5.metric("APY at Target", f"{detail.get('APY at Target', 0):.2%}")
s6.markdown(f"[Open in Monarch]({monarch_link(chain_id_int, market_id)})")

c1, c2 = st.columns([1, 3])
if c1.button("Add to Optimizer", type="primary"):
    add_to_basket(market_id, chain_id_int)
    st.toast(f"Added {label} to optimizer basket")
if c2.button("Go to Optimize page"):
    st.switch_page("pages/3_Optimize.py")

tab_overview, tab_rates, tab_risk = st.tabs(["Overview", "Rates", "Risk"])

with tab_overview:
    o1, o2 = st.columns(2)
    with o1:
        st.markdown("**Contracts**")
        st.write("Oracle", detail.get("Oracle Address") or "—")
        st.write("IRM", detail.get("IRM Address") or "—")
        st.write("Loan token", detail.get("Loan Address") or "—")
        st.write("Collateral", detail.get("Collateral Address") or "—")
    with o2:
        st.markdown("**Meta**")
        ts = detail.get("State Timestamp")
        if ts:
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            st.write("State updated", dt.strftime("%Y-%m-%d %H:%M UTC"))
        vaults = detail.get("Supplying Vaults") or []
        st.write("Supplying vaults", len(vaults))
        if vaults:
            for v in vaults[:5]:
                st.caption(v)
            if len(vaults) > 5:
                st.caption(f"…and {len(vaults) - 5} more")

with tab_rates:
    rolling = pd.DataFrame(
        [
            {
                "Window": "1d",
                "Supply APY": detail.get("Daily Supply APY", 0),
                "Borrow APY": detail.get("Daily Borrow APY", 0),
            },
            {
                "Window": "7d",
                "Supply APY": detail.get("Weekly Supply APY", 0),
                "Borrow APY": detail.get("Weekly Borrow APY", 0),
            },
            {
                "Window": "30d",
                "Supply APY": detail.get("Monthly Supply APY", 0),
                "Borrow APY": detail.get("Monthly Borrow APY", 0),
            },
        ]
    )
    st.dataframe(
        rolling.style.format({"Supply APY": "{:.2%}", "Borrow APY": "{:.2%}"}),
        hide_index=True,
        use_container_width=True,
    )

    hist = fetch_market_historical(chain_id_int, market_id, days=30)
    if hist:
        c_a, c_b = st.columns(2)
        supply_df = historical_series_to_df(hist.get("supplyApy"), "Supply APY")
        borrow_df = historical_series_to_df(hist.get("borrowApy"), "Borrow APY")
        util_df = historical_series_to_df(hist.get("utilization"), "Utilization")
        liq_df = historical_series_to_df(hist.get("liquidityAssetsUsd"), "Liquidity USD")

        chart_supply = build_historical_line_chart(supply_df, "Supply APY", "Supply APY")
        chart_borrow = build_historical_line_chart(borrow_df, "Borrow APY", "Borrow APY")
        if chart_supply:
            c_a.subheader("Supply APY (30d)")
            c_a.altair_chart(chart_supply, use_container_width=True)
        if chart_borrow:
            c_b.subheader("Borrow APY (30d)")
            c_b.altair_chart(chart_borrow, use_container_width=True)

        c_c, c_d = st.columns(2)
        chart_util = build_historical_line_chart(util_df, "Utilization", "Utilization")
        chart_liq = build_historical_line_chart(liq_df, "Liquidity USD", "Liquidity USD", fmt="$,.0f")
        if chart_util:
            c_c.subheader("Utilization (30d)")
            c_c.altair_chart(chart_util, use_container_width=True)
        if chart_liq:
            c_d.subheader("Liquidity USD (30d)")
            c_d.altair_chart(chart_liq, use_container_width=True)
    else:
        st.info("Historical timeseries unavailable for this market.")

with tab_risk:
    warnings = detail.get("Warnings") or []
    levels = detail.get("Warning Levels") or []
    if warnings:
        warn_rows = [{"Type": t, "Level": levels[i] if i < len(levels) else ""} for i, t in enumerate(warnings)]
        st.dataframe(pd.DataFrame(warn_rows), hide_index=True, use_container_width=True)
    else:
        st.success("No active warnings from Morpho API.")

    r1, r2, r3 = st.columns(3)
    r1.metric("Bad debt (USD)", f"${detail.get('Bad Debt (USD)', 0):,.2f}")
    r2.metric("Realized bad debt (USD)", f"${detail.get('Realized Bad Debt (USD)', 0):,.2f}")
    r3.metric("Utilization", f"{detail['Utilization']:.1%}")
    st.progress(min(1.0, detail["Utilization"]), text="Utilization gauge")
