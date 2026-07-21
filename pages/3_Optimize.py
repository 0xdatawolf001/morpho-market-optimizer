"""Portfolio optimization across selected Morpho markets."""

import time

import numpy as np
import pandas as pd
import streamlit as st

from lib.config import (
    MANUAL_SEP,
    SCOPE_LABELS,
    SECONDS_PER_YEAR,
    STRATEGY_NAMES,
    WALLET_SEP,
    monarch_link,
    scope_code,
)
from lib.math_utils import (
    apy_to_rate_per_second,
    extract_market_id_from_monarch_link,
    filter_small_moves,
    format_action_label,
)
from lib.morpho_api import (
    build_price_lookup,
    ensure_market_index,
    fetch_live_market_details,
    fetch_user_positions,
    refresh_market_index,
)
from lib.optimizer import RebalanceOptimizer
from lib.portfolio import (
    basket_market_ids,
    get_basket,
    handle_text_change,
    init_session_state,
    parse_market_ids_from_text,
    sync_portfolio_text_from_basket,
)
from lib.ui.charts import (
    render_allocation_bars,
    render_convergence_chart,
    render_efficiency_frontier,
)
from lib.ui.theme import (
    inject_theme,
    render_basket_sidebar,
    render_optimizer_basket,
    render_page_header,
    render_simulation_banner,
)

inject_theme()
init_session_state()

render_page_header(
    "⚖️ Optimize",
    "Model rebalance strategies across your Morpho lending portfolio.",
)

try:
    with st.spinner("Loading market index…"):
        df_all = ensure_market_index()
except RuntimeError as exc:
    st.error(str(exc))
    if st.button("Retry loading markets", key="opt_retry_api"):
        st.session_state.pop("market_dict", None)
        refresh_market_index()
        st.rerun()
    st.stop()

if df_all.empty:
    st.error("Could not load markets. Check your network connection and try again.")
    if st.button("Retry", key="opt_retry_empty"):
        st.session_state.pop("market_dict", None)
        refresh_market_index()
        st.rerun()
    st.stop()

render_basket_sidebar(df_all)

# --- Basket → portfolio text sync (preserve wallet section) ---
if get_basket():
    wallet_lines = []
    current_text = st.session_state.get("portfolio_input_text", "")
    if WALLET_SEP in current_text:
        wallet_lines = [
            ln
            for ln in current_text.split(WALLET_SEP, 1)[1].split("\n")
            if ln.strip() and MANUAL_SEP not in ln
        ]
    sync_portfolio_text_from_basket(df_all)
    if wallet_lines:
        st.session_state.portfolio_input_text = (
            st.session_state.portfolio_input_text.rstrip()
            + f"\n\n{WALLET_SEP}\n"
            + "\n".join(wallet_lines)
        )

basket_ids = basket_market_ids(df_all)
render_optimizer_basket(df_all)

# =============================================================================
# 1. Portfolio
# =============================================================================
st.subheader("1. Portfolio")

u_col1, u_col2, u_col3 = st.columns([2, 1, 1], vertical_alignment="bottom")
with u_col1:
    user_wallet = st.text_input("Auto-fill from wallet address", placeholder="0x…", key="opt_wallet")
with u_col2:
    scan_clicked = st.button("Import from wallet", type="primary", use_container_width=True)
with u_col3:
    clear_all_clicked = st.button("Clear all", type="secondary", use_container_width=True)

price_lookup = build_price_lookup(df_all)

if scan_clicked:
    if len(user_wallet) < 10:
        current_text = st.session_state.portfolio_input_text
        if WALLET_SEP in current_text:
            manual_part = current_text.split(WALLET_SEP)[0].strip()
            st.session_state.portfolio_input_text = manual_part
            st.warning("Address cleared: wallet markets removed.")
        else:
            st.info("No wallet markets to remove.")
        st.rerun()
    else:
        found_positions = fetch_user_positions(user_wallet, price_lookup)
        if found_positions:
            st.session_state.balance_cache.update(found_positions)
            new_wallet_ids = {k.lower() for k in found_positions.keys()}

            current_text = st.session_state.portfolio_input_text
            manual_ids = []
            if WALLET_SEP in current_text:
                manual_text = current_text.split(WALLET_SEP)[0]
            else:
                manual_text = current_text

            for line in manual_text.replace(",", "\n").split("\n"):
                potential = line.split("--")[0].strip()
                if not potential:
                    continue
                mid = extract_market_id_from_monarch_link(potential).lower()
                if mid.startswith("0x") and len(mid) > 20 and mid not in new_wallet_ids:
                    manual_ids.append(mid)

            final_parts = []
            if manual_ids:
                final_parts.append(MANUAL_SEP)
                seen_man = set()
                for mid in manual_ids:
                    if mid not in seen_man:
                        row = df_all[df_all["Market ID"].str.lower() == mid]
                        label = (
                            f" -- {row.iloc[0]['Loan Token']}/{row.iloc[0]['Collateral']}"
                            if not row.empty
                            else ""
                        )
                        final_parts.append(f"{mid}{label}")
                        seen_man.add(mid)
                final_parts.append("")

            final_parts.append(WALLET_SEP)
            for mid in new_wallet_ids:
                row = df_all[df_all["Market ID"].str.lower() == mid]
                label = (
                    f" -- {row.iloc[0]['Loan Token']}/{row.iloc[0]['Collateral']}"
                    if not row.empty
                    else ""
                )
                final_parts.append(f"{mid}{label}")

            st.session_state.portfolio_input_text = "\n".join(final_parts).strip()
            st.success(f"Loaded {len(found_positions)} markets from wallet.")
            time.sleep(0.5)
            st.rerun()
        else:
            st.warning("No active Morpho positions found for this address.")

if clear_all_clicked:
    st.session_state.portfolio_input_text = ""
    st.session_state.balance_cache = {}
    st.success("All markets and balances cleared.")
    time.sleep(0.5)
    st.rerun()

with st.expander("Advanced: paste market IDs or Monarch links", expanded=not bool(basket_ids)):
    st.text_area(
        "Paste market IDs or Monarch links (one per line)",
        height=200,
        placeholder="Market IDs or links…",
        key="portfolio_input_text",
        on_change=lambda: handle_text_change(df_all),
    )

clean_ids = parse_market_ids_from_text(st.session_state.get("portfolio_input_text", ""))
if not clean_ids and basket_ids:
    clean_ids = basket_ids

df_selected = df_all[df_all["Market ID"].str.lower().isin([c.lower() for c in clean_ids])].copy()

if df_selected.empty:
    st.info("Add markets from the Markets page basket, import a wallet, or paste IDs above.")
    st.stop()

# =============================================================================
# 2. Parameters
# =============================================================================
st.subheader("2. Parameters")

tab_scope, tab_dominance, tab_safety = st.tabs(["Scope & budget", "Dominance", "Safety"])

with tab_scope:
    rebalance_scope = st.selectbox(
        "Optimization constraint",
        options=SCOPE_LABELS,
        index=0,
        help="Limits where funds can move. The optimizer runs independently per group.",
    )
    new_cash = st.number_input(
        "Additional new cash / withdrawal (USD)",
        value=0.0,
        help="Positive = add capital. Negative = withdraw (distributed proportionally to silos).",
    )
    min_move_thresh = st.number_input(
        "Min rebalance threshold ($)",
        value=0.0,
        help="Only allocate when the move crosses this amount. Prevents dust moves.",
    )

with tab_dominance:
    max_supply_dom = st.number_input(
        "Max supply dominance %",
        value=10.0,
        help="You will never represent more than X% of the market's total supply.",
    )
    max_borrow_dom = st.number_input(
        "Max borrow dominance %",
        value=10.0,
        help="Your supply will never exceed X% of the market's total borrows.",
    )

with tab_safety:
    max_port_alloc = st.number_input(
        "Max portfolio allocation %",
        value=100.0,
        help="No single market can exceed this % of your total wealth.",
    )
    max_dominance = st.number_input(
        "Whale shield: max liquidity %",
        value=50.0,
        help="You will never own more than this % of available liquidity in a pool.",
    )

# Portfolio editor
df_selected["Existing Balance (USD)"] = df_selected["Market ID"].apply(
    lambda x: st.session_state.balance_cache.get(x, 0.0)
)
if "Force Exit" not in df_selected.columns:
    df_selected["Force Exit"] = False
if "Prevent Outflows" not in df_selected.columns:
    df_selected["Prevent Outflows"] = False
if "Prevent Inflows" not in df_selected.columns:
    df_selected["Prevent Inflows"] = False

df_selected["Link To Market"] = [
    monarch_link(row["ChainID"], row["Market ID"]) for _, row in df_selected.iterrows()
]
df_selected = df_selected.sort_values(by="Existing Balance (USD)", ascending=False)


def sync_portfolio_edits():
    if "portfolio_editor" not in st.session_state:
        return
    edits = st.session_state["portfolio_editor"].get("edited_rows", {})
    for idx, changes in edits.items():
        if "Existing Balance (USD)" in changes:
            m_id = df_selected.iloc[idx]["Market ID"]
            raw_val = changes["Existing Balance (USD)"]
            try:
                val = float(raw_val) if raw_val is not None else 0.0
            except (ValueError, TypeError):
                val = 0.0
            st.session_state.balance_cache[m_id] = val


st.markdown(
    """
    <div style="background-color: transparent; padding: 10px; border-radius: 5px;
                border-left: 5px solid #4caf50; margin-bottom: 10px;">
        <strong>Portfolio tip:</strong> Fill in
        <span style="color: #2e7d32; font-weight: bold;">Existing Balance (USD)</span>
        if holdings were not imported from your wallet, or to simulate positions.
    </div>
    """,
    unsafe_allow_html=True,
)

edited_df = st.data_editor(
    df_selected[
        [
            "Market ID",
            "Chain",
            "Loan Token",
            "Collateral",
            "Supply APY",
            "Utilization",
            "Total Supply (USD)",
            "Total Borrow (USD)",
            "Available Liquidity (USD)",
            "Existing Balance (USD)",
            "Prevent Outflows",
            "Prevent Inflows",
            "Force Exit",
            "Link To Market",
        ]
    ],
    column_config={
        "Market ID": st.column_config.TextColumn(width=100),
        "Chain": st.column_config.TextColumn(width=80),
        "Loan Token": st.column_config.TextColumn(width=80),
        "Collateral": st.column_config.TextColumn(width=80),
        "Supply APY": st.column_config.NumberColumn(format="percent", width=70),
        "Utilization": st.column_config.NumberColumn(format="percent", width=70),
        "Total Supply (USD)": st.column_config.NumberColumn(format="dollar", width=100),
        "Total Borrow (USD)": st.column_config.NumberColumn(format="dollar", width=100),
        "Available Liquidity (USD)": st.column_config.NumberColumn(format="dollar", width=100),
        "Existing Balance (USD)": st.column_config.NumberColumn(
            "Existing Balance (USD)",
            help="Current market balance used to calculate rebalance moves.",
            format="dollar",
            min_value=0.0,
            width=150,
        ),
        "Prevent Outflows": st.column_config.CheckboxColumn(
            "No outflows",
            help="Never sell or withdraw from this market (hold or buy only).",
            default=False,
            width=90,
        ),
        "Prevent Inflows": st.column_config.CheckboxColumn(
            "No inflows",
            help="Never add new funds to this market (hold or sell only).",
            default=False,
            width=90,
        ),
        "Force Exit": st.column_config.CheckboxColumn(
            "Force exit?",
            help="Forcefully sell 100% of this position.",
            default=False,
            width=80,
        ),
        "Link To Market": st.column_config.LinkColumn("Link", display_text="Link", width=80),
    },
    disabled=[
        "Market ID",
        "Chain",
        "Loan Token",
        "Collateral",
        "Supply APY",
        "Utilization",
        "Total Supply (USD)",
        "Total Borrow (USD)",
        "Available Liquidity (USD)",
        "Link To Market",
    ],
    width="stretch",
    hide_index=True,
    key="portfolio_editor",
    on_change=sync_portfolio_edits,
)

force_exit_map = dict(zip(edited_df["Market ID"], edited_df["Force Exit"]))
prevent_out_map = dict(zip(edited_df["Market ID"], edited_df["Prevent Outflows"]))
prevent_in_map = dict(zip(edited_df["Market ID"], edited_df["Prevent Inflows"]))

current_wealth = sum(st.session_state.balance_cache.get(m_id, 0.0) for m_id in df_selected["Market ID"])
total_optimizable = current_wealth + new_cash

# =============================================================================
# 3. Run
# =============================================================================
st.subheader("3. Run")

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Total current balance", f"${current_wealth:,.2f}")
if new_cash < 0:
    metric_col2.metric("Withdrawing", f"-${abs(new_cash):,.2f}")
else:
    metric_col2.metric("Additional cash", f"${new_cash:,.2f}")
metric_col3.metric("Target portfolio size", f"${total_optimizable:,.2f}")

if total_optimizable < 0:
    st.error("Withdrawal amount exceeds total current wealth. Please adjust.")
    st.stop()

with st.expander("About portfolio strategies", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Best yield**")
        st.caption("Aggressive. Maximizes mathematical APY, often concentrating in one market.")
    with c2:
        st.markdown("**Whale shield**")
        st.caption(
            f"Protective. Caps ownership at **{max_dominance}%** of available liquidity per pool."
        )
    with c3:
        st.markdown("**Frontier**")
        st.caption("Balanced Pareto trade-off between yield and diversity (1 − HHI).")
    with c4:
        st.markdown("**Liquid-yield**")
        st.caption("Depth-first. Favors deep liquidity over marginal APY.")

if st.button("Run optimization", type="primary", use_container_width=True):
    sync_portfolio_edits()

    market_data_list = fetch_live_market_details(df_selected)
    for m in market_data_list:
        m["existing_balance_usd"] = st.session_state.balance_cache.get(m["Market ID"], 0.0)
        m["force_exit"] = force_exit_map.get(m["Market ID"], False)
        m["prevent_outflows"] = prevent_out_map.get(m["Market ID"], False)
        m["prevent_inflows"] = prevent_in_map.get(m["Market ID"], False)

    current_annual_interest = sum(
        m["existing_balance_usd"] * m["current_supply_apy"] for m in market_data_list
    )
    current_blended_apy = current_annual_interest / current_wealth if current_wealth > 0 else 0.0

    silos: dict = {}
    for idx, m in enumerate(market_data_list):
        sc = scope_code(rebalance_scope)
        if sc == 0:
            key = "GLOBAL"
        elif sc == 1:
            key = (m["ChainID"], m["Loan Address"])
        elif sc == 2:
            key = m["ChainID"]
        elif sc == 3:
            key = m["Loan Address"]
        else:
            key = "GLOBAL"
        silos.setdefault(key, []).append(idx)

    n = len(market_data_list)
    final_y = np.zeros(n)
    final_f = np.zeros(n)
    final_l = np.zeros(n)
    final_w = np.zeros(n)

    global_yield_trace = []
    global_frontier_trace = []
    global_liquid_trace = []
    global_whale_trace = []
    global_attempts = []
    global_capacity_warnings = []

    my_bar = st.progress(0, text="Optimizing silos…")
    silo_count = len(silos)
    global_port_cap_usd = total_optimizable * (max_port_alloc / 100.0)

    for i, (key, indices) in enumerate(silos.items()):
        silo_markets = [market_data_list[idx] for idx in indices]
        silo_existing_wealth = sum(m["existing_balance_usd"] for m in silo_markets)

        if current_wealth > 0:
            share_of_port = silo_existing_wealth / current_wealth
            silo_budget = silo_existing_wealth + (new_cash * share_of_port)
        else:
            silo_budget = new_cash / len(silos)

        if silo_budget <= 0.01:
            my_bar.progress((i + 1) / silo_count, text=f"Skipping empty silo {key}…")
            continue

        adjusted_max_alloc_pct = (
            (global_port_cap_usd / silo_budget) * 100.0 if silo_budget > 0 else 100.0
        )

        opt = RebalanceOptimizer(
            total_budget=silo_budget,
            market_list=silo_markets,
            max_dominance_pct=max_dominance,
            max_port_alloc_pct=adjusted_max_alloc_pct,
            max_supply_pct=max_supply_dom,
            max_borrow_pct=max_borrow_dom,
        )
        r_y, r_f, r_l, r_w = opt.optimize()

        for local_idx, global_idx in enumerate(indices):
            final_y[global_idx] = r_y[local_idx]
            final_f[global_idx] = r_f[local_idx]
            final_l[global_idx] = r_l[local_idx]
            final_w[global_idx] = r_w[local_idx]

        global_yield_trace.extend(opt.yield_trace)
        global_frontier_trace.extend(opt.frontier_trace)
        global_liquid_trace.extend(opt.liquid_trace)
        global_whale_trace.extend(opt.whale_trace)
        global_attempts.extend(opt.all_attempts)
        if opt.capacity_warning:
            global_capacity_warnings.append(f"Silo [{key}]: {opt.capacity_warning}")

        my_bar.progress((i + 1) / silo_count, text=f"Optimized silo: {key}")

    my_bar.empty()

    cleaned_yield = filter_small_moves(final_y, market_data_list, min_move_thresh, total_optimizable)
    cleaned_frontier = filter_small_moves(final_f, market_data_list, min_move_thresh, total_optimizable)
    cleaned_liquid = filter_small_moves(final_l, market_data_list, min_move_thresh, total_optimizable)
    cleaned_whale = filter_small_moves(final_w, market_data_list, min_move_thresh, total_optimizable)

    dummy_opt = RebalanceOptimizer(total_optimizable, market_data_list, max_dominance, max_port_alloc)
    dummy_opt.yield_trace = global_yield_trace
    dummy_opt.frontier_trace = global_frontier_trace
    dummy_opt.liquid_trace = global_liquid_trace
    dummy_opt.whale_trace = global_whale_trace
    dummy_opt.all_attempts = global_attempts
    dummy_opt.capacity_warning = (
        " | ".join(global_capacity_warnings) if global_capacity_warnings else None
    )

    st.session_state["opt_results"] = {
        "market_data_list": market_data_list,
        "opt_object": dummy_opt,
        "best_yield_alloc": cleaned_yield,
        "frontier_alloc": cleaned_frontier,
        "liquid_alloc": cleaned_liquid,
        "whale_alloc": cleaned_whale,
        "current_metrics": {
            "annual_interest": current_annual_interest,
            "blended_apy": current_blended_apy,
        },
        "traces": {
            "yield": global_yield_trace,
            "frontier": global_frontier_trace,
            "liquid": global_liquid_trace,
            "whale": global_whale_trace,
        },
        "attempts": global_attempts,
    }

# =============================================================================
# 4. Results
# =============================================================================
if "opt_results" in st.session_state and st.session_state["opt_results"].get("best_yield_alloc") is not None:
    st.subheader("4. Results")
    render_simulation_banner()

    res_data = st.session_state["opt_results"]
    opt = res_data["opt_object"]
    market_data_list = res_data["market_data_list"]
    best_yield_alloc = res_data["best_yield_alloc"]
    frontier_alloc = res_data["frontier_alloc"]
    liquid_alloc = res_data["liquid_alloc"]
    whale_alloc = res_data["whale_alloc"]

    def get_stats(alloc):
        y = sum(val * opt.simulate_apy(market_data_list[i], val) for i, val in enumerate(alloc))
        weights = alloc / total_optimizable if total_optimizable > 0 else np.zeros_like(alloc)
        div = 1.0 - np.sum(weights**2)
        return y, y / total_optimizable if total_optimizable > 0 else 0, div

    y_abs, y_apy, y_div = get_stats(best_yield_alloc)
    w_abs, w_apy, w_div = get_stats(whale_alloc)
    f_abs, f_apy, f_div = get_stats(frontier_alloc)
    l_abs, l_apy, l_div = get_stats(liquid_alloc)

    current_blended = res_data["current_metrics"]["blended_apy"]

    st.markdown("#### Strategy comparison")
    s_col0, s_col1, s_col2, s_col3, s_col4 = st.columns(5)
    with s_col0:
        st.markdown("<h4 style='color:#9E9E9E; margin-bottom:0;'>Current</h4>", unsafe_allow_html=True)
        st.metric("APY", f"{current_blended:.2%}")
        st.caption("Baseline")
    with s_col1:
        st.markdown("<h4 style='color:#F44336; margin-bottom:0;'>Best yield</h4>", unsafe_allow_html=True)
        st.metric("APY", f"{y_apy:.2%}", delta=f"{(y_apy - current_blended):.2%}")
        st.caption(f"Diversity: **{y_div:.4f}**")
    with s_col2:
        st.markdown("<h4 style='color:#2979FF; margin-bottom:0;'>Whale shield</h4>", unsafe_allow_html=True)
        st.metric("APY", f"{w_apy:.2%}", delta=f"{(w_apy - current_blended):.2%}")
        st.caption(f"Diversity: **{w_div:.4f}**")
    with s_col3:
        st.markdown("<h4 style='color:#E040FB; margin-bottom:0;'>Frontier</h4>", unsafe_allow_html=True)
        st.metric("APY", f"{f_apy:.2%}", delta=f"{(f_apy - current_blended):.2%}")
        st.caption(f"Diversity: **{f_div:.4f}**")
    with s_col4:
        st.markdown("<h4 style='color:#00E676; margin-bottom:0;'>Liquid-yield</h4>", unsafe_allow_html=True)
        st.metric("APY", f"{l_apy:.2%}", delta=f"{(l_apy - current_blended):.2%}")
        st.caption(f"Diversity: **{l_div:.4f}**")

    st.divider()
    st.markdown("#### Optimization search space")
    col_graph1, col_graph2 = st.columns(2)

    df_scatter = pd.DataFrame(res_data["attempts"])
    highlights = pd.DataFrame(
        [
            {"Diversity Score": y_div, "Blended APY": y_apy, "Type": "Best Yield", "Size": 100},
            {"Diversity Score": w_div, "Blended APY": w_apy, "Type": "Whale Shield", "Size": 100},
            {"Diversity Score": f_div, "Blended APY": f_apy, "Type": "Frontier", "Size": 100},
            {"Diversity Score": l_div, "Blended APY": l_apy, "Type": "Liquid-Yield", "Size": 100},
        ]
    )

    with col_graph1:
        st.markdown("**Efficiency frontier**")
        render_efficiency_frontier(df_scatter, highlights)

    with col_graph2:
        st.markdown("**Solver convergence**")
        render_convergence_chart(res_data["traces"])

    st.divider()
    st.markdown("#### Allocation comparison")

    bar_data = []
    for idx, m in enumerate(market_data_list):
        short_id = m["Market ID"][0:6]
        m_name = f"{m['Loan Token']}/{m['Collateral']} ({short_id})"
        y_val = best_yield_alloc[idx]
        w_val = whale_alloc[idx]
        f_val = frontier_alloc[idx]
        l_val = liquid_alloc[idx]
        if max(y_val, w_val, f_val, l_val) > 1:
            bar_data.append({"Market": m_name, "Strategy": "Best Yield", "Alloc ($)": y_val})
            bar_data.append({"Market": m_name, "Strategy": "Whale Shield", "Alloc ($)": w_val})
            bar_data.append({"Market": m_name, "Strategy": "Frontier", "Alloc ($)": f_val})
            bar_data.append({"Market": m_name, "Strategy": "Liquid-Yield", "Alloc ($)": l_val})

    render_allocation_bars(bar_data)

    st.divider()
    st.markdown("#### Detailed results")

    strategy_choice = st.radio(
        "View details for:",
        [
            "Current portfolio",
            "Best yield",
            "Whale shield",
            "Frontier",
            "Liquid-yield",
        ],
        index=1,
        horizontal=True,
    )

    _strategy_map = {
        "Best yield": best_yield_alloc,
        "Whale shield": whale_alloc,
        "Frontier": frontier_alloc,
        "Liquid-yield": liquid_alloc,
    }
    final_alloc = _strategy_map.get(strategy_choice)
    if final_alloc is None:
        final_alloc = np.array([m["existing_balance_usd"] for m in market_data_list])
    elif strategy_choice == "Whale shield" and res_data["opt_object"].capacity_warning:
        st.warning(res_data["opt_object"].capacity_warning)

    results = []
    theoretical_annual_interest = 0.0
    realized_annual_interest = 0.0
    realized_annual_apr_interest = 0.0
    total_allocated_usd = 0.0
    total_stuck_usd = 0.0

    for i, target_val in enumerate(final_alloc):
        m = market_data_list[i]
        total_allocated_usd += target_val

        token_price = m["Price USD"]
        multiplier = 10 ** m["Decimals"]
        user_current = m["existing_balance_usd"]
        current_avail_usd = m.get("Available Liquidity (USD)", 0.0)
        current_apy_val = m.get("current_supply_apy", 0.0)

        net_move_usd = target_val - user_current
        requested_withdrawal = max(0, -net_move_usd)
        stuck_funds_usd = max(0, requested_withdrawal - current_avail_usd)
        actual_withdrawable_usd = requested_withdrawal - stuck_funds_usd
        liquid_move_usd = -actual_withdrawable_usd if net_move_usd < 0 else net_move_usd
        realized_target_val = user_current + liquid_move_usd
        total_stuck_usd += stuck_funds_usd

        user_existing_wei = (user_current / token_price) * multiplier if token_price > 0 else 0
        base_supply_wei = max(0, m["raw_supply"] - user_existing_wei)
        realized_target_wei = (realized_target_val / token_price) * multiplier if token_price > 0 else 0
        simulated_realized_supply_wei = base_supply_wei + realized_target_wei
        final_util = (
            m["raw_borrow"] / simulated_realized_supply_wei
            if simulated_realized_supply_wei > 0
            else 0
        )

        base_available_usd = max(0, current_avail_usd - user_current)
        final_available_usd = base_available_usd + realized_target_val
        liq_share = realized_target_val / final_available_usd if final_available_usd > 0 else 0

        if abs(liquid_move_usd) < 0.01:
            realized_apy = current_apy_val
        else:
            realized_apy = opt.simulate_apy(m, realized_target_val)

        realized_rate_sec = apy_to_rate_per_second(realized_apy)
        realized_apr = realized_rate_sec * SECONDS_PER_YEAR
        target_apy = current_apy_val if abs(net_move_usd) < 0.01 else opt.simulate_apy(m, target_val)

        if net_move_usd > 0.01:
            action = "🟢 DEPOSIT"
        elif net_move_usd < -0.01:
            action = "⚠️ STUCK" if actual_withdrawable_usd <= 0.01 else "🔴 WITHDRAW"
        else:
            action = "⚪ HOLD"

        current_contrib = (user_current / total_optimizable * current_apy_val) if total_optimizable > 0 else 0
        selected_contrib = (target_val / total_optimizable * target_apy) if total_optimizable > 0 else 0
        net_apy_impact = selected_contrib - current_contrib
        theoretical_annual_interest += target_val * target_apy

        realized_interest_item = realized_target_val * realized_apy
        realized_annual_interest += realized_interest_item
        realized_annual_apr_interest += realized_target_val * realized_apr

        results.append(
            {
                "Destination ID": str(m["Market ID"])[:7],
                "Market": f"{m['Loan Token']}/{m['Collateral']}",
                "Token": m["Loan Token"],
                "Chain": m["Chain"],
                "Action": format_action_label(action),
                "Weight": target_val / total_optimizable if total_optimizable > 0 else 0,
                "Net APY Impact": net_apy_impact,
                "Contribution (Target)": selected_contrib,
                "Contribution (Current)": current_contrib,
                "Current ($)": user_current,
                "Target ($)": target_val,
                "Realized ($)": realized_target_val,
                "Net Move ($)": net_move_usd,
                "Current APY": current_apy_val,
                "Simulated APY": target_apy,
                "Simulated APR": realized_apr,
                "Initial Utilization": m["raw_borrow"] / m["raw_supply"] if m["raw_supply"] > 0 else 0,
                "Final Utilization": final_util,
                "Liquid Move ($)": liquid_move_usd,
                "Stuck Funds ($)": stuck_funds_usd,
                "Ann. Yield": target_val * target_apy,
                "Ann. Yield (Realized)": realized_interest_item,
                "Initial Liq.": current_avail_usd,
                "Final Liq.": final_available_usd,
                "% Liq. Share": liq_share,
                "Market ID Full": m["Market ID"],
                "Loan Address": m.get("Loan Address"),
                "Price USD": m.get("Price USD"),
                "Decimals": m.get("Decimals"),
                "ChainID": m["ChainID"],
                "Link To Market": monarch_link(m["ChainID"], m["Market ID"]),
            }
        )

    unallocated_cash = total_optimizable - total_allocated_usd
    if unallocated_cash > 0.01:
        results.append(
            {
                "Market": "Unallocated cash",
                "Token": "CASH",
                "Chain": "Wallet",
                "Action": format_action_label("⚪ HOLD"),
                "Weight": unallocated_cash / total_optimizable if total_optimizable > 0 else 0,
                "Net APY Impact": 0.0,
                "Contribution (Target)": 0.0,
                "Contribution (Current)": 0.0,
                "Current ($)": 0.0,
                "Target ($)": unallocated_cash,
                "Realized ($)": unallocated_cash,
                "Net Move ($)": 0.0,
                "Liquid Move ($)": 0.0,
                "Current APY": 0.0,
                "Simulated APY": 0.0,
                "Simulated APR": 0.0,
                "Ann. Yield": 0.0,
                "Ann. Yield (Realized)": 0.0,
                "Stuck Funds ($)": 0.0,
                "Initial Utilization": 0.0,
                "Final Utilization": 0.0,
                "Initial Liq.": 0.0,
                "Final Liq.": 0.0,
                "% Liq. Share": 0.0,
            }
        )

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by=["Net APY Impact", "Weight"], ascending=[False, False])

    realized_blended_apy = (
        realized_annual_interest / total_optimizable if total_optimizable > 0 else 0.0
    )
    theoretical_blended_apy = (
        theoretical_annual_interest / total_optimizable if total_optimizable > 0 else 0.0
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Current APY", f"{current_blended:.4%}")
    m2.metric(
        "Realized APY",
        f"{realized_blended_apy:.4%}",
        delta=f"{(realized_blended_apy - current_blended):.4%}",
        help=(
            f"Theoretical target APY: {theoretical_blended_apy:.4%}. "
            "Realized APY accounts for liquidity-limited withdrawals."
        ),
    )
    m3.metric("Total wealth (1 yr)", f"${total_optimizable + realized_annual_interest:,.2f}")
    m4.metric(
        "Annual interest gain",
        f"${(realized_annual_interest - res_data['current_metrics']['annual_interest']):,.2f}",
        help=(
            f"Theoretical max gain: "
            f"${(theoretical_annual_interest - res_data['current_metrics']['annual_interest']):,.2f}"
        ),
    )
    selected_weights = np.array([r["Weight"] for r in results if "Weight" in r])
    selected_diversity = 1.0 - np.sum(selected_weights**2)
    m5.metric("Diversity score", f"{selected_diversity:.4f}")
    m6.metric(
        "Stuck capital",
        f"${total_stuck_usd:,.2f}",
        delta_color="inverse",
        delta="Liquidity issue" if total_stuck_usd > 0 else None,
    )

    st.markdown("---")
    st.markdown("#### Projected earnings")
    calc_basis = st.radio(
        "Rate calculation basis:",
        ["Compounded (APY)", "Linear (APR)"],
        index=1,
        horizontal=True,
        help="Linear (APR) is more conservative for short-term cashflow.",
    )
    active_interest = (
        realized_annual_apr_interest if "Linear" in calc_basis else realized_annual_interest
    )
    label_suffix = "(APR)" if "Linear" in calc_basis else "(APY)"

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric(f"Annual {label_suffix}", f"${active_interest:,.2f}")
    t2.metric(f"Monthly {label_suffix}", f"${active_interest / 12:,.2f}")
    t3.metric(f"Weekly {label_suffix}", f"${active_interest / 52:,.2f}")
    t4.metric(f"Daily {label_suffix}", f"${active_interest / 365:,.2f}")
    t5.metric(f"Hourly {label_suffix}", f"${active_interest / 8760:,.4f}")

    def style_impact(val):
        color = "#00E676" if val > 0.000001 else "#FF5252" if val < -0.000001 else "#9E9E9E"
        return f"color: {color}; font-weight: bold"

    with st.expander("Detailed allocation breakdown", expanded=False):
        st.dataframe(
            df_res.style.format(
                {
                    "Weight": "{:.2%}",
                    "Net APY Impact": "{:+.4%}",
                    "Contribution (Target)": "{:.4%}",
                    "Contribution (Current)": "{:.4%}",
                    "Current ($)": "${:,.2f}",
                    "Target ($)": "${:,.2f}",
                    "Realized ($)": "${:,.2f}",
                    "Net Move ($)": "${:,.2f}",
                    "Liquid Move ($)": "${:,.2f}",
                    "Stuck Funds ($)": "${:,.2f}",
                    "Current APY": "{:.4%}",
                    "Simulated APY": "{:.4%}",
                    "Simulated APR": "{:.4%}",
                    "Ann. Yield": "${:,.2f}",
                    "Ann. Yield (Realized)": "${:,.2f}",
                    "Initial Utilization": "{:.2%}",
                    "Final Utilization": "{:.2%}",
                    "Initial Liq.": "${:,.2f}",
                    "Final Liq.": "${:,.2f}",
                    "% Liq. Share": "{:.2%}",
                }
            ).map(style_impact, subset=["Net APY Impact"]),
            column_config={
                "Net APY Impact": st.column_config.NumberColumn(
                    "Net APY impact",
                    help="Change to total portfolio APY.",
                ),
                "Link To Market": st.column_config.LinkColumn("Link", display_text="Link"),
            },
            column_order=[
                "Destination ID",
                "Market",
                "Chain",
                "Action",
                "Weight",
                "Net APY Impact",
                "Contribution (Target)",
                "Contribution (Current)",
                "Current ($)",
                "Target ($)",
                "Realized ($)",
                "Net Move ($)",
                "Current APY",
                "Simulated APY",
                "Simulated APR",
                "Ann. Yield",
                "Ann. Yield (Realized)",
                "Initial Utilization",
                "Final Utilization",
                "Initial Liq.",
                "Final Liq.",
                "% Liq. Share",
                "Liquid Move ($)",
                "Stuck Funds ($)",
                "Link To Market",
            ],
            width="stretch",
            hide_index=True,
        )
