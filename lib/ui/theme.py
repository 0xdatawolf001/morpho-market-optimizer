"""Monarch-inspired Streamlit theme and layout helpers."""

import streamlit as st

from lib.config import STRATEGY_COLORS


def inject_theme():
    colors = ", ".join(f'"{k}": "{v}"' for k, v in STRATEGY_COLORS.items())
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }}
        [data-testid="stMetric"] {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.65rem 0.85rem;
        }}
        [data-testid="stMetric"] label {{
            font-size: 0.78rem;
            opacity: 0.75;
        }}
        div[data-testid="stDataFrame"] {{
            font-size: 0.85rem;
        }}
        .morpho-card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
        }}
        .morpho-card h3 {{
            margin: 0 0 0.35rem 0;
            font-size: 0.95rem;
        }}
        .morpho-card p {{
            margin: 0;
            opacity: 0.8;
            font-size: 0.82rem;
        }}
        .morpho-empty {{
            text-align: center;
            padding: 2.5rem 1rem;
            border: 1px dashed rgba(255,255,255,0.15);
            border-radius: 12px;
            opacity: 0.85;
        }}
        .morpho-disclaimer {{
            background: rgba(255, 193, 7, 0.12);
            border-left: 4px solid #ffc107;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin: 0.75rem 0 1rem 0;
            font-size: 0.9rem;
        }}
        .morpho-chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-bottom: 0.5rem;
        }}
        .morpho-summary-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.65rem;
            margin-bottom: 1rem;
        }}
        @media (max-width: 900px) {{
            .morpho-summary-row {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_summary_cards(metrics: list[tuple[str, str, str | None]]):
    cols = st.columns(len(metrics))
    for col, (label, value, help_text) in zip(cols, metrics):
        with col:
            st.metric(label, value, help=help_text)


def render_empty_state(title: str, body: str):
    st.markdown(
        f"""
        <div class="morpho-empty">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer(text: str = "Simulation only — not a transaction plan."):
    st.markdown(f'<div class="morpho-disclaimer">{text}</div>', unsafe_allow_html=True)


def render_page_header(title: str, caption: str | None = None):
    st.title(title)
    if caption:
        st.caption(caption)


def render_simulation_banner():
    render_disclaimer(
        "Simulation only — results show modeled allocations and yields, not executable transactions."
    )


def render_optimizer_basket(df_all, *, key_prefix: str = "opt"):
    """Editable optimizer basket with per-market remove actions."""
    from lib.portfolio import get_basket, remove_market_from_optimizer

    basket = get_basket()
    if not basket:
        st.info("No markets in basket. Add markets from the Markets page or Market Detail.")
        return

    st.markdown(f"**Optimizer basket ({len(basket)})**")
    for bkey in basket:
        if ":" in bkey:
            chain_id, mid = bkey.split(":", 1)
            row = df_all[
                (df_all["Market ID"].str.lower() == mid.lower())
                & (df_all["ChainID"] == int(chain_id))
            ]
            chain_id_int = int(chain_id)
        else:
            mid = bkey
            row = df_all[df_all["Market ID"].str.lower() == mid.lower()]
            chain_id_int = int(row.iloc[0]["ChainID"]) if not row.empty else None

        if row.empty:
            label = f"{mid[:10]}…"
            sub = "Unknown market"
        else:
            r = row.iloc[0]
            label = f"{r['Loan Token']}/{r['Collateral']}"
            sub = f"{r['Chain']} · {r['Supply APY']:.2%} supply APY"

        c1, c2, c3 = st.columns([3, 2, 1], vertical_alignment="center")
        c1.markdown(f"**{label}**")
        c2.caption(sub)
        if c3.button("Remove", key=f"{key_prefix}_rm_{bkey}", type="secondary"):
            remove_market_from_optimizer(mid, chain_id_int)
            st.toast(f"Removed {label}")
            st.rerun()

    if st.button("Clear basket", key=f"{key_prefix}_clear_basket", type="secondary"):
        for bkey in list(basket):
            if ":" in bkey:
                chain_id, mid = bkey.split(":", 1)
                remove_market_from_optimizer(mid, int(chain_id))
            else:
                remove_market_from_optimizer(bkey)
        st.toast("Cleared optimizer basket")
        st.rerun()


def render_basket_sidebar(df_all):
    """Show optimizer basket in sidebar."""
    basket = st.session_state.get("optimizer_basket", [])
    count = len(basket)
    st.sidebar.caption(f"Optimizer basket: **{count}** market{'s' if count != 1 else ''}")
    if not basket:
        st.sidebar.info("Add markets from Discovery or Market Detail.")
        return
    for key in basket:
        if ":" in key:
            chain_id, mid = key.split(":", 1)
            row = df_all[
                (df_all["Market ID"].str.lower() == mid.lower())
                & (df_all["ChainID"] == int(chain_id))
            ]
        else:
            mid = key
            row = df_all[df_all["Market ID"].str.lower() == key.lower()]
        label = f"{mid[:8]}…" if row.empty else f"{row.iloc[0]['Loan Token']}/{row.iloc[0]['Collateral']}"
        c1, c2 = st.sidebar.columns([4, 1])
        c1.caption(label)
        if c2.button("✕", key=f"rm_{key}"):
            from lib.portfolio import remove_market_from_optimizer

            if ":" in key:
                chain_str, mid = key.split(":", 1)
                remove_market_from_optimizer(mid, int(chain_str))
            else:
                remove_market_from_optimizer(key)
            st.rerun()
