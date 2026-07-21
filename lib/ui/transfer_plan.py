"""Render manual execution instructions for portfolio rebalancing."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.transfer_plan import build_transfer_plan


def render_execution_plan(
    df_res: pd.DataFrame,
    df_all: pd.DataFrame,
    *,
    new_cash: float,
    rebalance_scope: str,
    min_move_thresh: float,
    total_stuck_usd: float,
) -> None:
    """Show withdrawal split summary and ordered rebalance / bridge / swap steps."""
    transfer_steps, summary_map = build_transfer_plan(
        df_res, df_all, new_cash, rebalance_scope, min_move_thresh
    )

    st.divider()
    st.markdown("#### Manual execution plan")
    st.caption(
        "Follow these steps to rebalance manually on Monarch, a DEX, or a bridge. "
        "Steps respect your optimization scope constraints."
    )

    st.markdown("##### Withdrawal operations split")
    st.caption("How funds leave each source market, by operation type.")

    if summary_map:
        st.dataframe(
            pd.DataFrame(list(summary_map.values()))
            .style.format(
                {
                    "Total Withdrawal": "${:,.2f}",
                    "1. Internal Rebalance": "${:,.2f}",
                    "2. Internal Swap": "${:,.2f}",
                    "3. Bridge Out": "${:,.2f}",
                    "4. Cash Out": "${:,.2f}",
                }
            )
            .map(
                lambda x: "background-color: #1b5e20; color: white" if x > 0.01 else "",
                subset=["1. Internal Rebalance"],
            )
            .map(
                lambda x: "background-color: #01579b; color: white" if x > 0.01 else "",
                subset=["2. Internal Swap"],
            )
            .map(
                lambda x: "background-color: #b71c1c; color: white" if x > 0.01 else "",
                subset=["3. Bridge Out"],
            )
            .map(
                lambda x: "background-color: #424242; color: white" if x > 0.01 else "",
                subset=["4. Cash Out"],
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No withdrawals required for this strategy.")

    stuck_df = df_res[df_res["Stuck Funds ($)"] > 0.01].copy()
    if not stuck_df.empty:
        st.warning(
            f"Execution limited by liquidity: ${total_stuck_usd:,.2f} cannot be withdrawn "
            "until more liquidity is available in these markets."
        )
        st.dataframe(
            stuck_df[["Market", "Chain", "Stuck Funds ($)"]].style.format(
                {"Stuck Funds ($)": "${:,.2f}"}
            ),
            width="stretch",
            hide_index=True,
        )

    if transfer_steps:
        st.markdown("##### Market rebalance steps")
        df_actions = pd.DataFrame(transfer_steps)

        def highlight_op(val: str) -> str:
            if "1." in val:
                return "color: #00E676; font-weight: bold;"
            if "2." in val:
                return "color: #01579b; font-weight: bold;"
            if "3." in val:
                return "color: #b71c1c; font-weight: bold;"
            return ""

        st.dataframe(
            df_actions.style.format(
                {
                    "Amount to move ($)": "${:,.2f}",
                    "Remaining Funds In Source ($)": "${:,.2f}",
                }
            )
            .map(highlight_op, subset=["Operation Type"])
            .map(
                lambda x: "color: red" if x < 0.01 else "",
                subset=["Remaining Funds In Source ($)"],
            ),
            column_order=[
                "Ordering",
                "Operation Type",
                "From",
                "From Market ID",
                "From (Chain)",
                "From (Token)",
                "To",
                "To Market ID",
                "To (Chain)",
                "To (Token)",
                "Amount to move ($)",
                "Remaining Funds In Source ($)",
            ],
            width="stretch",
            hide_index=True,
        )
    elif not stuck_df.empty:
        st.info("No executable transfer steps — resolve stuck liquidity first.")
    else:
        st.success("Portfolio is already aligned with this strategy.")
