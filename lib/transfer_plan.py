"""Build step-by-step rebalance / bridge / swap instructions from optimizer output."""

from __future__ import annotations

import pandas as pd

from lib.config import scope_code


def build_transfer_plan(
    df_res: pd.DataFrame,
    df_all: pd.DataFrame,
    new_cash: float,
    rebalance_scope: str,
    min_move_thresh: float,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Build transfer steps from optimization results.

    Returns (transfer_steps, summary_map) where summary_map aggregates
    withdrawals by source market and operation type.
    """
    market_dict = df_all
    all_sources: list[dict] = []
    wallet_source: dict | None = None

    withdraw_df = df_res[df_res["Liquid Move ($)"] < -0.01]
    for _, row in withdraw_df.iterrows():
        try:
            m_row = market_dict[market_dict["Market ID"] == row["Market ID Full"]]
            if not m_row.empty:
                dec_val = int(m_row.iloc[0]["Decimals"])
                tok_addr = row.get("Loan Address") or m_row.iloc[0]["Loan Address"]
            else:
                dec_val = 18
                tok_addr = row.get("Loan Address")
        except Exception:
            dec_val = 18
            tok_addr = row.get("Loan Address")

        all_sources.append(
            {
                "id": row["Market ID Full"],
                "name": row["Market"],
                "chain": row["Chain"],
                "token": row["Token"],
                "available": abs(row["Liquid Move ($)"]),
                "running_balance": row["Current ($)"],
                "type": "Market",
                "chain_id": row.get("ChainID"),
                "token_address": tok_addr,
                "decimals": dec_val,
                "price": row.get("Price USD", 1.0),
            }
        )

    if new_cash > 0.01:
        wallet_source = {
            "id": "Wallet",
            "name": "New Capital",
            "chain": "Wallet",
            "token": "CASH",
            "available": new_cash,
            "running_balance": new_cash,
            "type": "Wallet",
            "chain_id": None,
            "token_address": None,
            "decimals": 18,
            "price": 1.0,
        }

    all_destinations: list[dict] = []
    deposit_df = df_res[df_res["Net Move ($)"] > 0.01]
    for _, row in deposit_df.iterrows():
        try:
            m_row = market_dict[market_dict["Market ID"] == row["Market ID Full"]]
            tok_addr = row.get("Loan Address") or (
                m_row.iloc[0]["Loan Address"] if not m_row.empty else None
            )
        except Exception:
            tok_addr = None

        all_destinations.append(
            {
                "id": row["Market ID Full"],
                "name": row["Market"],
                "chain": row["Chain"],
                "token": row["Token"],
                "needed": row["Net Move ($)"],
                "running_balance": row["Current ($)"],
                "chain_id": row.get("ChainID"),
                "token_address": tok_addr,
                "apy": row.get("Simulated APY", 0.0),
            }
        )

    sc = scope_code(rebalance_scope)

    def _group_key(item: dict) -> str | tuple:
        if sc == 0:
            return "GLOBAL"
        if sc == 1:
            return (item["chain"], item["token"])
        if sc == 2:
            return item["chain"]
        if sc == 3:
            return item["token"]
        return "GLOBAL"

    groups: dict = {}
    for src in all_sources:
        key = _group_key(src)
        groups.setdefault(key, {"src": [], "dst": []})["src"].append(src)
    for dst in all_destinations:
        key = _group_key(dst)
        groups.setdefault(key, {"src": [], "dst": []})["dst"].append(dst)

    transfer_steps: list[dict] = []
    ordering_counter = 1

    for _g_key, bucket in groups.items():
        g_srcs, g_dsts = bucket["src"], bucket["dst"]
        s_idx, d_idx = 0, 0
        while d_idx < len(g_dsts):
            dst = g_dsts[d_idx]
            amount_needed = dst["needed"]
            amount_from_src = 0.0
            src = None
            used_wallet = False

            if s_idx < len(g_srcs):
                src = g_srcs[s_idx]
                amount_from_src = min(src["available"], amount_needed)

            if amount_from_src < 0.01 and wallet_source and wallet_source["available"] > 0.01:
                src = wallet_source
                amount_from_src = min(wallet_source["available"], amount_needed)
                used_wallet = True

            if amount_from_src > 0.01:
                dst["running_balance"] += amount_from_src
                src["running_balance"] -= amount_from_src
                src["available"] -= amount_from_src
                dst["needed"] -= amount_from_src

                if src["type"] == "Wallet":
                    op_type, op_code = "New Deposit", 0
                elif src["chain"] != dst["chain"]:
                    op_type, op_code = "3. Bridge", 3
                elif src["token"] != dst["token"]:
                    op_type, op_code = "2. Swap", 2
                else:
                    op_type, op_code = "1. Rebalance", 1

                safe_decimals = int(src.get("decimals") or 18)
                mid = src.get("id", "")
                truncated_id = mid[:7] if mid and mid != "Wallet" else "N/A"

                transfer_steps.append(
                    {
                        "Ordering": ordering_counter,
                        "Operation Type": op_type,
                        "OpCode": op_code,
                        "From": src["name"],
                        "From ID": src.get("id"),
                        "From Market ID": truncated_id,
                        "From (Chain)": src["chain"],
                        "From (Token)": src["token"],
                        "To": dst["name"],
                        "To ID": dst.get("id"),
                        "To Market ID": dst.get("id", "N/A")[:7],
                        "To (Chain)": dst["chain"],
                        "To (Token)": dst["token"],
                        "Amount to move ($)": amount_from_src,
                        "Remaining Funds In Source ($)": src["running_balance"],
                        "src_chain_id": src.get("chain_id"),
                        "dst_chain_id": dst.get("chain_id"),
                        "src_token": src.get("token_address"),
                        "dst_token": dst.get("token_address"),
                        "decimals": safe_decimals,
                        "price": src.get("price", 1.0),
                        "dst_apy": dst.get("apy", 0.0),
                    }
                )
                ordering_counter += 1
                if not used_wallet and src["available"] < 0.01:
                    s_idx += 1
                if dst["needed"] < 0.01:
                    d_idx += 1
            else:
                d_idx += 1

    for _g_key, bucket in groups.items():
        for src in bucket["src"]:
            if src["available"] > 0.01 and src["available"] >= min_move_thresh:
                src["running_balance"] -= src["available"]
                mid = src.get("id", "")
                truncated_id = mid[:7] if mid else "N/A"
                transfer_steps.append(
                    {
                        "Ordering": ordering_counter,
                        "Operation Type": "Cash Out",
                        "OpCode": 0,
                        "From": src["name"],
                        "From Market ID": truncated_id,
                        "From (Chain)": src["chain"],
                        "From (Token)": src["token"],
                        "To": "Wallet (Unallocated)",
                        "To Market ID": "N/A",
                        "To (Chain)": "Wallet",
                        "To (Token)": "CASH",
                        "Amount to move ($)": src["available"],
                        "Remaining Funds In Source ($)": src["running_balance"],
                    }
                )
                ordering_counter += 1

    summary_map: dict[str, dict] = {}
    for step in transfer_steps:
        s_id = step.get("From ID")
        if not s_id or s_id == "Wallet":
            continue
        if s_id not in summary_map:
            summary_map[s_id] = {
                "Source Market": step["From"],
                "Market ID": s_id[:7],
                "Chain": step["From (Chain)"],
                "Asset": step["From (Token)"],
                "Total Withdrawal": 0.0,
                "1. Internal Rebalance": 0.0,
                "2. Internal Swap": 0.0,
                "3. Bridge Out": 0.0,
                "4. Cash Out": 0.0,
            }
        amt = step["Amount to move ($)"]
        summary_map[s_id]["Total Withdrawal"] += amt
        oc = step["OpCode"]
        if oc == 1:
            summary_map[s_id]["1. Internal Rebalance"] += amt
        elif oc == 2:
            summary_map[s_id]["2. Internal Swap"] += amt
        elif oc == 3:
            summary_map[s_id]["3. Bridge Out"] += amt
        elif step["Operation Type"] == "Cash Out":
            summary_map[s_id]["4. Cash Out"] += amt

    return transfer_steps, summary_map
