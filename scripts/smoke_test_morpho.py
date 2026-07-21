#!/usr/bin/env python3
"""Smoke test Morpho API integration without launching Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Minimal streamlit session shim for cached functions
import streamlit as st

if "morpho_api_base" not in st.session_state:
    st.session_state.morpho_api_base = None


def main() -> int:
    from lib.morpho_api import (
        fetch_market_detail,
        fetch_market_historical,
        get_market_dictionary,
    )

    print("1. Fetching market index…")
    get_market_dictionary.clear()
    st.session_state.pop("morpho_api_base", None)
    df = get_market_dictionary()
    if df.empty:
        print("FAIL: market index is empty")
        return 1
    print(f"   OK: {len(df):,} markets loaded via {st.session_state.get('morpho_api_base')}")

    sample = df.iloc[0]
    chain_id = int(sample["ChainID"])
    market_id = sample["Market ID"]
    print(f"2. Fetching market detail for {sample['Market Label']}…")
    detail = fetch_market_detail(chain_id, market_id)
    if not detail:
        print("FAIL: market detail returned None")
        return 1
    print(f"   OK: supply APY {detail['Supply APY']:.2%}")

    print("3. Fetching 30d historical series…")
    hist = fetch_market_historical(chain_id, market_id, days=30)
    series_count = sum(1 for v in hist.values() if v)
    print(f"   OK: {series_count} non-empty series")

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
