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
    from lib.morpho_api import get_market_dictionary

    print("1. Fetching market index…")
    get_market_dictionary.clear()
    st.session_state.pop("morpho_api_base", None)
    df = get_market_dictionary()
    if df.empty:
        print("FAIL: market index is empty")
        return 1
    print(f"   OK: {len(df):,} markets loaded via {st.session_state.get('morpho_api_base')}")

    sample = df.iloc[0]
    print(f"2. Sample market: {sample['Market Label']} on {sample['Chain']}")
    print(f"   OK: supply APY {sample['Supply APY']:.2%}")

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
