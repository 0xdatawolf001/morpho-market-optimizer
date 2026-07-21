"""Portfolio basket and text parsing helpers."""

import streamlit as st

from lib.config import MANUAL_SEP, WALLET_SEP
from lib.math_utils import extract_market_id_from_monarch_link


def init_session_defaults():
    defaults = {
        "balance_cache": {},
        "optimizer_basket": [],
        "portfolio_input_text": "",
        "filter_prefs": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def init_session_state():
    """Alias used by Streamlit pages."""
    init_session_defaults()


def get_basket() -> list[str]:
    """Return optimizer basket keys from session state."""
    init_session_defaults()
    return list(st.session_state.optimizer_basket)


def basket_key(market_id: str, chain_id: int | None = None) -> str:
    if chain_id is not None:
        return f"{int(chain_id)}:{market_id.lower()}"
    return market_id.lower()


def add_to_basket(market_id: str, chain_id: int | None = None) -> bool:
    init_session_defaults()
    key = basket_key(market_id, chain_id)
    if key not in st.session_state.optimizer_basket:
        st.session_state.optimizer_basket.append(key)
        return True
    return False


def remove_from_basket(key: str):
    init_session_defaults()
    st.session_state.optimizer_basket = [k for k in st.session_state.optimizer_basket if k != key]


def clear_basket():
    st.session_state.optimizer_basket = []


def _strip_market_from_text(raw_text: str, market_id: str) -> str:
    """Remove one market ID line from portfolio paste text (manual + wallet sections)."""
    mid = market_id.lower()
    kept = []
    for line in raw_text.split("\n"):
        if line.strip() in (MANUAL_SEP, WALLET_SEP):
            kept.append(line)
            continue
        clean = line.split("--")[0].strip()
        if not clean:
            continue
        extracted = extract_market_id_from_monarch_link(clean).lower()
        if extracted == mid:
            continue
        kept.append(line)

    # Drop dangling section headers when a section becomes empty.
    out: list[str] = []
    i = 0
    while i < len(kept):
        line = kept[i]
        if line.strip() in (MANUAL_SEP, WALLET_SEP):
            j = i + 1
            while j < len(kept) and kept[j].strip() not in (MANUAL_SEP, WALLET_SEP):
                if kept[j].strip():
                    break
                j += 1
            if j >= len(kept) or kept[j].strip() in (MANUAL_SEP, WALLET_SEP):
                i += 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out).strip()


def remove_market_from_optimizer(market_id: str, chain_id: int | None = None):
    """Remove a market from basket, portfolio text, and balance cache."""
    init_session_defaults()
    remove_from_basket(basket_key(market_id, chain_id))
    st.session_state.portfolio_input_text = _strip_market_from_text(
        st.session_state.get("portfolio_input_text", ""),
        market_id,
    )
    _clear_balance_cache_for_market(market_id)


def parse_basket_keys() -> list[tuple[int | None, str]]:
    init_session_defaults()
    parsed = []
    for key in st.session_state.optimizer_basket:
        if ":" in key:
            chain_str, mid = key.split(":", 1)
            try:
                parsed.append((int(chain_str), mid))
            except ValueError:
                parsed.append((None, key))
        else:
            parsed.append((None, key))
    return parsed


def basket_market_ids(df_all) -> list[str]:
    ids = []
    for chain_id, mid in parse_basket_keys():
        if chain_id is not None:
            match = df_all[
                (df_all["Market ID"].str.lower() == mid.lower())
                & (df_all["ChainID"] == chain_id)
            ]
        else:
            match = df_all[df_all["Market ID"].str.lower() == mid.lower()]
        if not match.empty:
            ids.append(match.iloc[0]["Market ID"])
        else:
            ids.append(mid)
    return list(dict.fromkeys(ids))


def parse_market_ids_from_text(raw_text: str) -> list[str]:
    clean_ids = []
    for line in raw_text.replace(",", "\n").split("\n"):
        if WALLET_SEP in line or MANUAL_SEP in line:
            continue
        line_clean = line.split("--")[0].strip()
        if not line_clean:
            continue
        extracted = extract_market_id_from_monarch_link(line_clean)
        if extracted and extracted not in clean_ids:
            clean_ids.append(extracted)
    return clean_ids


def parse_manual_market_ids(raw_text: str) -> list[str]:
    """Market IDs from the manual / basket section only (before wallet import block)."""
    manual = raw_text.split(WALLET_SEP)[0] if WALLET_SEP in raw_text else raw_text
    return parse_market_ids_from_text(manual)


def _clear_balance_cache_for_market(market_id: str):
    for key in list(st.session_state.balance_cache.keys()):
        if key.lower() == market_id.lower():
            del st.session_state.balance_cache[key]


def sync_basket_from_manual_text(df_all):
    """Keep optimizer basket aligned with the manual paste section."""
    init_session_defaults()
    raw = st.session_state.get("portfolio_input_text", "")
    manual_ids = parse_manual_market_ids(raw)
    new_keys: list[str] = []
    for mid in manual_ids:
        row = df_all[df_all["Market ID"].str.lower() == mid.lower()]
        if not row.empty:
            r = row.iloc[0]
            new_keys.append(basket_key(r["Market ID"], int(r["ChainID"])))
        else:
            new_keys.append(basket_key(mid))

    old_keys = set(st.session_state.optimizer_basket)
    new_key_set = set(new_keys)
    for key in old_keys - new_key_set:
        mid = key.split(":", 1)[1] if ":" in key else key
        _clear_balance_cache_for_market(mid)

    st.session_state.optimizer_basket = new_keys


def ensure_portfolio_text_has_basket(df_all):
    """Add basket markets to paste text without overwriting user removals."""
    if not get_basket():
        return
    manual_ids = {m.lower() for m in parse_manual_market_ids(st.session_state.get("portfolio_input_text", ""))}
    basket_ids = {m.lower() for m in basket_market_ids(df_all)}
    if basket_ids <= manual_ids:
        return

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


def sync_portfolio_text_from_basket(df_all):
    ids = basket_market_ids(df_all)
    if not ids:
        return
    lines = [MANUAL_SEP]
    for mid in ids:
        row = df_all[df_all["Market ID"].str.lower() == mid.lower()]
        label = ""
        if not row.empty:
            label = f" -- {row.iloc[0]['Loan Token']}/{row.iloc[0]['Collateral']}"
        lines.append(f"{mid}{label}")
    st.session_state.portfolio_input_text = "\n".join(lines)


def handle_text_change(df_all):
    raw_text = st.session_state.get("portfolio_input_text", "")
    if not raw_text or not raw_text.strip():
        st.session_state.optimizer_basket = []
        return

    if WALLET_SEP in raw_text:
        parts = raw_text.split(WALLET_SEP)
        manual_part = parts[0]
        wallet_part = parts[1]
    else:
        manual_part = raw_text
        wallet_part = ""

    def format_line(part_text):
        lines = []
        for line in part_text.split("\n"):
            clean = line.split("--")[0].strip()
            mid = extract_market_id_from_monarch_link(clean).lower()
            if mid.startswith("0x"):
                label = ""
                row = df_all[df_all["Market ID"].str.lower() == mid]
                if not row.empty:
                    label = f" -- {row.iloc[0]['Loan Token']}/{row.iloc[0]['Collateral']}"
                lines.append(f"{mid}{label}")
        return lines

    manual_lines = format_line(manual_part)
    wallet_lines = format_line(wallet_part)

    new_parts = []
    if manual_lines:
        new_parts.append(MANUAL_SEP)
        new_parts.extend(manual_lines)
        new_parts.append("")
    if wallet_lines:
        new_parts.append(WALLET_SEP)
        new_parts.extend(wallet_lines)

    st.session_state.portfolio_input_text = "\n".join(new_parts).strip()
    sync_basket_from_manual_text(df_all)
