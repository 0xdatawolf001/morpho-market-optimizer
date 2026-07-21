"""Morpho GraphQL fetch layer with Streamlit caching."""

from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

from lib.config import (
    BATCH_SIZE,
    CACHE_TTL,
    CHAIN_ID_TO_NAME,
    MORPHO_API_FALLBACK_URL,
    MORPHO_API_URL,
    TARGET_CHAINS,
    WAD,
    safe_float,
)
from lib.math_utils import apy_to_rate_per_second, compute_curve_multiplier

MARKET_INDEX_FIELDS = """
  marketId
  listed
  lltv
  irmAddress
  oracle { address }
  loanAsset {
    address
    symbol
    decimals
    price { usd }
    chain { id }
  }
  collateralAsset {
    address
    symbol
    decimals
  }
  supplyingVaults { address }
  warnings { type level }
  realizedBadDebt { underlying usd }
  badDebt { underlying usd }
  state {
    supplyApy
    borrowApy
    utilization
    supplyAssets
    borrowAssets
    supplyAssetsUsd
    borrowAssetsUsd
    liquidityAssetsUsd
    collateralAssetsUsd
    fee
    timestamp
    apyAtTarget
    rateAtTarget
  }
"""

MARKET_DETAIL_FIELDS = """
  marketId
  listed
  lltv
  irmAddress
  oracle { address }
  loanAsset {
    address
    symbol
    decimals
    price { usd }
    chain { id }
  }
  collateralAsset {
    address
    symbol
    decimals
  }
  supplyingVaults { address }
  warnings { type level }
  realizedBadDebt { underlying usd }
  badDebt { underlying usd }
  state {
    supplyApy
    borrowApy
    utilization
    supplyAssets
    borrowAssets
    supplyAssetsUsd
    borrowAssetsUsd
    liquidityAssetsUsd
    collateralAssetsUsd
    fee
    timestamp
    apyAtTarget
    rateAtTarget
    dailySupplyApy
    dailyBorrowApy
    weeklySupplyApy
    weeklyBorrowApy
    monthlySupplyApy
    monthlyBorrowApy
  }
"""


def _parse_gql_response(resp: requests.Response) -> dict:
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Morpho API returned non-JSON response (HTTP {resp.status_code})") from exc

    if resp.status_code >= 400 or payload.get("errors"):
        err = (payload.get("errors") or [{}])[0]
        message = err.get("message", f"HTTP {resp.status_code}")
        extensions = err.get("extensions") or payload.get("extensions") or {}
        complexity = extensions.get("complexity")
        maximum = extensions.get("maximumComplexity")
        if complexity and maximum:
            message = f"{message} (complexity {complexity:,} > max {maximum:,})"
        raise RuntimeError(f"Morpho API error: {message}")

    return payload.get("data") or {}


def _gql(query: str, variables: dict | None = None) -> dict:
    body = {"query": query, "variables": variables or {}}
    cached = st.session_state.get("morpho_api_base")
    urls: list[str] = []
    if cached:
        urls.append(cached)
    for url in (MORPHO_API_URL, MORPHO_API_FALLBACK_URL):
        if url not in urls:
            urls.append(url)

    last_error: Exception | None = None
    for url in urls:
        try:
            resp = requests.post(url, json=body, timeout=60)
            data = _parse_gql_response(resp)
            st.session_state.morpho_api_base = url
            return data
        except (RuntimeError, requests.RequestException) as exc:
            last_error = exc
            if url == urls[-1]:
                if isinstance(exc, requests.RequestException):
                    raise RuntimeError(f"Morpho API request failed: {exc}") from exc
                raise

    raise RuntimeError(f"Morpho API request failed: {last_error}")


def _price_usd_for_loan(loan: dict) -> float:
    price_obj = loan.get("price") or {}
    price_usd = safe_float(price_obj.get("usd"), default=None)
    if price_usd is None or price_usd <= 0:
        symbol = str(loan.get("symbol", "")).upper()
        if any(s in symbol for s in ["USD", "DAI", "PYUSD", "USDS", "USDT"]):
            return 1.0
        return 0.0
    return price_usd


def _process_market_row(m: dict) -> dict | None:
    loan = m.get("loanAsset") or {}
    state = m.get("state") or {}
    collateral = m.get("collateralAsset") or {}
    oracle = m.get("oracle") or {}
    warnings = m.get("warnings") or []
    bad_debt = m.get("badDebt") or {}
    realized_bad = m.get("realizedBadDebt") or {}
    vaults = m.get("supplyingVaults") or []

    loan_address = loan.get("address")
    decimals = loan.get("decimals")
    if not loan_address or decimals is None:
        return None

    price_usd = _price_usd_for_loan(loan)
    chain_id = (loan.get("chain") or {}).get("id")
    supply_usd = safe_float(state.get("supplyAssetsUsd"))
    borrow_usd = safe_float(state.get("borrowAssetsUsd"))
    liquidity_usd = safe_float(state.get("liquidityAssetsUsd"))
    if liquidity_usd <= 0:
        liquidity_usd = max(0.0, supply_usd - borrow_usd)

    supply_assets = safe_float(state.get("supplyAssets"))
    borrow_assets = safe_float(state.get("borrowAssets"))
    util = safe_float(state.get("utilization"))
    if util <= 0 and supply_assets > 0:
        util = borrow_assets / supply_assets

    lltv_raw = safe_float(m.get("lltv"))
    lltv_pct = (lltv_raw / 1e18 * 100) if lltv_raw > 1 else lltv_raw

    warning_types = [w.get("type", "") for w in warnings if w.get("type")]
    indicators = []
    if m.get("listed"):
        indicators.append("Whitelisted")
    if warning_types:
        indicators.append("Warning")
    if liquidity_usd >= 1_000_000:
        indicators.append("Deep Liq")

    return {
        "Market ID": m["marketId"],
        "Chain": CHAIN_ID_TO_NAME.get(chain_id, "Other"),
        "ChainID": chain_id,
        "Loan Token": loan.get("symbol"),
        "Loan Address": loan_address,
        "Collateral": collateral.get("symbol") or "Unknown",
        "Collateral Address": collateral.get("address"),
        "Decimals": int(decimals),
        "Price USD": float(price_usd),
        "Supply APY": safe_float(state.get("supplyApy")),
        "Borrow APY": safe_float(state.get("borrowApy")),
        "Utilization": util,
        "Total Supply (USD)": supply_usd,
        "Total Borrow (USD)": borrow_usd,
        "Available Liquidity (USD)": liquidity_usd,
        "Collateral (USD)": safe_float(state.get("collateralAssetsUsd")),
        "Whitelisted": bool(m.get("listed", False)),
        "LLTV": lltv_pct,
        "IRM Address": m.get("irmAddress"),
        "Oracle Address": oracle.get("address"),
        "Fee": safe_float(state.get("fee")) / 1e18 if safe_float(state.get("fee")) > 1 else safe_float(state.get("fee")),
        "APY at Target": safe_float(state.get("apyAtTarget")),
        "Daily Supply APY": safe_float(state.get("dailySupplyApy")),
        "Daily Borrow APY": safe_float(state.get("dailyBorrowApy")),
        "Weekly Supply APY": safe_float(state.get("weeklySupplyApy")),
        "Weekly Borrow APY": safe_float(state.get("weeklyBorrowApy")),
        "Monthly Supply APY": safe_float(state.get("monthlySupplyApy")),
        "Monthly Borrow APY": safe_float(state.get("monthlyBorrowApy")),
        "Bad Debt (USD)": safe_float(bad_debt.get("usd")),
        "Realized Bad Debt (USD)": safe_float(realized_bad.get("usd")),
        "Warnings": warning_types,
        "Warning Levels": [w.get("level", "") for w in warnings],
        "Supplying Vaults": [v.get("address") for v in vaults if v.get("address")],
        "State Timestamp": state.get("timestamp"),
        "Indicators": ", ".join(indicators) if indicators else "—",
        "Market Label": f"{loan.get('symbol')}/{collateral.get('symbol') or 'Unknown'}",
        "Short ID": str(m["marketId"])[:10],
    }


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_market_dictionary(_version: int = 1) -> pd.DataFrame:
    query = f"""
    query GetAllMarkets($first: Int, $skip: Int, $where: MarketFilters) {{
      markets(first: $first, skip: $skip, where: $where) {{
        items {{ {MARKET_INDEX_FIELDS} }}
      }}
    }}
    """
    all_items = []
    skip = 0
    load_status = st.empty()
    while True:
        load_status.info(f"Loading Morpho markets… {len(all_items):,} found")
        variables = {
            "first": BATCH_SIZE,
            "skip": skip,
            "where": {"chainId_in": TARGET_CHAINS},
        }
        data = _gql(query, variables)
        items = data.get("markets", {}).get("items", [])
        if not items:
            break
        all_items.extend(items)
        if len(items) < BATCH_SIZE:
            break
        skip += BATCH_SIZE
    load_status.empty()

    processed = []
    for m in all_items:
        row = _process_market_row(m)
        if row:
            processed.append(row)

    if not processed:
        return pd.DataFrame()

    return (
        pd.DataFrame(processed)
        .drop_duplicates(subset=["Market ID"], keep="first")
        .sort_values("Total Supply (USD)", ascending=False)
        .reset_index(drop=True)
    )


def refresh_market_index():
    get_market_dictionary.clear()
    st.session_state.pop("morpho_api_base", None)
    return get_market_dictionary()


def fetch_live_market_details(selected_df: pd.DataFrame) -> list[dict]:
    market_ids = selected_df["Market ID"].tolist()
    if not market_ids:
        return []

    query = """
    query GetMarketData($where: MarketFilters, $first: Int!) {
        markets(first: $first, where: $where) {
            items {
                marketId
                state { supplyAssets borrowAssets fee borrowApy supplyApy }
            }
        }
    }
    """
    details = []
    try:
        variables = {"where": {"uniqueKey_in": market_ids}, "first": len(market_ids) + 10}
        data = _gql(query, variables)
        state_lookup = {}
        for item in data.get("markets", {}).get("items", []):
            mid = item.get("marketId")
            if mid:
                state_lookup[mid] = item.get("state", {})

        for _, row in selected_df.iterrows():
            m_id = row["Market ID"]
            state = state_lookup.get(m_id)
            if state:
                sup, bor = float(state["supplyAssets"]), float(state["borrowAssets"])
                util = bor / sup if sup > 0 else 0
                curr_rate_sec = apy_to_rate_per_second(float(state["borrowApy"]))
                mult = compute_curve_multiplier(util)
                details.append(
                    {
                        **row.to_dict(),
                        "raw_supply": sup,
                        "raw_borrow": bor,
                        "fee": float(state["fee"]) / WAD,
                        "rate_at_target": (curr_rate_sec / mult if mult > 0 else 0),
                        "current_supply_apy": float(state["supplyApy"]),
                    }
                )
    except Exception as exc:
        st.error(f"Error fetching market details: {exc}")
    return details


def fetch_user_positions(user_address: str, price_lookup: dict | None = None) -> dict:
    user_address = user_address.lower()
    query = """
    query GetUserPositions($user: [String!]) {
      marketPositions(
        first: 1000,
        where: {
          userAddress_in: $user,
          chainId_in: [1, 8453, 42161, 999, 10, 130, 137, 480, 143]
        }
      ) {
        items {
          market { marketId loanAsset { symbol decimals } }
          state { supplyAssets }
        }
      }
    }
    """
    positions = {}
    price_lookup = price_lookup or {}
    try:
        data = _gql(query, {"user": [user_address]})
        items = data.get("marketPositions", {}).get("items", [])
        for item in items:
            market = item.get("market") or {}
            m_id = market.get("marketId")
            if not m_id:
                continue
            state = item.get("state") or {}
            loan_asset = market.get("loanAsset") or {}
            symbol = str(loan_asset.get("symbol", "")).upper()
            decimals = int(loan_asset.get("decimals", 18) or 18)
            raw_supply = float(state.get("supplyAssets") or 0)
            if raw_supply <= 0:
                continue
            token_amount = raw_supply / (10**decimals)
            price_usd = price_lookup.get(symbol)
            if price_usd is not None and price_usd > 0:
                bal = token_amount * price_usd
            elif any(s in symbol for s in ["USD", "DAI", "PYUSD", "USDS", "USDT"]):
                bal = token_amount * 1.0
            else:
                continue
            if bal > 0.01:
                positions[m_id] = bal
    except Exception as exc:
        st.error(f"Error parsing user positions: {exc}")
    return positions


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_market_detail(chain_id: int, market_id: str) -> dict | None:
    query = f"""
    query GetMarketDetail($uniqueKey: String!, $chainId: Int!) {{
      marketByUniqueKey: marketById(marketId: $uniqueKey, chainId: $chainId) {{
        {MARKET_DETAIL_FIELDS}
      }}
    }}
    """
    try:
        data = _gql(query, {"uniqueKey": market_id, "chainId": int(chain_id)})
        raw = data.get("marketByUniqueKey")
        if not raw:
            return None
        row = _process_market_row(raw)
        if row:
            row["_raw"] = raw
        return row
    except Exception:
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_market_historical(chain_id: int, market_id: str, days: int = 30) -> dict:
    query = """
    query getMarketHistoricalData($uniqueKey: String!, $options: TimeseriesOptions!, $chainId: Int!) {
      marketByUniqueKey: marketById(marketId: $uniqueKey, chainId: $chainId) {
        historicalState {
          supplyApy(options: $options) { x y }
          borrowApy(options: $options) { x y }
          utilization(options: $options) { x y }
          liquidityAssetsUsd(options: $options) { x y }
        }
      }
    }
    """
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - days * 86400
    variables = {
        "uniqueKey": market_id,
        "chainId": int(chain_id),
        "options": {"startTimestamp": start, "endTimestamp": end, "interval": "DAY"},
    }
    try:
        data = _gql(query, variables)
        return data.get("marketByUniqueKey", {}).get("historicalState") or {}
    except Exception:
        return {}


def ensure_market_index() -> pd.DataFrame:
    if "market_dict" not in st.session_state or st.session_state.market_dict is None:
        st.session_state.market_dict = get_market_dictionary()
    return st.session_state.market_dict


def build_price_lookup(df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in df.iterrows():
        sym = str(row.get("Loan Token", "")).upper()
        price = float(row.get("Price USD", 0) or 0)
        if price > 0:
            lookup[sym] = price
    return lookup
