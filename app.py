import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
import time

# ==========================================
# 0. STREAMLIT CONFIG & CONSTANTS
# ==========================================
st.set_page_config(page_title="Morpho Market Optimizer", layout="wide")

MORPHO_API_URL = "https://api.morpho.org/graphql"
BATCH_SIZE = 1000

# Constants for Morpho Math
WAD = 1e18
TARGET_UTILIZATION = 0.9
CURVE_STEEPNESS = 4.0
SECONDS_PER_YEAR = 31536000

CHAIN_ID_TO_NAME = {
    1: "Ethereum",
    10: "Optimism",
    130: "Unichain",
    137: "Polygon",
    143: "Monad",
    8453: "Base",
    42161: "Arbitrum",
    999: "HyperEVM"
}

# Default IDs from your script for the UI input
DEFAULT_IDS = """
0xc7d717f4052ac4e5463dcc58cea0f6b05dd7d8c67e0aee68ebe30a8af09b259f, 
0xa35d91efb3e284a0ab7098e8c5a65caf58ea0451073e36a544a821fd8f350953
""".strip()

# ==========================================
# 1. MATH HELPERS (Unchanged)
# ==========================================

def apy_to_rate_per_second(apy_float: float) -> float:
    if apy_float == 0: return 0.0
    return math.log(apy_float + 1) / SECONDS_PER_YEAR

def rate_per_second_to_apy(rate: float) -> float:
    return math.exp(rate * SECONDS_PER_YEAR) - 1

def compute_curve_multiplier(utilization: float) -> float:
    if utilization <= TARGET_UTILIZATION:
        numerator = (CURVE_STEEPNESS - 1) * (utilization / TARGET_UTILIZATION) + 1
        return numerator / CURVE_STEEPNESS
    else:
        if utilization >= 1.0: return CURVE_STEEPNESS
        error_ratio = (utilization - TARGET_UTILIZATION) / (1 - TARGET_UTILIZATION)
        return (CURVE_STEEPNESS - 1) * error_ratio + 1

# ==========================================
# 2. DATA FETCHING (Cached)
# ==========================================

@st.cache_data(ttl=3600)
def get_all_morpho_markets():
    """
    Retrieves ALL Morpho markets via GraphQL. Cached by Streamlit.
    """
    # Query template
    query = """
    query GetAllMarkets($first: Int, $skip: Int) {
      markets(first: $first, skip: $skip) {
        items {
          uniqueKey
          whitelisted
          lltv
          oracleAddress
          irmAddress
          loanAsset {
            address
            symbol
            name
            decimals
            chain { id }
          }
          collateralAsset {
            address
            symbol
            name
            decimals
          }
        }
      }
    }
    """

    all_markets_raw = []
    skip = 0
    status_text = st.empty()

    while True:
        status_text.text(f"Fetching batch starting at index {skip}...")
        
        variables = {"first": BATCH_SIZE, "skip": skip}

        try:
            response = requests.post(MORPHO_API_URL, json={'query': query, 'variables': variables})
            response.raise_for_status()
            data = response.json()
            
            batch_items = data['data']['markets']['items']
            
            if not batch_items:
                break
                
            all_markets_raw.extend(batch_items)
            
            if len(batch_items) < BATCH_SIZE:
                break
                
            skip += BATCH_SIZE
            time.sleep(0.1) 

        except Exception as e:
            st.error(f"API Request failed: {e}")
            break
            
    status_text.empty() # Clear status

    # Process data
    processed_data = []
    for market in all_markets_raw:
        loan_asset = market.get('loanAsset') or {}
        collateral_asset = market.get('collateralAsset') or {}
        chain_info = loan_asset.get('chain') or {}
        
        chain_id = chain_info.get('id')
        chain_name = CHAIN_ID_TO_NAME.get(chain_id, f"Unknown ({chain_id})")

        raw_lltv = market.get('lltv')
        formatted_lltv = float(raw_lltv) / 10**18 if raw_lltv else 0.0

        entry = {
            "market_id": market.get('uniqueKey'),
            "chain_id": chain_id,
            "chain_name": chain_name,
            "lltv_display": formatted_lltv,
            "loan_token_symbol": loan_asset.get('symbol'),
            "loan_token_decimals": loan_asset.get('decimals'),
            "collateral_token_symbol": collateral_asset.get('symbol'),
        }
        processed_data.append(entry)

    return pd.DataFrame(processed_data)

def fetch_live_market_details(market_df):
    """
    Fetches live supply/borrow/apy stats for specific markets.
    Not cached heavily as we want fresh rates for optimization.
    """
    market_data = []
    
    progress_bar = st.progress(0)
    total = len(market_df)
    
    for i, (index, row) in enumerate(market_df.iterrows()):
        query = """
        query GetMarketData($uniqueKey: String!, $chainId: Int!) {
            marketByUniqueKey(uniqueKey: $uniqueKey, chainId: $chainId) {
                state { supplyAssets borrowAssets fee borrowApy supplyApy }
            }
        }
        """
        
        try:
            response = requests.post(MORPHO_API_URL, json={
                'query': query, 
                'variables': {"uniqueKey": row['market_id'], "chainId": int(row['chain_id'])}
            })
            raw = response.json()
            data = raw.get('data', {}).get('marketByUniqueKey')
            
            if data:
                state = data['state']
                supply_assets = float(state['supplyAssets'])
                borrow_assets = float(state['borrowAssets'])
                fee_percent = float(state['fee']) / WAD
                curr_borrow_apy = float(state['borrowApy'])
                curr_supply_apy = float(state['supplyApy'])
                
                # Math preparation
                current_utilization = borrow_assets / supply_assets if supply_assets > 0 else 0
                curr_rate_per_sec = apy_to_rate_per_second(curr_borrow_apy)
                curve_mult = compute_curve_multiplier(current_utilization)
                rate_at_target = (curr_rate_per_sec / curve_mult) if curve_mult != 0 else 0

                pair_name = f"{row['loan_token_symbol']} / {row['collateral_token_symbol']}"

                market_data.append({
                    'id': row['market_id'],
                    'chain': row['chain_id'],
                    'chain_name': row['chain_name'],
                    'pair_name': pair_name,
                    'decimals': row['loan_token_decimals'],
                    'raw_supply': supply_assets,
                    'raw_borrow': borrow_assets,
                    'fee': fee_percent,
                    'rate_at_target': rate_at_target,
                    'current_supply_apy': curr_supply_apy
                })
        except Exception as e:
            st.warning(f"Failed to fetch details for {row['market_id']}: {e}")

        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    return market_data

# ==========================================
# 3. OPTIMIZER CLASS
# ==========================================

class MarketOptimizer:
    def __init__(self, budget, market_data_list):
        self.budget = budget
        self.market_data = market_data_list

    def calculate_new_metrics(self, market, alloc_usd):
        alloc_wei = alloc_usd * (10 ** market['decimals'])
        new_supply_wei = market['raw_supply'] + alloc_wei
        
        if new_supply_wei == 0: return 0.0

        new_util = market['raw_borrow'] / new_supply_wei
        new_mult = compute_curve_multiplier(new_util)
        new_borrow_rate = market['rate_at_target'] * new_mult
        new_supply_rate = new_borrow_rate * new_util * (1 - market['fee'])
        return rate_per_second_to_apy(new_supply_rate)

    def objective_function(self, allocations):
        total_yield_usd = 0.0
        for i, alloc_usd in enumerate(allocations):
            market = self.market_data[i]
            new_apy = self.calculate_new_metrics(market, alloc_usd)
            total_yield_usd += (alloc_usd * new_apy)
        return -total_yield_usd

    def run_optimization(self):
        num_markets = len(self.market_data)
        if num_markets == 0:
            return None, None
            
        initial_guess = [self.budget / num_markets] * num_markets
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self.budget})
        bounds = tuple((0, self.budget) for _ in range(num_markets))

        result = minimize(
            self.objective_function,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100}
        )
        return result.x, result.success

# ==========================================
# 4. STREAMLIT UI
# ==========================================

st.title("🤖 Morpho Market Optimizer")

# Inserted text block
st.info("""
Please use [Monarch lend](https://www.monarchlend.xyz/markets) to find the markets you need. 
This page assumes that your budgets and optimizations are in USD stables. 
**DO NOT USE IT FOR NON-USD STABLES**
        
You can add in market ID across multi-chains. Chain coverage is the same as Monarch. 
You can technically do multi-assets only for USD stables (DAI, USDC, USDT etc).
This tool mostly relies on the user to add in appropriate markets, so you are whitelisting yourself
""")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
total_budget = st.sidebar.number_input("Total Budget (USD)", value=10000.0, step=1000.0)

st.sidebar.subheader("Target Market IDs")
st.sidebar.caption("Paste IDs separated by newlines or commas")
raw_ids = st.sidebar.text_area("Market IDs", value=DEFAULT_IDS, height=150)

# Process ID input
clean_ids = [x.strip() for x in raw_ids.replace(',', '\n').split('\n') if x.strip()]

# --- Main Logic ---

# 1. Load Dictionary
with st.spinner("Loading Market Dictionary..."):
    df_all_markets = get_all_morpho_markets()

# 2. Filter DataFrame
if not clean_ids:
    st.warning("Please enter at least one Market ID in the sidebar.")
    st.stop()

# Case-insensitive match
mask = df_all_markets['market_id'].str.lower().isin([x.lower() for x in clean_ids])
df_candidate = df_all_markets[mask].copy()

# Show found vs missing
found_ids = set(df_candidate['market_id'].str.lower())
missing = [x for x in clean_ids if x.lower() not in found_ids]

if missing:
    st.error(f"Could not find {len(missing)} IDs in the API data.")
    with st.expander("See missing IDs"):
        st.write(missing)

st.subheader(f"Selected Markets ({len(df_candidate)})")
st.dataframe(df_candidate[['chain_name', 'loan_token_symbol', 'collateral_token_symbol', 'lltv_display', 'market_id']], use_container_width=True)

# 3. Action Button
if st.button("🚀 Fetch Live Data & Optimize Allocation", type="primary"):
    
    if df_candidate.empty:
        st.error("No valid markets selected.")
        st.stop()

    # A. Fetch Live Data
    with st.status("Fetching live on-chain data..."):
        market_data_list = fetch_live_market_details(df_candidate)
        if not market_data_list:
            st.error("Could not fetch live data for selected markets (likely 0 liquidity).")
            st.stop()
            
    # B. Run Optimizer
    optimizer = MarketOptimizer(total_budget, market_data_list)
    final_allocations, success = optimizer.run_optimization()

    if not success:
        st.error("Optimization failed to converge.")
    else:
        # C. Format Results
        st.divider()
        st.subheader("Optimization Results")

        results_data = []
        total_projected_yield = 0.0

        for i, alloc in enumerate(final_allocations):
            market = market_data_list[i]
            new_apy = optimizer.calculate_new_metrics(market, alloc)
            old_apy = market['current_supply_apy']
            yield_usd = alloc * new_apy
            total_projected_yield += yield_usd
            
            # Formating for table
            results_data.append({
                "Chain": market['chain_name'],
                "Pair": market['pair_name'],
                "Allocation ($)": alloc,
                "Allocation (%)": (alloc / total_budget),
                "Current APY": old_apy,
                "New APY (Diluted)": new_apy,
                "Proj. Yield ($/yr)": yield_usd,
                "Market ID": market['id']
            })

        df_results = pd.DataFrame(results_data)
        
        # Calculate Blended APY
        blended_apy = (total_projected_yield / total_budget) if total_budget > 0 else 0

        # Metrics Row
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Budget", f"${total_budget:,.2f}")
        col2.metric("Est. Annual Yield", f"${total_projected_yield:,.2f}")
        col3.metric("Blended APY", f"{blended_apy:.4%}")

        # Final Table
        st.dataframe(
            df_results.style.format({
                "Allocation ($)": "${:,.2f}",
                "Allocation (%)": "{:.1%}",
                "Current APY": "{:.4%}",
                "New APY (Diluted)": "{:.4%}",
                "Proj. Yield ($/yr)": "${:,.2f}"
            }),
            use_container_width=True,
            height=400
        )


