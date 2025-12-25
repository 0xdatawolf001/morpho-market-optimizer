import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
import time

# ==========================================
# 0. CONFIG & CONSTANTS
# ==========================================
st.set_page_config(page_title="Morpho Rebalance", layout="wide")

MORPHO_API_URL = "https://api.morpho.org/graphql"
BATCH_SIZE = 1000
WAD = 1e18
TARGET_UTILIZATION = 0.9
CURVE_STEEPNESS = 4.0
SECONDS_PER_YEAR = 31536000

CHAIN_ID_TO_NAME = {
    1: "Ethereum", 10: "Optimism", 130: "Unichain", 137: "Polygon",
    143: "Monad", 8453: "Base", 42161: "Arbitrum", 999: "HyperEVM",
    747474: "Katana", 988: "Stable", 98866: "Plume"
}

# ==========================================
# 1. MATH HELPERS
# ==========================================

def apy_to_rate_per_second(apy_float: float) -> float:
    if apy_float <= 0: return 0.0
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
# 2. DATA FETCHING
# ==========================================

def get_market_dictionary():
    query = """
    query GetAllMarkets($first: Int, $skip: Int) {
      markets(first: $first, skip: $skip) {
        items {
          uniqueKey
          loanAsset { symbol decimals chain { id } }
          collateralAsset { symbol }
        }
      }
    }
    """
    all_items = []
    skip = 0
    load_status = st.empty()
    
    while True:
        load_status.info(f"⏳ Indexing Morpho Markets... {len(all_items)} found.")
        resp = requests.post(MORPHO_API_URL, json={'query': query, 'variables': {"first": BATCH_SIZE, "skip": skip}})
        data = resp.json()['data']['markets']['items']
        if not data: break
        all_items.extend(data)
        if len(data) < BATCH_SIZE: break
        skip += BATCH_SIZE
        
    load_status.empty()
    
    processed = []
    for m in all_items:
        loan = m.get('loanAsset') or {}
        processed.append({
            "Market ID": m['uniqueKey'],
            "Chain": CHAIN_ID_TO_NAME.get(loan.get('chain', {}).get('id'), "Other"),
            "Loan Token": loan.get('symbol'),
            "Collateral": (m.get('collateralAsset') or {}).get('symbol'),
            "Decimals": loan.get('decimals'),
            "ChainID": loan.get('chain', {}).get('id')
        })
    return pd.DataFrame(processed)

def fetch_live_market_details(selected_df):
    details = []
    my_bar = st.progress(0, text="Fetching real-time yields...")
    for i, (_, row) in enumerate(selected_df.iterrows()):
        query = """
        query GetMarketData($uniqueKey: String!, $chainId: Int!) {
            marketByUniqueKey(uniqueKey: $uniqueKey, chainId: $chainId) {
                state { supplyAssets borrowAssets fee borrowApy supplyApy }
            }
        }
        """
        r = requests.post(MORPHO_API_URL, json={'query': query, 'variables': {"uniqueKey": row['Market ID'], "chainId": int(row['ChainID'])}}).json().get('data', {}).get('marketByUniqueKey')
        if r:
            state = r['state']
            sup, bor = float(state['supplyAssets']), float(state['borrowAssets'])
            util = bor / sup if sup > 0 else 0
            curr_rate_sec = apy_to_rate_per_second(float(state['borrowApy']))
            mult = compute_curve_multiplier(util)
            details.append({
                **row.to_dict(),
                'raw_supply': sup, 'raw_borrow': bor,
                'fee': float(state['fee']) / WAD,
                'rate_at_target': (curr_rate_sec / mult if mult > 0 else 0),
                'current_supply_apy': float(state['supplyApy'])
            })
        my_bar.progress((i + 1) / len(selected_df))
    my_bar.empty()
    return details

# ==========================================
# 3. OPTIMIZER
# ==========================================

class RebalanceOptimizer:
    def __init__(self, total_budget, market_list):
        self.total_budget = total_budget 
        self.markets = market_list

    def simulate_apy(self, market, user_new_alloc_usd):
        user_existing_wei = market['existing_balance_usd'] * (10**market['Decimals'])
        other_users_supply_wei = max(0, market['raw_supply'] - user_existing_wei)
        simulated_total_supply_wei = other_users_supply_wei + (user_new_alloc_usd * (10**market['Decimals']))
        
        if simulated_total_supply_wei <= 0: return 0.0
        new_util = market['raw_borrow'] / simulated_total_supply_wei
        new_mult = compute_curve_multiplier(new_util)
        new_borrow_rate = market['rate_at_target'] * new_mult
        new_supply_rate = new_borrow_rate * new_util * (1 - market['fee'])
        return rate_per_second_to_apy(new_supply_rate)

    def objective(self, x):
        total_yield = 0
        for i, alloc in enumerate(x):
            total_yield += (alloc * self.simulate_apy(self.markets[i], alloc))
        return -total_yield

    def optimize(self):
        n = len(self.markets)
        res = minimize(self.objective, [self.total_budget/n]*n, method='SLSQP', 
                       bounds=[(0, self.total_budget)]*n, 
                       constraints=({'type': 'eq', 'fun': lambda x: np.sum(x)-self.total_budget}))
        return res.x if res.success else None

# ==========================================
# 4. STREAMLIT UI
# ==========================================
if "balance_cache" not in st.session_state:
    st.session_state.balance_cache = {}

st.title("⚖️ Morpho Portfolio Optimizer")

# --- DATA INDEXING ---
if "market_dict" not in st.session_state:
    st.session_state.market_dict = get_market_dictionary()

df_all = st.session_state.market_dict

st.subheader("1. Market Discovery")
st.caption("Use the filters below to narrow down markets, then copy Market IDs for optimization.")
st.textg("The tool assumes that you have selected a candidate list of markets. The tool assumes USD stables = 1 USD and only works for USD stables. Do not use for any other loan token. The tool also assumes that you have sensibly chosen the pools you are comfortable to invest in")

# ==========================================
# ADD MULTI-SELECT FILTERS HERE
# ==========================================

# Prepare token display names with truncated addresses
def create_token_display(df, token_col, address_col='Market ID'):
    """Create display names with truncated addresses for duplicate token names"""
    token_data = []
    for _, row in df.iterrows():
        token_name = row[token_col]
        market_id = row[address_col]
        # Truncate address to first 6 chars (0x + 4 chars)
        truncated = market_id[:6] if pd.notna(market_id) else ""
        
        # Handle null/blank token names
        if pd.isna(token_name) or str(token_name).strip() == "":
            display_name = f"({truncated})"
            sort_key = f"zzz_{truncated}"  # Sort unnamed tokens last
        else:
            display_name = f"{token_name} ({truncated})"
            sort_key = token_name.lower()
        
        token_data.append({
            'display_name': display_name,
            'token_name': token_name,
            'market_id': market_id,
            'sort_key': sort_key
        })
    return token_data

# Create lookup dictionaries for loan and collateral tokens
loan_token_data = create_token_display(df_all, 'Loan Token')
collateral_token_data = create_token_display(df_all, 'Collateral')

# Get unique display names (sorted)
unique_loan_displays = sorted(list(set([x['display_name'] for x in loan_token_data])), 
                               key=lambda x: next(d['sort_key'] for d in loan_token_data if d['display_name'] == x))
unique_collateral_displays = sorted(list(set([x['display_name'] for x in collateral_token_data])), 
                                     key=lambda x: next(d['sort_key'] for d in collateral_token_data if d['display_name'] == x))
unique_chains = sorted(df_all['Chain'].dropna().unique().tolist())

# Display filters in three columns
filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    selected_chains = st.multiselect(
        "Filter by Chain",
        options=unique_chains,
        default=[],
        placeholder="Select chains..."
    )

with filter_col2:
    selected_loan_tokens = st.multiselect(
        "Filter by Loan Token",
        options=unique_loan_displays,
        default=[],
        placeholder="Select loan tokens..."
    )

with filter_col3:
    selected_collateral_tokens = st.multiselect(
        "Filter by Collateral Token",
        options=unique_collateral_displays,
        default=[],
        placeholder="Select collateral tokens..."
    )

# Apply filters to dataframe
df_filtered = df_all.copy()

# Filter by chain
if selected_chains:
    df_filtered = df_filtered[df_filtered['Chain'].isin(selected_chains)]

# Filter by loan token (match against display names)
if selected_loan_tokens:
    # Extract market IDs from selected display names
    selected_loan_market_ids = [d['market_id'] for d in loan_token_data 
                                if d['display_name'] in selected_loan_tokens]
    df_filtered = df_filtered[df_filtered['Market ID'].isin(selected_loan_market_ids)]

# Filter by collateral token (match against display names)
if selected_collateral_tokens:
    # Extract market IDs from selected display names
    selected_collateral_market_ids = [d['market_id'] for d in collateral_token_data 
                                      if d['display_name'] in selected_collateral_tokens]
    df_filtered = df_filtered[df_filtered['Market ID'].isin(selected_collateral_market_ids)]

# Display filtered results count
st.caption(f"📊 Showing {len(df_filtered)} of {len(df_all)} markets")

# Display the filtered dataframe
st.dataframe(
    df_filtered,
    use_container_width=True,
    hide_index=True,
    column_order=["Market ID", "Chain", "Loan Token", "Collateral"],
)


# ==========================================
# END OF FILTER SECTION
# ==========================================

st.divider()

# --- UI SECTION 2: SCOPE & REBALANCE ---
col_scope, col_cash = st.columns([2, 1])

with col_scope:
    st.subheader("2. Your Portfolio Scope")
    # Increased height from 100 to 300 to fit ~10+ lines
    raw_ids = st.text_area(
        "Paste Market IDs to optimize (one per line)", 
        value="", 
        height=300, 
        placeholder="""0xfea758e88403739fee1113b26623f43d3c37b51dc1e1e8231b78b23d1404e439 (Market ID)
0xf8c13c80ab8666c21fc5afa13105745cae7c1da13df596eb5054319f36655cc9"""
    )
    clean_ids = list(set([x.strip().lower() for x in raw_ids.replace(',', '\n').split('\n') if x.strip()]))
    df_selected = df_all[df_all['Market ID'].str.lower().isin(clean_ids)].copy()

with col_cash:
    st.subheader("3. Budget")
    new_cash = st.number_input("Additional New Cash (USD)", value=0.0, step=1000.0)

if not df_selected.empty:
    st.info("💡 Enter your current USD holdings for the selected markets below:")
    
    # Persistent balance logic
    df_selected['Existing Balance (USD) | Click cell to fill values'] = df_selected['Market ID'].apply(lambda x: st.session_state.balance_cache.get(x, 0.0))
    
    edited_df = st.data_editor(
        df_selected[['Market ID', 'Loan Token', 'Collateral', 'Existing Balance (USD)']],
        column_config={"Existing Balance (USD)": st.column_config.NumberColumn(format="$%.2f")},
        disabled=['Market ID', 'Loan Token', 'Collateral'],
        use_container_width=True, hide_index=True, key="portfolio_editor"
    )

    for _, row in edited_df.iterrows():
        st.session_state.balance_cache[row['Market ID']] = row['Existing Balance (USD)']

    current_wealth = edited_df['Existing Balance (USD)'].sum()
    total_optimizable = current_wealth + new_cash


    # ==========================================
    # LIVE METRICS DISPLAY
    # ==========================================
    st.divider()
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            label="💰 Total Current Balance",
            value=f"${current_wealth:,.2f}",
            help="Sum of all existing balances entered above"
        )
    
    with metric_col2:
        st.metric(
            label="💵 Additional cash to invest",
            value=f"${new_cash:,.2f}",
            help="New cash entered in Budget section"
        )
    
    with metric_col3:
        st.metric(
            label="📊 Total Balance",
            value=f"${total_optimizable:,.2f}",
            delta=f"${new_cash:,.2f}" if new_cash > 0 else None,
            delta_color="normal",
            help="Current Balance + Fresh Cash = Total available for optimization"
        )
    
    st.divider()

    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        market_data_list = fetch_live_market_details(df_selected)
        for m in market_data_list:
            m['existing_balance_usd'] = st.session_state.balance_cache.get(m['Market ID'], 0.0)

        opt = RebalanceOptimizer(total_optimizable, market_data_list)
        allocations = opt.optimize()

        if allocations is not None:
            st.divider()
            results = []
            total_yield = 0
            for i, target_alloc in enumerate(allocations):
                m = market_data_list[i]
                new_apy = opt.simulate_apy(m, target_alloc)
                net_move = target_alloc - m['existing_balance_usd']
                total_yield += (target_alloc * new_apy)
                
                # EMOJI ONLY ACTION
                action = "🟢 DEPOSIT" if net_move > 0.01 else ("🔴 WITHDRAW" if net_move < -0.01 else "⚪ HOLD")
                
                results.append({
                    "Market": f"{m['Loan Token']} / {m['Collateral']}",
                    "Action": action,
                    "Current ($)": m['existing_balance_usd'],
                    "Target ($)": target_alloc,
                    "Net Move ($)": net_move,
                    "Current APY": m['current_supply_apy'],
                    "New APY": new_apy,
                    "Market ID": m['Market ID']
                })
            
            df_res = pd.DataFrame(results)
            c1, c2, c3 = st.columns(3)
            c1.metric("Blended APY", f"{(total_yield/total_optimizable):.4%}")
            c2.metric("Annual Interest", f"${total_yield:,.2f}")
            c3.metric("Total Wealth", f"${total_optimizable:,.2f}")

            st.dataframe(
                df_res.style.format({
                    "Current ($)": "${:,.2f}", "Target ($)": "${:,.2f}", "Net Move ($)": "${:,.2f}",
                    "Current APY": "{:.4%}", "New APY": "{:.4%}"
                }), use_container_width=True, hide_index=True
            )
else:
    st.warning("Add Market IDs to section 2 to begin rebalancing.")
