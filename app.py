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
    
def extract_market_id_from_monarch_link(text: str) -> str:
    """Extract market ID from Monarch link or return original text if not a link"""
    text = text.strip()
    if 'monarchlend.xyz/market/' in text:
        # Extract the part after the last slash
        parts = text.rstrip('/').split('/')
        if len(parts) >= 2 and parts[-1].startswith('0x'):
            return parts[-1].lower()
    return text.lower()

# ==========================================
# 2. DATA FETCHING
# ==========================================

def get_market_dictionary():
    query = """
    query GetAllMarkets($first: Int, $skip: Int) {
      markets(first: $first, skip: $skip) {
        items {
          uniqueKey
          whitelisted
          loanAsset { symbol decimals chain { id } }
          collateralAsset { symbol }
          state {
            supplyApy
            supplyAssets
            borrowAssets
            supplyAssetsUsd
            borrowAssetsUsd
          }
        }
      }
    }
    """
    all_items = []
    skip = 0
    load_status = st.empty()
    
    while True:
        load_status.info(f"⏳ Retrieving Morpho Markets... {len(all_items)} of 3000+ markets found. Please wait as we are downloading 1000 market at a time...")
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
        state = m.get('state') or {}
        
        # Safe float conversion with None handling
        def safe_float(value, default=0.0):
            """Convert value to float, handling None and invalid values"""
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Get TVL values with safe conversion
        supply_usd = safe_float(state.get('supplyAssetsUsd'))
        borrow_usd = safe_float(state.get('borrowAssetsUsd'))
        available_liquidity = supply_usd - borrow_usd
        
        # Calculate utilization
        supply_assets = safe_float(state.get('supplyAssets'))
        borrow_assets = safe_float(state.get('borrowAssets'))
        utilization = (borrow_assets / supply_assets * 100) if supply_assets > 0 else 0
        
        processed.append({
            "Market ID": m['uniqueKey'],
            "Chain": CHAIN_ID_TO_NAME.get(loan.get('chain', {}).get('id'), "Other"),
            "Loan Token": loan.get('symbol'),
            "Collateral": (m.get('collateralAsset') or {}).get('symbol'),
            "Decimals": loan.get('decimals'),
            "ChainID": loan.get('chain', {}).get('id'),
            "Supply APY": safe_float(state.get('supplyApy')),
            "Utilization": utilization,
            "Total Supply (USD)": supply_usd,
            "Total Borrow (USD)": borrow_usd,
            "Available Liquidity (USD)": available_liquidity,
            "Whitelisted": m.get('whitelisted', False)
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
        res = minimize(
            self.objective,
            [self.total_budget/n]*n,
            method="SLSQP",
            bounds=[(0, self.total_budget)]*n,
            constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_budget}),
            options={"maxiter": 10000}   # <- set your cap here
        )
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
# --- ADD THIS AFTER df_all = st.session_state.market_dict ---
df_all = st.session_state.market_dict.copy()
df_all['Supply APY'] = pd.to_numeric(df_all['Supply APY'], errors='coerce').fillna(0.0)
df_all['Utilization'] = pd.to_numeric(df_all['Utilization'], errors='coerce').fillna(0.0)
df_all['Available Liquidity (USD)'] = pd.to_numeric(df_all['Available Liquidity (USD)'], errors='coerce').fillna(0.0)
df_all['Total Supply (USD)'] = pd.to_numeric(df_all['Total Supply (USD)'], errors='coerce').fillna(0.0)
df_all['Total Borrow (USD)'] = pd.to_numeric(df_all['Total Borrow (USD)'], errors='coerce').fillna(0.0)

st.subheader("1. Market Discovery")
st.caption("Use the filters below to narrow down markets, then copy Market IDs for optimization.")
st.markdown(
    "The tool assumes that you have selected a candidate list of markets. "
    "The tool assumes USD stables = 1 USD and only works for USD stables. "
    "Do not use for any other loan token. "
    "The tool also assumes that you have sensibly chosen the pools you are comfortable to invest in. "
    "You can use [Monarch Lend](https://monarchlend.xyz) to help you find markets as this should not be your primary market discovery tool"
)

# Prepare Token Display names for the multi-selects
def get_tokens(df, col):
    items = []
    for _, r in df.iterrows():
        name = str(r[col])
        mid = str(r['Market ID'])[:6]
        disp = f"{name} ({mid})" if name and name != "nan" else f"({mid})"
        items.append({'disp': disp, 'id': r['Market ID']})
    return items

loan_tokens = get_tokens(df_all, 'Loan Token')
collateral_tokens = get_tokens(df_all, 'Collateral')

unique_loan_disp = sorted(list(set(x['disp'] for x in loan_tokens)))
unique_coll_disp = sorted(list(set(x['disp'] for x in collateral_tokens)))
unique_chains = sorted(df_all['Chain'].dropna().unique().tolist())

# --- FILTER FORM ---
with st.form("market_filter_form"):
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        sel_chains = st.multiselect("Chains", options=unique_chains)
    with f_col2:
        sel_loans = st.multiselect("Loan Tokens", options=unique_loan_disp)
    with f_col3:
        sel_colls = st.multiselect("Collateral Tokens", options=unique_coll_disp)

    show_whitelisted = st.checkbox("Whitelisted?", value=False)

    st.markdown("---")
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        st.write("**Supply APY %**")
        sc1, sc2 = st.columns(2)
        m_apy_in = sc1.number_input("Min %", 0.0, 200.0, 0.0, step=1.0, key="m_apy_in")
        x_apy_in = sc2.number_input("Max %", 0.0, 200.0, 200.0, step=1.0, key="x_apy_in")
    
    with r1_c2:
        st.write("**Utilization %**")
        uc1, uc2 = st.columns(2)
        m_util_in = uc1.number_input("Min %", 0.0, 100.0, 0.0, step=5.0, key="m_util_in")
        x_util_in = uc2.number_input("Max %", 0.0, 100.0, 100.0, step=5.0, key="x_util_in")

    st.write("**Liquidity & Supply (USD)**")
    r2_c1, r2_c2 = st.columns(2)
    m_liq_in = r2_c1.number_input("Min Liquidity (USD)", 0.0, 1e15, 0.0, step=10000.0)
    m_sup_in = r2_c2.number_input("Min Total Supply (USD)", 0.0, 1e15, 0.0, step=10000.0)

    apply_btn = st.form_submit_button("Apply Filters", type="primary", use_container_width=True)

# --- FILTER APPLICATION ---
df_filtered = df_all.copy()
if sel_chains:
    df_filtered = df_filtered[df_filtered['Chain'].isin(sel_chains)]
if sel_loans:
    target_ids = [x['id'] for x in loan_tokens if x['disp'] in sel_loans]
    df_filtered = df_filtered[df_filtered['Market ID'].isin(target_ids)]
if sel_colls:
    target_ids = [x['id'] for x in collateral_tokens if x['disp'] in sel_colls]
    df_filtered = df_filtered[df_filtered['Market ID'].isin(target_ids)]
if show_whitelisted:
    df_filtered = df_filtered[df_filtered['Whitelisted'] == True]

df_filtered = df_filtered[
    (df_filtered['Supply APY'] >= (m_apy_in / 100.0)) & 
    (df_filtered['Supply APY'] <= (x_apy_in / 100.0)) &
    (df_filtered['Utilization'] >= m_util_in) & 
    (df_filtered['Utilization'] <= x_util_in) &
    (df_filtered['Available Liquidity (USD)'] >= m_liq_in) &
    (df_filtered['Total Supply (USD)'] >= m_sup_in)
]

# --- RESULTS DISPLAY ---
# Create a copy for display so we don't alter the actual data used in the optimizer
df_display = df_filtered.copy()

# Transform decimal APY (0.05) to percentage points (5.00) for display
df_display['Supply APY'] = df_display['Supply APY'] * 100 

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True,
    column_order=[
        "Whitelisted", "Chain", "Loan Token", "Collateral", 
        "Supply APY", "Utilization", 
        "Total Supply (USD)", "Total Borrow (USD)", "Available Liquidity (USD)",
        "Market ID"
    ],
    column_config={
        "Whitelisted": st.column_config.CheckboxColumn("Whitelisted?"),
        "Supply APY": st.column_config.NumberColumn(
            "Supply APY", 
            format="%.2f%%",
            help="Annual Percentage Yield (Percentage)"
        ),
        "Utilization": st.column_config.NumberColumn(
            "Utilization", 
            format="%.2f%%",
            help="Current market utilization"
        ),
        "Total Supply (USD)": st.column_config.NumberColumn("Total Supply (USD)", format="dollar"),
        "Total Borrow (USD)": st.column_config.NumberColumn("Total Borrow (USD)", format="dollar"),
        "Available Liquidity (USD)": st.column_config.NumberColumn("Avail. Liquidity (USD)", format="dollar")
    }
)

st.divider()

# --- UI SECTION 2: SCOPE & REBALANCE ---
col_scope, col_cash = st.columns([2, 1])

with col_scope:
    st.subheader("2. Your Portfolio Scope")
    raw_ids = st.text_area(
        "Paste Market IDs or Monarch links to optimize (one per line)", 
        value="", 
        height=300,
        placeholder="Market IDs\n0x347aa5f94a12dd46d3e17e542ca1c4033bd6952bde4b22af3caa33c82e52451a\n0x81b97c7305aca46c62f2ffce63a09c6a4d647163e25f31c44fadcbeab838b3f8\n\n- Monarch Links\nhttps://www.monarchlend.xyz/market/1/0x5ffdf15c5a4d7c6affb3f12634eeda1a20e60b92c0eb547f61754f656b55841e\nhttps://www.monarchlend.xyz/market/1/0xc7d717f4052ac4e5463dcc58cea0f6b05dd7d8c67e0aee68ebe30a8af09b259f\nhttps://www.monarchlend.xyz/market/1/0x0655e0c8686d94d9e0c0d2b78d7f99492676e52d712db5ac061b3c78da4b7587\nhttps://www.monarchlend.xyz/market/1/0x3f1d5c88c72432b04f2074499fe217468af49ddaa98bcb6ec80b08f76a82c6ec",
        help="💡 Copy Market IDs from the filtered table above OR paste Monarch links. The optimizer will find the best allocation across these markets."
    )
    # Process each line to extract market IDs from links or use raw IDs
    clean_ids = []
    for line in raw_ids.replace(',', '\n').split('\n'):
        line = line.strip()
        if line:
            extracted_id = extract_market_id_from_monarch_link(line)
            if extracted_id and extracted_id not in clean_ids:
                clean_ids.append(extracted_id)
    
    df_selected = df_all[df_all['Market ID'].str.lower().isin(clean_ids)].copy()


with col_cash:
    st.subheader("3. Budget")
    new_cash = st.number_input(
        "Additional New Cash (USD)", 
        value=0.0, 
        step=1000.0,
        help="💡 This is fresh capital you want to deploy. Set to $0 if only rebalancing existing positions."
    )

if not df_selected.empty:
    st.info("💡 Enter your current USD holdings for the selected markets below:")
    
    # Persistent balance logic
    df_selected['Existing Balance (USD)'] = df_selected['Market ID'].apply(lambda x: st.session_state.balance_cache.get(x, 0.0))
    
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

        # 1. Calculate Pre-Optimization Metrics (FIXED)
        # Current blended APY should ONLY consider existing deployed capital
        current_annual_interest = sum(m['existing_balance_usd'] * m['current_supply_apy'] for m in market_data_list)
        current_blended_apy = current_annual_interest / current_wealth if current_wealth > 0 else 0.0

        # 2. Run Optimizer
        opt = RebalanceOptimizer(total_optimizable, market_data_list)
        allocations = opt.optimize()

        if allocations is not None:
            st.divider()
            results = []
            new_annual_interest = 0
            
            for i, target_alloc in enumerate(allocations):
                m = market_data_list[i]
                new_apy = opt.simulate_apy(m, target_alloc)
                net_move = target_alloc - m['existing_balance_usd']
                
                new_annual_interest += (target_alloc * new_apy)
                
                action = "🟢 DEPOSIT" if net_move > 0.01 else ("🔴 WITHDRAW" if net_move < -0.01 else "⚪ HOLD")
                weight_pct = target_alloc / total_optimizable if total_optimizable > 0 else 0.0

                results.append({
                    "Market": f"{m['Loan Token']} / {m['Collateral']}",
                    "Chain": m['Chain'], 
                    "Suggested Action": action,
                    "Portfolio Weight": weight_pct,
                    "Current ($)": m['existing_balance_usd'],
                    "Target ($)": target_alloc,
                    "Net Move ($)": net_move,
                    "Current APY": m['current_supply_apy'],
                    "New APY": new_apy,
                    "Annual $ Yield Contribution": target_alloc * new_apy,
                    "Market ID": m['Market ID']
                })
            
            # 3. Calculate Post-Optimization Metrics
            new_blended_apy = new_annual_interest / total_optimizable if total_optimizable > 0 else 0.0
            
            # Calculate APY improvement correctly
            if current_wealth > 0:
                apy_diff = new_blended_apy - current_blended_apy
                interest_diff = new_annual_interest - current_annual_interest
            else:
                apy_diff = new_blended_apy  # All new deployment
                interest_diff = new_annual_interest

            # --- METRICS DISPLAY ---
            st.subheader("Results")
            
            # Show current vs optimized
            if current_wealth > 0:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric(
                    "Current Deployed APY", 
                    f"{current_blended_apy:.4%}",
                    help="Weighted APY of existing balances only"
                )
                m2.metric(
                    "Optimized Blended APY", 
                    f"{new_blended_apy:.4%}", 
                    delta=f"{apy_diff*100:.3f}%",
                    help="Projected APY after deploying all capital"
                )
                m3.metric("Current Deployed Capital", f"${current_wealth:,.2f}")
                m4.metric("Total Optimized Wealth (1 Yr)", f"${total_optimizable+new_annual_interest:,.2f}")
                m5.metric("Annual Interest Gain", f"+${interest_diff:,.2f}")
            else:
                # New portfolio - no "current" metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Optimized Blended APY", f"{new_blended_apy:.4%}")
                m2.metric("Total Capital Deployed", f"${total_optimizable:,.2f}")
                m3.metric("Annual Interest", f"${new_annual_interest:,.2f}")

            st.divider()

            # Row 2: Absolute Interest Breakdown (unchanged)
            r2_c1, r2_c2, r2_c3, r2_c4, r2_c5 = st.columns(5)
            r2_c1.metric("Annual Interest", f"${new_annual_interest:,.2f}")
            r2_c2.metric("Monthly Interest", f"${new_annual_interest/12:,.2f}")
            r2_c3.metric("Weekly Interest", f"${new_annual_interest/52:,.2f}")
            r2_c4.metric("Daily Interest", f"${new_annual_interest/365:,.2f}")
            r2_c5.metric("Hourly Interest", f"${new_annual_interest/8760:,.4f}")
            
            df_res = pd.DataFrame(results)

            # Sort by Suggested Action first, then Portfolio Weight descending
            df_res_sorted = df_res.sort_values(
                by=["Suggested Action", "Portfolio Weight"], 
                ascending=[False, False]  # True for Suggested Action, False for Portfolio Weight
            )

            st.dataframe(
                df_res_sorted.style.format({
                    "Portfolio Weight": "{:.2%}",
                    "Current ($)": "${:,.2f}", 
                    "Target ($)": "${:,.2f}", 
                    "Net Move ($)": "${:,.2f}",
                    "Current APY": "{:.4%}", 
                    "New APY": "{:.4%}",
                    "Annual $ Yield Contribution": "${:,.2f}",
                }), 
                use_container_width=True, 
                hide_index=True,
                column_order=[
                    "Market", "Chain", "Suggested Action", "Portfolio Weight", 
                    "Current ($)", "Target ($)", "Net Move ($)", 
                    "Current APY", "New APY", "Annual $ Yield Contribution"
                ]
            )





