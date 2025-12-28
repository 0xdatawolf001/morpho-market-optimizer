import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize, differential_evolution
import time
import altair as alt

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

# Default User Inputs
DEFAULT_HURDLE_RATE = 4.0

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
        
        def safe_float(value, default=0.0):
            if value is None: return default
            try: return float(value)
            except (ValueError, TypeError): return default
        
        supply_usd = safe_float(state.get('supplyAssetsUsd'))
        borrow_usd = safe_float(state.get('borrowAssetsUsd'))
        available_liquidity = supply_usd - borrow_usd
        
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
    def __init__(self, total_budget, market_list, hurdle_rate_pct):
        self.total_budget = total_budget 
        self.markets = market_list
        self.hurdle_rate = hurdle_rate_pct / 100.0
        
        # Tracking history for charts
        self.yield_trace = []    
        self.frontier_trace = [] 
        self.car_trace = []      
        self.all_attempts = []   
        
        # Current Optimization Context
        self.current_phase = "Initializing"
        self.start_time = time.time()

    def simulate_apy(self, market, user_new_alloc_usd):
        change_threshold = 1e-15 
        if abs(user_new_alloc_usd - market['existing_balance_usd']) < change_threshold:
            return market['current_supply_apy']
        
        multiplier = 10**market['Decimals']
        user_existing_wei = market['existing_balance_usd'] * multiplier
        user_new_wei = user_new_alloc_usd * multiplier
        
        other_users_supply_wei = max(0, market['raw_supply'] - user_existing_wei)
        simulated_total_supply_wei = other_users_supply_wei + user_new_wei
        
        if simulated_total_supply_wei <= 0: 
            return 0.0
        
        new_util = market['raw_borrow'] / simulated_total_supply_wei
        new_mult = compute_curve_multiplier(new_util)
        new_borrow_rate = market['rate_at_target'] * new_mult
        new_supply_rate = new_borrow_rate * new_util * (1 - market['fee'])
        
        return rate_per_second_to_apy(new_supply_rate)

    def _get_normalized_allocations(self, x):
        sum_x = np.sum(x)
        if sum_x == 0: return x 
        return x * (self.total_budget / sum_x)

    def _calculate_metrics(self, x):
        normalized_x = self._get_normalized_allocations(x)
        weights = normalized_x / self.total_budget if self.total_budget > 0 else np.zeros_like(x)
        
        total_yield = 0
        for i, alloc in enumerate(normalized_x):
            total_yield += (alloc * self.simulate_apy(self.markets[i], alloc))
            
        hhi = np.sum(weights**2)
        diversity = 1.0 - hhi
        
        hurdle_cost = self.total_budget * self.hurdle_rate
        excess_return = total_yield - hurdle_cost
        car_score = excess_return / (hhi + 1e-9)
        
        return normalized_x, total_yield, diversity, car_score

    def _record_attempt(self, total_yield, diversity):
        self.all_attempts.append({
            "Annual Yield ($)": total_yield,
            "Blended APY": total_yield / self.total_budget if self.total_budget > 0 else 0,
            "Diversity Score": diversity,
            "Type": "Explored" 
        })

    def objective_yield(self, x):
        norm_x, y_val, div_val, car_val = self._calculate_metrics(x)
        self._record_attempt(y_val, div_val)
        self.yield_trace.append(y_val)
        return -y_val

    def objective_frontier(self, x):
        norm_x, y_val, div_val, car_val = self._calculate_metrics(x)
        self._record_attempt(y_val, div_val)
        self.frontier_trace.append(y_val)
        return -(y_val * div_val) 

    def objective_car(self, x):
        norm_x, y_val, div_val, car_val = self._calculate_metrics(x)
        self._record_attempt(y_val, div_val)
        self.car_trace.append(y_val)
        return -car_val

    def optimize(self):
        n = len(self.markets)
        if n == 0: return None, None, None
        
        bounds = [(0, self.total_budget)] * n
        
        init_pop = []
        for i in range(min(n, 50)):
            g = np.zeros(n)
            g[i] = self.total_budget
            init_pop.append(g)
        init_pop.append(np.ones(n) * (self.total_budget/n))
        init = np.array(init_pop) if len(init_pop) >= 5 else None

        self.current_phase = "Max Yield"
        res_yield = differential_evolution(
            self.objective_yield, bounds, strategy='best1bin',
            maxiter=100000, popsize=10, tol=0.01,
            init=init if init is not None and init.shape[0] >= 5 else 'latinhypercube',
            mutation=(0.5, 1), recombination=0.7
        )
        best_yield_alloc = self._get_normalized_allocations(res_yield.x)

        self.current_phase = "Frontier"
        res_frontier = differential_evolution(
            self.objective_frontier, bounds, strategy='best1bin',
            maxiter=100000, popsize=10, tol=0.01,
            init=init if init is not None and init.shape[0] >= 5 else 'latinhypercube',
            mutation=(0.5, 1), recombination=0.7
        )
        best_frontier_alloc = self._get_normalized_allocations(res_frontier.x)

        self.current_phase = "Concentration-Adj"
        res_car = differential_evolution(
            self.objective_car, bounds, strategy='best1bin',
            maxiter=100000, popsize=10, tol=0.01,
            init=init if init is not None and init.shape[0] >= 5 else 'latinhypercube',
            mutation=(0.5, 1), recombination=0.7
        )
        best_car_alloc = self._get_normalized_allocations(res_car.x)
        
        return best_yield_alloc, best_frontier_alloc, best_car_alloc

# ==========================================
# 4. STREAMLIT UI
# ==========================================
if "balance_cache" not in st.session_state:
    st.session_state.balance_cache = {}

st.title("⚖️ Morpho Portfolio Optimizer")

# --- DATA INDEXING ---
if "market_dict" not in st.session_state:
    st.session_state.market_dict = get_market_dictionary()

df_all = st.session_state.market_dict.copy()
cols_to_num = ['Supply APY', 'Utilization', 'Available Liquidity (USD)', 'Total Supply (USD)', 'Total Borrow (USD)']
for c in cols_to_num:
    df_all[c] = pd.to_numeric(df_all[c], errors='coerce').fillna(0.0)

st.subheader("1. Market Discovery")
st.caption("Use the filters below to narrow down markets, then copy Market IDs for optimization.")

# Prepare Token Display names
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
    with f_col1: sel_chains = st.multiselect("Chains", options=unique_chains)
    with f_col2: sel_loans = st.multiselect("Loan Tokens", options=unique_loan_disp)
    with f_col3: sel_colls = st.multiselect("Collateral Tokens", options=unique_coll_disp)
    show_whitelisted = st.checkbox("Whitelisted?", value=False)
    st.markdown("---")
    r1_c1, r1_c2 = st.columns(2)
    m_apy_in = r1_c1.number_input("Min APY %", 0.0, 200.0, 0.0)
    x_apy_in = r1_c1.number_input("Max APY %", 0.0, 200.0, 200.0)
    m_util_in = r1_c2.number_input("Min Util %", 0.0, 100.0, 0.0)
    x_util_in = r1_c2.number_input("Max Util %", 0.0, 100.0, 100.0)
    apply_btn = st.form_submit_button("Apply Filters", type="primary", width='stretch')

# --- FILTER APPLICATION ---
df_filtered = df_all.copy()
if sel_chains: df_filtered = df_filtered[df_filtered['Chain'].isin(sel_chains)]
if sel_loans:
    target_ids = [x['id'] for x in loan_tokens if x['disp'] in sel_loans]
    df_filtered = df_filtered[df_filtered['Market ID'].isin(target_ids)]
if sel_colls:
    target_ids = [x['id'] for x in collateral_tokens if x['disp'] in sel_colls]
    df_filtered = df_filtered[df_filtered['Market ID'].isin(target_ids)]
if show_whitelisted: df_filtered = df_filtered[df_filtered['Whitelisted'] == True]

df_filtered = df_filtered[
    (df_filtered['Supply APY'] >= (m_apy_in / 100.0)) & 
    (df_filtered['Supply APY'] <= (x_apy_in / 100.0)) &
    (df_filtered['Utilization'] >= m_util_in) & 
    (df_filtered['Utilization'] <= x_util_in)
]

st.dataframe(df_filtered, width='stretch', hide_index=True)

st.divider()

# --- UI SECTION 2: SCOPE & REBALANCE ---
col_scope, col_param = st.columns([2, 1])

with col_scope:
    st.subheader("2. Your Portfolio Scope")
    raw_ids = st.text_area(
        "Paste Market IDs or Monarch links (one per line)", 
        value="", 
        height=200,
        placeholder="Market IDs or Links..."
    )
    clean_ids = []
    for line in raw_ids.replace(',', '\n').split('\n'):
        line = line.strip()
        if line:
            extracted_id = extract_market_id_from_monarch_link(line)
            if extracted_id and extracted_id not in clean_ids:
                clean_ids.append(extracted_id)
    
    df_selected = df_all[df_all['Market ID'].str.lower().isin(clean_ids)].copy()

with col_param:
    st.subheader("3. Parameters")
    new_cash = st.number_input("Additional New Cash (USD)", value=0.0, step=1000.0)
    # Hurdle Rate Box
    st.markdown("**Concentration-Adj Setting**")
    hurdle_rate = st.number_input(
        "Hurdle Rate (Risk Free %)", 
        # min_value=0.0, max_value=100.0, 
        value=DEFAULT_HURDLE_RATE, step=0.1,
        help="Used to calculate the Concentration-Adjusted Return. Represents the cost of capital."
    )

if not df_selected.empty:
    st.info("💡 Enter your current USD holdings for the selected markets:")
    
    # Track existing balances
    df_selected['Existing Balance (USD)'] = df_selected['Market ID'].apply(lambda x: st.session_state.balance_cache.get(x, 0.0))
    
    edited_df = st.data_editor(
        df_selected[['Market ID', 'Loan Token', 'Collateral', 'Existing Balance (USD)']],
        column_config={"Existing Balance (USD)": st.column_config.NumberColumn(format="$%.2f")},
        disabled=['Market ID', 'Loan Token', 'Collateral'],
        width='stretch', hide_index=True, key="portfolio_editor"
    )

    for _, row in edited_df.iterrows():
        st.session_state.balance_cache[row['Market ID']] = row['Existing Balance (USD)']

    current_wealth = edited_df['Existing Balance (USD)'].sum()
    total_optimizable = current_wealth + new_cash

    # ==========================================
    # LIVE METRICS & OPTIMIZATION LOGIC
    # ==========================================
    st.divider()
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("💰 Total Current Balance", f"${current_wealth:,.2f}")
    metric_col2.metric("💵 Additional cash", f"${new_cash:,.2f}")
    metric_col3.metric("📊 Total Optimizable", f"${total_optimizable:,.2f}")
    
    st.divider()

    # Portfolio Explanations placed above the button
    with st.expander("ℹ️ About the Portfolio Strategies", expanded=True):
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown("**🔴 Best Yield**")
            st.caption("Purely prioritizes raw APY. Finds the mathematical maximum yield, often resulting in 100% allocation to the single highest payer.")
        with col_info2:
            st.markdown("**🟣 Concentration-Adj Return**")
            st.caption("Maximizes `(Yield - Hurdle) / Concentration`. This penalizes high concentration (HHI) unless the excess return is massive.")
        with col_info3:
            st.markdown("**🌸 Frontier (Diversified)**")
            st.caption("Maximizes `Yield * Diversity`. Finds the sweet spot where you get good yield without putting all eggs in one basket.")

    # --- BUTTON TO RUN AND SAVE TO STATE ---
    if st.button("🚀 Run Optimization", type="primary", width='stretch'):
        market_data_list = fetch_live_market_details(df_selected)
        for m in market_data_list:
            m['existing_balance_usd'] = st.session_state.balance_cache.get(m['Market ID'], 0.0)

        # Pre-Optimization Stats
        current_annual_interest = sum(m['existing_balance_usd'] * m['current_supply_apy'] for m in market_data_list)
        current_blended_apy = current_annual_interest / current_wealth if current_wealth > 0 else 0.0

        # Run Optimizer (Status placeholder logic removed)
        opt = RebalanceOptimizer(total_optimizable, market_data_list, hurdle_rate)
        
        # RETURNS 3 DISTINCT SOLVED PORTFOLIOS
        best_yield_alloc, frontier_alloc, car_alloc = opt.optimize()
        
        # Save results to session state to prevent reset on interaction
        st.session_state['opt_results'] = {
            'market_data_list': market_data_list,
            'opt_object': opt,
            'best_yield_alloc': best_yield_alloc,
            'frontier_alloc': frontier_alloc,
            'car_alloc': car_alloc,
            'current_metrics': {
                'annual_interest': current_annual_interest,
                'blended_apy': current_blended_apy
            },
            'traces': {
                'yield': opt.yield_trace,
                'frontier': opt.frontier_trace,
                'car': opt.car_trace
            },
            'attempts': opt.all_attempts
        }

    # ==========================================
    # DISPLAY RESULTS (IF THEY EXIST)
    # ==========================================
    if 'opt_results' in st.session_state and st.session_state['opt_results']['best_yield_alloc'] is not None:
        
        # Load from State
        res_data = st.session_state['opt_results']
        opt = res_data['opt_object'] # NOTE: This object is stale but methods work if stateless
        market_data_list = res_data['market_data_list']
        best_yield_alloc = res_data['best_yield_alloc']
        frontier_alloc = res_data['frontier_alloc']
        car_alloc = res_data['car_alloc']
        
        # Helper to calc point stats for chart
        def get_stats(alloc):
            y = 0
            for i, val in enumerate(alloc):
                y += (val * opt.simulate_apy(market_data_list[i], val))
            weights = alloc/total_optimizable if total_optimizable > 0 else np.zeros_like(alloc)
            div = 1.0 - np.sum(weights**2)
            return y, y/total_optimizable if total_optimizable>0 else 0, div

        y_abs, y_apy, y_div = get_stats(best_yield_alloc)
        f_abs, f_apy, f_div = get_stats(frontier_alloc)
        c_abs, c_apy, c_div = get_stats(car_alloc)

        # Prepare Scatter Data
        df_scatter = pd.DataFrame(res_data['attempts'])
        
        highlights = pd.DataFrame([
            {"Diversity Score": y_div, "Blended APY": y_apy, "Type": "Best Yield", "Size": 100, "Color": "#F44336"},
            {"Diversity Score": f_div, "Blended APY": f_apy, "Type": "Frontier", "Size": 100, "Color": "#E040FB"}, # Magenta
            {"Diversity Score": c_div, "Blended APY": c_apy, "Type": "Concentration-Adj", "Size": 100, "Color": "#7C4DFF"} # Purple
        ])
        
        # --- CHART SECTION ---
        st.subheader("📊 Optimization Search Space")
        col_graph1, col_graph2 = st.columns(2)
        
        # 1. SCATTER PLOT
        with col_graph1:
            st.markdown("**Efficiency Frontier**")
            
            base = alt.Chart(df_scatter).mark_circle(opacity=0.3, color='#80DEEA').encode(
                x=alt.X('Diversity Score', title='Diversity (1 - HHI)'),
                y=alt.Y('Blended APY', title='APY', axis=alt.Axis(format='%')),
                tooltip=['Diversity Score', 'Blended APY', 'Annual Yield ($)']
            )
            
            points = alt.Chart(highlights).mark_circle(size=200, opacity=1.0).encode(
                x='Diversity Score',
                y='Blended APY',
                color=alt.Color('Type', scale=alt.Scale(
                    domain=['Best Yield', 'Frontier', 'Concentration-Adj'],
                    range=['#F44336', '#E040FB', '#7C4DFF'] 
                )),
                tooltip=['Type', 'Blended APY', 'Diversity Score']
            )
            
            st.altair_chart(base + points, width='stretch')
            st.caption("Purple Point represents the best Risk-Adjusted return based on Concentration.")

        # 2. LINE CHART (CONVERGENCE)
        with col_graph2:
            st.markdown("**Solver Convergence**")
            
            # Since we now run 3 separate loops, the traces might be different lengths
            # We will plot them as independent lines
            traces = res_data['traces']
            
            # Create separate DFs and concat
            d1 = pd.DataFrame({"Iteration": range(len(traces['yield'])), "Value": traces['yield'], "Strategy": "Best Yield Run"})
            d2 = pd.DataFrame({"Iteration": range(len(traces['frontier'])), "Value": traces['frontier'], "Strategy": "Frontier Run"})
            d3 = pd.DataFrame({"Iteration": range(len(traces['car'])), "Value": traces['car'], "Strategy": "Conc-Adj Run"})
            
            df_hist_long = pd.concat([d1, d2, d3])
            
            line_chart = alt.Chart(df_hist_long).mark_line().encode(
                x='Iteration',
                y=alt.Y('Value', title='Objective Yield ($)'),
                color=alt.Color('Strategy', scale=alt.Scale(range=['#F44336', '#E040FB', '#7C4DFF']))
            )
            st.altair_chart(line_chart, width='stretch')

        # --- MULTI-BAR CHART (ALL PORTFOLIOS) ---
        st.divider()
        st.markdown("**Allocation Comparison (All Strategies)**")
        
        # Construct long-format DF for Grouped Bar Chart
        bar_data = []
        for idx, m in enumerate(market_data_list):
            m_name = f"{m['Loan Token']}/{m['Collateral']}"
            
            # Data for the three strategies
            y_val = best_yield_alloc[idx]
            f_val = frontier_alloc[idx]
            c_val = car_alloc[idx]
            
            # Only include market if at least one strategy allocates more than $1
            if y_val > 1 or f_val > 1 or c_val > 1:
                bar_data.append({"Market": m_name, "Strategy": "Best Yield", "Alloc ($)": y_val})
                bar_data.append({"Market": m_name, "Strategy": "Frontier", "Alloc ($)": f_val})
                bar_data.append({"Market": m_name, "Strategy": "Conc-Adj", "Alloc ($)": c_val})
            
        if bar_data:
            df_bar = pd.DataFrame(bar_data)
            
            # Using xOffset for side-by-side grouping instead of faceted columns
            bar_chart = alt.Chart(df_bar).mark_bar().encode(
                x=alt.X('Market:N', 
                        title="Market Pair",
                        axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Alloc ($):Q', 
                        title="Allocation (USD)",
                        scale=alt.Scale(zero=True)),
                xOffset='Strategy:N',
                color=alt.Color('Strategy:N', scale=alt.Scale(
                    domain=['Best Yield', 'Frontier', 'Conc-Adj'],
                    range=['#F44336', '#E040FB', '#7C4DFF']
                )),
                tooltip=['Market', 'Strategy', alt.Tooltip('Alloc ($)', format='$,.2f')]
            ).properties(
                height=400  # Fixed height prevents the "squashed" look
            ).configure_view(
                stroke=None
            )
            
            st.altair_chart(bar_chart, width='stretch')
        else:
            st.info("No significant allocations to display in chart.")

        # --- SELECTION & TABLE RESULTS ---
        st.divider()
        
        st.subheader("🔍 Results")

        strategy_choice = st.radio(
            "View Details For:",
            ["Best Yield (Red)", "Frontier / Diversified (Magenta)", "Concentration-Adj (Purple)"],
            horizontal=True
        )
        
        if "Best Yield" in strategy_choice:
            final_alloc = best_yield_alloc
        elif "Frontier" in strategy_choice:
            final_alloc = frontier_alloc
        else:
            final_alloc = car_alloc

        # Calculate Final Table Details based on selection
        results = []
        new_annual_interest = 0
        
        for i, target_val in enumerate(final_alloc):
            m = market_data_list[i]
            # Use current stateless optimizer method for calc
            new_apy = opt.simulate_apy(m, target_val)
            new_annual_interest += (target_val * new_apy)
            net_move = target_val - m['existing_balance_usd']
            action = "🟢 DEPOSIT" if net_move > 0.01 else ("🔴 WITHDRAW" if net_move < -0.01 else "⚪ HOLD")
            
            results.append({
                "Market": f"{m['Loan Token']}/{m['Collateral']}",
                "Chain": m['Chain'], 
                "Suggested Action": action,
                "Portfolio Weight": target_val / total_optimizable if total_optimizable > 0 else 0,
                "Current ($)": m['existing_balance_usd'],
                "Target ($)": target_val,
                "Net Move ($)": net_move,
                "Current APY": m['current_supply_apy'],
                "New APY": new_apy,
                "Annual $ Yield": target_val * new_apy,
            })
        
        df_res = pd.DataFrame(results)
        
        # Add Yield Contribution Column
        total_gen_yield = df_res["Annual $ Yield"].sum()
        df_res["Yield Contribution"] = df_res["Annual $ Yield"].apply(
            lambda x: x / total_gen_yield if total_gen_yield > 0 else 0
        )
        
        current_blended = res_data['current_metrics']['blended_apy']
        current_ann = res_data['current_metrics']['annual_interest']
        new_blended_apy = new_annual_interest / total_optimizable if total_optimizable > 0 else 0.0
        
        # --- 1. Financial Summary Metrics (RESTORED) ---
        st.subheader("Optimization Summary")

        # Calculate differences for deltas
        apy_diff = new_blended_apy - current_blended
        interest_diff = new_annual_interest - current_ann

        # Calculate Diversity for the currently selected strategy
        selected_weights = final_alloc / total_optimizable if total_optimizable > 0 else np.zeros_like(final_alloc)
        selected_diversity = 1.0 - np.sum(selected_weights**2)

        m1, m2, m3, m4, m5 = st.columns(5) 
        m1.metric(
            "Current APY", 
            f"{current_blended:.4%}",
            help="Weighted APY of existing balances only"
        )
        m2.metric(
            "Optimized APY", 
            f"{new_blended_apy:.4%}", 
            delta=f"{apy_diff*100:.3f}%",
            help="Projected APY after rebalancing/deploying capital"
        )
        m3.metric("Total Wealth", f"${total_optimizable:,.2f}")
        m4.metric(
            "Total Wealth (1 Yr)", 
            f"${total_optimizable + new_annual_interest:,.2f}",
            help="Principal plus projected annual interest"
        )
        m5.metric(
            "Additional Gain", 
            f"${interest_diff:,.2f}",
            delta_color="normal",
            help="Extra annual interest earned compared to current setup"
        )
        
        # Diversity metric placed below or in a separate row if preferred
        st.caption(f"**Diversity Index:** {selected_diversity:.4f} (1.0 is perfectly diversified, closer to 0 is concentrated)")
        
        # 2. Detailed Interest Breakdown
        st.write("---")
        r2_c1, r2_c2, r2_c3, r2_c4, r2_c5 = st.columns(5)
        r2_c1.metric("Annual", f"${new_annual_interest:,.2f}")
        r2_c2.metric("Monthly", f"${new_annual_interest/12:,.2f}")
        r2_c3.metric("Weekly", f"${new_annual_interest/52:,.2f}")
        r2_c4.metric("Daily", f"${new_annual_interest/365:,.2f}")
        r2_c5.metric("Hourly", f"${new_annual_interest/8760:,.4f}")

        # 3. Sorted Dataframe with Column Ordering
        df_res = df_res.sort_values(by=["Suggested Action", "Portfolio Weight"], ascending=[False, False])

        st.dataframe(
            df_res.style.format({
                "Portfolio Weight": "{:.2%}", 
                "Current ($)": "${:,.2f}", 
                "Target ($)": "${:,.2f}", 
                "Net Move ($)": "${:,.2f}",
                "Current APY": "{:.4%}", 
                "New APY": "{:.4%}", 
                "Annual $ Yield": "${:,.2f}", 
                "Yield Contribution": "{:.2%}"
            }), 
            width='stretch', 
            hide_index=True,
            column_order=[
                "Market", "Chain", "Suggested Action", "Portfolio Weight", 
                "Current ($)", "Target ($)", "Net Move ($)", 
                "Current APY", "New APY", "Annual $ Yield", "Yield Contribution"
            ]
        )
else:
    st.warning("Please paste Market IDs or Monarch links in Section 2 to begin.")