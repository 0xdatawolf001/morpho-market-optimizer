import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize, differential_evolution
import time
from datetime import datetime, timedelta, timezone
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

# LI.FI API Configuration
MAX_RETRIES = 3
RETRY_BACKOFF = 2 # Seconds

# Chart Performance Constants
MAX_SCATTER_PLOT_POINTS = 5000 # Max points for the Efficiency Frontier plot
MAX_LINE_PLOT_POINTS_PER_STRATEGY = 1000 # Max points per strategy for the Solver Convergence plot
DEMAND_TREND_CHART_HEIGHT = 800 # Increased for better vertical visibility

# For text box
# Separator Constant
MANUAL_SEP = "-- Selected Markets --"
WALLET_SEP = "-- From User Wallet --"

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

def apy_to_apr(apy_float: float) -> float:
    """Converts compounded APY to Linear APR (Annual Percentage Rate)"""
    if apy_float <= 0: return 0.0
    rate_per_sec = apy_to_rate_per_second(apy_float)
    return rate_per_sec * SECONDS_PER_YEAR

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

def filter_small_moves(allocations, market_data_list, threshold_usd, total_budget):
    """
    Applies a post-optimization filter.
    1. If difference between Target and Current is < threshold, revert to Current (HOLD).
    2. If resulting Unallocated Cash (Wallet) is < threshold, force-reallocate it to markets.
    """
    cleaned_allocations = allocations.copy()
    
    # 1. Filter individual Market Moves
    for i, target_val in enumerate(cleaned_allocations):
        current_balance = market_data_list[i]['existing_balance_usd']
        
        # Calculate raw move size
        diff = abs(target_val - current_balance)
        
        # Logic: If move is significant enough (> 0.01 cents) but below user threshold
        # We revert the target to match the current balance exactly.
        if diff > 0.01 and diff < threshold_usd:
            cleaned_allocations[i] = current_balance

    # 2. Filter Wallet Dust (Unallocated Cash < Threshold)
    # If the optimizer (or the filter above) leaves a small amount of cash that is 
    # below the threshold, we push it back into the markets to avoid a tiny "Cash Out".
    current_total = np.sum(cleaned_allocations)
    unallocated = total_budget - current_total
    
    if unallocated > 0.01 and unallocated < threshold_usd:
        # Priority: Add back to a market we are withdrawing from (to reduce the withdrawal)
        # Fallback: Add to the largest allocation
        
        candidates = []
        for i, target_val in enumerate(cleaned_allocations):
            if target_val < market_data_list[i]['existing_balance_usd']:
                candidates.append(i)
        
        if candidates:
            # Reduce withdrawal on the first candidate
            cleaned_allocations[candidates[0]] += unallocated
        else:
            # No withdrawals, just add to largest bucket
            best_idx = np.argmax(cleaned_allocations)
            cleaned_allocations[best_idx] += unallocated
            
    return cleaned_allocations

# ==========================================
# 2. API HELPERS
# ==========================================

def to_atomic_units(amount_adjusted: float, decimals: int) -> str:
    """
    Safely converts decimal-adjusted amount to atomic units (string integer).
    Prevents floating point errors by using rounding before casting.
    """
    # FIX: Explicit check for missing decimals
    if decimals is None or not isinstance(decimals, (int, float)):
        raise ValueError("Invalid decimals provided for atomic conversion.")
        
    if not amount_adjusted or amount_adjusted <= 0:
        return "0"
        
    return str(int(round(amount_adjusted * (10**int(decimals)))))

def get_lifi_quote(from_chain, to_chain, from_token, to_token, amount_atomic, user_address, user_slippage_decimal):
    url = "https://li.quest/v1/advanced/routes"
    
    # Construct the Ladder
    ladder = [user_slippage_decimal]
    # Standard steps: 0.05%, 0.1%, 0.5%, 1%, 5%, 10%
    standard_tiers = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    for tier in standard_tiers:
        if tier > user_slippage_decimal:
            ladder.append(tier)
            
    # Unique, sorted values starting with the user's lowest preference
    ladder = sorted(list(set(ladder)))
    if ladder[0] != user_slippage_decimal and user_slippage_decimal in ladder:
        ladder.remove(user_slippage_decimal)
        ladder.insert(0, user_slippage_decimal)

    last_error = None

    # OUTER LOOP: The Slippage Ladder
    for slippage_tier in ladder:
        payload = {
            "fromChainId": int(from_chain),
            "fromAmount": amount_atomic,
            "fromTokenAddress": from_token,
            "toChainId": int(to_chain),
            "toTokenAddress": to_token,
            "options": {
                "slippage": slippage_tier,
                "order": "CHEAPEST",
                "allowSwitchChain": True,
                "allowDestinationCall": True,
                "timing": {
                    "routeTimingStrategies": [
                        {
                            "strategy": "minWaitTime",
                            "minWaitTimeMs": 1500,
                            "startingExpectedResults": 6,
                            "reduceEveryMs": 500
                        }
                    ]
                }
            }
        }

        if user_address and len(user_address) > 10:
            payload["fromAddress"] = user_address

        # INNER LOOP: Retries for Network/Rate Limits
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    routes = data.get('routes', [])
                    
                    if not routes:
                        last_error = f"No routes found for tier {slippage_tier*100}%"
                        break 
                    
                    best_route = routes[0]
                    
                    # Extraction
                    amount_in_usd = float(best_route.get('fromAmountUSD', 0))
                    amount_out_usd = float(best_route.get('toAmountUSD', 0))
                    gas_usd = float(best_route.get('gasCostUSD', 0))
                    
                    fee_usd = 0.0
                    path_steps = []
                    for step in best_route.get('steps', []):
                        step_fees = step.get('estimate', {}).get('feeCosts', [])
                        if step_fees:
                            fee_usd += sum([float(f.get('amountUSD', 0)) for f in step_fees])
                        tool_name = step.get('toolDetails', {}).get('name') or step.get('tool')
                        if tool_name: path_steps.append(tool_name)

                    return {
                        "success": True,
                        "total_cost": (amount_in_usd - amount_out_usd) + gas_usd,
                        "gas": gas_usd,
                        "fees": fee_usd,
                        "swap_and_impact_cost": max(0.0, amount_in_usd - amount_out_usd),
                        "amount_in": amount_in_usd,
                        "amount_out": amount_out_usd,
                        "ladder_tier_used": slippage_tier,
                        "execution_path": " ➡️ ".join(path_steps) if path_steps else "Optimized Route"
                    }
                
                elif response.status_code == 429:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
                else:
                    last_error = f"API Error {response.status_code}"
                    break 
                    
            except Exception as e:
                last_error = str(e)
                time.sleep(RETRY_BACKOFF)
    
    return {"success": False, "error": last_error}

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
          loanAsset { 
            address  # <--- ADDED THIS
            symbol 
            decimals 
            priceUsd 
            chain { id } 
          }
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
        load_status.info(f"⏳ Retrieving Morpho Markets... {len(all_items)} found. Please wait...")
        resp = requests.post(MORPHO_API_URL, json={'query': query, 'variables': {"first": BATCH_SIZE, "skip": skip}})
        json_data = resp.json()
        
        if 'data' not in json_data:
            st.error(f"API Error: {json_data.get('errors', 'Unknown Error')}")
            st.stop()
            
        items = json_data['data']['markets']['items']
        if not items: break
        all_items.extend(items)
        if len(items) < BATCH_SIZE: break
        skip += BATCH_SIZE
        
    load_status.empty()
    
    processed = []
    for m in all_items:
        loan = m.get('loanAsset') or {}
        state = m.get('state') or {}
        
        # FIX: Strict data validation. If loan metadata or price is missing, skip the market.
        price_usd = loan.get('priceUsd')
        loan_address = loan.get('address')
        decimals = loan.get('decimals')
        
        if price_usd is None or price_usd <= 0 or not loan_address or decimals is None:
            continue # Skip markets with "Ghost Data"

        def safe_float(value, default=0.0):
            if value is None: return default
            try: return float(value)
            except (ValueError, TypeError): return default
        
        supply_usd = safe_float(state.get('supplyAssetsUsd'))
        borrow_usd = safe_float(state.get('borrowAssetsUsd'))
        
        processed.append({
            "Market ID": m['uniqueKey'],
            "Chain": CHAIN_ID_TO_NAME.get(loan.get('chain', {}).get('id'), "Other"),
            "Loan Token": loan.get('symbol'),
            "Loan Address": loan_address, 
            "Collateral": (m.get('collateralAsset') or {}).get('symbol'),
            "Decimals": int(decimals),
            "Price USD": float(price_usd),
            "ChainID": loan.get('chain', {}).get('id'),
            "Supply APY": safe_float(state.get('supplyApy')),
            "Utilization": (safe_float(state.get('borrowAssets')) / safe_float(state.get('supplyAssets'))) if safe_float(state.get('supplyAssets')) > 0 else 0,
            "Total Supply (USD)": supply_usd,
            "Total Borrow (USD)": borrow_usd,
            "Available Liquidity (USD)": supply_usd - borrow_usd,
            "Whitelisted": m.get('whitelisted', False)
        })
    return (pd.DataFrame(processed) 
              .drop_duplicates(subset=["Market ID"], keep=False) 
              .query('Collateral.notna() & (Collateral != "") & (Collateral != " ")') 
              .sort_values('Total Supply (USD)', ascending=False)
           )

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

def fetch_user_positions(user_address):
    """
    Fetches all active supply positions for a specific user address across all chains.
    Returns a dict: {market_id: supply_usd_balance}
    """
    user_address = user_address.lower()
    query = """
    query GetUserPositions($user: [String!]) {
      marketPositions(first: 100, where: { userAddress_in: $user }) {
        items {
          market { uniqueKey }
          supplyAssetsUsd
        }
      }
    }
    """
    
    positions = {}
    try:
        # We fetch simply using the main endpoint; Morpho API aggregates chains usually
        variables = {"user": [user_address]}
        resp = requests.post(MORPHO_API_URL, json={'query': query, 'variables': variables})
        data = resp.json()
        
        if 'data' in data and 'marketPositions' in data['data']:
            items = data['data']['marketPositions']['items']
            for item in items:
                m_id = item['market']['uniqueKey']
                bal = float(item['supplyAssetsUsd'])
                if bal > 0.01: # Filter dust
                    positions[m_id] = bal
    except Exception as e:
        st.error(f"Error fetching user positions: {e}")
        
    return positions

def fetch_historical_flows(market_ids_and_chains):
    """
    Fetches 30-day history for supplyShares and borrowShares.
    Calculates a Balanced Demand Index using the Geometric Mean: sqrt(Supply * Borrow).
    This penalizes one-sided markets and rewards synergy between lenders and borrowers.
    """
    now = datetime.now(timezone.utc)
    start_time = int((now - timedelta(days=30)).timestamp())
    
    historical_data = []
    hist_bar = st.progress(0, text="Analyzing 30-day balanced demand trends...")
    
    for idx, m in enumerate(market_ids_and_chains):
        query = """
        query GetMarketHistory($uniqueKey: String!, $chainId: Int!, $options: TimeseriesOptions) {
            marketByUniqueKey(uniqueKey: $uniqueKey, chainId: $chainId) {
                historicalState {
                    supplyShares(options: $options) { x y }
                    borrowShares(options: $options) { x y }
                }
            }
        }
        """
        
        variables = {
            "uniqueKey": m['uniqueKey'],
            "chainId": int(m['chainId']),
            "options": {"startTimestamp": start_time, "interval": "DAY"}
        }
        
        try:
            resp = requests.post(MORPHO_API_URL, json={'query': query, 'variables': variables})
            data = resp.json()
            
            if 'data' in data and data['data'].get('marketByUniqueKey'):
                hist = data['data']['marketByUniqueKey']['historicalState']
                df_sup = pd.DataFrame(hist['supplyShares'])
                df_bor = pd.DataFrame(hist['borrowShares'])
                
                if not df_sup.empty and not df_bor.empty:
                    df_merged = pd.merge(df_sup, df_bor, on='x', suffixes=('_sup', '_bor'))
                    df_merged['y_sup'] = df_merged['y_sup'].astype(float)
                    df_merged['y_bor'] = df_merged['y_bor'].astype(float)
                    
                    # MATH UPDATE: Geometric Mean ensures both Supply and Borrow must exist to score
                    df_merged['balanced_activity'] = np.sqrt(df_merged['y_sup'] * df_merged['y_bor'])
                    df_merged = df_merged.sort_values('x', ascending=True)
                    
                    # Find first day with actual activity to avoid 0-division baseline
                    active_days = df_merged[df_merged['balanced_activity'] > 0]
                    if not active_days.empty:
                        initial_activity = active_days['balanced_activity'].iloc[0]
                        df_merged['indexed_demand'] = df_merged['balanced_activity'] / initial_activity
                    else:
                        df_merged['indexed_demand'] = 1.0
                        
                    short_id = m['uniqueKey'][0:6]
                    unique_label = f"{m['name']} [{short_id}]"
                    df_merged['Market'] = unique_label
                    df_merged['date'] = pd.to_datetime(df_merged['x'], unit='s')
                    historical_data.append(df_merged[['date', 'Market', 'indexed_demand']])
        except Exception as e:
            st.error(f"Error fetching history for {m['uniqueKey']}: {e}")
            
        hist_bar.progress((idx + 1) / len(market_ids_and_chains))
        
    hist_bar.empty()
    return pd.concat(historical_data) if historical_data else pd.DataFrame()

# ==========================================
# 3. OPTIMIZER
# ==========================================

class RebalanceOptimizer:
    def __init__(self, total_budget, market_list, max_dominance_pct=100.0, max_port_alloc_pct=100.0, max_supply_pct=100.0, max_borrow_pct=100.0):
        self.total_budget = total_budget 
        self.markets = market_list
        self.max_dominance_ratio = max_dominance_pct / 100.0 
        self.max_port_alloc_ratio = max_port_alloc_pct / 100.0
        # NEW: Store supply and borrow dominance ratios
        self.max_supply_ratio = max_supply_pct / 100.0
        self.max_borrow_ratio = max_borrow_pct / 100.0
        
        self.yield_trace = []    
        self.frontier_trace = [] 
        self.liquid_trace = []
        self.whale_trace = []
        self.capacity_warning = None 
        self.all_attempts = []

    def simulate_apy(self, market, user_new_alloc_usd):
        token_price = market['Price USD']
        if token_price <= 0: return 0.0
        decimals = market['Decimals']
        multiplier = 10**decimals

        user_existing_usd = market.get('existing_balance_usd', 0.0)
        user_existing_wei = (user_existing_usd / token_price) * multiplier
        
        current_total_supply_wei = market['raw_supply']
        base_supply_wei = max(0, current_total_supply_wei - user_existing_wei)
        
        user_new_wei = (user_new_alloc_usd / token_price) * multiplier
        simulated_total_supply_wei = base_supply_wei + user_new_wei
        
        if simulated_total_supply_wei <= 1: 
            return 0.0
        
        new_util = market['raw_borrow'] / simulated_total_supply_wei
        clamped_util = min(1.0, new_util)
        new_mult = compute_curve_multiplier(clamped_util)
        new_borrow_rate = market['rate_at_target'] * new_mult
        new_supply_rate = new_borrow_rate * new_util * (1 - market['fee'])
        
        return rate_per_second_to_apy(new_supply_rate)

    def _calculate_metrics(self, x):
        total_yield = 0
        for i, alloc in enumerate(x):
            total_yield += (alloc * self.simulate_apy(self.markets[i], alloc))
            
        weights = x / self.total_budget if self.total_budget > 0 else np.zeros_like(x)
        hhi = np.sum(weights**2)
        diversity = 1.0 - hhi
        
        return x, total_yield, diversity

    def _record_attempt(self, total_yield, diversity):
        self.all_attempts.append({
            "Annual Yield ($)": total_yield,
            "Blended APY": total_yield / self.total_budget if self.total_budget > 0 else 0,
            "Diversity Score": diversity,
            "Type": "Explored" 
        })

    def objective_yield(self, x):
        _, y_val, div_val = self._calculate_metrics(x)
        self._record_attempt(y_val, div_val)
        self.yield_trace.append(y_val)
        return -y_val

    def objective_frontier(self, x):
        _, y_val, div_val = self._calculate_metrics(x)
        self._record_attempt(y_val, div_val)
        score = (y_val + 1e-9) * (div_val + 1e-9)
        self.frontier_trace.append(y_val) 
        return -score 
    
    def objective_liquidity(self, x):
        _, y_val, div_val = self._calculate_metrics(x)
        self._record_attempt(y_val, div_val)
        self.liquid_trace.append(y_val)
        liq_score = 0
        for i, alloc in enumerate(x):
            if alloc > 1.0: 
                market = self.markets[i]
                sim_apy = self.simulate_apy(market, alloc)
                avail_liq = market.get('Available Liquidity (USD)', 0.0)
                liq_weight = math.log10(max(10.0, avail_liq)) 
                liq_score += (alloc * sim_apy * liq_weight)
        return -liq_score

    def objective_whale(self, x):
        _, y_val, div_val = self._calculate_metrics(x)
        self._record_attempt(y_val, div_val)
        self.whale_trace.append(y_val)
        return -y_val

    def optimize(self):
        n = len(self.markets)
        if n == 0: return None, None, None, None
        
        standard_bounds = []
        whale_bounds = []
        port_cap_usd = self.total_budget * self.max_port_alloc_ratio
        
        # Available Liquidity factor (Whale Shield only)
        if self.max_dominance_ratio >= 0.999:
            whale_cap_factor = 1e12 
        else:
            whale_cap_factor = self.max_dominance_ratio / (1.0 - self.max_dominance_ratio)

        # NEW: Supply dominance factor (Global)
        if self.max_supply_ratio >= 0.999:
            supply_cap_factor = 1e12
        else:
            supply_cap_factor = self.max_supply_ratio / (1.0 - self.max_supply_ratio)

        for m in self.markets:
            current_bal = m.get('existing_balance_usd', 0.0)
            token_price = m.get('Price USD', 0)
            multiplier = 10**m.get('Decimals', 18)
            
            # 1. Calc Metadata for Caps
            user_existing_wei = (current_bal / token_price) * multiplier if token_price > 0 else 0
            # Total Supply minus User
            base_supply_wei = max(0, m.get('raw_supply', 0) - user_existing_wei)
            base_supply_usd = (base_supply_wei / multiplier) * token_price if token_price > 0 else 0
            # Total Borrow
            total_borrow_usd = (m.get('raw_borrow', 0) / multiplier) * token_price if token_price > 0 else 0
            
            # 2. Global Caps (Applied to ALL strategies)
            # Cap based on Portfolio Size
            cap_port = min(self.total_budget, port_cap_usd)
            # Cap based on Total Supply %: UserSupply / (UserSupply + BaseSupply) <= MaxRatio
            cap_supply = max(0.0, base_supply_usd * supply_cap_factor)
            # Cap based on Total Borrow %: UserSupply / TotalBorrow <= MaxRatio
            cap_borrow = max(0.0, total_borrow_usd * self.max_borrow_ratio)
            
            # The strictly allowed upper bound for ALL strategies
            global_upper = min(cap_port, cap_supply, cap_borrow)

            std_lower, std_upper = 0.0, global_upper
            
            if m.get('force_exit', False):
                std_lower = std_upper = 0.0
            else:
                if m.get('prevent_outflows', False): std_lower = min(current_bal, global_upper)
                if m.get('prevent_inflows', False): std_upper = min(current_bal, global_upper)

            standard_bounds.append((std_lower, max(std_lower, std_upper)))
            
            # 3. Liquidity Shield Cap (Only applied to Whale/Liquid strategies)
            base_available_usd = ((max(0, m.get('raw_supply', 0) - m.get('raw_borrow', 0)) - user_existing_wei) / multiplier) * token_price if token_price > 0 else 0
            liq_shield_cap = max(0, base_available_usd * whale_cap_factor)
            
            # Whale bounds inherit the global restrictions but add the liquidity shield
            whale_upper = min(std_upper, liq_shield_cap)
            if std_lower > whale_upper: whale_upper = std_lower
            whale_bounds.append((std_lower, max(std_lower, whale_upper)))

        x0 = np.array([(b[0] + b[1]) / 2.0 for b in standard_bounds])
        if np.sum(x0) > 0: x0 = x0 * (self.total_budget / np.sum(x0))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_budget})
        options = {'maxiter': 2000, 'ftol': 1e-8}

        print(f"DEBUG: Optimization starting. Total Budget: ${self.total_budget:,.2f}")
        res_yield = minimize(self.objective_yield, x0, method='SLSQP', bounds=standard_bounds, constraints=constraints, options=options)
        res_frontier = minimize(self.objective_frontier, x0, method='SLSQP', bounds=standard_bounds, constraints=constraints, options=options)
        res_liq = minimize(self.objective_liquidity, x0, method='SLSQP', bounds=whale_bounds, constraints=constraints, options=options)
        res_whale = minimize(self.objective_whale, x0, method='SLSQP', bounds=whale_bounds, constraints=constraints, options=options)
        
        return res_yield.x, res_frontier.x, res_liq.x, res_whale.x
    
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

# Simplified token helper: returns sorted list of unique symbols
def get_tokens(df, col):
    """Extracted unique symbols without terminal logging."""
    unique_symbols = df[col].dropna().unique()
    return sorted([str(s) for s in unique_symbols if str(s).lower() != 'nan'])

loan_symbols = get_tokens(df_all, 'Loan Token')
collateral_symbols = get_tokens(df_all, 'Collateral')
unique_chains = sorted(df_all['Chain'].dropna().unique().tolist())

# --- FILTER FORM ---
with st.form("market_filter_form"):
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1: sel_chains = st.multiselect("Chains", options=unique_chains)
    with f_col2: sel_loans = st.multiselect("Loan Tokens", options=loan_symbols)
    with f_col3: sel_colls = st.multiselect("Collateral Tokens", options=collateral_symbols)
    show_whitelisted = st.checkbox("Whitelisted?", value=False)
    
    st.markdown("---")
    # Metric Range Row 1: APY
    r1_c1, r1_c2 = st.columns(2)
    m_apy_in = r1_c1.number_input("Min APY %", min_value=0.0, value = 0.0)
    x_apy_in = r1_c2.number_input("Max APY %", min_value=0.0, value=1000.0)
    
    # Metric Range Row 2: Utilization
    r2_c1, r2_c2 = st.columns(2)
    m_util_in = r2_c1.number_input("Min Util %", 0.0, 100.0, 0.0)
    x_util_in = r2_c2.number_input("Max Util %", 0.0, 100.0, 100.0)
    
    # Metric Row 3: Liquidity Thresholds
    r3_c1, r3_c2 = st.columns(2)
    m_supply_usd = r3_c1.number_input("Min Total Supply (USD)", 0.0, 10_000_000_000.0, 0.0)
    m_avail_usd = r3_c2.number_input("Min Available Liquidity (USD)", 0.0, 10_000_000_000.0, 0.0)
    
    apply_btn = st.form_submit_button("Apply Filters", type="primary", width='stretch')

# --- FILTER APPLICATION ---
df_filtered = df_all.copy()
if sel_chains: 
    df_filtered = df_filtered[df_filtered['Chain'].isin(sel_chains)]
    
if sel_loans:
    df_filtered = df_filtered[df_filtered['Loan Token'].isin(sel_loans)]
    
if sel_colls:
    df_filtered = df_filtered[df_filtered['Collateral'].isin(sel_colls)]

if show_whitelisted: 
    df_filtered = df_filtered[df_filtered['Whitelisted'] == True]

df_filtered = df_filtered[
    (df_filtered['Supply APY'] >= (m_apy_in / 100.0)) & 
    (df_filtered['Supply APY'] <= (x_apy_in / 100.0)) &
    (df_filtered['Utilization'] >= m_util_in / 100.0) & 
    (df_filtered['Utilization'] <= x_util_in / 100.0) &
    (df_filtered['Total Supply (USD)'] >= m_supply_usd) &
    (df_filtered['Available Liquidity (USD)'] >= m_avail_usd)
]

# Add the URL column
df_filtered['Link To Market'] = df_filtered.apply(
    lambda x: f"https://www.monarchlend.xyz/market/{int(x['ChainID'])}/{x['Market ID']}", axis=1
)

st.dataframe(
    df_filtered, 
    column_config={
        "Price (USD)": st.column_config.NumberColumn(format="dollar"),
        "Total Supply (USD)": st.column_config.NumberColumn(format="dollar"),
        "Total Borrow (USD)": st.column_config.NumberColumn(format="dollar"), 
        "Available Liquidity (USD)": st.column_config.NumberColumn(format="dollar"),
        "Supply APY": st.column_config.NumberColumn(format="percent"),  
        "Utilization": st.column_config.NumberColumn(format="percent"), 
        "Link To Market": st.column_config.LinkColumn("Link To Market", display_text="Link")
    },
    column_order=[
        "Market ID", "Chain", "Loan Token", "Collateral", "Price USD", 
        "Supply APY", "Utilization", "Total Supply (USD)", 
        "Total Borrow (USD)", "Available Liquidity (USD)", "Whitelisted", "Link To Market"
    ],
    width="stretch", 
    hide_index=True
)

st.divider()

# --- UI SECTION 2: SCOPE & REBALANCE ---
col_scope, col_param = st.columns([2, 1])

with col_scope:
    st.subheader("2. Your Portfolio Scope")
    
    # --- Wallet Scanner ---
    u_col1, u_col2, u_col3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    
    with u_col1:
        user_wallet = st.text_input("Auto-fill from Wallet Address", placeholder="0x...")
        
    with u_col2:
        scan_clicked = st.button("Import From Wallet", type="primary", use_container_width=True)

    with u_col3:
        clear_all_clicked = st.button("Clear All", type="secondary", use_container_width=True)

    if "portfolio_input_text" not in st.session_state:
        st.session_state.portfolio_input_text = ""

    def handle_text_change():
        """Extracts unique IDs only and reconstructs the UI text with proper headers/labels."""
        raw_text = st.session_state.portfolio_input_text
        if not raw_text.strip():
            return

        # 1. Extract all unique valid Market IDs (lowercase)
        found_ids = []
        seen = set()
        for line in raw_text.replace(',', '\n').split('\n'):
            # Strip comments and links
            potential = line.split('--')[0].strip()
            if not potential: continue
            mid = extract_market_id_from_monarch_link(potential).lower()
            if mid.startswith('0x') and len(mid) > 20 and mid not in seen:
                found_ids.append(mid)
                seen.add(mid)

        # 2. Separate into Manual vs Wallet based on existing cache
        current_text = st.session_state.portfolio_input_text
        
        manual_lines = []
        wallet_lines = []
        
        # Simple parse to preserve section location
        if WALLET_SEP in current_text:
            parts = current_text.split(WALLET_SEP)
            manual_part = parts[0]
            wallet_part = parts[1]
        else:
            manual_part = current_text
            wallet_part = ""

        # Re-enrich manual part
        for line in manual_part.split('\n'):
            clean = line.split('--')[0].strip()
            mid = extract_market_id_from_monarch_link(clean).lower()
            if mid.startswith('0x'):
                row = df_all[df_all['Market ID'].str.lower() == mid]
                label = f" -- {row.iloc[0]['Loan Token']}/{row.iloc[0]['Collateral']}" if not row.empty else ""
                manual_lines.append(f"{mid}{label}")

        # Re-enrich wallet part
        for line in wallet_part.split('\n'):
            clean = line.split('--')[0].strip()
            mid = extract_market_id_from_monarch_link(clean).lower()
            if mid.startswith('0x'):
                row = df_all[df_all['Market ID'].str.lower() == mid]
                label = f" -- {row.iloc[0]['Loan Token']}/{row.iloc[0]['Collateral']}" if not row.empty else ""
                wallet_lines.append(f"{mid}{label}")

        # 3. Reconstruct string
        new_parts = []
        if manual_lines:
            new_parts.append(MANUAL_SEP)
            new_parts.extend(manual_lines)
            new_parts.append("") # Spacer
            
        if wallet_lines:
            new_parts.append(WALLET_SEP)
            new_parts.extend(wallet_lines)

        st.session_state.portfolio_input_text = "\n".join(new_parts).strip()

    if scan_clicked:
        # Case 1: Empty Address -> Remove all wallet markets
        if len(user_wallet) < 10:
            current_text = st.session_state.portfolio_input_text
            if WALLET_SEP in current_text:
                # Keep only manual part
                manual_part = current_text.split(WALLET_SEP)[0].strip()
                st.session_state.portfolio_input_text = manual_part
                st.warning("Address cleared: Wallet markets removed.")
            else:
                st.info("No wallet markets to remove.")
            st.rerun()

        # Case 2: New Address -> Replace wallet markets
        else:
            found_positions = fetch_user_positions(user_wallet)
            
            if found_positions:
                # 1. Update Balances in cache
                st.session_state.balance_cache.update(found_positions)
                new_wallet_ids = {k.lower() for k in found_positions.keys()}
                
                # 2. Preserve Manual Inputs Only
                current_text = st.session_state.portfolio_input_text
                manual_ids = []
                
                # If separator exists, take everything before it
                if WALLET_SEP in current_text:
                    manual_text = current_text.split(WALLET_SEP)[0]
                else:
                    manual_text = current_text
                    
                for line in manual_text.replace(',', '\n').split('\n'):
                    potential = line.split('--')[0].strip()
                    if not potential: continue
                    mid = extract_market_id_from_monarch_link(potential).lower()
                    if mid.startswith('0x') and len(mid) > 20:
                        # Don't add if it's going to be in the new wallet section
                        if mid not in new_wallet_ids:
                            manual_ids.append(mid)

                # 3. Reconstruct with REPLACED wallet section
                final_parts = []
                
                # Add Manual
                if manual_ids:
                    final_parts.append(MANUAL_SEP)
                    # Remove duplicates in manual
                    seen_man = set()
                    for mid in manual_ids:
                        if mid not in seen_man:
                            row = df_all[df_all['Market ID'].str.lower() == mid]
                            label = f" -- {row.iloc[0]['Loan Token']}/{row.iloc[0]['Collateral']}" if not row.empty else ""
                            final_parts.append(f"{mid}{label}")
                            seen_man.add(mid)
                    final_parts.append("")

                # Add New Wallet
                final_parts.append(WALLET_SEP)
                for mid in list(new_wallet_ids):
                    row = df_all[df_all['Market ID'].str.lower() == mid]
                    label = f" -- {row.iloc[0]['Loan Token']}/{row.iloc[0]['Collateral']}" if not row.empty else ""
                    final_parts.append(f"{mid}{label}")

                st.session_state.portfolio_input_text = "\n".join(final_parts).strip()
                st.success(f"Loaded {len(found_positions)} markets from new wallet.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.warning("No active Morpho positions found for this address.")

    if clear_all_clicked:
        st.session_state.portfolio_input_text = ""
        st.session_state.balance_cache = {} # Wipe balance cache
        st.success("All markets and balances cleared.")
        time.sleep(0.5)
        st.rerun()

    # --- Market ID Input ---
    # FIXED: Removed value= argument. Key handles binding.
    raw_ids = st.text_area(
        "Paste Market IDs or Monarch links (one per line)", 
        height=320,
        placeholder="Market IDs or Links...",
        key="portfolio_input_text",
        on_change=handle_text_change 
    )
    
    clean_ids = []
    for line in raw_ids.replace(',', '\n').split('\n'):
        if WALLET_SEP in line or MANUAL_SEP in line: continue
        line_clean = line.split('--')[0].strip()
        if line_clean:
            extracted_id = extract_market_id_from_monarch_link(line_clean)
            if extracted_id and extracted_id not in clean_ids:
                clean_ids.append(extracted_id)
    
    df_selected = df_all[df_all['Market ID'].str.lower().isin(clean_ids)].copy()

with col_param:
    st.subheader("3. Parameters")
    
    # Organize parameters into logical tabs to save vertical space
    tab_scope, tab_dominance, tab_safety = st.tabs(["🌐 Scope & Budget", "🐳 Dominance", "🛡️ Safety"])

    with tab_scope:
        rebalance_scope = st.selectbox(
            "Optimization Constraint",
            options=[
                "1) Full Optimization",
                "2) Within Chain and Same Loan Token",
                "3) Within Chain and Different Loan Token",
                "4) Across Chain and Same Loan Token"
            ],
            index=0,
            help="Strictly limits where funds can move. The optimizer is run independently for each group."
        )

        new_cash = st.number_input(
            "Additional New Cash / Withdrawal (USD)", 
            value=0.0, 
            help="Positive = Add Capital. Negative = Withdraw. (Distributed proportionally to existing silos)."
        )

        min_move_thresh = st.number_input(
            "Min Rebalance Threshold ($)", 
            value=0.0, 
            help='The threshold where the optimizer only allocates if it crosses this amount. Prevents "Dust" moves.'
        )

    with tab_dominance:
        max_supply_dom = st.number_input(
            "Max Supply Dominance %",
            value=10.0,
            help="Constraint: You will never represent more than X% of the market's total supply (including your entry)."
        )

        max_borrow_dom = st.number_input(
            "Max Borrow Dominance %",
            value=10.0,
            help="Constraint: Your supply will never exceed X% of the market's total borrows (Debt pool)."
        )

    with tab_safety:
        max_port_alloc = st.number_input(
            "Max Portfolio Allocation %",
            value=100.0,
            help="Safety limit: No single market can exceed this % of your TOTAL wealth."
        )

        max_dominance = st.number_input(
            "Whale Shield: Max Liquidity %",
            value=30.0,
            help="Liquidity limit: You will never own more than this % of available (unused) liquidity in a pool."
        )

if not df_selected.empty:

    # 1. Ensure current state is mapped to the dataframe
    df_selected['Existing Balance (USD)'] = df_selected['Market ID'].apply(
        lambda x: st.session_state.balance_cache.get(x, 0.0)
    )
    
    # 2. Add Toggle Columns
    if 'Force Exit' not in df_selected:
        df_selected['Force Exit'] = False
    
    # NEW: Split Lock into Outflow/Inflow controls
    if 'Prevent Outflows' not in df_selected:
        df_selected['Prevent Outflows'] = False
    if 'Prevent Inflows' not in df_selected:
        df_selected['Prevent Inflows'] = False

    # Add the URL column
    df_selected['Link To Market'] = df_selected.apply(
        lambda x: f"https://www.monarchlend.xyz/market/{int(x['ChainID'])}/{x['Market ID']}", axis=1
    )

    df_selected = df_selected.sort_values(by='Existing Balance (USD)', ascending=False)

    def sync_portfolio_edits():
        if "portfolio_editor" in st.session_state:
            edits = st.session_state["portfolio_editor"].get("edited_rows", {})
            for idx, changes in edits.items():
                if "Existing Balance (USD)" in changes:
                    m_id = df_selected.iloc[idx]['Market ID']
                    raw_val = changes["Existing Balance (USD)"]
                    try:
                        val = float(raw_val) if raw_val is not None else 0.0
                    except (ValueError, TypeError):
                        val = 0.0
                    st.session_state.balance_cache[m_id] = val

# 1. Visual Tip for the User (Pastel Green Theme)
    st.markdown(
        """
        <div style="background-color: transparent; padding: 10px; border-radius: 5px; border-left: 5px solid #4caf50; margin-bottom: 10px;">
            <strong>ℹ️ Portfolio Tip:</strong> Please fill in your current holdings in the 
            <span style="color: #2e7d32; font-weight: bold;">Existing Balance (USD) 🟢</span> 
            column below if they were not automatically imported from your wallet. Or you can use it to simulate.
        </div>
        """, 
        unsafe_allow_html=True
    )

# 2. Tightened Data Editor
    edited_df = st.data_editor(
        df_selected[[
            'Market ID', 'Chain', 'Loan Token', 'Collateral', 
            'Supply APY', 'Utilization', 'Total Supply (USD)', 
            'Total Borrow (USD)', 'Available Liquidity (USD)', 
            'Existing Balance (USD)', 'Prevent Outflows', 'Prevent Inflows', 'Force Exit', 'Link To Market'
        ]],
        column_config={
            "Market ID": st.column_config.TextColumn(width=100),
            "Chain": st.column_config.TextColumn(width=80),
            "Loan Token": st.column_config.TextColumn(width=80),
            "Collateral": st.column_config.TextColumn(width=80),
            "Supply APY": st.column_config.NumberColumn(format="percent", width=70),
            "Utilization": st.column_config.NumberColumn(format="percent", width=70),
            "Total Supply (USD)": st.column_config.NumberColumn(format="dollar", width=100),
            "Total Borrow (USD)": st.column_config.NumberColumn(format="dollar", width=100),
            "Available Liquidity (USD)": st.column_config.NumberColumn(format="dollar", width=100),
            "Existing Balance (USD)": st.column_config.NumberColumn(
                "Existing Balance (USD) 🟢", 
                help="Fill in your current market balance here. The optimizer uses this to calculate rebalance moves.",
                format="dollar", 
                min_value=0.0,
                width=150
            ),
            "Prevent Outflows": st.column_config.CheckboxColumn(
                "No Outflows 🔒", 
                help="If checked, the optimizer will NEVER sell or withdraw funds from this market (Hold or Buy only).", 
                default=False, 
                width=90
            ),
            "Prevent Inflows": st.column_config.CheckboxColumn(
                "No Inflows 🛑", 
                help="If checked, the optimizer will NEVER add new funds to this market (Hold or Sell only).", 
                default=False, 
                width=90
            ),
            "Force Exit": st.column_config.CheckboxColumn("Force Exit?", help="Check to forcefully sell 100% of this position.", default=False, width=80),
            "Link To Market": st.column_config.LinkColumn("Link To Market", display_text="Link", width=80)
        },
        disabled=[
            'Market ID', 'Chain', 'Loan Token', 'Collateral', 
            'Supply APY', 'Utilization', 'Total Supply (USD)', 
            'Total Borrow (USD)', 'Available Liquidity (USD)', 'Link To Market'
        ],
        width='stretch', 
        hide_index=True, 
        key="portfolio_editor",
        on_change=sync_portfolio_edits,
        column_order=[
            'Market ID', 'Chain', 'Loan Token', 'Collateral', 
            'Supply APY', 'Utilization', 'Total Supply (USD)', 
            'Total Borrow (USD)', 'Available Liquidity (USD)', 
            'Existing Balance (USD)', 'Prevent Outflows', 'Prevent Inflows', 'Force Exit', 'Link To Market'
        ]
    )
    
    # Capture UI states
    force_exit_map = dict(zip(edited_df['Market ID'], edited_df['Force Exit']))
    # Update maps for new columns
    prevent_out_map = dict(zip(edited_df['Market ID'], edited_df['Prevent Outflows']))
    prevent_in_map = dict(zip(edited_df['Market ID'], edited_df['Prevent Inflows']))

    current_wealth = sum(st.session_state.balance_cache.get(m_id, 0.0) for m_id in df_selected['Market ID'])
    total_optimizable = current_wealth + new_cash

    # ==========================================
    # LIVE METRICS & OPTIMIZATION LOGIC
    # ==========================================
    st.divider()
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("💰 Total Current Balance", f"${current_wealth:,.2f}")
    
    # Logic to display Withdrawal vs Addition
    if new_cash < 0:
        metric_col2.metric("📤 Withdrawing", f"-${abs(new_cash):,.2f}")
    else:
        metric_col2.metric("💵 Additional cash", f"${new_cash:,.2f}")
        
    metric_col3.metric("📊 Target Portfolio Size", f"${total_optimizable:,.2f}")
    
    if total_optimizable < 0:
        st.error("⚠️ Withdrawal amount exceeds total current wealth! Please adjust.")
        st.stop()
    
    st.divider()

    with st.expander("ℹ️ About the Portfolio Strategies", expanded=True):
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        
        with col_info1:
            st.markdown("🔴 **Best Yield**")
            st.caption(
                "**Aggressive.** Purely maximizes mathematical APY. "
                "Often results in 100% allocation to the single highest payer, ignoring risk."
            )
            
        with col_info2:
            st.markdown("🔵 **Whale Shield**")
            st.caption(
                f"**Protective.** Maximizes yield but adds a hard constraint: "
                f"You will never own more than **{max_dominance}%** of a pool's **Available Liquidity**. "
                "Prevents you from getting stuck in markets with high Total Supply but 0 withdrawals available."
            )
            
        with col_info3:
            st.markdown("🌸 **Frontier**")
            st.caption(
                "**Balanced.** Finds the mathematical 'sweet spot' (Pareto Efficiency) between "
                "maximizing raw yield and maximizing the Diversity Score (1 - HHI)."
            )

        with col_info4:
            st.markdown("🟢 **Liquid-Yield**")
            st.caption(
                "**Depth-First.** Prioritizes markets with deep available liquidity. "
                "It sacrifices a small amount of APY to ensure you can exit large positions easily without slippage."
            )

    # --- FIND: "if st.button("🚀 Run Optimization"..." ---
    if st.button("🚀 Run Optimization", type="primary", width='stretch'):
        # FIX: Force a final sync of the portfolio editor before calculating
        sync_portfolio_edits()

        # 1. Clear State
        if 'hist_demand_df' in st.session_state:
            del st.session_state.hist_demand_df
        
        # Reset the demand trend trigger so it doesn't auto-run on new results
        st.session_state.run_demand_trend = False
        
        # 2. Prepare Data
        market_data_list = fetch_live_market_details(df_selected)
        
        # Inject user data and UI flags
        for m in market_data_list:
            m['existing_balance_usd'] = st.session_state.balance_cache.get(m['Market ID'], 0.0)
            m['force_exit'] = force_exit_map.get(m['Market ID'], False)
            # Inject new split constraints
            m['prevent_outflows'] = prevent_out_map.get(m['Market ID'], False)
            m['prevent_inflows'] = prevent_in_map.get(m['Market ID'], False)

        # Global Metrics for reference
        current_annual_interest = sum(m['existing_balance_usd'] * m['current_supply_apy'] for m in market_data_list)
        current_blended_apy = current_annual_interest / current_wealth if current_wealth > 0 else 0.0

        # 3. SILO LOGIC: Group the data BEFORE optimization
        silos = {}
        for idx, m in enumerate(market_data_list):
            # Define the "Key" that makes a market unique based on constraints
            if "1)" in rebalance_scope:
                key = "GLOBAL"
            elif "2)" in rebalance_scope:
                # Strictly same chain AND same asset
                key = (m['ChainID'], m['Loan Address']) 
            elif "3)" in rebalance_scope:
                # Same chain, any asset (Swapping allowed)
                key = m['ChainID']
            elif "4)" in rebalance_scope:
                # Any chain, same asset (Bridging allowed)
                key = m['Loan Address']
            
            if key not in silos: silos[key] = []
            silos[key].append(idx)

        # 4. Initialize Global Result Arrays (Same size as original list)
        n = len(market_data_list)
        final_y = np.zeros(n)
        final_f = np.zeros(n)
        final_l = np.zeros(n)
        final_w = np.zeros(n)
        
        # Trace accumulators
        global_yield_trace = []
        global_frontier_trace = []
        global_liquid_trace = []
        global_whale_trace = []
        global_attempts = []
        global_capacity_warnings = []

        # 5. EXECUTION LOOP: Run Optimizer for each Silo independently
        progress_text = "Optimizing Silos..."
        my_bar = st.progress(0, text=progress_text)
        silo_count = len(silos)
        
        # MATH FIX: Calculate the Global Portfolio Cap in USD first
        # This ensures safety limits are based on your TOTAL wealth, not the silo slice.
        global_port_cap_usd = total_optimizable * (max_port_alloc / 100.0)

        for i, (key, indices) in enumerate(silos.items()):
            # Subset the markets
            silo_markets = [market_data_list[idx] for idx in indices]
            
            # Calculate Silo Budget
            silo_existing_wealth = sum(m['existing_balance_usd'] for m in silo_markets)
            
            # FIX: Handle new users where current_wealth is 0
            if current_wealth > 0:
                share_of_port = silo_existing_wealth / current_wealth
                silo_budget = silo_existing_wealth + (new_cash * share_of_port)
            else:
                # If starting from scratch, distribute new cash equally among selected silos
                silo_budget = new_cash / len(silos)
            
            # Safety: Prevent negative budgets or tiny dust silos
            if silo_budget <= 0.01:
                my_bar.progress((i + 1) / silo_count, text=f"Skipping empty silo {key}...")
                continue
                
            # MATH FIX: Adjust the percentage for this specific silo
            # We want: (Silo_Budget * Adjusted_Pct) = Global_Cap_USD
            # Therefore: Adjusted_Pct = (Global_Cap_USD / Silo_Budget)
            # We multiply by 100 because the Optimizer expects a percentage input (0-100).
            if silo_budget > 0:
                adjusted_max_alloc_pct = (global_port_cap_usd / silo_budget) * 100.0
            else:
                adjusted_max_alloc_pct = 100.0

            # RUN OPTIMIZER ON SUBSET with ADJUSTED CAP and NEW DOMINANCE LIMITS
            opt = RebalanceOptimizer(
                total_budget=silo_budget, 
                market_list=silo_markets, 
                max_dominance_pct=max_dominance, 
                max_port_alloc_pct=adjusted_max_alloc_pct,
                max_supply_pct=max_supply_dom,
                max_borrow_pct=max_borrow_dom
            )
            r_y, r_f, r_l, r_w = opt.optimize()

            # Map results back to the Global Indices
            for local_idx, global_idx in enumerate(indices):
                final_y[global_idx] = r_y[local_idx]
                final_f[global_idx] = r_f[local_idx]
                final_l[global_idx] = r_l[local_idx]
                final_w[global_idx] = r_w[local_idx]

            # Aggregate metadata for charts
            global_yield_trace.extend(opt.yield_trace)
            global_frontier_trace.extend(opt.frontier_trace)
            global_liquid_trace.extend(opt.liquid_trace)
            global_whale_trace.extend(opt.whale_trace)
            global_attempts.extend(opt.all_attempts)
            if opt.capacity_warning:
                # Clarify which silo triggered the warning
                global_capacity_warnings.append(f"Silo [{key}]: {opt.capacity_warning}")
            
            my_bar.progress((i + 1) / silo_count, text=f"Optimized Silo: {key}")
        
        my_bar.empty()

        # 6. Apply Final Dust Filters to the Globals
        cleaned_yield = filter_small_moves(final_y, market_data_list, min_move_thresh, total_optimizable)
        cleaned_frontier = filter_small_moves(final_f, market_data_list, min_move_thresh, total_optimizable)
        cleaned_liquid = filter_small_moves(final_l, market_data_list, min_move_thresh, total_optimizable)
        cleaned_whale = filter_small_moves(final_w, market_data_list, min_move_thresh, total_optimizable)
        
        # 7. Store Results
        # We create a dummy optimizer object just to hold the traces/warnings for the UI to read
        dummy_opt = RebalanceOptimizer(total_optimizable, market_data_list, max_dominance, max_port_alloc)
        dummy_opt.yield_trace = global_yield_trace
        dummy_opt.frontier_trace = global_frontier_trace
        dummy_opt.liquid_trace = global_liquid_trace
        dummy_opt.whale_trace = global_whale_trace
        dummy_opt.all_attempts = global_attempts
        dummy_opt.capacity_warning = " | ".join(global_capacity_warnings) if global_capacity_warnings else None

        st.session_state['opt_results'] = {
            'market_data_list': market_data_list,
            'opt_object': dummy_opt, 
            'best_yield_alloc': cleaned_yield,
            'frontier_alloc': cleaned_frontier,
            'liquid_alloc': cleaned_liquid,
            'whale_alloc': cleaned_whale, 
            'current_metrics': {
                'annual_interest': current_annual_interest,
                'blended_apy': current_blended_apy
            },
            'traces': {
                'yield': global_yield_trace,
                'frontier': global_frontier_trace,
                'liquid': global_liquid_trace,
                'whale': global_whale_trace
            },
            'attempts': global_attempts
        }

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================
    if 'opt_results' in st.session_state and st.session_state['opt_results']['best_yield_alloc'] is not None:
        
        res_data = st.session_state['opt_results']
        opt = res_data['opt_object'] 
        market_data_list = res_data['market_data_list']
        best_yield_alloc = res_data['best_yield_alloc']
        frontier_alloc = res_data['frontier_alloc']
        liquid_alloc = res_data['liquid_alloc']
        whale_alloc = res_data['whale_alloc'] 
        
        # --- [SECTION START: METRIC SUMMARY & STATS] ---
        def get_stats(alloc):
            y = 0
            for i, val in enumerate(alloc):
                y += (val * opt.simulate_apy(market_data_list[i], val))
            # Honest APY: We always divide by total_optimizable budget 
            # This correctly accounts for unallocated cash in strict Whale Shield mode.
            weights = alloc / total_optimizable if total_optimizable > 0 else np.zeros_like(alloc)
            div = 1.0 - np.sum(weights**2)
            return y, y / total_optimizable if total_optimizable > 0 else 0, div

        y_abs, y_apy, y_div = get_stats(best_yield_alloc)
        w_abs, w_apy, w_div = get_stats(whale_alloc)
        f_abs, f_apy, f_div = get_stats(frontier_alloc)
        l_abs, l_apy, l_div = get_stats(liquid_alloc)

        # 5-Column High-Level Summary
        st.subheader("📋 Strategy Comparison")
        
        # Extract current APY early to use as reference for deltas
        current_blended = res_data['current_metrics']['blended_apy']
        
        s_col0, s_col1, s_col2, s_col3, s_col4 = st.columns(5)
        
        with s_col0:
            st.markdown("<h4 style='color:#9E9E9E; margin-bottom:0;'>Current</h4>", unsafe_allow_html=True)
            st.metric("APY", f"{current_blended:.2%}")
            st.caption("Baseline")
        
        with s_col1:
            st.markdown("<h4 style='color:#F44336; margin-bottom:0;'>Best Yield</h4>", unsafe_allow_html=True)
            st.metric("APY", f"{y_apy:.2%}", delta=f"{(y_apy - current_blended):.2%}")
            st.caption(f"Diversity: **{y_div:.4f}**")

        with s_col2:
            st.markdown("<h4 style='color:#2979FF; margin-bottom:0;'>Whale Shield</h4>", unsafe_allow_html=True)
            st.metric("APY", f"{w_apy:.2%}", delta=f"{(w_apy - current_blended):.2%}")
            st.caption(f"Diversity: **{w_div:.4f}**")

        with s_col3:
            st.markdown("<h4 style='color:#E040FB; margin-bottom:0;'>Frontier</h4>", unsafe_allow_html=True)
            st.metric("APY", f"{f_apy:.2%}", delta=f"{(f_apy - current_blended):.2%}")
            st.caption(f"Diversity: **{f_div:.4f}**")

        with s_col4:
            st.markdown("<h4 style='color:#00E676; margin-bottom:0;'>Liquid-Yield</h4>", unsafe_allow_html=True)
            st.metric("APY", f"{l_apy:.2%}", delta=f"{(l_apy - current_blended):.2%}")
            st.caption(f"Diversity: **{l_div:.4f}**")

        st.divider()
        
        df_scatter = pd.DataFrame(res_data['attempts'])
        # Downsample scatter plot points if too many, to prevent UI lag
        if len(df_scatter) > MAX_SCATTER_PLOT_POINTS:
            df_scatter = df_scatter.sample(MAX_SCATTER_PLOT_POINTS, random_state=42)
        
        # Updated Color Mapping
        STRAT_DOMAIN = ['Best Yield', 'Whale Shield', 'Frontier', 'Liquid-Yield']
        STRAT_RANGE = ['#F44336', '#2979FF', '#E040FB', '#00E676'] 
        
        st.subheader("📊 Optimization Search Space")
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            st.markdown("**Efficiency Frontier**")
            
            highlights = pd.DataFrame([
                {"Diversity Score": y_div, "Blended APY": y_apy, "Type": "Best Yield", "Size": 100},
                {"Diversity Score": w_div, "Blended APY": w_apy, "Type": "Whale Shield", "Size": 100}, 
                {"Diversity Score": f_div, "Blended APY": f_apy, "Type": "Frontier", "Size": 100},
                {"Diversity Score": l_div, "Blended APY": l_apy, "Type": "Liquid-Yield", "Size": 100}
            ])
            
            base = alt.Chart(df_scatter).mark_circle(opacity=0.3, color='#80DEEA').encode(
                x=alt.X('Diversity Score', title='Diversity (1 - HHI)'),
                y=alt.Y('Blended APY', title='APY', axis=alt.Axis(format='%')),
                tooltip=['Diversity Score', 'Blended APY', 'Annual Yield ($)']
            )
            
            points = alt.Chart(highlights).mark_circle(size=200, opacity=1.0).encode(
                x='Diversity Score',
                y='Blended APY',
                color=alt.Color('Type', scale=alt.Scale(domain=STRAT_DOMAIN, range=STRAT_RANGE)),
                tooltip=['Type', 'Blended APY', 'Diversity Score']
            )
            
            if not df_scatter.empty and len(df_scatter['Type'].unique()) > 0:
                base = alt.Chart(df_scatter).mark_circle(opacity=0.3, color='#80DEEA').encode(
                    x=alt.X('Diversity Score', title='Diversity (1 - HHI)'),
                    y=alt.Y('Blended APY', title='APY', axis=alt.Axis(format='%')),
                    tooltip=['Diversity Score', 'Blended APY', 'Annual Yield ($)']
                )
                
                points = alt.Chart(highlights).mark_circle(size=200, opacity=1.0).encode(
                    x='Diversity Score',
                    y='Blended APY',
                    color=alt.Color('Type', scale=alt.Scale(domain=STRAT_DOMAIN, range=STRAT_RANGE)),
                    tooltip=['Type', 'Blended APY', 'Diversity Score']
                )
                
                st.altair_chart(base + points, width='stretch')
            else:
                st.info("Not enough data to generate optimization charts. Please add capital or select more markets.")

        with col_graph2:
            st.markdown("**Solver Convergence**")
            traces = res_data['traces']
            
            d1 = pd.DataFrame({"Iteration": range(len(traces['yield'])), "Value": traces['yield'], "Strategy": "Best Yield"})
            d2 = pd.DataFrame({"Iteration": range(len(traces['frontier'])), "Value": traces['frontier'], "Strategy": "Frontier"})
            d4 = pd.DataFrame({"Iteration": range(len(traces['liquid'])), "Value": traces['liquid'], "Strategy": "Liquid-Yield"})
            d5 = pd.DataFrame({"Iteration": range(len(traces['whale'])), "Value": traces['whale'], "Strategy": "Whale Shield"})
            
            # Downsample each trace if it has too many points, THEN SORT
            if len(d1) > MAX_LINE_PLOT_POINTS_PER_STRATEGY: 
                d1 = d1.sample(MAX_LINE_PLOT_POINTS_PER_STRATEGY, random_state=42).sort_values('Iteration')
            else:
                d1 = d1.sort_values('Iteration')

            if len(d2) > MAX_LINE_PLOT_POINTS_PER_STRATEGY: 
                d2 = d2.sample(MAX_LINE_PLOT_POINTS_PER_STRATEGY, random_state=43).sort_values('Iteration')
            else:
                d2 = d2.sort_values('Iteration')

            if len(d4) > MAX_LINE_PLOT_POINTS_PER_STRATEGY: 
                d4 = d4.sample(MAX_LINE_PLOT_POINTS_PER_STRATEGY, random_state=45).sort_values('Iteration')
            else:
                d4 = d4.sort_values('Iteration')

            if len(d5) > MAX_LINE_PLOT_POINTS_PER_STRATEGY: 
                d5 = d5.sample(MAX_LINE_PLOT_POINTS_PER_STRATEGY, random_state=46).sort_values('Iteration')
            else:
                d5 = d5.sort_values('Iteration')
            
            df_hist_long = pd.concat([d1, d2, d4, d5])
            
            line_chart = alt.Chart(df_hist_long).mark_line().encode(
                x='Iteration',
                y=alt.Y('Value', title='Objective Yield ($)'),
                color=alt.Color('Strategy', scale=alt.Scale(domain=STRAT_DOMAIN, range=STRAT_RANGE))
            )
            st.altair_chart(line_chart, width='stretch')

        st.divider()
        
        # --- FULL WIDTH: ALLOCATION CHART ---
        st.subheader("⚖️ Allocation Comparison")
        
        bar_data = []
        for idx, m in enumerate(market_data_list):
            # Apply 6-char suffix here as well
            short_id = m['Market ID'][0:6]
            m_name = f"{m['Loan Token']}/{m['Collateral']} ({short_id})"
            
            y_val = best_yield_alloc[idx]
            w_val = whale_alloc[idx]
            f_val = frontier_alloc[idx]
            l_val = liquid_alloc[idx]
            
            if max(y_val, w_val, f_val, l_val) > 1:
                bar_data.append({"Market": m_name, "Strategy": "Best Yield", "Alloc ($)": y_val})
                bar_data.append({"Market": m_name, "Strategy": "Whale Shield", "Alloc ($)": w_val})
                bar_data.append({"Market": m_name, "Strategy": "Frontier", "Alloc ($)": f_val})
                bar_data.append({"Market": m_name, "Strategy": "Liquid-Yield", "Alloc ($)": l_val})
        
        if bar_data:
            df_bar = pd.DataFrame(bar_data)
            bar_chart = alt.Chart(df_bar).mark_bar().encode(
                x=alt.X('Market:N', title="Market Pair (ID)", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Alloc ($):Q', title="Allocation (USD)", scale=alt.Scale(zero=True)),
                xOffset='Strategy:N',
                color=alt.Color('Strategy:N', scale=alt.Scale(domain=STRAT_DOMAIN, range=STRAT_RANGE)),
                tooltip=['Market', 'Strategy', alt.Tooltip('Alloc ($)', format='$,.2f')]
            ).properties(height=400).configure_view(stroke=None) # Increased height
            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.info("No significant allocations.")

        st.divider()

        # --- FULL WIDTH: DEMAND TRENDS CHART ---
        st.subheader("📈 30-Day Balanced Demand Trend")
        st.caption("Growth of the square root of Supply x Borrow relative to start of period. Rewards balanced market synergy and penalizes ghost/diluted pools.")

        # Initialize trigger state
        if 'run_demand_trend' not in st.session_state:
            st.session_state.run_demand_trend = False

        # Red Primary Button for Trend Analysis
        if not st.session_state.run_demand_trend:
            if st.button("🔍 Run 30-Day Demand Trend Analysis", type="primary", use_container_width=True):
                st.session_state.run_demand_trend = True
                st.rerun()
        
        if st.session_state.run_demand_trend:
            # 1. Prepare list for fetching
            targets_for_history = []
            for m in market_data_list:
                targets_for_history.append({
                    "uniqueKey": m['Market ID'],
                    "chainId": m['ChainID'],
                    "name": f"{m['Loan Token']}/{m['Collateral']}"
                })
            
            # 2. Fetch Data (Cached)
            if 'hist_demand_df' not in st.session_state:
                st.session_state.hist_demand_df = fetch_historical_flows(targets_for_history)
            
            df_history = st.session_state.hist_demand_df
            
            if not df_history.empty:
                # --- FILTER LOGIC ---
                trend_summary = df_history.sort_values('date').groupby('Market')['indexed_demand'].last()
                
                d_col_empty, d_col_filter = st.columns([2, 1])
                with d_col_filter:
                    filter_mode = st.radio(
                        "Filter Trends:",
                        ["All", "Growth (> 1.0)", "Decay (≤ 1.0)"],
                        horizontal=True,
                        key="trend_filter_radio"
                    )

                # Apply Filter
                markets_to_show = trend_summary.index.tolist()
                
                if "Growth" in filter_mode:
                    markets_to_show = trend_summary[trend_summary > 1.0].index.tolist()
                elif "Decay" in filter_mode:
                    markets_to_show = trend_summary[trend_summary <= 1.0].index.tolist()
                
                df_hist_filtered = df_history[df_history['Market'].isin(markets_to_show)]

                if not df_hist_filtered.empty:
                    # --- INTERACTIVE ELEMENTS ---
                    selection = alt.selection_point(fields=['Market'], bind='legend')
                    hover = alt.selection_point(
                        fields=['date'], 
                        nearest=True, 
                        on='mouseover', 
                        empty=False, 
                        clear='mouseout'
                    )

                    # 3. Base Line Chart
                    base = alt.Chart(df_hist_filtered).encode(
                        x=alt.X('date:T', title=None, axis=alt.Axis(format='%b %d')),
                        y=alt.Y('indexed_demand:Q', 
                                title='Growth Index', 
                                axis=alt.Axis(format='.2f', grid=True),
                                scale=alt.Scale(zero=False)), 
                        color=alt.Color('Market:N', legend=alt.Legend(
                            orient='bottom', 
                            title='Market ID & Pair',
                            columns=4,
                            labelLimit=0,
                            symbolLimit=100,
                            padding=10
                        ))
                    )

                    lines = base.mark_line(interpolate='monotone').encode(
                        opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
                        strokeWidth=alt.condition(selection, alt.value(2.5), alt.value(1))
                    ).add_params(selection)

                    selectors = base.mark_point().encode(
                        opacity=alt.value(0),
                        tooltip=[
                            alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                            alt.Tooltip('Market:N', title='Market'),
                            alt.Tooltip('indexed_demand:Q', title='Index Value', format='.3f')
                        ]
                    ).add_params(hover)

                    points = base.mark_point().encode(
                        opacity=alt.condition(hover, alt.value(1), alt.value(0))
                    )

                    rules = base.mark_rule(color='white', strokeWidth=0.5).encode(
                        opacity=alt.condition(hover, alt.value(0.3), alt.value(0))
                    ).transform_filter(hover)

                    # Updated to use global height constant
                    final_chart = alt.layer(
                        lines, selectors, points, rules
                    ).properties(height=DEMAND_TREND_CHART_HEIGHT).interactive()
                    
                    st.altair_chart(final_chart, use_container_width=True)
                else:
                    st.info(f"No markets found matching filter: {filter_mode}")
            else:
                st.info("No historical data available for selected markets.")

        st.divider()
        st.subheader("🔍 Results")

        strategy_choice = st.radio(
            "View Details For:",
            [
                "Current Portfolio",
                "Best Yield (Red)", 
                "Whale Shield (Blue)", 
                "Frontier (Magenta)", 
                "Liquid-Yield (Green)"
            ],
            index=1,
            horizontal=True
        )
        
        if "Best Yield" in strategy_choice:
            final_alloc = best_yield_alloc
        elif "Whale" in strategy_choice:
            final_alloc = whale_alloc
            if hasattr(res_data['opt_object'], 'capacity_warning') and res_data['opt_object'].capacity_warning:
                st.warning(res_data['opt_object'].capacity_warning)
        elif "Frontier" in strategy_choice:
            final_alloc = frontier_alloc
        elif "Liquid" in strategy_choice:
            final_alloc = liquid_alloc
        else:
            final_alloc = np.array([m['existing_balance_usd'] for m in market_data_list])

        results = []
        theoretical_annual_interest = 0
        realized_annual_interest = 0
        realized_annual_apr_interest = 0
        total_allocated_usd = 0.0
        total_stuck_usd = 0.0
        
        for i, target_val in enumerate(final_alloc):
            m = market_data_list[i]
            total_allocated_usd += target_val
            
            # --- BASIC METADATA ---
            token_price = m['Price USD']
            multiplier = 10**m['Decimals']
            user_current = m['existing_balance_usd']
            current_avail_usd = m.get('Available Liquidity (USD)', 0.0)
            current_apy_val = m.get('current_supply_apy', 0.0)
            
            # --- 1. LIQUIDITY & REALIZATION LOGIC ---
            # Determine how much we actually moved
            net_move_usd = target_val - user_current
            requested_withdrawal = max(0, -net_move_usd)
            
            # If we want to withdraw more than available, we are stuck
            stuck_funds_usd = max(0, requested_withdrawal - current_avail_usd)
            actual_withdrawable_usd = requested_withdrawal - stuck_funds_usd
            
            # Realized move is the actual deposit OR the allowed withdrawal
            liquid_move_usd = -actual_withdrawable_usd if net_move_usd < 0 else net_move_usd
            realized_target_val = user_current + liquid_move_usd
            total_stuck_usd += stuck_funds_usd
            
            # --- 2. FINAL STATE CALCULATIONS (BASED ON REALIZED BALANCE) ---
            # Base Supply = Total Supply - User's Current Holdings
            user_existing_wei = (user_current / token_price) * multiplier if token_price > 0 else 0
            base_supply_wei = max(0, m['raw_supply'] - user_existing_wei)
            
            # Realized State: Uses the balance we actually HAVE in the market
            realized_target_wei = (realized_target_val / token_price) * multiplier if token_price > 0 else 0
            simulated_realized_supply_wei = base_supply_wei + realized_target_wei
            final_util = m['raw_borrow'] / simulated_realized_supply_wei if simulated_realized_supply_wei > 0 else 0
            
            # Final Liquidity = What's there now + our change in balance
            # (If we are stuck, liquid_move_usd is 0, so Final Liq == Initial Liq)
            base_available_usd = max(0, current_avail_usd - user_current)
            final_available_usd = base_available_usd + realized_target_val
            liq_share = realized_target_val / final_available_usd if final_available_usd > 0 else 0
            
            # --- 3. APY & APR (REALIZED) ---
            # If no move happened (stuck or hold), rate is exactly current market rate
            if abs(liquid_move_usd) < 0.01:
                realized_apy = current_apy_val
            else:
                realized_apy = opt.simulate_apy(m, realized_target_val)
            
            realized_rate_sec = apy_to_rate_per_second(realized_apy)
            realized_apr = realized_rate_sec * SECONDS_PER_YEAR
            
            # --- 4. TARGET STATE (THEORETICAL FOR OPTIMIZER CONTEXT) ---
            # We keep this so the user can see what the Optimizer "Wanted"
            target_apy = current_apy_val if abs(net_move_usd) < 0.01 else opt.simulate_apy(m, target_val)
            
            # --- 5. AGGREGATES & ACTION ---
            if net_move_usd > 0.01:
                action = "🟢 DEPOSIT"
            elif net_move_usd < -0.01:
                action = "⚠️ STUCK" if actual_withdrawable_usd <= 0.01 else "🔴 WITHDRAW"
            else:
                action = "⚪ HOLD"

            current_contrib = (user_current / total_optimizable * current_apy_val) if total_optimizable > 0 else 0
            # Target contribution uses theoretical APY
            selected_contrib = (target_val / total_optimizable * target_apy) if total_optimizable > 0 else 0
            net_apy_impact = selected_contrib - current_contrib
            
            theoretical_annual_interest += (target_val * target_apy)
            
            # Interest derived from what actually happened
            realized_interest_item = (realized_target_val * realized_apy)
            realized_annual_interest += realized_interest_item
            realized_annual_apr_interest += (realized_target_val * realized_apr)

            # --- 6. CONSTRUCT DICTIONARY ---
            results.append({
                "Destination ID": str(m['Market ID'])[:7],
                "Market": f"{m['Loan Token']}/{m['Collateral']}",
                "Token": m['Loan Token'],
                "Chain": m['Chain'], 
                "Action": action,
                "Weight": target_val / total_optimizable if total_optimizable > 0 else 0,
                "Net APY Impact": net_apy_impact,
                "Contribution (Target)": selected_contrib,
                "Contribution (Current)": current_contrib,
                "Current ($)": user_current,
                "Target ($)": target_val,
                "Realized ($)": realized_target_val,
                "Net Move ($)": net_move_usd,
                "Current APY": current_apy_val,
                "Simulated APY": target_apy, # Theoretical goal
                "Simulated APR": realized_apr, # Realized reality
                "Initial Utilization": m['raw_borrow'] / m['raw_supply'] if m['raw_supply'] > 0 else 0,
                "Final Utilization": final_util, # Realized reality
                "Liquid Move ($)": liquid_move_usd,
                "Stuck Funds ($)": stuck_funds_usd,
                "Ann. Yield": target_val * target_apy, # Theoretical goal
                "Ann. Yield (Realized)": realized_interest_item, # Realized reality
                "Initial Liq.": current_avail_usd,
                "Final Liq.": final_available_usd, # Realized reality
                "% Liq. Share": liq_share,
                "Market ID Full": m['Market ID'],
                "Loan Address": m.get('Loan Address'),
                "Price USD": m.get('Price USD'),
                "Decimals": m.get('Decimals'),
                "ChainID": m['ChainID'],
                "Link To Market": f"https://www.monarchlend.xyz/market/{int(m['ChainID'])}/{m['Market ID']}"
            })

        # 6. Check for Unallocated Cash
        unallocated_cash = total_optimizable - total_allocated_usd
        if unallocated_cash > 0.01:
            results.append({
                "Market": "⚠️ Unallocated Cash", "Token": "CASH", "Chain": "Wallet", "Action": "⚪ HOLD",
                "Weight": unallocated_cash / total_optimizable if total_optimizable > 0 else 0,
                "Net APY Impact": 0.0, "Contribution (Target)": 0.0, "Contribution (Current)": 0.0,
                "Current ($)": 0.0, "Target ($)": unallocated_cash, "Realized ($)": unallocated_cash,
                "Net Move ($)": 0.0, "Liquid Move ($)": 0.0, "Current APY": 0.0, "Simulated APY": 0.0, "Simulated APR": 0.0, 
                "Ann. Yield": 0.0, "Ann. Yield (Realized)": 0.0, "Stuck Funds ($)": 0.0,
                "Initial Utilization": 0.0, "Final Utilization": 0.0, "Initial Liq.": 0.0, "Final Liq.": 0.0, "% Liq. Share": 0.0
            })
        
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by=["Net APY Impact", "Weight"], ascending=[False, False])
        
        # Define legacy aliases for downstream calculations
        new_annual_interest = realized_annual_interest
        
        # Metric Calculations
        current_blended = res_data['current_metrics']['blended_apy']
        realized_blended_apy = realized_annual_interest / total_optimizable if total_optimizable > 0 else 0.0
        theoretical_blended_apy = theoretical_annual_interest / total_optimizable if total_optimizable > 0 else 0.0
        
        # UI Header
        m1, m2, m3, m4, m5, m6 = st.columns(6) 
        m1.metric("Current APY", f"{current_blended:.4%}")
        
        m2.metric(
            "Realized APY", 
            f"{realized_blended_apy:.4%}", 
            delta=f"{(realized_blended_apy - current_blended):.4%}",
            help=f"Theoretical Target APY: {theoretical_blended_apy:.4%}. Realized APY accounts for funds currently stuck due to market liquidity."
        )
        
        m3.metric("Total Wealth (1 Yr)", f"${total_optimizable + realized_annual_interest:,.2f}")
        
        m4.metric(
            "Annual Interest Gain", 
            f"${(realized_annual_interest - res_data['current_metrics']['annual_interest']):,.2f}",
            help=f"Theoretical Max Gain: ${(theoretical_annual_interest - res_data['current_metrics']['annual_interest']):,.2f}"
        )
        
        selected_weights = np.array([r['Weight'] for r in results if 'Weight' in r])
        selected_diversity = 1.0 - np.sum(selected_weights**2)
        m5.metric("Diversity Score", f"{selected_diversity:.4f}")
        
        m6.metric("Stuck Capital", f"${total_stuck_usd:,.2f}", delta_color="inverse", delta="Liquidity Issue" if total_stuck_usd > 0 else None)

        # Row 2: Time-Based Earnings Breakdown
        st.markdown("---")
        st.subheader("📈 Projected Earnings")
        
        calc_basis = st.radio(
            "Rate Calculation Basis:",
            ["Compounded (APY)", "Linear (APR)"],
            index=1, 
            horizontal=True,
            help="Linear (APR) is more conservative and accurate for short-term (daily/hourly) cashflow."
        )

        if "Linear" in calc_basis:
            active_interest = realized_annual_apr_interest
            label_suffix = "(APR)"
        else:
            active_interest = realized_annual_interest
            label_suffix = "(APY)"

        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric(f"Annual {label_suffix}", f"${active_interest:,.2f}")
        t2.metric(f"Monthly {label_suffix}", f"${active_interest/12:,.2f}")
        t3.metric(f"Weekly {label_suffix}", f"${active_interest/52:,.2f}")
        t4.metric(f"Daily {label_suffix}", f"${active_interest/365:,.2f}")
        t5.metric(f"Hourly {label_suffix}", f"${active_interest/8760:,.4f}")

        st.divider()
        st.subheader("⚖️ Allocations")

        def style_impact(val):
            color = '#00E676' if val > 0.000001 else '#FF5252' if val < -0.000001 else '#9E9E9E'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            df_res.style.format({
                "Weight": "{:.2%}", "Net APY Impact": "{:+.4%}",
                "Contribution (Target)": "{:.4%}", "Contribution (Current)": "{:.4%}",
                "Current ($)": "${:,.2f}", "Target ($)": "${:,.2f}", "Realized ($)": "${:,.2f}",
                "Net Move ($)": "${:,.2f}", "Liquid Move ($)": "${:,.2f}", "Stuck Funds ($)": "${:,.2f}",
                "Current APY": "{:.4%}", "Simulated APY": "{:.4%}", "Simulated APR": "{:.4%}",
                "Ann. Yield": "${:,.2f}", "Ann. Yield (Realized)": "${:,.2f}",
                "Initial Utilization": "{:.2%}", "Final Utilization": "{:.2%}",
                "Initial Liq.": "${:,.2f}", "Final Liq.": "${:,.2f}", "% Liq. Share": "{:.2%}"
            }).applymap(style_impact, subset=['Net APY Impact']), 
            column_config={
                "Net APY Impact": st.column_config.NumberColumn("Net APY Impact 🚀", help="Change to total Portfolio APY."),
                "Link To Market": st.column_config.LinkColumn("Link To Market", display_text="Link")
            },
            column_order=[
                "Destination ID", "Market", "Chain", "Action", "Weight", 
                "Net APY Impact", "Contribution (Target)", "Contribution (Current)",
                "Current ($)", "Target ($)", "Realized ($)", "Net Move ($)", 
                "Current APY", "Simulated APY", "Simulated APR", "Ann. Yield", "Ann. Yield (Realized)",
                "Initial Utilization", "Final Utilization", "Initial Liq.", "Final Liq.", "% Liq. Share",
                "Liquid Move ($)", "Stuck Funds ($)", "Link To Market"
            ],
            width='stretch', 
            hide_index=True
        )

        st.divider()

# ==========================================
# 5. EXECUTION LOGIC & TABLES
# ==========================================

        st.markdown("#### 📤 Withdrawal Operations Split")
        st.caption("How funds move from source markets to destinations based on your constraints.")
        
        def get_silo_key(row):
            if "1)" in rebalance_scope: return "GLOBAL"
            if "2)" in rebalance_scope: return (row['Chain'], row['Token'])
            if "3)" in rebalance_scope: return row['Chain']
            if "4)" in rebalance_scope: return row['Token']
            return "GLOBAL"

        silo_global_demand = {} 
        silo_token_demand = {}
        silo_chain_demand = {}
        
        # We use Net Move here because it represents what the optimizer INTENDED to move
        deposits = df_res[df_res['Net Move ($)'] > 0.01]
        for _, row in deposits.iterrows():
            c, t = row['Chain'], row['Token']
            amt = row['Net Move ($)']
            s_key = get_silo_key(row)
            
            silo_token_demand[(c, t)] = silo_token_demand.get((c, t), 0.0) + amt
            silo_chain_demand[c] = silo_chain_demand.get(c, 0.0) + amt
            silo_global_demand[s_key] = silo_global_demand.get(s_key, 0.0) + amt
            
        withdraw_logic = []
        # Filter withdrawals using the restored 'Liquid Move ($)' key
        withdrawals = df_res[df_res['Liquid Move ($)'] < -0.01].copy()
        
        for _, row in withdrawals.iterrows():
            src_chain, src_token = row['Chain'], row['Token']
            src_name, src_id = row['Market'], row['Destination ID']
            total_moved = abs(row['Liquid Move ($)'])
            remaining = total_moved
            s_key = get_silo_key(row)
            
            matched_reb = min(remaining, silo_token_demand.get((src_chain, src_token), 0.0))
            if matched_reb > 0.01:
                silo_token_demand[(src_chain, src_token)] -= matched_reb
                silo_chain_demand[src_chain] -= matched_reb
                silo_global_demand[s_key] -= matched_reb
                remaining -= matched_reb
            else: matched_reb = 0.0

            matched_swap = 0.0
            if "1)" in rebalance_scope or "3)" in rebalance_scope:
                matched_swap = min(remaining, silo_chain_demand.get(src_chain, 0.0))
                if matched_swap > 0.01:
                    silo_chain_demand[src_chain] -= matched_swap
                    silo_global_demand[s_key] -= matched_swap
                    remaining -= matched_swap

            matched_bridge = 0.0
            if "1)" in rebalance_scope or "4)" in rebalance_scope:
                matched_bridge = min(remaining, silo_global_demand.get(s_key, 0.0))
                if matched_bridge > 0.01:
                    silo_global_demand[s_key] -= matched_bridge
                    remaining -= matched_bridge

            matched_cash = remaining

            withdraw_logic.append({
                "Source Market": src_name, "Market ID": src_id, "Chain": src_chain, "Asset": src_token,
                "Total Withdrawal": total_moved, "1. Internal Rebalance": matched_reb,
                "2. Internal Swap": matched_swap, "3. Bridge Out": matched_bridge, "4. Cash Out": matched_cash
            })
            
        if withdraw_logic:
            st.dataframe(
                pd.DataFrame(withdraw_logic).style.format({
                    "Total Withdrawal": "${:,.2f}", "1. Internal Rebalance": "${:,.2f}",
                    "2. Internal Swap": "${:,.2f}", "3. Bridge Out": "${:,.2f}", "4. Cash Out": "${:,.2f}"
                }).applymap(lambda x: 'background-color: #1b5e20; color: white' if x > 0.01 else '', subset=["1. Internal Rebalance"])
                  .applymap(lambda x: 'background-color: #01579b; color: white' if x > 0.01 else '', subset=["2. Internal Swap"])
                  .applymap(lambda x: 'background-color: #b71c1c; color: white' if x > 0.01 else '', subset=["3. Bridge Out"])
                  .applymap(lambda x: 'background-color: #424242; color: white' if x > 0.01 else '', subset=["4. Cash Out"]),
                width='stretch', hide_index=True
            )
        else:
            st.info("No withdrawals required for this strategy.")

        # --- B. Calculate Actual Transfers (Grouped Match) ---
        
        # 1. Prepare Source and Destination Lists
        all_sources = []
        wallet_source = None
        
        # Enrich source data
        withdraw_df = df_res[df_res["Liquid Move ($)"] < -0.01]
        for _, row in withdraw_df.iterrows():
            
            # Safe Lookup for Decimals and Address
            # Because 'df_res' contains calculated rows, we must ensure we grab metadata safely
            try:
                m_row = st.session_state.market_dict[st.session_state.market_dict['Market ID'] == row['Market ID Full']]
                if not m_row.empty:
                    dec_val = int(m_row.iloc[0]['Decimals'])
                    tok_addr = row.get('Loan Address') or m_row.iloc[0]['Loan Address']
                else:
                    dec_val = 18 # Default Fallback
                    tok_addr = row.get('Loan Address')
            except Exception:
                dec_val = 18
                tok_addr = row.get('Loan Address')

            item = {
                "id": row['Market ID Full'],
                "name": row['Market'],
                "chain": row['Chain'],
                "token": row['Token'],
                "available": abs(row['Liquid Move ($)']),
                "running_balance": row['Current ($)'],
                "type": "Market",
                # Metadata
                "chain_id": row.get('ChainID'),
                "token_address": tok_addr,
                "decimals": dec_val,
                "price": row.get('Price USD', 1.0)
            }
            all_sources.append(item)
            
        # Prepare Wallet Source separately (Global Reserve)
        if new_cash > 0.01:
            wallet_source = { 
                "id": "Wallet", "name": "New Capital", "chain": "Wallet", "token": "CASH",
                "available": new_cash, "running_balance": new_cash, "type": "Wallet",
                "chain_id": None, "token_address": None, "decimals": 18, "price": 1.0
            }
            
        all_destinations = []
        deposit_df = df_res[df_res["Net Move ($)"] > 0.01]
        for _, row in deposit_df.iterrows():
            # Destination Lookup
            try:
                m_row = st.session_state.market_dict[st.session_state.market_dict['Market ID'] == row['Market ID Full']]
                tok_addr = row.get('Loan Address') or (m_row.iloc[0]['Loan Address'] if not m_row.empty else None)
            except:
                tok_addr = None

            all_destinations.append({
                "id": row['Market ID Full'],
                "name": row['Market'],
                "chain": row['Chain'],
                "token": row['Token'],
                "needed": row['Net Move ($)'],
                "running_balance": row['Current ($)'],
                # Metadata
                "chain_id": row.get('ChainID'),
                "token_address": tok_addr,
                "apy": row.get('Simulated APY', 0.0)
            })

        # 2. Define Grouping Key based on Scope
        def get_group_key(item):
            if "1)" in rebalance_scope: return "GLOBAL"
            if "2)" in rebalance_scope: return (item['chain'], item['token']) # Exact Match
            if "3)" in rebalance_scope: return item['chain'] # Same Chain, Swap OK
            if "4)" in rebalance_scope: return item['token'] # Same Token, Bridge OK
            return "GLOBAL"

        # 3. Bucketize
        groups = {}
        
        # Bucket Sources
        for src in all_sources:
            k = get_group_key(src)
            if k not in groups: groups[k] = {'src': [], 'dst': []}
            groups[k]['src'].append(src)
            
        # Bucket Destinations
        for dst in all_destinations:
            k = get_group_key(dst)
            if k not in groups: groups[k] = {'src': [], 'dst': []}
            groups[k]['dst'].append(dst)

        transfer_steps = []
        ordering_counter = 1

        # 4. Matching Logic (Per Group)
        for g_key, bucket in groups.items():
            g_srcs = bucket['src']
            g_dsts = bucket['dst']
            
            s_idx, d_idx = 0, 0
            
            while d_idx < len(g_dsts):
                dst = g_dsts[d_idx]
                amount_needed = dst['needed']
                
                # Try to fill from Group Source first
                amount_from_src = 0.0
                src = None
                
                if s_idx < len(g_srcs):
                    src = g_srcs[s_idx]
                    amount_from_src = min(src['available'], amount_needed)
                
                # If no group source available (or empty), try Wallet
                used_wallet = False
                if amount_from_src < 0.01 and wallet_source and wallet_source['available'] > 0.01:
                    src = wallet_source
                    amount_from_src = min(src['available'], amount_needed)
                    used_wallet = True
                
                if amount_from_src > 0.01:
                    # Update balances
                    dst['running_balance'] += amount_from_src
                    src['running_balance'] -= amount_from_src
                    src['available'] -= amount_from_src
                    dst['needed'] -= amount_from_src
                    
                    # Log Step
                    if src['type'] == "Wallet":
                        op_type, op_code = "New Deposit", 0
                    elif src['chain'] != dst['chain']:
                        op_type, op_code = "3. Bridge", 3
                    elif src['token'] != dst['token']:
                        op_type, op_code = "2. Swap", 2
                    else:
                        op_type, op_code = "1. Rebalance", 1
                    
                    # FIX: Enforce Integer for decimals
                    safe_decimals = int(src.get('decimals') or 18)

                    transfer_steps.append({
                        "Ordering": ordering_counter,
                        "Operation Type": op_type, "OpCode": op_code,
                        "From": src['name'], "From Market ID": src.get('id', 'N/A')[:7],
                        "From (Chain)": src['chain'], "From (Token)": src['token'],
                        "To": dst['name'], "To Market ID": dst.get('id', 'N/A')[:7],
                        "To (Chain)": dst['chain'], "To (Token)": dst['token'],
                        "Amount to move ($)": amount_from_src,
                        "Remaining Funds In Source ($)": src['running_balance'],
                        # Metadata for LI.FI
                        "src_chain_id": src.get('chain_id'), "dst_chain_id": dst.get('chain_id'),
                        "src_token": src.get('token_address'), "dst_token": dst.get('token_address'),
                        "decimals": safe_decimals, 
                        "price": src.get('price', 1.0), "dst_apy": dst.get('apy', 0.0)
                    })
                    ordering_counter += 1
                    
                    if src['available'] < 0.01 and not used_wallet: s_idx += 1
                    if dst['needed'] < 0.01: d_idx += 1
                else:
                    # Cannot fill this destination from current sources or wallet?
                    # This shouldn't happen if math is right, but break to avoid infinite loop
                    d_idx += 1

        # 5. Handle Leftover Sources (Cash Out)
        # Any source in any group that still has money -> Wallet
        for g_key, bucket in groups.items():
            for src in bucket['src']:
                if src['available'] > 0.01 and src['available'] >= min_move_thresh:
                    src['running_balance'] -= src['available']
                    transfer_steps.append({
                        "Ordering": ordering_counter,
                        "Operation Type": "Cash Out", "OpCode": 0,
                        "From": src['name'], "From Market ID": src.get('id', 'N/A')[:7],
                        "From (Chain)": src['chain'], "From (Token)": src['token'],
                        "To": "Wallet (Unallocated)", "To Market ID": "N/A",
                        "To (Chain)": "Wallet", "To (Token)": "CASH",
                        "Amount to move ($)": src['available'],
                        "Remaining Funds In Source ($)": src['running_balance']
                    })
                    ordering_counter += 1

        st.divider()
        st.subheader("📋 Execution Plan")

        # Stuck Warning
        stuck_df = df_res[df_res["Stuck Funds ($)"] > 0.01].copy()
        if not stuck_df.empty:
            st.warning(f"⚠️ **Execution Limited by Liquidity:** ${total_stuck_usd:,.2f} blocked.")
            st.dataframe(stuck_df[["Market", "Chain", "Stuck Funds ($)"]].style.format({"Stuck Funds ($)": "${:,.2f}"}), width='stretch', hide_index=True)

        if transfer_steps:
            st.markdown("#### Market Rebalance Steps")
            df_actions = pd.DataFrame(transfer_steps)
            
            def highlight_op(val):
                if "1." in val: return 'color: #00E676; font-weight: bold;'
                if "2." in val: return 'color: #01579b; font-weight: bold;'
                if "3." in val: return 'color: #b71c1c; font-weight: bold;'
                return ''

            st.dataframe(
                df_actions.style.format({
                    "Amount to move ($)": "${:,.2f}",
                    "Remaining Funds In Source ($)": "${:,.2f}"
                }).applymap(highlight_op, subset=['Operation Type']).applymap(lambda x: 'color: red' if x < 0.01 else '', subset=['Remaining Funds In Source ($)']), 
                column_order=["Ordering", "Operation Type", "From", "From Market ID", "From (Chain)", "To", "To Market ID", "To (Chain)", "Amount to move ($)", "Remaining Funds In Source ($)"],
                width='stretch', 
                hide_index=True
            )
            
# --- LI.FI ANALYSIS SECTION ---
            st.markdown("---")
            st.markdown("### 🕵️‍♀️ Bridging Cost Breakeven")
            st.caption("Determines if Bridging/Swapping is worth it. Aggregates moves by route. Simulates execution to find actual costs and break-even time. Note that this is just an estimate and Jumper may give different values")
            
            # Filter for Swaps/Bridges only (OpCode 2 or 3)
            complex_moves = [x for x in transfer_steps if x.get('OpCode') in [2, 3]]
            
            if complex_moves:
                c_an1, c_an2 = st.columns([1, 2])
                with c_an1:
                    # User inputs Percentage (e.g. 0.5), we convert to decimal (0.005) later
                    user_slippage_pct = st.number_input(
                        "Target Slippage Tolerance (%)", 
                        min_value = 0.0001,
                        value=0.0001, 
                        format="%.4f"
                    )
                    
                with c_an2:
                    st.write("") # Spacer
                    st.write("") # Spacer
                    run_analysis = st.button("Analyze Gas, Slippage & ROI via LI.FI", type="primary")

                if run_analysis:
                    analysis_results = []
                    grand_total_cost = 0.0 # <--- NEW: Accumulator for total costs
                    prog_bar = st.progress(0, text=f"Analyzing {len(complex_moves)} individual moves via LI.FI...")
                    
                    dummy_wallet = user_wallet if (user_wallet and len(user_wallet) > 10) else "0x0000000000000000000000000000000000000000"

                    for i, move in enumerate(complex_moves):
                        # FIX: Validation Check - If token addresses or decimals are missing, skip to avoid API crash
                        if not move.get('src_token') or not move.get('dst_token') or move.get('decimals') is None:
                            analysis_results.append({
                                "Signal": "🔴 Incompatible",
                                "From Market": f"{move['From']} ({move['From (Chain)']})",
                                "To Market": f"{move['To']} ({move['To (Chain)']})",
                                "Execution Path": "Missing Token Address/Decimals in Morpho API",
                                "Amount In": f"${move['Amount to move ($)']:,.2f}",
                                "Simulated Out": "$0.00",
                                "Net ROI (1yr $)": 0,
                                "Total Cost": "N/A",
                                "Gas": "$0.00",
                                "Fees": "$0.00",
                                "Swap & Impact Cost ($)": "$0.00",
                                "Slippage Tolerance": "N/A",
                                "Asset B/E": "N/A",
                                "Jumper Link": ""
                            })
                            prog_bar.progress((i + 1) / len(complex_moves))
                            continue

                        token_amt = move['Amount to move ($)'] / move['price'] if move['price'] > 0 else 0
                        
                        # Safe conversion using the sanitized 'decimals'
                        try:
                            atomic_amt = to_atomic_units(token_amt, move['decimals'])
                        except ValueError:
                             atomic_amt = "0"
                        
                        quote_res = get_lifi_quote(
                            move['src_chain_id'],
                            move['dst_chain_id'],
                            move['src_token'],
                            move['dst_token'],
                            atomic_amt,
                            dummy_wallet,
                            user_slippage_decimal=(user_slippage_pct / 100.0)
                        )
                        
                        # ROI Logic
                        move_usd = move['Amount to move ($)']
                        move_apy = move['dst_apy']
                        annual_interest_gain = move_usd * move_apy
                        portfolio_apy_contribution_pct = (annual_interest_gain / total_optimizable * 100) if total_optimizable > 0 else 0.0
                        
                        asset_breakeven_str = "N/A"
                        port_breakeven_str = "N/A"
                        net_roi_usd = 0.0
                        cost_display = "Quote Failed"
                        gas_display = fee_display = slip_display = "$0.00"
                        amt_in_display = f"${move_usd:,.2f}"
                        amt_out_display = "$0.00"
                        slippage_tolerance_display = "Failed"
                        execution_path_display = "No Route Found"
                        signal = "⚪ Unknown"

                        if quote_res['success']:
                            total_cost = quote_res['total_cost']
                            grand_total_cost += total_cost # <--- NEW: Add to global total
                            net_roi_usd = annual_interest_gain - total_cost
                            
                            cost_display = f"${total_cost:.2f}"
                            gas_display = f"${quote_res['gas']:.2f}"
                            fee_display = f"${quote_res['fees']:.2f}"
                            slip_display = f"${quote_res['swap_and_impact_cost']:.2f}"
                            amt_in_display = f"${quote_res['amount_in']:,.2f}"
                            amt_out_display = f"${quote_res['amount_out']:,.2f}"
                            slippage_tolerance_display = f"{quote_res['ladder_tier_used']*100:.4f}%"
                            execution_path_display = quote_res['execution_path']

                            # Asset Break-even
                            asset_hourly_yield = annual_interest_gain / 8760
                            if asset_hourly_yield > 0:
                                hours = total_cost / asset_hourly_yield
                                if hours < 1: asset_breakeven_str = "Instant"
                                elif hours < 24: asset_breakeven_str = f"{hours:.1f} hrs"
                                else: asset_breakeven_str = f"{hours/24:.1f} days"
                            
                            if net_roi_usd < 0: 
                                signal = "🔴 Negative ROI"
                            elif hours < 24: 
                                signal = "🟢 Strong"
                            elif hours < 168: 
                                signal = "🟡 Marginal"
                            else: 
                                signal = "🟠 Slow"
                            
                        link = f"https://jumper.exchange/?fromChain={move['src_chain_id']}&fromToken={move['src_token']}&toChain={move['dst_chain_id']}&toToken={move['dst_token']}&fromAmount={token_amt}"
                        
                        analysis_results.append({
                            "Signal": signal,
                            "From Market": f"{move['From']} ({move['From (Chain)']})",
                            "To Market": f"{move['To']} ({move['To (Chain)']})",
                            "Execution Path": execution_path_display,
                            "Amount In": amt_in_display,
                            "Simulated Out": amt_out_display,
                            "Net ROI (1yr $)": net_roi_usd,
                            "Total Cost": cost_display,
                            "Gas": gas_display,
                            "Fees": fee_display,
                            "Swap & Impact Cost ($)": slip_display,
                            "Slippage Tolerance": slippage_tolerance_display,
                            "Asset B/E": asset_breakeven_str,
                            "Jumper Link": link
                        })
                        prog_bar.progress((i + 1) / len(complex_moves))
                    
                    # ==========================================
# REPLACEMENT BLOCK: LI.FI ANALYSIS RESULT PROCESSING
# ==========================================
                    prog_bar.empty()
                    
                    if analysis_results:
                        st.dataframe(
                            pd.DataFrame(analysis_results),
                            column_config={
                                "Signal": st.column_config.TextColumn(
                                    help=(
                                        "🟢 Strong: Breaks even < 24h\n\n"
                                        "🟡 Marginal: Breaks even in 1-7 days\n\n"
                                        "🟠 Slow: Breaks even in > 7 days\n\n"
                                        "🔴 Negative ROI: Costs exceed 1-year yield gain"
                                    )
                                ),
                                "Execution Path": st.column_config.TextColumn(help="DEXs and Bridges used.", width="medium"),
                                "Net ROI (1yr $)": st.column_config.NumberColumn(
                                    help="Expected 1-year yield gain from this move minus the total execution costs.",
                                    format="$%.2f"
                                ),
                                "Swap & Impact Cost ($)": st.column_config.TextColumn(
                                    help="Difference between Amount In and Amount Out. Includes DEX provider fees and price impact due to liquidity depth."
                                ),
                                "Slippage Tolerance": st.column_config.TextColumn(
                                    help="The maximum price movement allowed before the transaction reverts (Safety Limit)."
                                ),
                                "Jumper Link": st.column_config.LinkColumn("Execute", display_text="Open Jumper"),
                                "Asset B/E": st.column_config.TextColumn(help="Time required for this position's yield to cover execution costs.")
                            },
                            hide_index=True,
                            use_container_width=True
                        )

                        # --- NEW SECTION: Portfolio Level Summary ---
                        # Logic: Calculate break-even based on the TOTAL projected yield of the new strategy
                        if grand_total_cost > 0:
                            if new_annual_interest > 0:
                                # (Total Cost) / (Total Annual Yield / Hours in Year)
                                hours_to_breakeven = grand_total_cost / (new_annual_interest / 8760)
                                
                                if hours_to_breakeven < 24:
                                    time_str = f"{hours_to_breakeven:.1f} Hours"
                                else:
                                    time_str = f"{hours_to_breakeven/24:.1f} Days"
                                
                                # Simplified string to fix rendering artifacts
                                summary_msg = (
                                    f"💰 Total Execution Cost: ${grand_total_cost:,.2f}  |  "
                                    f"⏳ Portfolio Break-Even: {time_str} "
                                    f"(based on total annual interest of ${new_annual_interest:,.2f})"
                                )
                                st.info(summary_msg)
                            else:
                                st.warning(f"💰 Total Execution Cost: ${grand_total_cost:,.2f}. Note: Strategy yield is $0, costs cannot be recovered.")
            else:
                st.info("No cross-chain or swap moves detected in this plan. Costs are negligible.")
            
        else:
            st.success("✅ Portfolio is aligned with this strategy.")