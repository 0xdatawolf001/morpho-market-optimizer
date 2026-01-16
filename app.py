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

# Chart Performance Constants (NEW)
MAX_SCATTER_PLOT_POINTS = 5000 # Max points for the Efficiency Frontier plot
MAX_LINE_PLOT_POINTS_PER_STRATEGY = 1000 # Max points per strategy for the Solver Convergence plot

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

def enrich_market_ids(text, df_all):
    """Adds -- Loan/Collateral labels to raw Market IDs in text."""
    lines = text.split('\n')
    enriched_lines = []
    for line in lines:
        clean = line.strip()
        # If it looks like a Market ID and doesn't already have a label/comment
        if clean.startswith('0x') and '--' not in clean and len(clean) >= 60:
            row = df_all[df_all['Market ID'].str.lower() == clean.lower()]
            if not row.empty:
                loan = row.iloc[0]['Loan Token']
                coll = row.iloc[0]['Collateral']
                enriched_lines.append(f"{clean} -- {loan}/{coll}")
                continue
        enriched_lines.append(line)
    return "\n".join(enriched_lines)

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
# 1. MATH HELPERS
# ==========================================

def to_atomic_units(amount_adjusted: float, decimals: int) -> str:
    """
    Safely converts decimal-adjusted amount to atomic units (string integer).
    Prevents floating point errors by using rounding before casting.
    """
    if not amount_adjusted or amount_adjusted <= 0:
        return "0"
    return str(int(round(amount_adjusted * (10**decimals))))

def get_lifi_quote(from_chain, to_chain, from_token, to_token, amount_atomic, user_address, user_slippage_decimal):
    """
    Calls LI.FI /v1/advanced/routes API to match Jumper.exchange efficiency.
    Includes timing strategies and multi-step route options.
    """
    url = "https://li.quest/v1/advanced/routes"
    
    # Slippage Ladder
    ladder = [user_slippage_decimal]
    standard_tiers = [0.0005, 0.001, 0.005, 0.01, 0.05]
    
    for tier in standard_tiers:
        if tier > user_slippage_decimal:
            ladder.append(tier)
            
    ladder = sorted(list(set(ladder)))
    if ladder[0] != user_slippage_decimal:
        ladder.remove(user_slippage_decimal)
        ladder.insert(0, user_slippage_decimal)

    last_error = None

    for slippage_try in ladder:
        # Construct Payload for /advanced/routes
        payload = {
            "fromChainId": int(from_chain),
            "fromAmount": amount_atomic,
            "fromTokenAddress": from_token,
            "toChainId": int(to_chain),
            "toTokenAddress": to_token,
            "options": {
                "slippage": slippage_try,
                "order": "CHEAPEST", # Matches Jumper defaults for best rate
                "allowSwitchChain": True, # Allows multi-step bridge + swap
                "allowDestinationCall": True, # Essential for efficient solving
                # "fee": 0.0, # Explicitly 0 ensures we see the raw cost without API fees
                "timing": {
                    "routeTimingStrategies": [
                        {
                            "strategy": "minWaitTime",
                            "minWaitTimeMs": 1500, # Wait 1.5s to gather best quotes
                            "startingExpectedResults": 6,
                            "reduceEveryMs": 500
                        }
                    ]
                }
            }
        }

        # Add fromAddress if valid (helps with balance checks/gas estimation)
        if user_address and len(user_address) > 10:
            payload["fromAddress"] = user_address

        headers = {
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    routes = data.get('routes', [])
                    
                    if not routes:
                        last_error = "No routes found."
                        break # Try next slippage tier or exit
                    
                    # We selected "CHEAPEST" in options, so routes[0] is the best one
                    best_route = routes[0]
                    
                    # 1. Amounts
                    amount_in_usd = float(best_route.get('fromAmountUSD', 0))
                    amount_out_usd = float(best_route.get('toAmountUSD', 0))
                    
                    # 2. Costs
                    # gasCostUSD is provided at the top level of the route object
                    gas_usd = float(best_route.get('gasCostUSD', 0))
                    
                    # Fee costs are nested in steps. We sum them up for display.
                    fee_usd = 0.0
                    steps = best_route.get('steps', [])
                    path_steps = []
                    
                    for step in steps:
                        # Sum Fees
                        step_fees = step.get('estimate', {}).get('feeCosts', [])
                        if step_fees:
                            fee_usd += sum([float(f.get('amountUSD', 0)) for f in step_fees])
                        
                        # Build Execution Path String
                        tool_name = step.get('toolDetails', {}).get('name') or step.get('tool')
                        if tool_name:
                            path_steps.append(tool_name)

                    # 3. Total Economic Cost
                    # Value Loss (Slippage + Token Fees) + Gas + Explicit Fees (if any separate)
                    # Note: amount_out_usd usually already accounts for token-deducted fees.
                    slippage_loss = max(0.0, amount_in_usd - amount_out_usd)
                    
                    # Total Cost for ROI calc = Gas + Value Loss (which includes slippage & fees taken from token)
                    # We add fee_usd only if it wasn't already deducted from toAmount (usually feeCosts explains the diff)
                    # To be safe for ROI: Input Value - Output Value + Gas Paid is the user's net cost.
                    total_cost = (amount_in_usd - amount_out_usd) + gas_usd

                    execution_path = " ➡️ ".join(path_steps) if path_steps else "Optimized Route"
                    
                    return {
                        "success": True,
                        "total_cost": total_cost,
                        "gas": gas_usd,
                        "fees": fee_usd, # For display purposes
                        "slippage_cost": slippage_loss,
                        "amount_in": amount_in_usd,
                        "amount_out": amount_out_usd,
                        "slippage_used": slippage_try,
                        "execution_path": execution_path
                    }
                
                elif response.status_code == 429:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
                
                else:
                    last_error = f"API Error {response.status_code}: {response.text}"
                    break 
                    
            except Exception as e:
                print(f"LI.FI Exception: {e}")
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
        
        def safe_float(value, default=0.0):
            if value is None: return default
            try: return float(value)
            except (ValueError, TypeError): return default
        
        supply_usd = safe_float(state.get('supplyAssetsUsd'))
        borrow_usd = safe_float(state.get('borrowAssetsUsd'))
        
        # Capture the price from the API
        price_usd = safe_float(loan.get('priceUsd'), default=1.0)
        
        processed.append({
            "Market ID": m['uniqueKey'],
            "Chain": CHAIN_ID_TO_NAME.get(loan.get('chain', {}).get('id'), "Other"),
            "Loan Token": loan.get('symbol'),
            "Loan Address": loan.get('address'), 
            "Collateral": (m.get('collateralAsset') or {}).get('symbol'),
            "Decimals": loan.get('decimals'),
            "Price USD": price_usd,
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
    Calculates a growth index starting at 1.0 for demand trends.
    """
    now = datetime.now(timezone.utc)
    start_time = int((now - timedelta(days=30)).timestamp())
    
    historical_data = []
    hist_bar = st.progress(0, text="Analyzing 30-day demand trends...")
    
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
                    # Merge supply and borrow shares
                    df_merged = pd.merge(df_sup, df_bor, on='x', suffixes=('_sup', '_bor'))
                    
                    # Convert to numeric
                    df_merged['y_sup'] = df_merged['y_sup'].astype(float)
                    df_merged['y_bor'] = df_merged['y_bor'].astype(float)
                    
                    # Calculate Combined Total Shares
                    df_merged['total_shares'] = df_merged['y_sup'] + df_merged['y_bor']
                    
                    # --- FIX: ENSURE ASCENDING SORT FOR INDEX BASELINE ---
                    df_merged = df_merged.sort_values('x', ascending=True)
                    
                    # Calculate Growth Index relative to the START of the window
                    initial_total = df_merged['total_shares'].iloc[0]
                    
                    if initial_total > 0:
                        df_merged['indexed_demand'] = df_merged['total_shares'] / initial_total
                    else:
                        # Safety: If market starts at 0, index stays at 1.0 until first movement
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
    
    if historical_data:
        return pd.concat(historical_data)
    return pd.DataFrame()

# ==========================================
# 3. OPTIMIZER
# ==========================================

class RebalanceOptimizer:
    def __init__(self, total_budget, market_list, max_dominance_pct=100.0, max_port_alloc_pct=100.0, allow_overflow=False):
        self.total_budget = total_budget 
        self.markets = market_list
        self.max_dominance_ratio = max_dominance_pct / 100.0 
        self.max_port_alloc_ratio = max_port_alloc_pct / 100.0
        self.allow_overflow = allow_overflow 
        
        self.yield_trace = []    
        self.frontier_trace = [] 
        self.liquid_trace = []
        self.whale_trace = []
        self.capacity_warning = None 
        
        self.all_attempts = []   

    def simulate_apy(self, market, user_new_alloc_usd):
        # 1. Get Market Constants
        token_price = market['Price USD']
        if token_price <= 0: return 0.0
        decimals = market['Decimals']
        multiplier = 10**decimals

        # 2. Determine "World Without User" (Base State)
        user_existing_usd = market.get('existing_balance_usd', 0.0)
        user_existing_wei = (user_existing_usd / token_price) * multiplier
        
        current_total_supply_wei = market['raw_supply']
        
        # Base Supply = Total Supply - User's Current Holdings
        base_supply_wei = max(0, current_total_supply_wei - user_existing_wei)
        
        # 3. Add "New" Allocation
        user_new_wei = (user_new_alloc_usd / token_price) * multiplier
        simulated_total_supply_wei = base_supply_wei + user_new_wei
        
        # 4. Standard Morpho Math
        if simulated_total_supply_wei <= 0: 
            return 0.0
        
        new_util = market['raw_borrow'] / simulated_total_supply_wei
        
        new_mult = compute_curve_multiplier(new_util)
        new_borrow_rate = market['rate_at_target'] * new_mult
        new_supply_rate = new_borrow_rate * new_util * (1 - market['fee'])
        
        return rate_per_second_to_apy(new_supply_rate)

    def _get_normalized_allocations(self, x):
        x = np.maximum(x, 0)
        sum_x = np.sum(x)
        if sum_x == 0: return x 
        # Only normalize if we are not strictly capped by safety constraints
        if not self.allow_overflow and sum_x < (self.total_budget - 0.01):
            return x
        return x * (self.total_budget / sum_x)

    def _calculate_metrics(self, x):
        normalized_x = self._get_normalized_allocations(x)
        weights = normalized_x / self.total_budget if self.total_budget > 0 else np.zeros_like(x)
        
        total_yield = 0
        for i, alloc in enumerate(normalized_x):
            total_yield += (alloc * self.simulate_apy(self.markets[i], alloc))
            
        hhi = np.sum(weights**2)
        diversity = 1.0 - hhi
        
        return normalized_x, total_yield, diversity

    def _record_attempt(self, total_yield, diversity):
        self.all_attempts.append({
            "Annual Yield ($)": total_yield,
            "Blended APY": total_yield / self.total_budget if self.total_budget > 0 else 0,
            "Diversity Score": diversity,
            "Type": "Explored" 
        })

    def objective_yield(self, x):
        norm_x, y_val, div_val = self._calculate_metrics(x)
        self._record_attempt(y_val, div_val)
        self.yield_trace.append(y_val)
        return -y_val

    def objective_frontier(self, x):
        norm_x, y_val, div_val = self._calculate_metrics(x)
        self._record_attempt(y_val, div_val)
        self.frontier_trace.append(y_val) 
        return -(y_val * div_val) 
    
    def objective_liquidity(self, x):
        norm_x, y_val, div_val = self._calculate_metrics(x)
        self._record_attempt(y_val, div_val)
        self.liquid_trace.append(y_val)
        liq_score = 0
        for i, alloc in enumerate(norm_x):
            if alloc > 1.0: 
                market = self.markets[i]
                sim_apy = self.simulate_apy(market, alloc)
                avail_liq = market.get('Available Liquidity (USD)', 0.0)
                liq_weight = math.log10(max(10.0, avail_liq)) 
                liq_score += (alloc * sim_apy * liq_weight)
        return -liq_score

    def objective_whale(self, x):
        norm_x, y_val, div_val = self._calculate_metrics(x)
        self._record_attempt(y_val, div_val)
        self.whale_trace.append(y_val)
        return -y_val

    def optimize(self):
        n = len(self.markets)
        if n == 0: return None, None, None, None
        
        # --- 1. Define Combined Bounds ---
        standard_bounds = []
        whale_bounds = []
        
        # Portfolio Cap (USD)
        port_cap_usd = self.total_budget * self.max_port_alloc_ratio
        
        # Whale Shield Logic
        if self.max_dominance_ratio >= 0.99:
            whale_cap_factor = float('inf')
        else:
            whale_cap_factor = self.max_dominance_ratio / (1.0 - self.max_dominance_ratio)

        for m in self.markets:
            # A. Standard Bound: Min(Budget, Portfolio Cap)
            standard_bounds.append((0.0, min(self.total_budget, port_cap_usd)))
            
            # B. Whale Bound: Min(Budget, Portfolio Cap, Liquidity Shield)
            token_price = m.get('Price USD', 0)
            multiplier = 10**m.get('Decimals', 18)
            user_existing_usd = m.get('existing_balance_usd', 0.0)
            user_existing_wei = (user_existing_usd / token_price) * multiplier if token_price > 0 else 0
            
            total_supply_wei = m.get('raw_supply', 0)
            total_borrow_wei = m.get('raw_borrow', 0)
            current_available_wei = max(0, total_supply_wei - total_borrow_wei)
            base_available_wei = max(0, current_available_wei - user_existing_wei)
            base_available_usd = (base_available_wei / multiplier) * token_price if token_price > 0 else 0
            
            liq_shield_cap = base_available_usd * whale_cap_factor
            
            whale_bounds.append((0.0, min(self.total_budget, port_cap_usd, liq_shield_cap)))

        # --- Capacity Checks & Warnings ---
        total_standard_cap = sum([b[1] for b in standard_bounds])
        total_whale_cap = sum([b[1] for b in whale_bounds])
        
        if total_whale_cap < (self.total_budget):
            if self.allow_overflow:
                self.capacity_warning = (
                    f"⚠️ **Safety Limits Ignored:** Total Budget (\${self.total_budget:,.0f}) exceeds your safety caps. "
                    "Optimizer reverted to standard bounds to ensure full investment."
                )
                whale_bounds = [(0.0, self.total_budget)] * n
                standard_bounds = [(0.0, self.total_budget)] * n
            else:
                self.capacity_warning = (
                    f"⚠️ **Budget Limited by Safety Caps:** Your total budget (\${self.total_budget:,.0f}) exceeds the "
                    f"combined capacity of your chosen markets (\${total_whale_cap:,.0f}) under current safety rules. "
                    "Some cash will remain unallocated."
                )

        x0 = np.full(n, self.total_budget / n)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_budget})
        options = {'maxiter': 5000, 'ftol': 1e-6}

        # Optimizer execution
        res_yield = minimize(self.objective_yield, x0, method='SLSQP', bounds=standard_bounds, constraints=constraints, options=options)
        best_yield_alloc = self._get_normalized_allocations(res_yield.x)

        res_frontier = minimize(self.objective_frontier, x0, method='SLSQP', bounds=standard_bounds, constraints=constraints, options=options)
        best_frontier_alloc = self._get_normalized_allocations(res_frontier.x)
        
        res_liq = minimize(self.objective_liquidity, x0, method='SLSQP', bounds=whale_bounds, constraints=constraints, options=options)
        best_liquid_alloc = self._get_normalized_allocations(res_liq.x)
        
        res_whale = minimize(self.objective_whale, x0, method='SLSQP', bounds=whale_bounds, constraints=constraints, options=options)
        best_whale_alloc = self._get_normalized_allocations(res_whale.x)
        
        return best_yield_alloc, best_frontier_alloc, best_liquid_alloc, best_whale_alloc
    
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
    m_apy_in = r1_c1.number_input("Min APY %", 0.0, 1000.0, 0.0, step=0.5)
    x_apy_in = r1_c2.number_input("Max APY %", 0.0, 1000.0, 1000.0, step=0.5)
    
    # Metric Range Row 2: Utilization
    r2_c1, r2_c2 = st.columns(2)
    m_util_in = r2_c1.number_input("Min Util %", 0.0, 100.0, 0.0)
    x_util_in = r2_c2.number_input("Max Util %", 0.0, 100.0, 100.0)
    
    # Metric Row 3: Liquidity Thresholds
    r3_c1, r3_c2 = st.columns(2)
    m_supply_usd = r3_c1.number_input("Min Total Supply (USD)", 0.0, 10_000_000_000.0, 0.0, step=100000.0)
    m_avail_usd = r3_c2.number_input("Min Available Liquidity (USD)", 0.0, 10_000_000_000.0, 0.0, step=100000.0)
    
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
        # NEW: Clear All button replaces Clear Wallet Markets
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
        height=300,
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
    new_cash = st.number_input("Additional New Cash (USD)", value=0.0, step=1000.0)

    # NEW: Portfolio Cap
    max_port_alloc = st.number_input(
        "Max Portfolio Allocation %",
        min_value=0.01,
        max_value=100.0,
        value=100.0,
        step=5.0,
        help="Hard safety limit: No single market can ever exceed this % of your TOTAL portfolio. Useful for capping protocol risk."
    )

    # Whale Shield Input
    max_dominance = st.number_input(
        "Whale Shield: Max Dominance %",
        min_value=0.001,
        max_value=1000.0,
        value=30.0,
        step=5.0,
        help="Hard liquidity limit: You will never own more than this % of a market's AVAILABLE liquidity (Supply - Borrow)."
    )

    min_move_thresh = st.number_input(
        "Min Rebalance Threshold ($)", 
        value=0.0, 
        step=50.0
    )

    allow_break = st.checkbox(
        "Allow Safety Cap Overflow?",
        value=False,
        help="If checked, the optimizer will ignore both Whale Shield and Portfolio Caps to ensure all money is invested. Useful if your safety rules are too strict for your budget size."
    )

if not df_selected.empty:

    # 1. Ensure current state is mapped to the dataframe
    df_selected['Existing Balance (USD)'] = df_selected['Market ID'].apply(
        lambda x: st.session_state.balance_cache.get(x, 0.0)
    )

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

# Locate the st.data_editor block and replace it with this version
    edited_df = st.data_editor(
        df_selected[[
            'Market ID', 'Chain', 'Loan Token', 'Collateral', 
            'Supply APY', 'Utilization', 'Total Supply (USD)', 
            'Total Borrow (USD)', 'Available Liquidity (USD)', 
            'Existing Balance (USD)', 'Link To Market'
        ]],
        column_config={
            "Supply APY": st.column_config.NumberColumn(format="percent"),
            "Utilization": st.column_config.NumberColumn(format="percent"),
            "Total Supply (USD)": st.column_config.NumberColumn(format="dollar"),
            "Total Borrow (USD)": st.column_config.NumberColumn(format="dollar"),
            "Available Liquidity (USD)": st.column_config.NumberColumn(format="dollar"),
            "Existing Balance (USD)": st.column_config.NumberColumn(format="dollar", min_value=0.0),
            "Link To Market": st.column_config.LinkColumn("Link To Market", display_text="Link")
        },
        disabled=[
            'Market ID', 'Chain', 'Loan Token', 'Collateral', 
            'Supply APY', 'Utilization', 'Total Supply (USD)', 
            'Total Borrow (USD)', 'Available Liquidity (USD)', 'Link To Market'
        ],
        width='stretch', 
        hide_index=True, 
        key="portfolio_editor",
        on_change=sync_portfolio_edits
    )

    current_wealth = sum(st.session_state.balance_cache.get(m_id, 0.0) for m_id in df_selected['Market ID'])
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

    if st.button("🚀 Run Optimization", type="primary", width='stretch'):
        # Add this line to clear previous history cache
        if 'hist_demand_df' in st.session_state:
            del st.session_state.hist_demand_df
        market_data_list = fetch_live_market_details(df_selected)
        for m in market_data_list:
            m['existing_balance_usd'] = st.session_state.balance_cache.get(m['Market ID'], 0.0)

        current_annual_interest = sum(m['existing_balance_usd'] * m['current_supply_apy'] for m in market_data_list)
        current_blended_apy = current_annual_interest / current_wealth if current_wealth > 0 else 0.0

        # Pass max_dominance, max_port_alloc, and allow_overflow to Optimizer
        opt = RebalanceOptimizer(
            total_optimizable, 
            market_data_list, 
            max_dominance, 
            max_port_alloc, 
            allow_break
        )
        
        # Unpack 4 results
        r_yield, r_frontier, r_liquid, r_whale = opt.optimize()

        # MODIFIED: Pass total_optimizable to filter_small_moves to handle Unallocated Cash
        cleaned_yield = filter_small_moves(r_yield, market_data_list, min_move_thresh, total_optimizable)
        cleaned_frontier = filter_small_moves(r_frontier, market_data_list, min_move_thresh, total_optimizable)
        cleaned_liquid = filter_small_moves(r_liquid, market_data_list, min_move_thresh, total_optimizable)
        cleaned_whale = filter_small_moves(r_whale, market_data_list, min_move_thresh, total_optimizable)
        
        st.session_state['opt_results'] = {
            'market_data_list': market_data_list,
            'opt_object': opt,
            'best_yield_alloc': cleaned_yield,
            'frontier_alloc': cleaned_frontier,
            'liquid_alloc': cleaned_liquid,
            'whale_alloc': cleaned_whale, 
            'current_metrics': {
                'annual_interest': current_annual_interest,
                'blended_apy': current_blended_apy
            },
            'traces': {
                'yield': opt.yield_trace,
                'frontier': opt.frontier_trace,
                'liquid': opt.liquid_trace,
                'whale': opt.whale_trace
            },
            'attempts': opt.all_attempts
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
            
            st.altair_chart(base + points, width='stretch')

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
        d_col_header, d_col_filter = st.columns([2, 1])
        with d_col_header:
            st.subheader("📈 30-Day Demand Trend (Indexed)")
            st.caption("Growth of Supply + Borrow shares relative to start of period (Baseline = 1.0).")
        
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
            # Calculate the final value for every market to determine trend
            trend_summary = df_history.sort_values('date').groupby('Market')['indexed_demand'].last()
            
            with d_col_filter:
                filter_mode = st.radio(
                    "Filter Trends:",
                    ["All", "Growth (> 1.0)", "Decay (≤ 1.0)"],
                    horizontal=True,
                    key="trend_filter_radio"
                )

            # Apply Filter
            markets_to_show = trend_summary.index.tolist() # Default All
            
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
                        columns=4, # More columns since it's full width
                        labelLimit=0,
                        symbolLimit=100,
                        padding=10
                    ))
                )

                # The visible lines
                lines = base.mark_line(interpolate='monotone').encode(
                    opacity=alt.condition(selection, alt.value(1), alt.value(0.15)),
                    strokeWidth=alt.condition(selection, alt.value(2.5), alt.value(1))
                ).add_params(selection)

                # Transparent selectors for hover behavior
                selectors = base.mark_point().encode(
                    opacity=alt.value(0),
                    tooltip=[
                        alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                        alt.Tooltip('Market:N', title='Market'),
                        alt.Tooltip('indexed_demand:Q', title='Index Value', format='.3f')
                    ]
                ).add_params(hover)

                # Points that appear on hover
                points = base.mark_point().encode(
                    opacity=alt.condition(hover, alt.value(1), alt.value(0))
                )

                # Vertical rule that follows mouse
                rules = base.mark_rule(color='white', strokeWidth=0.5).encode(
                    opacity=alt.condition(hover, alt.value(0.3), alt.value(0))
                ).transform_filter(hover)

                # Layering the chart components - Height increased to 450
                final_chart = alt.layer(
                    lines, selectors, points, rules
                ).properties(height=450).interactive()
                
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
            # Show capacity warnings if they exist
            if hasattr(res_data['opt_object'], 'capacity_warning') and res_data['opt_object'].capacity_warning:
                st.warning(res_data['opt_object'].capacity_warning)
        elif "Frontier" in strategy_choice:
            final_alloc = frontier_alloc
        elif "Liquid" in strategy_choice:
            final_alloc = liquid_alloc
        else:
            # Current Portfolio: Target = Current Balance
            final_alloc = np.array([m['existing_balance_usd'] for m in market_data_list])

        results = []
        new_annual_interest = 0
        total_allocated_usd = 0.0
        total_stuck_usd = 0.0
        
        for i, target_val in enumerate(final_alloc):
            m = market_data_list[i]
            total_allocated_usd += target_val
            
            # --- UTILIZATION MATH ---
            token_price = m['Price USD']
            multiplier = 10**m['Decimals']
            user_current = m['existing_balance_usd']
            
            # Initial Util (from raw market data)
            initial_util = m['raw_borrow'] / m['raw_supply'] if m['raw_supply'] > 0 else 0
            
            # Final Util (accounting for user's net change in supply)
            user_existing_wei = (user_current / token_price) * multiplier if token_price > 0 else 0
            target_wei = (target_val / token_price) * multiplier if token_price > 0 else 0
            base_supply_wei = max(0, m['raw_supply'] - user_existing_wei)
            simulated_total_supply_wei = base_supply_wei + target_wei
            
            final_util = m['raw_borrow'] / simulated_total_supply_wei if simulated_total_supply_wei > 0 else 0
            
            # --- LIQUIDITY MATH ---
            current_avail = m.get('Available Liquidity (USD)', 0.0)
            base_available = max(0, current_avail - user_current)
            final_available = base_available + target_val
            liq_share = target_val / final_available if final_available > 0 else 0
            
            # Movement and Stuck logic
            net_move = target_val - user_current
            requested_withdrawal = max(0, -net_move)
            stuck_funds = max(0, requested_withdrawal - current_avail)
            actual_withdrawable = requested_withdrawal - stuck_funds
            liquid_move = -actual_withdrawable if net_move < 0 else net_move
            total_stuck_usd += stuck_funds
            
            # Action State Logic
            if net_move > 0.01:
                action = "🟢 DEPOSIT"
            elif net_move < -0.01:
                if actual_withdrawable <= 0.01:
                    action = "⚠️ STUCK"
                elif stuck_funds > 0.01:
                    action = "🟠 PARTIAL"
                else:
                    action = "🔴 WITHDRAW"
            else:
                action = "⚪ HOLD"

            # APY Calculations
            current_apy_val = m.get('current_supply_apy', 0.0)
            new_apy = current_apy_val if abs(net_move) < 0.01 else opt.simulate_apy(m, target_val)
            new_annual_interest += (target_val * new_apy)

            results.append({
                "Destination ID": str(m['Market ID'])[:7],
                "Market": f"{m['Loan Token']}/{m['Collateral']}",
                "Token": m['Loan Token'],
                "Chain": m['Chain'], 
                "Action": action,
                "Weight": target_val / total_optimizable if total_optimizable > 0 else 0,
                "Initial Utilization": initial_util,
                "Final Utilization": final_util,
                "Current ($)": user_current,
                "Target ($)": target_val,
                "Net Move ($)": net_move,
                "Liquid Move ($)": liquid_move,
                "Stuck Funds ($)": stuck_funds,
                "Current APY": current_apy_val,
                "Simulated APY": new_apy,
                "Ann. Yield": target_val * new_apy,
                "Initial Liq.": current_avail,
                "Final Liq.": final_available,
                "% Liq. Share": liq_share,
                "Market ID Full": m['Market ID'],
                "ChainID": m['ChainID'],
                "Link To Market": f"https://www.monarchlend.xyz/market/{int(m['ChainID'])}/{m['Market ID']}"
            })

        # --- Check for Unallocated Capital ---
        unallocated_cash = total_optimizable - total_allocated_usd
        if unallocated_cash > 0.01:
            results.append({
                "Market": "⚠️ Unallocated Cash",
                "Token": "CASH",
                "Chain": "Wallet",
                "Action": "⚪ HOLD",
                "Weight": unallocated_cash / total_optimizable if total_optimizable > 0 else 0,
                "Current ($)": 0.0, 
                "Target ($)": unallocated_cash,
                "Net Move ($)": 0.0, 
                "Current APY": 0.0,
                "Simulated APY": 0.0,
                "Ann. Yield": 0.0,
                "Initial Liq.": 0.0,
                "Final Liq.": 0.0,
                "% Liq. Share": 0.0
            })
        
        df_res = pd.DataFrame(results)
        
        if total_optimizable > 0:
            df_res["Contribution to Portfolio APY"] = (df_res["Target ($)"] / total_optimizable) * df_res["Simulated APY"]
        else:
            df_res["Contribution to Portfolio APY"] = 0.0
        
        current_blended = res_data['current_metrics']['blended_apy']
        current_ann = res_data['current_metrics']['annual_interest']
        new_blended_apy = new_annual_interest / total_optimizable if total_optimizable > 0 else 0.0
        
        apy_diff = new_blended_apy - current_blended
        interest_diff = new_annual_interest - current_ann
        selected_weights = np.array([r['Weight'] for r in results if 'Weight' in r])
        selected_diversity = 1.0 - np.sum(selected_weights**2)

        m1, m2, m3, m4, m5, m6 = st.columns(6) 
        m1.metric("Current APY", f"{current_blended:.4%}")
        m2.metric("Optimized APY", f"{new_blended_apy:.4%}", delta=f"{apy_diff*100:.3f}%")
        m3.metric("Total Wealth (1 Yr)", f"${total_optimizable + new_annual_interest:,.2f}")
        m4.metric("Annual Interest Gain", f"${interest_diff:,.2f}")
        m5.metric("Diversity Score", f"{selected_diversity:.4f}")
        m6.metric("Stuck Capital", f"${total_stuck_usd:,.2f}", delta_color="inverse", delta="Liquidity Issue" if total_stuck_usd > 0 else None)

        # Row 2: Time-Based Earnings Breakdown
        st.markdown("---")
        st.subheader("📈 Projected Earnings")
        
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Annual", f"${new_annual_interest:,.2f}")
        t2.metric("Monthly", f"${new_annual_interest/12:,.2f}")
        t3.metric("Weekly", f"${new_annual_interest/52:,.2f}")
        t4.metric("Daily", f"${new_annual_interest/365:,.2f}")
        t5.metric("Hourly", f"${new_annual_interest/8760:,.4f}")

        st.divider()
        st.subheader("⚖️ Allocations")
        df_res = df_res.sort_values(by=["Action", "Weight"], ascending=[False, False])
        
        st.dataframe(
            df_res.style.format({
                "Weight": "{:.2%}", 
                "Initial Utilization": "{:.2%}",      
                "Final Utilization": "{:.2%}",        
                "Current ($)": "${:,.2f}", 
                "Target ($)": "${:,.2f}", 
                "Net Move ($)": "${:,.2f}",
                "Liquid Move ($)": "${:,.2f}",
                "Stuck Funds ($)": "${:,.2f}",
                "Current APY": "{:.4%}",
                "Simulated APY": "{:.4%}",
                "Ann. Yield": "${:,.2f}",
                "Initial Liq.": "${:,.2f}",
                "Final Liq.": "${:,.2f}",
                "% Liq. Share": "{:.2%}",
                "Contribution to Portfolio APY": "{:.4%}" 
            }), 
            column_config={
                "Link To Market": st.column_config.LinkColumn("Link To Market", display_text="Link")
            },
            column_order=["Destination ID", "Market", "Chain", "Action", "Weight", 
                          "Initial Utilization", "Final Utilization", 
                          "Current ($)", "Target ($)", "Net Move ($)", "Liquid Move ($)", 
                          "Stuck Funds ($)", "Current APY", "Simulated APY", "Ann. Yield", 
                          "Initial Liq.", "Final Liq.", "% Liq. Share", "Contribution to Portfolio APY", "Link To Market"],
            width='stretch', 
            hide_index=True
        )

        st.divider()

# ==========================================
        # 5. EXECUTION LOGIC & TABLES
        # ==========================================
        
        # 2. Operational Split (Rebalance vs Swap vs Bridge)
        st.markdown("#### 📤 Withdrawal Operations Split")
        st.caption("Breakdown of where your withdrawn funds should go.")
        
        # --- A. Theoretical Split ---
        chain_capacities = {} 
        token_capacities = {} 
        
        deposits = df_res[df_res['Net Move ($)'] > 0.01]
        for _, row in deposits.iterrows():
            c = row['Chain']
            t = row['Token']
            amt = row['Net Move ($)']
            chain_capacities[c] = chain_capacities.get(c, 0.0) + amt
            token_capacities[(c, t)] = token_capacities.get((c, t), 0.0) + amt
            
        withdraw_logic = []
        withdrawals = df_res[df_res['Net Move ($)'] < -0.01].copy()
        withdrawals = withdrawals.sort_values(['Chain', 'Token'])
        grouped_withdrawals = withdrawals.groupby(['Chain', 'Token'])['Net Move ($)'].sum().reset_index()
        
        for _, row in grouped_withdrawals.iterrows():
            src_chain = row['Chain']
            src_token = row['Token']
            total_moved = abs(row['Net Move ($)'])
            remaining_val = total_moved
            
            # 1. Match: Rebalance (Same Chain, Same Token)
            specific_cap = token_capacities.get((src_chain, src_token), 0.0)
            matched_rebalance = min(remaining_val, specific_cap)
            
            if matched_rebalance > 0:
                token_capacities[(src_chain, src_token)] -= matched_rebalance
                chain_capacities[src_chain] -= matched_rebalance
                remaining_val -= matched_rebalance
            
            # 2. Match: Swap (Same Chain, Different Token)
            general_cap = chain_capacities.get(src_chain, 0.0)
            matched_swap = min(remaining_val, general_cap)
            
            if matched_swap > 0:
                chain_capacities[src_chain] -= matched_swap
                remaining_val -= matched_swap
                
            # 3. Match: Bridge (Exit Chain)
            matched_bridge = remaining_val
            
            withdraw_logic.append({
                "Source Chain": src_chain,
                "Asset": src_token,
                "Total Capital Affected": total_moved,
                "1. Keep In Chain And Same Asset (Rebalance)": matched_rebalance,
                "2. Swap Asset But In Same Chain (Internal)": matched_swap,
                "3. Bridge Out": matched_bridge
            })
            
        if withdraw_logic:
            df_ops = pd.DataFrame(withdraw_logic)
            st.dataframe(
                df_ops.style.format({
                    "Total Capital Affected": "${:,.2f}",
                    "1. Keep In Chain And Same Asset (Rebalance)": "${:,.2f}",
                    "2. Swap Asset But In Same Chain (Internal)": "${:,.2f}",
                    "3. Bridge Out": "${:,.2f}"
                }).applymap(lambda x: 'background-color: #1b5e20; color: white' if x > 0.01 else '', subset=["1. Keep In Chain And Same Asset (Rebalance)"])
                  .applymap(lambda x: 'background-color: #01579b; color: white' if x > 0.01 else '', subset=["2. Swap Asset But In Same Chain (Internal)"])
                  .applymap(lambda x: 'background-color: #b71c1c; color: white' if x > 0.01 else '', subset=["3. Bridge Out"]),
                width='stretch',
                hide_index=True
            )
        else:
            st.info("No withdrawals required for this strategy.")

        # --- B. Calculate Actual Transfers (Greedy Match) ---
        sources = []
        # Enrich source data with necessary metadata for LI.FI
        withdraw_df = df_res[df_res["Liquid Move ($)"] < -0.01]
        for _, row in withdraw_df.iterrows():
            sources.append({
                "id": row['Market ID Full'],
                "name": row['Market'],
                "chain": row['Chain'],
                "token": row['Token'],
                "available": abs(row['Liquid Move ($)']),
                "running_balance": row['Current ($)'],
                "type": "Market",
                # Metadata for LI.FI
                "chain_id": row.get('ChainID'),
                "token_address": row.get('Loan Address', None) or st.session_state.market_dict[st.session_state.market_dict['Market ID'] == row['Market ID Full']].iloc[0]['Loan Address'],
                "decimals": st.session_state.market_dict[st.session_state.market_dict['Market ID'] == row['Market ID Full']].iloc[0]['Decimals'],
                "price": row.get('Price USD', 1.0)
            })
            
        if new_cash > 0.01:
            sources.append({ 
                "id": "Wallet",
                "name": "New Capital", 
                "chain": "Wallet",
                "token": "CASH",
                "available": new_cash, 
                "running_balance": new_cash, 
                "type": "Wallet",
                "chain_id": None, "token_address": None, "decimals": 18, "price": 1.0
            })
            
        destinations = []
        deposit_df = df_res[df_res["Net Move ($)"] > 0.01]
        for _, row in deposit_df.iterrows():
            destinations.append({
                "id": row['Market ID Full'],
                "name": row['Market'],
                "chain": row['Chain'],
                "token": row['Token'],
                "needed": row['Net Move ($)'],
                "running_balance": row['Current ($)'],
                # Metadata for LI.FI
                "chain_id": row.get('ChainID'),
                "token_address": row.get('Loan Address', None) or st.session_state.market_dict[st.session_state.market_dict['Market ID'] == row['Market ID Full']].iloc[0]['Loan Address'],
                "apy": row.get('Simulated APY', 0.0)
            })

        transfer_steps = []
        src_idx, dst_idx = 0, 0
        ordering_counter = 1
        
        # 1. Match Sources to Destinations
        while src_idx < len(sources) and dst_idx < len(destinations):
            src, dst = sources[src_idx], destinations[dst_idx]
            amount_to_move = min(src['available'], dst['needed'])
            
            if amount_to_move > 0.01: 
                dst['running_balance'] += amount_to_move
                src['running_balance'] -= amount_to_move
                
                # Determine Operation Type
                if src['type'] == "Wallet":
                    op_type = "New Deposit"
                    op_code = 0
                elif src['chain'] != dst['chain']:
                    op_type = "3. Bridge"
                    op_code = 3
                else:
                    if src['token'] == dst['token']:
                        op_type = "1. Rebalance"
                        op_code = 1
                    else:
                        op_type = "2. Swap"
                        op_code = 2

                transfer_steps.append({
                    "Ordering": ordering_counter,
                    "From": src['name'],
                    "From Market ID": src.get('id', 'N/A')[:7],
                    "From (Chain)": src['chain'],
                    "To": dst['name'],
                    "To Market ID": dst.get('id', 'N/A')[:7],
                    "To (Chain)": dst['chain'],
                    "Amount to move ($)": amount_to_move,
                    "Remaining Funds In Source ($)": src['running_balance'],
                    "Operation Type": op_type,
                    "OpCode": op_code,
                    # Hidden Metadata for Analysis
                    "src_chain_id": src.get('chain_id'),
                    "dst_chain_id": dst.get('chain_id'),
                    "src_token": src.get('token_address'),
                    "dst_token": dst.get('token_address'),
                    "decimals": src.get('decimals', 18),
                    "price": src.get('price', 1.0),
                    "dst_apy": dst.get('apy', 0.0)
                })
                ordering_counter += 1
            
            src['available'] -= amount_to_move
            dst['needed'] -= amount_to_move
            
            if src['available'] < 0.01: src_idx += 1
            if dst['needed'] < 0.01: dst_idx += 1

        # 2. Handle Leftover Sources (Market -> Wallet)
        while src_idx < len(sources):
            src = sources[src_idx]
            if src['available'] > 0.01 and src['available'] >= min_move_thresh:
                src['running_balance'] -= src['available']
                transfer_steps.append({
                    "Ordering": ordering_counter,
                    "From": src['name'],
                    "From (Chain)": src['chain'],
                    "To": "Wallet (Unallocated)",
                    "To (Chain)": "Wallet",
                    "Amount to move ($)": src['available'],
                    "Remaining Funds In Source ($)": src['running_balance'],
                    "Operation Type": "Cash Out",
                    "OpCode": 0
                })
                ordering_counter += 1
            src_idx += 1

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
                if "1." in val: return 'color: #00E676; font-weight: bold;' # Brighter Green
                if "2." in val: return 'color: #01579b; font-weight: bold;' # Blue
                if "3." in val: return 'color: #b71c1c; font-weight: bold;' # Red
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
            st.markdown("### 🕵️‍♀️ Briding Cost Breakeven")
            st.caption("Determines if Bridging/Swapping is worth it. Aggregates moves by route. Simulates execution to find actual costs and break-even time. Note that this is just an estimate and Jumper may give different values")
            
            # Filter for Swaps/Bridges only (OpCode 2 or 3)
            complex_moves = [x for x in transfer_steps if x.get('OpCode') in [2, 3]]
            
            if complex_moves:
                c_an1, c_an2 = st.columns([1, 2])
                with c_an1:
                    # User inputs Percentage (e.g. 0.5), we convert to decimal (0.005) later
                    user_slippage_pct = st.number_input("Target Slippage Tolerance (%)", min_value=0.0001, max_value=5.0, value=0.01, step=0.01)
                    
                with c_an2:
                    st.write("") # Spacer
                    st.write("") # Spacer
                    run_analysis = st.button("Analyze Gas, Slippage & ROI via LI.FI", type="primary")

                if run_analysis:
                    # 1. Batching Logic: Group by Route
                    batches = {}
                    for move in complex_moves:
                        # Define unique route key
                        key = (move['src_chain_id'], move['dst_chain_id'], move['src_token'], move['dst_token'])
                        
                        if key not in batches:
                            batches[key] = {
                                "src_chain_name": move['From (Chain)'],
                                "dst_chain_name": move['To (Chain)'],
                                "src_token_symbol": move['From (Token)'],
                                "dst_token_symbol": move['To (Token)'],
                                "total_usd": 0.0,
                                "total_token_amount": 0.0,
                                "weighted_yield_sum": 0.0,
                                "decimals": move['decimals'],
                                "price": move['price']
                            }
                        
                        batches[key]["total_usd"] += move['Amount to move ($)']
                        # Decimal adjusted token amount
                        token_amt = move['Amount to move ($)'] / move['price'] if move['price'] > 0 else 0
                        batches[key]["total_token_amount"] += token_amt
                        # Yield contribution for weighted APY
                        batches[key]["weighted_yield_sum"] += (move['Amount to move ($)'] * move['dst_apy'])

                    analysis_results = []
                    batch_list = list(batches.values())
                    prog_bar = st.progress(0, text="Requesting batched quotes from LI.FI...")
                    dummy_wallet = user_wallet if (user_wallet and len(user_wallet) > 10) else "0x0000000000000000000000000000000000000000"
                    
                    # Portfolio level earnings rate (per hour)
                    portfolio_hourly_earnings = new_annual_interest / 8760 if new_annual_interest > 0 else 0

                    for i, batch in enumerate(batch_list):
                        # 2. Convert to Atomic Units for LI.FI
                        atomic_amt = to_atomic_units(batch['total_token_amount'], batch['decimals'])
                        
                        # Find the key keys for the API call
                        route_key = [k for k, v in batches.items() if v == batch][0]
                        
                        # 3. Call API with Smart Slippage
                        quote_res = get_lifi_quote(
                            route_key[0], # src_chain_id
                            route_key[1], # dst_chain_id
                            route_key[2], # src_token (address)
                            route_key[3], # dst_token (address)
                            atomic_amt,   # ATOMIC UNITS STRING
                            dummy_wallet,
                            user_slippage_decimal=(user_slippage_pct / 100.0)
                        )
                        
                        # 4. ROI Logic using Batched Totals
                        batch_total_usd = batch['total_usd']
                        batch_weighted_apy = batch['weighted_yield_sum'] / batch_total_usd if batch_total_usd > 0 else 0
                        
                        # New: Portfolio APY Contribution (Multiplied by 100 for formatter)
                        portfolio_apy_contribution_pct = (batch['weighted_yield_sum'] / total_optimizable * 100) if total_optimizable > 0 else 0.0

                        # Asset-Chain level earnings
                        asset_hourly_yield_usd = (batch_total_usd * batch_weighted_apy) / 8760
                        
                        asset_breakeven_str = "N/A"
                        port_breakeven_str = "N/A"
                        
                        cost_display = "Quote Failed"
                        gas_display = fee_display = slip_display = "$0.00"
                        amt_in_display = f"${batch_total_usd:,.2f}" # Default to theoretical if fail
                        amt_out_display = "$0.00"
                        slippage_used_display = "Failed"
                        execution_path_display = "No Route Found"
                        
                        signal = "⚪ Unknown"

                        if quote_res['success']:
                            total_cost = quote_res['total_cost']
                            cost_display = f"${total_cost:.2f}"
                            
                            gas_display = f"${quote_res['gas']:.2f}"
                            fee_display = f"${quote_res['fees']:.2f}"
                            slip_display = f"${quote_res['slippage_cost']:.2f}"
                            
                            amt_in_display = f"${quote_res['amount_in']:,.2f}"
                            amt_out_display = f"${quote_res['amount_out']:,.2f}"
                            
                            # Show Slippage Used as Percentage
                            slippage_used_display = f"{quote_res['slippage_used']*100:.2f}%"
                            execution_path_display = quote_res['execution_path']

                            # --- A. Asset-Chain Break-even ---
                            if asset_hourly_yield_usd > 0:
                                hours_to_recover = total_cost / asset_hourly_yield_usd
                                if hours_to_recover < 24:
                                    asset_breakeven_str = f"{hours_to_recover:.1f} hrs"
                                    signal = "🟢 Good"
                                elif hours_to_recover < 168:
                                    asset_breakeven_str = f"{hours_to_recover/24:.1f} days"
                                    signal = "🟡 Okay"
                                else:
                                    asset_breakeven_str = f"{hours_to_recover/24:.0f} days"
                                    signal = "🔴 Poor"
                            else:
                                asset_breakeven_str = "Never"
                                signal = "🔴 Bad"

                            # --- B. Portfolio Break-even ---
                            if portfolio_hourly_earnings > 0:
                                port_hours = total_cost / portfolio_hourly_earnings
                                if port_hours < 1:
                                    port_breakeven_str = "Instant"
                                elif port_hours < 24:
                                    port_breakeven_str = f"{port_hours:.1f} hrs"
                                else:
                                    port_breakeven_str = f"{port_hours/24:.1f} days"
                            
                        # 5. Generate Jumper Link
                        link = f"https://jumper.exchange/?fromChain={route_key[0]}&fromToken={route_key[2]}&toChain={route_key[1]}&toToken={route_key[3]}&fromAmount={batch['total_token_amount']}"
                        
                        analysis_results.append({
                            "Signal": signal,
                            "Route": f"{batch['src_token_symbol']} ({batch['src_chain_name']}) ➡️ {batch['dst_token_symbol']} ({batch['dst_chain_name']})",
                            "Execution Path": execution_path_display,
                            "Amount In": amt_in_display,
                            "Simulated Out": amt_out_display,
                            "Portfolio APY Contribution": portfolio_apy_contribution_pct,
                            "Total Cost": cost_display,
                            "Gas": gas_display,
                            "Fees": fee_display,
                            "Slippage ($)": slip_display,
                            "Slippage Used": slippage_used_display,
                            "Asset B/E": asset_breakeven_str,
                            "Portfolio B/E": port_breakeven_str,
                            "Jumper Link": link
                        })
                        prog_bar.progress((i + 1) / len(batch_list))
                    
                    prog_bar.empty()
                    
                    if analysis_results:
                        st.dataframe(
                            pd.DataFrame(analysis_results),
                            column_config={
                                "Execution Path": st.column_config.TextColumn(help="The sequence of DEXs and Bridges used in this route.", width="large"),
                                "Amount In": st.column_config.TextColumn(help="USD Value of tokens sent"),
                                "Simulated Out": st.column_config.TextColumn(help="USD Value of tokens received (after fees/slippage)"),
                                "Portfolio APY Contribution": st.column_config.NumberColumn(
                                    help="How much this specific route contributes to your TOTAL portfolio APY (Gross annual yield / Total Budget).",
                                    format="%.4f%%"
                                ),
                                "Slippage Used": st.column_config.TextColumn(help="The slippage tolerance required to get this quote"),
                                "Jumper Link": st.column_config.LinkColumn("Execute Batch", display_text="Open Jumper"),
                                "Asset B/E": st.column_config.TextColumn(help="Time for this specific position's yield to pay off the transfer cost."),
                                "Portfolio B/E": st.column_config.TextColumn(help="Time for your TOTAL portfolio yield to pay off this specific transfer cost.")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
            else:
                st.info("No cross-chain or swap moves detected in this plan. Costs are negligible.")
            
        else:
            st.success("✅ Portfolio is aligned with this strategy.")
