import math

import numpy as np
from scipy.optimize import minimize

from lib.math_utils import compute_curve_multiplier, rate_per_second_to_apy


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

        res_yield = minimize(self.objective_yield, x0, method='SLSQP', bounds=standard_bounds, constraints=constraints, options=options)
        res_frontier = minimize(self.objective_frontier, x0, method='SLSQP', bounds=standard_bounds, constraints=constraints, options=options)
        res_liq = minimize(self.objective_liquidity, x0, method='SLSQP', bounds=whale_bounds, constraints=constraints, options=options)
        res_whale = minimize(self.objective_whale, x0, method='SLSQP', bounds=whale_bounds, constraints=constraints, options=options)
        
        return res_yield.x, res_frontier.x, res_liq.x, res_whale.x
