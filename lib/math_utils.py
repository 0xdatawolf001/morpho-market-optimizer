import math

import numpy as np

from lib.config import CURVE_STEEPNESS, SECONDS_PER_YEAR, TARGET_UTILIZATION


def apy_to_rate_per_second(apy_float: float) -> float:
    if apy_float <= 0:
        return 0.0
    return math.log(apy_float + 1) / SECONDS_PER_YEAR


def rate_per_second_to_apy(rate: float) -> float:
    return math.exp(rate * SECONDS_PER_YEAR) - 1


def apy_to_apr(apy_float: float) -> float:
    if apy_float <= 0:
        return 0.0
    rate_per_sec = apy_to_rate_per_second(apy_float)
    return rate_per_sec * SECONDS_PER_YEAR


def compute_curve_multiplier(utilization: float) -> float:
    if utilization <= TARGET_UTILIZATION:
        numerator = (CURVE_STEEPNESS - 1) * (utilization / TARGET_UTILIZATION) + 1
        return numerator / CURVE_STEEPNESS
    if utilization >= 1.0:
        return CURVE_STEEPNESS
    error_ratio = (utilization - TARGET_UTILIZATION) / (1 - TARGET_UTILIZATION)
    return (CURVE_STEEPNESS - 1) * error_ratio + 1


def extract_market_id_from_monarch_link(text: str) -> str:
    text = text.strip()
    if "monarchlend.xyz/market/" in text:
        parts = text.rstrip("/").split("/")
        if len(parts) >= 2 and parts[-1].startswith("0x"):
            return parts[-1].lower()
    return text.lower()


def filter_small_moves(allocations, market_data_list, threshold_usd, total_budget):
    cleaned_allocations = allocations.copy()

    for i, target_val in enumerate(cleaned_allocations):
        current_balance = market_data_list[i]["existing_balance_usd"]
        diff = abs(target_val - current_balance)
        if diff > 0.01 and diff < threshold_usd:
            cleaned_allocations[i] = current_balance

    current_total = np.sum(cleaned_allocations)
    unallocated = total_budget - current_total

    if unallocated > 0.01 and unallocated < threshold_usd:
        candidates = [
            i
            for i, target_val in enumerate(cleaned_allocations)
            if target_val < market_data_list[i]["existing_balance_usd"]
        ]
        if candidates:
            cleaned_allocations[candidates[0]] += unallocated
        else:
            cleaned_allocations[int(np.argmax(cleaned_allocations))] += unallocated

    return cleaned_allocations


def format_lltv(lltv_raw) -> float:
    try:
        return float(lltv_raw) / 1e18
    except (TypeError, ValueError):
        return 0.0


def format_action_label(action: str) -> str:
    mapping = {
        "🟢 DEPOSIT": "Allocate",
        "🔴 WITHDRAW": "Reduce",
        "⚪ HOLD": "Hold",
        "⚠️ STUCK": "Liquidity-limited",
    }
    return mapping.get(action, action)
