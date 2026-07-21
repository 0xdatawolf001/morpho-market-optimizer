MORPHO_API_URL = "https://api.morpho.org/graphql"
MORPHO_API_FALLBACK_URL = "https://blue-api.morpho.org/graphql"
BATCH_SIZE = 500
WAD = 1e18
TARGET_UTILIZATION = 0.9
CURVE_STEEPNESS = 4.0
SECONDS_PER_YEAR = 31536000
CACHE_TTL = 300

MAX_SCATTER_PLOT_POINTS = 5000
MAX_LINE_PLOT_POINTS_PER_STRATEGY = 1000

MANUAL_SEP = "-- Selected Markets --"
WALLET_SEP = "-- From User Wallet --"

TARGET_CHAINS = [1, 8453, 42161, 999]
SUPPORTED_CHAIN_IDS = [1, 8453, 42161, 999, 10, 130, 137, 480, 143]

CHAIN_ID_TO_NAME = {
    1: "Ethereum",
    10: "Optimism",
    130: "Unichain",
    137: "Polygon",
    143: "Monad",
    8453: "Base",
    42161: "Arbitrum",
    999: "HyperEVM",
    747474: "Katana",
    988: "Stable",
    98866: "Plume",
}

NAME_TO_CHAIN_ID = {v: k for k, v in CHAIN_ID_TO_NAME.items()}

STRATEGY_NAMES = ["Best Yield", "Whale Shield", "Frontier", "Liquid-Yield"]
STRATEGY_COLORS = {
    "Best Yield": "#F44336",
    "Whale Shield": "#2979FF",
    "Frontier": "#E040FB",
    "Liquid-Yield": "#00E676",
}

SCOPE_LABELS = [
    "1) Full Optimization",
    "2) Within Chain and Same Loan Token",
    "3) Within Chain and Different Loan Token",
    "4) Across Chain and Same Loan Token",
]

PAGE_SIZE = 25

MARKET_FIELDS_FRAGMENT = """
  marketId
  listed
  lltv
  irmAddress
  oracle { address }
  loanAsset {
    address
    symbol
    name
    decimals
    price { usd }
    chain { id }
  }
  collateralAsset {
    address
    symbol
    name
    decimals
  }
  supplyingVaults { address }
  warnings { type level }
  realizedBadDebt { underlying usd }
  badDebt { underlying usd }
  state {
    borrowAssets
    supplyAssets
    borrowAssetsUsd
    supplyAssetsUsd
    borrowShares
    supplyShares
    liquidityAssets
    liquidityAssetsUsd
    collateralAssets
    collateralAssetsUsd
    utilization
    supplyApy
    borrowApy
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


def scope_code(scope_str: str) -> int:
    for prefix, code in [("1)", 0), ("2)", 1), ("3)", 2), ("4)", 3)]:
        if prefix in scope_str:
            return code
    return 0


def monarch_link(chain_id, market_id: str) -> str:
    return f"https://www.monarchlend.xyz/market/{int(chain_id)}/{market_id}"


def safe_float(value, default=0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def format_action(action: str) -> str:
    from lib.math_utils import format_action_label

    emoji_map = {
        "DEPOSIT": "🟢 DEPOSIT",
        "WITHDRAW": "🔴 WITHDRAW",
        "HOLD": "⚪ HOLD",
        "STUCK": "⚠️ STUCK",
    }
    for key, emoji_action in emoji_map.items():
        if key in action.upper():
            return format_action_label(emoji_action)
    return format_action_label(action)
