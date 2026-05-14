# API Services Module
# Handles all external API interactions including LI.FI, Morpho GraphQL, and other services.

import requests
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

# Configuration
MORPHO_API_URL = "https://api.morpho.org/graphql"
BATCH_SIZE = 1000
WAD = 1e18
TARGET_UTILIZATION = 0.9
CURVE_STEEPNESS = 4.0

@dataclass
class MarketData:
    """Represents structured market data from Morpho API."""
    market_id: str
    chain: str
    loan_token: str
    loan_address: str
    collateral: str
    decimals: int
    price_usd: float
    supply_apy: float
    utilization: float
    supply_usd: float
    borrow_usd: float
    available_usd: float
    whitelisted: bool

class APIServiceError(Exception):
    """Custom exception for API service errors."""
    pass
    """Represents structured market data from Morpho API."""
    market_id: str
    chain: str
    loan_token: str
    loan_address: str
    collateral: str
    decimals: int
    price_usd: float
    supply_apy: float
    utilization: float
    supply_usd: float
    borrow_usd: float
    available_usd: float
    whitelisted: bool

class APIServiceError(Exception):
    """Custom exception for API service errors."""
    pass