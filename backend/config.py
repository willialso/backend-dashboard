# backend/config.py

import os
from typing import List, Dict

# === CORE PLATFORM SETTINGS ===
PLATFORM_NAME = "Atticus"
VERSION = "2.2" # Version updated to reflect new engine capabilities
DEMO_MODE = True

# === WEBSOCKET FEED URLS === (FIXED: Use alternative Coinbase endpoint)
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"  # ← FIXED: Changed from blocked endpoint
KRAKEN_WS_URL_V2 = "wss://ws.kraken.com/v2"
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

# === EXCHANGE-SPECIFIC PRODUCT/TICKER SYMBOLS ===
COINBASE_PRODUCT_ID = "BTC-USD"           # Required by coinbase_client.py
KRAKEN_TICKER_SYMBOL = "BTC/USD"          # Required by kraken_client.py (v2 format)

# Backup URLs for reference
COINBASE_WS_URL_BACKUP = "wss://ws-feed.pro.coinbase.com"        # CloudFlare blocked
COINBASE_WS_URL_ALTERNATIVE = "wss://ws-feed.exchange.coinbase.com"  # Working alternative

# === CONTRACT SPECIFICATIONS ===
# *** Updated contract size to 0.1 BTC for institutional standard ***
STANDARD_CONTRACT_SIZE_BTC = 0.1 # Each contract represents 0.1 BTC
CONTRACT_SIZES_AVAILABLE = [0.1] # Institutional contract size

# === EXPIRY CONFIGURATIONS ===
# *** UPDATED: Removed 5-minute "gambling" expiry for professional credibility ***
AVAILABLE_EXPIRIES_MINUTES = [
    15, # 15 minutes - shortest professional timeframe
    60, # 1 hour
    240, # 4 hours
    480 # 8 hours
]

# *** UPDATED: Removed "Turbo" label, keeping professional naming ***
EXPIRY_LABELS = {
    15: "Express",
    60: "Hourly",
    240: "4-Hour",
    480: "8-Hour"
}

# === PRICING ENGINE SETTINGS ===
RISK_FREE_RATE = 0.05 # 5% annual risk-free rate

# Adjusted for more realistic smile effects and broader range for BTC
MIN_VOLATILITY = 0.15 # 15% minimum annualized volatility floor
MAX_VOLATILITY = 3.00 # 300% maximum volatility cap
DEFAULT_VOLATILITY = 0.80 # 80% default if calculation fails or for initial warm-up
DEFAULT_VOLATILITY_FOR_BASIC_BS = 0.80 # If using a very basic BS model elsewhere

# Advanced volatility settings
VOLATILITY_REGIME_DETECTION = True
VOLATILITY_EWMA_ALPHA = 0.1 # Decay factor for EWMA volatility

# Regime multipliers (can be tuned based on market observation)
VOLATILITY_REGIME_MULTIPLIER_LOW = 0.85 # Applied to EWMA vol in low vol regime
VOLATILITY_REGIME_MULTIPLIER_MEDIUM = 1.0 # Applied to EWMA vol in medium vol regime
VOLATILITY_REGIME_MULTIPLIER_HIGH = 1.25 # Applied to EWMA vol in high vol regime

# Short-term expiry volatility adjustments (applied to ATM vol before smile/skew)
# These factors boost ATM volatility for options very close to expiry.
VOLATILITY_SHORT_EXPIRY_ADJUSTMENTS = { # expiry_minutes: multiplier
    15: 1.15, # For 15 min expiry, boost ATM vol by 15%
    60: 1.05 # For 1 hour expiry, boost ATM vol by 5%
    # Add other expiries if specific short-term boosts are desired
}

# Fallback multiplier if an expiry is not explicitly listed above (no adjustment)
VOLATILITY_DEFAULT_SHORT_EXPIRY_ADJUSTMENT = 1.0

# Volatility Smile/Skew Parameters (applied to ATM vol for a given expiry)
# Model: Vol_strike = ATM_Vol_expiry * (1 + CURVATURE * ln(K/S)^2 + SKEW * ln(K/S))
# These parameters define the general shape of the smile/skew.
# Positive CURVATURE creates a "smile" (higher vol for OTM/ITM).
# Negative SKEW_FACTOR typically creates "reverse skew" (OTM puts > OTM calls IV), common in equity/crypto.
VOLATILITY_SMILE_CURVATURE = 0.15 # Higher value = more pronounced smile. Example: 0.1 to 0.3
VOLATILITY_SKEW_FACTOR = -0.10 # Example: -0.05 to -0.2 for crypto/equity.

# Bounds for the smile/skew multiplicative adjustment factor.
# This prevents the smile/skew model from pushing strike-specific volatility to absurd levels
# relative to the adjusted ATM volatility for that expiry.
MIN_SMILE_ADJUSTMENT_FACTOR = 0.70 # Volatility for a strike won't drop below 70% of its expiry's ATM vol due to smile/skew
MAX_SMILE_ADJUSTMENT_FACTOR = 1.50 # Volatility for a strike won't exceed 150% of its expiry's ATM vol due to smile/skew

# GARCH and ML Volatility flags (can be enabled if respective components are implemented)
VOLATILITY_GARCH_ENABLED = False # Set to True if GARCH models are integrated
ML_VOL_TRAINING_INTERVAL = 500 # Example: Retrain ML vol model every 500 data points
PRICE_CHANGE_THRESHOLD_FOR_BROADCAST = 0.0001 # Threshold for broadcasting price updates

# === STRIKE GENERATION ===
# *** UPDATED: Removed 5-minute strike config, optimized for professional expiries ***
STRIKE_RANGES_BY_EXPIRY = {
    15: {"num_itm": 7, "num_otm": 7, "step_pct": 0.005}, # Express - tight spacing
    60: {"num_itm": 10, "num_otm": 10, "step_pct": 0.01}, # Hourly - moderate spacing
    240: {"num_itm": 12, "num_otm": 12, "step_pct": 0.02}, # 4-Hour - wider spacing
    480: {"num_itm": 15, "num_otm": 15, "step_pct": 0.03} # 8-Hour - widest spacing
}

STRIKE_ROUNDING_NEAREST = 10 # Round strikes to nearest $10

# === DATA FEED SETTINGS === (FIXED: Prioritize OKX)
EXCHANGES_ENABLED = ["okx", "coinbase", "kraken"]  # ← FIXED: OKX first (most real-time)
PRIMARY_EXCHANGE = "okx"                           # ← FIXED: Changed from coinbase to okx
DATA_BROADCAST_INTERVAL_SECONDS = 1.0 # Interval at which price data is assumed to arrive/be processed
PRICE_HISTORY_MAX_POINTS = 10000 # Max number of price points to store for volatility calculations

# === ALPHA SIGNAL SETTINGS ===
ALPHA_SIGNALS_ENABLED = False # Set to True to enable alpha signal adjustments in pricing
BAR_PORTION_LOOKBACK = 20
REGIME_DETECTION_LOOKBACK = 100 # Lookback for regime detection in alpha signals
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
REGIME_TRAINING_INTERVAL = 1000

# === FEATURE FLAGS ===
SENTIMENT_ANALYSIS_ENABLED = False
USE_RL_HEDGER = False
USE_ML_VOLATILITY = False # Set to True if ML-based volatility forecasting is implemented and active
REGIME_DETECTION_ENABLED = True # General flag for regime detection features (volatility engine also uses its own)

# === HEDGING SIMULATION ===
HEDGING_ENABLED = True
DELTA_HEDGE_FREQUENCY_MINUTES = 5
HEDGE_SLIPPAGE_BPS = 2 # Basis points

# === RISK MANAGEMENT ===
MAX_SINGLE_USER_EXPOSURE_BTC = 10.0
MAX_PLATFORM_NET_DELTA_BTC = 100.0
MARGIN_REQUIREMENT_MULTIPLIER = 1.5

# === API SETTINGS ===
API_PORT = 8000
API_HOST = "localhost"
CORS_ORIGINS = [
    "*", # Allow all for local development, restrict in production
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://preview--atticus-option-flow.lovable.app"
]

API_STARTUP_TIMEOUT = 20
WEBSOCKET_TIMEOUT_SECONDS = 30

# === LOGGING ===
LOG_LEVEL = "INFO" # Can be "DEBUG" for more verbose output
LOG_FILE = "logs/atticus.log"

# === DATABASE ===
DATABASE_URL = "sqlite:///./atticus_demo.db"

# === ATTICUS BACKEND SETTINGS ===
ATTICUS_BACKGROUND_TASK_INTERVAL = 30 # Seconds

# === CONFIGURATION HELPERS ===
def get_config_value(key: str, default=None):
    """Helper function to get config values with defaults."""
    return globals().get(key, default)

def update_config(key: str, value):
    """Helper function to update config values at runtime (use with caution)."""
    globals()[key] = value

# === ENVIRONMENT-BASED OVERRIDES ===
# This section allows overriding config values using environment variables
# E.g., ATTICUS_RISK_FREE_RATE=0.04 will override RISK_FREE_RATE
# Ensure environment variable names are prefixed with "ATTICUS_"
for key, value in globals().copy().items():
    if key.isupper() and isinstance(value, (str, int, float, bool, list, dict)): # Added dict support
        env_value = os.environ.get(f"ATTICUS_{key}")
        if env_value is not None:
            current_value = globals()[key]
            try:
                if isinstance(current_value, bool):
                    globals()[key] = env_value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int):
                    globals()[key] = int(env_value)
                elif isinstance(current_value, float):
                    globals()[key] = float(env_value)
                elif isinstance(current_value, list):
                    # Assumes comma-separated values for lists, may need more robust parsing for complex lists
                    globals()[key] = [item.strip() for item in env_value.split(',')]
                elif isinstance(current_value, dict):
                    # For dicts, expects a JSON string in the env var.
                    import json
                    globals()[key] = json.loads(env_value)
                else: # Handles string type
                    globals()[key] = env_value
            except (ValueError, TypeError, json.JSONDecodeError) as e: # Added JSONDecodeError
                print(f"Warning: Could not cast env var ATTICUS_{key}='{env_value}' to type {type(current_value)}. Error: {e}")
