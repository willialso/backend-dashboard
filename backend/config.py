# backend/config.py

import os
import json
from typing import List, Dict, Any
import logging

# === LOGGER SETUP (MUST BE AT THE TOP BEFORE ANY LOGGER CALLS) ===
config_logger = logging.getLogger(__name__)

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - CONFIG - %(levelname)s - %(message)s")

# =================================================================

# === CORE PLATFORM SETTINGS ===
PLATFORM_NAME = "Atticus"
VERSION = "2.3.1"
DEMO_MODE = True
DEBUG_MODE = True

# === API & SERVER SETTINGS ===
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("PORT", 8001))
LOVABLE_PREVIEW_URL = "https://preview--atticus-insight-hub.lovable.app"
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    LOVABLE_PREVIEW_URL,
    "https://atticus-insight-hub.lovable.app",
]

RENDER_BACKEND_URL = "https://atticus-demo-dashboard.onrender.com"
if RENDER_BACKEND_URL and RENDER_BACKEND_URL not in CORS_ORIGINS:
    CORS_ORIGINS.append(RENDER_BACKEND_URL)

API_STARTUP_TIMEOUT = 30
WEBSOCKET_UPDATE_INTERVAL_SECONDS = 2.0
BACKGROUND_UPDATE_INTERVAL_SECONDS = 1.0
WEBSOCKET_TIMEOUT_SECONDS = 60

# === DATA FEED SETTINGS ===
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
KRAKEN_WS_URL_V2 = "wss://ws.kraken.com/v2"
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
COINBASE_PRODUCT_ID = "BTC-USD"
KRAKEN_TICKER_SYMBOL = "BTC/USD"
EXCHANGES_ENABLED = ["okx", "coinbase", "kraken"]
PRIMARY_EXCHANGE = "okx"
DATA_BROADCAST_INTERVAL_SECONDS = 1.0
PRICE_HISTORY_MAX_POINTS = 10000
PRICE_CHANGE_THRESHOLD_FOR_BROADCAST = 0.0001

# === CONTRACT SPECIFICATIONS ===
STANDARD_CONTRACT_SIZE_BTC = 0.1
CONTRACT_SIZES_AVAILABLE = [0.01, 0.1, 1.0]

# === EXPIRY CONFIGURATIONS ===
AVAILABLE_EXPIRIES_MINUTES = [5, 15, 30, 60, 120, 240, 480, 1440]
EXPIRY_LABELS = {
    5: "5 Minute", 15: "15 Minute", 30: "30 Minute", 60: "1 Hour", 120: "2 Hour",
    240: "4 Hour", 480: "8 Hour", 1440: "1 Day"
}

# === PRICING ENGINE & VOLATILITY SETTINGS ===
RISK_FREE_RATE = 0.05
MIN_VOLATILITY = 0.15
MAX_VOLATILITY = 3.00
DEFAULT_VOLATILITY = 0.80
VOLATILITY_EWMA_ALPHA = 0.06
VOLATILITY_REGIME_MULTIPLIER_LOW = 0.85
VOLATILITY_REGIME_MULTIPLIER_MEDIUM = 1.00
VOLATILITY_REGIME_MULTIPLIER_HIGH = 1.25
VOLATILITY_SHORT_EXPIRY_ADJUSTMENTS = {
    5: 1.50, 15: 1.30, 30: 1.15, 60: 1.05, 120: 1.02
}
VOLATILITY_DEFAULT_SHORT_EXPIRY_ADJUSTMENT = 1.0
VOLATILITY_SMILE_CURVATURE = 0.15
VOLATILITY_SKEW_FACTOR = -0.10
MIN_SMILE_ADJUSTMENT_FACTOR = 0.70
MAX_SMILE_ADJUSTMENT_FACTOR = 1.50
VOLATILITY_GARCH_ENABLED = False

# === STRIKE GENERATION ===
STRIKE_RANGES_BY_EXPIRY = {
    5: {"num_one_side": 5, "step_pct": 0.005},
    15: {"num_one_side": 7, "step_pct": 0.005},
    30: {"num_one_side": 8, "step_pct": 0.01},
    60: {"num_one_side": 10, "step_pct": 0.01},
    120: {"num_one_side": 10, "step_pct": 0.015},
    240: {"num_one_side": 12, "step_pct": 0.02},
    480: {"num_one_side": 15, "step_pct": 0.03},
    1440: {"num_one_side": 20, "step_pct": 0.04}
}
STRIKE_ROUNDING_NEAREST = 10
PRICE_ROUNDING_DP = 2

# === REVENUE ENGINE ===
REVENUE_BASE_MARKUP_PERCENTAGE = 0.035
STANDARD_METRICS_EXPIRY_MINUTES = 60
ESTIMATED_DAILY_CONTRACTS_SOLD = 150

# === BOT TRADER SIMULATOR ===
TRADER_DIST_ADV_PCT = 50.0
TRADER_DIST_INT_PCT = 35.0
TRADER_DIST_BEG_PCT = 15.0
BASE_TOTAL_SIMULATED_TRADERS = 50
TRADER_COUNT_ADV = int(BASE_TOTAL_SIMULATED_TRADERS * (TRADER_DIST_ADV_PCT / 100))
TRADER_COUNT_INT = int(BASE_TOTAL_SIMULATED_TRADERS * (TRADER_DIST_INT_PCT / 100))
TRADER_COUNT_BEG = BASE_TOTAL_SIMULATED_TRADERS - TRADER_COUNT_ADV - TRADER_COUNT_INT
BOT_SIM_LOOP_MIN_SLEEP_SEC = 50
BOT_SIM_LOOP_MAX_SLEEP_SEC = 70
MAX_RECENT_TRADES_LOG_SIZE_BOTSIM = 250
BOT_OTM_MIN_FACTOR = 0.005
BOT_OTM_MAX_FACTORS = {
    "Beginner": 0.07, "Intermediate": 0.05, "Advanced": 0.03
}

# === BOT TRADING CONFIGURATION (FIXED MISSING VARIABLE) ===
TRADE_SIMULATION_INTERVAL_SECONDS = 30  # Bot trader checks for new trades every 30 seconds

# === HEDGE FEED MANAGER & POSITION MANAGER ===
MAX_PLATFORM_NET_DELTA_BTC = 0.5
HEDGE_LOOP_MIN_SLEEP_SEC = 8
HEDGE_LOOP_MAX_SLEEP_SEC = 25
HEDGE_PROBABILISTIC_TRIGGER_PCT = 0.05
HEDGE_DELTA_PROPORTION_MIN = 0.6
HEDGE_DELTA_PROPORTION_MAX = 0.9
HEDGE_MIN_SIZE_BTC = 0.01
HEDGE_MAX_SIZE_BTC = 2.5
HEDGE_NON_DELTA_MAX_SIZE_BTC = 0.5
HEDGE_QUANTITY_ROUNDING_DP = 4
HEDGE_INSTRUMENT_SPOT = "BTC-SPOT"
HEDGE_INSTRUMENT_PERP = "BTC-PERP"
HEDGE_SLIPPAGE_MIN_PCT = -0.0003
HEDGE_SLIPPAGE_MAX_PCT = 0.0003
HEDGE_TRANSACTION_FEE_PCT = 0.0005
EXCHANGE_NAME_COINBASE_PRO = "Coinbase Pro"
EXCHANGE_NAME_KRAKEN = "Kraken"
EXCHANGE_NAME_OKX = "OKX"
EXCHANGE_NAME_DERIBIT = "Deribit"
HEDGE_EXCHANGE_WEIGHTS = {
    EXCHANGE_NAME_COINBASE_PRO: 0.4,
    EXCHANGE_NAME_KRAKEN: 0.25,
    EXCHANGE_NAME_OKX: 0.20,
    EXCHANGE_NAME_DERIBIT: 0.15
}
HEDGE_EXECUTION_TIMES_MS = {
    EXCHANGE_NAME_COINBASE_PRO: 180, EXCHANGE_NAME_KRAKEN: 220,
    EXCHANGE_NAME_OKX: 150, EXCHANGE_NAME_DERIBIT: 120,
    "Simulated Internal": 10
}
MAX_RECENT_HEDGES_LOG_SIZE = 150

# === LIQUIDITY MANAGER (FIXED MISSING VARIABLES) ===
LM_INITIAL_TOTAL_POOL_USD = 1500000.0
LM_INITIAL_ACTIVE_USERS = BASE_TOTAL_SIMULATED_TRADERS
LM_BASE_LIQUIDITY_PER_USER_USD = 10000.0
LM_VOLUME_FACTOR_PER_USER_USD = 500.0
LM_OPTIONS_EXPOSURE_FACTOR = 0.25
LM_STRESS_TEST_BUFFER_PCT = 0.20
LM_MIN_LIQUIDITY_RATIO = 0.20  # FIXED: Added missing variable
LM_MAX_LIQUIDITY_RATIO = 0.80  # FIXED: Added missing variable
MIN_LIQUIDITY_RATIO = 0.20     # FIXED: Added missing variable for audit engine
MAX_LIQUIDITY_RATIO = 0.80     # FIXED: Added missing variable for audit engine
LM_PROFIT_ALLOCATION_PCT = 0.05
LM_OPERATIONS_ALLOCATION_PCT = 0.10
LM_LIQUIDITY_REINVESTMENT_PCT = 0.85

# === AUDIT ENGINE CONFIGURATION (FIXED MISSING VARIABLES) ===
AUDIT_ENABLED = True
AUDIT_LOG_RETENTION_DAYS = 90
AUDIT_COMPLIANCE_THRESHOLD = 95.0
AUDIT_CRITICAL_ISSUE_THRESHOLD = 0
AUDIT_WARNING_THRESHOLD = 5

# === RISK MANAGEMENT ===
MAX_SINGLE_USER_EXPOSURE_BTC = 10.0
MARGIN_REQUIREMENT_MULTIPLIER = 1.5

# === ALPHA SIGNAL & ML SETTINGS ===
ALPHA_SIGNALS_ENABLED = False
BAR_PORTION_LOOKBACK = 20
REGIME_DETECTION_LOOKBACK = 100
MACD_FAST = 12; MACD_SLOW = 26; MACD_SIGNAL = 9
REGIME_TRAINING_INTERVAL = 1000
SENTIMENT_ANALYSIS_ENABLED = False
USE_RL_HEDGER = False
USE_ML_VOLATILITY = False
REGIME_DETECTION_ENABLED = True

# === LOGGING & DATABASE ===
LOG_LEVEL = "INFO"
LOG_FILE = "logs/atticus_platform.log"
DATABASE_URL = "sqlite:///./atticus_platform_data.db"

# === MISC ===
ATTICUS_BACKGROUND_TASK_INTERVAL = 30

# === CONFIGURATION HELPERS ===
def get_config_value(key: str, default: Any = None) -> Any:
    return globals().get(key, default)

def update_config(key: str, value: Any):
    globals()[key] = value
    config_logger.info(f"CONFIG: Updated '{key}' to '{value}' at runtime.")

# === ENVIRONMENT-BASED OVERRIDES ===
config_logger.info("CONFIG: Checking for environment variable overrides...")

_PREFIX = "ATTICUS_"
for key, current_value in list(globals().items()):
    if key.isupper() and isinstance(current_value, (str, int, float, bool, list, dict)):
        env_key = f"{_PREFIX}{key}"
        env_value_str = os.environ.get(env_key)
        if env_value_str is not None:
            try:
                new_value: Any
                if isinstance(current_value, bool):
                    new_value = env_value_str.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int):
                    new_value = int(env_value_str)
                elif isinstance(current_value, float):
                    new_value = float(env_value_str)
                elif isinstance(current_value, list):
                    new_value = [item.strip() for item in env_value_str.split(',')]
                elif isinstance(current_value, dict):
                    new_value = json.loads(env_value_str)
                else:
                    new_value = env_value_str

                globals()[key] = new_value
                config_logger.info(f"CONFIG: Overrode '{key}' with env var '{env_key}' = '{new_value}' (was '{current_value}')")

            except (ValueError, TypeError, json.JSONDecodeError) as e:
                config_logger.warning(f"CONFIG: Could not cast env var {env_key}='{env_value_str}' to type {type(current_value)}. Error: {e}")

config_logger.info("CONFIG: Environment variable override check complete.")
