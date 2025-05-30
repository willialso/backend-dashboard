# main_dashboard.py

import time
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import config
from backend.advanced_volatility_engine import AdvancedVolatilityEngine
from backend.advanced_pricing_engine import AdvancedPricingEngine
from data_feeds.data_feed_manager import DataFeedManager
from investor_dashboard.revenue_engine import RevenueEngine
from investor_dashboard.position_manager import PositionManager
from investor_dashboard.liquidity_manager import LiquidityManager
from investor_dashboard.bot_trader_simulator import BotTraderSimulator
from investor_dashboard.audit_engine import AuditEngine
from investor_dashboard.hedge_feed_manager import HedgeFeedManager
from investor_dashboard.dashboard_api import router as dashboard_router

# Configure root logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global component instances
data_feed_manager: Optional[DataFeedManager] = None
volatility_engine: Optional[AdvancedVolatilityEngine] = None
pricing_engine: Optional[AdvancedPricingEngine] = None
revenue_engine: Optional[RevenueEngine] = None
position_manager: Optional[PositionManager] = None
liquidity_manager: Optional[LiquidityManager] = None
bot_trader_simulator: Optional[BotTraderSimulator] = None
audit_engine: Optional[AuditEngine] = None
hedge_feed_manager: Optional[HedgeFeedManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown for backend components."""
    global data_feed_manager, volatility_engine, pricing_engine
    global revenue_engine, position_manager, liquidity_manager
    global bot_trader_simulator, audit_engine, hedge_feed_manager

    try:
        logger.info("🔧 Initializing backend components...")

        # Data feed manager
        data_feed_manager = DataFeedManager(
            enabled_exchanges=config.EXCHANGES_ENABLED,
            primary_exchange=config.PRIMARY_EXCHANGE,
            coinbase_product_id=config.COINBASE_PRODUCT_ID,
            kraken_ticker_symbol=config.KRAKEN_TICKER_SYMBOL,
            okx_ws_url=config.OKX_WS_URL,
            coinbase_ws_url=config.COINBASE_WS_URL,
            kraken_ws_url_v2=config.KRAKEN_WS_URL_V2,
            data_broadcast_interval_seconds=config.DATA_BROADCAST_INTERVAL_SECONDS,
            price_change_threshold_for_broadcast=config.PRICE_CHANGE_THRESHOLD_FOR_BROADCAST,
            price_history_max_points=config.PRICE_HISTORY_MAX_POINTS
        )

        # FIXED: Audit engine FIRST
        audit_engine = AuditEngine()
        logger.info("✅ Audit Engine initialized")

        # Volatility & pricing engines
        volatility_engine = AdvancedVolatilityEngine(
            price_history_max_points=config.PRICE_HISTORY_MAX_POINTS,
            default_vol=config.DEFAULT_VOLATILITY,
            min_vol=config.MIN_VOLATILITY,
            max_vol=config.MAX_VOLATILITY,
            ewma_alpha=config.VOLATILITY_EWMA_ALPHA,
            short_expiry_adjustments=config.VOLATILITY_SHORT_EXPIRY_ADJUSTMENTS,
            default_short_expiry_adjustment=config.VOLATILITY_DEFAULT_SHORT_EXPIRY_ADJUSTMENT,
            smile_curvature=config.VOLATILITY_SMILE_CURVATURE,
            skew_factor=config.VOLATILITY_SKEW_FACTOR,
            min_smile_adj=config.MIN_SMILE_ADJUSTMENT_FACTOR,
            max_smile_adj=config.MAX_SMILE_ADJUSTMENT_FACTOR
        )
        pricing_engine = AdvancedPricingEngine(
            volatility_engine_instance=volatility_engine,
            alpha_signal_generator_instance=None
        )

        # Position manager BEFORE revenue engine
        try:
            position_manager = PositionManager(pricing_engine_instance=pricing_engine)
        except TypeError:
            position_manager = PositionManager(pricing_engine_instance=pricing_engine)
        position_manager.pricing_engine = pricing_engine
        logger.info("✅ Position Manager initialized")

        # FIXED: Revenue engine with ALL required parameters including audit_engine_instance
        try:
            revenue_engine = RevenueEngine(
                volatility_engine_instance=volatility_engine,
                pricing_engine_instance=pricing_engine,
                position_manager_instance=position_manager,
                audit_engine_instance=audit_engine  # CRITICAL FIX: This was missing!
            )
        except TypeError as e:
            logger.warning(f"RevenueEngine constructor TypeError: {e}, trying fallback")
            # Fallback if constructor doesn't accept all parameters
            revenue_engine = RevenueEngine(
                volatility_engine_instance=volatility_engine,
                pricing_engine_instance=pricing_engine
            )
            # Manually set missing references
            revenue_engine.position_manager = position_manager
            revenue_engine.audit_engine = audit_engine

        # CRITICAL: Verify audit engine connection
        if revenue_engine.audit_engine is None:
            logger.error("CRITICAL: RevenueEngine audit_engine is None!")
            revenue_engine.audit_engine = audit_engine
            logger.warning("Fixed: Manually assigned audit_engine to revenue_engine")
        else:
            logger.info(f"✅ RevenueEngine connected to audit_engine: {type(revenue_engine.audit_engine)}")

        # Liquidity manager
        try:
            liquidity_manager = LiquidityManager(
                initial_total_pool_usd=config.LM_INITIAL_TOTAL_POOL_USD,
                initial_active_users=config.LM_INITIAL_ACTIVE_USERS,
                base_liquidity_per_user_usd=config.LM_BASE_LIQUIDITY_PER_USER_USD,
                volume_factor_per_user_usd=config.LM_VOLUME_FACTOR_PER_USER_USD,
                options_exposure_factor=config.LM_OPTIONS_EXPOSURE_FACTOR,
                stress_test_buffer_pct=config.LM_STRESS_TEST_BUFFER_PCT,
                min_liquidity_ratio=config.LM_MIN_LIQUIDITY_RATIO,
                audit_engine_instance=audit_engine
            )
        except TypeError:
            liquidity_manager = LiquidityManager()

        # Bot trader simulator
        try:
            bot_trader_simulator = BotTraderSimulator(
                revenue_engine=revenue_engine,
                position_manager=position_manager,
                data_feed_manager=data_feed_manager,
                audit_engine_instance=audit_engine
            )
        except TypeError:
            bot_trader_simulator = BotTraderSimulator()
        bot_trader_simulator.revenue_engine = revenue_engine
        bot_trader_simulator.position_manager = position_manager
        bot_trader_simulator.audit_engine = audit_engine
        bot_trader_simulator.current_btc_price = getattr(position_manager, 'current_btc_price', 108000.0)

        # Hedge feed manager
        try:
            hedge_feed_manager = HedgeFeedManager(
                position_manager_instance=position_manager,
                audit_engine_instance=audit_engine
            )
        except TypeError:
            hedge_feed_manager = HedgeFeedManager()
        hedge_feed_manager.position_manager = position_manager

        # Inject into API module
        import investor_dashboard.dashboard_api as api_mod
        api_mod.revenue_engine = revenue_engine
        api_mod.position_manager = position_manager
        api_mod.liquidity_manager = liquidity_manager
        api_mod.bot_trader_simulator = bot_trader_simulator
        api_mod.audit_engine = audit_engine
        api_mod.hedge_feed_manager = hedge_feed_manager

        # FINAL VERIFICATION: Check all connections
        logger.info("🔍 FINAL VERIFICATION:")
        logger.info(f"  - revenue_engine.audit_engine: {revenue_engine.audit_engine is not None}")
        logger.info(f"  - revenue_engine.position_manager: {revenue_engine.position_manager is not None}")
        logger.info(f"  - bot_trader_simulator.revenue_engine: {bot_trader_simulator.revenue_engine is not None}")

        # Start background services
        data_feed_manager.start()
        bot_trader_simulator.start()
        hedge_feed_manager.start()

        logger.info("✅ Backend components initialized with full audit integration")
        yield

        # Shutdown
        logger.info("🔄 Shutting down backend components")
        data_feed_manager.stop()
        bot_trader_simulator.stop()
        hedge_feed_manager.stop()
        logger.info("✅ Shutdown complete")

    except Exception as e:
        logger.error("❌ Initialization failed", exc_info=True)
        raise

# Create FastAPI app
app = FastAPI(
    title="Atticus Options Trading Platform",
    version="2.3.1",
    lifespan=lifespan
)

# CORS middleware - Explicit origins including Lovable preview
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://preview--atticus-insight-hub.lovable.app",
    "https://atticus-insight-hub.lovable.app",
    "https://atticus-demo-dashboard.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=86400,
)

# Mount dashboard router
app.include_router(dashboard_router, prefix="/api/dashboard")

@app.get("/health")
async def health_check():
    """Overall system health."""
    try:
        statuses = {
            "data_feed": data_feed_manager is not None and data_feed_manager.get_current_price() > 0,
            "position_manager": position_manager is not None,
            "revenue_engine": revenue_engine is not None,
            "pricing_engine": pricing_engine is not None,
            "liquidity_manager": liquidity_manager is not None,
            "bot_simulator": bot_trader_simulator is not None,
            "hedge_manager": hedge_feed_manager is not None,
            "audit_engine": audit_engine is not None
        }
        up = sum(statuses.values())
        total = len(statuses)
        pct = (up/total)*100
        status = "healthy" if pct >= 90 else ("degraded" if pct >= 70 else "unhealthy")
        return {
            "status": status,
            "health_percentage": round(pct,1),
            "services": statuses,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error("Health check failed", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Atticus Options Trading Platform API",
        "version": "2.3.1",
        "status": "operational",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_dashboard:app", host=config.API_HOST, port=config.API_PORT, reload=True)
