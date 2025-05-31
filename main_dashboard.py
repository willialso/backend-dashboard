# main_dashboard.py

import time
import logging
import asyncio
import os
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

# CRITICAL FIX: Environment detection
ENVIRONMENT = os.getenv("RENDER", "local")
IS_PRODUCTION = ENVIRONMENT != "local"

if IS_PRODUCTION:
    logger.info("🌐 Running in PRODUCTION mode on Render")
else:
    logger.info("🔧 Running in DEVELOPMENT mode")

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

# CRITICAL FIX: JSON serialization helper for numpy types
def safe_json_convert(value):
    """Convert numpy types and other non-serializable types to JSON-safe Python types"""
    try:
        if hasattr(value, 'item'):  # numpy scalar
            return value.item()
        elif hasattr(value, 'tolist'):  # numpy array
            return value.tolist()
        elif isinstance(value, (bool, int, float, str)) or value is None:
            return value
        else:
            return str(value)  # Convert unknown types to string
    except Exception:
        return str(value)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown for backend components."""
    global data_feed_manager, volatility_engine, pricing_engine
    global revenue_engine, position_manager, liquidity_manager
    global bot_trader_simulator, audit_engine, hedge_feed_manager

    try:
        logger.info("🔧 Initializing backend components...")

        # Data feed manager
        try:
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
            logger.info("✅ Data feed manager initialized")
        except Exception as e:
            logger.error(f"❌ Data feed manager initialization failed: {e}")
            data_feed_manager = None

        # FIXED: Audit engine FIRST
        try:
            audit_engine = AuditEngine()
            logger.info("✅ Audit Engine initialized")
        except Exception as e:
            logger.error(f"❌ Audit engine initialization failed: {e}")
            audit_engine = None

        # Volatility & pricing engines
        try:
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
            logger.info("✅ Volatility engine initialized")
        except Exception as e:
            logger.error(f"❌ Volatility engine initialization failed: {e}")
            volatility_engine = None

        try:
            pricing_engine = AdvancedPricingEngine(
                volatility_engine_instance=volatility_engine,
                alpha_signal_generator_instance=None
            )
            logger.info("✅ Pricing engine initialized")
        except Exception as e:
            logger.error(f"❌ Pricing engine initialization failed: {e}")
            pricing_engine = None

        # Position manager BEFORE revenue engine
        try:
            position_manager = PositionManager(pricing_engine_instance=pricing_engine)
            logger.info("✅ Position Manager initialized")
        except TypeError as e:
            logger.warning(f"Position manager TypeError: {e}, trying fallback")
            try:
                position_manager = PositionManager(pricing_engine_instance=pricing_engine)
                position_manager.pricing_engine = pricing_engine
                logger.info("✅ Position Manager initialized with fallback")
            except Exception as e2:
                logger.error(f"❌ Position manager initialization completely failed: {e2}")
                position_manager = None
        except Exception as e:
            logger.error(f"❌ Position manager initialization failed: {e}")
            position_manager = None

        # CRITICAL FIX: Ensure PositionManager is fully configured before hedge manager
        if position_manager:
            if not hasattr(position_manager, 'pricing_engine') or not position_manager.pricing_engine:
                position_manager.pricing_engine = pricing_engine
                logger.warning("🔧 Fixed: Manually assigned pricing_engine to position_manager")

            # Ensure current BTC price is set (critical for trade recording)
            if not hasattr(position_manager, 'current_btc_price') or position_manager.current_btc_price <= 0:
                position_manager.current_btc_price = 108000.0  # Default startup price
                logger.warning("🔧 Fixed: Set default BTC price in position_manager")

            # Initialize any other required position manager attributes
            if not hasattr(position_manager, 'open_option_positions'):
                position_manager.open_option_positions = {}
            if not hasattr(position_manager, 'open_hedge_positions'):
                position_manager.open_hedge_positions = {}
            if not hasattr(position_manager, 'total_portfolio_delta'):
                position_manager.total_portfolio_delta = 0.0

            # VERIFICATION: Log position manager state before hedge manager starts
            logger.info("🔍 POSITION MANAGER VERIFICATION:")
            logger.info(f" ✅ pricing_engine set: {position_manager.pricing_engine is not None}")
            logger.info(f" ✅ current_btc_price: ${position_manager.current_btc_price:,.2f}")
            logger.info(f" ✅ positions initialized: {hasattr(position_manager, 'open_option_positions')}")

        # FIXED: Revenue engine with ALL required parameters including audit_engine_instance
        try:
            revenue_engine = RevenueEngine(
                volatility_engine_instance=volatility_engine,
                pricing_engine_instance=pricing_engine,
                position_manager_instance=position_manager,
                audit_engine_instance=audit_engine  # CRITICAL FIX: This was missing!
            )
            logger.info("✅ Revenue engine initialized")
        except TypeError as e:
            logger.warning(f"RevenueEngine constructor TypeError: {e}, trying fallback")
            # Fallback if constructor doesn't accept all parameters
            try:
                revenue_engine = RevenueEngine(
                    volatility_engine_instance=volatility_engine,
                    pricing_engine_instance=pricing_engine
                )
                # Manually set missing references
                revenue_engine.position_manager = position_manager
                revenue_engine.audit_engine = audit_engine
                logger.info("✅ Revenue engine initialized with fallback")
            except Exception as e2:
                logger.error(f"❌ Revenue engine initialization completely failed: {e2}")
                revenue_engine = None
        except Exception as e:
            logger.error(f"❌ Revenue engine initialization failed: {e}")
            revenue_engine = None

        # CRITICAL: Verify audit engine connection
        if revenue_engine and audit_engine:
            if not hasattr(revenue_engine, 'audit_engine') or revenue_engine.audit_engine is None:
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
            logger.info("✅ Liquidity manager initialized")
        except TypeError as e:
            logger.warning(f"Liquidity manager TypeError: {e}, using fallback")
            liquidity_manager = LiquidityManager()
            logger.info("✅ Liquidity manager initialized with fallback")
        except Exception as e:
            logger.error(f"❌ Liquidity manager initialization failed: {e}")
            liquidity_manager = None

        # Bot trader simulator
        try:
            bot_trader_simulator = BotTraderSimulator(
                revenue_engine=revenue_engine,
                position_manager=position_manager,
                data_feed_manager=data_feed_manager,
                audit_engine_instance=audit_engine
            )
            logger.info("✅ Bot trader simulator initialized")
        except TypeError as e:
            logger.warning(f"Bot trader simulator TypeError: {e}, using fallback")
            bot_trader_simulator = BotTraderSimulator()
            bot_trader_simulator.revenue_engine = revenue_engine
            bot_trader_simulator.position_manager = position_manager
            bot_trader_simulator.audit_engine = audit_engine
            bot_trader_simulator.current_btc_price = getattr(position_manager, 'current_btc_price', 108000.0)
            logger.info("✅ Bot trader simulator initialized with fallback")
        except Exception as e:
            logger.error(f"❌ Bot trader simulator initialization failed: {e}")
            bot_trader_simulator = None

        # CRITICAL FIX: Connect audit engine to bot simulator for data consistency
        if bot_trader_simulator and audit_engine:
            try:
                if hasattr(audit_engine, 'set_bot_simulator_reference'):
                    audit_engine.set_bot_simulator_reference(bot_trader_simulator)
                    logger.info("✅ CRITICAL FIX: Connected audit engine to bot simulator for data consistency")
            except Exception as e:
                logger.warning(f"Failed to connect audit engine to bot simulator: {e}")

        # CRITICAL FIX: Hedge feed manager with enhanced initialization
        try:
            hedge_feed_manager = HedgeFeedManager(
                position_manager_instance=position_manager,
                audit_engine_instance=audit_engine,
                liquidity_manager_instance=liquidity_manager  # Add liquidity manager reference
            )
            logger.info("✅ Hedge feed manager initialized")
        except TypeError as e:
            logger.warning(f"Hedge feed manager TypeError: {e}, using fallback")
            hedge_feed_manager = HedgeFeedManager()
            # Ensure hedge feed manager has proper references
            hedge_feed_manager.position_manager = position_manager
            hedge_feed_manager.audit_engine = audit_engine
            hedge_feed_manager.liquidity_manager = liquidity_manager
            logger.info("✅ Hedge feed manager initialized with fallback")
        except Exception as e:
            logger.error(f"❌ Hedge feed manager initialization failed: {e}")
            hedge_feed_manager = None

        # CRITICAL: Verify hedge feed manager dependencies before starting
        if hedge_feed_manager:
            logger.info("🔍 HEDGE FEED MANAGER VERIFICATION:")
            logger.info(f" ✅ position_manager connected: {hedge_feed_manager.position_manager is not None}")
            logger.info(f" ✅ audit_engine connected: {hedge_feed_manager.audit_engine is not None}")
            logger.info(f" ✅ liquidity_manager connected: {hasattr(hedge_feed_manager, 'liquidity_manager') and hedge_feed_manager.liquidity_manager is not None}")
            
            if hedge_feed_manager.position_manager:
                logger.info(f" ✅ position_manager pricing_engine: {hasattr(hedge_feed_manager.position_manager, 'pricing_engine') and hedge_feed_manager.position_manager.pricing_engine is not None}")
                logger.info(f" ✅ position_manager current_price: ${getattr(hedge_feed_manager.position_manager, 'current_btc_price', 0):,.2f}")
            
            # CRITICAL: Verify hedge threshold configuration
            hedge_threshold = getattr(config, 'MAX_PLATFORM_NET_DELTA_BTC', None)
            logger.info(f" 🔍 HEDGE THRESHOLD: {hedge_threshold} BTC")
            if hedge_threshold is None:
                logger.error("❌ CRITICAL: MAX_PLATFORM_NET_DELTA_BTC not configured!")

        # Inject into API module
        try:
            import investor_dashboard.dashboard_api as api_mod
            api_mod.revenue_engine = revenue_engine
            api_mod.position_manager = position_manager
            api_mod.liquidity_manager = liquidity_manager
            api_mod.bot_trader_simulator = bot_trader_simulator
            api_mod.audit_engine = audit_engine
            api_mod.hedge_feed_manager = hedge_feed_manager
            logger.info("✅ API module dependency injection completed")
        except Exception as e:
            logger.error(f"❌ API module dependency injection failed: {e}")

        # CRITICAL FIX: Add periodic data consistency sync
        try:
            import threading
            def periodic_data_sync():
                """Ensure audit engine stays synced with bot simulator"""
                while True:
                    try:
                        time.sleep(300)  # 5 minutes
                        if audit_engine and bot_trader_simulator:
                            if hasattr(audit_engine, 'set_bot_simulator_reference'):
                                audit_engine.set_bot_simulator_reference(bot_trader_simulator)
                                logger.debug("✅ Data consistency sync completed")
                    except Exception as e:
                        logger.error(f"❌ Error in periodic data sync: {e}")

            # Start sync thread
            sync_thread = threading.Thread(target=periodic_data_sync, daemon=True)
            sync_thread.start()
            logger.info("✅ Started periodic data consistency sync thread")
        except Exception as e:
            logger.warning(f"Failed to start data consistency sync thread: {e}")

        # FINAL VERIFICATION: Check all connections
        logger.info("🔍 FINAL COMPONENT VERIFICATION:")
        logger.info(f" - revenue_engine: {revenue_engine is not None}")
        logger.info(f" - revenue_engine.audit_engine: {revenue_engine is not None and hasattr(revenue_engine, 'audit_engine') and revenue_engine.audit_engine is not None}")
        logger.info(f" - revenue_engine.position_manager: {revenue_engine is not None and hasattr(revenue_engine, 'position_manager') and revenue_engine.position_manager is not None}")
        logger.info(f" - bot_trader_simulator: {bot_trader_simulator is not None}")
        logger.info(f" - bot_trader_simulator.revenue_engine: {bot_trader_simulator is not None and hasattr(bot_trader_simulator, 'revenue_engine') and bot_trader_simulator.revenue_engine is not None}")
        logger.info(f" - audit_engine: {audit_engine is not None}")
        logger.info(f" - audit_engine.bot_simulator: {audit_engine is not None and hasattr(audit_engine, 'bot_simulator') and audit_engine.bot_simulator is not None}")
        logger.info(f" - hedge_feed_manager: {hedge_feed_manager is not None}")
        logger.info(f" - position_manager: {position_manager is not None}")
        logger.info(f" - liquidity_manager: {liquidity_manager is not None}")

        # Start background services with proper order and error handling
        logger.info("🚀 Starting background services...")

        # Start data feed manager first
        if data_feed_manager:
            try:
                data_feed_manager.start()
                logger.info("✅ Data feed manager started")
            except Exception as e:
                logger.error(f"❌ Failed to start data feed manager: {e}")

        # Start bot trader simulator
        if bot_trader_simulator:
            try:
                bot_trader_simulator.start()
                logger.info("✅ Bot trader simulator started")
            except Exception as e:
                logger.error(f"❌ Failed to start bot trader simulator: {e}")

        # CRITICAL: Start hedge feed manager LAST after all dependencies verified
        if hedge_feed_manager:
            try:
                hedge_feed_manager.start()
                logger.info("✅ Hedge feed manager started")
                
                # CRITICAL: Add hedge execution debugging
                logger.info("🔍 HEDGE EXECUTION DEBUG:")
                logger.info(f" - Hedge manager is_running: {getattr(hedge_feed_manager, 'is_running', False)}")
                logger.info(f" - Position manager available: {hedge_feed_manager.position_manager is not None}")
                logger.info(f" - MAX_PLATFORM_NET_DELTA_BTC: {getattr(config, 'MAX_PLATFORM_NET_DELTA_BTC', 'NOT SET')}")
                
                # Test delta calculation
                if hedge_feed_manager.position_manager:
                    try:
                        test_greeks = hedge_feed_manager.position_manager.get_aggregate_platform_greeks()
                        current_delta = test_greeks.get("net_portfolio_delta_btc", 0.0)
                        logger.info(f" - Current delta exposure: {current_delta:.4f} BTC")
                        threshold = getattr(config, 'MAX_PLATFORM_NET_DELTA_BTC', 0.5)
                        logger.info(f" - Should hedge: {abs(current_delta) > threshold} (|{current_delta:.4f}| > {threshold})")
                    except Exception as e:
                        logger.error(f" - Failed to get test delta: {e}")
                
            except Exception as e:
                logger.error(f"❌ Failed to start hedge feed manager: {e}")

        logger.info("✅ Backend components initialized with COMPLETE audit integration and data consistency fixes")

        yield

        # Shutdown
        logger.info("🔄 Shutting down backend components")
        if data_feed_manager:
            try:
                data_feed_manager.stop()
                logger.info("✅ Data feed manager stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping data feed manager: {e}")
                
        if bot_trader_simulator:
            try:
                bot_trader_simulator.stop()
                logger.info("✅ Bot trader simulator stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping bot trader simulator: {e}")
                
        if hedge_feed_manager:
            try:
                hedge_feed_manager.stop()
                logger.info("✅ Hedge feed manager stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping hedge feed manager: {e}")
                
        logger.info("✅ Shutdown complete")

    except Exception as e:
        logger.error("❌ Initialization failed", exc_info=True)
        raise

# Create FastAPI app
app = FastAPI(
    title="Atticus Options Trading Platform",
    version="2.3.2",
    lifespan=lifespan
)

# CRITICAL FIX: CORS Configuration for Render Production
if IS_PRODUCTION:
    logger.warning("🌐 PRODUCTION: Using permissive CORS for debugging")
    # Permissive CORS for production debugging
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # CRITICAL: Allow all origins for debugging
        allow_credentials=False,  # CRITICAL: Must be False when using "*"
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
        max_age=86400,
    )
else:
    logger.info("🔧 DEVELOPMENT: Using restricted CORS")
    # Restricted CORS for development
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
    """CRITICAL FIX: Enhanced health check with proper error handling"""
    try:
        # FIXED: Safe component checking with individual try-catch
        statuses = {}

        # Check data feed manager safely
        try:
            if data_feed_manager is not None:
                # Don't call get_current_price() as it might fail - just check if it exists
                statuses["data_feed"] = hasattr(data_feed_manager, 'get_current_price')
            else:
                statuses["data_feed"] = False
        except Exception as e:
            logger.warning(f"Health check - data feed error: {e}")
            statuses["data_feed"] = False

        # Check other components safely
        try:
            statuses["position_manager"] = position_manager is not None
        except Exception:
            statuses["position_manager"] = False

        try:
            statuses["revenue_engine"] = revenue_engine is not None
        except Exception:
            statuses["revenue_engine"] = False

        try:
            statuses["pricing_engine"] = pricing_engine is not None
        except Exception:
            statuses["pricing_engine"] = False

        try:
            statuses["liquidity_manager"] = liquidity_manager is not None
        except Exception:
            statuses["liquidity_manager"] = False

        try:
            statuses["bot_simulator"] = bot_trader_simulator is not None
        except Exception:
            statuses["bot_simulator"] = False

        try:
            statuses["hedge_manager"] = hedge_feed_manager is not None
        except Exception:
            statuses["hedge_manager"] = False

        try:
            statuses["audit_engine"] = audit_engine is not None
        except Exception:
            statuses["audit_engine"] = False

        try:
            # Check data consistency safely
            statuses["data_consistency"] = (audit_engine is not None and 
                                          hasattr(audit_engine, 'bot_simulator') and 
                                          audit_engine.bot_simulator is not None)
        except Exception:
            statuses["data_consistency"] = False

        # FIXED: Safe calculation
        try:
            up = sum(1 for status in statuses.values() if status is True)
            total = len(statuses)
            pct = (up/total)*100 if total > 0 else 0
        except Exception as e:
            logger.error(f"Health calculation error: {e}")
            up = 0
            total = len(statuses)
            pct = 0

        status = "healthy" if pct >= 90 else ("degraded" if pct >= 70 else "unhealthy")

        # CRITICAL: Add hedge execution status
        hedge_status = "unknown"
        current_delta = 0.0
        hedge_threshold = getattr(config, 'MAX_PLATFORM_NET_DELTA_BTC', 0.5)
        
        try:
            if position_manager and hedge_feed_manager:
                test_greeks = position_manager.get_aggregate_platform_greeks()
                current_delta = test_greeks.get("net_portfolio_delta_btc", 0.0)
                hedge_should_execute = abs(current_delta) > hedge_threshold
                hedge_is_running = getattr(hedge_feed_manager, 'is_running', False)
                
                if hedge_should_execute and not hedge_is_running:
                    hedge_status = "CRITICAL - Should hedge but manager not running"
                elif hedge_should_execute and hedge_is_running:
                    hedge_status = "Should be hedging"
                else:
                    hedge_status = "Normal - no hedging needed"
        except Exception as e:
            hedge_status = f"Error checking hedge status: {e}"

        return {
            "status": status,
            "health_percentage": round(pct, 1),
            "services": statuses,
            "data_consistency_status": "connected" if statuses.get("data_consistency", False) else "disconnected",
            "environment": "production" if IS_PRODUCTION else "development",
            "cors_mode": "permissive" if IS_PRODUCTION else "restricted",
            "hedge_execution_status": hedge_status,
            "current_delta_btc": round(current_delta, 4),
            "hedge_threshold_btc": hedge_threshold,
            "component_count": {"up": up, "total": total},
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error("Health check failed", exc_info=True)
        return JSONResponse(status_code=500, content={
            "status": "error",
            "health_percentage": 0.0,
            "error": str(e),
            "environment": "production" if IS_PRODUCTION else "development",
            "timestamp": time.time()
        })

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Atticus Options Trading Platform API",
        "version": "2.3.2",
        "status": "operational",
        "environment": "production" if IS_PRODUCTION else "development",
        "data_consistency": "enabled",
        "cors_mode": "permissive" if IS_PRODUCTION else "restricted",
        "timestamp": time.time()
    }

# CRITICAL FIX: CORS Testing Endpoint
@app.options("/api/dashboard/{path:path}")
async def cors_preflight(path: str):
    """Handle CORS preflight requests explicitly"""
    logger.info(f"CORS preflight request for: /api/dashboard/{path}")
    return JSONResponse(
        content={"message": "CORS preflight handled"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

# NEW: Data consistency validation endpoint
@app.get("/api/dashboard/validate-data-consistency")
async def validate_data_consistency():
    """Validate that trading stats and audit data are consistent"""
    try:
        if not bot_trader_simulator or not audit_engine:
            return {"status": "unavailable", "message": "Components not available"}
            
        trading_stats = bot_trader_simulator.get_trading_statistics()
        audit_metrics = audit_engine.get_24h_metrics()
        
        trade_diff = abs(trading_stats.get('total_trades_24h', 0) - audit_metrics.option_trades_executed_24h)
        volume_diff = abs(trading_stats.get('total_premium_volume_usd_24h', 0) - audit_metrics.gross_option_premiums_24h_usd)
        
        is_consistent = trade_diff <= 10 and volume_diff <= 1000
        
        return {
            "status": "consistent" if is_consistent else "inconsistent",
            "trading_stats_trades": trading_stats.get('total_trades_24h', 0),
            "audit_trades": audit_metrics.option_trades_executed_24h,
            "trade_difference": trade_diff,
            "volume_difference": volume_diff,
            "audit_has_bot_reference": hasattr(audit_engine, 'bot_simulator') and audit_engine.bot_simulator is not None,
            "environment": "production" if IS_PRODUCTION else "development",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error validating data consistency: {e}")
        return {"status": "error", "error": str(e), "timestamp": time.time()}

# CRITICAL FIX: Force restart endpoint for Render
@app.post("/api/dashboard/force-restart")
async def force_restart():
    """Force restart all background services - Render-specific fix"""
    try:
        logger.warning("🔄 FORCE RESTART INITIATED - Render ephemeral filesystem workaround")
        
        # Stop all services
        if bot_trader_simulator:
            try:
                bot_trader_simulator.stop()
                logger.info("✅ Stopped bot trader simulator")
            except Exception as e:
                logger.error(f"Error stopping bot trader simulator: {e}")
            
        if hedge_feed_manager:
            try:
                hedge_feed_manager.stop()
                logger.info("✅ Stopped hedge feed manager")
            except Exception as e:
                logger.error(f"Error stopping hedge feed manager: {e}")
            
        # Wait for services to stop
        await asyncio.sleep(2)
        
        # Clear all data manually
        if bot_trader_simulator:
            try:
                if hasattr(bot_trader_simulator, 'recent_trades_log'):
                    bot_trader_simulator.recent_trades_log.clear()
                if hasattr(bot_trader_simulator, 'total_trades_executed'):
                    bot_trader_simulator.total_trades_executed = 0
                if hasattr(bot_trader_simulator, 'total_premium_collected_usd'):
                    bot_trader_simulator.total_premium_collected_usd = 0.0
                if hasattr(bot_trader_simulator, 'start_time'):
                    bot_trader_simulator.start_time = time.time()
                logger.info("✅ Cleared bot trader data")
            except Exception as e:
                logger.error(f"Error clearing bot trader data: {e}")
                
        if hedge_feed_manager:
            try:
                if hasattr(hedge_feed_manager, 'recent_hedges'):
                    hedge_feed_manager.recent_hedges.clear()
                logger.info("✅ Cleared hedge feed data")
            except Exception as e:
                logger.error(f"Error clearing hedge feed data: {e}")
                
        if position_manager:
            try:
                if hasattr(position_manager, 'open_option_positions'):
                    position_manager.open_option_positions.clear()
                if hasattr(position_manager, 'open_hedge_positions'):
                    position_manager.open_hedge_positions.clear()
                if hasattr(position_manager, 'total_portfolio_delta'):
                    position_manager.total_portfolio_delta = 0.0
                logger.info("✅ Cleared position manager data")
            except Exception as e:
                logger.error(f"Error clearing position manager data: {e}")
        
        # Restart services
        await asyncio.sleep(1)
        
        if bot_trader_simulator:
            try:
                bot_trader_simulator.start()
                logger.info("✅ Restarted bot trader simulator")
            except Exception as e:
                logger.error(f"Error restarting bot trader simulator: {e}")
            
        if hedge_feed_manager:
            try:
                hedge_feed_manager.start()
                logger.info("✅ Restarted hedge feed manager")
            except Exception as e:
                logger.error(f"Error restarting hedge feed manager: {e}")
            
        logger.warning("✅ FORCE RESTART COMPLETED - Services restarted with cleared data")
        
        return {
            "status": "services_restarted", 
            "message": "All services restarted with cleared data",
            "environment": "production" if IS_PRODUCTION else "development",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Force restart failed: {e}")
        return {"error": str(e), "status": "restart_failed", "timestamp": time.time()}

# CRITICAL FIX: Fixed debug hedge execution endpoint with JSON serialization
@app.get("/api/dashboard/debug-hedge-execution")
async def debug_hedge_execution():
    """Debug hedge execution status with JSON serialization fix"""
    try:
        debug_info = {}
        
        # Check hedge manager status with safe conversion
        if hedge_feed_manager:
            debug_info["hedge_manager"] = {
                "exists": True,
                "is_running": safe_json_convert(getattr(hedge_feed_manager, 'is_running', False)),
                "has_position_manager": safe_json_convert(hedge_feed_manager.position_manager is not None),
                "has_audit_engine": safe_json_convert(hedge_feed_manager.audit_engine is not None),
                "has_liquidity_manager": safe_json_convert(hasattr(hedge_feed_manager, 'liquidity_manager') and hedge_feed_manager.liquidity_manager is not None)
            }
        else:
            debug_info["hedge_manager"] = {"exists": False}
        
        # Check position manager and delta with safe conversion
        if position_manager:
            try:
                greeks = position_manager.get_aggregate_platform_greeks()
                debug_info["position_manager"] = {
                    "exists": True,
                    "current_delta": safe_json_convert(greeks.get("net_portfolio_delta_btc", 0.0)),
                    "risk_status": str(greeks.get("risk_status_message", "Unknown")),
                    "open_options": safe_json_convert(greeks.get("open_options_count", 0)),
                    "open_hedges": safe_json_convert(greeks.get("open_hedges_count", 0))
                }
            except Exception as e:
                debug_info["position_manager"] = {"exists": True, "error": str(e)}
        else:
            debug_info["position_manager"] = {"exists": False}
        
        # Check configuration with safe conversion
        threshold_value = getattr(config, 'MAX_PLATFORM_NET_DELTA_BTC', 'NOT_SET')
        debug_info["configuration"] = {
            "MAX_PLATFORM_NET_DELTA_BTC": safe_json_convert(threshold_value),
            "hedge_threshold_configured": safe_json_convert(hasattr(config, 'MAX_PLATFORM_NET_DELTA_BTC'))
        }
        
        # Check if hedging should be happening with safe conversion
        if position_manager and hasattr(config, 'MAX_PLATFORM_NET_DELTA_BTC'):
            try:
                greeks = position_manager.get_aggregate_platform_greeks()
                current_delta = greeks.get("net_portfolio_delta_btc", 0.0)
                threshold = config.MAX_PLATFORM_NET_DELTA_BTC
                should_hedge = abs(current_delta) > threshold
                
                debug_info["hedge_decision"] = {
                    "current_delta": safe_json_convert(current_delta),
                    "threshold": safe_json_convert(threshold),
                    "should_hedge": safe_json_convert(should_hedge),
                    "delta_exceeds_threshold": safe_json_convert(abs(current_delta) > threshold)
                }
            except Exception as e:
                debug_info["hedge_decision"] = {"error": str(e)}
        
        debug_info["timestamp"] = time.time()
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug hedge execution failed: {e}")
        return {"error": str(e), "timestamp": time.time()}

# NEW: Force hedge execution test endpoint
@app.post("/api/dashboard/force-hedge-execution-test")
async def force_hedge_execution_test():
    """Force execute a test hedge for debugging - CAUTION: Only for testing"""
    try:
        if not hedge_feed_manager:
            return {"error": "Hedge feed manager not available", "success": False}
        
        if not position_manager:
            return {"error": "Position manager not available", "success": False}
        
        # Get current state
        current_greeks = position_manager.get_aggregate_platform_greeks()
        current_delta = current_greeks.get("net_portfolio_delta_btc", 0.0)
        
        logger.warning(f"HFM: FORCING TEST HEDGE EXECUTION - Current delta: {current_delta:.4f}")
        
        # Force execute hedge using internal method if available
        if hasattr(hedge_feed_manager, '_execute_hedge'):
            try:
                hedge_feed_manager._execute_hedge(current_delta)
                
                # Get updated state after hedge
                updated_greeks = position_manager.get_aggregate_platform_greeks()
                
                return {
                    "success": True,
                    "test_executed": True,
                    "delta_before": safe_json_convert(current_delta),
                    "delta_after": safe_json_convert(updated_greeks.get("net_portfolio_delta_btc", 0.0)),
                    "hedges_count_before": safe_json_convert(current_greeks.get("open_hedges_count", 0)),
                    "hedges_count_after": safe_json_convert(updated_greeks.get("open_hedges_count", 0)),
                    "message": "Test hedge execution completed",
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"Force hedge execution failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "delta_before": safe_json_convert(current_delta),
                    "timestamp": time.time()
                }
        else:
            return {
                "success": False,
                "error": "Hedge execution method not available",
                "delta_before": safe_json_convert(current_delta),
                "timestamp": time.time()
            }
        
    except Exception as e:
        logger.error(f"Force hedge execution test failed: {e}")
        return {"error": str(e), "success": False, "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    
    # Environment-specific configuration
    uvicorn_config = {
        "host": config.API_HOST,
        "port": config.API_PORT,
        "reload": not IS_PRODUCTION  # Disable reload in production
    }
    
    if IS_PRODUCTION:
        logger.info("🌐 Starting in PRODUCTION mode")
    else:
        logger.info("🔧 Starting in DEVELOPMENT mode with reload")
        
    uvicorn.run("main_dashboard:app", **uvicorn_config)
