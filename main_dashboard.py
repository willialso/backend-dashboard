# main_dashboard.py

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import all backend components
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

# Import API router
from investor_dashboard.dashboard_api import router as dashboard_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global component instances (will be initialized in lifespan)
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
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("🔧 Initializing global backend components with enhanced engines...")
    
    global data_feed_manager, volatility_engine, pricing_engine, revenue_engine
    global position_manager, liquidity_manager, bot_trader_simulator, audit_engine, hedge_feed_manager
    
    try:
        # Initialize DataFeedManager
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
        
        # Initialize AuditEngine - SIMPLE
        audit_engine = AuditEngine()
        
        # Initialize AdvancedVolatilityEngine
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
        
        # Initialize AdvancedPricingEngine
        pricing_engine = AdvancedPricingEngine(
            volatility_engine_instance=volatility_engine,
            alpha_signal_generator_instance=None
        )
        
        # Initialize RevenueEngine with CORRECT parameter names - FIXED
        revenue_engine = RevenueEngine(
            volatility_engine_instance=volatility_engine,  # CORRECTED: was volatility_engine
            pricing_engine_instance=pricing_engine          # CORRECTED: was pricing_engine
        )
        
        # Initialize PositionManager - Try different parameter names
        try:
            position_manager = PositionManager(
                pricing_engine_instance=pricing_engine
            )
        except TypeError:
            try:
                position_manager = PositionManager(
                    pricing_engine=pricing_engine
                )
            except TypeError:
                position_manager = PositionManager()
                if hasattr(position_manager, 'pricing_engine'):
                    position_manager.pricing_engine = pricing_engine
        
        # CRITICAL: Ensure PositionManager has all dependencies
        if position_manager:
            if not hasattr(position_manager, 'pricing_engine') or position_manager.pricing_engine is None:
                position_manager.pricing_engine = pricing_engine
                logger.info("🔧 Manually set pricing_engine on PositionManager")
            if hasattr(position_manager, 'current_btc_price'):
                position_manager.current_btc_price = 108000.0
                logger.info("🔧 Set default BTC price on PositionManager")
        
        # NEW FIX: Ensure RevenueEngine has PositionManager reference
        if revenue_engine and position_manager:
            if hasattr(revenue_engine, 'position_manager'):
                revenue_engine.position_manager = position_manager
                logger.info("🔧 Connected PositionManager to RevenueEngine")
        
        # Initialize LiquidityManager with full parameters
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
        
        # Initialize BotTraderSimulator with CORRECT dependencies - CRITICAL FIX
        try:
            bot_trader_simulator = BotTraderSimulator(
                revenue_engine=revenue_engine,
                position_manager=position_manager,
                data_feed_manager=data_feed_manager,
                audit_engine_instance=audit_engine
            )
            logger.info(f"✅ BotTraderSimulator initialized with dependencies: revenue_engine={revenue_engine is not None}, position_manager={position_manager is not None}")
        except TypeError as e:
            logger.warning(f"BotTraderSimulator constructor failed: {e}")
            # Fallback: create with no parameters and set dependencies manually
            bot_trader_simulator = BotTraderSimulator()
            bot_trader_simulator.revenue_engine = revenue_engine
            bot_trader_simulator.position_manager = position_manager
            bot_trader_simulator.audit_engine = audit_engine
            logger.info("✅ BotTraderSimulator initialized with manual dependency injection")
        
        # CRITICAL: Verify dependencies are properly set
        if bot_trader_simulator:
            if not hasattr(bot_trader_simulator, 'revenue_engine') or bot_trader_simulator.revenue_engine is None:
                bot_trader_simulator.revenue_engine = revenue_engine
                logger.info("🔧 Manually set revenue_engine on BotTraderSimulator")
            
            if not hasattr(bot_trader_simulator, 'position_manager') or bot_trader_simulator.position_manager is None:
                bot_trader_simulator.position_manager = position_manager
                logger.info("🔧 Manually set position_manager on BotTraderSimulator")
            
            # Ensure BTC price is set for trading
            if hasattr(bot_trader_simulator, 'current_btc_price') and bot_trader_simulator.current_btc_price <= 0:
                bot_trader_simulator.current_btc_price = 108000.0
                logger.info("🔧 Set default BTC price on BotTraderSimulator")
        
        # Initialize HedgeFeedManager
        try:
            hedge_feed_manager = HedgeFeedManager(
                position_manager_instance=position_manager,
                audit_engine_instance=audit_engine
            )
        except TypeError:
            hedge_feed_manager = HedgeFeedManager()
            if hasattr(hedge_feed_manager, 'position_manager'):
                hedge_feed_manager.position_manager = position_manager
        
        # Inject global instances into dashboard_api module
        import investor_dashboard.dashboard_api as dashboard_api_module
        dashboard_api_module.revenue_engine = revenue_engine
        dashboard_api_module.liquidity_manager = liquidity_manager
        dashboard_api_module.bot_trader_simulator = bot_trader_simulator
        dashboard_api_module.audit_engine = audit_engine
        dashboard_api_module.hedge_feed_manager = hedge_feed_manager
        dashboard_api_module.position_manager = position_manager
        
        # Start background services
        data_feed_manager.start()
        
        # Start bot trader simulator
        if bot_trader_simulator and hasattr(bot_trader_simulator, 'start'):
            bot_trader_simulator.start()
            logger.info("✅ BotTraderSimulator started")
        
        # Start hedge feed manager
        if hedge_feed_manager and hasattr(hedge_feed_manager, 'start'):
            hedge_feed_manager.start()
            logger.info("✅ HedgeFeedManager started")
        
        logger.info("✅ All backend components initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}", exc_info=True)
        raise
    
    # Shutdown
    logger.info("🔄 Shutting down backend components...")
    try:
        if data_feed_manager and hasattr(data_feed_manager, 'stop'):
            data_feed_manager.stop()
        if bot_trader_simulator and hasattr(bot_trader_simulator, 'stop'):
            bot_trader_simulator.stop()
        if hedge_feed_manager and hasattr(hedge_feed_manager, 'stop'):
            hedge_feed_manager.stop()
        logger.info("✅ All components shut down successfully!")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}", exc_info=True)

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Atticus Options Trading Platform",
    description="Professional-grade BTC options trading platform with real-time market data",
    version="2.3.1",
    lifespan=lifespan
)

# CORS configuration
origins = [
    "http://localhost:3000",
    "https://localhost:3000",
    "http://127.0.0.1:3000",
    "https://127.0.0.1:3000",
    "http://localhost:8000",
    "https://localhost:8000",
    "http://127.0.0.1:8000",
    "https://127.0.0.1:8000",
    "https://preview--brilliant-croissant-de6cba.lovable.app",
    "https://brilliant-croissant-de6cba.lovable.app",
    "*"  # Allow all origins for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=86400,
)

# Include the dashboard router with proper prefix
app.include_router(dashboard_router, prefix="/api/dashboard")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - Atticus Platform API."""
    return {
        "message": "Atticus Options Trading Platform API",
        "version": "2.3.1",
        "status": "operational",
        "documentation": "/docs",
        "dashboard_endpoints": "/api/dashboard/*",
        "health_check": "/health",
        "timestamp": time.time()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Complete health check implementation."""
    try:
        pm_ok = position_manager is not None and hasattr(position_manager, 'get_aggregate_platform_greeks')
        
        data_feed_ok = False
        if data_feed_manager:
            current_price = data_feed_manager.get_current_price()
            data_feed_ok = current_price > 0
        
        services_status = {
            "data_feed": data_feed_ok,
            "position_manager": pm_ok,
            "revenue_engine": revenue_engine is not None,
            "volatility_engine": volatility_engine is not None,
            "pricing_engine": pricing_engine is not None,
            "liquidity_manager": liquidity_manager is not None,
            "bot_trader_simulator": bot_trader_simulator is not None,
            "hedge_feed_manager": hedge_feed_manager is not None,
            "audit_engine": audit_engine is not None
        }
        
        operational_count = sum(1 for status in services_status.values() if status)
        total_services = len(services_status)
        health_percentage = (operational_count / total_services) * 100 if total_services > 0 else 0
        
        if health_percentage >= 90:
            overall_status = "healthy"
        elif health_percentage >= 70:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        market_data = {}
        if data_feed_manager:
            try:
                market_data = {
                    "current_btc_price": data_feed_manager.get_current_price(),
                    "exchange_status": data_feed_manager.get_exchange_status()
                }
            except Exception as e:
                market_data = {"error": f"Failed to get market data: {str(e)}"}
        
        return {
            "status": overall_status,
            "health_percentage": round(health_percentage, 1),
            "operational_services": operational_count,
            "total_services": total_services,
            "services_status": services_status,
            "market_data": market_data,
            "platform_info": {
                "version": "2.3.1",
                "uptime_seconds": time.time(),
                "environment": "development"
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "timestamp": time.time()
            }
        )

# Additional utility endpoints
@app.get("/api/status")
async def api_status():
    """API status endpoint."""
    return {
        "api_version": "2.3.1",
        "status": "operational",
        "endpoints": {
            "dashboard": "/api/dashboard/*",
            "health": "/health",
            "docs": "/docs"
        },
        "timestamp": time.time()
    }

@app.get("/api/config")
async def get_config():
    """Get public configuration information."""
    return {
        "platform_name": "Atticus Options Trading Platform",
        "version": "2.3.1",
        "supported_exchanges": config.EXCHANGES_ENABLED,
        "primary_exchange": config.PRIMARY_EXCHANGE,
        "api_endpoints": {
            "dashboard_prefix": "/api/dashboard",
            "health_check": "/health",
            "documentation": "/docs"
        },
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Atticus Platform server...")
    uvicorn.run(
        "main_dashboard:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    )
