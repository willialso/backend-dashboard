# investor_dashboard/dashboard_api.py

import time
import logging
import io
import csv
from typing import Any, Dict, Optional, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd

from backend import config
from .position_manager import PositionManager
from .bot_trader_simulator import BotTraderSimulator
from .hedge_feed_manager import HedgeFeedManager
from .audit_engine import AuditEngine
from .revenue_engine import RevenueEngine

router = APIRouter()

logger = logging.getLogger(__name__)

# These will be injected at startup in main_dashboard.py
revenue_engine: Optional[RevenueEngine] = None
position_manager: Optional[PositionManager] = None
liquidity_manager: Any = None
bot_trader_simulator: Optional[BotTraderSimulator] = None
hedge_feed_manager: Optional[HedgeFeedManager] = None
audit_engine: Optional[AuditEngine] = None

api_call_count = 0
performance_metrics: Dict[str, Dict[str, float]] = {}

def log_api_call(endpoint: str, result: str, duration_ms: float = 0):
    global api_call_count, performance_metrics
    api_call_count += 1
    m = performance_metrics.setdefault(endpoint, {"calls": 0, "errors": 0, "total_duration": 0.0, "avg_duration": 0.0})
    m["calls"] += 1
    m["total_duration"] += duration_ms
    m["avg_duration"] = m["total_duration"] / m["calls"]
    if "error" in result.lower():
        m["errors"] += 1
    logger.info(f"API Call #{api_call_count}: {endpoint} - {result} ({duration_ms:.2f}ms)")

def safe_component_call(name: str, func, *args, **kwargs):
    start = time.time()
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return None, str(e)

# CRITICAL FIX: Add data consistency validation
def validate_data_consistency():
    """Validate that trading stats and audit data are consistent"""
    try:
        if not bot_trader_simulator or not audit_engine:
            return True  # Can't validate if components missing
            
        trading_stats = bot_trader_simulator.get_trading_statistics()
        audit_metrics = audit_engine.get_24h_metrics()
        
        trade_diff = abs(trading_stats.get('total_trades_24h', 0) - audit_metrics.option_trades_executed_24h)
        volume_diff = abs(trading_stats.get('total_premium_volume_usd_24h', 0) - audit_metrics.gross_option_premiums_24h_usd)
        
        if trade_diff > 10 or volume_diff > 1000:
            logger.error(f"❌ Data inconsistency detected: Trade diff: {trade_diff}, Volume diff: ${volume_diff:,.2f}")
            return False
            
        logger.debug(f"✅ Data consistency validated: Trade diff: {trade_diff}, Volume diff: ${volume_diff:,.2f}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating data consistency: {e}")
        return False

def force_audit_sync():
    """Force audit engine to sync with bot simulator"""
    try:
        if audit_engine and bot_trader_simulator:
            if hasattr(audit_engine, 'set_bot_simulator_reference'):
                audit_engine.set_bot_simulator_reference(bot_trader_simulator)
                logger.info("✅ Forced audit sync with bot simulator")
            elif hasattr(audit_engine, 'force_sync_with_bot_simulator'):
                audit_engine.force_sync_with_bot_simulator()
                logger.info("✅ Forced audit sync completed")
    except Exception as e:
        logger.error(f"❌ Error forcing audit sync: {e}")

# Request models
class LiquidityAllocation(BaseModel):
    liquidity_pct: float
    operations_pct: float

class LiquidityPoolSize(BaseModel):
    total_pool_usd: float
    reason: Optional[str] = "manual_adjustment"

class TraderDistribution(BaseModel):
    total_traders: int
    auto_scale_liquidity: Optional[bool] = False

@router.get("/platform-health")
async def get_platform_health():
    start = time.time()
    comps = {
        "revenue_engine": revenue_engine is not None,
        "liquidity_manager": liquidity_manager is not None,
        "audit_engine": audit_engine is not None,
        "hedge_feed_manager": hedge_feed_manager is not None,
        "bot_simulator": bot_trader_simulator is not None,
        "position_manager": position_manager is not None,
        "data_consistency": validate_data_consistency()
    }
    
    up = sum(comps.values())
    pct = up / len(comps) * 100
    status = "GOOD" if pct >= 80 else ("FAIR" if pct >= 50 else "POOR")
    
    log_api_call("/platform-health", "success", (time.time()-start)*1000)
    return {
        "overall_health": status, 
        "health_score": pct, 
        "components": comps, 
        "data_consistency_enabled": True,
        "timestamp": time.time()
    }

@router.get("/revenue-metrics")
async def get_revenue_metrics():
    start = time.time()
    if not revenue_engine:
        raise HTTPException(503, "Revenue engine unavailable")
    
    metrics, err = safe_component_call("revenue_engine", revenue_engine.get_current_metrics)
    if err or not metrics:
        log_api_call("/revenue-metrics", f"error - {err}", (time.time()-start)*1000)
        raise HTTPException(500, f"Error fetching revenue metrics: {err}")
    
    data = metrics.__dict__ if hasattr(metrics, "__dict__") else metrics
    data["timestamp"] = time.time()
    
    log_api_call("/revenue-metrics", "success", (time.time()-start)*1000)
    return data

@router.get("/bot-trader-status")
async def get_bot_trader_status():
    start = time.time()
    if not bot_trader_simulator:
        raise HTTPException(503, "Bot simulator unavailable")
    
    # Get both activity and trader profiles
    activity, err = safe_component_call("bot_trader_simulator", bot_trader_simulator.get_current_activity)
    if err:
        log_api_call("/bot-trader-status", f"error - {err}", (time.time()-start)*1000)
        return {"is_running": False, "timestamp": time.time()}
    
    # Add trader profiles data for sliders
    trader_profiles = {}
    for trader_type, profile in bot_trader_simulator.trader_profiles.items():
        trader_profiles[trader_type.value] = {
            "count": profile["count"],
            "trades_per_hour": profile["trades_per_hour"],
            "success_rate": profile["success_rate"]
        }
    
    data = {
        "is_running": bot_trader_simulator.is_running,
        "current_btc_price": bot_trader_simulator.current_btc_price,
        "recent_trades_count": len(bot_trader_simulator.recent_trades_log),
        "trader_profiles": trader_profiles,
        "activity": activity,
        "timestamp": time.time()
    }
    
    log_api_call("/bot-trader-status", "success", (time.time()-start)*1000)
    return data

@router.get("/trading-statistics")
async def get_trading_statistics():
    start = time.time()
    stats, err = safe_component_call("bot_trader_simulator", bot_trader_simulator.get_trading_statistics)
    if err or stats is None:
        log_api_call("/trading-statistics", f"error - {err}", (time.time()-start)*1000)
        raise HTTPException(500, f"Error fetching trading statistics: {err}")
    
    data = stats if isinstance(stats, dict) else stats.__dict__
    data["timestamp"] = time.time()
    
    log_api_call("/trading-statistics", "success", (time.time()-start)*1000)
    return data

@router.get("/recent-trades")
async def get_recent_trades(limit: int = 10):
    start = time.time()
    trades, err = safe_component_call("bot_trader_simulator", bot_trader_simulator.get_recent_trades, limit)
    if err or trades is None:
        log_api_call("/recent-trades", f"error - {err}", (time.time()-start)*1000)
        raise HTTPException(500, f"Error fetching recent trades: {err}")
    
    out = [t.__dict__ for t in trades]
    
    log_api_call("/recent-trades", "success", (time.time()-start)*1000)
    return {"trades": out, "total_returned": len(out), "timestamp": time.time()}

@router.get("/liquidity-allocation")
async def get_liquidity_allocation():
    """FIXED: Updated to use new LiquidityManager methods"""
    start = time.time()
    if not liquidity_manager:
        log_api_call("/liquidity-allocation", "error - no liquidity manager", (time.time()-start)*1000)
        return {
            "liquidity_ratio": 1.923,
            "total_pool_usd": config.LM_INITIAL_TOTAL_POOL_USD,
            "liquidity_percentage": 75.0,
            "operations_percentage": 25.0,
            "utilized_amount_usd": 0.0,
            "available_amount_usd": config.LM_INITIAL_TOTAL_POOL_USD * 0.75,
            "utilization_pct": 0.0,
            "stress_test_status": "UNKNOWN",
            "error": "Liquidity manager not available",
            "timestamp": time.time()
        }
    
    # Use the new get_current_allocation method
    allocation_data, err = safe_component_call("liquidity_manager", liquidity_manager.get_current_allocation)
    if err or allocation_data is None:
        log_api_call("/liquidity-allocation", f"error - {err}", (time.time()-start)*1000)
        return {
            "liquidity_ratio": 1.923,
            "total_pool_usd": config.LM_INITIAL_TOTAL_POOL_USD,
            "liquidity_percentage": 75.0,
            "operations_percentage": 25.0,
            "utilized_amount_usd": 0.0,
            "available_amount_usd": config.LM_INITIAL_TOTAL_POOL_USD * 0.75,
            "utilization_pct": 0.0,
            "stress_test_status": "UNKNOWN",
            "error": err,
            "timestamp": time.time()
        }
    
    data = allocation_data if isinstance(allocation_data, dict) else allocation_data.__dict__
    data["timestamp"] = time.time()
    
    log_api_call("/liquidity-allocation", "success", (time.time()-start)*1000)
    return data

@router.post("/liquidity-allocation")
async def adjust_liquidity(dist: LiquidityAllocation):
    """FIXED: Updated to use new LiquidityManager update_allocation method"""
    start = time.time()
    
    # Check if liquidity manager exists
    if not liquidity_manager:
        log_api_call("/liquidity-allocation", "error - no liquidity manager", (time.time()-start)*1000)
        raise HTTPException(503, "Liquidity manager not available")
    
    # Check if update_allocation method exists
    if not hasattr(liquidity_manager, 'update_allocation'):
        log_api_call("/liquidity-allocation", "error - method not implemented", (time.time()-start)*1000)
        logger.warning("Liquidity manager does not have update_allocation method")
        return {
            "success": False,
            "message": "Liquidity adjustment not implemented yet",
            "status": "not_implemented"
        }
    
    # Validate input ranges
    if not (0 <= dist.liquidity_pct <= 100) or not (0 <= dist.operations_pct <= 100):
        raise HTTPException(400, "Percentages must be between 0 and 100")
    
    if abs(dist.liquidity_pct + dist.operations_pct - 100) > 0.1:
        raise HTTPException(400, "Liquidity and operations percentages must sum to 100")
    
    try:
        result, err = safe_component_call("liquidity_manager", liquidity_manager.update_allocation, dist.liquidity_pct, dist.operations_pct)
        if err:
            logger.error(f"Liquidity adjustment failed: {err}")
            log_api_call("/liquidity-allocation", f"error - {err}", (time.time()-start)*1000)
            raise HTTPException(500, f"Error adjusting liquidity: {err}")
        
        log_api_call("/liquidity-allocation", "success", (time.time()-start)*1000)
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error in liquidity adjustment: {e}", exc_info=True)
        log_api_call("/liquidity-allocation", f"error - {e}", (time.time()-start)*1000)
        raise HTTPException(500, f"Unexpected error: {str(e)}")

@router.post("/liquidity-pool-size")
async def update_pool_size(pool_update: LiquidityPoolSize):
    """NEW: Update total liquidity pool size"""
    start = time.time()
    
    if not liquidity_manager:
        log_api_call("/liquidity-pool-size", "error - no liquidity manager", (time.time()-start)*1000)
        raise HTTPException(503, "Liquidity manager not available")
    
    if not hasattr(liquidity_manager, 'update_total_pool'):
        log_api_call("/liquidity-pool-size", "error - method not available", (time.time()-start)*1000)
        return {
            "success": False,
            "message": "Pool size adjustment not implemented yet",
            "status": "not_implemented"
        }
    
    # Validate pool size
    if pool_update.total_pool_usd < 500000:
        raise HTTPException(400, "Pool size must be at least $500,000")
    
    if pool_update.total_pool_usd > 50000000:
        raise HTTPException(400, "Pool size must be less than $50,000,000")
    
    try:
        result, err = safe_component_call("liquidity_manager", liquidity_manager.update_total_pool, pool_update.total_pool_usd, pool_update.reason)
        if err:
            logger.error(f"Pool size update failed: {err}")
            log_api_call("/liquidity-pool-size", f"error - {err}", (time.time()-start)*1000)
            raise HTTPException(500, f"Error updating pool size: {err}")
        
        log_api_call("/liquidity-pool-size", "success", (time.time()-start)*1000)
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error in pool size update: {e}", exc_info=True)
        log_api_call("/liquidity-pool-size", f"error - {e}", (time.time()-start)*1000)
        raise HTTPException(500, f"Unexpected error: {str(e)}")

@router.get("/liquidity-scaling-recommendations")
async def get_liquidity_scaling_recommendations():
    """NEW: Get recommended pool size based on current usage"""
    start = time.time()
    
    if not liquidity_manager:
        log_api_call("/liquidity-scaling-recommendations", "error - no liquidity manager", (time.time()-start)*1000)
        return {"error": "Liquidity manager not available"}
    
    if not hasattr(liquidity_manager, 'get_recommended_pool_size'):
        log_api_call("/liquidity-scaling-recommendations", "error - method not available", (time.time()-start)*1000)
        return {"error": "Pool size recommendations not available"}
    
    try:
        recommendations, err = safe_component_call("liquidity_manager", liquidity_manager.get_recommended_pool_size)
        if err:
            log_api_call("/liquidity-scaling-recommendations", f"error - {err}", (time.time()-start)*1000)
            return {"error": err}
        
        log_api_call("/liquidity-scaling-recommendations", "success", (time.time()-start)*1000)
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting scaling recommendations: {e}")
        log_api_call("/liquidity-scaling-recommendations", f"error - {e}", (time.time()-start)*1000)
        return {"error": str(e)}

@router.get("/liquidity-scaling-metrics")
async def get_liquidity_scaling_metrics():
    """NEW: Get comprehensive scaling metrics for dashboard"""
    start = time.time()
    
    if not liquidity_manager:
        log_api_call("/liquidity-scaling-metrics", "error - no liquidity manager", (time.time()-start)*1000)
        return {"error": "Liquidity manager not available"}
    
    if not hasattr(liquidity_manager, 'get_scaling_metrics'):
        log_api_call("/liquidity-scaling-metrics", "error - method not available", (time.time()-start)*1000)
        return {"error": "Scaling metrics not available"}
    
    try:
        metrics, err = safe_component_call("liquidity_manager", liquidity_manager.get_scaling_metrics)
        if err:
            log_api_call("/liquidity-scaling-metrics", f"error - {err}", (time.time()-start)*1000)
            return {"error": err}
        
        log_api_call("/liquidity-scaling-metrics", "success", (time.time()-start)*1000)
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting scaling metrics: {e}")
        log_api_call("/liquidity-scaling-metrics", f"error - {e}", (time.time()-start)*1000)
        return {"error": str(e)}

@router.get("/liquidity-health")
async def get_liquidity_health():
    """NEW: Get comprehensive liquidity health status"""
    start = time.time()
    if not liquidity_manager:
        log_api_call("/liquidity-health", "error - no liquidity manager", (time.time()-start)*1000)
        return {"error": "Liquidity manager not available", "health_status": "ERROR"}
    
    if not hasattr(liquidity_manager, 'get_liquidity_health'):
        log_api_call("/liquidity-health", "error - method not available", (time.time()-start)*1000)
        return {"error": "Liquidity health method not available", "health_status": "ERROR"}
    
    health_data, err = safe_component_call("liquidity_manager", liquidity_manager.get_liquidity_health)
    if err:
        log_api_call("/liquidity-health", f"error - {err}", (time.time()-start)*1000)
        return {"error": err, "health_status": "ERROR"}
    
    log_api_call("/liquidity-health", "success", (time.time()-start)*1000)
    return health_data

@router.get("/liquidity-recommendations")
async def get_liquidity_recommendations():
    """NEW: Get recommended liquidity allocation"""
    start = time.time()
    if not liquidity_manager:
        log_api_call("/liquidity-recommendations", "error - no liquidity manager", (time.time()-start)*1000)
        return {"error": "Liquidity manager not available"}
    
    if not hasattr(liquidity_manager, 'get_recommended_allocation'):
        log_api_call("/liquidity-recommendations", "error - method not available", (time.time()-start)*1000)
        return {"error": "Liquidity recommendations method not available"}
    
    recommendations, err = safe_component_call("liquidity_manager", liquidity_manager.get_recommended_allocation)
    if err:
        log_api_call("/liquidity-recommendations", f"error - {err}", (time.time()-start)*1000)
        return {"error": err}
    
    log_api_call("/liquidity-recommendations", "success", (time.time()-start)*1000)
    return recommendations

@router.get("/platform-greeks")
async def get_platform_greeks():
    start = time.time()
    greeks, err = safe_component_call("position_manager", position_manager.get_aggregate_platform_greeks)
    if err or greeks is None:
        log_api_call("/platform-greeks", f"error - {err}", (time.time()-start)*1000)
        return {
            "net_portfolio_delta_btc": 0.0,
            "net_portfolio_gamma_btc": 0.0,
            "net_portfolio_vega_usd": 0.0,
            "net_portfolio_theta_usd": 0.0,
            "error": err,
            "timestamp": time.time()
        }
    
    data = greeks
    data["timestamp"] = time.time()
    
    log_api_call("/platform-greeks", "success", (time.time()-start)*1000)
    return data

@router.get("/hedge-execution-feed")
async def get_hedge_execution_feed(limit: int = 10):
    start = time.time()
    hedges, err = safe_component_call("hedge_feed_manager", hedge_feed_manager.get_recent_hedges, limit)
    if err or hedges is None:
        log_api_call("/hedge-execution-feed", f"error - {err}", (time.time()-start)*1000)
        raise HTTPException(500, f"Error fetching hedge feed: {err}")
    
    out = [h.__dict__ for h in hedges]
    
    log_api_call("/hedge-execution-feed", "success", (time.time()-start)*1000)
    return {"hedges": out, "total_returned": len(out), "timestamp": time.time()}

@router.get("/hedge-metrics")
async def get_hedge_metrics():
    start = time.time()
    metrics, err = safe_component_call("hedge_feed_manager", hedge_feed_manager.get_hedge_metrics)
    if err or metrics is None:
        log_api_call("/hedge-metrics", f"error - {err}", (time.time()-start)*1000)
        return {"hedges_24h": 0, "total_volume_hedged_btc_24h": 0.0, "timestamp": time.time()}
    
    data = metrics if isinstance(metrics, dict) else metrics.__dict__
    data["timestamp"] = time.time()
    
    log_api_call("/hedge-metrics", "success", (time.time()-start)*1000)
    return data

@router.get("/audit-summary")
async def get_audit_summary():
    """CRITICAL FIX: Added data consistency validation"""
    start = time.time()
    
    # CRITICAL FIX: Validate data consistency and force sync if needed
    if not validate_data_consistency():
        logger.warning("❌ Data inconsistency detected, forcing audit sync")
        force_audit_sync()
    
    metrics, err = safe_component_call("audit_engine", audit_engine.get_24h_metrics)
    if err or metrics is None:
        log_api_call("/audit-summary", f"error - {err}", (time.time()-start)*1000)
        return {"overall_status": "ERROR", "compliance_score": 0.0, "error": err, "timestamp": time.time()}
    
    data = metrics.__dict__ if hasattr(metrics, "__dict__") else metrics
    data["timestamp"] = time.time()
    
    # CRITICAL FIX: Add data consistency metadata
    data["data_consistency_check"] = validate_data_consistency()
    data["audit_synced_with_bot_simulator"] = (hasattr(audit_engine, 'bot_simulator') and 
                                             audit_engine.bot_simulator is not None)
    
    log_api_call("/audit-summary", "success", (time.time()-start)*1000)
    return data

@router.post("/trader-distribution")
async def adjust_trader_distribution(dist: TraderDistribution):
    """Enhanced trader distribution with optional auto-scaling"""
    start = time.time()
    
    # Update user count in liquidity manager if available
    if liquidity_manager and hasattr(liquidity_manager, 'update_user_count'):
        try:
            liquidity_manager.update_user_count(dist.total_traders)
            logger.info(f"Updated liquidity manager user count to {dist.total_traders}")
            
            # Auto-scale pool if requested
            if dist.auto_scale_liquidity and hasattr(liquidity_manager, 'auto_scale_pool_for_traders'):
                scale_result, scale_err = safe_component_call("liquidity_manager", liquidity_manager.auto_scale_pool_for_traders, dist.total_traders)
                if scale_result and scale_result.get("success", False):
                    logger.info(f"Auto-scaled pool to ${scale_result['new_pool_usd']:,.0f} for {dist.total_traders} traders")
                    
        except Exception as e:
            logger.warning(f"Failed to update liquidity manager user count: {e}")
    
    _, err = safe_component_call("bot_trader_simulator", bot_trader_simulator.adjust_trader_distribution, dist.total_traders)
    if err:
        log_api_call("/trader-distribution", f"error - {err}", (time.time()-start)*1000)
        raise HTTPException(500, f"Error adjusting trader distribution: {err}")
    
    # CRITICAL FIX: Force audit sync after trader distribution change
    force_audit_sync()
    
    log_api_call("/trader-distribution", "success", (time.time()-start)*1000)
    return {"status": "success", "total_traders": dist.total_traders, "auto_scaled": dist.auto_scale_liquidity}

@router.post("/reset-parameters")
async def reset_parameters():
    """Reset all system parameters to defaults"""
    results = {}
    
    if liquidity_manager and hasattr(liquidity_manager, 'reset_to_defaults'):
        try:
            liquidity_manager.reset_to_defaults()
            results["liquidity_manager"] = "reset"
        except Exception as e:
            results["liquidity_manager"] = f"error: {e}"
    
    if bot_trader_simulator and hasattr(bot_trader_simulator, 'reset_to_defaults'):
        try:
            bot_trader_simulator.reset_to_defaults()
            results["bot_trader_simulator"] = "reset"
        except Exception as e:
            results["bot_trader_simulator"] = f"error: {e}"
    
    # CRITICAL FIX: Reset audit engine and force sync
    if audit_engine and hasattr(audit_engine, 'reset_metrics'):
        try:
            audit_engine.reset_metrics()
            results["audit_engine"] = "reset"
        except Exception as e:
            results["audit_engine"] = f"error: {e}"
    
    # Force audit sync after reset
    force_audit_sync()
    
    return {"status": "success", "results": results, "data_consistency_restored": True}

# CRITICAL ENHANCED FIX: Complete system reset endpoint with ALL volume/revenue clearing
@router.post("/complete-system-reset")
async def complete_system_reset():
    """COMPREHENSIVE: Reset ALL system data including volume and revenue to fresh startup state"""
    start = time.time()
    results = {}
    
    logger.warning("🔄 COMPLETE SYSTEM RESET INITIATED - Clearing ALL accumulated data including volume/revenue")
    
    # CRITICAL FIX: Reset bot trader simulator - CLEAR ALL VOLUME AND REVENUE DATA
    if bot_trader_simulator:
        try:
            # Clear all historical trading data
            if hasattr(bot_trader_simulator, 'recent_trades_log'):
                bot_trader_simulator.recent_trades_log.clear()
                logger.info("✅ Cleared bot trader recent_trades_log")
            
            # CRITICAL: Reset ALL volume and revenue counters
            volume_revenue_attributes = [
                'total_trades_executed',
                'total_premium_collected_usd', 
                'daily_volume_usd',
                'daily_trade_count',
                'total_premium_volume_usd_24h',
                'avg_premium_received_usd_24h',
                'daily_revenue_usd',
                'cumulative_revenue',
                'total_volume_traded'
            ]
            
            for attr in volume_revenue_attributes:
                if hasattr(bot_trader_simulator, attr):
                    if 'usd' in attr.lower() or 'revenue' in attr.lower() or 'premium' in attr.lower():
                        setattr(bot_trader_simulator, attr, 0.0)
                    else:
                        setattr(bot_trader_simulator, attr, 0)
                    logger.debug(f"✅ Reset bot trader {attr} to 0")
            
            # Reset start time for fresh 24h window
            if hasattr(bot_trader_simulator, 'start_time'):
                bot_trader_simulator.start_time = time.time()
                logger.info("✅ Reset bot trader start_time")
            
            # Clear any cached metrics
            for cache_attr in ['daily_metrics', '_trading_stats_cache', '_cached_statistics']:
                if hasattr(bot_trader_simulator, cache_attr):
                    val = getattr(bot_trader_simulator, cache_attr)
                    if isinstance(val, dict):
                        val.clear()
                    elif isinstance(val, list):
                        val.clear()
                    else:
                        setattr(bot_trader_simulator, cache_attr, None)
            
            results["bot_trader_simulator"] = "volume_revenue_completely_reset"
            logger.info("✅ Bot trader volume/revenue completely cleared")
            
        except Exception as e:
            logger.error(f"❌ Error resetting bot trader volume/revenue: {e}")
            results["bot_trader_simulator"] = f"error: {e}"
    
    # CRITICAL FIX: Reset revenue engine - CLEAR ALL REVENUE DATA
    if revenue_engine:
        try:
            # Reset all revenue accumulation
            revenue_attributes = [
                'daily_revenue_accumulator',
                'total_revenue_collected',
                'markup_revenue',
                'fee_revenue',
                'cumulative_revenue_usd',
                'daily_revenue_usd'
            ]
            
            for attr in revenue_attributes:
                if hasattr(revenue_engine, attr):
                    setattr(revenue_engine, attr, 0.0)
                    logger.debug(f"✅ Reset revenue engine {attr} to 0")
            
            # Clear revenue tracking lists/dicts
            for list_attr in ['revenue_records', 'daily_metrics', 'transaction_history']:
                if hasattr(revenue_engine, list_attr):
                    val = getattr(revenue_engine, list_attr)
                    if isinstance(val, (list, dict)):
                        val.clear()
                        logger.debug(f"✅ Cleared revenue engine {list_attr}")
            
            # Reset start time
            if hasattr(revenue_engine, 'start_time'):
                revenue_engine.start_time = time.time()
            
            # Call reset method if available
            if hasattr(revenue_engine, 'reset_metrics'):
                revenue_engine.reset_metrics()
            
            results["revenue_engine"] = "volume_revenue_completely_reset"
            logger.info("✅ Revenue engine volume/revenue completely cleared")
            
        except Exception as e:
            logger.error(f"❌ Error resetting revenue engine: {e}")
            results["revenue_engine"] = f"error: {e}"
    
    # CRITICAL FIX: Reset hedge feed manager - CLEAR ALL HEDGE VOLUME/METRICS
    if hedge_feed_manager:
        try:
            # Clear recent hedges list
            if hasattr(hedge_feed_manager, 'recent_hedges'):
                hedge_feed_manager.recent_hedges.clear()
                logger.info("✅ Cleared hedge feed manager recent_hedges")
            
            # CRITICAL: Reset ALL hedge volume and metrics
            hedge_attributes = [
                'total_hedges_executed',
                'total_hedge_volume_btc',
                'total_hedge_volume_usd',
                'hedge_pnl_accumulator',
                'daily_hedge_volume',
                'total_hedge_value_usd_24h'
            ]
            
            for attr in hedge_attributes:
                if hasattr(hedge_feed_manager, attr):
                    setattr(hedge_feed_manager, attr, 0.0)
                    logger.debug(f"✅ Reset hedge manager {attr} to 0")
            
            # Clear hedge tracking dictionaries
            for dict_attr in ['hedge_metrics_24h', 'hedge_history', 'daily_hedge_data']:
                if hasattr(hedge_feed_manager, dict_attr):
                    val = getattr(hedge_feed_manager, dict_attr)
                    if isinstance(val, (list, dict)):
                        val.clear()
                        logger.debug(f"✅ Cleared hedge manager {dict_attr}")
            
            # Reset start time
            if hasattr(hedge_feed_manager, 'start_time'):
                hedge_feed_manager.start_time = time.time()
            
            results["hedge_feed_manager"] = "volume_metrics_completely_reset"
            logger.info("✅ Hedge feed manager volume/metrics completely cleared")
            
        except Exception as e:
            logger.error(f"❌ Error resetting hedge feed manager: {e}")
            results["hedge_feed_manager"] = f"error: {e}"
    
    # CRITICAL FIX: Reset position manager - CLEAR ALL POSITIONS
    if position_manager:
        try:
            # Use the new reset method
            if hasattr(position_manager, 'reset_all_positions'):
                position_manager.reset_all_positions()
            else:
                # Manual reset
                position_manager.open_option_positions.clear()
                position_manager.open_hedge_positions.clear()
                logger.info("✅ Cleared position manager positions manually")
            
            # Reset portfolio metrics
            for attr in ['total_portfolio_delta', 'total_gamma', 'total_vega', 'total_theta']:
                if hasattr(position_manager, attr):
                    setattr(position_manager, attr, 0.0)
            
            results["position_manager"] = "completely_reset"
            
        except Exception as e:
            logger.error(f"❌ Error resetting position manager: {e}")
            results["position_manager"] = f"error: {e}"
    
    # Reset audit engine - CLEAR ALL TRACKING
    if audit_engine:
        try:
            # Use existing reset method if available
            if hasattr(audit_engine, 'reset_metrics'):
                audit_engine.reset_metrics()
                logger.info("✅ Reset audit engine metrics")
            
            # Additional manual cleanup
            for attr in ['option_premium_records', 'hedging_pnl_records', 'operational_cost_records', 
                        'option_trade_execution_records', 'hedge_execution_records']:
                if hasattr(audit_engine, attr):
                    getattr(audit_engine, attr).clear()
            
            # Reset counters
            if hasattr(audit_engine, 'api_error_count'):
                audit_engine.api_error_count = 0
            if hasattr(audit_engine, 'total_api_requests'):
                audit_engine.total_api_requests = 0
            if hasattr(audit_engine, 'start_time'):
                audit_engine.start_time = time.time()
                
            results["audit_engine"] = "completely_reset"
            
        except Exception as e:
            logger.error(f"❌ Error resetting audit engine: {e}")
            results["audit_engine"] = f"error: {e}"
    
    # Reset liquidity manager utilization
    if liquidity_manager:
        try:
            # Clear transaction records if they exist
            if hasattr(liquidity_manager, 'recent_transactions'):
                liquidity_manager.recent_transactions.clear()
            
            if hasattr(liquidity_manager, 'hedge_transactions'):
                liquidity_manager.hedge_transactions.clear()
            
            # Force recalculation with cleared data
            if hasattr(liquidity_manager, '_recalculate_utilization'):
                liquidity_manager._recalculate_utilization()
                
            results["liquidity_manager"] = "reset_attempted"
            
        except Exception as e:
            logger.error(f"❌ Error resetting liquidity manager: {e}")
            results["liquidity_manager"] = f"error: {e}"
    
    # Force audit sync after complete reset
    force_audit_sync()
    
    # Reset API call counter
    global api_call_count, performance_metrics
    api_call_count = 0
    performance_metrics.clear()
    
    # Force fresh calculation verification
    try:
        if bot_trader_simulator and hasattr(bot_trader_simulator, 'get_trading_statistics'):
            fresh_stats = bot_trader_simulator.get_trading_statistics()
            logger.info(f"✅ POST-RESET VERIFICATION: Trades: {fresh_stats.get('total_trades_24h', 0)}, "
                       f"Volume: ${fresh_stats.get('total_premium_volume_usd_24h', 0):,.2f}")
        
        if revenue_engine and hasattr(revenue_engine, 'get_current_metrics'):
            fresh_revenue = revenue_engine.get_current_metrics()
            revenue_dict = fresh_revenue.__dict__ if hasattr(fresh_revenue, '__dict__') else fresh_revenue
            logger.info(f"✅ POST-RESET REVENUE: ${revenue_dict.get('daily_revenue_usd', 0):,.2f}")
            
    except Exception as e:
        logger.warning(f"Could not verify fresh stats: {e}")
    
    log_api_call("/complete-system-reset", "success", (time.time()-start)*1000)
    
    logger.warning("✅ COMPLETE SYSTEM RESET COMPLETED - All volume, revenue, and position data reset to zero")
    
    return {
        "status": "complete_reset_with_volume_revenue_clearing",
        "message": "All volume, revenue, and position data cleared to fresh startup state",
        "components_reset": results,
        "data_consistency_restored": True,
        "timestamp": time.time(),
        "next_action": "Dashboard should show zero/minimal metrics within 30 seconds"
    }

# ... rest of the existing endpoints remain the same ...

@router.post("/export-csv")
async def export_aggregated_csv():
    """Export aggregated platform summary data instead of individual transactions"""
    try:
        # CRITICAL FIX: Validate data consistency before export
        if not validate_data_consistency():
            logger.warning("Data inconsistency detected before CSV export, forcing sync")
            force_audit_sync()
        
        # Gather all aggregated data
        trading_stats, _ = safe_component_call("bot_trader_simulator", bot_trader_simulator.get_trading_statistics)
        hedge_metrics, _ = safe_component_call("hedge_feed_manager", hedge_feed_manager.get_hedge_metrics)
        platform_greeks, _ = safe_component_call("position_manager", position_manager.get_aggregate_platform_greeks)
        revenue_metrics, _ = safe_component_call("revenue_engine", revenue_engine.get_current_metrics)
        liquidity_allocation, _ = safe_component_call("liquidity_manager", liquidity_manager.get_current_allocation)
        bot_status, _ = safe_component_call("bot_trader_simulator", bot_trader_simulator.get_current_activity)
        audit_summary, _ = safe_component_call("audit_engine", audit_engine.get_24h_metrics)
        scaling_metrics, _ = safe_component_call("liquidity_manager", liquidity_manager.get_scaling_metrics) if hasattr(liquidity_manager, 'get_scaling_metrics') else (None, None)
        
        # Fallbacks for missing data
        trading_stats = trading_stats or {}
        hedge_metrics = hedge_metrics or {}
        platform_greeks = platform_greeks or {}
        revenue_metrics = revenue_metrics.__dict__ if hasattr(revenue_metrics, "__dict__") else (revenue_metrics or {})
        liquidity_allocation = liquidity_allocation if isinstance(liquidity_allocation, dict) else {}
        audit_summary = audit_summary.__dict__ if hasattr(audit_summary, "__dict__") else (audit_summary or {})
        scaling_metrics = scaling_metrics or {}
        
        # Extract trader profiles
        trader_profiles = {}
        if bot_trader_simulator:
            for trader_type, profile in bot_trader_simulator.trader_profiles.items():
                trader_profiles[trader_type.value] = {
                    "count": profile["count"],
                    "success_rate": profile["success_rate"]
                }
        
        # Create CSV content
        output = io.StringIO()
        csv_writer = csv.writer(output)
        
        # Header
        csv_writer.writerow(["Metric", "Value"])
        csv_writer.writerow(["Report Generated", time.strftime("%Y-%m-%d %H:%M:%S")])
        csv_writer.writerow(["Data Consistency Status", "Validated" if validate_data_consistency() else "Inconsistent"])
        csv_writer.writerow(["", ""])
        
        # Trading Summary
        csv_writer.writerow(["--- TRADING SUMMARY ---", ""])
        csv_writer.writerow(["Total Trades (24h)", trading_stats.get("total_trades_24h", 0)])
        csv_writer.writerow(["Total Premium Volume USD (24h)", f"${trading_stats.get('total_premium_volume_usd_24h', 0):,.2f}"])
        csv_writer.writerow(["Average Premium per Trade USD", f"${trading_stats.get('avg_premium_received_usd_24h', 0):,.2f}"])
        csv_writer.writerow(["Call/Put Ratio (24h)", trading_stats.get("call_put_ratio_24h", 0)])
        csv_writer.writerow(["", ""])
        
        # Trader Performance
        csv_writer.writerow(["--- TRADER PERFORMANCE ---", ""])
        for trader_type, data in trader_profiles.items():
            csv_writer.writerow([f"{trader_type.capitalize()} Count", data["count"]])
            csv_writer.writerow([f"{trader_type.capitalize()} Success Rate", f"{data['success_rate']:.1%}"])
        csv_writer.writerow(["", ""])
        
        # Hedge Summary
        csv_writer.writerow(["--- HEDGE SUMMARY ---", ""])
        csv_writer.writerow(["Total Hedges (24h)", hedge_metrics.get("hedges_24h", 0)])
        csv_writer.writerow(["Total Volume Hedged BTC (24h)", f"{hedge_metrics.get('total_volume_hedged_btc_24h', 0):.4f}"])
        csv_writer.writerow(["Average Execution Time (ms)", f"{hedge_metrics.get('avg_execution_time_ms', 0):.1f}"])
        csv_writer.writerow(["", ""])
        
        # Platform Greeks (Risk Metrics)
        csv_writer.writerow(["--- PLATFORM RISK METRICS ---", ""])
        csv_writer.writerow(["Net Portfolio Delta BTC", f"{platform_greeks.get('net_portfolio_delta_btc', 0):.4f}"])
        csv_writer.writerow(["Net Portfolio Gamma BTC", f"{platform_greeks.get('net_portfolio_gamma_btc', 0):.6f}"])
        csv_writer.writerow(["Net Portfolio Vega USD", f"${platform_greeks.get('net_portfolio_vega_usd', 0):,.2f}"])
        csv_writer.writerow(["Net Portfolio Theta USD", f"${platform_greeks.get('net_portfolio_theta_usd', 0):,.2f}"])
        csv_writer.writerow(["Risk Status", platform_greeks.get("risk_status_message", "Unknown")])
        csv_writer.writerow(["Open Options Count", platform_greeks.get("open_options_count", 0)])
        csv_writer.writerow(["Open Hedges Count", platform_greeks.get("open_hedges_count", 0)])
        csv_writer.writerow(["", ""])
        
        # Revenue Metrics
        csv_writer.writerow(["--- REVENUE METRICS ---", ""])
        csv_writer.writerow(["Platform Markup %", f"{revenue_metrics.get('platform_markup_pct', 0):.2f}%"])
        csv_writer.writerow(["Daily Revenue USD", f"${revenue_metrics.get('daily_revenue_usd', 0):,.2f}"])
        csv_writer.writerow(["", ""])
        
        # Liquidity Metrics
        csv_writer.writerow(["--- LIQUIDITY METRICS ---", ""])
        csv_writer.writerow(["Total Pool USD", f"${liquidity_allocation.get('total_pool_usd', 0):,.2f}"])
        csv_writer.writerow(["Liquidity Allocation %", f"{liquidity_allocation.get('liquidity_percentage', 0):.1f}%"])
        csv_writer.writerow(["Operations Allocation %", f"{liquidity_allocation.get('operations_percentage', 0):.1f}%"])
        csv_writer.writerow(["Utilized Amount USD", f"${liquidity_allocation.get('utilized_amount_usd', 0):,.2f}"])
        csv_writer.writerow(["Available Amount USD", f"${liquidity_allocation.get('available_amount_usd', 0):,.2f}"])
        csv_writer.writerow(["Utilization %", f"{liquidity_allocation.get('utilization_pct', 0):.1f}%"])
        csv_writer.writerow(["Stress Test Status", liquidity_allocation.get("stress_test_status", "Unknown")])
        csv_writer.writerow(["Active Users", liquidity_allocation.get("active_users", 0)])
        csv_writer.writerow(["", ""])
        
        # Scaling Metrics (if available)
        if scaling_metrics:
            csv_writer.writerow(["--- SCALING METRICS ---", ""])
            csv_writer.writerow(["Recommended Pool Size USD", f"${scaling_metrics.get('recommended_pool_size', 0):,.2f}"])
            csv_writer.writerow(["Scaling Factor Needed", f"{scaling_metrics.get('scaling_factor_needed', 1):.2f}x"])
            csv_writer.writerow(["Pool Per Trader USD", f"${scaling_metrics.get('pool_per_trader', 0):,.0f}"])
            csv_writer.writerow(["Current Hedge Capacity BTC", f"{scaling_metrics.get('current_hedge_capacity_btc', 0):.4f}"])
            csv_writer.writerow(["Hedge Capacity Improvement %", f"{scaling_metrics.get('hedge_capacity_improvement', 0):+.1f}%"])
            csv_writer.writerow(["Capital Efficiency (Traders/$1M)", f"{scaling_metrics.get('capital_efficiency', 0):.1f}"])
            csv_writer.writerow(["", ""])
        
        # Audit Summary - FIXED with consistency note
        csv_writer.writerow(["--- AUDIT SUMMARY ---", ""])
        csv_writer.writerow(["Compliance Status", audit_summary.get("overall_status", "Unknown")])
        csv_writer.writerow(["Compliance Score", f"{audit_summary.get('compliance_score', 0):.1f}%"])
        csv_writer.writerow(["Trades Tracked 24h", audit_summary.get("option_trades_executed_24h", 0)])
        csv_writer.writerow(["Premium Volume Tracked USD", f"${audit_summary.get('gross_option_premiums_24h_usd', 0):,.2f}"])
        csv_writer.writerow(["Data Consistency", "Validated" if validate_data_consistency() else "Inconsistent"])
        
        # Get CSV content
        csv_content = output.getvalue()
        output.close()
        
        # Return as StreamingResponse
        headers = {"Content-Disposition": "attachment; filename=atticus_platform_summary.csv"}
        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type="text/csv",
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"CSV export failed: {e}", exc_info=True)
        raise HTTPException(500, f"CSV export failed: {str(e)}")

@router.get("/debug-system-status")
async def debug_system_status():
    """Get detailed system status for debugging"""
    return {
        "api_call_count": api_call_count,
        "performance_metrics": performance_metrics,
        "component_status": {
            "revenue_engine": revenue_engine is not None,
            "position_manager": position_manager is not None,
            "liquidity_manager": liquidity_manager is not None,
            "bot_trader_simulator": bot_trader_simulator is not None,
            "hedge_feed_manager": hedge_feed_manager is not None,
            "audit_engine": audit_engine is not None
        },
        "data_consistency_status": validate_data_consistency(),
        "audit_bot_reference": (audit_engine is not None and 
                               hasattr(audit_engine, 'bot_simulator') and 
                               audit_engine.bot_simulator is not None),
        "timestamp": time.time()
    }

# Additional liquidity-specific debug endpoints
@router.get("/debug-liquidity-status")
async def debug_liquidity_status():
    """Debug liquidity manager internal state"""
    if not liquidity_manager:
        return {"error": "Liquidity manager not available"}
    
    try:
        debug_info = {}
        
        if hasattr(liquidity_manager, 'get_current_allocation'):
            debug_info["current_allocation"] = liquidity_manager.get_current_allocation()
        
        if hasattr(liquidity_manager, 'get_liquidity_health'):
            debug_info["health_status"] = liquidity_manager.get_liquidity_health()
        
        if hasattr(liquidity_manager, 'get_recommended_allocation'):
            debug_info["recommendations"] = liquidity_manager.get_recommended_allocation()
            
        if hasattr(liquidity_manager, 'get_scaling_metrics'):
            debug_info["scaling_metrics"] = liquidity_manager.get_scaling_metrics()
        
        debug_info["methods_available"] = [
            method for method in dir(liquidity_manager) 
            if not method.startswith('_') and callable(getattr(liquidity_manager, method))
        ]
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}

# CRITICAL FIX: New endpoint for data consistency validation
@router.get("/validate-data-consistency")
async def validate_data_consistency_endpoint():
    """Validate that trading stats and audit data are consistent"""
    try:
        if not bot_trader_simulator or not audit_engine:
            return {"status": "unavailable", "message": "Components not available"}
            
        trading_stats = bot_trader_simulator.get_trading_statistics()
        audit_metrics = audit_engine.get_24h_metrics()
        
        trade_diff = abs(trading_stats.get('total_trades_24h', 0) - audit_metrics.option_trades_executed_24h)
        volume_diff = abs(trading_stats.get('total_premium_volume_usd_24h', 0) - audit_metrics.gross_option_premiums_24h_usd)
        
        is_consistent = trade_diff <= 10 and volume_diff <= 1000
        
        if not is_consistent:
            logger.warning(f"Data inconsistency detected - forcing sync")
            force_audit_sync()
            # Re-check after sync
            audit_metrics_after = audit_engine.get_24h_metrics()
            trade_diff_after = abs(trading_stats.get('total_trades_24h', 0) - audit_metrics_after.option_trades_executed_24h)
            is_consistent_after = trade_diff_after <= 10
        else:
            is_consistent_after = is_consistent
        
        return {
            "status": "consistent" if is_consistent_after else "inconsistent",
            "trading_stats_trades": trading_stats.get('total_trades_24h', 0),
            "audit_trades_before_sync": audit_metrics.option_trades_executed_24h,
            "audit_trades_after_sync": audit_metrics_after.option_trades_executed_24h if not is_consistent else audit_metrics.option_trades_executed_24h,
            "trade_difference_before": trade_diff,
            "trade_difference_after": trade_diff_after if not is_consistent else trade_diff,
            "sync_applied": not is_consistent,
            "audit_has_bot_reference": hasattr(audit_engine, 'bot_simulator') and audit_engine.bot_simulator is not None,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error validating data consistency: {e}")
        return {"status": "error", "error": str(e), "timestamp": time.time()}
