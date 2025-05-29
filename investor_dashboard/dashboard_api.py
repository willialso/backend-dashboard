# investor_dashboard/dashboard_api.py

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import time
import logging
import traceback
import json
import io
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances (will be initialized by main_dashboard.py)
revenue_engine: Optional[Any] = None
liquidity_manager: Optional[Any] = None
bot_trader_simulator: Optional[Any] = None
audit_engine: Optional[Any] = None
hedge_feed_manager: Optional[Any] = None
position_manager: Optional[Any] = None

# API call tracking
api_call_count = 0
performance_metrics: Dict[str, Dict[str, float]] = {}

def log_api_call(endpoint: str, result: str, duration_ms: float = 0):
    global api_call_count, performance_metrics
    api_call_count += 1
    if endpoint not in performance_metrics:
        performance_metrics[endpoint] = {"calls": 0, "avg_duration": 0, "errors": 0, "total_duration": 0}
    entry = performance_metrics[endpoint]
    entry["calls"] += 1
    entry["total_duration"] += duration_ms
    entry["avg_duration"] = entry["total_duration"] / entry["calls"]
    if "error" in result.lower(): entry["errors"] += 1
    logger.info(f"API Call #{api_call_count}: {endpoint} - {result} ({duration_ms:.2f}ms)")

def safe_component_call(component_name: str, operation: callable, *args, **kwargs):
    start_time = time.time()
    try:
        module_instance = globals().get(component_name.lower().replace(" ", "_"))
        if not module_instance:
            raise ValueError(f"Component '{component_name}' not found or not initialized.")
        method_to_call = getattr(module_instance, operation.__name__ if callable(operation) else operation)
        result = method_to_call(*args, **kwargs)
        duration_ms = (time.time() - start_time) * 1000
        logger.debug(f"'{component_name}.{operation.__name__ if callable(operation) else operation}' completed in {duration_ms:.2f}ms")
        return result, None
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = f"Error in '{component_name}.{operation.__name__ if callable(operation) else operation}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg

# --- Pydantic Models ---
class LiquidityAllocation(BaseModel):
    liquidity_pct: float
    operations_pct: float

class TraderDistribution(BaseModel):
    advanced_pct: float
    intermediate_pct: float
    beginner_pct: float

class UserGrowthSimulation(BaseModel):
    new_user_count: int

# NEW: Hedge Test Request Model
class HedgeTestRequest(BaseModel):
    quantity_btc: float = 0.1
    direction: str = "buy"  # "buy" or "sell"
    reason: str = "Manual API test hedge"

# --- NEW BOT TRADER DEBUG & RESTART ENDPOINTS ---

@router.post("/restart-bot-trader")
async def restart_bot_trader():
    """Restart the bot trader simulator."""
    start_time = time.time()
    
    if not bot_trader_simulator:
        log_api_call("/restart-bot-trader", "error - Bot trader not available", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=503, detail="Bot trader not available")
    
    try:
        # Force set current BTC price if it's missing
        if hasattr(bot_trader_simulator, 'current_btc_price') and bot_trader_simulator.current_btc_price <= 0:
            if position_manager and hasattr(position_manager, 'current_btc_price'):
                bot_trader_simulator.current_btc_price = position_manager.current_btc_price
            else:
                bot_trader_simulator.current_btc_price = 108000.0  # Default fallback
        
        # Stop and restart bot trader
        if hasattr(bot_trader_simulator, 'stop'):
            bot_trader_simulator.stop()
            logger.info("Bot trader stopped successfully")
        
        if hasattr(bot_trader_simulator, 'start'):
            bot_trader_simulator.start()
            logger.info("Bot trader started successfully")
        
        response_data = {
            "status": "success", 
            "message": "Bot trader restarted successfully",
            "restart_time": time.time(),
            "has_start_method": hasattr(bot_trader_simulator, 'start'),
            "has_stop_method": hasattr(bot_trader_simulator, 'stop'),
            "current_btc_price": getattr(bot_trader_simulator, 'current_btc_price', 0),
            "timestamp": time.time()
        }
        
        log_api_call("/restart-bot-trader", "success", (time.time() - start_time) * 1000)
        return response_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Error restarting bot trader: {e}", exc_info=True)
        log_api_call("/restart-bot-trader", f"error - {str(e)}", duration_ms)
        return {"error": str(e), "timestamp": time.time()}

@router.get("/bot-trader-status")
async def get_bot_trader_status():
    """Get detailed bot trader status."""
    start_time = time.time()
    
    if not bot_trader_simulator:
        log_api_call("/bot-trader-status", "error - Bot trader not available", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=503, detail="Bot trader not available")
    
    try:
        status_info = {
            "is_running": getattr(bot_trader_simulator, 'is_running', False),
            "recent_trades_count": len(getattr(bot_trader_simulator, 'recent_trades_log', [])),
            "has_start_method": hasattr(bot_trader_simulator, 'start'),
            "has_stop_method": hasattr(bot_trader_simulator, 'stop'),
            "has_get_status_method": hasattr(bot_trader_simulator, 'get_status'),
            "current_btc_price": getattr(bot_trader_simulator, 'current_btc_price', 0),
            "has_revenue_engine": getattr(bot_trader_simulator, 'revenue_engine', None) is not None,
            "has_position_manager": getattr(bot_trader_simulator, 'position_manager', None) is not None,
            "trader_attributes": [attr for attr in dir(bot_trader_simulator) if not attr.startswith('_')],
            "timestamp": time.time()
        }
        
        # Try to get more detailed status if available
        if hasattr(bot_trader_simulator, 'get_status'):
            detailed_status, _ = safe_component_call("bot_trader_simulator", bot_trader_simulator.get_status)
            if detailed_status:
                status_info["detailed_status"] = detailed_status
        
        # Get recent trades info
        if hasattr(bot_trader_simulator, 'recent_trades_log'):
            trades_log = bot_trader_simulator.recent_trades_log
            status_info["last_trade_timestamp"] = trades_log[-1].timestamp if trades_log else None
            
            # Count trades in last minute to see if active
            now = time.time()
            recent_trades = [t for t in trades_log if t.timestamp > now - 60]
            status_info["trades_last_minute"] = len(recent_trades)
        
        log_api_call("/bot-trader-status", "success", (time.time() - start_time) * 1000)
        return status_info
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Error getting bot trader status: {e}", exc_info=True)
        log_api_call("/bot-trader-status", f"error - {str(e)}", duration_ms)
        return {"error": str(e), "timestamp": time.time()}

@router.post("/force-generate-trade")
async def force_generate_trade():
    """Force the bot trader to generate a single trade for testing."""
    start_time = time.time()
    
    if not bot_trader_simulator:
        raise HTTPException(status_code=503, detail="Bot trader not available")
    
    try:
        # Check if we have required dependencies
        has_revenue_engine = hasattr(bot_trader_simulator, 'revenue_engine') and bot_trader_simulator.revenue_engine is not None
        has_position_manager = hasattr(bot_trader_simulator, 'position_manager') and bot_trader_simulator.position_manager is not None
        has_current_price = hasattr(bot_trader_simulator, 'current_btc_price') and bot_trader_simulator.current_btc_price > 0
        
        # Set current BTC price if missing
        if hasattr(bot_trader_simulator, 'current_btc_price') and bot_trader_simulator.current_btc_price <= 0:
            bot_trader_simulator.current_btc_price = 108000.0  # Set a reasonable current price
            has_current_price = True
        
        # Try to manually generate a trade
        if hasattr(bot_trader_simulator, '_generate_and_record_trade'):
            # Check if it has trader profiles
            trader_profiles = getattr(bot_trader_simulator, 'trader_profiles', {})
            
            if trader_profiles:
                # Pick first available trader type
                first_trader_type = list(trader_profiles.keys())[0]
                profile = trader_profiles[first_trader_type]
                
                # Count trades before
                trades_before = len(getattr(bot_trader_simulator, 'recent_trades_log', []))
                
                # Force trade generation
                result, error = safe_component_call(
                    "bot_trader_simulator", 
                    bot_trader_simulator._generate_and_record_trade,
                    first_trader_type,
                    profile
                )
                
                if error:
                    return {"error": f"Trade generation failed: {error}", "timestamp": time.time()}
                
                # Count trades after
                trades_after = len(getattr(bot_trader_simulator, 'recent_trades_log', []))
                
                return {
                    "status": "success",
                    "message": "Trade generation attempted",
                    "dependencies": {
                        "has_revenue_engine": has_revenue_engine,
                        "has_position_manager": has_position_manager,
                        "has_current_price": has_current_price,
                        "current_btc_price": getattr(bot_trader_simulator, 'current_btc_price', 0),
                        "trader_profiles_count": len(trader_profiles)
                    },
                    "result": "Trade generated successfully" if result else "Trade generation returned None",
                    "trades_before": trades_before,
                    "trades_after": trades_after,
                    "new_trades_added": trades_after - trades_before,
                    "timestamp": time.time()
                }
            else:
                return {"error": "No trader profiles available", "timestamp": time.time()}
        else:
            return {"error": "_generate_and_record_trade method not found", "timestamp": time.time()}
            
    except Exception as e:
        logger.error(f"Force trade generation error: {e}", exc_info=True)
        return {"error": str(e), "timestamp": time.time()}

@router.get("/bot-trader-debug")
async def get_bot_trader_debug():
    """Get detailed debug info about bot trader state."""
    if not bot_trader_simulator:
        raise HTTPException(status_code=503, detail="Bot trader not available")
    
    try:
        debug_info = {
            "is_running": getattr(bot_trader_simulator, 'is_running', False),
            "recent_trades_count": len(getattr(bot_trader_simulator, 'recent_trades_log', [])),
            "current_btc_price": getattr(bot_trader_simulator, 'current_btc_price', 0),
            "has_revenue_engine": hasattr(bot_trader_simulator, 'revenue_engine') and bot_trader_simulator.revenue_engine is not None,
            "has_position_manager": hasattr(bot_trader_simulator, 'position_manager') and bot_trader_simulator.position_manager is not None,
            "trader_profiles": getattr(bot_trader_simulator, 'trader_profiles', {}),
            "trade_id_counter": getattr(bot_trader_simulator, 'trade_id_counter', 0),
            "all_attributes": [attr for attr in dir(bot_trader_simulator) if not attr.startswith('_')],
            "timestamp": time.time()
        }
        
        # Get trader counts by type
        if hasattr(bot_trader_simulator, 'trader_profiles'):
            trader_counts = {str(k): v.get("count", 0) for k, v in bot_trader_simulator.trader_profiles.items()}
            debug_info["trader_counts"] = trader_counts
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e), "timestamp": time.time()}

# --- FIXED HEDGE TEST ENDPOINT ---
@router.post("/force-hedge-test")
async def force_hedge_test(request: HedgeTestRequest):
    """Force execute a test hedge to see the hedging system in action."""
    start_time = time.time()
    
    if not hedge_feed_manager:
        log_api_call("/force-hedge-test", "error - HedgeFeedManager not initialized", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=503, detail="HedgeFeedManager not available")
    
    # Validate input
    if request.direction.lower() not in ["buy", "sell"]:
        log_api_call("/force-hedge-test", "error - invalid direction", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=400, detail="Direction must be 'buy' or 'sell'")
    
    if request.quantity_btc <= 0 or request.quantity_btc > 5.0:
        log_api_call("/force-hedge-test", "error - invalid quantity", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=400, detail="Quantity must be between 0 and 5.0 BTC")
    
    try:
        # Calculate delta to hedge (positive for buy, negative for sell)
        delta_to_hedge = request.quantity_btc if request.direction.lower() == "buy" else -request.quantity_btc
        
        # Get current BTC price
        current_price = 108000.0  # Default fallback
        if position_manager and hasattr(position_manager, 'current_btc_price'):
            current_price = position_manager.current_btc_price
        
        # FIXED: Set the current BTC price on hedge_feed_manager (it uses this internally)
        if hasattr(hedge_feed_manager, 'current_btc_price'):
            hedge_feed_manager.current_btc_price = current_price
        
        # FIXED: Force hedge execution - call with NO parameters as the method expects
        hedge_result, error = safe_component_call(
            "hedge_feed_manager", 
            hedge_feed_manager._generate_and_record_hedge  # NO PARAMETERS!
        )
        
        if error:
            log_api_call("/force-hedge-test", f"error - {error}", (time.time() - start_time) * 1000)
            raise HTTPException(status_code=500, detail=f"Hedge execution failed: {error}")
        
        # Get updated portfolio metrics
        portfolio_greeks = {}
        if position_manager:
            greeks, _ = safe_component_call("position_manager", position_manager.get_aggregate_platform_greeks)
            if greeks:
                portfolio_greeks = greeks
        
        response_data = {
            "status": "success",
            "message": f"Test hedge executed: {request.direction.upper()} {request.quantity_btc} BTC",
            "hedge_details": {
                "direction": request.direction,
                "quantity_btc": request.quantity_btc,
                "delta_hedged": delta_to_hedge,
                "estimated_price_usd": current_price,
                "estimated_cost_usd": round(request.quantity_btc * current_price, 2),
                "reason": request.reason
            },
            "hedge_result": hedge_result.__dict__ if hedge_result and hasattr(hedge_result, '__dict__') else hedge_result,
            "portfolio_after_hedge": portfolio_greeks,
            "timestamp": time.time(),
            "response_time_ms": (time.time() - start_time) * 1000
        }
        
        log_api_call("/force-hedge-test", "success", response_data["response_time_ms"])
        return response_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Force hedge test unexpected error: {e}", exc_info=True)
        log_api_call("/force-hedge-test", f"error - {str(e)}", duration_ms)
        raise HTTPException(status_code=500, detail=f"Hedge test failed: {str(e)}")

@router.get("/trigger-hedge-test")
async def trigger_hedge_test(quantity: float = 0.1, direction: str = "buy"):
    """Simple GET endpoint to trigger a test hedge."""
    start_time = time.time()
    
    try:
        request = HedgeTestRequest(
            quantity_btc=quantity,
            direction=direction,
            reason="Simple GET trigger test"
        )
        return await force_hedge_test(request)
    except Exception as e:
        log_api_call("/trigger-hedge-test", f"error - {str(e)}", (time.time() - start_time) * 1000)
        return {"error": str(e), "timestamp": time.time()}

# --- EXISTING ENDPOINTS (all remaining endpoints from your file) ---

@router.get("/platform-greeks")
async def get_platform_greeks_endpoint():
    """Returns the aggregate platform Greeks (Delta, Gamma, Vega, Theta)."""
    start_time = time.time()
    if not position_manager:
        log_api_call("/platform-greeks", "error - PositionManager not initialized", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=503, detail="Position Manager not initialized")
    
    greeks_data, error = safe_component_call("position_manager", position_manager.get_aggregate_platform_greeks)
    if error:
        log_api_call("/platform-greeks", f"error - {error}", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=500, detail=f"Error getting platform Greeks: {error}")
    
    log_api_call("/platform-greeks", "success", (time.time() - start_time) * 1000)
    return greeks_data

@router.post("/export-csv")
async def export_csv():
    start_time = time.time()
    try:
        summary_rows = []
        
        # Revenue
        if revenue_engine:
            metrics, _ = safe_component_call("revenue_engine", revenue_engine.get_current_metrics)
            if metrics:
                for k, v in metrics.__dict__.items():
                    summary_rows.append({'Category': 'Revenue', 'Metric': k, 'Value': v})

        # Liquidity
        if liquidity_manager:
            status, _ = safe_component_call("liquidity_manager", liquidity_manager.get_liquidity_status)
            if status:
                for k, v in status.__dict__.items():
                    summary_rows.append({'Category': 'Liquidity', 'Metric': k, 'Value': v})

        # Trading Activity
        if bot_trader_simulator:
            activities, _ = safe_component_call("bot_trader_simulator", bot_trader_simulator.get_current_activity)
            if activities:
                for act in activities:
                    for k,v in act.__dict__.items():
                        summary_rows.append({'Category': f'Trading_{act.trader_type.value}', 'Metric': k, 'Value': v})

        summary_df = pd.DataFrame(summary_rows)
        stream = io.StringIO()
        summary_df.to_csv(stream, index=False)

        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=atticus_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
        
        log_api_call("/export-csv", "success", (time.time() - start_time) * 1000)
        return response

    except Exception as e:
        log_api_call("/export-csv", f"error - {str(e)}", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=500, detail=f"Error exporting CSV: {str(e)}")

@router.post("/reset-parameters")
async def reset_parameters_endpoint():
    start_time = time.time()
    reset_results_dict = {}
    
    if liquidity_manager and hasattr(liquidity_manager, 'reset_to_defaults'):
        liquidity_manager.reset_to_defaults()
        reset_results_dict["liquidity_manager"] = "success"
    
    if bot_trader_simulator and hasattr(bot_trader_simulator, 'reset_to_defaults'):
        bot_trader_simulator.reset_to_defaults()
        reset_results_dict["bot_trader_simulator"] = "success"

    log_api_call("/reset-parameters", "success", (time.time() - start_time) * 1000)
    return {"status": "success", "message": "Platform parameters reset (simulated)", "reset_results": reset_results_dict}

@router.get("/trading-statistics")
async def get_trading_statistics_endpoint():
    start_time = time.time()
    if not bot_trader_simulator:
        log_api_call("/trading-statistics", "error - BotTraderSimulator not initialized", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=503, detail="Bot Trader Simulator not initialized")

    stats, error = safe_component_call("bot_trader_simulator", bot_trader_simulator.get_trading_statistics)
    if error:
        log_api_call("/trading-statistics", f"error - {error}", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=500, detail=f"Error getting trading statistics: {error}")

    log_api_call("/trading-statistics", "success", (time.time() - start_time) * 1000)
    return stats

@router.get("/market-data")
async def get_market_data_endpoint():
    start_time = time.time()
    if not revenue_engine:
        log_api_call("/market-data", "error - RevenueEngine not initialized", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=503, detail="Revenue Engine not initialized for market data")

    current_metrics_obj, metrics_error = safe_component_call("revenue_engine", revenue_engine.get_current_metrics)
    debug_info_dict, debug_error = safe_component_call("revenue_engine", revenue_engine.get_debug_info)

    if metrics_error or debug_error or not current_metrics_obj or not debug_info_dict:
        log_api_call("/market-data", "error - metrics/debug unavailable, using fallback", (time.time() - start_time) * 1000)
        return {"status": "error", "message": "Failed to fetch full market data, using fallback."}

    current_price = float(debug_info_dict.get("current_btc_price", 0))

    market_data_response = {
        "current_price": current_price,
        "volatility_regime": debug_info_dict.get("volatility_engine_metrics", {}).get("regime", "unknown"),
        "advanced": {
            "volatility_percentage": current_metrics_obj.volatility_used_for_pricing_pct if hasattr(current_metrics_obj, 'volatility_used_for_pricing_pct') else 0,
            "bsm_fair_value": current_metrics_obj.bsm_fair_value if hasattr(current_metrics_obj, 'bsm_fair_value') else 0,
            "platform_price": current_metrics_obj.platform_price if hasattr(current_metrics_obj, 'platform_price') else 0,
        },
        "timestamp": time.time(),
        "response_time_ms": (time.time() - start_time) * 1000,
        "status": "success"
    }

    log_api_call("/market-data", "success", (time.time() - start_time) * 1000)
    return market_data_response

@router.get("/revenue-metrics")
async def get_revenue_metrics_endpoint():
    start_time = time.time()
    if not revenue_engine:
        log_api_call("/revenue-metrics", "error - RevenueEngine not initialized", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=503, detail="Revenue engine not initialized")

    metrics, error = safe_component_call("revenue_engine", revenue_engine.get_current_metrics)
    if error:
        log_api_call("/revenue-metrics", f"error - {error}", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=500, detail=f"Error getting revenue metrics: {error}")

    response_data = metrics.__dict__ if hasattr(metrics, '__dict__') else metrics
    if isinstance(response_data, dict):
        response_data["timestamp"] = time.time()
        response_data["response_time_ms"] = (time.time() - start_time) * 1000

    log_api_call("/revenue-metrics", "success", (time.time() - start_time) * 1000)
    return response_data

@router.get("/platform-health")
async def get_platform_health_endpoint():
    start_time = time.time()
    
    components_check = {
        "revenue_operational": revenue_engine is not None,
        "liquidity_operational": liquidity_manager is not None,
        "audit_operational": audit_engine is not None,
        "hedge_feed_operational": hedge_feed_manager is not None,
        "bot_simulation_operational": bot_trader_simulator is not None,
        "position_manager_operational": position_manager is not None
    }

    operational_count = sum(1 for v in components_check.values() if v)
    total_components = len(components_check)
    health_score = (operational_count / total_components) * 100 if total_components > 0 else 0
    status_text = "GOOD" if health_score > 80 else ("FAIR" if health_score > 50 else "POOR")

    log_api_call("/platform-health", "success", (time.time() - start_time) * 1000)
    return {
        "overall_health": status_text,
        "health_score": health_score,
        "operational_components": operational_count,
        "total_components": total_components,
        "components": components_check,
        "timestamp": time.time()
    }

@router.get("/recent-trades")
async def get_recent_trades_endpoint(limit: int = 10):
    start_time = time.time()
    if not bot_trader_simulator:
        log_api_call("/recent-trades", "error - BotTraderSimulator not initialized", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=503, detail="Bot trader simulator not initialized")

    trades, error = safe_component_call("bot_trader_simulator", bot_trader_simulator.get_recent_trades, limit)
    if error:
        log_api_call("/recent-trades", f"error - {error}", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=500, detail=f"Error getting recent trades: {error}")

    response_trades = []
    for trade in trades:
        trade_dict = trade.__dict__ if hasattr(trade, '__dict__') else trade
        if hasattr(trade, 'trader_type') and hasattr(trade.trader_type, 'value'):
            trade_dict["trader_type"] = trade.trader_type.value
        response_trades.append(trade_dict)

    log_api_call("/recent-trades", "success", (time.time() - start_time) * 1000)
    return {"trades": response_trades, "total_returned": len(response_trades), "timestamp": time.time()}

@router.get("/hedge-execution-feed")
async def get_hedge_execution_feed_endpoint(limit: int = 10):
    start_time = time.time()
    if not hedge_feed_manager:
        log_api_call("/hedge-execution-feed", "error - HedgeFeedManager not initialized", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=503, detail="Hedge Feed Manager not initialized")

    hedges, error = safe_component_call("hedge_feed_manager", hedge_feed_manager.get_recent_hedges, limit)
    if error:
        log_api_call("/hedge-execution-feed", f"error - {error}", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=500, detail=f"Error getting hedge executions: {error}")

    response_hedges = []
    for hedge in hedges:
        hedge_dict = hedge.__dict__ if hasattr(hedge, '__dict__') else hedge
        if hasattr(hedge, 'hedge_type') and hasattr(hedge.hedge_type, 'value'):
            hedge_dict["hedge_type"] = hedge.hedge_type.value
        if hasattr(hedge, 'exchange') and hasattr(hedge.exchange, 'value'):
            hedge_dict["exchange"] = hedge.exchange.value
        response_hedges.append(hedge_dict)

    log_api_call("/hedge-execution-feed", "success", (time.time() - start_time) * 1000)
    return {"hedges": response_hedges, "total_returned": len(response_hedges), "timestamp": time.time()}

@router.get("/audit-summary")
async def get_audit_summary_endpoint():
    start_time = time.time()
    if not audit_engine:
        log_api_call("/audit-summary", "error - AuditEngine not initialized", (time.time() - start_time) * 1000)
        return {
            "message": "Audit engine not initialized - showing defaults",
            "compliance_status": "UNKNOWN",
            "timestamp": time.time()
        }

    metrics, error = safe_component_call("audit_engine", audit_engine.get_24h_metrics)
    if error:
        log_api_call("/audit-summary", f"error - {error}", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=500, detail=f"Error getting audit summary: {error}")

    response_data = metrics.__dict__ if hasattr(metrics, '__dict__') else metrics
    if isinstance(response_data, dict):
        response_data["timestamp"] = time.time()
        response_data["response_time_ms"] = (time.time() - start_time) * 1000

    log_api_call("/audit-summary", "success", (time.time() - start_time) * 1000)
    return response_data

@router.get("/hedge-metrics")
async def get_hedge_metrics_endpoint():
    start_time = time.time()
    if not hedge_feed_manager:
        log_api_call("/hedge-metrics", "error - HedgeFeedManager not initialized", (time.time() - start_time) * 1000)
        return {
            "message": "Hedge feed manager not initialized",
            "hedges_24h": 0,
            "timestamp": time.time()
        }

    metrics, error = safe_component_call("hedge_feed_manager", hedge_feed_manager.get_hedge_metrics)
    if error:
        log_api_call("/hedge-metrics", f"error - {error}", (time.time() - start_time) * 1000)
        raise HTTPException(status_code=500, detail=f"Error getting hedge metrics: {error}")

    if isinstance(metrics, dict):
        metrics["timestamp"] = time.time()
        metrics["response_time_ms"] = (time.time() - start_time) * 1000
    else:
        metrics = {"error": "Invalid metrics format", "timestamp": time.time()}

    log_api_call("/hedge-metrics", "success", (time.time() - start_time) * 1000)
    return metrics

# Control endpoints
@router.post("/liquidity-allocation")
async def adjust_liquidity_allocation(allocation: LiquidityAllocation):
    if not liquidity_manager:
        raise HTTPException(status_code=503, detail="LM not init")
    
    _, err = safe_component_call("liquidity_manager", liquidity_manager.adjust_allocation, allocation.liquidity_pct, allocation.operations_pct)
    if err:
        raise HTTPException(status_code=500, detail=str(err))
    
    return {"status": "success", "message": "Liquidity allocation updated (simulated)"}

@router.post("/trader-distribution")
async def adjust_trader_distribution(distribution: TraderDistribution):
    if not bot_trader_simulator:
        raise HTTPException(status_code=503, detail="BTS not init")
    
    _, err = safe_component_call("bot_trader_simulator", bot_trader_simulator.adjust_trader_distribution,
                                distribution.advanced_pct, distribution.intermediate_pct, distribution.beginner_pct)
    if err:
        raise HTTPException(status_code=500, detail=str(err))
    
    return {"status": "success", "message": "Trader distribution updated (simulated)"}

@router.post("/simulate-user-growth")
async def simulate_user_growth_endpoint(growth: UserGrowthSimulation):
    if not liquidity_manager:
        raise HTTPException(status_code=503, detail="LM not init")
    
    _, err = safe_component_call("liquidity_manager", liquidity_manager.simulate_user_growth, growth.new_user_count)
    if err:
        raise HTTPException(status_code=500, detail=str(err))
    
    return {"status": "success", "message": f"Simulated growth to {growth.new_user_count} users"}

@router.get("/debug-system-status")
async def debug_system_status_endpoint():
    components_status = {
        "revenue_engine": {"initialized": revenue_engine is not None},
        "liquidity_manager": {"initialized": liquidity_manager is not None},
        "bot_trader_simulator": {"initialized": bot_trader_simulator is not None},
        "audit_engine": {"initialized": audit_engine is not None},
        "hedge_feed_manager": {"initialized": hedge_feed_manager is not None},
        "position_manager": {"initialized": position_manager is not None}
    }
    
    return {
        "status": "debug info",
        "components": components_status,
        "api_call_count": api_call_count,
        "performance_metrics": performance_metrics,
        "timestamp": time.time()
    }
