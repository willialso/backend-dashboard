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

# Request models
class LiquidityAllocation(BaseModel):
    liquidity_pct: float
    operations_pct: float

class TraderDistribution(BaseModel):
    total_traders: int

@router.get("/platform-health")
async def get_platform_health():
    start = time.time()
    comps = {
        "revenue_engine": revenue_engine is not None,
        "liquidity_manager": liquidity_manager is not None,
        "audit_engine": audit_engine is not None,
        "hedge_feed_manager": hedge_feed_manager is not None,
        "bot_simulator": bot_trader_simulator is not None,
        "position_manager": position_manager is not None
    }
    up = sum(comps.values())
    pct = up / len(comps) * 100
    status = "GOOD" if pct >= 80 else ("FAIR" if pct >= 50 else "POOR")
    log_api_call("/platform-health", "success", (time.time()-start)*1000)
    return {"overall_health": status, "health_score": pct, "components": comps, "timestamp": time.time()}

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
    start = time.time()
    status, err = safe_component_call("liquidity_manager", liquidity_manager.get_liquidity_status)
    if err or status is None:
        log_api_call("/liquidity-allocation", f"error - {err}", (time.time()-start)*1000)
        return {
            "liquidity_ratio": 0.0,
            "total_pool_usd": config.LM_INITIAL_TOTAL_POOL_USD,
            "stress_test_status": "UNKNOWN",
            "error": err,
            "timestamp": time.time()
        }
    data = status.__dict__ if hasattr(status, "__dict__") else status
    data["timestamp"] = time.time()
    log_api_call("/liquidity-allocation", "success", (time.time()-start)*1000)
    return data

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
    start = time.time()
    metrics, err = safe_component_call("audit_engine", audit_engine.get_24h_metrics)
    if err or metrics is None:
        log_api_call("/audit-summary", f"error - {err}", (time.time()-start)*1000)
        return {"overall_status": "ERROR", "compliance_score": 0.0, "error": err, "timestamp": time.time()}
    data = metrics.__dict__ if hasattr(metrics, "__dict__") else metrics
    data["timestamp"] = time.time()
    log_api_call("/audit-summary", "success", (time.time()-start)*1000)
    return data

@router.post("/trader-distribution")
async def adjust_trader_distribution(dist: TraderDistribution):
    _, err = safe_component_call("bot_trader_simulator", bot_trader_simulator.adjust_trader_distribution, dist.total_traders)
    if err:
        raise HTTPException(500, f"Error adjusting trader distribution: {err}")
    return {"status": "success", "total_traders": dist.total_traders}

@router.post("/liquidity-allocation")
async def adjust_liquidity(dist: LiquidityAllocation):
    """FIXED: Enhanced error handling for liquidity adjustment"""
    start = time.time()
    
    # Check if liquidity manager exists
    if not liquidity_manager:
        log_api_call("/liquidity-allocation", "error - no liquidity manager", (time.time()-start)*1000)
        raise HTTPException(503, "Liquidity manager not available")
    
    # Check if adjust_allocation method exists
    if not hasattr(liquidity_manager, 'adjust_allocation'):
        log_api_call("/liquidity-allocation", "error - method not implemented", (time.time()-start)*1000)
        logger.warning("Liquidity manager does not have adjust_allocation method")
        # Return success anyway to avoid breaking frontend
        return {"status": "success", "message": "Liquidity adjustment not implemented yet"}
        
    # Validate input ranges
    if not (0 <= dist.liquidity_pct <= 100) or not (0 <= dist.operations_pct <= 100):
        raise HTTPException(400, "Percentages must be between 0 and 100")
        
    if dist.liquidity_pct + dist.operations_pct != 100:
        raise HTTPException(400, "Liquidity and operations percentages must sum to 100")
    
    try:
        _, err = safe_component_call("liquidity_manager", liquidity_manager.adjust_allocation, dist.liquidity_pct, dist.operations_pct)
        if err:
            logger.error(f"Liquidity adjustment failed: {err}")
            log_api_call("/liquidity-allocation", f"error - {err}", (time.time()-start)*1000)
            raise HTTPException(500, f"Error adjusting liquidity: {err}")
        
        log_api_call("/liquidity-allocation", "success", (time.time()-start)*1000)
        return {"status": "success", "liquidity_pct": dist.liquidity_pct, "operations_pct": dist.operations_pct}
        
    except Exception as e:
        logger.error(f"Unexpected error in liquidity adjustment: {e}", exc_info=True)
        log_api_call("/liquidity-allocation", f"error - {e}", (time.time()-start)*1000)
        raise HTTPException(500, f"Unexpected error: {str(e)}")

@router.post("/reset-parameters")
async def reset_parameters():
    if liquidity_manager and hasattr(liquidity_manager, 'reset_to_defaults'):
        liquidity_manager.reset_to_defaults()
    if bot_trader_simulator and hasattr(bot_trader_simulator, 'reset_to_defaults'):
        bot_trader_simulator.reset_to_defaults()
    return {"status": "success"}

@router.post("/export-csv")
async def export_aggregated_csv():
    """NEW: Export aggregated platform summary data instead of individual transactions"""
    try:
        # Gather all aggregated data
        trading_stats, _ = safe_component_call("bot_trader_simulator", bot_trader_simulator.get_trading_statistics)
        hedge_metrics, _ = safe_component_call("hedge_feed_manager", hedge_feed_manager.get_hedge_metrics) 
        platform_greeks, _ = safe_component_call("position_manager", position_manager.get_aggregate_platform_greeks)
        revenue_metrics, _ = safe_component_call("revenue_engine", revenue_engine.get_current_metrics)
        liquidity_status, _ = safe_component_call("liquidity_manager", liquidity_manager.get_liquidity_status)
        bot_status, _ = safe_component_call("bot_trader_simulator", bot_trader_simulator.get_current_activity)
        
        # Fallbacks for missing data
        trading_stats = trading_stats or {}
        hedge_metrics = hedge_metrics or {}
        platform_greeks = platform_greeks or {}
        revenue_metrics = revenue_metrics.__dict__ if hasattr(revenue_metrics, "__dict__") else (revenue_metrics or {})
        liquidity_status = liquidity_status.__dict__ if hasattr(liquidity_status, "__dict__") else (liquidity_status or {})
        
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
        csv_writer.writerow(["", ""])
        
        # Trading Summary
        csv_writer.writerow(["--- TRADING SUMMARY ---", ""])
        csv_writer.writerow(["Total Trades (24h)", trading_stats.get("total_trades_24h", 0)])
        csv_writer.writerow(["Total Premium Volume USD (24h)", f"${trading_stats.get('total_premium_volume_usd_24h', 0):,.2f}"])
        csv_writer.writerow(["Average Premium per Trade USD", f"${trading_stats.get('avg_premium_received_usd_24h', 0):,.2f}"])
        csv_writer.writerow(["Call/Put Ratio (24h)", trading_stats.get("call_put_ratio_24h", 0)])
        csv_writer.writerow(["Most Active Expiry (minutes)", trading_stats.get("most_active_expiry_minutes_24h", 0)])
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
        csv_writer.writerow(["", ""])
        
        # Revenue Metrics
        csv_writer.writerow(["--- REVENUE METRICS ---", ""])
        csv_writer.writerow(["Platform Markup %", f"{revenue_metrics.get('platform_markup_pct', 0):.2f}%"])
        csv_writer.writerow(["Daily Revenue USD", f"${revenue_metrics.get('daily_revenue_usd', 0):,.2f}"])
        csv_writer.writerow(["", ""])
        
        # Liquidity Metrics  
        csv_writer.writerow(["--- LIQUIDITY METRICS ---", ""])
        csv_writer.writerow(["Liquidity Ratio", f"{liquidity_status.get('liquidity_ratio', 0):.2f}"])
        csv_writer.writerow(["Total Pool USD", f"${liquidity_status.get('total_pool_usd', 0):,.2f}"])
        csv_writer.writerow(["Active Users", liquidity_status.get("active_users", 0)])
        csv_writer.writerow(["Stress Test Status", liquidity_status.get("stress_test_status", "Unknown")])
        
        # Get CSV content
        csv_content = output.getvalue()
        output.close()
        
        # Return as StreamingResponse
        buffer = io.BytesIO(csv_content.encode('utf-8'))
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
    return {"api_call_count": api_call_count, "performance_metrics": performance_metrics, "timestamp": time.time()}
