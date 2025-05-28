# investor_dashboard/dashboard_api.py

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel

# Import your dashboard managers
from .revenue_engine import RevenueEngine
from .liquidity_manager import LiquidityManager
from .bot_trader_simulator import BotTraderSimulator
from .audit_engine import AuditEngine
from .hedge_feed_manager import HedgeFeedManager

router = APIRouter()

# Global instances (will be initialized by main_dashboard.py)
revenue_engine = None
liquidity_manager = None
bot_trader_simulator = None
audit_engine = None
hedge_feed_manager = None

# Pydantic models for API requests
class LiquidityAllocation(BaseModel):
    liquidity_pct: float
    operations_pct: float

class TraderDistribution(BaseModel):
    advanced_pct: float
    intermediate_pct: float
    beginner_pct: float

class UserGrowthSimulation(BaseModel):
    new_user_count: int

@router.get("/revenue-metrics")
async def get_revenue_metrics():
    """Get current revenue metrics."""
    if not revenue_engine:
        raise HTTPException(status_code=503, detail="Revenue engine not initialized")
    
    metrics = revenue_engine.get_current_metrics()
    return {
        "bsm_fair_value": metrics.bsm_fair_value,
        "platform_price": metrics.platform_price,
        "markup_percentage": metrics.markup_percentage,
        "revenue_per_contract": metrics.revenue_per_contract,
        "daily_revenue_estimate": metrics.daily_revenue_estimate,
        "contracts_sold_24h": metrics.contracts_sold_24h,
        "average_markup": metrics.average_markup
    }

@router.get("/liquidity-status")
async def get_liquidity_status():
    """Get current liquidity pool status."""
    if not liquidity_manager:
        raise HTTPException(status_code=503, detail="Liquidity manager not initialized")
    
    status = liquidity_manager.get_status()
    return {
        "total_pool_usd": status.total_pool_usd,
        "allocated_to_liquidity_pct": status.allocated_to_liquidity_pct,
        "allocated_to_operations_pct": status.allocated_to_operations_pct,
        "allocated_to_profit_pct": status.allocated_to_profit_pct,
        "active_users": status.active_users,
        "required_liquidity_usd": status.required_liquidity_usd,
        "liquidity_ratio": status.liquidity_ratio,
        "stress_test_buffer_usd": status.stress_test_buffer_usd
    }

@router.post("/liquidity-allocation")
async def adjust_liquidity_allocation(allocation: LiquidityAllocation):
    """Adjust liquidity allocation percentages."""
    if not liquidity_manager:
        raise HTTPException(status_code=503, detail="Liquidity manager not initialized")
    
    try:
        liquidity_manager.adjust_allocation(
            allocation.liquidity_pct,
            allocation.operations_pct
        )
        return {"status": "success", "message": "Liquidity allocation updated"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/bot-trader-activity")
async def get_bot_trader_activity():
    """Get current bot trader activity."""
    if not bot_trader_simulator:
        raise HTTPException(status_code=503, detail="Bot trader simulator not initialized")
    
    activities = bot_trader_simulator.get_current_activity()
    return {
        "activities": [
            {
                "trader_type": activity.trader_type.value,
                "active_count": activity.active_count,
                "percentage": activity.percentage,
                "avg_trade_size_usd": activity.avg_trade_size_usd,
                "trades_per_hour": activity.trades_per_hour,
                "success_rate": activity.success_rate
            }
            for activity in activities
        ]
    }

@router.get("/recent-trades")
async def get_recent_trades(limit: int = 10):
    """Get recent bot trades for live feed."""
    if not bot_trader_simulator:
        raise HTTPException(status_code=503, detail="Bot trader simulator not initialized")
    
    trades = bot_trader_simulator.get_recent_trades(limit)
    return {
        "trades": [
            {
                "trader_type": trade.trader_type.value,
                "option_type": trade.option_type,
                "strike": trade.strike,
                "expiry_minutes": trade.expiry_minutes,
                "premium_paid": trade.premium_paid,
                "timestamp": trade.timestamp,
                "trader_id": trade.trader_id
            }
            for trade in trades
        ]
    }

@router.post("/trader-distribution")
async def adjust_trader_distribution(distribution: TraderDistribution):
    """Adjust bot trader type distribution."""
    if not bot_trader_simulator:
        raise HTTPException(status_code=503, detail="Bot trader simulator not initialized")
    
    if (distribution.advanced_pct + distribution.intermediate_pct + distribution.beginner_pct) != 100:
        raise HTTPException(status_code=400, detail="Percentages must sum to 100")
    
    bot_trader_simulator.adjust_trader_distribution(
        distribution.advanced_pct,
        distribution.intermediate_pct,
        distribution.beginner_pct
    )
    return {"status": "success", "message": "Trader distribution updated"}

@router.get("/hedge-execution-feed")
async def get_hedge_execution_feed(limit: int = 10):
    """Get recent hedge executions."""
    if not hedge_feed_manager:
        raise HTTPException(status_code=503, detail="Hedge feed manager not initialized")
    
    hedges = hedge_feed_manager.get_recent_hedges(limit)
    return {
        "hedges": [
            {
                "timestamp": hedge.timestamp,
                "hedge_type": hedge.hedge_type.value,
                "side": hedge.side,
                "quantity_btc": hedge.quantity_btc,
                "price_usd": hedge.price_usd,
                "exchange": hedge.exchange.value,
                "cost_usd": hedge.cost_usd,
                "reasoning": hedge.reasoning,
                "execution_time_ms": hedge.execution_time_ms
            }
            for hedge in hedges
        ]
    }

@router.get("/hedge-metrics")
async def get_hedge_metrics():
    """Get hedge execution metrics."""
    if not hedge_feed_manager:
        raise HTTPException(status_code=503, detail="Hedge feed manager not initialized")
    
    return hedge_feed_manager.get_hedge_metrics()

@router.get("/audit-summary")
async def get_audit_summary():
    """Get 24-hour audit metrics."""
    if not audit_engine:
        raise HTTPException(status_code=503, detail="Audit engine not initialized")
    
    metrics = audit_engine.get_24h_metrics()
    return {
        "revenue_24h": metrics.revenue_24h,
        "trades_24h": metrics.trades_24h,
        "total_pnl_24h": metrics.total_pnl_24h,
        "hedge_efficiency_pct": metrics.hedge_efficiency_pct,
        "platform_uptime_pct": metrics.platform_uptime_pct,
        "compliance_status": metrics.compliance_status,
        "avg_response_time_ms": metrics.avg_response_time_ms,
        "error_rate_pct": metrics.error_rate_pct
    }

@router.get("/compliance-checks")
async def get_compliance_checks():
    """Get detailed compliance checks."""
    if not audit_engine:
        raise HTTPException(status_code=503, detail="Audit engine not initialized")
    
    checks = audit_engine.run_compliance_checks()
    return {
        "checks": [
            {
                "check_name": check.check_name,
                "status": check.status,
                "value": check.value,
                "threshold": check.threshold,
                "last_check": check.last_check
            }
            for check in checks
        ]
    }

@router.get("/platform-health")
async def get_platform_health():
    """Get overall platform health summary."""
    try:
        # Aggregate data from all components
        revenue_metrics = revenue_engine.get_current_metrics() if revenue_engine else None
        liquidity_status = liquidity_manager.get_status() if liquidity_manager else None
        audit_metrics = audit_engine.get_24h_metrics() if audit_engine else None
        
        health_score = 100  # Start with perfect score
        
        # Deduct points for issues
        if audit_metrics:
            if audit_metrics.compliance_status == "NON_COMPLIANT":
                health_score -= 20
            elif audit_metrics.compliance_status == "WARNING":
                health_score -= 5
                
            if audit_metrics.platform_uptime_pct < 99.5:
                health_score -= 10
                
            if audit_metrics.error_rate_pct > 1.0:
                health_score -= 5
        
        if liquidity_status and liquidity_status.liquidity_ratio < 1.2:
            health_score -= 15
        
        status = "EXCELLENT" if health_score >= 95 else \
                "GOOD" if health_score >= 85 else \
                "FAIR" if health_score >= 70 else "POOR"
        
        return {
            "overall_health": status,
            "health_score": health_score,
            "revenue_operational": revenue_engine is not None,
            "liquidity_operational": liquidity_manager is not None,
            "audit_operational": audit_engine is not None,
            "hedge_feed_operational": hedge_feed_manager is not None,
            "bot_simulation_operational": bot_trader_simulator is not None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/simulate-user-growth")
async def simulate_user_growth(growth: UserGrowthSimulation):
    """Simulate platform scaling with user growth."""
    if not liquidity_manager:
        raise HTTPException(status_code=503, detail="Liquidity manager not initialized")
    
    liquidity_manager.simulate_user_growth(growth.new_user_count)
    return {"status": "success", "message": f"Simulated growth to {growth.new_user_count} users"}
