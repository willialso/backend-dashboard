# investor_dashboard/__init__.py

"""
Atticus Investor Dashboard Backend

This module provides investor-focused analytics and controls for the BTC options platform.
"""

__version__ = "1.0.0"
__author__ = "Atticus Team"

from .revenue_engine import RevenueEngine, RevenueMetrics
from .liquidity_manager import LiquidityManager, LiquidityStatus
from .bot_trader_simulator import BotTraderSimulator, BotTrade, TraderActivity
from .audit_engine import AuditEngine, AuditMetrics
from .hedge_feed_manager import HedgeFeedManager, HedgeExecution
from .dashboard_api import router as dashboard_router

__all__ = [
    "RevenueEngine",
    "RevenueMetrics", 
    "LiquidityManager",
    "LiquidityStatus",
    "BotTraderSimulator",
    "BotTrade",
    "TraderActivity",
    "AuditEngine",
    "AuditMetrics",
    "HedgeFeedManager", 
    "HedgeExecution",
    "dashboard_router"
]
