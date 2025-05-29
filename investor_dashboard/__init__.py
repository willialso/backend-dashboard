# investor_dashboard/__init__.py

"""
Investor Dashboard Package for Atticus Platform

This package contains all investor-facing dashboard components including
revenue tracking, liquidity management, position management, and trading simulation.
"""

__version__ = "1.0.0"
__author__ = "Atticus Team"

# Import the main classes from this package
from .audit_engine import AuditEngine
from .revenue_engine import RevenueEngine, RevenueMetrics
from .position_manager import PositionManager
from .liquidity_manager import LiquidityManager, LiquidityStatus
from .hedge_feed_manager import HedgeFeedManager
from .bot_trader_simulator import BotTraderSimulator, EnrichedTradeData, TraderType  # FIXED: Changed TraderActivity to TraderType

# Define the public API
__all__ = [
    "AuditEngine",
    "RevenueEngine", 
    "RevenueMetrics",
    "PositionManager",
    "LiquidityManager",
    "LiquidityStatus", 
    "HedgeFeedManager",
    "BotTraderSimulator",
    "EnrichedTradeData",
    "TraderType"  # FIXED: Changed from TraderActivity to TraderType
]
