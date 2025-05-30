# investor_dashboard/__init__.py

"""
Investor Dashboard Module

Provides comprehensive trading, risk management, and liquidity tracking
for the Atticus Options Trading Platform.
"""

from .revenue_engine import RevenueEngine
from .position_manager import PositionManager, EnrichedTradeDataInPM, EnrichedHedgePositionInPM
from .bot_trader_simulator import BotTraderSimulator, TraderType
from .hedge_feed_manager import HedgeFeedManager, Exchange, HedgeExecution
from .audit_engine import AuditEngine, AuditMetrics, ComplianceCheck
from .liquidity_manager import LiquidityManager, LiquidityAllocation  # FIXED: Removed LiquidityStatus

__all__ = [
    # Core engines
    "RevenueEngine",
    "PositionManager", 
    "LiquidityManager",
    "BotTraderSimulator",
    "HedgeFeedManager",
    "AuditEngine",
    
    # Data classes
    "EnrichedTradeDataInPM",
    "EnrichedHedgePositionInPM", 
    "LiquidityAllocation",  # FIXED: Updated to correct class name
    "AuditMetrics",
    "ComplianceCheck",
    "HedgeExecution",
    
    # Enums
    "TraderType",
    "Exchange"
]

__version__ = "2.3.1"
