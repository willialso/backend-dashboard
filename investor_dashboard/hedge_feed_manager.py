# investor_dashboard/hedge_feed_manager.py

import time
import random
import threading
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from backend import config
from .position_manager import PositionManager

logger = logging.getLogger(__name__)

class Exchange(Enum):
    COINBASE_PRO = config.EXCHANGE_NAME_COINBASE_PRO
    KRAKEN = config.EXCHANGE_NAME_KRAKEN
    OKX = config.EXCHANGE_NAME_OKX
    DERIBIT = config.EXCHANGE_NAME_DERIBIT
    SIMULATED = "Simulated Internal"

@dataclass
class HedgeExecution:
    hedge_id: str
    exchange: Exchange
    quantity_btc: float
    price_usd: float
    delta_impact: float
    timestamp: float

class HedgeFeedManager:
    """Monitors net delta and executes hedges when thresholds are exceeded."""

    def __init__(
        self,
        position_manager_instance: PositionManager,
        audit_engine_instance: Optional[Any] = None
    ):
        logger.info("Initializing HedgeFeedManager...")
        self.position_manager = position_manager_instance
        self.audit_engine = audit_engine_instance
        self.is_running = False
        self.recent_hedges: List[HedgeExecution] = []

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        threading.Thread(target=self._hedging_loop, daemon=True).start()
        logger.info("HedgeFeedManager: Started")

    def stop(self):
        self.is_running = False
        logger.info("HedgeFeedManager: Stopped")

    def _hedging_loop(self):
        try:
            while self.is_running:
                data = self.position_manager.get_aggregate_platform_greeks()
                net_delta = data.get("net_portfolio_delta_btc", 0.0)

                if abs(net_delta) > config.MAX_PLATFORM_NET_DELTA_BTC:
                    self._execute_hedge(net_delta)

                time.sleep(random.uniform(
                    config.HEDGE_LOOP_MIN_SLEEP_SEC,
                    config.HEDGE_LOOP_MAX_SLEEP_SEC
                ))
        except Exception as e:
            logger.error(f"HedgeFeedManager loop error: {e}", exc_info=True)
            self.is_running = False

    def _execute_hedge(self, net_delta: float):
        # CORRECT: If net_delta > 0 (long), we sell to offset. If net_delta < 0 (short), we buy to offset.
        direction = "sell" if net_delta > 0 else "buy"
        size = min(abs(net_delta), config.HEDGE_MAX_SIZE_BTC)

        exchanges = list(Exchange)
        weights = [config.HEDGE_EXCHANGE_WEIGHTS.get(ex.value, 0) for ex in exchanges]
        exch = random.choices(exchanges, weights=weights, k=1)[0]

        price = self.position_manager.current_btc_price
        slippage = random.uniform(
            config.HEDGE_SLIPPAGE_MIN_PCT,
            config.HEDGE_SLIPPAGE_MAX_PCT
        )
        exec_price = price * (1 + slippage)

        hedge_id = f"HG{int(time.time()*1000)}"

        # CRITICAL FIX: Corrected delta_impact logic
        # When buying BTC (direction=="buy"), delta_impact should be positive (+size)
        # When selling BTC (direction=="sell"), delta_impact should be negative (-size)
        hed = HedgeExecution(
            hedge_id=hedge_id,
            exchange=exch,
            quantity_btc=size,
            price_usd=round(exec_price, config.PRICE_ROUNDING_DP),
            delta_impact=round(size if direction=="buy" else -size, 4),  # ✅ FIXED: Was backwards
            timestamp=time.time()
        )

        # Add to recent hedges log
        self.recent_hedges.append(hed)
        if len(self.recent_hedges) > config.MAX_RECENT_HEDGES_LOG_SIZE:
            self.recent_hedges.pop(0)

        logger.info(f"Hedge executed {hedge_id}: {direction} {size:.4f} BTC on {exch.name} @ ${hed.price_usd:,.2f} (Δ impact: {hed.delta_impact:+.4f})")

        # CRITICAL FIX: Record hedge in position manager
        if self.position_manager and hasattr(self.position_manager, 'add_hedge_position'):
            try:
                hedge_data = {
                    "hedge_id": hedge_id,
                    "exchange": str(exch.value),  # Convert enum to string
                    "quantity_btc": size,
                    "price_usd": hed.price_usd,
                    "timestamp": hed.timestamp,
                    "instrument_type": "BTC_HEDGE",
                    "delta_impact": hed.delta_impact  # Use the corrected delta_impact
                }
                self.position_manager.add_hedge_position(hedge_data)
                logger.info(f"HFM: ✅ Recorded hedge {hedge_id} in position manager (Δ: {hed.delta_impact:+.4f} BTC)")
            except Exception as e:
                logger.error(f"HFM: ❌ Failed to record hedge in position manager: {e}", exc_info=True)
        else:
            logger.error("HFM: ❌ Cannot record hedge - position manager missing add_hedge_position method")

        # Record in audit engine
        if self.audit_engine:
            try:
                self.audit_engine.record_hedge(hed)
                logger.debug(f"HFM: ✅ Recorded hedge {hedge_id} in audit engine")
            except Exception as e:
                logger.warning(f"HFM: ⚠️ Audit engine record_hedge failed: {e}")

    def get_recent_hedges(self, limit: int = 10) -> List[HedgeExecution]:
        return sorted(self.recent_hedges, key=lambda h: h.timestamp, reverse=True)[:limit]

    def get_hedge_metrics(self) -> Dict[str, Any]:
        now = time.time()
        cutoff = now - 24*3600

        hedges_24h = [h for h in self.recent_hedges if h.timestamp >= cutoff]
        total_volume = sum(h.quantity_btc for h in hedges_24h)
        avg_exec_time = (
            sum(config.HEDGE_EXECUTION_TIMES_MS.get(h.exchange.value, 0) for h in hedges_24h)
            / max(len(hedges_24h), 1)
        )

        return {
            "hedges_24h": len(hedges_24h),
            "total_volume_hedged_btc_24h": round(total_volume, 4),
            "avg_execution_time_ms": round(avg_exec_time, 2),
            "timestamp": now
        }

    def get_hedge_status_summary(self) -> Dict[str, Any]:
        """Get summary of hedge manager status for debugging"""
        return {
            "is_running": self.is_running,
            "total_hedges_in_log": len(self.recent_hedges),
            "position_manager_connected": self.position_manager is not None,
            "position_manager_has_add_method": (
                self.position_manager is not None and 
                hasattr(self.position_manager, 'add_hedge_position')
            ),
            "audit_engine_connected": self.audit_engine is not None,
            "current_btc_price": getattr(self.position_manager, 'current_btc_price', 0.0),
            "last_hedge_delta_impact": self.recent_hedges[-1].delta_impact if self.recent_hedges else 0.0,
            "timestamp": time.time()
        }

    def force_hedge_execution_test(self) -> Dict[str, Any]:
        """Force execute a test hedge for debugging - CAUTION: Only for testing"""
        if not self.position_manager:
            return {"error": "No position manager available"}
            
        current_greeks = self.position_manager.get_aggregate_platform_greeks()
        current_delta = current_greeks.get("net_portfolio_delta_btc", 0.0)
        
        logger.warning(f"HFM: FORCING TEST HEDGE EXECUTION - Current delta: {current_delta:.4f}")
        
        # Force execute hedge regardless of threshold
        self._execute_hedge(current_delta)
        
        # Get updated greeks after hedge
        updated_greeks = self.position_manager.get_aggregate_platform_greeks()
        
        return {
            "test_executed": True,
            "delta_before": current_delta,
            "delta_after": updated_greeks.get("net_portfolio_delta_btc", 0.0),
            "hedges_count_before": current_greeks.get("open_hedges_count", 0),
            "hedges_count_after": updated_greeks.get("open_hedges_count", 0),
            "last_hedge_delta_impact": self.recent_hedges[-1].delta_impact if self.recent_hedges else 0.0,
            "timestamp": time.time()
        }

    def clear_all_hedges(self) -> Dict[str, Any]:
        """Clear all hedge history - for testing/debugging only"""
        logger.warning("HFM: CLEARING ALL HEDGE HISTORY")
        hedge_count = len(self.recent_hedges)
        self.recent_hedges.clear()
        
        # Also clear position manager hedge positions if method exists
        if self.position_manager and hasattr(self.position_manager, 'open_hedge_positions'):
            position_count = len(self.position_manager.open_hedge_positions)
            self.position_manager.open_hedge_positions.clear()
            logger.warning(f"HFM: Cleared {position_count} hedge positions from position manager")
        
        return {
            "cleared_hedge_history": hedge_count,
            "timestamp": time.time()
        }
