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
    KRAKEN       = config.EXCHANGE_NAME_KRAKEN
    OKX          = config.EXCHANGE_NAME_OKX
    DERIBIT      = config.EXCHANGE_NAME_DERIBIT
    SIMULATED    = "Simulated Internal"

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
        hed = HedgeExecution(
            hedge_id=hedge_id,
            exchange=exch,
            quantity_btc=size,
            price_usd=round(exec_price, config.PRICE_ROUNDING_DP),
            delta_impact=round(-size if direction=="buy" else size, 4),
            timestamp=time.time()
        )
        self.recent_hedges.append(hed)
        if len(self.recent_hedges) > config.MAX_RECENT_HEDGES_LOG_SIZE:
            self.recent_hedges.pop(0)
        logger.info(f"Hedge executed {hedge_id}: {direction} {size} BTC on {exch.name} @ {hed.price_usd}")
        if self.audit_engine:
            try:
                self.audit_engine.record_hedge(hed)
            except Exception as e:
                logger.warning(f"Audit engine record_hedge failed: {e}")

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
