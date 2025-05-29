# investor_dashboard/hedge_feed_manager.py

import time
import random
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import logging
from collections import defaultdict

# from .position_manager import PositionManager # For type hinting
# from .audit_engine import AuditEngine     # For type hinting
from backend import config

logger = logging.getLogger(__name__)

class HedgeType(Enum): # Using your config.py version would be better if it defines these
    DELTA_HEDGE = "Delta Hedge"; RL_HEDGE = "RL Hedge"; PORTFOLIO_REBALANCE = "Portfolio Rebalance"
    GAMMA_HEDGE = "Gamma Hedge"; VEGA_HEDGE = "Vega Hedge"; LIQUIDITY_PROVISION = "Liquidity Provision Hedge"

class Exchange(Enum):
    COINBASE_PRO = config.get_config_value("EXCHANGE_NAME_COINBASE_PRO", "Coinbase Pro")
    KRAKEN = config.get_config_value("EXCHANGE_NAME_KRAKEN", "Kraken")
    OKX = config.get_config_value("EXCHANGE_NAME_OKX", "OKX")
    DERIBIT = config.get_config_value("EXCHANGE_NAME_DERIBIT", "Deribit")
    # SIMULATED_INTERNAL = "Simulated Internal" # Not in your example config

@dataclass
class EnrichedHedgeExecution:
    timestamp: float; hedge_id: str; hedge_type: HedgeType; side: str 
    instrument: str; quantity: float; price_usd: float; exchange: Exchange
    cost_usd: float; reasoning: str; execution_time_ms: float
    delta_before_hedge_btc: Optional[float] = None
    delta_impact_of_hedge_btc: Optional[float] = None
    delta_after_hedge_btc: Optional[float] = None
    estimated_pnl_usd: Optional[float] = field(default=0.0) # PNL of THIS hedge transaction/position

class HedgeFeedManager:
    def __init__(self,
                 position_manager_instance: Optional[Any] = None, # PositionManager
                 audit_engine_instance: Optional[Any] = None):    # AuditEngine
        logger.info("🔧 Initializing HedgeFeedManager...")
        self.recent_hedges_log: List[EnrichedHedgeExecution] = []
        self.is_running = False
        self.current_btc_price: float = 0.0
        self.hedge_id_counter = int(time.time())
        
        self.position_manager = position_manager_instance
        self.audit_engine = audit_engine_instance

        self.exchange_weights = { # From your config.py structure for weights
            Exchange.COINBASE_PRO: config.HEDGE_EXCHANGE_WEIGHT_CBPRO if hasattr(config, 'HEDGE_EXCHANGE_WEIGHT_CBPRO') else 0.4,
            Exchange.KRAKEN: config.HEDGE_EXCHANGE_WEIGHT_KRAKEN if hasattr(config, 'HEDGE_EXCHANGE_WEIGHT_KRAKEN') else 0.25,
            Exchange.OKX: config.HEDGE_EXCHANGE_WEIGHT_OKX if hasattr(config, 'HEDGE_EXCHANGE_WEIGHT_OKX') else 0.20,
            Exchange.DERIBIT: config.HEDGE_EXCHANGE_WEIGHT_DERIBIT if hasattr(config, 'HEDGE_EXCHANGE_WEIGHT_DERIBIT') else 0.15
        }
        self._normalize_exchange_weights()
        logger.info("✅ HedgeFeedManager initialized.")

    def _normalize_exchange_weights(self):
        total_weight = sum(self.exchange_weights.values())
        if total_weight == 0: return
        for exch in self.exchange_weights: self.exchange_weights[exch] /= total_weight
            
    def start(self):
        if not self.position_manager:
             logger.warning("⚠️ HFM: Started WITHOUT PositionManager. Delta hedging decisions will be impaired.")
        self.is_running = True
        threading.Thread(target=self._hedge_simulation_loop, daemon=True).start()
        logger.info("🛡️ HedgeFeedManager simulation loop started.")

    def _hedge_simulation_loop(self):
        while self.is_running:
            try:
                if self._should_generate_hedge():
                    self._generate_and_record_hedge()
            except Exception as e: logger.error(f"HFM loop error: {e}", exc_info=True)
            time.sleep(random.uniform(config.HEDGE_LOOP_MIN_SLEEP_SEC, config.HEDGE_LOOP_MAX_SLEEP_SEC))

    def _should_generate_hedge(self) -> bool:
        if not self.position_manager or self.current_btc_price <= 0:
            return random.random() < 0.005 # Very low chance if no data

        greeks = self.position_manager.get_aggregate_platform_greeks()
        net_delta_btc = greeks.get("net_portfolio_delta_btc", 0.0)
        threshold = self.position_manager.delta_hedge_threshold # PM gets from config
        
        if abs(net_delta_btc) > threshold:
            logger.info(f"💡 HFM Trigger: Net Delta {net_delta_btc:.2f} BTC > Threshold +/-{threshold:.2f} BTC.")
            return True
        return random.random() < config.HEDGE_PROBABILISTIC_TRIGGER_PCT

    def _generate_and_record_hedge(self) -> Optional[EnrichedHedgeExecution]:
        if self.current_btc_price <= 0: return None
        
        self.hedge_id_counter += 1
        hedge_id = f"H{self.hedge_id_counter}"

        delta_before_btc = 0.0
        quantity_to_hedge_btc_signed = 0.0
        hedge_type = HedgeType.DELTA_HEDGE # Default

        if self.position_manager:
            greeks = self.position_manager.get_aggregate_platform_greeks()
            delta_before_btc = greeks.get("net_portfolio_delta_btc", 0.0)
            threshold = self.position_manager.delta_hedge_threshold

            if abs(delta_before_btc) > threshold:
                proportion = random.uniform(config.HEDGE_DELTA_PROPORTION_MIN, config.HEDGE_DELTA_PROPORTION_MAX)
                quantity_to_hedge_btc_signed = round(-delta_before_btc * proportion, config.HEDGE_QUANTITY_ROUNDING_DP)
                quantity_to_hedge_btc_signed = max(min(quantity_to_hedge_btc_signed, config.HEDGE_MAX_SIZE_BTC), -config.HEDGE_MAX_SIZE_BTC)
                if abs(quantity_to_hedge_btc_signed) < config.HEDGE_MIN_SIZE_BTC: quantity_to_hedge_btc_signed = 0.0
            else: # Probabilistic non-delta hedge
                if random.random() < 0.3: # Chance for other types if delta is fine
                    hedge_type = random.choice([HedgeType.GAMMA_HEDGE, HedgeType.VEGA_HEDGE, HedgeType.PORTFOLIO_REBALANCE])
                    quantity_to_hedge_btc_signed = round(random.uniform(config.HEDGE_MIN_SIZE_BTC, config.HEDGE_NON_DELTA_MAX_SIZE_BTC) * random.choice([-1,1]), config.HEDGE_QUANTITY_ROUNDING_DP)
                else: return None # No hedge this cycle
        else: # No PM, random hedge
            quantity_to_hedge_btc_signed = round(random.uniform(config.HEDGE_MIN_SIZE_BTC, config.HEDGE_NON_DELTA_MAX_SIZE_BTC) * random.choice([-1,1]), config.HEDGE_QUANTITY_ROUNDING_DP)
            hedge_type = random.choice(list(HedgeType))

        if abs(quantity_to_hedge_btc_signed) < config.HEDGE_MIN_SIZE_BTC: return None

        side = "Buy" if quantity_to_hedge_btc_signed > 0 else "Sell"
        quantity_abs_btc = abs(quantity_to_hedge_btc_signed)
        delta_impact_of_hedge_btc = quantity_to_hedge_btc_signed
        delta_after_btc = delta_before_btc + delta_impact_of_hedge_btc
        
        exchange = random.choices(list(self.exchange_weights.keys()), weights=list(self.exchange_weights.values()))[0]
        instrument = config.HEDGE_INSTRUMENT_SPOT if exchange != Exchange.DERIBIT else config.HEDGE_INSTRUMENT_PERP
        
        # Slippage applied to make buy price higher, sell price lower
        slippage_factor = 1 + (config.HEDGE_SLIPPAGE_MAX_PCT if side == "Buy" else -config.HEDGE_SLIPPAGE_MAX_PCT) 
        execution_price = round(self.current_btc_price * slippage_factor, config.PRICE_ROUNDING_DP)
        cost_of_trade_usd = quantity_abs_btc * execution_price
        
        exec_times = config.HEDGE_EXECUTION_TIMES_MS
        execution_time_ms = round(random.uniform(exec_times.get(exchange, 200)*0.8, exec_times.get(exchange, 200)*1.2), 1)
        reasoning_str = self._generate_hedge_reasoning(hedge_type, side, quantity_abs_btc, delta_before_btc, delta_after_btc)

        hedge = EnrichedHedgeExecution(
            timestamp=time.time(), hedge_id=hedge_id, hedge_type=hedge_type, side=side,
            instrument=instrument, quantity=quantity_abs_btc, price_usd=execution_price,
            exchange=exchange, cost_usd=cost_of_trade_usd, reasoning=reasoning_str, 
            execution_time_ms=execution_time_ms,
            delta_before_hedge_btc=round(delta_before_btc, 4),
            delta_impact_of_hedge_btc=round(delta_impact_of_hedge_btc, 4),
            delta_after_hedge_btc=round(delta_after_btc, 4),
            estimated_pnl_usd=0.0 # Initial PNL; MTM handled by PM for open hedges
        )
        self.recent_hedges_log.append(hedge)
        if len(self.recent_hedges_log) > config.MAX_RECENT_HEDGES_LOG_SIZE:
            self.recent_hedges_log = self.recent_hedges_log[-config.MAX_RECENT_HEDGES_LOG_SIZE:]

        if self.position_manager:
            pm_hedge_data = {
                "hedge_id": hedge.hedge_id, "instrument_type": hedge.instrument,
                "quantity_btc_signed": quantity_to_hedge_btc_signed,
                "entry_price_usd": hedge.price_usd, "timestamp": hedge.timestamp
            }
            self.position_manager.add_hedge_position(pm_hedge_data)

        if self.audit_engine:
            transaction_fee = cost_of_trade_usd * config.HEDGE_TRANSACTION_FEE_PCT
            self.audit_engine.track_hedging_activity(pnl_usd=-transaction_fee, cost_usd=transaction_fee, hedge_id=hedge_id) # Initial impact is cost
        
        logger.info(f"🛡️ HFM Hedge: {hedge_id} {side} {quantity_abs_btc:.2f} {instrument} on {exchange.value} @ ${execution_price:,.2f}. Delta {delta_before_btc:.2f} -> {delta_after_btc:.2f}")
        return hedge

    def _generate_hedge_reasoning(self, ht: HedgeType, s: str, q: float, db: float, da: float) -> str:
        return f"{ht.value}: {s} {q:.2f} BTC. Delta {db:.2f} -> {da:.2f}."

    def update_btc_price(self, price: float):
        if price > 0: self.current_btc_price = price
        # PositionManager is responsible for MTM updates of its open hedges

    def get_recent_hedges(self, limit: int = 10) -> List[EnrichedHedgeExecution]:
        # If PM is source of truth for MTM PNL of open hedges, could enrich here before returning
        # For now, returns from local log. PM will have the MTM PNL.
        return sorted(self.recent_hedges_log, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_hedge_metrics(self) -> Dict: # From your Attachment [4], adapted
        cutoff_24h = time.time() - (24 * 60 * 60)
        hedges_24h = [h for h in self.recent_hedges_log if h.timestamp >= cutoff_24h]

        if not hedges_24h:
            return {"hedges_24h": 0, "total_volume_hedged_btc_24h":0, "total_gross_value_usd_24h": 0,
                    "avg_execution_time_ms": 0, "exchange_distribution": {}, "hedge_type_distribution": {},
                    "net_hedging_pnl_24h_usd": 0.0, "data_source": "hfm_log_no_hedges_24h"}

        total_abs_volume_btc = sum(h.quantity for h in hedges_24h) # Absolute BTC volume transacted
        total_gross_value = sum(h.cost_usd for h in hedges_24h) # Gross USD value of transactions
        avg_exec_time = sum(h.execution_time_ms for h in hedges_24h) / len(hedges_24h)
        
        net_hedging_pnl = 0.0
        if self.audit_engine: # Get from AuditEngine for consistency
            metrics_24h = self.audit_engine.get_24h_metrics()
            net_hedging_pnl = metrics_24h.net_hedging_pnl_24h_usd
        else: # Placeholder if no AuditEngine
            logger.warning("HFM: AuditEngine not available for net_hedging_pnl_24h_usd in metrics.")

        exchange_dist = defaultdict(int); type_dist = defaultdict(int)
        for h in hedges_24h:
            exchange_dist[h.exchange.value] += 1; type_dist[h.hedge_type.value] += 1
        
        return {
            "hedges_executed_24h": len(hedges_24h), # Renamed for clarity
            "total_btc_volume_hedged_24h": round(total_abs_volume_btc, 2),
            "total_gross_value_transacted_usd_24h": round(total_gross_value, 2),
            "avg_execution_time_ms": round(avg_exec_time, 1),
            "exchange_distribution_count": dict(exchange_dist),
            "hedge_type_distribution_count": dict(type_dist),
            "net_hedging_pnl_24h_usd": round(net_hedging_pnl, 2), # From AuditEngine
            "data_source": "hfm_log_and_audit_engine"
        }

    def stop(self):
        self.is_running = False; logger.info("🛡️ HedgeFeedManager simulation loop stopped.")

