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
    # REMOVED: DERIBIT = config.EXCHANGE_NAME_DERIBIT  # User prefers Coinbase Pro/Kraken
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
        audit_engine_instance: Optional[Any] = None,
        liquidity_manager_instance: Optional[Any] = None
    ):
        logger.info("Initializing HedgeFeedManager...")
        
        self.position_manager = position_manager_instance
        self.audit_engine = audit_engine_instance
        self.liquidity_manager = liquidity_manager_instance
        self.is_running = False
        self.recent_hedges: List[HedgeExecution] = []

        # CRITICAL FIX: Initialize all hedge tracking attributes that dashboard_api.py expects
        self.total_hedges_executed = 0
        self.total_hedge_volume_btc = 0.0
        self.total_hedge_volume_usd = 0.0
        self.hedge_pnl_accumulator = 0.0
        self.daily_hedge_volume = 0.0
        self.total_hedge_value_usd_24h = 0.0
        
        # CRITICAL FIX: Initialize hedge tracking dictionaries
        self.hedge_metrics_24h = {}
        self.hedge_history = []
        self.daily_hedge_data = {}
        
        # CRITICAL FIX: Add start time for 24h calculations
        self.start_time = time.time()

        # CRITICAL: Liquidity-based hedge limits
        self.min_liquidity_buffer_pct = 0.20  # Keep 20% buffer
        self.max_liquidity_per_hedge_pct = 0.05  # REDUCED: Max 5% of available per hedge
        
        # CRITICAL FIX: Hedge sizing parameters
        self.hedge_coverage_ratio = 0.8  # Hedge 80% of delta exposure
        self.min_hedge_size_btc = 0.005  # Minimum 0.005 BTC ($500 at $100K)
        self.max_single_hedge_btc = 0.1  # REDUCED: Maximum 0.1 BTC per hedge

    def reset_hedge_metrics(self):
        """CRITICAL FIX: Reset all hedge metrics for complete system reset"""
        try:
            logger.warning("HFM: Resetting all hedge metrics to zero")
            
            # Clear hedge history
            self.recent_hedges.clear()
            logger.info("✅ HFM: Cleared recent_hedges")
            
            # CRITICAL: Reset ALL hedge volume tracking attributes
            self.total_hedges_executed = 0
            self.total_hedge_volume_btc = 0.0
            self.total_hedge_volume_usd = 0.0
            self.hedge_pnl_accumulator = 0.0
            self.daily_hedge_volume = 0.0
            self.total_hedge_value_usd_24h = 0.0
            
            # Clear tracking dictionaries
            self.hedge_metrics_24h.clear()
            self.hedge_history.clear()
            self.daily_hedge_data.clear()
            
            # Reset start time for fresh 24h window
            self.start_time = time.time()
            
            logger.info("✅ HFM: All hedge metrics reset to zero")
            return {"status": "success", "message": "All hedge metrics reset to zero"}
            
        except Exception as e:
            logger.error(f"❌ HFM: Error resetting hedge metrics: {e}")
            return {"status": "error", "message": str(e)}

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

                # ENHANCED LOGGING: Debug hedge trigger logic
                threshold = config.MAX_PLATFORM_NET_DELTA_BTC
                logger.debug(f"HFM: Delta check - Current: {net_delta:.4f} BTC, Threshold: {threshold:.4f} BTC")

                if abs(net_delta) > threshold:
                    logger.warning(f"HFM: HEDGE TRIGGERED - Delta {net_delta:.4f} BTC exceeds threshold {threshold:.4f} BTC")
                    self._execute_hedge(net_delta)
                else:
                    logger.debug(f"HFM: No hedge needed - Delta {net_delta:.4f} within threshold ±{threshold:.4f}")

                time.sleep(random.uniform(
                    config.HEDGE_LOOP_MIN_SLEEP_SEC,
                    config.HEDGE_LOOP_MAX_SLEEP_SEC
                ))

        except Exception as e:
            logger.error(f"HedgeFeedManager loop error: {e}", exc_info=True)
            self.is_running = False

    def _get_available_liquidity_btc(self) -> float:
        """CRITICAL: Get available liquidity for hedging in BTC terms"""
        try:
            if not self.liquidity_manager:
                # Fallback: Use static calculation if no liquidity manager
                default_pool_usd = 1500000.0  # $1.5M default
                available_usd = default_pool_usd * 0.75  # 75% available for trading
                available_btc = available_usd / self.position_manager.current_btc_price
                logger.warning(f"HFM: Using fallback liquidity calculation: {available_btc:.4f} BTC")
                return available_btc

            # Get current liquidity allocation
            liquidity_data = self.liquidity_manager.get_current_allocation()
            total_pool_usd = liquidity_data.get("total_pool_usd", 1500000.0)
            liquidity_ratio = liquidity_data.get("liquidity_ratio", 1.923)

            # Calculate available for hedging
            if liquidity_ratio > 1.0:
                utilized_usd = total_pool_usd * (liquidity_ratio - 1.0)
                available_usd = total_pool_usd - utilized_usd
            else:
                available_usd = total_pool_usd * 0.75  # Default 75% available

            # Apply safety buffer
            safe_available_usd = available_usd * (1.0 - self.min_liquidity_buffer_pct)

            # Convert to BTC
            if self.position_manager.current_btc_price > 0:
                available_btc = safe_available_usd / self.position_manager.current_btc_price
            else:
                available_btc = 0.0

            logger.debug(f"HFM: Available liquidity: ${safe_available_usd:,.0f} = {available_btc:.4f} BTC")
            return available_btc

        except Exception as e:
            logger.error(f"HFM: Error calculating available liquidity: {e}")
            # Emergency fallback
            return 10.0  # 10 BTC fallback limit

    def _execute_hedge(self, net_delta: float):
        """CRITICAL FIX: Execute hedge with PROPORTIONAL sizing based on actual delta exposure"""
        # CORRECT: If net_delta > 0 (long), we sell to offset. If net_delta < 0 (short), we buy to offset.
        direction = "sell" if net_delta > 0 else "buy"

        # CRITICAL FIX: Proportional hedge sizing based on actual delta exposure
        abs_delta = abs(net_delta)
        
        # Calculate proportional hedge size (hedge a percentage of the delta exposure)
        proportional_hedge_size = abs_delta * self.hedge_coverage_ratio
        
        logger.info(f"HFM: PROPORTIONAL SIZING - Delta: {abs_delta:.4f} BTC, Coverage: {self.hedge_coverage_ratio:.1%}, Target: {proportional_hedge_size:.4f} BTC")

        # CRITICAL: Check minimum hedge size (don't hedge tiny amounts)
        if proportional_hedge_size < self.min_hedge_size_btc:
            logger.info(f"HFM: Hedge size {proportional_hedge_size:.4f} BTC < minimum {self.min_hedge_size_btc:.4f} BTC, skipping execution")
            return

        # Apply maximum single hedge limit
        capped_hedge_size = min(proportional_hedge_size, self.max_single_hedge_btc)
        
        if capped_hedge_size < proportional_hedge_size:
            logger.warning(f"HFM: Hedge size capped: {proportional_hedge_size:.4f} → {capped_hedge_size:.4f} BTC")

        # CRITICAL: Get available liquidity for hedging
        available_liquidity_btc = self._get_available_liquidity_btc()
        max_hedge_per_execution = available_liquidity_btc * self.max_liquidity_per_hedge_pct

        # Apply liquidity constraints
        liquidity_limited_size = min(capped_hedge_size, max_hedge_per_execution)
        
        if liquidity_limited_size < capped_hedge_size:
            logger.warning(f"HFM: Hedge size limited by liquidity: {capped_hedge_size:.4f} → {liquidity_limited_size:.4f} BTC")

        # Final hedge size
        final_size = liquidity_limited_size

        # Log liquidity constraints if applied
        if available_liquidity_btc < self.max_single_hedge_btc:
            logger.warning(f"HFM: Low liquidity for hedging: {available_liquidity_btc:.4f} BTC available")

        # FINAL CHECK: Ensure hedge size makes economic sense
        if final_size < self.min_hedge_size_btc:
            logger.info(f"HFM: Final hedge size {final_size:.4f} BTC < minimum {self.min_hedge_size_btc:.4f} BTC, skipping execution")
            return

        # FIXED: Use only Coinbase Pro and Kraken (removed Deribit per user request)
        exchanges = [Exchange.COINBASE_PRO, Exchange.KRAKEN]
        weights = [
            config.HEDGE_EXCHANGE_WEIGHTS.get(Exchange.COINBASE_PRO.value, 50),
            config.HEDGE_EXCHANGE_WEIGHTS.get(Exchange.KRAKEN.value, 50)
        ]
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
            quantity_btc=final_size,
            price_usd=round(exec_price, config.PRICE_ROUNDING_DP),
            delta_impact=round(final_size if direction=="buy" else -final_size, 4),  # ✅ FIXED: Was backwards
            timestamp=time.time()
        )

        # Add to recent hedges log
        self.recent_hedges.append(hed)
        if len(self.recent_hedges) > config.MAX_RECENT_HEDGES_LOG_SIZE:
            self.recent_hedges.pop(0)

        # CRITICAL FIX: Update hedge tracking metrics
        transaction_value = final_size * exec_price
        self.total_hedges_executed += 1
        self.total_hedge_volume_btc += final_size
        self.total_hedge_volume_usd += transaction_value
        self.daily_hedge_volume += transaction_value
        self.total_hedge_value_usd_24h += transaction_value
        
        # Add to hedge history for tracking
        hedge_record = {
            "hedge_id": hedge_id,
            "timestamp": hed.timestamp,
            "quantity_btc": final_size,
            "price_usd": exec_price,
            "value_usd": transaction_value,
            "delta_impact": hed.delta_impact,
            "exchange": exch.value,
            "original_delta": abs_delta,
            "coverage_ratio": self.hedge_coverage_ratio,
            "proportional_size": proportional_hedge_size
        }
        self.hedge_history.append(hedge_record)
        
        # Keep hedge history manageable
        if len(self.hedge_history) > 1000:
            self.hedge_history = self.hedge_history[-1000:]

        # Enhanced logging with proportional sizing info
        coverage_pct = (final_size / abs_delta) * 100 if abs_delta > 0 else 0
        logger.info(f"Hedge executed {hedge_id}: {direction} {final_size:.4f} BTC on {exch.name} @ ${hed.price_usd:,.2f}")
        logger.info(f"  ├─ Original Delta: {abs_delta:.4f} BTC")
        logger.info(f"  ├─ Coverage: {coverage_pct:.1f}% of delta exposure") 
        logger.info(f"  ├─ Value: ${transaction_value:,.0f}")
        logger.info(f"  ├─ Delta Impact: {hed.delta_impact:+.4f} BTC")
        logger.info(f"  └─ Available Liquidity: {available_liquidity_btc:.2f} BTC")

        # CRITICAL FIX: Record hedge in position manager
        if self.position_manager and hasattr(self.position_manager, 'add_hedge_position'):
            try:
                hedge_data = {
                    "hedge_id": hedge_id,
                    "exchange": str(exch.value),  # Convert enum to string
                    "quantity_btc": final_size,
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

        # CRITICAL: Update liquidity manager with hedge transaction
        if self.liquidity_manager and hasattr(self.liquidity_manager, 'record_hedge_transaction'):
            try:
                self.liquidity_manager.record_hedge_transaction(
                    amount_usd=transaction_value,
                    hedge_id=hedge_id,
                    transaction_type="hedge_execution"
                )
            except Exception as e:
                logger.warning(f"HFM: Failed to record hedge in liquidity manager: {e}")

    def get_recent_hedges(self, limit: int = 10) -> List[HedgeExecution]:
        return sorted(self.recent_hedges, key=lambda h: h.timestamp, reverse=True)[:limit]

    def get_hedge_metrics(self) -> Dict[str, Any]:
        """FIXED: Calculate hedge metrics using both recent hedges and accumulated data"""
        try:
            now = time.time()
            cutoff = now - 24*3600

            # Filter recent hedges (24h)
            hedges_24h = [h for h in self.recent_hedges if h.timestamp >= cutoff]
            
            # Use accumulated data for accurate totals
            hedges_count_24h = max(len(hedges_24h), self.total_hedges_executed)
            
            # Calculate volumes using both recent and accumulated data
            recent_volume_btc = sum(h.quantity_btc for h in hedges_24h)
            total_volume_btc = max(recent_volume_btc, self.total_hedge_volume_btc)
            
            recent_value_usd = sum(h.quantity_btc * h.price_usd for h in hedges_24h)
            total_value_usd = max(recent_value_usd, self.total_hedge_value_usd_24h)

            # Calculate average execution time
            avg_exec_time = (
                sum(config.HEDGE_EXECUTION_TIMES_MS.get(h.exchange.value, 0) for h in hedges_24h)
                / max(len(hedges_24h), 1)
            )

            # Calculate liquidity metrics
            available_liquidity = self._get_available_liquidity_btc()
            
            # CRITICAL FIX: Estimate P&L for hedge metrics
            hedge_pnl_24h = self.hedge_pnl_accumulator

            # ENHANCED: Add hedge efficiency metrics
            total_delta_hedged = sum(abs(record.get("original_delta", 0)) for record in self.hedge_history[-hedges_count_24h:])
            hedge_efficiency = (total_volume_btc / max(total_delta_hedged, 0.001)) * 100 if total_delta_hedged > 0 else 0

            return {
                "hedges_24h": hedges_count_24h,
                "total_volume_hedged_btc_24h": round(total_volume_btc, 4),
                "total_hedge_value_usd_24h": round(total_value_usd, 2),
                "avg_execution_time_ms": round(avg_exec_time, 2),
                "hedge_pnl_24h": round(hedge_pnl_24h, 2),
                "available_liquidity_btc": round(available_liquidity, 4),
                "liquidity_utilization_pct": round((total_volume_btc / max(available_liquidity, 0.1)) * 100, 2),
                "hedge_efficiency_pct": round(hedge_efficiency, 1),  # NEW: Hedge efficiency metric
                "avg_hedge_size_btc": round(total_volume_btc / max(hedges_count_24h, 1), 4),  # NEW: Average hedge size
                "timestamp": now
            }
            
        except Exception as e:
            logger.error(f"HFM: Error calculating hedge metrics: {e}")
            return {
                "hedges_24h": 0,
                "total_volume_hedged_btc_24h": 0.0,
                "total_hedge_value_usd_24h": 0.0,
                "avg_execution_time_ms": 0.0,
                "hedge_pnl_24h": 0.0,
                "available_liquidity_btc": 0.0,
                "liquidity_utilization_pct": 0.0,
                "hedge_efficiency_pct": 0.0,
                "avg_hedge_size_btc": 0.0,
                "error": str(e),
                "timestamp": time.time()
            }

    def get_liquidity_status(self) -> Dict[str, Any]:
        """Get detailed liquidity status for hedging"""
        try:
            available_btc = self._get_available_liquidity_btc()
            max_single_hedge = min(available_btc * self.max_liquidity_per_hedge_pct, self.max_single_hedge_btc)
            current_price = self.position_manager.current_btc_price

            # Calculate recent usage
            now = time.time()
            recent_hedges = [h for h in self.recent_hedges if h.timestamp >= now - 3600]  # Last hour
            recent_volume = sum(h.quantity_btc for h in recent_hedges)

            status = "HEALTHY"
            if available_btc < 1.0:
                status = "CRITICAL"
            elif available_btc < 5.0:
                status = "LOW"
            elif available_btc < 10.0:
                status = "MODERATE"

            return {
                "status": status,
                "available_liquidity_btc": round(available_btc, 4),
                "available_liquidity_usd": round(available_btc * current_price, 0),
                "max_single_hedge_btc": round(max_single_hedge, 4),
                "max_single_hedge_usd": round(max_single_hedge * current_price, 0),
                "recent_usage_1h_btc": round(recent_volume, 4),
                "usage_rate_pct": round((recent_volume / max(available_btc, 0.1)) * 100, 2),
                "buffer_percentage": self.min_liquidity_buffer_pct * 100,
                "hedge_coverage_ratio": self.hedge_coverage_ratio * 100,
                "min_hedge_size_btc": self.min_hedge_size_btc,
                "max_hedge_size_btc": self.max_single_hedge_btc,
                "timestamp": now
            }

        except Exception as e:
            logger.error(f"HFM: Error getting liquidity status: {e}")
            return {"status": "ERROR", "error": str(e), "timestamp": time.time()}

    def get_hedge_status_summary(self) -> Dict[str, Any]:
        """Get summary of hedge manager status for debugging"""
        liquidity_status = self.get_liquidity_status()
        
        return {
            "is_running": self.is_running,
            "total_hedges_in_log": len(self.recent_hedges),
            "total_hedges_executed": self.total_hedges_executed,
            "total_hedge_volume_btc": self.total_hedge_volume_btc,
            "total_hedge_volume_usd": self.total_hedge_volume_usd,
            "position_manager_connected": self.position_manager is not None,
            "position_manager_has_add_method": (
                self.position_manager is not None and
                hasattr(self.position_manager, 'add_hedge_position')
            ),
            "liquidity_manager_connected": self.liquidity_manager is not None,
            "audit_engine_connected": self.audit_engine is not None,
            "current_btc_price": getattr(self.position_manager, 'current_btc_price', 0.0),
            "last_hedge_delta_impact": self.recent_hedges[-1].delta_impact if self.recent_hedges else 0.0,
            "liquidity_status": liquidity_status["status"],
            "available_liquidity_btc": liquidity_status.get("available_liquidity_btc", 0),
            "hedge_history_count": len(self.hedge_history),
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "hedge_coverage_ratio": self.hedge_coverage_ratio,
            "min_hedge_size_btc": self.min_hedge_size_btc,
            "max_single_hedge_btc": self.max_single_hedge_btc,
            "timestamp": time.time()
        }

    def get_24h_hedge_summary(self) -> Dict[str, Any]:
        """CRITICAL FIX: Get 24h hedge summary using accumulated data"""
        try:
            metrics = self.get_hedge_metrics()
            
            return {
                "total_hedges": metrics["hedges_24h"],
                "total_volume_btc": metrics["total_volume_hedged_btc_24h"],
                "total_value_usd": metrics["total_hedge_value_usd_24h"],
                "hedge_pnl": metrics["hedge_pnl_24h"],
                "avg_execution_time": metrics["avg_execution_time_ms"],
                "liquidity_utilization": metrics["liquidity_utilization_pct"],
                "hedge_efficiency": metrics.get("hedge_efficiency_pct", 0),
                "avg_hedge_size_btc": metrics.get("avg_hedge_size_btc", 0),
                "uptime_hours": (time.time() - self.start_time) / 3600,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"HFM: Error getting 24h summary: {e}")
            return {
                "total_hedges": 0,
                "total_volume_btc": 0.0,
                "total_value_usd": 0.0,
                "hedge_pnl": 0.0,
                "avg_execution_time": 0.0,
                "liquidity_utilization": 0.0,
                "hedge_efficiency": 0.0,
                "avg_hedge_size_btc": 0.0,
                "uptime_hours": 0.0,
                "error": str(e),
                "timestamp": time.time()
            }

    def force_hedge_execution_test(self) -> Dict[str, Any]:
        """Force execute a test hedge for debugging - CAUTION: Only for testing"""
        if not self.position_manager:
            return {"error": "No position manager available"}

        current_greeks = self.position_manager.get_aggregate_platform_greeks()
        current_delta = current_greeks.get("net_portfolio_delta_btc", 0.0)
        liquidity_before = self.get_liquidity_status()

        logger.warning(f"HFM: FORCING TEST HEDGE EXECUTION - Current delta: {current_delta:.4f}")

        # Force execute hedge regardless of threshold
        self._execute_hedge(current_delta)

        # Get updated state after hedge
        updated_greeks = self.position_manager.get_aggregate_platform_greeks()
        liquidity_after = self.get_liquidity_status()

        return {
            "test_executed": True,
            "delta_before": current_delta,
            "delta_after": updated_greeks.get("net_portfolio_delta_btc", 0.0),
            "hedges_count_before": current_greeks.get("open_hedges_count", 0),
            "hedges_count_after": updated_greeks.get("open_hedges_count", 0),
            "liquidity_before_btc": liquidity_before.get("available_liquidity_btc", 0),
            "liquidity_after_btc": liquidity_after.get("available_liquidity_btc", 0),
            "last_hedge_delta_impact": self.recent_hedges[-1].delta_impact if self.recent_hedges else 0.0,
            "timestamp": time.time()
        }

    def clear_all_hedges(self) -> Dict[str, Any]:
        """ENHANCED: Clear all hedge history and reset metrics - for testing/debugging"""
        logger.warning("HFM: CLEARING ALL HEDGE HISTORY")

        hedge_count = len(self.recent_hedges)
        self.recent_hedges.clear()

        # Also clear position manager hedge positions if method exists
        if self.position_manager and hasattr(self.position_manager, 'open_hedge_positions'):
            position_count = len(self.position_manager.open_hedge_positions)
            self.position_manager.open_hedge_positions.clear()
            logger.warning(f"HFM: Cleared {position_count} hedge positions from position manager")

        # CRITICAL FIX: Reset all hedge metrics
        self.reset_hedge_metrics()

        return {
            "cleared_hedge_history": hedge_count,
            "metrics_reset": True,
            "timestamp": time.time()
        }

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information about hedge feed manager state"""
        return {
            "is_running": self.is_running,
            "recent_hedges_count": len(self.recent_hedges),
            "hedge_history_count": len(self.hedge_history),
            "total_hedges_executed": self.total_hedges_executed,
            "total_hedge_volume_btc": self.total_hedge_volume_btc,
            "total_hedge_volume_usd": self.total_hedge_volume_usd,
            "hedge_pnl_accumulator": self.hedge_pnl_accumulator,
            "daily_hedge_volume": self.daily_hedge_volume,
            "min_liquidity_buffer_pct": self.min_liquidity_buffer_pct,
            "max_liquidity_per_hedge_pct": self.max_liquidity_per_hedge_pct,
            "hedge_coverage_ratio": self.hedge_coverage_ratio,
            "min_hedge_size_btc": self.min_hedge_size_btc,
            "max_single_hedge_btc": self.max_single_hedge_btc,
            "position_manager_connected": self.position_manager is not None,
            "liquidity_manager_connected": self.liquidity_manager is not None,
            "audit_engine_connected": self.audit_engine is not None,
            "current_btc_price": getattr(self.position_manager, 'current_btc_price', 0.0),
            "uptime_seconds": time.time() - self.start_time,
            "hedge_metrics_24h_keys": list(self.hedge_metrics_24h.keys()) if self.hedge_metrics_24h else [],
            "daily_hedge_data_keys": list(self.daily_hedge_data.keys()) if self.daily_hedge_data else [],
            "exchanges_enabled": [ex.value for ex in [Exchange.COINBASE_PRO, Exchange.KRAKEN]],
            "timestamp": time.time()
        }
