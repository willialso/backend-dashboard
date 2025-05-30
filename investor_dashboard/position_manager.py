# investor_dashboard/position_manager.py

import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from backend.advanced_pricing_engine import AdvancedPricingEngine
from backend import config

logger = logging.getLogger(__name__)

@dataclass
class EnrichedTradeDataInPM:
    trade_id: str
    option_type: str
    strike: float
    expiry_timestamp: float
    quantity: float
    is_short: bool
    initial_premium_total: float
    initial_delta_total: Optional[float] = 0.0
    current_mtm_value_usd_total: float = 0.0
    unrealized_pnl_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    status: str = "OPEN"
    underlying_price_at_trade: float = 0.0
    volatility_at_trade: float = 0.0
    time_to_expiry_at_trade_years: float = 0.0

@dataclass
class EnrichedHedgePositionInPM:
    hedge_id: str
    instrument_type: str
    quantity_btc: float
    entry_price_usd: float
    entry_timestamp: float
    current_mtm_pnl_usd: float = 0.0
    current_delta_contribution_btc: float = 0.0
    status: str = "OPEN"

class PositionManager:
    """Tracks open option and hedge positions and computes portfolio Greeks."""

    def __init__(self, pricing_engine_instance: AdvancedPricingEngine):
        self.open_option_positions: Dict[str, EnrichedTradeDataInPM] = {}
        self.open_hedge_positions: Dict[str, EnrichedHedgePositionInPM] = {}
        self.pricing_engine = pricing_engine_instance
        self.current_btc_price = 0.0
        self.delta_hedge_threshold = config.MAX_PLATFORM_NET_DELTA_BTC
        logger.info(f"PositionManager initialized. Hedge threshold: {self.delta_hedge_threshold} BTC")

    def update_market_data(self, price: float):
        """Receive live BTC price updates, propagate to pricing engine and update positions."""
        if price <= 0:
            logger.warning(f"PM: Ignoring invalid price {price}")
            return

        self.current_btc_price = price
        if hasattr(self.pricing_engine, 'update_market_data'):
            self.pricing_engine.update_market_data(price)
        
        self._update_all_positions()

    def add_option_trade(self, details: Dict[str, Any]):
        """Record a new option trade, compute its initial delta & MTM value."""
        if not self.pricing_engine or self.current_btc_price <= 0:
            logger.error("PM: Cannot record trade—pricing engine or price missing")
            return

        now = time.time()
        expiry_ts = details["expiry_timestamp"]
        T = max((expiry_ts - now) / (365.25*24*3600), 0.0)

        # Black-Scholes to get price and Greeks
        price_per_unit, greeks = self.pricing_engine.black_scholes_with_greeks(
            S=details["underlying_price_at_trade"],
            K=details["strike"],
            T=T,
            r=config.RISK_FREE_RATE,
            sigma=details["volatility_at_trade"],
            option_type=details["option_type"]
        )

        # Delta: negative if short
        delta = greeks.get("delta", 0.0) * details["quantity"]
        if details["is_short"]:
            delta = -delta

        pos = EnrichedTradeDataInPM(
            trade_id=details["trade_id"],
            option_type=details["option_type"],
            strike=details["strike"],
            expiry_timestamp=expiry_ts,
            quantity=details["quantity"],
            is_short=details["is_short"],
            initial_premium_total=details["initial_premium_total"],
            initial_delta_total=round(delta, 8),
            underlying_price_at_trade=details["underlying_price_at_trade"],
            volatility_at_trade=details["volatility_at_trade"],
            time_to_expiry_at_trade_years=T,
            current_mtm_value_usd_total=round(price_per_unit * details["quantity"], 2)
        )

        self.open_option_positions[pos.trade_id] = pos
        logger.info(f"PM: Recorded trade {pos.trade_id}, Δ={pos.initial_delta_total}, "
                   f"premium=${pos.initial_premium_total:.2f}")

    def add_hedge_position(self, hedge_data: Dict[str, Any]):
        """CRITICAL FIX: Record hedge execution as a position that offsets delta"""
        try:
            hedge_id = hedge_data.get("hedge_id", f"hedge_{int(time.time()*1000)}")
            quantity_btc = hedge_data.get("quantity_btc", 0.0)

            # Create hedge position
            pos = EnrichedHedgePositionInPM(
                hedge_id=hedge_id,
                instrument_type=hedge_data.get("instrument_type", "BTC_HEDGE"),
                quantity_btc=quantity_btc,
                entry_price_usd=hedge_data.get("price_usd", 0.0),
                entry_timestamp=hedge_data.get("timestamp", time.time()),
                current_delta_contribution_btc=hedge_data.get("delta_impact", quantity_btc),  # ✅ FIXED - use actual delta
                status="OPEN"
            )

            # Store hedge position
            self.open_hedge_positions[hedge_id] = pos

            # Update MTM if current price available
            if self.current_btc_price > 0:
                current_value = quantity_btc * self.current_btc_price
                entry_value = quantity_btc * pos.entry_price_usd
                pos.current_mtm_pnl_usd = current_value - entry_value

            logger.info(f"PM: ✅ Recorded hedge {hedge_id}: +{quantity_btc:.4f} BTC delta, "
                       f"price=${pos.entry_price_usd:,.2f}")

        except Exception as e:
            logger.error(f"PM: ❌ Failed to record hedge position: {e}", exc_info=True)

    def _safe_cleanup_expired_positions(self):
        """CRITICAL FIX: Safe cleanup that doesn't modify dict during iteration"""
        try:
            now = time.time()
            expired_option_ids = []
            
            # FIXED: Safe iteration - don't modify during loop
            for pos_id, position in list(self.open_option_positions.items()):
                try:
                    if position.expiry_timestamp <= now:
                        expired_option_ids.append(pos_id)
                        position.status = "EXPIRED"
                except Exception as e:
                    logger.warning(f"PM: Error checking position {pos_id}: {e}")
                    expired_option_ids.append(pos_id)  # Remove problematic positions
            
            # Now safely remove collected positions
            for pos_id in expired_option_ids:
                if pos_id in self.open_option_positions:
                    removed_pos = self.open_option_positions.pop(pos_id, None)
                    if removed_pos:
                        logger.debug(f"PM: Cleaned up expired position {pos_id}")
                        
        except Exception as e:
            logger.error(f"PM: Error in safe position cleanup: {e}")

    def _update_all_positions(self):
        """FIXED: Recompute MTM and Greeks for all open positions without iteration errors."""
        now = time.time()
        
        # CRITICAL FIX: Safe cleanup first
        self._safe_cleanup_expired_positions()

        # FIXED: Use copy for safe iteration during updates
        option_positions_copy = dict(self.open_option_positions)
        
        # Update option positions
        for pos_id, pos in option_positions_copy.items():
            # Skip if position was removed during cleanup
            if pos_id not in self.open_option_positions:
                continue
                
            expiry_ts = pos.expiry_timestamp
            T = max((expiry_ts - now) / (365.25*24*3600), 0.0)
            
            if T <= 0:  # Expired
                if pos_id in self.open_option_positions:
                    self.open_option_positions[pos_id].status = "EXPIRED"
                continue

            try:
                price_per_unit, greeks = self.pricing_engine.black_scholes_with_greeks(
                    S=self.current_btc_price,
                    K=pos.strike,
                    T=T,
                    r=config.RISK_FREE_RATE,
                    sigma=pos.volatility_at_trade,
                    option_type=pos.option_type
                )

                multiplier = -1.0 if pos.is_short else 1.0
                current_mtm = round(price_per_unit * pos.quantity * multiplier, 2)
                
                # Update position if it still exists
                if pos_id in self.open_option_positions:
                    self.open_option_positions[pos_id].current_mtm_value_usd_total = current_mtm
                    self.open_option_positions[pos_id].unrealized_pnl_usd = round(current_mtm - pos.initial_premium_total, 2)

            except Exception as e:
                logger.warning(f"PM: Error updating position {pos.trade_id}: {e}")

        # Update hedge positions MTM (these are safer as they're not removed automatically)
        if self.current_btc_price > 0:
            for hedge_pos in self.open_hedge_positions.values():
                try:
                    current_value = hedge_pos.quantity_btc * self.current_btc_price
                    entry_value = hedge_pos.quantity_btc * hedge_pos.entry_price_usd
                    hedge_pos.current_mtm_pnl_usd = round(current_value - entry_value, 2)
                except Exception as e:
                    logger.warning(f"PM: Error updating hedge {hedge_pos.hedge_id}: {e}")

    def get_aggregate_platform_greeks(self) -> Dict[str, Any]:
        """FIXED: Safe aggregation without iteration errors"""
        try:
            # Update positions first
            self._update_all_positions()

            # FIXED: Use safe iteration with list() to avoid "dict changed size" error
            option_positions_safe = list(self.open_option_positions.values())
            hedge_positions_safe = list(self.open_hedge_positions.values())

            # Calculate option positions delta and other Greeks
            net_delta_from_options = sum(
                p.initial_delta_total or 0.0 for p in option_positions_safe 
                if p.status == "OPEN"
            )

            net_gamma = 0.0
            net_vega = 0.0
            net_theta = 0.0
            now = time.time()

            for pos in option_positions_safe:
                if pos.status != "OPEN":
                    continue

                T = max((pos.expiry_timestamp - now) / (365.25*24*3600), 0.0)
                if T <= 0:
                    continue

                try:
                    _, greeks = self.pricing_engine.black_scholes_with_greeks(
                        S=self.current_btc_price,
                        K=pos.strike,
                        T=T,
                        r=config.RISK_FREE_RATE,
                        sigma=pos.volatility_at_trade,
                        option_type=pos.option_type
                    )

                    multiplier = -1.0 if pos.is_short else 1.0
                    net_gamma += greeks.get("gamma", 0.0) * pos.quantity * multiplier
                    net_vega += greeks.get("vega", 0.0) * pos.quantity * multiplier
                    net_theta += greeks.get("theta", 0.0) * pos.quantity * multiplier

                except Exception as e:
                    logger.warning(f"PM: Error calculating Greeks for {pos.trade_id}: {e}")

            # CRITICAL FIX: Add hedge positions delta contribution
            net_delta_from_hedges = sum(
                h.current_delta_contribution_btc for h in hedge_positions_safe 
                if h.status == "OPEN"
            )

            # Total portfolio delta = options delta + hedges delta
            net_delta_total = net_delta_from_options + net_delta_from_hedges

            # Determine risk status
            status = (
                "Requires Hedging" 
                if abs(net_delta_total) > self.delta_hedge_threshold 
                else "Within Limits"
            )

            # Count positions
            active_options = sum(1 for p in option_positions_safe if p.status == "OPEN")
            active_hedges = sum(1 for h in hedge_positions_safe if h.status == "OPEN")

            result = {
                "timestamp": now,
                "net_portfolio_delta_btc": round(net_delta_total, 4),
                "net_portfolio_gamma_btc": round(net_gamma, 4),
                "net_portfolio_vega_usd": round(net_vega, 2),
                "net_portfolio_theta_usd": round(net_theta, 2),
                "risk_status_message": status,
                "open_options_count": active_options,
                "open_hedges_count": active_hedges,
                # Debug info
                "debug_options_delta": round(net_delta_from_options, 4),
                "debug_hedges_delta": round(net_delta_from_hedges, 4)
            }

            logger.debug(f"PM Greeks: Total Δ={net_delta_total:.4f} (Options: {net_delta_from_options:.4f}, "
                        f"Hedges: {net_delta_from_hedges:.4f}), Status: {status}")

            return result

        except Exception as e:
            logger.error(f"PM: Error in get_aggregate_platform_greeks: {e}", exc_info=True)
            return self._get_fallback_greeks()

    def _get_fallback_greeks(self) -> Dict[str, Any]:
        """Fallback Greeks when calculation fails"""
        return {
            "timestamp": time.time(),
            "net_portfolio_delta_btc": 0.0,
            "net_portfolio_gamma_btc": 0.0,
            "net_portfolio_vega_usd": 0.0,
            "net_portfolio_theta_usd": 0.0,
            "risk_status_message": "Error",
            "open_options_count": 0,
            "open_hedges_count": 0,
            "debug_options_delta": 0.0,
            "debug_hedges_delta": 0.0
        }

    def close_expired_positions(self):
        """FIXED: Remove expired option positions from tracking safely."""
        try:
            now = time.time()
            expired_ids = []

            # FIXED: Safe iteration
            for trade_id, pos in list(self.open_option_positions.items()):
                if pos.expiry_timestamp <= now:
                    expired_ids.append(trade_id)
                    pos.status = "EXPIRED"
                    logger.info(f"PM: Expired option {trade_id}")

            # Now safely remove
            for trade_id in expired_ids:
                self.open_option_positions.pop(trade_id, None)

        except Exception as e:
            logger.error(f"PM: Error in close_expired_positions: {e}")

    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions for debugging/monitoring."""
        try:
            # FIXED: Safe iteration
            active_options = [p for p in list(self.open_option_positions.values()) if p.status == "OPEN"]
            active_hedges = [h for h in list(self.open_hedge_positions.values()) if h.status == "OPEN"]

            total_option_pnl = sum(p.unrealized_pnl_usd for p in active_options)
            total_hedge_pnl = sum(h.current_mtm_pnl_usd for h in active_hedges)

            return {
                "active_options_count": len(active_options),
                "active_hedges_count": len(active_hedges),
                "total_option_unrealized_pnl": round(total_option_pnl, 2),
                "total_hedge_pnl": round(total_hedge_pnl, 2),
                "net_portfolio_pnl": round(total_option_pnl + total_hedge_pnl, 2),
                "current_btc_price": self.current_btc_price,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"PM: Error in get_position_summary: {e}")
            return {
                "active_options_count": 0,
                "active_hedges_count": 0,
                "total_option_unrealized_pnl": 0.0,
                "total_hedge_pnl": 0.0,
                "net_portfolio_pnl": 0.0,
                "current_btc_price": self.current_btc_price,
                "timestamp": time.time()
            }

    def force_delta_recalculation(self):
        """Force recalculation of all position deltas - for debugging."""
        logger.info("PM: Force recalculating all position deltas")
        self._update_all_positions()
        greeks = self.get_aggregate_platform_greeks()
        logger.info(f"PM: Recalculated - Total Delta: {greeks['net_portfolio_delta_btc']:.4f} BTC, "
                   f"Options: {greeks.get('debug_options_delta', 0):.4f}, "
                   f"Hedges: {greeks.get('debug_hedges_delta', 0):.4f}")
        return greeks

    def reset_all_positions(self):
        """CRITICAL FIX: Reset all positions for complete system reset"""
        try:
            logger.warning("PM: Resetting all positions to zero")
            
            # Clear all position dictionaries
            self.open_option_positions.clear()
            self.open_hedge_positions.clear()
            
            # Reset any other tracking attributes
            for attr in ['total_portfolio_delta', 'total_gamma', 'total_vega', 'total_theta']:
                if hasattr(self, attr):
                    setattr(self, attr, 0.0)
                    
            logger.info("✅ PM: All positions reset to zero")
            
        except Exception as e:
            logger.error(f"❌ PM: Error resetting positions: {e}")
