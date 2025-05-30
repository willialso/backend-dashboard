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

    def _update_all_positions(self):
        """Recompute MTM and Greeks for all open positions."""
        now = time.time()
        for pos in self.open_option_positions.values():
            expiry_ts = pos.expiry_timestamp
            T = max((expiry_ts - now) / (365.25*24*3600), 0.0)
            price_per_unit, greeks = self.pricing_engine.black_scholes_with_greeks(
                S=self.current_btc_price,
                K=pos.strike,
                T=T,
                r=config.RISK_FREE_RATE,
                sigma=pos.volatility_at_trade,
                option_type=pos.option_type
            )
            multiplier = -1.0 if pos.is_short else 1.0
            pos.current_mtm_value_usd_total = round(price_per_unit * pos.quantity * multiplier, 2)
            # Update PnL
            pos.unrealized_pnl_usd = round(pos.current_mtm_value_usd_total - pos.initial_premium_total, 2)

    def get_aggregate_platform_greeks(self) -> Dict[str, Any]:
        """Sum up all option position Greeks into portfolio-level risk metrics."""
        self._update_all_positions()
        net_delta = sum(p.initial_delta_total or 0.0 for p in self.open_option_positions.values())
        net_gamma = 0.0
        net_vega = 0.0
        net_theta = 0.0
        now = time.time()

        for pos in self.open_option_positions.values():
            T = max((pos.expiry_timestamp - now) / (365.25*24*3600), 0.0)
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
            net_vega  += greeks.get("vega",  0.0) * pos.quantity * multiplier
            net_theta += greeks.get("theta", 0.0) * pos.quantity * multiplier

        status = ("Requires Hedging"
                  if abs(net_delta) > self.delta_hedge_threshold
                  else "Delta Neutral")

        return {
            "timestamp": now,
            "net_portfolio_delta_btc": round(net_delta, 4),
            "net_portfolio_gamma_btc": round(net_gamma, 4),
            "net_portfolio_vega_usd": round(net_vega,  2),
            "net_portfolio_theta_usd": round(net_theta, 2),
            "risk_status_message": status,
            "open_options_count": len(self.open_option_positions),
            "open_hedges_count": len(self.open_hedge_positions)
        }
