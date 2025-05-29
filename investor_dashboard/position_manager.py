# investor_dashboard/position_manager.py

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import logging
import math
from scipy.stats import norm
import random  # <-- ADD THIS LINE

from backend.advanced_pricing_engine import AdvancedPricingEngine
from backend import config # Import your config

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
    initial_delta_total: Optional[float] = None
    current_mtm_value_usd_total: Optional[float] = 0.0
    unrealized_pnl_usd: Optional[float] = 0.0
    realized_pnl_usd: Optional[float] = 0.0
    status: str = "OPEN"
    underlying_price_at_trade: float = 0.0
    volatility_at_trade: float = 0.0
    time_to_expiry_at_trade_years: float = 0.0

@dataclass
class EnrichedHedgePositionInPM:
    hedge_id: str
    instrument_type: str
    quantity_btc: float # Signed: positive for long, negative for short
    entry_price_usd: float
    entry_timestamp: float
    current_mtm_pnl_usd: Optional[float] = 0.0
    current_delta_contribution_btc: Optional[float] = 0.0
    status: str = "OPEN"

class PositionManager:
    def __init__(self, pricing_engine_instance: AdvancedPricingEngine):
        self.open_option_positions: Dict[str, EnrichedTradeDataInPM] = {}
        self.open_hedge_positions: Dict[str, EnrichedHedgePositionInPM] = {}
        self.pricing_engine = pricing_engine_instance
        self.current_btc_price = 0.0
        self.current_risk_free_rate = config.RISK_FREE_RATE
        self.delta_hedge_threshold = config.MAX_PLATFORM_NET_DELTA_BTC # Using config value
        logger.info(f"PositionManager initialized. Delta Hedge Threshold: {self.delta_hedge_threshold} BTC.")

    def update_market_data(self, current_btc_price: float):
        if current_btc_price <= 0:
            logger.warning(f"PM: Invalid BTC price update: {current_btc_price}")
            return
        self.current_btc_price = current_btc_price
        if hasattr(self.pricing_engine, 'update_market_data'): # APE needs this if it caches prices/vol
             self.pricing_engine.update_market_data(current_btc_price)
        self._update_all_positions_mtm_and_greeks()

    def add_option_trade(self, trade_details: Dict):
        if not self.pricing_engine or self.current_btc_price == 0:
            logger.error("PM: Cannot add option trade, pricing engine or BTC price unavailable.")
            return

        trade_id = trade_details.get("trade_id", f"opt_{int(time.time()*1000)}_{random.randint(100,999)}")
        time_to_expiry_years = (trade_details["expiry_timestamp"] - trade_details["timestamp"]) / (365.25 * 24 * 60 * 60)

        # APE's black_scholes_with_greeks is static, but relies on vol_engine within APE for sigma if not provided
        # We need the sigma that will be used. RE should pass vol_at_trade.
        initial_value_per_unit, initial_greeks_per_unit = self.pricing_engine.black_scholes_with_greeks(
            S=trade_details["underlying_price_at_trade"],
            K=trade_details["strike"],
            T=time_to_expiry_years,
            r=self.current_risk_free_rate,
            sigma=trade_details["volatility_at_trade"], # This sigma comes from RevenueEngine via VolatilityEngine
            option_type=trade_details["option_type"]
        )
        
        initial_delta_total = initial_greeks_per_unit.get("delta", 0.0) * trade_details["quantity"]
        if trade_details["is_short"]:
            initial_delta_total *= -1

        pm_trade = EnrichedTradeDataInPM(
            trade_id=trade_id,
            option_type=trade_details["option_type"],
            strike=trade_details["strike"],
            expiry_timestamp=trade_details["expiry_timestamp"],
            quantity=trade_details["quantity"],
            is_short=trade_details["is_short"],
            initial_premium_total=trade_details["initial_premium_total"],
            initial_delta_total=initial_delta_total,
            underlying_price_at_trade=trade_details["underlying_price_at_trade"],
            volatility_at_trade=trade_details["volatility_at_trade"],
            time_to_expiry_at_trade_years=time_to_expiry_years,
            current_mtm_value_usd_total=initial_value_per_unit * trade_details["quantity"] if not trade_details["is_short"] else -initial_value_per_unit * trade_details["quantity"],
            unrealized_pnl_usd=0.0,
            status="OPEN"
        )
        self.open_option_positions[trade_id] = pm_trade
        logger.info(f"PM: Added option {trade_id} K{pm_trade.strike} {pm_trade.option_type}. Initial Delta: {initial_delta_total:.4f}. Premium: ${pm_trade.initial_premium_total:.2f}")

    def add_hedge_position(self, hedge_details: Dict):
        hedge_id = hedge_details.get("hedge_id", f"h_{int(time.time()*1000)}_{random.randint(100,999)}")
        quantity_btc_signed = hedge_details["quantity_btc_signed"]

        pm_hedge = EnrichedHedgePositionInPM(
            hedge_id=hedge_id,
            instrument_type=hedge_details["instrument_type"],
            quantity_btc=quantity_btc_signed,
            entry_price_usd=hedge_details["entry_price_usd"],
            entry_timestamp=hedge_details["timestamp"],
            current_delta_contribution_btc=quantity_btc_signed,
            current_mtm_pnl_usd=0.0,
            status="OPEN"
        )
        self.open_hedge_positions[hedge_id] = pm_hedge
        logger.info(f"PM: Added hedge {hedge_id}. Qty: {quantity_btc_signed:.4f} BTC @ ${pm_hedge.entry_price_usd:.2f}. Delta Impact: {quantity_btc_signed:.4f}")

    def _update_all_positions_mtm_and_greeks(self):
        if not self.pricing_engine or self.current_btc_price == 0:
            return

        logger.debug(f"PM: Updating MTM & Greeks for all positions. BTC Price: ${self.current_btc_price:.2f}")
        current_time = time.time()
        audit_engine_instance = getattr(self, 'audit_engine', None) # Get if injected

        for trade_id, trade in list(self.open_option_positions.items()):
            if trade.status != "OPEN": continue

            time_to_expiry_years = (trade.expiry_timestamp - current_time) / (365.25 * 24 * 60 * 60)

            if time_to_expiry_years <= 1e-6: # Effectively expired
                final_payout_per_unit = 0.0
                if trade.option_type.lower() == "call":
                    final_payout_per_unit = max(0, self.current_btc_price - trade.strike)
                elif trade.option_type.lower() == "put":
                    final_payout_per_unit = max(0, trade.strike - self.current_btc_price)
                
                final_payout_total = final_payout_per_unit * trade.quantity
                
                if trade.is_short:
                    trade.realized_pnl_usd = trade.initial_premium_total - final_payout_total
                else: # Platform bought (unlikely for current model)
                    trade.realized_pnl_usd = final_payout_total - trade.initial_premium_total
                
                trade.status = "EXPIRED_ITM" if final_payout_total > 0 else "EXPIRED_OTM"
                trade.current_mtm_value_usd_total = 0.0
                trade.unrealized_pnl_usd = 0.0 # Becomes realized
                logger.info(f"PM: Option {trade_id} EXPIRED. PayoutTotal=${final_payout_total:.2f}, Platform RealizedPNL=${trade.realized_pnl_usd:.2f}")
                
                if audit_engine_instance: # Log realized PNL to AuditEngine
                    # This needs a specific method in AuditEngine or use a general transaction logger
                    # audit_engine_instance.track_realized_option_pnl(trade.realized_pnl_usd, trade_id)
                    pass # For now, PM handles it internally
                del self.open_option_positions[trade_id]
                continue

            # Use the APE's vol_engine to get current sigma for this option
            current_sigma_for_option = self.pricing_engine.vol_engine.get_expiry_adjusted_volatility(
                expiry_minutes=time_to_expiry_years * 365.25 * 24 * 60,
                strike_price=trade.strike,
                underlying_price=self.current_btc_price
            )
            if current_sigma_for_option <=0: current_sigma_for_option = config.MIN_VOLATILITY

            mtm_value_per_unit, _ = self.pricing_engine.black_scholes_with_greeks(
                S=self.current_btc_price, K=trade.strike, T=time_to_expiry_years,
                r=self.current_risk_free_rate, sigma=current_sigma_for_option, option_type=trade.option_type
            )
            trade.current_mtm_value_usd_total = mtm_value_per_unit * trade.quantity

            if trade.is_short:
                trade.unrealized_pnl_usd = trade.initial_premium_total - trade.current_mtm_value_usd_total
            else:
                trade.unrealized_pnl_usd = trade.current_mtm_value_usd_total - trade.initial_premium_total
        
        for hedge_id, hedge in self.open_hedge_positions.items():
            if hedge.status != "OPEN": continue
            hedge.current_mtm_pnl_usd = (self.current_btc_price - hedge.entry_price_usd) * hedge.quantity_btc
            hedge.current_delta_contribution_btc = hedge.quantity_btc # Assuming spot/linear futures

    def get_aggregate_platform_greeks(self) -> Dict:
        self._update_all_positions_mtm_and_greeks() # Ensure all data is fresh

        net_delta_btc = 0.0
        net_gamma_usd_per_1_pct_sq = 0.0 # This unit/calculation needs careful thought for portfolio level
        net_vega_usd_per_1_vol_pt = 0.0
        net_theta_usd_per_day = 0.0
        current_time = time.time()

        for trade in self.open_option_positions.values():
            if trade.status != "OPEN": continue
            time_to_expiry_years = (trade.expiry_timestamp - current_time) / (365.25 * 24 * 60 * 60)
            if time_to_expiry_years <= 1e-6: continue

            current_sigma_for_option = self.pricing_engine.vol_engine.get_expiry_adjusted_volatility(
                expiry_minutes=time_to_expiry_years * 365.25 * 24 * 60,
                strike_price=trade.strike,
                underlying_price=self.current_btc_price
            )
            if current_sigma_for_option <=0: current_sigma_for_option = config.MIN_VOLATILITY

            _, greeks_per_unit = self.pricing_engine.black_scholes_with_greeks(
                S=self.current_btc_price, K=trade.strike, T=time_to_expiry_years,
                r=self.current_risk_free_rate, sigma=current_sigma_for_option, option_type=trade.option_type
            )
            
            multiplier = trade.quantity * (-1 if trade.is_short else 1)
            net_delta_btc += greeks_per_unit.get("delta", 0.0) * multiplier
            
            # Gamma (BS output is dDelta/dS). Portfolio Gamma ($ value for 1% move): 0.5 * S^2 * Gamma_BS * (0.01)^2 * total_contracts_in_btc_equiv
            # For simplicity, we'll provide gamma in terms of dDelta_BTC / dS_USD (from BS output * quantity)
            # Then Lovable can display it or derive $Gamma.
            # net_gamma_btc_per_usd_S_change += greeks_per_unit.get("gamma", 0.0) * multiplier 
            # Sticking to original units for now to match APE output
            net_gamma_usd_per_1_pct_sq += greeks_per_unit.get("gamma", 0.0) * multiplier # This needs unit consistency with APE output. APE gamma is d2V/dS2.
                                                                                      # Dashboard typically wants $ change for 1% S move.
                                                                                      # Dollar Gamma = Gamma_APE * S^2 * 0.01
            net_vega_usd_per_1_vol_pt += greeks_per_unit.get("vega", 0.0) * multiplier   # APE Vega is dV/dSigma (per 1 abs vol change). To get per 1 vol pt (1%), divide by 100.
            net_theta_usd_per_day += greeks_per_unit.get("theta", 0.0) * multiplier # APE Theta is dV/dT (per 1 year). To get per day, divide by 365.

        for hedge in self.open_hedge_positions.values():
            if hedge.status != "OPEN" or hedge.current_delta_contribution_btc is None: continue
            net_delta_btc += hedge.current_delta_contribution_btc
            # Spot/linear futures have minimal other Greeks usually ignored at portfolio level for simplicity

        risk_status = "Requires Hedging" if abs(net_delta_btc) > self.delta_hedge_threshold else "Delta Neutral"
        if not self.open_option_positions and not self.open_hedge_positions:
            risk_status = "Flat - No Open Positions"

        logger.info(f"PM Agg Greeks: Delta={net_delta_btc:.4f}, Gamma(raw)={net_gamma_usd_per_1_pct_sq:.4f}, Vega(raw)={net_vega_usd_per_1_vol_pt:.2f}, Theta(raw)={net_theta_usd_per_day:.2f}")
        return {
            "timestamp": current_time,
            "net_portfolio_delta_btc": round(net_delta_btc, 4),
            "net_portfolio_gamma_raw": round(net_gamma_usd_per_1_pct_sq, 6), # Raw BS Gamma (d2V/dS2), sum over positions
            "net_portfolio_vega_raw_per_abs_vol": round(net_vega_usd_per_1_vol_pt, 2), # Raw BS Vega (dV/dSigma), sum over positions
            "net_portfolio_theta_raw_per_year": round(net_theta_usd_per_day, 2), # Raw BS Theta (dV/dT), sum over positions
            "risk_status_message": risk_status,
            "open_options_count": len([p for p in self.open_option_positions.values() if p.status == "OPEN"]),
            "open_hedges_count": len([p for p in self.open_hedge_positions.values() if p.status == "OPEN"]),
        }

    def get_open_option_positions_detail(self, limit: int = config.MAX_RECENT_TRADES_LOG_SIZE_BOTSIM) -> List[Dict]:
        self._update_all_positions_mtm_and_greeks()
        open_options = [pos for pos_id, pos in self.open_option_positions.items() if pos.status == "OPEN"]
        return [pos.__dict__ for pos in sorted(open_options, key=lambda x: x.expiry_timestamp)[:limit]]

    def get_open_hedge_positions_detail(self, limit: int = config.MAX_RECENT_HEDGES_LOG_SIZE) -> List[Dict]:
        self._update_all_positions_mtm_and_greeks()
        open_hedges = [pos for pos_id, pos in self.open_hedge_positions.items() if pos.status == "OPEN"]
        return [pos.__dict__ for pos in sorted(open_hedges, key=lambda x: x.entry_timestamp, reverse=True)[:limit]]

