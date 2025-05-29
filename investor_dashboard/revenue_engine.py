# investor_dashboard/revenue_engine.py

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
import random  # <-- ADD THIS LINE

from backend.advanced_pricing_engine import AdvancedPricingEngine, OptionQuote # Make sure OptionQuote is used or defined
from backend.advanced_volatility_engine import AdvancedVolatilityEngine, VolatilityMetrics #removed volitilityregime
from backend import config # Use your config file

logger = logging.getLogger(__name__)

@dataclass
class RevenueMetrics:
    bsm_fair_value_per_contract: float
    platform_price_per_contract: float
    markup_percentage_applied: float
    revenue_captured_per_contract: float
    daily_revenue_estimate_usd: float
    simulated_contracts_sold_24h: int
    average_markup_on_fair_value_pct: float
    gross_premium_per_contract: float
    volatility_used_for_pricing_pct: float # Actual sigma used (annualized decimal)
    calculation_method_used: str

class RevenueEngine:
    def __init__(self,
                 volatility_engine_instance: AdvancedVolatilityEngine,
                 pricing_engine_instance: AdvancedPricingEngine,
                 position_manager_instance: Optional[Any] = None, # PositionManager
                 audit_engine_instance: Optional[Any] = None):    # AuditEngine
        logger.info("🔧 Initializing Revenue Engine with Advanced Engines & Dependencies...")
        self.vol_engine = volatility_engine_instance
        self.pricing_engine = pricing_engine_instance
        self.position_manager = position_manager_instance
        self.audit_engine = audit_engine_instance
        
        self.current_btc_price: float = 0.0
        self.base_markup_percentage: float = config.REVENUE_BASE_MARKUP_PERCENTAGE
        self.last_price_update_ts: float = 0.0
        self.price_update_count: int = 0
        self.debug_mode: bool = config.DEBUG_MODE
        logger.info("✅ Revenue Engine initialized.")

    def update_price(self, btc_price: float):
        if btc_price <= 0: return
        self.current_btc_price = btc_price
        self.last_price_update_ts = time.time()
        self.price_update_count += 1
        # APE and VolEngine are primarily updated by the main loop via PositionManager or directly.
        # This method is just for RE's internal state of current_btc_price.
        if self.debug_mode and self.price_update_count % 10 == 0:
             logger.debug(f"💰 RE: Price updated to ${btc_price:,.2f}")

    def price_option_for_platform_sale(self,
                                       strike_price: float,
                                       expiry_minutes: int,
                                       option_type: str = 'call',
                                       quantity: float = 1.0,
                                       trade_id_prefix: str = "opt") -> Optional[Dict]:
        if self.current_btc_price <= 0:
            logger.error("❌ RE: No valid BTC price for option pricing.")
            return None
        if not self.pricing_engine:
            logger.error("❌ RE: AdvancedPricingEngine not available.")
            return None

        time_to_expiry_years = expiry_minutes / (60 * 24 * 365.25)
        if time_to_expiry_years <= 1e-7: # Effectively zero or past
             logger.warning(f"RE: Option K={strike_price} TTM={expiry_minutes}min is expired or too short to price.")
             return None

        # Get strike-specific, expiry-adjusted, regime-aware volatility
        # This is the sigma APE's BSM will use internally via its vol_engine instance
        sigma_for_pricing = self.vol_engine.get_expiry_adjusted_volatility(
            expiry_minutes=expiry_minutes,
            strike_price=strike_price,
            underlying_price=self.current_btc_price
        )
        if sigma_for_pricing <= config.MIN_VOLATILITY / 10: # Safety for very low/zero vol
            logger.warning(f"RE: Calculated sigma {sigma_for_pricing:.4f} too low for K={strike_price}. Using MIN_VOLATILITY {config.MIN_VOLATILITY:.4f}.")
            sigma_for_pricing = config.MIN_VOLATILITY

        # Use APE's BSM which includes Greeks
        fair_value_per_unit, greeks_per_unit = self.pricing_engine.black_scholes_with_greeks(
            S=self.current_btc_price, K=strike_price, T=time_to_expiry_years,
            r=config.RISK_FREE_RATE, sigma=sigma_for_pricing, option_type=option_type
        )
        
        platform_price_per_unit = fair_value_per_unit * (1 + self.base_markup_percentage)
        revenue_captured_per_unit = platform_price_per_unit - fair_value_per_unit

        trade_id_full = f"{trade_id_prefix}_{int(time.time()*1000)}_{random.randint(100,999)}"
        current_timestamp = time.time()
        expiry_abs_timestamp = current_timestamp + (expiry_minutes * 60)

        # Data to be sent to PositionManager
        trade_details_for_pm = {
            "trade_id": trade_id_full,
            "timestamp": current_timestamp, # Time of trade
            "underlying_price_at_trade": self.current_btc_price,
            "option_type": option_type,
            "strike": strike_price,
            "expiry_timestamp": expiry_abs_timestamp, # Absolute expiry time
            "quantity": quantity,
            "is_short": True, # Platform is selling this option
            "initial_premium_total": platform_price_per_unit * quantity,
            "volatility_at_trade": sigma_for_pricing, # Actual sigma used
            # initial_delta_total is now calculated by PositionManager based on greeks_per_unit
        }

        if self.position_manager:
            self.position_manager.add_option_trade(trade_details_for_pm)
        else:
            logger.warning("RE: PositionManager N/A. Cannot register new option position.")

        if self.audit_engine:
            self.audit_engine.track_option_premium(
                amount=trade_details_for_pm["initial_premium_total"],
                trade_id=trade_id_full, is_credit=True
            )
            self.audit_engine.track_option_trade_executed(trade_id=trade_id_full)
        
        logger.info(f"💰 RE Sold: {trade_id_full} ({option_type.upper()} K{strike_price} Qty{quantity}). FairVal/u=${fair_value_per_unit:.2f}, PlatPrice/u=${platform_price_per_unit:.2f}, Sigma={sigma_for_pricing:.4f}, InitialOptionDelta/u={greeks_per_unit['delta']:.4f}")

        return { # Data returned by RE for internal use or if BotSim needs it
            "trade_id": trade_id_full, "timestamp": current_timestamp,
            "btc_price_at_trade": self.current_btc_price, "strike_price": strike_price,
            "expiry_minutes": expiry_minutes, "option_type": option_type, "quantity": quantity,
            "bsm_fair_value_per_unit": fair_value_per_unit,
            "platform_price_per_unit": platform_price_per_unit,
            "markup_pct_applied": self.base_markup_percentage * 100,
            "revenue_captured_per_unit": revenue_captured_per_unit,
            "total_gross_premium_trade": platform_price_per_unit * quantity,
            "total_revenue_captured_trade": revenue_captured_per_unit * quantity,
            "volatility_used_for_pricing_pct": sigma_for_pricing * 100, # As percentage
            "initial_greeks_per_unit": greeks_per_unit, # Full greeks dict from APE
            "calculation_method_used": "APE_BS_with_strike_specific_vol"
        }

    def get_current_metrics(self) -> Optional[RevenueMetrics]:
        if self.current_btc_price <= 0:
            logger.warning("⚠️ RE: No BTC price for current metrics.")
            return RevenueMetrics(0,0,0,0,0,0,0,0,0,"no_btc_price_for_metrics")

        atm_strike = round(self.current_btc_price / config.STRIKE_ROUNDING_NEAREST) * config.STRIKE_ROUNDING_NEAREST
        atm_strike = max(atm_strike, config.STRIKE_ROUNDING_NEAREST)
        standard_expiry_minutes = config.STANDARD_METRICS_EXPIRY_MINUTES

        # Use the main pricing method to get consistent details for a standard option
        pricing_info_dict = self.price_option_for_platform_sale(
            strike_price=atm_strike, expiry_minutes=standard_expiry_minutes,
            option_type='call', quantity=1.0, trade_id_prefix="METRICS"
        )

        if not pricing_info_dict:
            logger.error("❌ RE: Failed to get standard pricing info for metrics output.")
            return RevenueMetrics(0,0,0,0,0,0,0,0,0,"pricing_error_for_metrics")
        
        # Use AuditEngine to get 24h contract count if available
        simulated_daily_contracts = config.ESTIMATED_DAILY_CONTRACTS_SOLD # Default
        if self.audit_engine:
            try:
                audit_24h_metrics = self.audit_engine.get_24h_metrics()
                simulated_daily_contracts = audit_24h_metrics.option_trades_executed_24h
            except Exception as e:
                logger.warning(f"RE: Could not get trade count from AuditEngine for daily estimate: {e}")
        
        daily_revenue_estimate = pricing_info_dict["revenue_captured_per_unit"] * simulated_daily_contracts
        
        metrics = RevenueMetrics(
            bsm_fair_value_per_contract=pricing_info_dict["bsm_fair_value_per_unit"],
            platform_price_per_contract=pricing_info_dict["platform_price_per_unit"],
            markup_percentage_applied=pricing_info_dict["markup_pct_applied"],
            revenue_captured_per_contract=pricing_info_dict["revenue_captured_per_unit"],
            daily_revenue_estimate_usd=daily_revenue_estimate,
            simulated_contracts_sold_24h=simulated_daily_contracts,
            average_markup_on_fair_value_pct=self.base_markup_percentage * 100,
            gross_premium_per_contract=pricing_info_dict["platform_price_per_unit"],
            volatility_used_for_pricing_pct=pricing_info_dict["volatility_used_for_pricing_pct"], # Already %
            calculation_method_used=pricing_info_dict["calculation_method_used"]
        )
        if self.debug_mode:
            logger.debug(f"💰 RE Metrics: PlatPrice/u=${metrics.platform_price_per_contract:.2f}, DailyEstRev=${metrics.daily_revenue_estimate_usd:.2f}, VolUsed={metrics.volatility_used_for_pricing_pct:.2f}%")
        return metrics

    def get_debug_info(self) -> Dict:
        vol_metrics = self.vol_engine.get_volatility_metrics()
        return {
            "current_btc_price": self.current_btc_price,
            "last_price_update_ts": self.last_price_update_ts,
            "time_since_last_update_s": time.time() - self.last_price_update_ts if self.last_price_update_ts > 0 else -1,
            "price_update_count": self.price_update_count,
            "base_markup_percentage": self.base_markup_percentage,
            "pricing_engine_type": type(self.pricing_engine).__name__ if self.pricing_engine else "None",
            "volatility_engine_metrics": vol_metrics.__dict__ if hasattr(vol_metrics, '__dict__') else str(vol_metrics),
            "debug_mode_active": self.debug_mode
        }

    def test_volatility_engine(self, test_price: Optional[float] = None) -> Dict:
        # ... (Keep your existing test_volatility_engine method, ensure it uses self.vol_engine)
        # ... Use logger.info for outputs.
        # Example from previous revision:
        effective_test_price = test_price if test_price is not None else (self.current_btc_price if self.current_btc_price > 0 else 108000.0)
        logger.info(f"🧪 RE: Testing Volatility Engine with BTC price: ${effective_test_price:,.2f}")
        self.vol_engine.update_price(effective_test_price) 
        results = {}
        results["simple_historical_vol"] = self.vol_engine.calculate_simple_historical_vol() * 100
        results["ewma_volatility"] = self.vol_engine.calculate_ewma_volatility() * 100
        results["regime"] = str(self.vol_engine.detect_volatility_regime())
        atm_strike = round(effective_test_price / config.STRIKE_ROUNDING_NEAREST) * config.STRIKE_ROUNDING_NEAREST
        results["expiry_adjusted_60m_atm_vol_pct"] = self.vol_engine.get_expiry_adjusted_volatility(60, atm_strike, effective_test_price) * 100
        full_metrics_obj = self.vol_engine.get_volatility_metrics()
        results["full_metrics"] = full_metrics_obj.__dict__ if hasattr(full_metrics_obj, '__dict__') else str(full_metrics_obj)
        logger.info(f"🧪 RE: Volatility Test Results: {results}")
        return results


    def force_price_update(self, btc_price: float) -> Dict:
        # ... (Keep your existing force_price_update method)
        # ... Use logger.info.
        logger.info(f"🔧 RE: Force updating with BTC price: ${btc_price:,.2f}")
        self.update_price(btc_price)
        current_metrics = self.get_current_metrics()
        return {
            **(current_metrics.__dict__ if current_metrics else {}),
            "force_update_timestamp": time.time(),
            "message": f"Price forced to {btc_price}"
        }


    def toggle_debug_mode(self, enabled: Optional[bool] = None) -> Dict:
        # ... (Keep your existing toggle_debug_mode method)
        # ... Use logger.info.
        if enabled is None: self.debug_mode = not self.debug_mode
        else: self.debug_mode = enabled
        logger.info(f"🔧 RE: Debug mode: {'ENABLED' if self.debug_mode else 'DISABLED'}")
        return {"debug_mode_active": self.debug_mode}

