# investor_dashboard/revenue_engine.py

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
import random

from backend.advanced_pricing_engine import AdvancedPricingEngine, OptionQuote
from backend.advanced_volatility_engine import AdvancedVolatilityEngine, VolatilityMetrics
from backend import config

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
    volatility_used_for_pricing_pct: float
    calculation_method_used: str
    # FIXED: Add fields that dashboard_api.py expects
    daily_revenue_usd: float = 0.0
    platform_markup_pct: float = 0.0

    def __post_init__(self):
        # Ensure dashboard compatibility
        self.daily_revenue_usd = self.daily_revenue_estimate_usd
        self.platform_markup_pct = self.markup_percentage_applied

class RevenueEngine:
    def __init__(self,
                 volatility_engine_instance: AdvancedVolatilityEngine,
                 pricing_engine_instance: AdvancedPricingEngine,
                 position_manager_instance: Optional[Any] = None,
                 audit_engine_instance: Optional[Any] = None):
        
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
        
        # FIXED: Initialize daily revenue tracking
        self.total_premium_volume_24h: float = 0.0
        self.total_trades_24h: int = 0
        self.last_24h_reset: float = time.time()
        
        # CRITICAL: Log audit engine connection status
        if self.audit_engine is not None:
            logger.info(f"✅ Revenue Engine connected to audit engine: {type(self.audit_engine)}")
        else:
            logger.warning("⚠️ Revenue Engine initialized WITHOUT audit engine")
        
        logger.info("✅ Revenue Engine initialized.")

    def update_price(self, btc_price: float):
        if btc_price <= 0: 
            return
        self.current_btc_price = btc_price
        self.last_price_update_ts = time.time()
        self.price_update_count += 1
        
        if self.debug_mode and self.price_update_count % 10 == 0:
            logger.debug(f"💰 RE: Price updated to ${btc_price:,.2f}")

    def _reset_24h_metrics_if_needed(self):
        """Reset 24h metrics if more than 24 hours have passed"""
        now = time.time()
        if now - self.last_24h_reset > 86400:  # 24 hours
            self.total_premium_volume_24h = 0.0
            self.total_trades_24h = 0
            self.last_24h_reset = now

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
        if time_to_expiry_years <= 1e-7:
            logger.warning(f"RE: Option K={strike_price} TTM={expiry_minutes}min is expired or too short to price.")
            return None

        # Get volatility
        try:
            sigma_for_pricing = self.vol_engine.get_expiry_adjusted_volatility(
                expiry_minutes=expiry_minutes,
                strike_price=strike_price,
                underlying_price=self.current_btc_price
            )
        except Exception as e:
            logger.warning(f"RE: Volatility calculation failed: {e}, using default")
            sigma_for_pricing = config.DEFAULT_VOLATILITY

        if sigma_for_pricing <= config.MIN_VOLATILITY / 10:
            logger.warning(f"RE: Calculated sigma {sigma_for_pricing:.4f} too low for K={strike_price}. Using MIN_VOLATILITY {config.MIN_VOLATILITY:.4f}.")
            sigma_for_pricing = config.MIN_VOLATILITY

        # Calculate fair value and Greeks
        try:
            fair_value_per_unit, greeks_per_unit = self.pricing_engine.black_scholes_with_greeks(
                S=self.current_btc_price, K=strike_price, T=time_to_expiry_years,
                r=config.RISK_FREE_RATE, sigma=sigma_for_pricing, option_type=option_type
            )
        except Exception as e:
            logger.error(f"RE: Black-Scholes calculation failed: {e}")
            return None

        # Apply markup
        platform_price_per_unit = fair_value_per_unit * (1 + self.base_markup_percentage)
        revenue_captured_per_unit = platform_price_per_unit - fair_value_per_unit

        # Generate trade ID
        trade_id_full = f"{trade_id_prefix}_{int(time.time()*1000)}_{random.randint(100,999)}"
        current_timestamp = time.time()
        expiry_abs_timestamp = current_timestamp + (expiry_minutes * 60)

        # FIXED: Track revenue for daily calculations
        total_premium = platform_price_per_unit * quantity
        self._reset_24h_metrics_if_needed()
        self.total_premium_volume_24h += total_premium
        self.total_trades_24h += 1

        # Prepare trade details for position manager
        trade_details_for_pm = {
            "trade_id": trade_id_full,
            "timestamp": current_timestamp,
            "underlying_price_at_trade": self.current_btc_price,
            "option_type": option_type,
            "strike": strike_price,
            "expiry_timestamp": expiry_abs_timestamp,
            "quantity": quantity,
            "is_short": True,
            "initial_premium_total": total_premium,
            "volatility_at_trade": sigma_for_pricing,
        }

        # Register with position manager
        if self.position_manager:
            try:
                self.position_manager.add_option_trade(trade_details_for_pm)
            except Exception as e:
                logger.warning(f"RE: Position manager registration failed: {e}")
        else:
            logger.warning("RE: PositionManager N/A. Cannot register new option position.")

        # FIXED: Enhanced audit engine integration with detailed logging
        if self.audit_engine:
            try:
                logger.info(f"RE: Attempting audit tracking for trade {trade_id_full}, premium=${total_premium:.2f}")
                
                if hasattr(self.audit_engine, 'track_option_premium'):
                    self.audit_engine.track_option_premium(
                        amount=total_premium,
                        trade_id=trade_id_full, 
                        is_credit=True
                    )
                    logger.info(f"RE: ✅ Successfully tracked premium ${total_premium:.2f} in audit engine")
                else:
                    logger.error("RE: ❌ audit_engine missing track_option_premium method")
                    
                if hasattr(self.audit_engine, 'track_option_trade_executed'):
                    self.audit_engine.track_option_trade_executed(trade_id=trade_id_full)
                    logger.info(f"RE: ✅ Successfully tracked trade execution {trade_id_full} in audit engine")
                else:
                    logger.error("RE: ❌ audit_engine missing track_option_trade_executed method")
                    
            except Exception as e:
                logger.error(f"RE: ❌ Audit engine registration failed: {e}", exc_info=True)
        else:
            logger.error(f"RE: ❌ audit_engine is None! Cannot track trade {trade_id_full}")

        logger.info(f"💰 RE Sold: {trade_id_full} ({option_type.upper()} K{strike_price} Qty{quantity}). FairVal/u=${fair_value_per_unit:.2f}, PlatPrice/u=${platform_price_per_unit:.2f}, Sigma={sigma_for_pricing:.4f}, InitialOptionDelta/u={greeks_per_unit.get('delta', 0):.4f}")

        # FIXED: Return comprehensive data with all expected fields
        return {
            "trade_id": trade_id_full,
            "timestamp": current_timestamp,
            "btc_price_at_trade": self.current_btc_price,
            "strike_price": strike_price,
            "expiry_minutes": expiry_minutes,
            "option_type": option_type,
            "quantity": quantity,
            "fair_value": fair_value_per_unit,
            "platform_price": platform_price_per_unit,
            "platform_price_per_contract": platform_price_per_unit,  # Alias for bot_trader_simulator
            "markup_pct": self.base_markup_percentage * 100,
            "platform_profit": revenue_captured_per_unit * quantity,
            "delta": greeks_per_unit.get('delta', 0),
            "gamma": greeks_per_unit.get('gamma', 0),
            "theta": greeks_per_unit.get('theta', 0),
            "vega": greeks_per_unit.get('vega', 0),
            "iv": sigma_for_pricing,
            "bsm_fair_value_per_unit": fair_value_per_unit,
            "platform_price_per_unit": platform_price_per_unit,
            "markup_pct_applied": self.base_markup_percentage * 100,
            "revenue_captured_per_unit": revenue_captured_per_unit,
            "total_gross_premium_trade": total_premium,
            "total_revenue_captured_trade": revenue_captured_per_unit * quantity,
            "volatility_used_for_pricing_pct": sigma_for_pricing * 100,
            "initial_greeks_per_unit": greeks_per_unit,
            "calculation_method_used": "APE_BS_with_strike_specific_vol"
        }

    def get_current_metrics(self) -> Optional[RevenueMetrics]:
        """FIXED: Calculate real revenue metrics from actual trading data"""
        try:
            if self.current_btc_price <= 0:
                logger.warning("⚠️ RE: No BTC price for current metrics.")
                # Return default metrics instead of None
                return RevenueMetrics(
                    bsm_fair_value_per_contract=0.0,
                    platform_price_per_contract=0.0,
                    markup_percentage_applied=self.base_markup_percentage * 100,
                    revenue_captured_per_contract=0.0,
                    daily_revenue_estimate_usd=0.0,
                    simulated_contracts_sold_24h=0,
                    average_markup_on_fair_value_pct=self.base_markup_percentage * 100,
                    gross_premium_per_contract=0.0,
                    volatility_used_for_pricing_pct=config.DEFAULT_VOLATILITY * 100,
                    calculation_method_used="no_btc_price_for_metrics"
                )

            # Reset 24h metrics if needed
            self._reset_24h_metrics_if_needed()

            # Get standard pricing for metrics calculation
            atm_strike = round(self.current_btc_price / config.STRIKE_ROUNDING_NEAREST) * config.STRIKE_ROUNDING_NEAREST
            atm_strike = max(atm_strike, config.STRIKE_ROUNDING_NEAREST)
            standard_expiry_minutes = config.STANDARD_METRICS_EXPIRY_MINUTES

            # Use internal pricing calculation without recording trade
            time_to_expiry_years = standard_expiry_minutes / (60 * 24 * 365.25)
            
            try:
                sigma_for_pricing = self.vol_engine.get_expiry_adjusted_volatility(
                    expiry_minutes=standard_expiry_minutes,
                    strike_price=atm_strike,
                    underlying_price=self.current_btc_price
                )
            except Exception as e:
                logger.warning(f"RE: Volatility calculation failed for metrics: {e}")
                sigma_for_pricing = config.DEFAULT_VOLATILITY

            try:
                fair_value_per_unit, greeks_per_unit = self.pricing_engine.black_scholes_with_greeks(
                    S=self.current_btc_price, K=atm_strike, T=time_to_expiry_years,
                    r=config.RISK_FREE_RATE, sigma=sigma_for_pricing, option_type='call'
                )
            except Exception as e:
                logger.warning(f"RE: BS calculation failed for metrics: {e}")
                fair_value_per_unit = 100.0  # Fallback
                
            platform_price_per_unit = fair_value_per_unit * (1 + self.base_markup_percentage)
            revenue_captured_per_unit = platform_price_per_unit - fair_value_per_unit

            # Calculate daily revenue from actual trading activity
            daily_revenue_estimate = self.total_premium_volume_24h * (self.base_markup_percentage / (1 + self.base_markup_percentage))
            
            # Use actual trade count or estimated
            contracts_sold_24h = max(self.total_trades_24h, config.ESTIMATED_DAILY_CONTRACTS_SOLD)

            # Create metrics object
            metrics = RevenueMetrics(
                bsm_fair_value_per_contract=fair_value_per_unit,
                platform_price_per_contract=platform_price_per_unit,
                markup_percentage_applied=self.base_markup_percentage * 100,
                revenue_captured_per_contract=revenue_captured_per_unit,
                daily_revenue_estimate_usd=daily_revenue_estimate,
                simulated_contracts_sold_24h=contracts_sold_24h,
                average_markup_on_fair_value_pct=self.base_markup_percentage * 100,
                gross_premium_per_contract=platform_price_per_unit,
                volatility_used_for_pricing_pct=sigma_for_pricing * 100,
                calculation_method_used="live_trading_data_based"
            )

            if self.debug_mode:
                logger.debug(f"💰 RE Metrics: PlatPrice/u=${metrics.platform_price_per_contract:.2f}, DailyRev=${metrics.daily_revenue_usd:.2f}, Vol24h=${self.total_premium_volume_24h:.2f}")

            return metrics

        except Exception as e:
            logger.error(f"RE: Error calculating metrics: {e}", exc_info=True)
            # Return default metrics on error
            return RevenueMetrics(
                bsm_fair_value_per_contract=0.0,
                platform_price_per_contract=0.0,
                markup_percentage_applied=self.base_markup_percentage * 100,
                revenue_captured_per_contract=0.0,
                daily_revenue_estimate_usd=0.0,
                simulated_contracts_sold_24h=0,
                average_markup_on_fair_value_pct=self.base_markup_percentage * 100,
                gross_premium_per_contract=0.0,
                volatility_used_for_pricing_pct=config.DEFAULT_VOLATILITY * 100,
                calculation_method_used="error_fallback"
            )

    def get_debug_info(self) -> Dict:
        """Get debugging information about the revenue engine state"""
        try:
            vol_metrics = self.vol_engine.get_volatility_metrics()
        except Exception as e:
            vol_metrics = f"Error: {e}"
            
        return {
            "current_btc_price": self.current_btc_price,
            "last_price_update_ts": self.last_price_update_ts,
            "time_since_last_update_s": time.time() - self.last_price_update_ts if self.last_price_update_ts > 0 else -1,
            "price_update_count": self.price_update_count,
            "base_markup_percentage": self.base_markup_percentage,
            "total_premium_volume_24h": self.total_premium_volume_24h,
            "total_trades_24h": self.total_trades_24h,
            "audit_engine_connected": self.audit_engine is not None,
            "audit_engine_type": type(self.audit_engine).__name__ if self.audit_engine else "None",
            "pricing_engine_type": type(self.pricing_engine).__name__ if self.pricing_engine else "None",
            "volatility_engine_metrics": vol_metrics.__dict__ if hasattr(vol_metrics, '__dict__') else str(vol_metrics),
            "debug_mode_active": self.debug_mode
        }

    def test_volatility_engine(self, test_price: Optional[float] = None) -> Dict:
        """Test volatility engine functionality"""
        effective_test_price = test_price if test_price is not None else (self.current_btc_price if self.current_btc_price > 0 else 108000.0)
        
        logger.info(f"🧪 RE: Testing Volatility Engine with BTC price: ${effective_test_price:,.2f}")
        
        try:
            self.vol_engine.update_price(effective_test_price)
            
            results = {
                "test_price": effective_test_price,
                "simple_historical_vol": self.vol_engine.calculate_simple_historical_vol() * 100,
                "ewma_volatility": self.vol_engine.calculate_ewma_volatility() * 100,
                "regime": str(self.vol_engine.detect_volatility_regime()),
            }
            
            atm_strike = round(effective_test_price / config.STRIKE_ROUNDING_NEAREST) * config.STRIKE_ROUNDING_NEAREST
            results["expiry_adjusted_60m_atm_vol_pct"] = self.vol_engine.get_expiry_adjusted_volatility(60, atm_strike, effective_test_price) * 100
            
            full_metrics_obj = self.vol_engine.get_volatility_metrics()
            results["full_metrics"] = full_metrics_obj.__dict__ if hasattr(full_metrics_obj, '__dict__') else str(full_metrics_obj)
            
            logger.info(f"🧪 RE: Volatility Test Results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"RE: Volatility test failed: {e}")
            return {"error": str(e), "test_price": effective_test_price}

    def force_price_update(self, btc_price: float) -> Dict:
        """Force update BTC price and return current metrics"""
        logger.info(f"🔧 RE: Force updating with BTC price: ${btc_price:,.2f}")
        
        self.update_price(btc_price)
        current_metrics = self.get_current_metrics()
        
        return {
            **(current_metrics.__dict__ if current_metrics else {}),
            "force_update_timestamp": time.time(),
            "message": f"Price forced to {btc_price}"
        }

    def toggle_debug_mode(self, enabled: Optional[bool] = None) -> Dict:
        """Toggle debug mode on/off"""
        if enabled is None: 
            self.debug_mode = not self.debug_mode
        else: 
            self.debug_mode = enabled
            
        logger.info(f"🔧 RE: Debug mode: {'ENABLED' if self.debug_mode else 'DISABLED'}")
        return {"debug_mode_active": self.debug_mode}
