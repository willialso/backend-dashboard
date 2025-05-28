# backend/advanced_pricing_engine.py

import numpy as np
import pandas as pd
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from scipy.stats import norm

from backend import config
from backend.volatility_engine import AdvancedVolatilityEngine # Assuming this path is correct
from backend.alpha_signals import AlphaSignalGenerator # Assuming this path is correct
from backend.utils import setup_logger # Assuming this path is correct

logger = setup_logger(__name__)

# --- Data Classes for Option Quotes and Chains ---

@dataclass
class OptionQuote:
    symbol: str
    option_type: str # "call" or "put"
    strike: float
    expiry_minutes: int
    expiry_label: str
    premium_usd: float # Premium per contract
    premium_btc: float # Premium in BTC terms for the contract
    delta: float # Scaled for contract size
    gamma: float # Scaled for contract size
    theta: float # Scaled for contract size (per day, USD value change)
    vega: float # Scaled for contract size (per 1% vol change, USD value change)
    rho: float  # Scaled for contract size (per 1% rate change, USD value change)
    implied_vol: float # Annualized, strike-specific volatility used for this quote's calculation
    moneyness: str # "ITM", "ATM", "OTM"

    def dict(self):
        return asdict(self)

@dataclass
class OptionChain:
    underlying_price: float
    timestamp: float
    expiry_minutes: int
    expiry_label: str
    calls: List[OptionQuote]
    puts: List[OptionQuote]
    volatility_used: float # Represents ATM volatility for this expiry
    alpha_adjustment_applied: bool

    def dict(self):
        return {
            "underlying_price": self.underlying_price,
            "timestamp": self.timestamp,
            "expiry_minutes": self.expiry_minutes,
            "expiry_label": self.expiry_label,
            "calls": [c.dict() for c in self.calls],
            "puts": [p.dict() for p in self.puts],
            "volatility_used": self.volatility_used,
            "alpha_adjustment_applied": self.alpha_adjustment_applied
        }

# --- Main Pricing Engine Class ---

class AdvancedPricingEngine:
    """
    Advanced Black-Scholes pricing engine with critical fixes for mini-contracts,
    Rho calculation, and integration with strike-specific volatility.
    """
    def __init__(self, volatility_engine: AdvancedVolatilityEngine, alpha_signal_generator: AlphaSignalGenerator):
        self.vol_engine = volatility_engine
        self.alpha_generator = alpha_signal_generator
        self.current_price = 0.0
        logger.info("AdvancedPricingEngine initialized with CRITICAL FIXES, Rho enhancement, and strike-specific volatility.")

    def update_market_data(self, price: float, volume: float = 0) -> None:
        """Updates the engine with the latest market price and volume."""
        logger.info(f"APE: update_market_data called with price: {price}, volume: {volume}") # DIAGNOSTIC LOG
        if price <= 0:
            logger.warning(f"APE: Invalid price received in update_market_data: {price}. Ignoring.")
            return

        self.current_price = price
        if hasattr(self.vol_engine, 'update_price') and callable(getattr(self.vol_engine, 'update_price')):
            self.vol_engine.update_price(price) # Pass price to vol engine if it needs it
        if hasattr(self.alpha_generator, 'update_tick') and callable(getattr(self.alpha_generator, 'update_tick')):
            self.alpha_generator.update_tick(price, volume)

    @staticmethod
    def black_scholes_with_greeks(S: float, K: float, T: float, r: float,
                                  sigma: float, option_type: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculates Black-Scholes option price and Greeks (Delta, Gamma, Theta, Vega, Rho).
        *** ENHANCED with Rho and robust error handling ***
        """
        default_greeks = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

        if S <= 0 or K <= 0 or T <= 0:
            price_at_extremes = 0.0
            if T > 0:
                if option_type.lower() == "call":
                    price_at_extremes = max(0, S - K * math.exp(-r * T)) if S > 0 else 0
                    default_greeks["delta"] = 1.0 if S > K else 0.0
                elif option_type.lower() == "put":
                    price_at_extremes = max(0, K * math.exp(-r * T) - S) if K > 0 else 0
                    default_greeks["delta"] = -1.0 if K > S else 0.0
            return price_at_extremes, default_greeks

        if sigma <= 1e-6: sigma = 1e-4

        try:
            if T < 1e-9: T = 1e-9

            d1_numerator = math.log(S / K) + (r + 0.5 * sigma ** 2) * T
            d1_denominator = sigma * math.sqrt(T)

            if abs(d1_denominator) < 1e-9:
                price_at_extremes = 0.0
                delta_extreme = 0.0
                if option_type.lower() == "call":
                    price_at_extremes = max(0, S - K * math.exp(-r * T))
                    if S >= K: delta_extreme = 1.0
                elif option_type.lower() == "put":
                    price_at_extremes = max(0, K * math.exp(-r * T) - S)
                    if K >= S: delta_extreme = -1.0
                return price_at_extremes, {"delta": delta_extreme, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

            d1 = d1_numerator / d1_denominator
            d2 = d1 - sigma * math.sqrt(T)

            if option_type.lower() == "call":
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                delta = norm.cdf(d1)
                rho_annual = K * T * math.exp(-r * T) * norm.cdf(d2)
            elif option_type.lower() == "put":
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                delta = norm.cdf(d1) - 1
                rho_annual = -K * T * math.exp(-r * T) * norm.cdf(-d2)
            else:
                raise ValueError("option_type must be 'call' or 'put'")

            gamma_val = norm.pdf(d1) / (S * sigma * math.sqrt(T)) if S > 0 and sigma > 0 and T > 0 else 0.0
            theta_term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) if T > 0 and sigma > 0 else 0.0
            theta_term2 = -r * K * math.exp(-r * T) * norm.cdf(d2) if option_type.lower() == "call" else r * K * math.exp(-r * T) * norm.cdf(-d2)
            theta_annual = theta_term1 + theta_term2
            theta_per_day = theta_annual / 365.25
            vega_val = S * norm.pdf(d1) * math.sqrt(T) if T > 0 else 0.0
            vega_per_1_pct_vol_change = vega_val / 100.0
            rho_per_1_pct_rate_change = rho_annual / 100.0

            greeks = {"delta": delta, "gamma": gamma_val, "theta": theta_per_day, "vega": vega_per_1_pct_vol_change, "rho": rho_per_1_pct_rate_change}
            return max(price, 1e-8), greeks
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            logger.error(f"BS calc error: S={S}, K={K}, T={T}, r={r}, sigma={sigma}, type={option_type}: {e}", exc_info=True)
            return 1e-8, default_greeks

    def generate_strikes_for_expiry(self, expiry_minutes: int) -> List[float]:
        logger.info(f"APE: generate_strikes_for_expiry called for {expiry_minutes}min. Current price: {self.current_price}") # DIAGNOSTIC LOG
        if self.current_price <= 0:
            logger.warning(f"APE: Cannot generate strikes for {expiry_minutes}min: current_price is invalid ({self.current_price}).")
            return []

        strike_params_config = config.STRIKE_RANGES_BY_EXPIRY
        strike_params = strike_params_config.get(expiry_minutes)
        default_fallback_expiry = 15

        if strike_params is None:
            logger.warning(f"Strike params for expiry {expiry_minutes}min not found. Using fallback for {default_fallback_expiry}min.")
            strike_params = strike_params_config.get(default_fallback_expiry)
        if strike_params is None:
            logger.error(f"No strike params for {expiry_minutes}min or fallback {default_fallback_expiry}min.")
            return []

        num_itm_strikes = strike_params["num_itm"]
        num_otm_strikes = strike_params["num_otm"]
        step_percentage = strike_params["step_pct"]
        rounding_val = config.STRIKE_ROUNDING_NEAREST
        generated_strikes = set()

        atm_strike_price = round(self.current_price / rounding_val) * rounding_val
        if atm_strike_price <= 0: atm_strike_price = rounding_val
        generated_strikes.add(atm_strike_price)

        actual_step_value = max(atm_strike_price * step_percentage, rounding_val * step_percentage)
        if actual_step_value <= 0: actual_step_value = rounding_val * 0.01

        for i in range(1, num_itm_strikes + 1):
            itm_k_val = round((atm_strike_price - i * actual_step_value) / rounding_val) * rounding_val
            if itm_k_val > 0: generated_strikes.add(itm_k_val)
        for i in range(1, num_otm_strikes + 1):
            otm_k_val = round((atm_strike_price + i * actual_step_value) / rounding_val) * rounding_val
            if otm_k_val > 0 : generated_strikes.add(otm_k_val) # Ensure OTM strikes are positive too

        positive_strikes = sorted([s for s in generated_strikes if s > 0])
        if not positive_strikes:
            logger.warning(f"APE: No positive strikes generated for {expiry_minutes}min with current price {self.current_price}.")
        return positive_strikes

    def classify_moneyness(self, strike: float, option_type: str) -> str:
        if self.current_price <= 0: return "N/A"
        atm_threshold_percentage = 0.005
        lower_atm_bound = self.current_price * (1 - atm_threshold_percentage)
        upper_atm_bound = self.current_price * (1 + atm_threshold_percentage)
        if option_type.lower() == "call":
            if strike < lower_atm_bound: return "ITM"
            elif strike > upper_atm_bound: return "OTM"
            else: return "ATM"
        elif option_type.lower() == "put":
            if strike > upper_atm_bound: return "ITM"
            elif strike < lower_atm_bound: return "OTM"
            else: return "ATM"
        return "N/A"

    def apply_alpha_adjustment(self, base_premium_usd_on_contract: float, option_type: str,
                               moneyness_status: str, expiry_minutes: int) -> Tuple[float, float]:
        if not config.ALPHA_SIGNALS_ENABLED: return base_premium_usd_on_contract, 0.0
        try:
            primary_alpha_signal = self.alpha_generator.generate_primary_signal()
            if not (primary_alpha_signal and hasattr(primary_alpha_signal, 'value') and hasattr(primary_alpha_signal, 'confidence')):
                return base_premium_usd_on_contract, 0.0
            base_adjustment_percentage = 0.05
            signal_confidence_weight = primary_alpha_signal.confidence
            signal_value_effect = primary_alpha_signal.value if option_type.lower() == "call" else -primary_alpha_signal.value
            total_adjustment_factor = signal_value_effect * signal_confidence_weight * base_adjustment_percentage
            if moneyness_status == "OTM": total_adjustment_factor *= 1.5
            elif moneyness_status == "ATM": total_adjustment_factor *= 1.2
            adjusted_premium_val = base_premium_usd_on_contract * (1 + total_adjustment_factor)
            min_floor_price = 1e-5 * config.STANDARD_CONTRACT_SIZE_BTC * self.current_price if self.current_price > 0 else 1e-5
            min_premium_floor_val = max(base_premium_usd_on_contract * 0.5, min_floor_price)
            final_adjusted_premium_val = max(adjusted_premium_val, min_premium_floor_val)
            actual_adj_factor = (final_adjusted_premium_val / base_premium_usd_on_contract) - 1 if base_premium_usd_on_contract > 1e-9 else 0.0
            return final_adjusted_premium_val, actual_adj_factor
        except Exception as e_alpha:
            logger.error(f"Alpha adjustment error: {e_alpha}", exc_info=True)
            return base_premium_usd_on_contract, 0.0

    def generate_option_chain(self, expiry_minutes: int) -> Optional[OptionChain]:
        logger.info(f"APE: generate_option_chain START for {expiry_minutes}min. Current price: {self.current_price}") # DIAGNOSTIC LOG
        if self.current_price <= 0:
            logger.warning(f"APE: Cannot generate chain for {expiry_minutes}min: No valid current_price ({self.current_price}). Returning None.")
            return None

        try:
            # Calculate ATM volatility for this expiry (to be stored in OptionChain.volatility_used)
            # This assumes vol_engine can handle current_price being strike for ATM calculation.
            atm_volatility_for_chain = self.vol_engine.get_expiry_adjusted_volatility(
                expiry_minutes=expiry_minutes,
                strike_price=self.current_price, # Use current price as strike for ATM vol
                underlying_price=self.current_price
            )
            atm_volatility_for_chain = max(config.MIN_VOLATILITY, min(atm_volatility_for_chain, config.MAX_VOLATILITY))
            logger.info(f"APE: For {expiry_minutes}min, ATM Volatility for chain: {atm_volatility_for_chain:.4f}") # DIAGNOSTIC LOG

            strike_prices_list = self.generate_strikes_for_expiry(expiry_minutes)
            logger.info(f"APE: For {expiry_minutes}min, generated strikes: {strike_prices_list}") # DIAGNOSTIC LOG
            if not strike_prices_list:
                logger.warning(f"APE: No strikes generated for {expiry_minutes}min. Returning None.")
                return None

            time_to_expiry_years = expiry_minutes / (60 * 24 * 365.25)
            call_quotes_list: List[OptionQuote] = []
            put_quotes_list: List[OptionQuote] = []
            any_alpha_adjustment_applied_in_chain = False

            for K_strike in strike_prices_list:
                # *** CRITICAL FIX: Get strike-specific volatility ***
                strike_specific_sigma = self.vol_engine.get_expiry_adjusted_volatility(
                    expiry_minutes=expiry_minutes,
                    strike_price=K_strike,
                    underlying_price=self.current_price
                )
                # Clamp volatility to configured min/max
                strike_specific_sigma = max(config.MIN_VOLATILITY, min(strike_specific_sigma, config.MAX_VOLATILITY))
                logger.debug(f"APE: For K={K_strike}, {expiry_minutes}min, Strike-Specific Sigma: {strike_specific_sigma:.4f}") # DIAGNOSTIC LOG

                for option_contract_type in ["call", "put"]:
                    base_premium_per_unit, greeks_per_unit = self.black_scholes_with_greeks(
                        S=self.current_price, K=K_strike, T=time_to_expiry_years,
                        r=config.RISK_FREE_RATE, sigma=strike_specific_sigma, option_type=option_contract_type
                    )

                    intrinsic_value_per_unit = max(0, self.current_price - K_strike) if option_contract_type == "call" else max(0, K_strike - self.current_price)
                    if base_premium_per_unit < intrinsic_value_per_unit:
                        logger.debug(f"BS premium ${base_premium_per_unit:.6f} < intrinsic ${intrinsic_value_per_unit:.6f} for {K_strike} {option_contract_type}. Adjusting.")
                        base_premium_per_unit = intrinsic_value_per_unit
                    
                    base_premium_usd_for_contract = base_premium_per_unit * config.STANDARD_CONTRACT_SIZE_BTC
                    option_moneyness = self.classify_moneyness(K_strike, option_contract_type)
                    adjusted_premium_usd_for_contract, alpha_adj_factor = self.apply_alpha_adjustment(
                        base_premium_usd_for_contract, option_contract_type, option_moneyness, expiry_minutes
                    )
                    if abs(alpha_adj_factor) > 1e-6: any_alpha_adjustment_applied_in_chain = True

                    final_premium_btc_for_contract = adjusted_premium_usd_for_contract / self.current_price if self.current_price > 0 else 0.0
                    
                    scaled_greeks_values = {
                        name: val * config.STANDARD_CONTRACT_SIZE_BTC for name, val in greeks_per_unit.items()
                    }
                    
                    if option_moneyness == "ITM":
                        if option_contract_type == "call":
                            min_delta_h = 0.7 * config.STANDARD_CONTRACT_SIZE_BTC
                            scaled_greeks_values["delta"] = max(scaled_greeks_values["delta"], min_delta_h)
                        else:
                            max_delta_h = -0.7 * config.STANDARD_CONTRACT_SIZE_BTC
                            scaled_greeks_values["delta"] = min(scaled_greeks_values["delta"], max_delta_h)

                    logger.debug(f"Strike {K_strike} {option_contract_type.upper()}: PremPU=${base_premium_per_unit:.4f} AdjPremCont=${adjusted_premium_usd_for_contract:.2f} DeltaCont={scaled_greeks_values['delta']:.4f} ({option_moneyness}) IV={strike_specific_sigma:.4f}")

                    option_quote_obj = OptionQuote(
                        symbol=f"BTC-{config.EXPIRY_LABELS.get(expiry_minutes, str(expiry_minutes)+'M')}-{int(K_strike)}-{option_contract_type[0].upper()}",
                        option_type=option_contract_type, strike=K_strike, expiry_minutes=expiry_minutes,
                        expiry_label=config.EXPIRY_LABELS.get(expiry_minutes, f"{expiry_minutes}min"),
                        premium_usd=round(adjusted_premium_usd_for_contract, 2),
                        premium_btc=round(final_premium_btc_for_contract, 8),
                        delta=round(scaled_greeks_values["delta"], 4), gamma=round(scaled_greeks_values["gamma"], 6),
                        theta=round(scaled_greeks_values["theta"], 4), vega=round(scaled_greeks_values["vega"], 4),
                        rho=round(scaled_greeks_values["rho"], 4),
                        implied_vol=strike_specific_sigma, # Store the strike-specific sigma
                        moneyness=option_moneyness
                    )
                    if option_contract_type == "call": call_quotes_list.append(option_quote_obj)
                    else: put_quotes_list.append(option_quote_obj)
            
            call_quotes_list.sort(key=lambda q: q.strike)
            put_quotes_list.sort(key=lambda q: q.strike)

            if not call_quotes_list and not put_quotes_list:
                logger.warning(f"APE: For {expiry_minutes}min, no call or put quotes were generated despite having strikes. Returning None.")
                return None

            logger.info(f"APE: Successfully generated option chain for {expiry_minutes}min with {len(call_quotes_list)} calls and {len(put_quotes_list)} puts.")
            return OptionChain(
                underlying_price=self.current_price,
                timestamp=pd.Timestamp.now(tz='UTC').timestamp(),
                expiry_minutes=expiry_minutes,
                expiry_label=config.EXPIRY_LABELS.get(expiry_minutes, f"{expiry_minutes}min"),
                calls=call_quotes_list, puts=put_quotes_list,
                volatility_used=atm_volatility_for_chain, # Store the calculated ATM vol for the chain
                alpha_adjustment_applied=(config.ALPHA_SIGNALS_ENABLED and any_alpha_adjustment_applied_in_chain)
            )
        except Exception as e_chain_gen:
            logger.error(f"APE: Option chain generation CRITICAL error for {expiry_minutes}min: {e_chain_gen}", exc_info=True)
            return None

    def generate_all_chains(self) -> Dict[int, Optional[OptionChain]]:
        all_generated_chains: Dict[int, Optional[OptionChain]] = {}
        for expiry_duration_minutes in config.AVAILABLE_EXPIRIES_MINUTES:
            chain = self.generate_option_chain(expiry_duration_minutes)
            all_generated_chains[expiry_duration_minutes] = chain
            if not chain:
                logger.warning(f"APE: Failed to generate chain for expiry in ALL_CHAINS loop: {expiry_duration_minutes} minutes.")
        return all_generated_chains

