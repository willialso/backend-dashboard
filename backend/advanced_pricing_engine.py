# backend/advanced_pricing_engine.py

import math
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from scipy.stats import norm

from backend import config

logger = logging.getLogger(__name__)

@dataclass
class OptionQuote:
    strike: float
    expiry_minutes: int
    option_type: str
    bid: float
    ask: float
    mid: float
    implied_vol: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    underlying_price: float

class AdvancedPricingEngine:
    def __init__(self,
                 volatility_engine_instance,  # CORRECTED: Accept this parameter
                 alpha_signal_generator_instance: Optional[Any] = None):
        
        self.vol_engine = volatility_engine_instance
        self.alpha_signal_generator = alpha_signal_generator_instance
        self.current_underlying_price: float = 0.0
        self.risk_free_rate: float = config.RISK_FREE_RATE
        
        logger.info("AdvancedPricingEngine initialized with volatility engine integration.")

    def update_market_data(self, underlying_price: float) -> None:
        """Update the current underlying price."""
        if underlying_price > 0:
            self.current_underlying_price = underlying_price
            if hasattr(self.vol_engine, 'update_price'):
                self.vol_engine.update_price(underlying_price)

    @staticmethod
    def black_scholes_with_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Tuple[float, Dict[str, float]]:
        """Calculate Black-Scholes option price and Greeks."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            logger.warning(f"Invalid BS parameters: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
            return 0.0, {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            n_d1 = norm.pdf(d1)
            
            if option_type.lower() == "call":
                price = S * N_d1 - K * math.exp(-r * T) * N_d2
                delta = N_d1
                rho = K * T * math.exp(-r * T) * N_d2
            else:  # put
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                delta = N_d1 - 1
                rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
            
            gamma = n_d1 / (S * sigma * math.sqrt(T))
            theta = (-S * n_d1 * sigma / (2 * math.sqrt(T)) - 
                    r * K * math.exp(-r * T) * N_d2 if option_type.lower() == "call" 
                    else -S * n_d1 * sigma / (2 * math.sqrt(T)) + 
                    r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            vega = S * n_d1 * math.sqrt(T) / 100
            
            greeks = {
                "delta": round(delta, 6),
                "gamma": round(gamma, 8),
                "theta": round(theta, 6),
                "vega": round(vega, 6),
                "rho": round(rho, 6)
            }
            
            return max(0.0, price), greeks
            
        except Exception as e:
            logger.error(f"Black-Scholes calculation error: {e}")
            return 0.0, {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    def price_option(self, strike: float, expiry_minutes: int, option_type: str, 
                    underlying_price: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        """Price a single option using volatility engine."""
        S = underlying_price if underlying_price is not None else self.current_underlying_price
        if S <= 0:
            logger.warning("No valid underlying price for option pricing")
            return 0.0, {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        
        sigma = self.vol_engine.get_expiry_adjusted_volatility(
            expiry_minutes=expiry_minutes,
            strike_price=strike,
            underlying_price=S
        )
        
        T = expiry_minutes / (365.25 * 24 * 60)
        return self.black_scholes_with_greeks(S, strike, T, self.risk_free_rate, sigma, option_type)

    def generate_option_chain(self, expiry_minutes: int, underlying_price: Optional[float] = None,
                            strikes: Optional[List[float]] = None) -> List[OptionQuote]:
        """Generate a full option chain for a given expiry."""
        S = underlying_price if underlying_price is not None else self.current_underlying_price
        if S <= 0:
            logger.warning("No valid underlying price for option chain generation")
            return []
        
        if strikes is None:
            strikes = self._generate_strikes_for_expiry(S, expiry_minutes)
        
        option_chain = []
        
        for strike in strikes:
            for option_type in ["call", "put"]:
                try:
                    price, greeks = self.price_option(strike, expiry_minutes, option_type, S)
                    
                    sigma = self.vol_engine.get_expiry_adjusted_volatility(
                        expiry_minutes=expiry_minutes,
                        strike_price=strike,
                        underlying_price=S
                    )
                    
                    spread_pct = config.get_config_value("OPTION_BID_ASK_SPREAD_PCT", 0.02)
                    spread = price * spread_pct
                    bid = max(0.0, price - spread/2)
                    ask = price + spread/2
                    
                    quote = OptionQuote(
                        strike=strike,
                        expiry_minutes=expiry_minutes,
                        option_type=option_type,
                        bid=round(bid, 2),
                        ask=round(ask, 2),
                        mid=round(price, 2),
                        implied_vol=round(sigma * 100, 2),
                        delta=greeks["delta"],
                        gamma=greeks["gamma"],
                        theta=greeks["theta"],
                        vega=greeks["vega"],
                        rho=greeks["rho"],
                        underlying_price=S
                    )
                    option_chain.append(quote)
                    
                except Exception as e:
                    logger.error(f"Error generating quote for {option_type} K={strike}: {e}")
                    continue
        
        return option_chain

    def _generate_strikes_for_expiry(self, underlying_price: float, expiry_minutes: int) -> List[float]:
        """Generate strikes based on expiry and config."""
        if expiry_minutes not in config.STRIKE_RANGES_BY_EXPIRY:
            closest_expiry = min(config.STRIKE_RANGES_BY_EXPIRY.keys(),
                                key=lambda x: abs(x - expiry_minutes))
            strike_config = config.STRIKE_RANGES_BY_EXPIRY[closest_expiry]
        else:
            strike_config = config.STRIKE_RANGES_BY_EXPIRY[expiry_minutes]
        
        num_strikes_one_side = strike_config["num_one_side"]
        step_pct = strike_config["step_pct"]
        
        strikes = []
        atm_strike = round(underlying_price / config.STRIKE_ROUNDING_NEAREST) * config.STRIKE_ROUNDING_NEAREST
        strikes.append(atm_strike)
        
        for i in range(1, num_strikes_one_side + 1):
            higher_strike = round(underlying_price * (1 + i * step_pct) / config.STRIKE_ROUNDING_NEAREST) * config.STRIKE_ROUNDING_NEAREST
            strikes.append(higher_strike)
            
            lower_strike = round(underlying_price * (1 - i * step_pct) / config.STRIKE_ROUNDING_NEAREST) * config.STRIKE_ROUNDING_NEAREST
            if lower_strike > 0:
                strikes.append(lower_strike)
        
        return sorted(list(set(strikes)))

    def get_pricing_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the pricing engine state."""
        vol_metrics = self.vol_engine.get_volatility_metrics() if self.vol_engine else None
        
        return {
            "current_underlying_price": self.current_underlying_price,
            "risk_free_rate": self.risk_free_rate,
            "volatility_engine_connected": self.vol_engine is not None,
            "volatility_metrics": vol_metrics.__dict__ if vol_metrics else None,
            "alpha_signal_generator_connected": self.alpha_signal_generator is not None
        }
