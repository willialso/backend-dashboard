# backend/volatility_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

from backend import config # Ensure this imports your updated config
from backend.utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class VolatilityMetrics:
    current_vol: float # Simple historical volatility (annualized)
    regime_vol: float  # ATM volatility for a standard expiry (e.g., 1hr), adjusted for regime and time
    ewma_vol: float    # EWMA volatility (annualized, typically ATM)
    garch_vol: Optional[float] # Placeholder for future GARCH model
    confidence: float  # Confidence in the calculated volatilities (0.0 to 1.0)
    regime: str        # Detected volatility regime ("low", "medium", "high")

class AdvancedVolatilityEngine:
    def __init__(self):
        self.price_history: List[float] = []
        self.return_history: List[float] = []
        self.regime_history: List[str] = [] # Not strictly used by current logic but good for tracking
        self.last_regime = "medium" # Store last detected regime
        logger.info("AdvancedVolatilityEngine initialized with smile/skew modeling capabilities.")

    def update_price(self, price: float) -> None:
        """Update with new price and calculate returns."""
        if price <= 0:
            logger.debug(f"Invalid price received in update_price: {price}. Ignoring.")
            return

        self.price_history.append(price)

        if len(self.price_history) >= 2:
            # Use log returns for better properties
            log_return = math.log(price / self.price_history[-2])
            self.return_history.append(log_return)

        # Limit history size as defined in config
        max_points = config.PRICE_HISTORY_MAX_POINTS
        if len(self.price_history) > max_points:
            self.price_history = self.price_history[-max_points:]
        if len(self.return_history) > max_points: # Ensure return history is also capped
            self.return_history = self.return_history[-max_points:]

    def calculate_simple_historical_vol(self, periods: Optional[int] = None) -> float:
        """Calculate simple historical volatility."""
        if len(self.return_history) < 20: # Need sufficient data
            logger.debug("Not enough return data for historical vol, using default.")
            return config.DEFAULT_VOLATILITY

        if periods is None:
            # Default to a reasonable lookback, e.g., 100 periods or all available if less
            periods = min(len(self.return_history), 100)
        
        # Ensure periods requested does not exceed available data
        actual_periods = min(periods, len(self.return_history))
        if actual_periods < 2: # Need at least 2 returns for std dev
            return config.DEFAULT_VOLATILITY
            
        returns_subset = np.array(self.return_history[-actual_periods:])

        sampling_freq_seconds = config.DATA_BROADCAST_INTERVAL_SECONDS
        periods_per_year = (365 * 24 * 60 * 60) / sampling_freq_seconds
        annualization_factor = math.sqrt(periods_per_year)

        vol = np.std(returns_subset) * annualization_factor
        return max(vol, config.MIN_VOLATILITY) # Apply floor

    def calculate_ewma_volatility(self, alpha: Optional[float] = None) -> float:
        """Calculate Exponentially Weighted Moving Average (EWMA) volatility."""
        if alpha is None:
            alpha = config.VOLATILITY_EWMA_ALPHA

        if len(self.return_history) < 10: # Need some data for EWMA
            logger.debug("Not enough return data for EWMA vol, falling back to simple historical.")
            return self.calculate_simple_historical_vol()

        returns = np.array(self.return_history)
        
        # EWMA variance calculation (more direct method)
        # sigma_sq_t = alpha * r_t-1^2 + (1-alpha) * sigma_sq_t-1
        # Here, we use a weighted average of squared returns (assuming mean return is close to 0 for high-freq)
        num_returns = len(returns)
        weights = np.array([(1 - alpha) ** i for i in range(num_returns)])
        weights = weights[::-1] # Reverse to give more weight to recent returns
        weights /= np.sum(weights) # Normalize weights

        # Weighted average of squared returns (approximates variance if mean return is small)
        # For financial returns, often E[r] is assumed small for vol calcs
        ewma_variance = np.average(returns**2, weights=weights)

        sampling_freq_seconds = config.DATA_BROADCAST_INTERVAL_SECONDS
        periods_per_year = (365 * 24 * 60 * 60) / sampling_freq_seconds
        annualized_ewma_vol = math.sqrt(ewma_variance * periods_per_year)
        
        return max(annualized_ewma_vol, config.MIN_VOLATILITY) # Apply floor

    def detect_volatility_regime(self) -> str:
        """Detect current volatility regime using recent vol vs long-term vol."""
        if len(self.return_history) < 50: # Need sufficient data for short and long term vol
            logger.debug("Not enough data for regime detection, defaulting to medium.")
            return "medium" # Default regime

        # Using shorter and longer lookbacks for comparison
        short_vol = self.calculate_simple_historical_vol(periods=20) 
        long_vol = self.calculate_simple_historical_vol(periods=100)

        if long_vol < 1e-6: # Avoid division by zero if long_vol is extremely small
            logger.warning("Long-term volatility is near zero, cannot determine regime accurately. Defaulting to medium.")
            return "medium"

        ratio = short_vol / long_vol

        if ratio > 1 + (config.VOLATILITY_REGIME_MULTIPLIER_HIGH -1) * 0.75 : # e.g. > 1.1875 if HIGH is 1.25
            regime = "high"
        elif ratio < 1 - (1- config.VOLATILITY_REGIME_MULTIPLIER_LOW) * 0.75: # e.g. < 0.8875 if LOW is 0.85
            regime = "low"
        else:
            regime = "medium"
        
        self.last_regime = regime # Store for potential future use or logging
        return regime

    def get_expiry_adjusted_volatility(self, expiry_minutes: int, strike_price: float, underlying_price: float) -> float:
        """
        Calculate strike-specific volatility adjusted for expiry, regime, and smile/skew.
        This is the primary method called by the pricing engine.
        """

        # 1. Calculate base At-The-Money (ATM) volatility (e.g., EWMA)
        base_atm_vol = self.calculate_ewma_volatility()

        # 2. Apply regime adjustment to the base ATM volatility
        regime = self.detect_volatility_regime()
        regime_multipliers = {
            "low": config.VOLATILITY_REGIME_MULTIPLIER_LOW,
            "medium": config.VOLATILITY_REGIME_MULTIPLIER_MEDIUM,
            "high": config.VOLATILITY_REGIME_MULTIPLIER_HIGH
        }
        atm_vol_regime_adjusted = base_atm_vol * regime_multipliers.get(regime, config.VOLATILITY_REGIME_MULTIPLIER_MEDIUM)

        # 3. Apply time-based adjustments for very short expiries to the ATM volatility
        # This typically boosts volatility for options very close to expiry.
        atm_vol_for_expiry = atm_vol_regime_adjusted
        short_term_multiplier = config.VOLATILITY_SHORT_EXPIRY_ADJUSTMENTS.get(
            expiry_minutes, config.VOLATILITY_DEFAULT_SHORT_EXPIRY_ADJUSTMENT
        )
        atm_vol_for_expiry *= short_term_multiplier
        
        # Clamp ATM volatility for this expiry to be within global bounds before applying smile.
        # This ensures the basis for the smile is itself reasonable.
        atm_vol_for_expiry = max(min(atm_vol_for_expiry, config.MAX_VOLATILITY), config.MIN_VOLATILITY)

        # 4. Calculate strike-specific adjustments (smile/skew)
        strike_specific_vol = atm_vol_for_expiry # Default to ATM vol if moneyness cannot be calculated reliably

        log_moneyness = 0.0 # Default for ATM or problematic inputs
        adjustment_factor = 1.0 # Default no adjustment
        bounded_adjustment_factor = 1.0

        if underlying_price > 1e-6 and strike_price > 1e-6: # Ensure prices are positive and non-trivial
            # Using a small tolerance for ATM to prevent float precision issues with log(1)
            if abs(strike_price - underlying_price) / underlying_price < 1e-5: # Considered ATM
                log_moneyness = 0.0
            else:
                log_moneyness = math.log(strike_price / underlying_price)

            smile_component = config.VOLATILITY_SMILE_CURVATURE * (log_moneyness ** 2)
            skew_component = config.VOLATILITY_SKEW_FACTOR * log_moneyness
            
            adjustment_factor = 1.0 + smile_component + skew_component
            
            # Bound the multiplicative adjustment factor to prevent extreme volatilities
            bounded_adjustment_factor = max(config.MIN_SMILE_ADJUSTMENT_FACTOR, 
                                            min(adjustment_factor, config.MAX_SMILE_ADJUSTMENT_FACTOR))

            strike_specific_vol = atm_vol_for_expiry * bounded_adjustment_factor
        else:
            logger.warning(
                f"Cannot apply smile/skew for T={expiry_minutes}min: invalid strike ({strike_price:.2f}) "
                f"or underlying ({underlying_price:.2f}). Using ATM vol for expiry ({atm_vol_for_expiry:.4f})."
            )
        
        # 5. Apply final global MIN/MAX volatility clamps to the strike-specific volatility
        final_strike_vol = max(min(strike_specific_vol, config.MAX_VOLATILITY), config.MIN_VOLATILITY)
        
        logger.debug(
           f"VolCalc: T={expiry_minutes}min, K={strike_price:.2f}, S={underlying_price:.2f} | "
           f"BaseEWMA={base_atm_vol:.4f}, Regime='{regime}', ATMRegAdj={atm_vol_regime_adjusted:.4f}, "
           f"ATMShortTermAdj={short_term_multiplier:.2f}, ATMFinalExp={atm_vol_for_expiry:.4f} | "
           f"LogMny={log_moneyness:.4f}, SmileComp={config.VOLATILITY_SMILE_CURVATURE * (log_moneyness ** 2):.4f}, "
           f"SkewComp={config.VOLATILITY_SKEW_FACTOR * log_moneyness:.4f}, "
           f"RawAdjFactor={adjustment_factor:.4f}, BoundedAdjFactor={bounded_adjustment_factor:.4f} | "
           f"StrikeSpecificPreClamp={strike_specific_vol:.4f} -> FinalVol={final_strike_vol:.4f}"
        )
        
        return final_strike_vol

    def get_volatility_metrics(self) -> VolatilityMetrics:
        """Get comprehensive volatility metrics."""
        simple_hist_vol = self.calculate_simple_historical_vol() 
        ewma_atm_vol = self.calculate_ewma_volatility() # Base EWMA ATM vol
        current_regime = self.detect_volatility_regime()
        
        # For representative regime_vol, calculate ATM vol for a standard expiry (e.g., 1hr)
        regime_vol_atm_1hr = ewma_atm_vol # Start with EWMA
        if self.price_history:
            latest_s = self.price_history[-1]
            if latest_s > 0:
                regime_vol_atm_1hr = self.get_expiry_adjusted_volatility(
                    expiry_minutes=60, # Standard 1-hour expiry
                    strike_price=latest_s, # ATM strike
                    underlying_price=latest_s
                )
            else: # Fallback if price is somehow invalid
                 logger.warning("Invalid latest price in history for metrics, regime_vol might be less representative.")
        else: # No price history yet
            logger.warning("No price history for metrics, regime_vol might be less representative.")


        # Confidence based on data sufficiency (simple measure)
        # Normalize against a reasonable number of points needed for "full" confidence, e.g., 200
        confidence = min(len(self.return_history) / 200.0, 1.0) 

        return VolatilityMetrics(
            current_vol=simple_hist_vol,
            regime_vol=regime_vol_atm_1hr, 
            ewma_vol=ewma_atm_vol,
            garch_vol=None, # Placeholder
            confidence=confidence,
            regime=current_regime
        )

