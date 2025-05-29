# backend/volatility_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import math
import logging

from backend import config

logger = logging.getLogger(__name__)

@dataclass
class VolatilityMetrics:
    current_vol: float
    regime_vol: float
    ewma_vol: float
    garch_vol: Optional[float]
    confidence: float
    regime: str

class AdvancedVolatilityEngine:
    def __init__(self,
                 price_history_max_points: int = config.PRICE_HISTORY_MAX_POINTS,
                 default_vol: float = config.DEFAULT_VOLATILITY,
                 min_vol: float = config.MIN_VOLATILITY,
                 max_vol: float = config.MAX_VOLATILITY,
                 ewma_alpha: float = config.VOLATILITY_EWMA_ALPHA,
                 short_expiry_adjustments: Dict[int, float] = config.VOLATILITY_SHORT_EXPIRY_ADJUSTMENTS,
                 default_short_expiry_adjustment: float = config.VOLATILITY_DEFAULT_SHORT_EXPIRY_ADJUSTMENT,
                 smile_curvature: float = config.VOLATILITY_SMILE_CURVATURE,
                 skew_factor: float = config.VOLATILITY_SKEW_FACTOR,
                 min_smile_adj: float = config.MIN_SMILE_ADJUSTMENT_FACTOR,
                 max_smile_adj: float = config.MAX_SMILE_ADJUSTMENT_FACTOR):
        
        # Store all parameters
        self.price_history_max_points = price_history_max_points
        self.default_vol = default_vol
        self.min_vol = min_vol
        self.max_vol = max_vol
        self.ewma_alpha = ewma_alpha
        self.short_expiry_adjustments = short_expiry_adjustments
        self.default_short_expiry_adjustment = default_short_expiry_adjustment
        self.smile_curvature = smile_curvature
        self.skew_factor = skew_factor
        self.min_smile_adj = min_smile_adj
        self.max_smile_adj = max_smile_adj
        
        # Initialize data storage
        self.price_history: List[float] = []
        self.return_history: List[float] = []
        self.last_detected_regime = "medium"
        
        # Calculate annualization factor
        self.periods_per_year = (365.25 * 24 * 60 * 60) / config.DATA_BROADCAST_INTERVAL_SECONDS \
                               if config.DATA_BROADCAST_INTERVAL_SECONDS > 0 else 252 * 24 * 60
        self.annualization_factor = math.sqrt(self.periods_per_year)
        
        logger.info(f"AdvancedVolatilityEngine initialized. Max points: {self.price_history_max_points}")

    def update_price(self, price: float) -> None:
        if not isinstance(price, (int, float)) or price <= 0:
            return

        if self.price_history:
            log_return = math.log(price / self.price_history[-1])
            self.return_history.append(log_return)
        
        self.price_history.append(price)

        # Maintain size limits
        if len(self.price_history) > self.price_history_max_points:
            self.price_history = self.price_history[-self.price_history_max_points:]
        if len(self.return_history) > self.price_history_max_points:
            self.return_history = self.return_history[-self.price_history_max_points:]

    def calculate_simple_historical_vol(self, periods: Optional[int] = None) -> float:
        if len(self.return_history) < 20:
            return self.default_vol

        periods_to_use = periods if periods is not None else min(len(self.return_history), 100)
        actual_periods = min(periods_to_use, len(self.return_history))
        
        if actual_periods < 2:
            return self.default_vol
            
        returns_subset = np.array(self.return_history[-actual_periods:])
        std_dev = np.std(returns_subset, ddof=1)
        vol = std_dev * self.annualization_factor
        
        return max(self.min_vol, min(vol, self.max_vol))

    def calculate_ewma_volatility(self, alpha: Optional[float] = None) -> float:
        alpha_to_use = alpha if alpha is not None else self.ewma_alpha
        
        if len(self.return_history) < 10:
            return self.calculate_simple_historical_vol()

        returns_sq = np.array(self.return_history)**2
        
        try:
            ewm_series = pd.Series(returns_sq).ewm(alpha=alpha_to_use, adjust=True, min_periods=1).mean()
            variance_ewma = ewm_series.iloc[-1]
        except Exception:
            if not returns_sq.any():
                return self.default_vol
            variance_ewma = returns_sq[0]
            for r_sq in returns_sq[1:]:
                variance_ewma = alpha_to_use * r_sq + (1 - alpha_to_use) * variance_ewma
        
        vol = math.sqrt(max(0, variance_ewma)) * self.annualization_factor
        return max(self.min_vol, min(vol, self.max_vol))

    def detect_volatility_regime(self) -> str:
        if len(self.return_history) < 100:
            return self.last_detected_regime

        short_vol = self.calculate_simple_historical_vol(periods=20)
        long_vol = self.calculate_simple_historical_vol(periods=100)

        if long_vol < 1e-7:
            return "medium"

        ratio = short_vol / long_vol
        
        if ratio > 1.125:
            regime = "high"
        elif ratio < 0.925:
            regime = "low"
        else:
            regime = "medium"
        
        self.last_detected_regime = regime
        return regime

    def get_expiry_adjusted_volatility(self, expiry_minutes: int, strike_price: float, underlying_price: float) -> float:
        if underlying_price <= 1e-6:
            return self.default_vol

        # Base volatility
        base_vol = self.calculate_ewma_volatility()

        # Regime adjustment
        regime = self.detect_volatility_regime()
        regime_multipliers = {"low": 0.85, "medium": 1.0, "high": 1.25}
        regime_adjusted_vol = base_vol * regime_multipliers.get(regime, 1.0)

        # Expiry adjustment
        expiry_multiplier = self.short_expiry_adjustments.get(expiry_minutes, self.default_short_expiry_adjustment)
        time_adjusted_vol = regime_adjusted_vol * expiry_multiplier
        time_adjusted_vol = max(self.min_vol, min(time_adjusted_vol, self.max_vol))

        # Smile/skew adjustment for non-ATM options
        if abs(strike_price - underlying_price) / underlying_price >= 1e-5:
            log_moneyness = math.log(strike_price / underlying_price)
            smile_component = self.smile_curvature * (log_moneyness ** 2)
            skew_component = self.skew_factor * log_moneyness
            adjustment_factor = 1.0 + smile_component + skew_component
            bounded_adjustment = max(self.min_smile_adj, min(adjustment_factor, self.max_smile_adj))
            final_vol = time_adjusted_vol * bounded_adjustment
        else:
            final_vol = time_adjusted_vol
        
        return max(self.min_vol, min(final_vol, self.max_vol))

    def get_volatility_metrics(self) -> VolatilityMetrics:
        simple_vol = self.calculate_simple_historical_vol()
        ewma_vol = self.calculate_ewma_volatility()
        regime = self.detect_volatility_regime()
        
        # Calculate regime vol for a standard case
        if self.price_history and self.price_history[-1] > 0:
            current_price = self.price_history[-1]
            atm_strike = round(current_price / config.STRIKE_ROUNDING_NEAREST) * config.STRIKE_ROUNDING_NEAREST
            regime_vol = self.get_expiry_adjusted_volatility(60, atm_strike, current_price)
        else:
            regime_vol = ewma_vol

        confidence = min(len(self.return_history) / 200, 1.0)

        return VolatilityMetrics(
            current_vol=round(simple_vol, 4),
            regime_vol=round(regime_vol, 4),
            ewma_vol=round(ewma_vol, 4),
            garch_vol=None,
            confidence=round(confidence, 2),
            regime=regime
        )
