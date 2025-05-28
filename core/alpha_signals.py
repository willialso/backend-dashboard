# backend/alpha_signals.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from backend import config
from backend.utils import setup_logger

logger = setup_logger(__name__)

@dataclass 
class AlphaSignal:
    name: str
    value: float
    confidence: float
    direction: int  # -1, 0, 1
    timestamp: float

@dataclass
class CandlestickData:
    timestamp: float
    open: float
    high: float  
    low: float
    close: float
    volume: float

class AlphaSignalGenerator:
    """Implementation of the Bar Portion and other alpha signals from research."""
    
    def __init__(self):
        self.candlestick_history: List[CandlestickData] = []
        self.price_history: List[float] = []
        
    def update_tick(self, price: float, volume: float = 0) -> None:
        """Update with new tick data."""
        self.price_history.append(price)
        
        # For demo, create synthetic candlestick from price movements
        # In production, this would come from proper OHLCV feed
        if len(self.price_history) >= 2:
            # Create 1-minute synthetic candles
            self._create_synthetic_candle(price, volume)
            
    def _create_synthetic_candle(self, current_price: float, volume: float) -> None:
        """Create synthetic candlestick for demo purposes."""
        if len(self.candlestick_history) == 0:
            # First candle
            candle = CandlestickData(
                timestamp=pd.Timestamp.now().timestamp(),
                open=current_price,
                high=current_price,
                low=current_price, 
                close=current_price,
                volume=volume
            )
            self.candlestick_history.append(candle)
        else:
            # Update current candle or create new one
            last_candle = self.candlestick_history[-1]
            time_diff = pd.Timestamp.now().timestamp() - last_candle.timestamp
            
            if time_diff > 60:  # New minute candle
                candle = CandlestickData(
                    timestamp=pd.Timestamp.now().timestamp(),
                    open=last_candle.close,
                    high=current_price,
                    low=current_price,
                    close=current_price,
                    volume=volume
                )
                self.candlestick_history.append(candle)
            else:
                # Update current candle
                last_candle.high = max(last_candle.high, current_price)
                last_candle.low = min(last_candle.low, current_price)
                last_candle.close = current_price
                last_candle.volume += volume
                
        # Limit history
        if len(self.candlestick_history) > 1000:
            self.candlestick_history = self.candlestick_history[-1000:]
            
    def calculate_bar_portion(self, lookback: int = 1) -> float:
        """
        Calculate Bar Portion signal from research paper.
        Bar Portion = (Close - Open) / (High - Low)
        Range: -1 to 1
        """
        if len(self.candlestick_history) < lookback:
            return 0.0
            
        candle = self.candlestick_history[-lookback]
        
        if candle.high == candle.low:
            return 0.0  # No range
            
        bar_portion = (candle.close - candle.open) / (candle.high - candle.low)
        return max(-1.0, min(1.0, bar_portion))
    
    def calculate_bar_position(self, lookback: int = 1) -> float:
        """Calculate Bar Position signal."""
        if len(self.candlestick_history) < lookback:
            return 0.5
            
        candle = self.candlestick_history[-lookback]
        
        if candle.high == candle.low:
            return 0.5
            
        mid_price = (candle.open + candle.close) / 2
        bar_position = (mid_price - candle.low) / (candle.high - candle.low)
        return max(0.0, min(1.0, bar_position))
    
    def calculate_stick_length(self, lookback: int = 1, atr_period: int = 10) -> float:
        """Calculate normalized stick length."""
        if len(self.candlestick_history) < max(lookback, atr_period):
            return 1.0
            
        candle = self.candlestick_history[-lookback]
        
        # Calculate ATR
        atr = self._calculate_atr(atr_period)
        if atr == 0:
            return 1.0
            
        stick_length = (candle.high - candle.low) / atr
        return stick_length
    
    def _calculate_atr(self, period: int) -> float:
        """Calculate Average True Range."""
        if len(self.candlestick_history) < period + 1:
            return 0.0
            
        true_ranges = []
        for i in range(-period, 0):
            current = self.candlestick_history[i]
            previous = self.candlestick_history[i-1] if i > -len(self.candlestick_history) else current
            
            tr1 = current.high - current.low
            tr2 = abs(current.high - previous.close)
            tr3 = abs(current.low - previous.close)
            
            true_ranges.append(max(tr1, tr2, tr3))
            
        return np.mean(true_ranges) if true_ranges else 0.0
    
    def calculate_slope(self, period: int = 3) -> float:
        """Calculate price slope (momentum)."""
        if len(self.candlestick_history) < period:
            return 0.0
            
        recent_closes = [c.close for c in self.candlestick_history[-period:]]
        
        if len(recent_closes) < 2:
            return 0.0
            
        # Simple slope calculation
        slope = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
        return slope
    
    def calculate_curvature(self, period: int = 3) -> float:
        """Calculate curvature (acceleration)."""
        if len(self.candlestick_history) < period + 1:
            return 0.0
            
        current_slope = self.calculate_slope(period)
        
        # Calculate previous slope
        if len(self.candlestick_history) >= period + 1:
            prev_closes = [c.close for c in self.candlestick_history[-(period+1):-1]]
            if len(prev_closes) >= 2:
                prev_slope = (prev_closes[-1] - prev_closes[0]) / prev_closes[0]
                return current_slope - prev_slope
                
        return 0.0
    
    def generate_primary_signal(self) -> AlphaSignal:
        """Generate the primary Bar Portion signal."""
        bar_portion = self.calculate_bar_portion()
        
        # Based on research: Bar Portion shows monotonic decreasing relationship
        # Higher bar portion (bullish candle) tends to predict lower next returns
        direction = -1 if bar_portion > 0.1 else (1 if bar_portion < -0.1 else 0)
        
        confidence = min(abs(bar_portion) * 2, 1.0)  # Higher magnitude = higher confidence
        
        return AlphaSignal(
            name="bar_portion",
            value=bar_portion,
            confidence=confidence,
            direction=direction,
            timestamp=pd.Timestamp.now().timestamp()
        )
    
    def generate_all_signals(self) -> Dict[str, AlphaSignal]:
        """Generate all available alpha signals."""
        signals = {}
        
        # Primary signal
        signals["bar_portion"] = self.generate_primary_signal()
        
        # Additional signals
        bar_position = self.calculate_bar_position()
        signals["bar_position"] = AlphaSignal(
            name="bar_position", 
            value=bar_position,
            confidence=0.3,  # Lower confidence
            direction=0,
            timestamp=pd.Timestamp.now().timestamp()
        )
        
        stick_length = self.calculate_stick_length()
        signals["stick_length"] = AlphaSignal(
            name="stick_length",
            value=stick_length, 
            confidence=0.4,
            direction=1 if stick_length > 1.5 else 0,
            timestamp=pd.Timestamp.now().timestamp()
        )
        
        slope = self.calculate_slope()
        signals["slope"] = AlphaSignal(
            name="slope",
            value=slope,
            confidence=0.5,
            direction=1 if slope > 0.001 else (-1 if slope < -0.001 else 0),
            timestamp=pd.Timestamp.now().timestamp()
        )
        
        return signals
