# backend/utils.py
import logging
import os
import time
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up logger with consistent formatting."""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class TimeSeriesAnalyzer:
    """Utility class for time series analysis."""
    
    @staticmethod
    def calculate_returns(prices: List[float]) -> List[float]:
        """Calculate log returns from price series."""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                ret = np.log(prices[i] / prices[i-1])
                returns.append(ret)
            else:
                returns.append(0.0)
        
        return returns
    
    @staticmethod
    def calculate_volatility(returns: List[float], annualize: bool = True) -> float:
        """Calculate volatility from returns."""
        if len(returns) < 2:
            return 0.0
        
        vol = np.std(returns)
        
        if annualize:
            # Assume returns are at 1-second frequency for high-freq data
            periods_per_year = 365 * 24 * 60 * 60
            vol *= np.sqrt(periods_per_year)
        
        return vol
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        vol = np.std(returns)
        
        if vol == 0:
            return 0.0
        
        # Annualize if needed
        periods_per_year = 365 * 24 * 60 * 60  # Assuming second-level data
        annual_return = mean_return * periods_per_year
        annual_vol = vol * np.sqrt(periods_per_year)
        
        sharpe = (annual_return - risk_free_rate) / annual_vol
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> Dict[str, float]:
        """Calculate maximum drawdown."""
        if len(prices) < 2:
            return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0}
        
        prices_array = np.array(prices)
        cumulative_max = np.maximum.accumulate(prices_array)
        drawdown = (cumulative_max - prices_array) / cumulative_max
        
        max_drawdown_pct = np.max(drawdown)
        max_drawdown_abs = np.max(cumulative_max - prices_array)
        
        return {
            "max_drawdown": max_drawdown_abs,
            "max_drawdown_pct": max_drawdown_pct
        }

class PerformanceMetrics:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def calculate_all_metrics(trades: List[Dict], initial_balance: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return {}
        
        # Extract P&L values
        pnl_values = [trade.get('realized_pnl_usd', 0) for trade in trades]
        
        total_pnl = sum(pnl_values)
        total_trades = len(trades)
        winning_trades = [p for p in pnl_values if p > 0]
        losing_trades = [p for p in pnl_values if p < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        
        # Calculate returns for Sharpe ratio
        returns = []
        for trade in trades:
            trade_return = trade.get('realized_pnl_usd', 0) / initial_balance
            returns.append(trade_return)
        
        sharpe_ratio = TimeSeriesAnalyzer.calculate_sharpe_ratio(returns)
        
        metrics = {
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "sharpe_ratio": sharpe_ratio,
            "roi": (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0
        }
        
        return metrics

class DataValidator:
    """Validate data integrity and consistency."""
    
    @staticmethod
    def validate_price_data(price: float) -> bool:
        """Validate price data."""
        if not isinstance(price, (int, float)):
            return False
        
        if price <= 0:
            return False
        
        if np.isnan(price) or np.isinf(price):
            return False
        
        # Reasonable bounds for BTC price (adjust as needed)
        if price < 1000 or price > 1000000:
            return False
        
        return True
    
    @staticmethod
    def validate_option_parameters(strike: float, premium: float, expiry_minutes: int) -> bool:
        """Validate option parameters."""
        if not all(isinstance(x, (int, float)) for x in [strike, premium]):
            return False
        
        if strike <= 0 or premium < 0:
            return False
        
        if not isinstance(expiry_minutes, int) or expiry_minutes <= 0:
            return False
        
        return True

class ConfigManager:
    """Manage configuration and settings."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return config_data
        except Exception as e:
            logging.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    @staticmethod
    def save_config(config_data: Dict[str, Any], config_path: str) -> bool:
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save config to {config_path}: {e}")
            return False

class PriceDataCache:
    """Cache for price data with time-based expiry."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
    
    def set(self, key: str, value: Any) -> None:
        """Set cache value with timestamp."""
        current_time = time.time()
        
        # Clean expired entries
        self._clean_expired()
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            self.remove(oldest_key)
        
        self.cache[key] = value
        self.timestamps[key] = current_time
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache value if not expired."""
        if key not in self.cache:
            return None
        
        current_time = time.time()
        if current_time - self.timestamps[key] > self.ttl_seconds:
            self.remove(key)
            return None
        
        return self.cache[key]
    
    def remove(self, key: str) -> None:
        """Remove cache entry."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def _clean_expired(self) -> None:
        """Clean expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self.remove(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.timestamps.clear()

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount for display."""
    if currency == "USD":
        return f"${amount:,.2f}"
    elif currency == "BTC":
        if amount >= 0.01:
            return f"{amount:.4f} BTC"
        else:
            sats = amount * 100000000
            return f"{sats:,.0f} sats"
    else:
        return f"{amount:.8f} {currency}"

def calculate_time_to_expiry(expiry_timestamp: float) -> Dict[str, int]:
    """Calculate time remaining to expiry."""
    current_time = time.time()
    time_diff = expiry_timestamp - current_time
    
    if time_diff <= 0:
        return {"days": 0, "hours": 0, "minutes": 0, "seconds": 0}
    
    days = int(time_diff // (24 * 3600))
    hours = int((time_diff % (24 * 3600)) // 3600)
    minutes = int((time_diff % 3600) // 60)
    seconds = int(time_diff % 60)
    
    return {"days": days, "hours": hours, "minutes": minutes, "seconds": seconds}

def create_option_symbol(option_type: str, strike: float, expiry_minutes: int) -> str:
    """Create standardized option symbol."""
    return f"BTC-{expiry_minutes}min-{int(strike)}-{option_type.upper()}"

def parse_option_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Parse option symbol into components."""
    try:
        parts = symbol.split('-')
        if len(parts) != 4:
            return None
        
        asset, expiry_str, strike_str, option_type = parts
        
        expiry_minutes = int(expiry_str.replace('min', ''))
        strike = float(strike_str)
        
        return {
            "asset": asset,
            "expiry_minutes": expiry_minutes,
            "strike": strike,
            "option_type": option_type.lower()
        }
    except Exception:
        return None
