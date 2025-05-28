# backend/data_feed_manager.py

import threading
import time
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

from backend import config
from backend.utils import setup_logger

# Import the WebSocket clients and their global price info
from backend.coinbase_client import start_coinbase_ws, coinbase_btc_price_info
from backend.kraken_client import start_kraken_ws, kraken_btc_price_info
from backend.okx_client import start_okx_ws, okx_btc_price_info

logger = setup_logger(__name__)

@dataclass
class PriceData:
    symbol: str
    price: float
    volume: float
    timestamp: float
    exchange: str

class DataFeedManager:
    """Manages live data feeds from multiple exchanges with REAL BTC prices via WebSocket."""

    def __init__(self):
        self.price_callbacks: List[Callable[[PriceData], None]] = []
        self.latest_prices: Dict[str, PriceData] = {}
        self.is_running = False
        self.consolidated_price = 0.0
        self.price_history: List[PriceData] = []
        self.last_real_price = 0.0

    def add_price_callback(self, callback: Callable[[PriceData], None]) -> None:
        """Add callback function to receive price updates."""
        self.price_callbacks.append(callback)

    def _get_ws_btc_prices(self):
        """Aggregate latest BTC prices from all WebSocket clients with freshness check."""
        sources = []
        current_time = time.time()
        
        # 1. PRIORITIZE OKX (most real-time and accurate)
        if okx_btc_price_info.get('price'):
            # Use 'price' field if available (from 'last' in okx_client)
            last_update = okx_btc_price_info.get('last_update', 0)
            if current_time - last_update < 60:  # Fresh data (< 60 seconds old)
                sources.append(('OKX', okx_btc_price_info['price']))
        elif okx_btc_price_info.get('bid') and okx_btc_price_info.get('ask'):
            # Fallback to bid/ask average if 'price' not available
            last_update = okx_btc_price_info.get('last_update', 0)
            if current_time - last_update < 60:  # Fresh data
                bid = okx_btc_price_info['bid']
                ask = okx_btc_price_info['ask']
                sources.append(('OKX', (bid + ask) / 2))
        
        # 2. Then Coinbase (if OKX not available)
        if not sources and coinbase_btc_price_info.get('price'):
            last_update = coinbase_btc_price_info.get('last_update_time', 0)
            if current_time - last_update < 60:  # Fresh data
                sources.append(('Coinbase', coinbase_btc_price_info['price']))
        
        # 3. Then Kraken (last resort)
        if not sources and kraken_btc_price_info.get('price'):
            last_update = kraken_btc_price_info.get('last_update_time', 0)
            if current_time - last_update < 60:  # Fresh data
                sources.append(('Kraken', kraken_btc_price_info['price']))
        
        return sources

    def start(self) -> None:
        """Start real BTC price feeds via WebSocket."""
        logger.info("Starting data feed manager with REAL BTC prices (WebSocket)...")
        self.is_running = True

        # Start WebSocket clients as background threads
        threading.Thread(target=start_coinbase_ws, daemon=True, name="CoinbaseWSThread").start()
        threading.Thread(target=start_kraken_ws, daemon=True, name="KrakenWSThread").start()
        threading.Thread(target=start_okx_ws, daemon=True, name="OKXWSThread").start()

        # Start real price feed thread
        threading.Thread(target=self._real_price_feed, daemon=True).start()
        logger.info("Real BTC price feed started")

    def _real_price_feed(self) -> None:
        """Stream real BTC price updates from WebSocket clients."""
        consecutive_failures = 0
        max_failures = 5
        while self.is_running:
            try:
                ws_prices = self._get_ws_btc_prices()
                if ws_prices:
                    # Use the first available, or average/median if you prefer
                    exchange_name, real_price = ws_prices[0]
                    logger.info(f"Using real BTC price from {exchange_name} (WebSocket): ${real_price:,.2f}")

                    price_data = PriceData(
                        symbol="BTC-USD",
                        price=real_price,
                        volume=self._estimate_volume(),  # Estimated volume
                        timestamp=time.time(),
                        exchange=exchange_name
                    )
                    
                    self.consolidated_price = real_price
                    self.last_real_price = real_price
                    self.latest_prices["real"] = price_data
                    # Store in history
                    self.price_history.append(price_data)
                    if len(self.price_history) > config.PRICE_HISTORY_MAX_POINTS:
                        self.price_history = self.price_history[-config.PRICE_HISTORY_MAX_POINTS:]
                    # Notify all callbacks with REAL price
                    for callback in self.price_callbacks:
                        try:
                            callback(price_data)
                        except Exception as e:
                            logger.error(f"Price callback error: {e}")
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.warning(f"Failed to get WebSocket BTC price (attempt {consecutive_failures})")
                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive failures, stopping price feed")
                        break
            except Exception as e:
                logger.error(f"Real price feed error: {e}")
                consecutive_failures += 1
            # Update frequency: every 1 second for true real-time
            time.sleep(1)

    def _estimate_volume(self) -> float:
        """Estimate trading volume (can be enhanced with real volume data later)."""
        import random
        # Realistic BTC trading volume range
        return random.uniform(15000, 45000)

    def stop(self) -> None:
        """Stop all connections."""
        logger.info("Stopping data feed manager...")
        self.is_running = False

    def get_current_price(self) -> float:
        """Get current real BTC price."""
        return self.consolidated_price

    def get_price_history(self, minutes: int = 60) -> List[PriceData]:
        """Get price history for specified minutes."""
        cutoff_time = time.time() - (minutes * 60)
        return [p for p in self.price_history if p.timestamp >= cutoff_time]

    def get_exchange_status(self) -> Dict[str, Dict]:
        """Get status of real exchange connections."""
        current_time = time.time()
        latest = self.latest_prices.get("real")
        return {
            "real_market_data": {
                "connected": self.is_running,
                "last_price": latest.price if latest else self.consolidated_price,
                "last_update": latest.timestamp if latest else current_time,
                "stale": (current_time - latest.timestamp) > 60 if latest else False,
                "source": "WebSocket"
            }
        }
