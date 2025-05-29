# data_feeds/data_feed_manager.py

import threading
import time
from typing import List, Dict, Callable, Optional, Any, Tuple # Ensure Tuple is imported
from dataclasses import dataclass
import logging
from collections import deque
import random # For fallback volume estimation

from backend import config
logger = logging.getLogger(__name__)

from .coinbase_client import start_coinbase_ws, coinbase_btc_price_info
from .kraken_client import start_kraken_ws, kraken_btc_price_info
from .okx_client import start_okx_ws, okx_btc_price_info

@dataclass
class PriceData:
    symbol: str
    price: float
    volume: float
    timestamp: float
    exchange: str

class DataFeedManager:
    """Manages live data feeds from multiple exchanges using dedicated WebSocket client modules."""

    # CORRECTED __init__ METHOD WITH MATCHING PARAMETER NAMES
    def __init__(self,
                 enabled_exchanges: List[str] = config.EXCHANGES_ENABLED,  # Changed from enabled_exchanges_list
                 primary_exchange: str = config.PRIMARY_EXCHANGE,         # Changed from primary_exchange_name
                 coinbase_product_id: str = config.COINBASE_PRODUCT_ID,   # Changed from coinbase_product_id_str
                 kraken_ticker_symbol: str = config.KRAKEN_TICKER_SYMBOL, # Changed from kraken_ticker_symbol_str
                 okx_ws_url: str = config.OKX_WS_URL,                     # Added (main_dashboard.py passes this)
                 coinbase_ws_url: str = config.COINBASE_WS_URL,           # Added
                 kraken_ws_url_v2: str = config.KRAKEN_WS_URL_V2,         # Added
                 data_broadcast_interval_seconds: float = config.DATA_BROADCAST_INTERVAL_SECONDS,  # Added
                 price_change_threshold_for_broadcast: float = config.PRICE_CHANGE_THRESHOLD_FOR_BROADCAST,  # Added
                 price_history_max_points: int = config.PRICE_HISTORY_MAX_POINTS  # Changed from price_history_max_items
                 ):
        
        self.enabled_exchanges = [ex.lower() for ex in enabled_exchanges]  # Use the parameter directly
        self.primary_exchange = primary_exchange.lower()                   # Use the parameter directly
        
        # Store URL/config values for potential use
        self._coinbase_product_id = coinbase_product_id
        self._kraken_ticker_symbol = kraken_ticker_symbol
        self._okx_ws_url = okx_ws_url
        self._coinbase_ws_url = coinbase_ws_url
        self._kraken_ws_url_v2 = kraken_ws_url_v2
        self._data_broadcast_interval = data_broadcast_interval_seconds
        self._price_change_threshold = price_change_threshold_for_broadcast
        
        self.price_callbacks: List[Callable[[PriceData], None]] = []
        self.latest_prices_from_ws: Dict[str, PriceData] = {}
        self.is_running = False
        self.consolidated_btc_price: float = 0.0
        self.price_history = deque(maxlen=price_history_max_points)  # Use the parameter directly

        self.last_broadcast_price: float = 0.0
        self.last_broadcast_time: float = 0.0

        self._active_ws_clients: Dict[str, Callable] = {}
        if "coinbase" in self.enabled_exchanges: self._active_ws_clients["coinbase"] = start_coinbase_ws
        if "kraken" in self.enabled_exchanges: self._active_ws_clients["kraken"] = start_kraken_ws
        if "okx" in self.enabled_exchanges: self._active_ws_clients["okx"] = start_okx_ws

        logger.info(f"DataFeedManager initialized. Enabled WS: {list(self._active_ws_clients.keys())}. Primary: {self.primary_exchange}.")

    def add_price_callback(self, callback: Callable[[PriceData], None]) -> None:
        self.price_callbacks.append(callback)
        logger.info(f"DFM: Added price callback.")

    def _get_prioritized_ws_price(self) -> Optional[Tuple[str, float, float]]:
        sources_data: List[Tuple[str, float, float]] = []
        current_time = time.time()
        freshness_threshold_seconds = 60

        # OKX
        if "okx" in self.enabled_exchanges:
            okx_price = okx_btc_price_info.get('price')
            okx_last_update = okx_btc_price_info.get('last_update', 0)
            if okx_price and okx_last_update and (current_time - okx_last_update < freshness_threshold_seconds):
                sources_data.append(("okx", float(okx_price), okx_last_update))
            elif okx_btc_price_info.get('bid') and okx_btc_price_info.get('ask'):
                bid, ask = okx_btc_price_info.get('bid'), okx_btc_price_info.get('ask')
                if bid and ask and okx_last_update and (current_time - okx_last_update < freshness_threshold_seconds):
                     sources_data.append(("okx", (float(bid) + float(ask)) / 2.0, okx_last_update))
        
        # Coinbase
        if "coinbase" in self.enabled_exchanges:
            cb_price = coinbase_btc_price_info.get('price')
            cb_last_update = coinbase_btc_price_info.get('last_update_time', 0)
            if cb_price and cb_last_update and (current_time - cb_last_update < freshness_threshold_seconds):
                sources_data.append(("coinbase", float(cb_price), cb_last_update))
        
        # Kraken
        if "kraken" in self.enabled_exchanges:
            kr_price = kraken_btc_price_info.get('price')
            kr_last_update = kraken_btc_price_info.get('last_update_time', 0)
            if kr_price and kr_last_update and (current_time - kr_last_update < freshness_threshold_seconds):
                sources_data.append(("kraken", float(kr_price), kr_last_update))
        
        if not sources_data: return None
        primary_source = next((s for s in sources_data if s[0] == self.primary_exchange), None)
        if primary_source: return primary_source
        sources_data.sort(key=lambda x: x[2], reverse=True)
        return sources_data[0]

    def start(self) -> None:
        if self.is_running: 
            logger.info("DFM: Already running.")
            return
        
        logger.info("DFM: Starting data feed manager with REAL BTC prices (WebSocket)...")
        self.is_running = True
        
        for ex_name, start_func in self._active_ws_clients.items():
            logger.info(f"DFM: Starting WebSocket client thread for {ex_name.upper()}...")
            threading.Thread(target=start_func, daemon=True, name=f"{ex_name.capitalize()}WSThread").start()
        
        threading.Thread(target=self._real_price_feed_loop, daemon=True, name="DFMPriceFeedLoop").start()
        logger.info("DFM: Real BTC price feed loop started.")

    def _real_price_feed_loop(self) -> None:
        consecutive_failures = 0
        max_failures = 10
        loop_interval = max(0.1, self._data_broadcast_interval / 2.0)

        while self.is_running:
            try:
                price_info = self._get_prioritized_ws_price()
                if price_info:
                    exchange, real_price, update_ts = price_info
                    consecutive_failures = 0
                    
                    volume = 0.0
                    source_info_dict = None
                    if exchange == "okx": source_info_dict = okx_btc_price_info
                    elif exchange == "coinbase": source_info_dict = coinbase_btc_price_info
                    elif exchange == "kraken": source_info_dict = kraken_btc_price_info
                    
                    if source_info_dict:
                        volume = float(source_info_dict.get("vol24h", source_info_dict.get("volume_24h", 0.0)))

                    price_data = PriceData(
                        symbol="BTC-USD", price=real_price,
                        volume=volume if volume else self._estimate_fallback_volume(),
                        timestamp=update_ts, exchange=exchange.upper()
                    )

                    with threading.Lock():
                        self.consolidated_btc_price = price_data.price
                        self.latest_prices_from_ws[exchange] = price_data
                        self.price_history.append(price_data)

                    price_change_pct = abs(price_data.price - self.last_broadcast_price) / self.last_broadcast_price if self.last_broadcast_price > 0 else float('inf')
                    time_since_bcast = time.time() - self.last_broadcast_time

                    if price_change_pct >= self._price_change_threshold or time_since_bcast >= self._data_broadcast_interval:
                        self.last_broadcast_price = price_data.price
                        self.last_broadcast_time = time.time()
                        logger.info(f"DFM Broadcast: {price_data.exchange} ${price_data.price:,.2f} Vol:{price_data.volume:.2f}")
                        
                        for cb in self.price_callbacks:
                            try: cb(price_data)
                            except Exception as e: logger.error(f"DFM CallbackErr: {e}")
                    else:
                        logger.debug(f"DFM Price: {price_data.exchange} ${price_data.price:,.2f}. No broadcast.")
                else:
                    consecutive_failures += 1
                    logger.warning(f"DFM: No fresh WS BTC price (Attempt {consecutive_failures}/{max_failures}).")
                    if consecutive_failures >= max_failures:
                        logger.error("DFM: Max WS failures. Price feed potentially stale.")
                        time.sleep(10)  # Sleep longer on total failure
                        
            except Exception as e: 
                logger.error(f"DFM LoopErr: {e}")
                consecutive_failures += 1
                
            time.sleep(loop_interval)

    def _estimate_fallback_volume(self) -> float:
        return random.uniform(15000, 45000) * config.STANDARD_CONTRACT_SIZE_BTC

    def stop(self) -> None:
        logger.info("DFM: Stopping...")
        self.is_running = False
        logger.info("DFM: Stopped.")

    def get_current_price(self) -> float:
        return self.consolidated_btc_price

    def get_price_history(self, limit: int = 100) -> List[PriceData]:
        with threading.Lock(): 
            return list(self.price_history)[-limit:]

    def get_exchange_status(self) -> Dict[str, Dict[str, Any]]:
        status: Dict[str, Dict[str, Any]] = {}
        current_time = time.time()

        for ex_name in self.enabled_exchanges:
            info_dict: Optional[Dict] = None
            if ex_name == "okx": info_dict = okx_btc_price_info
            elif ex_name == "coinbase": info_dict = coinbase_btc_price_info
            elif ex_name == "kraken": info_dict = kraken_btc_price_info
            
            price, last_upd = 0.0, 0.0
            if info_dict:
                price = info_dict.get('price', 0.0)
                if not price and info_dict.get('bid') and info_dict.get('ask'):
                    price = (float(info_dict['bid']) + float(info_dict['ask'])) / 2.0
                last_upd = info_dict.get('last_update', info_dict.get('last_update_time', 0.0))
            
            status[ex_name] = {
                "is_enabled": ex_name in self._active_ws_clients,
                "reported_price": float(price) if price else 0.0,
                "last_update_ts": last_upd,
                "is_stale": (current_time - last_upd) > 60 if last_upd > 0 else True,
                "secs_since_update": round(current_time - last_upd, 1) if last_upd > 0 else -1.0
            }

        status["consolidated_feed"] = {
            "current_price": self.get_current_price(),
            "last_broadcast_time": self.last_broadcast_time,
            "last_broadcast_price": self.last_broadcast_price,
            "primary_exchange_target": self.primary_exchange.upper()
        }
        return status
