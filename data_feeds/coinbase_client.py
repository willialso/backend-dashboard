# backend/coinbase_client.py
import websocket
import json
import time
import threading
import logging # For the __main__ block for direct testing

from backend import config # Import project-specific config
from backend.utils import setup_logger # Import project-specific logger

logger = setup_logger(__name__)

coinbase_btc_price_info = {'bid': None, 'ask': None, 'price': None, 'last_update_time': None}

def on_coinbase_message(ws, message):
    global coinbase_btc_price_info
    try:
        data = json.loads(message)
        # logger.debug(f"Coinbase Raw Message: {data}") # Uncomment for intense debugging
        if data.get("type") == "ticker" and data.get("product_id") == config.COINBASE_PRODUCT_ID:
            if "price" in data:
                coinbase_btc_price_info['price'] = float(data["price"])
            if "best_bid" in data:
                coinbase_btc_price_info['bid'] = float(data["best_bid"])
            if "best_ask" in data:
                coinbase_btc_price_info['ask'] = float(data["best_ask"])
            
            coinbase_btc_price_info['last_update_time'] = time.time()
            logger.debug(
                f"Coinbase Update: Product: {data.get('product_id')}, "
                f"Price: {coinbase_btc_price_info.get('price')}, "
                f"Bid: {coinbase_btc_price_info.get('bid')}, "
                f"Ask: {coinbase_btc_price_info.get('ask')}"
            )
    except Exception as e:
        logger.error(f"Coinbase: Error processing message: {message} - {e}", exc_info=True)

def on_coinbase_error(ws, error):
    logger.error(f"Coinbase WebSocket error: {error}")

def on_coinbase_close(ws, close_status_code, close_msg):
    logger.info(f"Coinbase WebSocket closed. Code: {close_status_code}, Message: {close_msg}")

def on_coinbase_open(ws_app):
    logger.info(f"Coinbase WebSocket connection opened to {config.COINBASE_WS_URL}")
    # This subscription message structure was from your previous working example,
    # typically used with ws-feed.exchange.coinbase.com.
    subscribe_message = {
        "type": "subscribe",
        "product_ids": [config.COINBASE_PRODUCT_ID],
        "channels": [{"name": "ticker", "product_ids": [config.COINBASE_PRODUCT_ID]}]
    }
    # If config.COINBASE_WS_URL was set to "wss://ws-feed.pro.coinbase.com",
    # a simpler subscribe might be:
    # subscribe_message = {
    #     "type": "subscribe",
    #     "product_ids": [config.COINBASE_PRODUCT_ID],
    #     "channels": ["ticker"]
    # }
    try:
        ws_app.send(json.dumps(subscribe_message))
        logger.info(
            f"Sent subscription to Coinbase ticker for {config.COINBASE_PRODUCT_ID} "
            f"via {config.COINBASE_WS_URL}"
        )
    except Exception as e:
        logger.error(f"Coinbase: Error sending subscription message: {e}", exc_info=True)

def start_coinbase_ws():
    logger.info(f"Attempting to connect to Coinbase WebSocket at {config.COINBASE_WS_URL}")
    while True:
        try:
            ws = websocket.WebSocketApp(config.COINBASE_WS_URL, # Use URL from config
                                        on_open=on_coinbase_open,
                                        on_message=on_coinbase_message,
                                        on_error=on_coinbase_error,
                                        on_close=on_coinbase_close)
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            logger.error(f"Coinbase: Unhandled exception in WebSocket run_forever loop: {e}", exc_info=True)
        
        logger.info("Coinbase: Connection attempt ended. Reconnecting in 10 seconds...")
        time.sleep(10)

if __name__ == '__main__':
    # This block is for testing this script directly.
    # It sets up basic logging if utils.py isn't available in the Python path.
    if not logger.handlers: # Check if logger from setup_logger already has handlers
        logging.basicConfig(level=config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else logging.DEBUG,
                            format=config.LOG_FORMAT if hasattr(config, 'LOG_FORMAT') else '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # If running directly and utils.setup_logger wasn't fully effective due to path issues,
        # re-get the logger with the basicConfig.
        logger = logging.getLogger(__name__) 
        if not hasattr(config, 'COINBASE_WS_URL'): # Minimal mock config if config.py somehow not loaded
            class MockConfig: COINBASE_PRODUCT_ID, COINBASE_WS_URL = "BTC-USD", "wss://ws-feed.exchange.coinbase.com"
            config = MockConfig()
            logger.warning("Running with MOCK config values for direct test.")

    logger.info(f"Running coinbase_client.py directly for testing (URL: {config.COINBASE_WS_URL})...")
    start_coinbase_ws()
