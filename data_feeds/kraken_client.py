# backend/kraken_client.py
import websocket
import json
import time
import threading
import logging # For the __main__ block for direct testing

from backend import config # Import project-specific config
from backend.utils import setup_logger # Import project-specific logger

logger = setup_logger(__name__)

kraken_btc_price_info = {'bid': None, 'ask': None, 'price': None, 'last_update_time': None}

def on_kraken_message(ws, message):
    global kraken_btc_price_info
    try:
        data = json.loads(message)
        # logger.debug(f"Kraken Raw Message: {data}") # Uncomment for intense debugging

        if isinstance(data, dict) and data.get("channel") == "ticker" and "data" in data:
            payload = data["data"][0] # Ticker data is usually in a list
            
            current_bid, current_ask, current_price = None, None, None

            if "bid" in payload:
                current_bid = float(payload["bid"])
            if "ask" in payload:
                current_ask = float(payload["ask"])
            
            if "last_trade" in payload and isinstance(payload["last_trade"], dict) and "price" in payload["last_trade"]:
                current_price = float(payload["last_trade"]["price"])
            elif "last" in payload: # Fallback
                current_price = float(payload["last"])
            
            if current_bid is not None: kraken_btc_price_info['bid'] = current_bid
            if current_ask is not None: kraken_btc_price_info['ask'] = current_ask
            if current_price is not None: kraken_btc_price_info['price'] = current_price
            
            if current_bid or current_ask or current_price:
                kraken_btc_price_info['last_update_time'] = time.time()
                logger.debug(
                    f"Kraken Update: Pair: {payload.get('symbol')}, "
                    f"Price: {kraken_btc_price_info.get('price')}, "
                    f"Bid: {kraken_btc_price_info.get('bid')}, "
                    f"Ask: {kraken_btc_price_info.get('ask')}"
                )
        
        elif isinstance(data, dict) and data.get("method") == "subscribe":
            if data.get("success", False):
                subscription_details = data.get('result', {}).get('subscription', {})
                channel_name = subscription_details.get('name', data.get('result', {}).get('channel', 'Unknown Channel'))
                logger.info(f"Kraken: Successfully subscribed to {channel_name}")
            else:
                logger.error(f"Kraken: Subscription failed: {data.get('error', 'Unknown subscription error')}")
        elif isinstance(data, dict) and data.get("channel") == "heartbeat":
            logger.debug("Kraken: Received heartbeat")

    except Exception as e:
        logger.error(f"Kraken: Error processing message: {message} - {e}", exc_info=True)

def on_kraken_error(ws, error):
    logger.error(f"Kraken WebSocket error: {error}")

def on_kraken_close(ws, close_status_code, close_msg):
    logger.info(f"Kraken WebSocket closed. Code: {close_status_code}, Message: {close_msg}")

def on_kraken_open(ws_app):
    logger.info(f"Kraken WebSocket connection opened to {config.KRAKEN_WS_URL_V2}")
    subscribe_message = {
        "method": "subscribe",
        "params": {
            "channel": "ticker",
            "symbol": [config.KRAKEN_TICKER_SYMBOL],
        }
    }
    try:
        ws_app.send(json.dumps(subscribe_message))
        logger.info(
            f"Sent subscription to Kraken ticker for {config.KRAKEN_TICKER_SYMBOL} "
            f"via {config.KRAKEN_WS_URL_V2}"
        )
    except Exception as e:
        logger.error(f"Kraken: Error sending subscription message: {e}", exc_info=True)

def start_kraken_ws():
    logger.info(f"Attempting to connect to Kraken WebSocket at {config.KRAKEN_WS_URL_V2}")
    while True:
        try:
            ws = websocket.WebSocketApp(config.KRAKEN_WS_URL_V2, # Use URL from config
                                        on_open=on_kraken_open,
                                        on_message=on_kraken_message,
                                        on_error=on_kraken_error,
                                        on_close=on_kraken_close)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.error(f"Kraken: Unhandled exception in WebSocket run_forever loop: {e}", exc_info=True)
        
        logger.info("Kraken: Connection attempt ended. Reconnecting in 10 seconds...")
        time.sleep(10)

if __name__ == '__main__':
    if not logger.handlers:
        logging.basicConfig(level=config.LOG_LEVEL if hasattr(config, 'LOG_LEVEL') else logging.DEBUG,
                            format=config.LOG_FORMAT if hasattr(config, 'LOG_FORMAT') else '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        if not hasattr(config, 'KRAKEN_WS_URL_V2'):
            class MockConfig: KRAKEN_TICKER_SYMBOL, KRAKEN_WS_URL_V2 = "BTC/USD", "wss://ws.kraken.com/v2"
            config = MockConfig()
            logger.warning("Running with MOCK config values for direct test.")
            
    logger.info(f"Running kraken_client.py directly for testing (URL: {config.KRAKEN_WS_URL_V2})...")
    start_kraken_ws()
