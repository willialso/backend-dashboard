# okx_client.py

import websocket
import json
import time
import threading

OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

# FIXED: Added 'price' field that data_feed_manager expects
okx_btc_price_info = {'bid': None, 'ask': None, 'price': None, 'last_update': None}

def on_okx_message(ws, message):
    global okx_btc_price_info
    
    try:
        data = json.loads(message)
        
        if data.get('arg', {}).get('channel') == 'tickers' and data.get('data'):
            ticker_data = data['data'][0]
            
            # Get all price data
            bid = float(ticker_data.get('bidPx', 0))
            ask = float(ticker_data.get('askPx', 0))
            last_price = float(ticker_data.get('last', 0))  # Actual last traded price
            
            # Update global info with all price data
            okx_btc_price_info['bid'] = bid
            okx_btc_price_info['ask'] = ask
            okx_btc_price_info['price'] = last_price  # FIXED: Use last price, not mid
            okx_btc_price_info['last_update'] = time.time()
            
            print(f"OKX Update: Last: ${last_price:,.2f}, Bid: ${bid:,.2f}, Ask: ${ask:,.2f}")
            
        elif data.get('event') == 'error':
            print(f"OKX WS Error: {data.get('msg')}")
        elif data.get('event') == 'subscribe':
            print(f"OKX Subscription successful: {data}")
            
    except Exception as e:
        print(f"Error parsing OKX message: {e}")
        print(f"Raw message: {message}")

def on_okx_error(ws, error):
    print(f"OKX Error: {error}")

def on_okx_close(ws, close_status_code, close_msg):
    print(f"OKX WebSocket connection closed: {close_status_code} - {close_msg}")

def on_okx_open(ws):
    print("OKX WebSocket connection opened.")
    
    # FIXED: Added missing closing bracket
    subscribe_message = {
        "op": "subscribe",
        "args": [
            {"channel": "tickers", "instId": "BTC-USDT"}
        ]
    }  # ← FIXED: Added missing bracket
    
    print(f"Sending subscription: {subscribe_message}")
    ws.send(json.dumps(subscribe_message))

def start_okx_ws():
    ws = websocket.WebSocketApp(
        OKX_WS_URL,
        on_open=on_okx_open,
        on_message=on_okx_message,
        on_error=on_okx_error,
        on_close=on_okx_close
    )
    ws.run_forever()
