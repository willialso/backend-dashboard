# main_dashboard.py

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import threading
import time

from core.config import *
from data_feeds.data_feed_manager import DataFeedManager
from investor_dashboard.dashboard_api import router as dashboard_router
from investor_dashboard.revenue_engine import RevenueEngine
from investor_dashboard.liquidity_manager import LiquidityManager
from investor_dashboard.bot_trader_simulator import BotTraderSimulator

app = FastAPI(title="Atticus Investor Dashboard", version="1.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
data_feed_manager = DataFeedManager()
revenue_engine = RevenueEngine()
liquidity_manager = LiquidityManager()
bot_trader_simulator = BotTraderSimulator()

# WebSocket connections for dashboard
dashboard_connections = set()

@app.on_event("startup")
async def startup_event():
    """Initialize dashboard backend services."""
    print("🚀 Starting Atticus Investor Dashboard Backend...")
    
    # Start data feeds
    data_feed_manager.start()
    
    # Start bot trader simulation
    bot_trader_simulator.start()
    
    # Start background dashboard updates
    threading.Thread(target=dashboard_update_loop, daemon=True).start()
    
    print("✅ Dashboard backend ready for investor connections")

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for dashboard real-time updates."""
    await websocket.accept()
    dashboard_connections.add(websocket)
    
    try:
        while True:
            # Send dashboard updates every 2 seconds
            dashboard_data = {
                "type": "dashboard_update",
                "data": {
                    "current_price": data_feed_manager.get_current_price(),
                    "revenue_metrics": revenue_engine.get_current_metrics(),
                    "liquidity_status": liquidity_manager.get_status(),
                    "bot_activity": bot_trader_simulator.get_current_activity(),
                    "timestamp": time.time()
                }
            }
            await websocket.send_text(json.dumps(dashboard_data))
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        dashboard_connections.remove(websocket)

def dashboard_update_loop():
    """Background loop for dashboard calculations."""
    while True:
        try:
            # Update revenue calculations
            current_price = data_feed_manager.get_current_price()
            if current_price > 0:
                revenue_engine.update_price(current_price)
                
            # Update bot trading simulation
            bot_trader_simulator.process_trades(current_price)
            
            # Update liquidity calculations
            liquidity_manager.update_metrics()
            
        except Exception as e:
            print(f"Dashboard update error: {e}")
        
        time.sleep(1)

# Include dashboard API routes
app.include_router(dashboard_router, prefix="/api/dashboard")

@app.get("/")
async def root():
    return {"message": "Atticus Investor Dashboard Backend", "status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
