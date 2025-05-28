# main_dashboard.py

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import threading
import time

from backend.config import *
from data_feeds.data_feed_manager import DataFeedManager
from investor_dashboard.dashboard_api import router as dashboard_router
from investor_dashboard.revenue_engine import RevenueEngine
from investor_dashboard.liquidity_manager import LiquidityManager
from investor_dashboard.bot_trader_simulator import BotTraderSimulator
from investor_dashboard.audit_engine import AuditEngine
from investor_dashboard.hedge_feed_manager import HedgeFeedManager

app = FastAPI(title="Atticus Investor Dashboard", version="1.0.0")

# ← ENHANCED CORS CONFIGURATION FOR LOVABLE INTEGRATION
origins = [
    # Local development
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    
    # Lovable domains (CRITICAL for frontend integration)
    "https://preview--atticusq-live-view.lovable.app",
    "https://atticusq-live-view.lovable.app",
    "https://preview--*.lovable.app",
    "https://*.lovable.app",
    
    # Your Render deployment
    "https://atticus-demo-dashboard.onrender.com",
    "https://*.onrender.com",
    
    # Development domains
    "http://localhost",
    "https://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Global instances
data_feed_manager = DataFeedManager()
revenue_engine = RevenueEngine()
liquidity_manager = LiquidityManager()
bot_trader_simulator = BotTraderSimulator()
audit_engine = AuditEngine()
hedge_feed_manager = HedgeFeedManager()

# ← FIX: Initialize dashboard API globals
from investor_dashboard import dashboard_api
dashboard_api.revenue_engine = revenue_engine
dashboard_api.liquidity_manager = liquidity_manager
dashboard_api.bot_trader_simulator = bot_trader_simulator
dashboard_api.audit_engine = audit_engine
dashboard_api.hedge_feed_manager = hedge_feed_manager

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
    
    # Start hedge feed manager
    hedge_feed_manager.start()
    
    # Start background dashboard updates
    threading.Thread(target=dashboard_update_loop, daemon=True).start()
    
    print("✅ Dashboard backend ready for investor connections")
    print("🌐 CORS configured for Lovable frontend integration")

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for dashboard real-time updates."""
    await websocket.accept()
    dashboard_connections.add(websocket)
    
    try:
        while True:
            # Send dashboard updates every 2 seconds
            try:
                dashboard_data = {
                    "type": "dashboard_update",
                    "data": {
                        "current_price": data_feed_manager.get_current_price(),
                        "revenue_metrics": revenue_engine.get_current_metrics().__dict__,
                        "liquidity_status": liquidity_manager.get_status().__dict__,
                        "bot_activity": [activity.__dict__ for activity in bot_trader_simulator.get_current_activity()],
                        "timestamp": time.time()
                    }
                }
                await websocket.send_text(json.dumps(dashboard_data, default=str))
            except Exception as e:
                print(f"WebSocket data error: {e}")
                # Send basic update if full data fails
                basic_data = {
                    "type": "dashboard_update",
                    "data": {
                        "current_price": data_feed_manager.get_current_price(),
                        "timestamp": time.time()
                    }
                }
                await websocket.send_text(json.dumps(basic_data))
            
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
            
            # Update hedge feed manager
            hedge_feed_manager.update_btc_price(current_price)
            
            # Update liquidity calculations
            liquidity_manager.update_metrics()
            
        except Exception as e:
            print(f"Dashboard update error: {e}")
        
        time.sleep(1)

# Include dashboard API routes
app.include_router(dashboard_router, prefix="/api/dashboard")

@app.get("/")
async def root():
    return {
        "message": "Atticus Investor Dashboard Backend", 
        "status": "running",
        "version": "1.0.0",
        "cors_configured": True,
        "api_endpoints": [
            "/api/dashboard/revenue-metrics",
            "/api/dashboard/liquidity-status",
            "/api/dashboard/platform-health",
            "/api/dashboard/bot-trader-activity",
            "/api/dashboard/hedge-execution-feed",
            "/api/dashboard/audit-summary",
            "/api/dashboard/recent-trades",
            "/api/dashboard/hedge-metrics"
        ]
    }

# ← ENHANCED: Health check endpoint for monitoring
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring services."""
    return {
        "status": "healthy",
        "services": {
            "data_feed": data_feed_manager.get_current_price() > 0,
            "revenue_engine": revenue_engine is not None,
            "liquidity_manager": liquidity_manager is not None,
            "bot_trader": bot_trader_simulator is not None,
            "hedge_manager": hedge_feed_manager is not None
        },
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
