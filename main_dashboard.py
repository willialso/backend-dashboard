# main_dashboard.py

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import threading
import time

# Assuming your project structure, adjust if needed
from backend.config import * # If config.py is in a 'backend' subfolder
from data_feeds.data_feed_manager import DataFeedManager
from investor_dashboard.dashboard_api import router as dashboard_router
from investor_dashboard.revenue_engine import RevenueEngine
from investor_dashboard.liquidity_manager import LiquidityManager
from investor_dashboard.bot_trader_simulator import BotTraderSimulator
from investor_dashboard.audit_engine import AuditEngine
from investor_dashboard.hedge_feed_manager import HedgeFeedManager

app = FastAPI(title="Atticus Investor Dashboard", version="1.0.0")

# ← CRITICAL CORS FIX: Only current Lovable domain and essentials
origins = [
    # Local development (essential for testing)
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    
    # CURRENT Lovable dashboard domain (from your error logs)
    "https://preview--atticus-insight-hub.lovable.app",  # PREVIEW URL
    "https://atticus-insight-hub.lovable.app",           # PRODUCTION URL (assumed)
    
    # Your Render deployment domain (essential for backend hosting)
    "https://atticus-demo-dashboard.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"], # Ensure OPTIONS is included
    allow_headers=["*"], # Allow all headers, including custom ones
    expose_headers=["*"], # Allow frontend to access headers like Content-Disposition for CSV
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Global instances (ensure these are correctly initialized)
data_feed_manager = DataFeedManager()
revenue_engine = RevenueEngine()
liquidity_manager = LiquidityManager()
bot_trader_simulator = BotTraderSimulator()
audit_engine = AuditEngine()
hedge_feed_manager = HedgeFeedManager()

# Ensure dashboard_api.py has access to these global instances
# This is crucial for your API endpoints to function
from investor_dashboard import dashboard_api
dashboard_api.revenue_engine = revenue_engine
dashboard_api.liquidity_manager = liquidity_manager
dashboard_api.bot_trader_simulator = bot_trader_simulator
dashboard_api.audit_engine = audit_engine
dashboard_api.hedge_feed_manager = hedge_feed_manager

# WebSocket connections for dashboard (if you plan to use WebSockets)
dashboard_connections = set()

@app.on_event("startup")
async def startup_event():
    """Initialize dashboard backend services."""
    print("🚀 Starting Atticus Investor Dashboard Backend...")
    print(f"✅ CORS Middleware configured. Allowing origins from: {origins}")
    
    # Start data feeds (if not already started by DataFeedManager constructor)
    if hasattr(data_feed_manager, 'start') and callable(getattr(data_feed_manager, 'start')):
        data_feed_manager.start()
    
    # Start bot trader simulation
    if hasattr(bot_trader_simulator, 'start') and callable(getattr(bot_trader_simulator, 'start')):
        bot_trader_simulator.start()
    
    # Start hedge feed manager
    if hasattr(hedge_feed_manager, 'start') and callable(getattr(hedge_feed_manager, 'start')):
        hedge_feed_manager.start()
    
    # Start background dashboard updates
    threading.Thread(target=dashboard_update_loop, daemon=True).start()
    
    print("✅ Dashboard backend services initialized and ready for investor connections.")

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for dashboard real-time updates."""
    await websocket.accept()
    dashboard_connections.add(websocket)
    print(f"WebSocket connection established: {websocket.client}")
    
    try:
        while True:
            # Send dashboard updates every 2 seconds
            try:
                # Prepare data safely, ensuring components are initialized
                current_price_data = data_feed_manager.get_current_price() if data_feed_manager else 0
                revenue_metrics_data = revenue_engine.get_current_metrics().__dict__ if revenue_engine else {}
                liquidity_status_data = liquidity_manager.get_status().__dict__ if liquidity_manager else {}
                bot_activity_data = [activity.__dict__ for activity in bot_trader_simulator.get_current_activity()] if bot_trader_simulator else []

                dashboard_data = {
                    "type": "dashboard_update",
                    "data": {
                        "current_price": current_price_data,
                        "revenue_metrics": revenue_metrics_data,
                        "liquidity_status": liquidity_status_data,
                        "bot_activity": bot_activity_data,
                        "timestamp": time.time()
                    }
                }
                await websocket.send_text(json.dumps(dashboard_data, default=str))
            except Exception as e:
                print(f"Error sending WebSocket data: {e}")
                # Send a minimal update or error message if full data fails
                error_data = {
                    "type": "error_update",
                    "message": f"Failed to fetch full dashboard data: {str(e)}",
                    "timestamp": time.time()
                }
                await websocket.send_text(json.dumps(error_data))
            
            await asyncio.sleep(2) # Interval for sending updates
    except WebSocketDisconnect:
        print(f"WebSocket connection closed: {websocket.client}")
    finally:
        dashboard_connections.remove(websocket)

def dashboard_update_loop():
    """Background loop for dashboard calculations and data refreshes."""
    print("🔄 Background dashboard update loop started.")
    while True:
        try:
            # Ensure components are initialized before calling methods
            current_price = data_feed_manager.get_current_price() if data_feed_manager else 0
            
            if revenue_engine and current_price > 0:
                revenue_engine.update_price(current_price)
            
            if bot_trader_simulator:
                bot_trader_simulator.process_trades(current_price)
            
            if hedge_feed_manager:
                hedge_feed_manager.update_btc_price(current_price)
            
            if liquidity_manager:
                liquidity_manager.update_metrics()
            
        except Exception as e:
            print(f"Error in dashboard update loop: {e}")
        
        time.sleep(1) # Frequency of background updates

# Include dashboard API routes from investor_dashboard/dashboard_api.py
app.include_router(dashboard_router, prefix="/api/dashboard")

# Root endpoint providing API information
@app.get("/")
async def root():
    # Dynamically list registered API endpoints for better maintainability
    api_routes = []
    for route in dashboard_router.routes:
        if hasattr(route, "path"):
            api_routes.append(f"/api/dashboard{route.path}")

    return {
        "message": "Atticus Investor Dashboard Backend", 
        "status": "running",
        "version": "1.0.0",
        "cors_configured": True,
        "active_cors_origins_count": len(origins), # Provides count, not the full list for brevity/security
        "lovable_integration_domain": "https://preview--atticus-insight-hub.lovable.app",
        "api_endpoints": sorted(list(set(api_routes))) # Ensure unique and sorted
    }

# Health check endpoint for monitoring
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    # Check basic operational status of components
    services_status = {}
    try:
        services_status["data_feed_operational"] = data_feed_manager.get_current_price() > 0 if data_feed_manager else False
        services_status["revenue_engine_operational"] = revenue_engine is not None and hasattr(revenue_engine, 'get_current_metrics')
        services_status["liquidity_manager_operational"] = liquidity_manager is not None and hasattr(liquidity_manager, 'get_status')
        services_status["bot_trader_simulator_operational"] = bot_trader_simulator is not None and hasattr(bot_trader_simulator, 'get_current_activity')
        services_status["hedge_feed_manager_operational"] = hedge_feed_manager is not None # Add specific check if available
        services_status["audit_engine_operational"] = audit_engine is not None # Add specific check if available
    except Exception as e:
        print(f"Health check component error: {e}")
        # If any component check fails, mark as unhealthy
        return {"status": "unhealthy", "detail": f"Component check failed: {str(e)}", "timestamp": time.time()}

    all_services_ok = all(services_status.values())

    return {
        "status": "healthy" if all_services_ok else "degraded",
        "detail": "All services operational" if all_services_ok else "One or more services may have issues",
        "services_status": services_status,
        "timestamp": time.time()
    }

# Main execution block
if __name__ == "__main__":
    print(f"Starting Uvicorn server on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
