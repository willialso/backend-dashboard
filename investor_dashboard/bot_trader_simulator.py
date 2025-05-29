# investor_dashboard/bot_trader_simulator.py

import random
import time
import threading
import math  # ADDED: Missing import
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
import logging
from collections import defaultdict

from backend import config
from .revenue_engine import RevenueEngine

logger = logging.getLogger(__name__)

class TraderType(Enum):
    ADVANCED = "advanced"
    INTERMEDIATE = "intermediate"
    BEGINNER = "beginner"

@dataclass
class EnrichedTradeData:
    trade_id: str
    trader_type: TraderType
    option_type: str  # "call" or "put"
    strike: float
    expiry_minutes: int
    premium_usd: float
    quantity_contracts: float
    underlying_price: float
    timestamp: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv_pct: float = 0.0
    fair_value_usd: float = 0.0
    platform_markup_pct: float = 0.0
    trader_id: str = ""
    
    # Add any additional trade details needed for the dashboard
    is_successful: bool = True  # Was trade execution successful
    trader_experience_level: str = ""  # Additional demographic info
    trader_success_rate: float = 0.0  # Historical success rate
    platform_profit_usd: float = 0.0  # Platform's profit on this trade

class BotTraderSimulator:
    """Simulates bot trading activity to generate realistic options trade data."""
    
    def __init__(self,
                 revenue_engine: Optional[Any] = None,  # FIXED: Parameter name matches constructor call
                 position_manager: Optional[Any] = None,
                 data_feed_manager: Optional[Any] = None,
                 audit_engine_instance: Optional[Any] = None):
        
        logger.info("Initializing Bot Trader Simulator...")
        self.revenue_engine = revenue_engine
        self.position_manager = position_manager
        self.audit_engine = audit_engine_instance
        
        self.is_running = False
        self.current_btc_price = 0.0
        self.trade_id_counter = int(time.time() * 100)
        self.recent_trades_log: List[EnrichedTradeData] = [] # Log of trades initiated by bots
        
        # Set up trader profiles
        self.trader_profiles = {
            TraderType.ADVANCED: {"count": config.TRADER_COUNT_ADV, "avg_trade_premium": 2500,
                                 "trades_per_hour": 8, "success_rate": 0.78,
                                 "preferred_expiries": [e for e in config.AVAILABLE_EXPIRIES_MINUTES if e >=60],
                                 "risk_tolerance_otm_pct": 0.03},
            TraderType.INTERMEDIATE: {"count": config.TRADER_COUNT_INT, "avg_trade_premium": 1000,
                                     "trades_per_hour": 4, "success_rate": 0.65,
                                     "preferred_expiries": [e for e in config.AVAILABLE_EXPIRIES_MINUTES if e >=15 and e <= 240],
                                     "risk_tolerance_otm_pct": 0.05},
            TraderType.BEGINNER: {"count": config.TRADER_COUNT_BEG, "avg_trade_premium": 250,
                                 "trades_per_hour": 2, "success_rate": 0.52,
                                 "preferred_expiries": [e for e in config.AVAILABLE_EXPIRIES_MINUTES if e <= 60],
                                 "risk_tolerance_otm_pct": 0.07}
        }  # FIXED: Added missing closing brace
        
        # If data feed manager provided, register for price updates
        if data_feed_manager and hasattr(data_feed_manager, 'add_price_callback'):
            data_feed_manager.add_price_callback(self.update_market_data)
    
    def start(self):
        """Start the bot trader simulator."""
        if self.is_running:
            logger.info("BTS: Already running")
            return
        
        self.is_running = True
        # Force an initial price update to start with reasonable values
        if self.current_btc_price <= 0:
            self.current_btc_price = 108000.0  # FIXED: Set a default price
        
        # Start the trading simulation in a background thread
        threading.Thread(target=self._trading_loop, daemon=True).start()
        logger.info(f"BTS: Started with {sum(p['count'] for p in self.trader_profiles.values())} virtual traders")
    
    def stop(self):
        """Stop the bot trader simulator."""
        self.is_running = False
        logger.info("BTS: Stopped")
    
    def update_market_data(self, price_data: Any) -> None:
        """Update the current market price data from feed."""
        if hasattr(price_data, 'price') and price_data.price > 0:
            self.current_btc_price = price_data.price
    
    def _trading_loop(self):
        """Main loop that generates simulated trading activity."""
        logger.info("BTS: Trading loop started")
        
        try:
            while self.is_running:
                # Skip if price isn't available yet
                if self.current_btc_price <= 0:
                    logger.debug("BTS: No price data yet, waiting...")
                    time.sleep(config.DATA_BROADCAST_INTERVAL_SECONDS)
                    continue
                
                # Skip if dependencies aren't available
                if not self.revenue_engine or not self.position_manager:
                    logger.warning("BTS: Missing revenue_engine or position_manager, waiting...")
                    time.sleep(5)
                    continue
                
                # Generate trades for each trader type
                for trader_type, profile in self.trader_profiles.items():
                    # Calculate number of trades for this interval
                    trades_this_interval = self._calculate_trades_for_interval(profile)
                    
                    # Generate each trade
                    for _ in range(trades_this_interval):
                        if not self.is_running:
                            break
                        
                        self._generate_and_record_trade(trader_type, profile)
                        # Small sleep between trades to avoid spikes
                        time.sleep(random.uniform(0.05, 0.2))
                
                # Wait for next interval
                time.sleep(getattr(config, "TRADE_SIMULATION_INTERVAL_SECONDS", 30))
        except Exception as e:
            logger.error(f"BTS loop error: {e}", exc_info=True)
            self.is_running = False
    
    def _calculate_trades_for_interval(self, profile: Dict) -> int:
        """Calculate how many trades should be generated in this interval."""
        num_bots_of_type = profile["count"]
        trades_per_bot_ph = profile["trades_per_hour"]
        
        # Average trades per minute across all bots of this type
        avg_type_trades_pm = (trades_per_bot_ph * num_bots_of_type) / 60.0
        
        # Scale to our interval duration
        interval_minutes = getattr(config, "TRADE_SIMULATION_INTERVAL_SECONDS", 30) / 60.0
        avg_trades_per_interval = avg_type_trades_pm * interval_minutes
        
        # Add some randomness with normal distribution 
        # (mean = avg, std = sqrt(avg) for Poisson-like behavior)
        return max(0, int(random.normalvariate(avg_trades_per_interval, math.sqrt(max(0.1, avg_trades_per_interval)))))
    
    def _generate_and_record_trade(self, trader_type: TraderType, profile: Dict):
        """Generate a single simulated trade for the given trader type."""
        if self.current_btc_price <= 0:
            logger.warning("BTS: Cannot generate trade without valid price")
            return
        
        try:
            # Get a random expiry from preferred expiries
            expiry_minutes = random.choice(profile["preferred_expiries"])
            
            # Random call or put
            option_type = random.choice(["call", "put"])
            
            # Calculate strike based on risk tolerance (more OTM for higher risk)
            risk_tolerance = profile["risk_tolerance_otm_pct"]
            if option_type == "call":
                # OTM calls have strike > current price
                strike_multiplier = 1.0 + random.uniform(0, risk_tolerance * 2)
            else:
                # OTM puts have strike < current price
                strike_multiplier = 1.0 - random.uniform(0, risk_tolerance * 2)
            
            # Assign initial strike near current price
            strike_raw = self.current_btc_price * strike_multiplier
            
            # Round to nearest config.STRIKE_ROUNDING_NEAREST
            strike = round(strike_raw / config.STRIKE_ROUNDING_NEAREST) * config.STRIKE_ROUNDING_NEAREST
            strike = max(strike, config.STRIKE_ROUNDING_NEAREST)  # Ensure minimum positive strike
            
            # Ensure revenue engine has current BTC price
            if hasattr(self.revenue_engine, 'current_btc_price'):
                self.revenue_engine.current_btc_price = self.current_btc_price
            
            # Calculate trade premium using the revenue engine
            pricing_and_trade_info = self.revenue_engine.price_option_for_platform_sale(
                strike_price=strike,
                expiry_minutes=expiry_minutes,
                option_type=option_type,
                quantity=1.0  # Each bot trade is for 1 contract
            )
            
            # Create the trade log entry
            self.trade_id_counter += 1
            trade_id = f"BT{self.trade_id_counter}"
            
            trade_log_entry = EnrichedTradeData(
                trade_id=trade_id,
                trader_type=trader_type,
                option_type=option_type,
                strike=strike,
                expiry_minutes=expiry_minutes,
                premium_usd=pricing_and_trade_info.get("platform_price", 0),
                quantity_contracts=1.0,
                underlying_price=self.current_btc_price,
                timestamp=time.time(),
                delta=pricing_and_trade_info.get("delta", 0),
                gamma=pricing_and_trade_info.get("gamma", 0),
                theta=pricing_and_trade_info.get("theta", 0),
                vega=pricing_and_trade_info.get("vega", 0),
                iv_pct=pricing_and_trade_info.get("iv", 0) * 100,  # Convert to percentage
                fair_value_usd=pricing_and_trade_info.get("fair_value", 0),
                platform_markup_pct=pricing_and_trade_info.get("markup_pct", 0),
                trader_id=f"simbot{random.randint(1000,9999)}",
                is_successful=random.random() < profile["success_rate"],
                trader_experience_level=trader_type.value,
                trader_success_rate=profile["success_rate"],
                platform_profit_usd=pricing_and_trade_info.get("platform_profit", 0)
            )
            
            # Add to recent trades log
            self.recent_trades_log.append(trade_log_entry)
            if len(self.recent_trades_log) > config.MAX_RECENT_TRADES_LOG_SIZE_BOTSIM:
                self.recent_trades_log = self.recent_trades_log[-config.MAX_RECENT_TRADES_LOG_SIZE_BOTSIM:]
            
            # CRITICAL FIX: Ensure PositionManager has current BTC price before adding trade
            if hasattr(self.position_manager, 'current_btc_price'):
                self.position_manager.current_btc_price = self.current_btc_price
            
            # Add to position manager
            current_timestamp = time.time()
            expiry_timestamp = current_timestamp + (expiry_minutes * 60)  # Convert minutes to seconds
            self.position_manager.add_option_trade({
                "trade_id": trade_id,
                "option_type": option_type,
                "strike": strike,
                "expiry_minutes": expiry_minutes,
                "premium": pricing_and_trade_info.get("platform_price", 0),
                "quantity": 1.0,
                "underlying_price_at_trade": self.current_btc_price,
                "delta": pricing_and_trade_info.get("delta", 0),
                "gamma": pricing_and_trade_info.get("gamma", 0),
                "theta": pricing_and_trade_info.get("theta", 0),
                "vega": pricing_and_trade_info.get("vega", 0),
                "volatility_at_trade": pricing_and_trade_info.get("iv", 0.8),
                "is_short": True,
                "initial_premium_total": pricing_and_trade_info.get("initial_premium_total", pricing_and_trade_info.get("platform_price", 0)),
                "timestamp": current_timestamp,
                "expiry_timestamp": expiry_timestamp
            })
            
            logger.info(f"BTS: Generated {trader_type.value} {option_type.upper()} K{strike} {expiry_minutes}min trade, premium=${trade_log_entry.premium_usd:.2f}")
            return trade_log_entry
            
        except Exception as e:
            logger.error(f"BTS trade generation error: {e}", exc_info=True)
            return None
    
    def get_current_activity(self) -> List[Any]:
        """Get current activity stats for each trader type."""
        activity = []
        for trader_type, profile_data in self.trader_profiles.items():
            activity.append({
                "trader_type": trader_type,
                "count": profile_data["count"],
                "trades_per_hour_per_bot": profile_data["trades_per_hour"],
                "success_rate": profile_data["success_rate"],
                "timestamp": time.time()
            })
        return activity
    
    def get_recent_trades(self, limit: int = 10) -> List[EnrichedTradeData]:
        """Get most recent simulated trades."""
        # This log is now mostly for BTS's own record of what it *initiated*.
        return sorted(self.recent_trades_log, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def adjust_trader_distribution(self, advanced_pct: float, intermediate_pct: float, beginner_pct: float):
        # Your existing logic using config for base total traders is fine.
        if abs((advanced_pct + intermediate_pct + beginner_pct) - 100.0) > 0.1:
            logger.error("BTS: Trader distribution percentages must sum to 100.")
            raise ValueError("Percentages must sum to 100")
        
        total_traders = config.BASE_TOTAL_SIMULATED_TRADERS
        
        # Update counts in trader profiles
        self.trader_profiles[TraderType.ADVANCED]["count"] = int((advanced_pct / 100.0) * total_traders)
        self.trader_profiles[TraderType.INTERMEDIATE]["count"] = int((intermediate_pct / 100.0) * total_traders)
        self.trader_profiles[TraderType.BEGINNER]["count"] = total_traders - self.trader_profiles[TraderType.ADVANCED]["count"] - self.trader_profiles[TraderType.INTERMEDIATE]["count"]
        
        logger.info(f"BTS: Adjusted trader distribution to {self.trader_profiles[TraderType.ADVANCED]['count']} adv, {self.trader_profiles[TraderType.INTERMEDIATE]['count']} int, {self.trader_profiles[TraderType.BEGINNER]['count']} beg")
        
        return {
            "advanced_count": self.trader_profiles[TraderType.ADVANCED]["count"],
            "intermediate_count": self.trader_profiles[TraderType.INTERMEDIATE]["count"],
            "beginner_count": self.trader_profiles[TraderType.BEGINNER]["count"],
            "total_count": total_traders
        }
    
    def get_trading_statistics(self):
        """Get statistics about trading activity in the last 24 hours."""
        cutoff_24h = time.time() - (24 * 60 * 60)
        
        # Filter trades from last 24h
        trades_for_stats = [t for t in self.recent_trades_log if t.timestamp >= cutoff_24h]
        
        if not trades_for_stats:
            return {
                "total_trades_24h": 0,
                "total_premium_volume_usd_24h": 0,
                "avg_premium_received_usd_24h": 0,
                "call_put_ratio_24h": 0,
                "most_active_expiry_minutes_24h": 0,
                "data_source": "bts_log_no_trades_24h"
            }
        
        # Calculate call/put ratio
        calls = len([t for t in trades_for_stats if t.option_type == "call"])
        puts = len([t for t in trades_for_stats if t.option_type == "put"])
        call_put_ratio = calls / puts if puts > 0 else float('inf')
        
        # Find most active expiry
        expiry_counts = defaultdict(int)
        for t in trades_for_stats: expiry_counts[t.expiry_minutes] += 1
        most_active_expiry = max(expiry_counts.items(), key=lambda x: x[1])[0] if expiry_counts else 0
        
        # Calculate volume and average premium
        total_premium = sum(t.premium_usd for t in trades_for_stats)
        avg_premium = total_premium / len(trades_for_stats) if trades_for_stats else 0
        
        return {
            "total_trades_24h": len(trades_for_stats),
            "total_premium_volume_usd_24h": round(total_premium, 2),
            "avg_premium_received_usd_24h": round(avg_premium, 2),
            "call_put_ratio_24h": round(call_put_ratio, 2),
            "most_active_expiry_minutes_24h": most_active_expiry,
            "data_source": "bts_log_24h"
        }
