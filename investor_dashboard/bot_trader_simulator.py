# investor_dashboard/bot_trader_simulator.py

import random
import time
import threading
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

class TraderType(Enum):
    ADVANCED = "Advanced"
    INTERMEDIATE = "Intermediate" 
    BEGINNER = "Beginner"

@dataclass
class BotTrade:
    trader_type: TraderType
    option_type: str  # "call" or "put"
    strike: float
    expiry_minutes: int
    premium_paid: float
    timestamp: float
    trader_id: str

@dataclass
class TraderActivity:
    trader_type: TraderType
    active_count: int
    percentage: float
    avg_trade_size_usd: float
    trades_per_hour: int
    success_rate: float

class BotTraderSimulator:
    """Simulates realistic trading activity from different trader types."""
    
    def __init__(self):
        # Trader distribution and characteristics
        self.trader_profiles = {
            TraderType.ADVANCED: {
                "percentage": 55,
                "count": 127,
                "avg_trade_size": 5000,
                "trades_per_hour": 8,
                "success_rate": 78,
                "preferred_expiries": [60, 240, 480],  # Longer timeframes
                "risk_tolerance": 0.15  # 15% OTM strikes
            },
            TraderType.INTERMEDIATE: {
                "percentage": 33,
                "count": 78,
                "avg_trade_size": 2000,
                "trades_per_hour": 4,
                "success_rate": 65,
                "preferred_expiries": [15, 60, 240],
                "risk_tolerance": 0.10  # 10% OTM strikes
            },
            TraderType.BEGINNER: {
                "percentage": 12,
                "count": 37,
                "avg_trade_size": 500,
                "trades_per_hour": 2,
                "success_rate": 52,
                "preferred_expiries": [15, 60],  # Shorter timeframes
                "risk_tolerance": 0.05  # 5% OTM strikes
            }
        }
        
        self.recent_trades = []
        self.is_running = False
        self.current_btc_price = 108000  # Starting price
        
    def start(self):
        """Start bot trading simulation."""
        self.is_running = True
        threading.Thread(target=self._trading_loop, daemon=True).start()
        
    def _trading_loop(self):
        """Main trading simulation loop."""
        while self.is_running:
            try:
                # Generate trades based on trader profiles
                for trader_type, profile in self.trader_profiles.items():
                    trades_this_minute = self._calculate_trades_per_minute(profile)
                    
                    for _ in range(trades_this_minute):
                        trade = self._generate_trade(trader_type, profile)
                        if trade:
                            self.recent_trades.append(trade)
                            
                # Keep only last 100 trades
                if len(self.recent_trades) > 100:
                    self.recent_trades = self.recent_trades[-100:]
                    
            except Exception as e:
                print(f"Bot trading simulation error: {e}")
                
            time.sleep(60)  # Run every minute
    
    def _calculate_trades_per_minute(self, profile: Dict) -> int:
        """Calculate how many trades to generate this minute."""
        trades_per_hour = profile["trades_per_hour"]
        active_traders = profile["count"]
        
        # Average trades per minute across all traders of this type
        avg_trades_per_minute = (trades_per_hour * active_traders) / 60
        
        # Add some randomness
        return max(0, int(random.poisson(avg_trades_per_minute)))
    
    def _generate_trade(self, trader_type: TraderType, profile: Dict) -> BotTrade:
        """Generate a realistic trade for a trader type."""
        if self.current_btc_price == 0:
            return None
            
        # Choose option type (slightly favor calls in crypto)
        option_type = random.choices(["call", "put"], weights=[0.6, 0.4])[0]
        
        # Choose expiry based on trader preferences
        expiry_minutes = random.choice(profile["preferred_expiries"])
        
        # Choose strike based on risk tolerance
        risk_tolerance = profile["risk_tolerance"]
        if option_type == "call":
            # OTM calls (above current price)
            strike_multiplier = 1 + random.uniform(0.01, risk_tolerance)
        else:
            # OTM puts (below current price)
            strike_multiplier = 1 - random.uniform(0.01, risk_tolerance)
            
        strike = round(self.current_btc_price * strike_multiplier, -1)  # Round to $10
        
        # Estimate premium (simplified)
        moneyness = abs(strike - self.current_btc_price) / self.current_btc_price
        time_value = expiry_minutes / 480  # Normalize to 8-hour max
        estimated_premium = self.current_btc_price * 0.1 * (0.03 + moneyness + time_value * 0.02)
        
        # Add trader-specific variations
        trade_size_variation = random.uniform(0.7, 1.3)
        premium_paid = estimated_premium * trade_size_variation
        
        return BotTrade(
            trader_type=trader_type,
            option_type=option_type,
            strike=strike,
            expiry_minutes=expiry_minutes,
            premium_paid=premium_paid,
            timestamp=time.time(),
            trader_id=f"{trader_type.value}_{random.randint(1000, 9999)}"
        )
    
    def process_trades(self, current_btc_price: float):
        """Process trades with current BTC price."""
        self.current_btc_price = current_btc_price
    
    def get_current_activity(self) -> List[TraderActivity]:
        """Get current trader activity summary."""
        activities = []
        
        for trader_type, profile in self.trader_profiles.items():
            # Count recent trades (last 5 minutes)
            recent_cutoff = time.time() - 300  # 5 minutes
            recent_trades_count = len([
                t for t in self.recent_trades 
                if t.trader_type == trader_type and t.timestamp > recent_cutoff
            ])
            
            activity = TraderActivity(
                trader_type=trader_type,
                active_count=profile["count"],
                percentage=profile["percentage"],
                avg_trade_size_usd=profile["avg_trade_size"],
                trades_per_hour=profile["trades_per_hour"],
                success_rate=profile["success_rate"]
            )
            activities.append(activity)
            
        return activities
    
    def get_recent_trades(self, limit: int = 10) -> List[BotTrade]:
        """Get most recent trades for live feed."""
        return sorted(self.recent_trades, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def adjust_trader_distribution(self, advanced_pct: float, intermediate_pct: float, beginner_pct: float):
        """Adjust trader type distribution (investor controls)."""
        total_traders = sum(profile["count"] for profile in self.trader_profiles.values())
        
        # Update counts based on new percentages
        self.trader_profiles[TraderType.ADVANCED]["count"] = int(total_traders * advanced_pct / 100)
        self.trader_profiles[TraderType.INTERMEDIATE]["count"] = int(total_traders * intermediate_pct / 100)
        self.trader_profiles[TraderType.BEGINNER]["count"] = int(total_traders * beginner_pct / 100)
        
        # Update percentages
        self.trader_profiles[TraderType.ADVANCED]["percentage"] = advanced_pct
        self.trader_profiles[TraderType.INTERMEDIATE]["percentage"] = intermediate_pct
        self.trader_profiles[TraderType.BEGINNER]["percentage"] = beginner_pct
