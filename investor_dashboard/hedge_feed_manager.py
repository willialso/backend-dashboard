# investor_dashboard/hedge_feed_manager.py

import time
import random
import threading
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class HedgeType(Enum):
    DELTA_HEDGE = "Delta Hedge"
    RL_HEDGE = "RL Hedge"
    PORTFOLIO_REBALANCE = "Portfolio Rebal"
    GAMMA_HEDGE = "Gamma Hedge"
    VEGA_HEDGE = "Vega Hedge"

class Exchange(Enum):
    COINBASE_PRO = "CB Pro"
    KRAKEN = "Kraken"
    OKX = "OKX"

@dataclass
class HedgeExecution:
    timestamp: float
    hedge_type: HedgeType
    side: str  # "Buy" or "Sell"
    quantity_btc: float
    price_usd: float
    exchange: Exchange
    cost_usd: float
    reasoning: str
    execution_time_ms: float

class HedgeFeedManager:
    """Manages real-time hedge execution feed for dashboard."""
    
    def __init__(self):
        self.recent_hedges = []
        self.is_running = False
        self.current_btc_price = 108000
        self.total_hedge_cost_24h = 0.0
        self.hedge_count_24h = 0
        
        # Exchange routing preferences
        self.exchange_weights = {
            Exchange.COINBASE_PRO: 0.5,  # 50% - primary liquidity
            Exchange.KRAKEN: 0.3,        # 30% - backup
            Exchange.OKX: 0.2            # 20% - arbitrage
        }
        
    def start(self):
        """Start hedge execution simulation."""
        self.is_running = True
        threading.Thread(target=self._hedge_simulation_loop, daemon=True).start()
        
    def _hedge_simulation_loop(self):
        """Simulate realistic hedge executions."""
        while self.is_running:
            try:
                # Generate hedge executions based on market activity
                if self._should_generate_hedge():
                    hedge = self._generate_realistic_hedge()
                    if hedge:
                        self.recent_hedges.append(hedge)
                        self._update_24h_metrics(hedge)
                        
                # Keep only last 50 hedges
                if len(self.recent_hedges) > 50:
                    self.recent_hedges = self.recent_hedges[-50:]
                    
            except Exception as e:
                print(f"Hedge simulation error: {e}")
                
            # Check for hedges every 15-45 seconds
            sleep_time = random.uniform(15, 45)
            time.sleep(sleep_time)
    
    def _should_generate_hedge(self) -> bool:
        """Determine if a hedge should be executed now."""
        # Realistic hedge frequency - not every minute
        # Based on actual option flow and delta changes
        
        hedge_probability = 0.15  # 15% chance per check
        
        # Increase probability during volatile periods
        volatility_factor = random.uniform(0.8, 1.5)
        adjusted_probability = hedge_probability * volatility_factor
        
        return random.random() < adjusted_probability
    
    def _generate_realistic_hedge(self) -> HedgeExecution:
        """Generate a realistic hedge execution."""
        if self.current_btc_price == 0:
            return None
            
        # Choose hedge type based on realistic probabilities
        hedge_type = random.choices(
            list(HedgeType),
            weights=[0.6, 0.2, 0.15, 0.03, 0.02]  # Delta most common
        )[0]
        
        # Choose exchange based on weights
        exchange = random.choices(
            list(Exchange),
            weights=list(self.exchange_weights.values())
        )[0]
        
        # Generate realistic hedge parameters
        side = random.choice(["Buy", "Sell"])
        
        # Hedge quantities typically 0.1 to 2.5 BTC
        quantity_btc = round(random.uniform(0.1, 2.5), 2)
        
        # Price with realistic slippage
        slippage_factor = random.uniform(0.9998, 1.0002)  # ±0.02% slippage
        execution_price = self.current_btc_price * slippage_factor
        
        # Execution cost
        cost_usd = quantity_btc * execution_price
        
        # Execution time (realistic for each exchange)
        execution_times = {
            Exchange.COINBASE_PRO: random.uniform(150, 350),
            Exchange.KRAKEN: random.uniform(200, 500),
            Exchange.OKX: random.uniform(100, 250)
        }
        execution_time_ms = execution_times[exchange]
        
        # Generate reasoning based on hedge type
        reasoning = self._generate_hedge_reasoning(hedge_type, side, quantity_btc)
        
        return HedgeExecution(
            timestamp=time.time(),
            hedge_type=hedge_type,
            side=side,
            quantity_btc=quantity_btc,
            price_usd=execution_price,
            exchange=exchange,
            cost_usd=cost_usd,
            reasoning=reasoning,
            execution_time_ms=execution_time_ms
        )
    
    def _generate_hedge_reasoning(self, hedge_type: HedgeType, side: str, quantity: float) -> str:
        """Generate realistic hedge reasoning."""
        reasons = {
            HedgeType.DELTA_HEDGE: [
                f"Portfolio delta exceeded threshold",
                f"Rebalancing {quantity:.2f} BTC position",
                f"Delta-neutral adjustment required",
                f"Option flow imbalance correction"
            ],
            HedgeType.RL_HEDGE: [
                f"ML model recommended {side.lower()}",
                f"Risk-reward optimization signal",
                f"Predictive model hedge trigger",
                f"AI-driven position adjustment"
            ],
            HedgeType.PORTFOLIO_REBALANCE: [
                f"Quarterly rebalancing schedule",
                f"Risk allocation adjustment",
                f"Portfolio optimization required",
                f"Strategic position sizing"
            ],
            HedgeType.GAMMA_HEDGE: [
                f"Gamma exposure management",
                f"Convexity risk mitigation",
                f"Second-order sensitivity hedge",
                f"Options portfolio gamma neutral"
            ],
            HedgeType.VEGA_HEDGE: [
                f"Volatility exposure reduction",
                f"Implied vol risk management",
                f"Vega-neutral positioning",
                f"Vol skew protection"
            ]
        }
        
        return random.choice(reasons[hedge_type])
    
    def _update_24h_metrics(self, hedge: HedgeExecution):
        """Update 24-hour hedge metrics."""
        self.total_hedge_cost_24h += hedge.cost_usd
        self.hedge_count_24h += 1
        
        # Clean up old metrics (keep only 24h)
        cutoff_24h = time.time() - (24 * 60 * 60)
        self.recent_hedges = [
            h for h in self.recent_hedges 
            if h.timestamp > cutoff_24h
        ]
    
    def update_btc_price(self, price: float):
        """Update current BTC price for hedge calculations."""
        self.current_btc_price = price
    
    def get_recent_hedges(self, limit: int = 10) -> List[HedgeExecution]:
        """Get most recent hedge executions."""
        return sorted(self.recent_hedges, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_hedge_metrics(self) -> Dict:
        """Get hedge execution metrics."""
        cutoff_24h = time.time() - (24 * 60 * 60)
        recent_hedges_24h = [
            h for h in self.recent_hedges 
            if h.timestamp > cutoff_24h
        ]
        
        if not recent_hedges_24h:
            return {
                "hedges_24h": 0,
                "total_cost_24h": 0,
                "avg_execution_time_ms": 0,
                "exchange_distribution": {},
                "hedge_type_distribution": {}
            }
        
        # Calculate metrics
        total_cost = sum(h.cost_usd for h in recent_hedges_24h)
        avg_execution_time = sum(h.execution_time_ms for h in recent_hedges_24h) / len(recent_hedges_24h)
        
        # Exchange distribution
        exchange_dist = {}
        for exchange in Exchange:
            count = len([h for h in recent_hedges_24h if h.exchange == exchange])
            exchange_dist[exchange.value] = count
            
        # Hedge type distribution
        hedge_type_dist = {}
        for hedge_type in HedgeType:
            count = len([h for h in recent_hedges_24h if h.hedge_type == hedge_type])
            hedge_type_dist[hedge_type.value] = count
        
        return {
            "hedges_24h": len(recent_hedges_24h),
            "total_cost_24h": total_cost,
            "avg_execution_time_ms": avg_execution_time,
            "exchange_distribution": exchange_dist,
            "hedge_type_distribution": hedge_type_dist
        }
