# investor_dashboard/bot_trader_simulator.py

import random
import time
import threading
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import logging
from collections import defaultdict

from backend import config
from .revenue_engine import RevenueEngine
from investor_dashboard.position_manager import PositionManager

logger = logging.getLogger(__name__)

class TraderType(Enum):
    ADVANCED = "advanced"
    INTERMEDIATE = "intermediate"
    BEGINNER = "beginner"

@dataclass
class EnrichedTradeData:
    trade_id: str
    trader_type: TraderType
    option_type: str
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
    is_successful: bool = True
    trader_experience_level: str = ""
    trader_success_rate: float = 0.0
    platform_profit_usd: float = 0.0

class BotTraderSimulator:
    """Simulates automated bot trading and records positions in PositionManager."""

    def __init__(
        self,
        revenue_engine: Optional[RevenueEngine] = None,
        position_manager: Optional[PositionManager] = None,
        data_feed_manager: Optional[Any] = None,
        audit_engine_instance: Optional[Any] = None
    ):
        logger.info("Initializing Bot Trader Simulator...")
        
        self.revenue_engine = revenue_engine
        self.position_manager = position_manager
        self.audit_engine = audit_engine_instance
        self.is_running = False
        self.current_btc_price = 0.0
        self.trade_id_counter = int(time.time() * 100)
        self.recent_trades_log: List[EnrichedTradeData] = []

        # CRITICAL FIX: Initialize all volume and revenue tracking attributes
        self.total_trades_executed = 0
        self.total_premium_collected_usd = 0.0
        self.daily_volume_usd = 0.0
        self.daily_trade_count = 0
        self.total_premium_volume_usd_24h = 0.0
        self.avg_premium_received_usd_24h = 0.0
        self.daily_revenue_usd = 0.0
        self.cumulative_revenue = 0.0
        self.total_volume_traded = 0.0
        
        # CRITICAL FIX: Initialize cached metrics and tracking
        self.daily_metrics = {}
        self._trading_stats_cache = None
        self._cached_statistics = None
        self.start_time = time.time()

        # 50% Advanced, 35% Intermediate, 15% Beginner
        total = config.BASE_TOTAL_SIMULATED_TRADERS
        self.trader_profiles = {
            TraderType.ADVANCED: {
                "count": int(total * 0.50),
                "trades_per_hour": config.TRADER_COUNT_ADV,
                "success_rate": config.TRADER_SUCCESS_RATE,
                "preferred_expiries": [e for e in config.AVAILABLE_EXPIRIES_MINUTES if e >= 60],
                "risk_tolerance_otm_pct": config.BOT_OTM_MAX_FACTORS["Advanced"]
            },
            TraderType.INTERMEDIATE: {
                "count": int(total * 0.35),
                "trades_per_hour": config.TRADER_COUNT_INT,
                "success_rate": config.TRADER_SUCCESS_RATE,
                "preferred_expiries": [e for e in config.AVAILABLE_EXPIRIES_MINUTES if 15 <= e <= 240],
                "risk_tolerance_otm_pct": config.BOT_OTM_MAX_FACTORS["Intermediate"]
            },
            TraderType.BEGINNER: {
                "count": total - int(total*0.50) - int(total*0.35),
                "trades_per_hour": config.TRADER_COUNT_BEG,
                "success_rate": config.TRADER_SUCCESS_RATE,
                "preferred_expiries": [e for e in config.AVAILABLE_EXPIRIES_MINUTES if e <= 60],
                "risk_tolerance_otm_pct": config.BOT_OTM_MAX_FACTORS["Beginner"]
            }
        }

        if data_feed_manager and hasattr(data_feed_manager, 'add_price_callback'):
            data_feed_manager.add_price_callback(self.update_market_data)

    def reset_to_defaults(self):
        """CRITICAL FIX: Reset all trading data to defaults for complete system reset"""
        try:
            logger.warning("BTS: Resetting all trading data to defaults")
            
            # Clear trade log
            self.recent_trades_log.clear()
            logger.info("✅ BTS: Cleared recent_trades_log")
            
            # CRITICAL: Reset ALL volume and revenue counters
            volume_revenue_attributes = [
                'total_trades_executed',
                'total_premium_collected_usd', 
                'daily_volume_usd',
                'daily_trade_count',
                'total_premium_volume_usd_24h',
                'avg_premium_received_usd_24h',
                'daily_revenue_usd',
                'cumulative_revenue',
                'total_volume_traded'
            ]
            
            for attr in volume_revenue_attributes:
                if hasattr(self, attr):
                    if 'usd' in attr.lower() or 'revenue' in attr.lower() or 'premium' in attr.lower():
                        setattr(self, attr, 0.0)
                    else:
                        setattr(self, attr, 0)
                    logger.debug(f"✅ BTS: Reset {attr} to 0")
            
            # Clear any cached metrics
            for cache_attr in ['daily_metrics', '_trading_stats_cache', '_cached_statistics']:
                if hasattr(self, cache_attr):
                    val = getattr(self, cache_attr)
                    if isinstance(val, dict):
                        val.clear()
                    elif isinstance(val, list):
                        val.clear()
                    else:
                        setattr(self, cache_attr, None)
                    logger.debug(f"✅ BTS: Cleared {cache_attr}")
            
            # Reset start time for fresh 24h window
            self.start_time = time.time()
            
            # Reset trade ID counter
            self.trade_id_counter = int(time.time() * 100)
            
            logger.info("✅ BTS: All trading data reset to defaults")
            return {"status": "success", "message": "All trading data reset to defaults"}
            
        except Exception as e:
            logger.error(f"❌ BTS: Error resetting to defaults: {e}")
            return {"status": "error", "message": str(e)}

    def start(self):
        if self.is_running:
            return
        
        self.is_running = True
        if self.current_btc_price <= 0:
            self.current_btc_price = 108000.0
        
        threading.Thread(target=self._trading_loop, daemon=True).start()
        total_bots = sum(p["count"] for p in self.trader_profiles.values())
        logger.info(f"BTS: Started with {total_bots} virtual traders")

    def stop(self):
        self.is_running = False
        logger.info("BTS: Stopped")

    def update_market_data(self, price_data: Any):
        if isinstance(price_data, dict) and price_data.get("price", 0) > 0:
            self.current_btc_price = price_data["price"]

    def _trading_loop(self):
        logger.info("BTS: Trading loop running")
        try:
            while self.is_running:
                if self.current_btc_price <= 0 or not self.revenue_engine or not self.position_manager:
                    time.sleep(config.DATA_BROADCAST_INTERVAL_SECONDS)
                    continue

                interval = getattr(config, "TRADE_SIMULATION_INTERVAL_SECONDS", 30)
                
                # FIXED: Read current trader profiles each loop iteration
                current_total = sum(p["count"] for p in self.trader_profiles.values())
                
                for trader_type, profile in self.trader_profiles.items():
                    # FIXED: Use current profile count, not cached value
                    current_count = profile["count"]
                    if current_count <= 0:
                        continue

                    n_trades = self._calculate_trades_for_interval(profile)
                    
                    # Log trade generation for debugging
                    if n_trades > 0:
                        logger.info(f"BTS: Generating {n_trades} trades for {trader_type.value} ({current_count} traders)")

                    for _ in range(n_trades):
                        if not self.is_running:
                            break
                        self._generate_and_record_trade(trader_type, profile)
                        time.sleep(random.uniform(0.05, 0.2))

                # Log total traders for debugging
                if current_total != config.BASE_TOTAL_SIMULATED_TRADERS:
                    logger.info(f"BTS: Operating with {current_total} total traders (scaled from {config.BASE_TOTAL_SIMULATED_TRADERS})")

                time.sleep(interval)

        except Exception as e:
            logger.error(f"BTS loop error: {e}", exc_info=True)
            self.is_running = False

    def _calculate_trades_for_interval(self, profile: Dict) -> int:
        """FIXED: Properly calculate trades based on current trader count"""
        current_bots = profile["count"]  # Use current count, not cached
        base_trades_per_hour = profile.get("trades_per_hour", 1)
        
        # Calculate trades per minute for this trader type
        trades_per_minute = (base_trades_per_hour * current_bots) / 60.0
        
        # Calculate expected trades for this interval
        interval_minutes = getattr(config, "TRADE_SIMULATION_INTERVAL_SECONDS", 30) / 60.0
        expected_trades = trades_per_minute * interval_minutes
        
        # Add some randomness but ensure we hit the expected average
        if expected_trades < 1.0:
            # For low rates, use probability-based generation
            return 1 if random.random() < expected_trades else 0
        else:
            # For higher rates, use normal distribution around expected value
            variance = max(0.1, expected_trades * 0.1)  # 10% variance
            actual_trades = max(0, int(random.normalvariate(expected_trades, math.sqrt(variance))))
            
            # Debug logging for scaling verification
            if current_bots != config.BASE_TOTAL_SIMULATED_TRADERS // 3:  # Detect when scaled
                logger.debug(f"BTS: {current_bots} bots → {actual_trades} trades this interval")
            
            return actual_trades

    def _generate_and_record_trade(self, trader_type: TraderType, profile: Dict) -> Optional[EnrichedTradeData]:
        try:
            expiry = random.choice(profile["preferred_expiries"])
            opt_type = random.choice(["call", "put"])
            tol = profile["risk_tolerance_otm_pct"]
            multiplier = (1 + random.uniform(0, tol*2)) if opt_type == "call" else (1 - random.uniform(0, tol*2))
            strike = round(self.current_btc_price * multiplier / config.STRIKE_ROUNDING_NEAREST) * config.STRIKE_ROUNDING_NEAREST

            # Try to price via revenue engine with debugging
            info = {}
            premium = 0.0
            markup = 0.0
            
            if self.revenue_engine:
                try:
                    self.revenue_engine.current_btc_price = self.current_btc_price
                    
                    # Try different method names that might exist
                    if hasattr(self.revenue_engine, 'price_option_for_platform_sale'):
                        info = self.revenue_engine.price_option_for_platform_sale(
                            strike_price=strike,
                            expiry_minutes=expiry,
                            option_type=opt_type,
                            quantity=1.0
                        )
                    elif hasattr(self.revenue_engine, 'get_option_pricing'):
                        info = self.revenue_engine.get_option_pricing(
                            strike=strike,
                            expiry_minutes=expiry,
                            option_type=opt_type
                        )
                    elif hasattr(self.revenue_engine, 'calculate_option_price'):
                        info = self.revenue_engine.calculate_option_price(
                            strike, expiry, opt_type
                        )

                    # Extract premium from various possible key names
                    premium = (
                        info.get("platform_price_per_contract", 0) or
                        info.get("platform_price", 0) or
                        info.get("price", 0) or
                        info.get("premium", 0) or
                        info.get("cost", 0)
                    )
                    markup = info.get("markup_pct", 0) or info.get("markup", 0)
                    
                except Exception as e:
                    logger.error(f"BTS: Revenue engine error: {e}")
                    info = {}

            # Fallback pricing if revenue engine fails
            if premium == 0:
                # Simple Black-Scholes approximation for demonstration
                S = self.current_btc_price
                K = strike
                T = expiry / (365.25 * 24 * 60)  # Convert minutes to years
                r = config.RISK_FREE_RATE
                sigma = config.DEFAULT_VOLATILITY

                # Simplified option pricing
                from math import sqrt, exp, log
                try:
                    import scipy.stats as stats
                    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
                    d2 = d1 - sigma*sqrt(T)
                    
                    if opt_type == "call":
                        premium = S*stats.norm.cdf(d1) - K*exp(-r*T)*stats.norm.cdf(d2)
                    else:
                        premium = K*exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
                    
                    premium = max(premium, 0.01)  # Minimum $0.01
                    markup = config.REVENUE_BASE_MARKUP_PERCENTAGE * 100
                    
                    # Add markup
                    premium_with_markup = premium * (1 + markup/100)
                    premium = round(premium_with_markup, 2)
                    
                except ImportError:
                    # Ultra-simple fallback
                    moneyness = abs(S - K) / S
                    time_value = sqrt(T) * 0.1 * S
                    intrinsic = max(S - K if opt_type == "call" else K - S, 0)
                    premium = round(intrinsic + time_value + moneyness * 100, 2)
                    premium = max(premium, 1.0)  # Minimum $1
                    markup = 3.5

            self.trade_id_counter += 1
            tid = f"BT{self.trade_id_counter}"

            trade = EnrichedTradeData(
                trade_id=tid,
                trader_type=trader_type,
                option_type=opt_type,
                strike=strike,
                expiry_minutes=expiry,
                premium_usd=premium,
                quantity_contracts=1.0,
                underlying_price=self.current_btc_price,
                timestamp=time.time(),
                delta=info.get("delta", 0),
                gamma=info.get("gamma", 0),
                theta=info.get("theta", 0),
                vega=info.get("vega", 0),
                iv_pct=info.get("iv", 0)*100 if info.get("iv") else 0,
                fair_value_usd=info.get("fair_value", premium * 0.9),
                platform_markup_pct=markup,
                trader_id=f"simbot{random.randint(1000,9999)}",
                is_successful=random.random() < profile["success_rate"],
                trader_experience_level=trader_type.value,
                trader_success_rate=profile["success_rate"],
                platform_profit_usd=info.get("platform_profit", premium * markup/100)
            )

            # CRITICAL FIX: Update volume and revenue tracking
            self.total_trades_executed += 1
            self.total_premium_collected_usd += premium
            self.daily_volume_usd += premium
            self.daily_trade_count += 1
            self.total_premium_volume_usd_24h += premium
            self.cumulative_revenue += info.get("platform_profit", premium * markup/100)
            self.total_volume_traded += premium

            # Append to log
            self.recent_trades_log.append(trade)
            if len(self.recent_trades_log) > config.MAX_RECENT_TRADES_LOG_SIZE_BOTSIM:
                self.recent_trades_log.pop(0)

            # Record in PositionManager
            if self.position_manager:
                now = time.time()
                self.position_manager.current_btc_price = self.current_btc_price
                self.position_manager.add_option_trade({
                    "trade_id": tid,
                    "option_type": opt_type,
                    "strike": strike,
                    "expiry_timestamp": now + expiry*60,
                    "quantity": 1.0,
                    "is_short": True,
                    "initial_premium_total": premium,
                    "underlying_price_at_trade": self.current_btc_price,
                    "volatility_at_trade": info.get("iv", config.DEFAULT_VOLATILITY),
                    "timestamp": now
                })

            return trade

        except Exception as e:
            logger.error(f"BTS trade generation error: {e}", exc_info=True)
            return None

    def get_current_activity(self) -> List[Dict[str, Any]]:
        return [
            {
                "trader_type": t.value,
                "count": p["count"],
                "trades_per_hour": p["trades_per_hour"],
                "success_rate": p["success_rate"],
                "timestamp": time.time()
            }
            for t, p in self.trader_profiles.items()
        ]

    def get_recent_trades(self, limit: int = 10) -> List[EnrichedTradeData]:
        return sorted(self.recent_trades_log, key=lambda x: x.timestamp, reverse=True)[:limit]

    def adjust_trader_distribution(self, total_traders: int) -> Dict[str, int]:
        """FIXED: Immediately update trader counts and log the change"""
        adv = int(total_traders * 0.50)
        inter = int(total_traders * 0.35)
        beg = total_traders - adv - inter

        # Update the actual trader profiles used by the trading loop
        self.trader_profiles[TraderType.ADVANCED]["count"] = adv
        self.trader_profiles[TraderType.INTERMEDIATE]["count"] = inter
        self.trader_profiles[TraderType.BEGINNER]["count"] = beg

        # Log the change for immediate verification
        logger.info(f"BTS: Distribution updated - Adv={adv}, Int={inter}, Beg={beg}, Total={total_traders}")
        logger.info(f"BTS: Expected trades/hour: {(adv*25 + inter*17 + beg*8):,}")

        return {"advanced_count": adv, "intermediate_count": inter, "beginner_count": beg, "total_count": total_traders}

    def get_trading_statistics(self) -> Dict[str, Any]:
        """FIXED: Calculate statistics from actual recent trades and accumulated data"""
        try:
            # Use 24-hour cutoff for recent trades
            cutoff = time.time() - 86400
            trades = [t for t in self.recent_trades_log if t.timestamp >= cutoff]

            # CRITICAL FIX: Use accumulated data for accurate statistics
            total_trades_24h = max(len(trades), self.total_trades_executed)
            total_premium_volume = max(
                sum(t.premium_usd for t in trades) if trades else 0,
                self.total_premium_volume_usd_24h
            )

            if not trades and total_trades_24h == 0:
                return {
                    "total_trades_24h": 0,
                    "total_premium_volume_usd_24h": 0.0,
                    "avg_premium_received_usd_24h": 0.0,
                    "call_put_ratio_24h": 0.0,
                    "most_active_expiry_minutes_24h": 0,
                    "current_btc_price": self.current_btc_price,
                    "timestamp": time.time()
                }

            # Calculate ratios from recent trades
            calls = sum(1 for t in trades if t.option_type == "call")
            puts = sum(1 for t in trades if t.option_type == "put")
            ratio = round(calls / puts, 2) if puts > 0 else calls

            # Calculate averages
            avg_prem = round(total_premium_volume / max(total_trades_24h, 1), 2)

            # Find most active expiry
            expiry_counts = defaultdict(int)
            for t in trades:
                expiry_counts[t.expiry_minutes] += 1
            most_active = max(expiry_counts, key=expiry_counts.get) if expiry_counts else 0

            # Log current statistics for debugging
            current_total = sum(p["count"] for p in self.trader_profiles.values())
            logger.debug(f"BTS Stats: {total_trades_24h} trades from {current_total} traders, ${total_premium_volume:,.2f} volume")

            return {
                "total_trades_24h": total_trades_24h,
                "total_premium_volume_usd_24h": round(total_premium_volume, 2),
                "avg_premium_received_usd_24h": avg_prem,
                "call_put_ratio_24h": ratio,
                "most_active_expiry_minutes_24h": most_active,
                "current_btc_price": self.current_btc_price,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"BTS: Error calculating trading statistics: {e}")
            return {
                "total_trades_24h": 0,
                "total_premium_volume_usd_24h": 0.0,
                "avg_premium_received_usd_24h": 0.0,
                "call_put_ratio_24h": 0.0,
                "most_active_expiry_minutes_24h": 0,
                "current_btc_price": self.current_btc_price,
                "error": str(e),
                "timestamp": time.time()
            }

    def get_24h_summary(self) -> Dict[str, Any]:
        """CRITICAL FIX: Get 24h summary using accumulated data"""
        try:
            stats = self.get_trading_statistics()
            
            return {
                "total_trades": stats["total_trades_24h"],
                "total_volume_usd": stats["total_premium_volume_usd_24h"],
                "total_revenue_usd": self.cumulative_revenue,
                "active_traders": sum(p["count"] for p in self.trader_profiles.values()),
                "avg_trades_per_trader": stats["total_trades_24h"] / max(sum(p["count"] for p in self.trader_profiles.values()), 1),
                "uptime_hours": (time.time() - self.start_time) / 3600,
                "trades_per_hour": stats["total_trades_24h"] / max((time.time() - self.start_time) / 3600, 1),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"BTS: Error getting 24h summary: {e}")
            return {
                "total_trades": 0,
                "total_volume_usd": 0.0,
                "total_revenue_usd": 0.0,
                "active_traders": 0,
                "avg_trades_per_trader": 0.0,
                "uptime_hours": 0.0,
                "trades_per_hour": 0.0,
                "error": str(e),
                "timestamp": time.time()
            }

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information about the bot trader simulator state"""
        return {
            "is_running": self.is_running,
            "current_btc_price": self.current_btc_price,
            "trade_id_counter": self.trade_id_counter,
            "total_trades_executed": self.total_trades_executed,
            "total_premium_collected_usd": self.total_premium_collected_usd,
            "daily_volume_usd": self.daily_volume_usd,
            "recent_trades_count": len(self.recent_trades_log),
            "trader_profiles": {t.value: p for t, p in self.trader_profiles.items()},
            "revenue_engine_connected": self.revenue_engine is not None,
            "position_manager_connected": self.position_manager is not None,
            "audit_engine_connected": self.audit_engine is not None,
            "uptime_seconds": time.time() - self.start_time,
            "cache_status": {
                "daily_metrics_keys": list(self.daily_metrics.keys()) if self.daily_metrics else [],
                "trading_stats_cache": self._trading_stats_cache is not None,
                "cached_statistics": self._cached_statistics is not None
            },
            "timestamp": time.time()
        }

    def force_trade_generation(self, count: int = 1) -> List[EnrichedTradeData]:
        """Force generate trades for testing - CAUTION: Only for testing"""
        logger.warning(f"BTS: FORCING {count} TEST TRADES")
        
        trades_generated = []
        for _ in range(count):
            # Use advanced trader profile for testing
            trader_type = TraderType.ADVANCED
            profile = self.trader_profiles[trader_type]
            
            trade = self._generate_and_record_trade(trader_type, profile)
            if trade:
                trades_generated.append(trade)
                
        logger.info(f"BTS: Generated {len(trades_generated)} test trades")
        return trades_generated

    def clear_all_trades(self) -> Dict[str, Any]:
        """Clear all trade history - for testing/debugging only"""
        logger.warning("BTS: CLEARING ALL TRADE HISTORY")
        
        trade_count = len(self.recent_trades_log)
        self.recent_trades_log.clear()
        
        # Reset counters
        self.total_trades_executed = 0
        self.total_premium_collected_usd = 0.0
        self.daily_volume_usd = 0.0
        self.daily_trade_count = 0
        self.total_premium_volume_usd_24h = 0.0
        self.cumulative_revenue = 0.0
        
        return {
            "cleared_trades": trade_count,
            "counters_reset": True,
            "timestamp": time.time()
        }
