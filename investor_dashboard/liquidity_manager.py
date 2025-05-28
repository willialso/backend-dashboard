# investor_dashboard/liquidity_manager.py (REALISTIC PARAMETERS VERSION)

from dataclasses import dataclass
from typing import Dict
import math
import time

@dataclass
class LiquidityStatus:
    total_pool_usd: float
    allocated_to_liquidity_pct: float
    allocated_to_operations_pct: float
    allocated_to_profit_pct: float
    active_users: int
    required_liquidity_usd: float
    liquidity_ratio: float
    stress_test_buffer_usd: float
    # Enhanced: Add options-specific metrics
    options_exposure_usd: float
    hedge_coverage_ratio: float
    volatility_buffer_usd: float
    risk_level: str

class LiquidityManager:
    """Realistic liquidity manager for BTC options trading platform."""

    def __init__(self):
        # ← REALISTIC: Allocation based on actual trading patterns
        self.liquidity_allocation_pct = 75.0
        self.operations_allocation_pct = 20.0
        self.profit_allocation_pct = 5.0
        
        # ← REALISTIC: Base pool sized for actual trading volumes
        self.base_liquidity_pool = 2_000_000  # $2M base (down from $5M)
        self.max_liquidity_pool = 10_000_000  # $10M maximum cap
        
        # ← REALISTIC: User and volume parameters
        self.active_users = 100  # Start with 100 users
        self.max_users = 1000   # Support up to 1000 users
        
        # ← REALISTIC: Trading volume per user (based on real BTC options data)
        self.avg_trade_per_user_per_day = 750  # $750/user/day (realistic)
        self.avg_daily_volume = self.active_users * self.avg_trade_per_user_per_day
        
        # ← REALISTIC: Options-specific parameters
        self.avg_option_premium = 500    # $500 average (down from $1200)
        self.contracts_per_user = 1.5    # 1.5 contracts per user average (down from 2.5)
        self.volatility_multiplier = 1.5 # 150% volatility factor (down from 300%)
        self.hedge_efficiency = 0.85     # 85% hedge coverage
        
        # Auto-scaling parameters
        self.last_pool_update = time.time()
        self.pool_growth_rate = 0.02  # 2% monthly growth
        
        print(f"💰 Initialized realistic liquidity manager: ${self.base_liquidity_pool:,.0f} pool, {self.active_users} users")

    def calculate_options_specific_requirements(self) -> Dict[str, float]:
        """Calculate realistic liquidity requirements for options trading."""
        
        # ← REALISTIC: Base options exposure
        estimated_open_contracts = self.active_users * self.contracts_per_user
        base_exposure = estimated_open_contracts * self.avg_option_premium
        
        # ← REALISTIC: Volatility buffer (much smaller)
        volatility_buffer = base_exposure * (self.volatility_multiplier / 100) * 0.3  # Reduced factor
        
        # ← REALISTIC: Hedge coverage shortfall buffer
        hedge_shortfall_buffer = base_exposure * (1 - self.hedge_efficiency)
        
        # ← REALISTIC: ITM payout reserve (conservative)
        itm_payout_reserve = base_exposure * 0.2  # 20% of contracts could go ITM (down from 30%)
        
        total_options_requirement = base_exposure + volatility_buffer + hedge_shortfall_buffer + itm_payout_reserve
        
        return {
            "base_exposure": base_exposure,
            "volatility_buffer": volatility_buffer,
            "hedge_shortfall_buffer": hedge_shortfall_buffer,
            "itm_payout_reserve": itm_payout_reserve,
            "total_options_requirement": total_options_requirement
        }

    def calculate_required_liquidity(self) -> float:
        """Realistic liquidity calculation preventing exponential growth."""
        
        # ← REALISTIC: Base requirement
        base_requirement = 500_000  # $500K base (down from $2M)
        
        # ← REALISTIC: Volume coverage (linear scaling)
        daily_volume = self.active_users * self.avg_trade_per_user_per_day
        volume_coverage = daily_volume * 15  # 15x daily volume (industry standard)
        
        # ← FIXED: Linear user scaling with diminishing returns
        if self.active_users <= 100:
            user_multiplier = 1.0
        elif self.active_users <= 300:
            user_multiplier = 1.0 + ((self.active_users - 100) / 200) * 0.5  # Slower growth
        elif self.active_users <= 600:
            user_multiplier = 1.5 + ((self.active_users - 300) / 300) * 0.3  # Even slower
        else:
            user_multiplier = 1.8 + ((self.active_users - 600) / 400) * 0.2  # Minimal growth
        
        user_buffer = 300_000 * user_multiplier  # $300K base with multiplier
        
        # ← REALISTIC: Stress test buffer with cap
        stress_buffer = min(daily_volume * 10, 2_000_000)  # 10x daily volume, capped at $2M
        
        # ← REALISTIC: Options-specific requirements
        options_requirements = self.calculate_options_specific_requirements()
        options_buffer = options_requirements["total_options_requirement"]
        
        # Calculate total
        total_required = base_requirement + volume_coverage + user_buffer + stress_buffer + options_buffer
        
        # ← CRITICAL: Maximum cap to prevent runaway calculations
        MAX_LIQUIDITY_REQUIREMENT = 8_000_000  # $8M maximum (realistic)
        total_required = min(total_required, MAX_LIQUIDITY_REQUIREMENT)
        
        # ← DEBUG: Log if calculations seem high
        if total_required > 5_000_000:
            print(f"⚠️ High liquidity requirement: ${total_required:,.0f} for {self.active_users} users")
            print(f"   Daily volume: ${daily_volume:,.0f}, Options exposure: ${options_buffer:,.0f}")
        
        return total_required

    def auto_scale_pool(self):
        """Smart auto-scaling with realistic limits."""
        required_liquidity = self.calculate_required_liquidity()
        current_liquidity = self.base_liquidity_pool * (self.liquidity_allocation_pct / 100)
        
        # Check if we need more liquidity
        if required_liquidity > current_liquidity:
            # Calculate needed pool size
            needed_pool = required_liquidity * 1.2 / (self.liquidity_allocation_pct / 100)  # 20% buffer
            
            # Cap at maximum pool size
            target_pool = min(needed_pool, self.max_liquidity_pool)
            
            # Gradual scaling (max 20% increase at a time)
            max_increase = self.base_liquidity_pool * 0.2
            actual_increase = min(target_pool - self.base_liquidity_pool, max_increase)
            
            if actual_increase > 0:
                old_pool = self.base_liquidity_pool
                self.base_liquidity_pool += actual_increase
                print(f"💰 Auto-scaled pool: ${old_pool:,.0f} → ${self.base_liquidity_pool:,.0f}")

    def manage_user_growth(self):
        """Realistic user growth management."""
        import random
        
        # ← REALISTIC: Smaller, more realistic variations
        if self.active_users < 200:
            base_variation = random.randint(-1, 3)  # Can grow faster when small
        elif self.active_users < 500:
            base_variation = random.randint(-2, 2)  # Moderate growth
        else:
            base_variation = random.randint(-2, 1)  # Slower growth when large
        
        # Apply variation with bounds
        self.active_users = max(50, min(self.max_users, self.active_users + base_variation))
        
        # Update daily volume based on new user count
        self.avg_daily_volume = self.active_users * self.avg_trade_per_user_per_day
        
        # Auto-scale pool if needed
        self.auto_scale_pool()

    def simulate_user_growth(self, new_user_count: int):
        """Simulate user growth with realistic constraints."""
        if new_user_count > self.max_users:
            print(f"⚠️ User count {new_user_count} exceeds maximum {self.max_users}. Setting to {self.max_users}.")
            new_user_count = self.max_users
        
        if new_user_count < 50:
            print(f"⚠️ User count {new_user_count} below minimum 50. Setting to 50.")
            new_user_count = 50
        
        old_users = self.active_users
        self.active_users = new_user_count
        
        # ← REALISTIC: Update volume based on realistic per-user trading
        self.avg_daily_volume = self.active_users * self.avg_trade_per_user_per_day
        
        # ← REALISTIC: Adjust option parameters gradually
        # Premium increases slightly with platform size (network effect)
        base_premium = 500
        network_bonus = min((self.active_users / 100) * 25, 200)  # Max $200 bonus
        self.avg_option_premium = base_premium + network_bonus
        
        # Trigger auto-scaling
        self.auto_scale_pool()
        
        print(f"👥 User growth: {old_users} → {new_user_count} users")
        print(f"📊 Daily volume: ${self.avg_daily_volume:,.0f}")
        print(f"💰 Pool size: ${self.base_liquidity_pool:,.0f}")

    def adjust_allocation(self, liquidity_pct: float, operations_pct: float):
        """Adjust allocation with safety checks."""
        profit_pct = 100.0 - liquidity_pct - operations_pct
        
        if profit_pct < 0:
            raise ValueError("Allocation percentages exceed 100%")
        
        # Safety check for options platform
        if liquidity_pct < 60.0:
            raise ValueError("Liquidity allocation cannot be below 60% for options platform safety")
        
        if liquidity_pct > 90.0:
            raise ValueError("Liquidity allocation cannot exceed 90% (need operational funds)")
        
        self.liquidity_allocation_pct = liquidity_pct
        self.operations_allocation_pct = operations_pct
        self.profit_allocation_pct = profit_pct
        
        # Trigger scaling check
        self.auto_scale_pool()
        
        print(f"📊 Allocation updated: {liquidity_pct}% liquidity, {operations_pct}% operations, {profit_pct}% profit")

    def get_status(self) -> LiquidityStatus:
        """Get current liquidity status with realistic metrics."""
        
        # Apply user management
        self.manage_user_growth()
        
        required_liquidity = self.calculate_required_liquidity()
        options_data = self.calculate_options_specific_requirements()
        
        # Calculate allocated amounts
        liquidity_amount = self.base_liquidity_pool * (self.liquidity_allocation_pct / 100)
        
        # Calculate liquidity ratio
        liquidity_ratio = liquidity_amount / required_liquidity if required_liquidity > 0 else 1.0
        
        # ← REALISTIC: Risk assessment
        if liquidity_ratio >= 1.5:
            risk_level = "LOW"
        elif liquidity_ratio >= 1.0:
            risk_level = "MEDIUM"
        elif liquidity_ratio >= 0.7:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return LiquidityStatus(
            total_pool_usd=self.base_liquidity_pool,
            allocated_to_liquidity_pct=self.liquidity_allocation_pct,
            allocated_to_operations_pct=self.operations_allocation_pct,
            allocated_to_profit_pct=self.profit_allocation_pct,
            active_users=self.active_users,
            required_liquidity_usd=required_liquidity,
            liquidity_ratio=liquidity_ratio,
            stress_test_buffer_usd=min(self.avg_daily_volume * 10, 2_000_000),
            # Enhanced fields
            options_exposure_usd=options_data["base_exposure"],
            hedge_coverage_ratio=self.hedge_efficiency,
            volatility_buffer_usd=options_data["volatility_buffer"],
            risk_level=risk_level
        )

    def update_metrics(self):
        """Update metrics with realistic market variations."""
        import random
        
        # Realistic user variations (managed by manage_user_growth)
        self.manage_user_growth()
        
        # ← REALISTIC: Volume variation (smaller range)
        volatility_factor = random.uniform(0.95, 1.05)  # ±5% variation (down from ±15%)
        base_volume_per_user = self.avg_trade_per_user_per_day
        self.avg_trade_per_user_per_day = base_volume_per_user * volatility_factor
        
        # Update total daily volume
        self.avg_daily_volume = self.active_users * self.avg_trade_per_user_per_day
        
        # ← REALISTIC: Option premium variation (smaller)
        premium_variation = random.uniform(0.95, 1.05)  # ±5% variation (down from ±30%)
        base_premium = 500 + (self.active_users / 100) * 25  # Base + network effect
        self.avg_option_premium = max(300, base_premium * premium_variation)
        
        # Periodic auto-scaling check (reduced frequency)
        if random.random() < 0.05:  # 5% chance each update (down from 10%)
            self.auto_scale_pool()

    def get_liquidity_recommendations(self) -> Dict[str, str]:
        """Provide realistic liquidity management recommendations."""
        status = self.get_status()
        recommendations = []
        
        if status.liquidity_ratio < 0.7:
            recommendations.append("URGENT: Increase liquidity pool or reduce user onboarding")
        elif status.liquidity_ratio < 1.0:
            recommendations.append("WARNING: Liquidity below safe threshold - consider pool increase")
        elif status.liquidity_ratio > 2.0:
            recommendations.append("INFO: Liquidity pool may be over-provisioned - consider profit allocation increase")
        
        if status.hedge_coverage_ratio < 0.8:
            recommendations.append("Improve hedge coverage to reduce liquidity requirements")
        
        if status.risk_level in ["HIGH", "CRITICAL"]:
            recommendations.append("Consider increasing liquidity allocation percentage")
        
        if status.active_users > 800:
            recommendations.append("High user count - monitor liquidity requirements closely")
        
        if not recommendations:
            recommendations.append("Liquidity management is healthy - maintain current strategy")
        
        return {
            "recommendations": recommendations,
            "suggested_pool_size": max(status.required_liquidity_usd * 1.2, 2_000_000),
            "optimal_allocation": "75% liquidity, 20% operations, 5% profit",
            "max_safe_users": min(1000, int(self.max_liquidity_pool * 0.75 / (self.avg_trade_per_user_per_day * 20)))
        }

    def get_debug_info(self) -> Dict:
        """Get debug information for troubleshooting."""
        return {
            "current_users": self.active_users,
            "max_users": self.max_users,
            "daily_volume": self.avg_daily_volume,
            "avg_trade_per_user": self.avg_trade_per_user_per_day,
            "avg_option_premium": self.avg_option_premium,
            "contracts_per_user": self.contracts_per_user,
            "base_pool": self.base_liquidity_pool,
            "max_pool": self.max_liquidity_pool,
            "pool_utilization_pct": (self.base_liquidity_pool / self.max_liquidity_pool) * 100
        }
