# investor_dashboard/liquidity_manager.py (ENHANCED VERSION)

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
    # ← ENHANCED: Add options-specific metrics
    options_exposure_usd: float
    hedge_coverage_ratio: float
    volatility_buffer_usd: float
    risk_level: str

class LiquidityManager:
    """Enhanced liquidity manager for options trading platform."""

    def __init__(self):
        # ← ENHANCED: Dynamic allocation based on risk
        self.liquidity_allocation_pct = 75.0  # Increased for options risk
        self.operations_allocation_pct = 20.0
        self.profit_allocation_pct = 5.0  # Reduced to prioritize safety
        
        # ← ENHANCED: Larger base pool for options platform
        self.base_liquidity_pool = 5_000_000  # $5M base (increased from $1.5M)
        self.active_users = 234
        self.avg_daily_volume = 2_500_000
        
        # ← NEW: Options-specific parameters
        self.avg_option_premium = 1200  # Average option price from revenue engine
        self.open_interest_multiplier = 150  # Contracts * average premium
        self.volatility_multiplier = 3.0  # 300% BTC volatility factor
        self.hedge_efficiency = 0.85  # 85% hedge coverage
        
        # ← NEW: Dynamic scaling parameters
        self.last_pool_update = time.time()
        self.pool_growth_rate = 0.02  # 2% monthly growth
        
    def calculate_options_specific_requirements(self) -> Dict[str, float]:
        """Calculate liquidity requirements specific to options trading."""
        
        # Base options exposure (open contracts * avg premium)
        estimated_open_contracts = self.active_users * 2.5  # 2.5 contracts per user avg
        base_exposure = estimated_open_contracts * self.avg_option_premium
        
        # Volatility buffer (for extreme moves)
        volatility_buffer = base_exposure * (self.volatility_multiplier / 100) * 0.5
        
        # Hedge coverage shortfall buffer
        hedge_shortfall_buffer = base_exposure * (1 - self.hedge_efficiency)
        
        # ITM payout reserve (worst-case scenario)
        itm_payout_reserve = base_exposure * 0.3  # 30% of contracts could go ITM
        
        return {
            "base_exposure": base_exposure,
            "volatility_buffer": volatility_buffer,
            "hedge_shortfall_buffer": hedge_shortfall_buffer,
            "itm_payout_reserve": itm_payout_reserve,
            "total_options_requirement": base_exposure + volatility_buffer + hedge_shortfall_buffer + itm_payout_reserve
        }

    def calculate_required_liquidity(self) -> float:
        """Enhanced liquidity calculation including options risks."""
        
        # Original base calculation
        base_requirement = 2_000_000  # Increased base: $2M minimum
        volume_coverage = self.avg_daily_volume * 0.20  # Increased to 20%
        
        # User scaling with diminishing returns
        user_scaling_factor = math.sqrt(self.active_users / 100)
        user_buffer = 750_000 * user_scaling_factor  # Increased buffer
        
        # Stress test buffer (3x peak volume for options)
        stress_buffer = self.avg_daily_volume * 0.5  # 50% stress buffer
        
        # ← NEW: Add options-specific requirements
        options_requirements = self.calculate_options_specific_requirements()
        options_buffer = options_requirements["total_options_requirement"]
        
        total_required = (base_requirement + volume_coverage + 
                         user_buffer + stress_buffer + options_buffer)
        
        return total_required

    def auto_scale_pool(self):
        """Automatically scale liquidity pool based on requirements."""
        current_time = time.time()
        time_since_update = current_time - self.last_pool_update
        
        # Check if pool needs scaling (monthly check)
        if time_since_update > (30 * 24 * 3600):  # 30 days
            required_liquidity = self.calculate_required_liquidity()
            current_liquidity = self.base_liquidity_pool * (self.liquidity_allocation_pct / 100)
            
            if current_liquidity < required_liquidity:
                # Scale pool to meet requirements + 20% buffer
                new_pool_size = required_liquidity * 1.2 / (self.liquidity_allocation_pct / 100)
                growth_needed = new_pool_size - self.base_liquidity_pool
                
                # Gradual scaling (max 10% increase per month)
                max_growth = self.base_liquidity_pool * 0.1
                actual_growth = min(growth_needed, max_growth)
                
                self.base_liquidity_pool += actual_growth
                self.last_pool_update = current_time
                
                print(f"💰 Auto-scaled liquidity pool by ${actual_growth:,.0f} to ${self.base_liquidity_pool:,.0f}")

    def adjust_allocation(self, liquidity_pct: float, operations_pct: float):
        """Enhanced allocation adjustment with safety checks."""
        profit_pct = 100.0 - liquidity_pct - operations_pct
        
        if profit_pct < 0:
            raise ValueError("Allocation percentages exceed 100%")
        
        # ← NEW: Safety check for options platform
        if liquidity_pct < 60.0:
            raise ValueError("Liquidity allocation cannot be below 60% for options platform safety")
        
        self.liquidity_allocation_pct = liquidity_pct
        self.operations_allocation_pct = operations_pct
        self.profit_allocation_pct = profit_pct
        
        # Trigger auto-scaling check when allocation changes
        self.auto_scale_pool()

    def simulate_user_growth(self, new_user_count: int):
        """Enhanced user growth simulation with pool scaling."""
        old_users = self.active_users
        self.active_users = new_user_count
        
        # Scale volume more realistically
        base_volume_per_user = 8_000  # $8K per user (reduced from $10K)
        network_effect = math.log10(new_user_count / 10) if new_user_count > 10 else 1
        self.avg_daily_volume = new_user_count * base_volume_per_user * network_effect * 0.75
        
        # Update option parameters based on user growth
        self.avg_option_premium = 1200 + (new_user_count / 100) * 50  # Premium increases with platform size
        
        # Trigger auto-scaling
        self.auto_scale_pool()
        
        print(f"👥 User growth: {old_users} → {new_user_count} users")
        print(f"📊 Volume scaled to: ${self.avg_daily_volume:,.0f}")

    def get_status(self) -> LiquidityStatus:
        """Enhanced status with options-specific metrics."""
        
        # Auto-scale check
        self.auto_scale_pool()
        
        required_liquidity = self.calculate_required_liquidity()
        options_data = self.calculate_options_specific_requirements()
        
        # Calculate allocated amounts
        liquidity_amount = self.base_liquidity_pool * (self.liquidity_allocation_pct / 100)
        
        # Enhanced liquidity ratio
        liquidity_ratio = liquidity_amount / required_liquidity if required_liquidity > 0 else 1.0
        
        # Risk assessment
        if liquidity_ratio >= 1.2:
            risk_level = "LOW"
        elif liquidity_ratio >= 1.0:
            risk_level = "MEDIUM"
        elif liquidity_ratio >= 0.7:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Hedge coverage ratio
        hedge_coverage_ratio = self.hedge_efficiency
        
        return LiquidityStatus(
            total_pool_usd=self.base_liquidity_pool,
            allocated_to_liquidity_pct=self.liquidity_allocation_pct,
            allocated_to_operations_pct=self.operations_allocation_pct,
            allocated_to_profit_pct=self.profit_allocation_pct,
            active_users=self.active_users,
            required_liquidity_usd=required_liquidity,
            liquidity_ratio=liquidity_ratio,
            stress_test_buffer_usd=self.avg_daily_volume * 0.5,
            # Enhanced fields
            options_exposure_usd=options_data["base_exposure"],
            hedge_coverage_ratio=hedge_coverage_ratio,
            volatility_buffer_usd=options_data["volatility_buffer"],
            risk_level=risk_level
        )

    def update_metrics(self):
        """Enhanced metrics update with options market data."""
        import random
        
        # More realistic variations
        user_variation = random.randint(-1, 2)  # Smaller variations
        self.active_users = max(50, self.active_users + user_variation)
        
        # Volume variation with volatility correlation
        volatility_factor = random.uniform(0.95, 1.15)  # ±15% variation
        base_volume = 2_500_000
        self.avg_daily_volume = base_volume * volatility_factor
        
        # Update option premium based on market volatility
        premium_variation = random.uniform(0.9, 1.3)  # ±30% variation
        self.avg_option_premium = max(500, self.avg_option_premium * premium_variation)
        
        # Periodic auto-scaling check
        if random.random() < 0.1:  # 10% chance each update
            self.auto_scale_pool()

    def get_liquidity_recommendations(self) -> Dict[str, str]:
        """Provide liquidity management recommendations."""
        status = self.get_status()
        recommendations = []
        
        if status.liquidity_ratio < 0.8:
            recommendations.append("URGENT: Increase liquidity pool or reduce user onboarding")
        elif status.liquidity_ratio < 1.0:
            recommendations.append("WARNING: Liquidity below safe threshold - consider pool increase")
        
        if status.hedge_coverage_ratio < 0.8:
            recommendations.append("Improve hedge coverage to reduce liquidity requirements")
        
        if status.risk_level in ["HIGH", "CRITICAL"]:
            recommendations.append("Consider increasing liquidity allocation percentage")
        
        if not recommendations:
            recommendations.append("Liquidity management is healthy - maintain current strategy")
        
        return {
            "recommendations": recommendations,
            "suggested_pool_size": status.required_liquidity_usd * 1.2,
            "optimal_allocation": "75% liquidity, 20% operations, 5% profit"
        }
