# investor_dashboard/liquidity_manager.py

from dataclasses import dataclass
from typing import Dict
import math

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

class LiquidityManager:
    """Manages platform liquidity allocation and requirements."""
    
    def __init__(self):
        # Default allocation percentages (adjustable)
        self.liquidity_allocation_pct = 70.0
        self.operations_allocation_pct = 20.0
        self.profit_allocation_pct = 10.0
        
        # Platform metrics
        self.base_liquidity_pool = 1_500_000  # $1.5M base pool
        self.active_users = 234  # Simulated user count
        self.avg_daily_volume = 2_500_000  # $2.5M daily volume
        
    def calculate_required_liquidity(self) -> float:
        """Calculate required liquidity based on user count and volume."""
        # Professional liquidity model
        # Base + Volume Coverage + Risk Buffer
        
        base_requirement = 1_000_000  # $1M minimum
        
        # Volume coverage: 15% of daily volume
        volume_coverage = self.avg_daily_volume * 0.15
        
        # User scaling with square root (diversification effect)
        user_scaling_factor = math.sqrt(self.active_users / 100)  # Baseline 100 users
        user_buffer = 500_000 * user_scaling_factor
        
        # Stress test buffer (2x peak volume days)
        stress_buffer = self.avg_daily_volume * 0.3
        
        return base_requirement + volume_coverage + user_buffer + stress_buffer
    
    def adjust_allocation(self, liquidity_pct: float, operations_pct: float):
        """Adjust allocation percentages (investor controls)."""
        # Ensure percentages add to 100%
        profit_pct = 100.0 - liquidity_pct - operations_pct
        
        if profit_pct < 0:
            raise ValueError("Allocation percentages exceed 100%")
            
        self.liquidity_allocation_pct = liquidity_pct
        self.operations_allocation_pct = operations_pct
        self.profit_allocation_pct = profit_pct
    
    def simulate_user_growth(self, new_user_count: int):
        """Simulate platform scaling with more users."""
        self.active_users = new_user_count
        
        # Scale volume with user growth (sub-linear)
        base_volume_per_user = 10_000  # $10K per user average
        self.avg_daily_volume = new_user_count * base_volume_per_user * 0.8  # 80% efficiency
    
    def get_status(self) -> LiquidityStatus:
        """Get current liquidity status for dashboard."""
        required_liquidity = self.calculate_required_liquidity()
        
        # Calculate allocated amounts
        liquidity_amount = self.base_liquidity_pool * (self.liquidity_allocation_pct / 100)
        
        # Liquidity ratio (how well covered we are)
        liquidity_ratio = liquidity_amount / required_liquidity if required_liquidity > 0 else 1.0
        
        # Stress test buffer
        stress_buffer = self.avg_daily_volume * 0.5  # 50% of daily volume
        
        return LiquidityStatus(
            total_pool_usd=self.base_liquidity_pool,
            allocated_to_liquidity_pct=self.liquidity_allocation_pct,
            allocated_to_operations_pct=self.operations_allocation_pct,
            allocated_to_profit_pct=self.profit_allocation_pct,
            active_users=self.active_users,
            required_liquidity_usd=required_liquidity,
            liquidity_ratio=liquidity_ratio,
            stress_test_buffer_usd=stress_buffer
        )
    
    def update_metrics(self):
        """Update liquidity metrics (called by background loop)."""
        # Simulate minor fluctuations in user count and volume
        import random
        
        # Small random variations
        user_variation = random.randint(-2, 3)
        self.active_users = max(50, self.active_users + user_variation)
        
        # Volume variation
        volume_variation = random.uniform(-0.05, 0.05)  # ±5%
        base_volume = 2_500_000
        self.avg_daily_volume = base_volume * (1 + volume_variation)
