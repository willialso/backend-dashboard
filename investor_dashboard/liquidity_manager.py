# investor_dashboard/liquidity_manager.py

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
import random

from backend import config

logger = logging.getLogger(__name__)

@dataclass
class LiquidityStatus:
    total_pool_usd: float
    available_liquidity_usd: float
    reserved_liquidity_usd: float
    utilization_percentage: float
    active_users: int
    required_liquidity_usd: float
    liquidity_ratio: float
    stress_test_buffer_usd: float
    status_message: str
    last_update_timestamp: float

class LiquidityManager:
    def __init__(self,
                 initial_total_pool_usd: float = config.LM_INITIAL_TOTAL_POOL_USD,
                 initial_active_users: int = config.LM_INITIAL_ACTIVE_USERS,
                 base_liquidity_per_user_usd: float = config.LM_BASE_LIQUIDITY_PER_USER_USD,
                 volume_factor_per_user_usd: float = config.LM_VOLUME_FACTOR_PER_USER_USD,
                 options_exposure_factor: float = config.LM_OPTIONS_EXPOSURE_FACTOR,
                 stress_test_buffer_pct: float = config.LM_STRESS_TEST_BUFFER_PCT,
                 min_liquidity_ratio: float = config.LM_MIN_LIQUIDITY_RATIO,
                 audit_engine_instance: Optional[Any] = None):
        
        # Store configuration parameters
        self.total_pool_usd = initial_total_pool_usd
        self.active_users = initial_active_users
        self.base_liquidity_per_user = base_liquidity_per_user_usd
        self.volume_factor_per_user = volume_factor_per_user_usd
        self.options_exposure_factor = options_exposure_factor
        self.stress_test_buffer_pct = stress_test_buffer_pct
        self.min_liquidity_ratio = min_liquidity_ratio
        self.audit_engine = audit_engine_instance
        
        # Initialize state
        self.reserved_liquidity_usd = 0.0
        self.last_update_time = time.time()
        
        # Calculate initial required liquidity
        self.required_liquidity_usd = self._calculate_required_liquidity()
        
        logger.info(f"LiquidityManager initialized. Pool: ${self.total_pool_usd:,.2f}, Users: {self.active_users}, Min Ratio: {self.min_liquidity_ratio}")

    def _calculate_required_liquidity(self) -> float:
        """Calculate the required liquidity based on current state."""
        # Base liquidity requirement
        base_requirement = self.active_users * self.base_liquidity_per_user
        
        # Volume-based requirement
        volume_requirement = self.active_users * self.volume_factor_per_user
        
        # Options exposure requirement (placeholder - would be calculated from actual positions)
        # For now, use a simple estimate based on active users and exposure factor
        estimated_options_exposure = self.active_users * self.base_liquidity_per_user * self.options_exposure_factor
        
        total_required = base_requirement + volume_requirement + estimated_options_exposure
        
        # Add stress test buffer
        stress_buffer = total_required * self.stress_test_buffer_pct
        
        return total_required + stress_buffer

    def update_metrics(self) -> None:
        """Update liquidity metrics and status."""
        current_time = time.time()
        
        # Recalculate required liquidity
        self.required_liquidity_usd = self._calculate_required_liquidity()
        
        # Add some realistic variation to simulate market conditions
        if hasattr(self, '_last_variation_time'):
            time_since_last = current_time - self._last_variation_time
            if time_since_last > 300:  # Update every 5 minutes
                variation = random.uniform(-0.02, 0.02)  # ±2% variation
                self.total_pool_usd *= (1 + variation)
                self._last_variation_time = current_time
        else:
            self._last_variation_time = current_time
        
        self.last_update_time = current_time
        
        # Log periodic updates
        if not hasattr(self, '_last_log_time') or (current_time - self._last_log_time) > 3600:  # Log every hour
            status = self.get_liquidity_status()
            logger.info(f"LM: Pool=${status.total_pool_usd:,.2f}, Ratio={status.liquidity_ratio:.2f}, Util={status.utilization_percentage:.1f}%")
            self._last_log_time = current_time

    def get_liquidity_status(self) -> LiquidityStatus:
        """Get current liquidity status."""
        available_liquidity = self.total_pool_usd - self.reserved_liquidity_usd
        
        # Calculate liquidity ratio
        liquidity_ratio = self.total_pool_usd / self.required_liquidity_usd if self.required_liquidity_usd > 0 else float('inf')
        
        # Calculate utilization percentage
        utilization_pct = (self.reserved_liquidity_usd / self.total_pool_usd) * 100 if self.total_pool_usd > 0 else 0
        
        # Calculate stress test buffer
        stress_buffer = self.required_liquidity_usd * self.stress_test_buffer_pct
        
        # Determine status message
        if liquidity_ratio >= self.min_liquidity_ratio:
            if liquidity_ratio >= self.min_liquidity_ratio * 1.5:
                status_msg = "Excellent - High Liquidity"
            else:
                status_msg = "Good - Adequate Liquidity"
        elif liquidity_ratio >= self.min_liquidity_ratio * 0.8:
            status_msg = "Warning - Low Liquidity"
        else:
            status_msg = "Critical - Insufficient Liquidity"
        
        return LiquidityStatus(
            total_pool_usd=round(self.total_pool_usd, 2),
            available_liquidity_usd=round(available_liquidity, 2),
            reserved_liquidity_usd=round(self.reserved_liquidity_usd, 2),
            utilization_percentage=round(utilization_pct, 2),
            active_users=self.active_users,
            required_liquidity_usd=round(self.required_liquidity_usd, 2),
            liquidity_ratio=round(liquidity_ratio, 3),
            stress_test_buffer_usd=round(stress_buffer, 2),
            status_message=status_msg,
            last_update_timestamp=self.last_update_time
        )

    def reserve_liquidity(self, amount_usd: float, reason: str = "General Reserve") -> bool:
        """Reserve liquidity for a specific purpose."""
        if amount_usd <= 0:
            return False
        
        available = self.total_pool_usd - self.reserved_liquidity_usd
        if available >= amount_usd:
            self.reserved_liquidity_usd += amount_usd
            logger.info(f"LM: Reserved ${amount_usd:,.2f} for {reason}. Total reserved: ${self.reserved_liquidity_usd:,.2f}")
            return True
        else:
            logger.warning(f"LM: Cannot reserve ${amount_usd:,.2f} for {reason}. Available: ${available:,.2f}")
            return False

    def release_liquidity(self, amount_usd: float, reason: str = "General Release") -> bool:
        """Release previously reserved liquidity."""
        if amount_usd <= 0 or amount_usd > self.reserved_liquidity_usd:
            return False
        
        self.reserved_liquidity_usd -= amount_usd
        logger.info(f"LM: Released ${amount_usd:,.2f} for {reason}. Total reserved: ${self.reserved_liquidity_usd:,.2f}")
        return True

    def add_liquidity(self, amount_usd: float, source: str = "External") -> None:
        """Add liquidity to the pool."""
        if amount_usd > 0:
            self.total_pool_usd += amount_usd
            logger.info(f"LM: Added ${amount_usd:,.2f} from {source}. Total pool: ${self.total_pool_usd:,.2f}")

    def update_active_users(self, new_user_count: int) -> None:
        """Update the active user count."""
        if new_user_count >= 0:
            old_count = self.active_users
            self.active_users = new_user_count
            logger.info(f"LM: Updated active users: {old_count} -> {new_user_count}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of key liquidity metrics."""
        status = self.get_liquidity_status()
        
        return {
            "pool_total_usd": status.total_pool_usd,
            "available_usd": status.available_liquidity_usd,
            "utilization_pct": status.utilization_percentage,
            "liquidity_ratio": status.liquidity_ratio,
            "active_users": status.active_users,
            "status": status.status_message,
            "last_update": status.last_update_timestamp,
            "meets_minimum_ratio": status.liquidity_ratio >= self.min_liquidity_ratio
        }

    def stress_test_liquidity(self, scenario_multiplier: float = 2.0) -> Dict[str, Any]:
        """Perform a stress test on liquidity requirements."""
        current_required = self.required_liquidity_usd
        stress_required = current_required * scenario_multiplier
        stress_ratio = self.total_pool_usd / stress_required if stress_required > 0 else float('inf')
        
        return {
            "scenario_multiplier": scenario_multiplier,
            "current_required_usd": current_required,
            "stress_required_usd": stress_required,
            "current_ratio": self.total_pool_usd / current_required if current_required > 0 else float('inf'),
            "stress_ratio": stress_ratio,
            "passes_stress_test": stress_ratio >= self.min_liquidity_ratio,
            "additional_liquidity_needed_usd": max(0, stress_required - self.total_pool_usd)
        }
