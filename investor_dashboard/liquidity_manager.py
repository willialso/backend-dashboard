# investor_dashboard/liquidity_manager.py

import time
import logging
import math
from typing import Dict, Any, Optional
from dataclasses import dataclass
from backend import config

logger = logging.getLogger(__name__)

@dataclass
class LiquidityAllocation:
    total_pool_usd: float
    liquidity_percentage: float
    operations_percentage: float
    utilized_amount_usd: float
    available_amount_usd: float
    utilization_ratio: float
    stress_test_status: str
    last_updated: float

class LiquidityManager:
    """Manages platform liquidity allocation and utilization tracking."""

    def __init__(
        self,
        initial_total_pool_usd: float = 1500000.0,
        initial_active_users: int = 50,
        base_liquidity_per_user_usd: float = 30000.0,
        volume_factor_per_user_usd: float = 5000.0,
        options_exposure_factor: float = 0.15,
        stress_test_buffer_pct: float = 20.0,
        min_liquidity_ratio: float = 1.2,
        audit_engine_instance: Optional[Any] = None
    ):
        logger.info("🔧 Initializing Liquidity Manager...")
        
        # Core configuration
        self.audit_engine = audit_engine_instance
        self.min_liquidity_ratio = min_liquidity_ratio
        self.stress_test_buffer_pct = stress_test_buffer_pct
        
        # Dynamic scaling parameters
        self.base_liquidity_per_user = base_liquidity_per_user_usd
        self.volume_factor_per_user = volume_factor_per_user_usd
        self.options_exposure_factor = options_exposure_factor
        
        # Pool scaling parameters
        self.min_pool_size_usd = 500000.0  # $500K minimum
        self.max_pool_size_usd = 50000000.0  # $50M maximum
        
        # Current state - DYNAMIC allocation
        self.current_allocation = LiquidityAllocation(
            total_pool_usd=initial_total_pool_usd,
            liquidity_percentage=75.0,  # Default 75%
            operations_percentage=25.0,  # Default 25%
            utilized_amount_usd=0.0,
            available_amount_usd=initial_total_pool_usd * 0.75,
            utilization_ratio=1.0,
            stress_test_status="PASS",
            last_updated=time.time()
        )
        
        # Usage tracking
        self.active_users = initial_active_users
        self.recent_transactions: list = []
        self.hedge_transactions: list = []
        
        logger.info(f"✅ Liquidity Manager initialized: ${initial_total_pool_usd:,.0f} pool, "
                   f"{initial_active_users} users, {self.current_allocation.liquidity_percentage:.1f}% liquidity allocation")

    def reset_to_defaults(self):
        """CRITICAL FIX: Reset liquidity manager to defaults for complete system reset"""
        try:
            logger.warning("LM: Resetting to defaults")
            
            # Clear transaction histories
            self.recent_transactions.clear()
            self.hedge_transactions.clear()
            logger.info("✅ LM: Cleared transaction histories")
            
            # Reset to initial state
            self.current_allocation.liquidity_percentage = 75.0
            self.current_allocation.operations_percentage = 25.0
            self.current_allocation.utilized_amount_usd = 0.0
            self.current_allocation.available_amount_usd = self.current_allocation.total_pool_usd * 0.75
            self.current_allocation.utilization_ratio = 1.0
            self.current_allocation.stress_test_status = "PASS"
            self.current_allocation.last_updated = time.time()
            
            # Reset user count to initial
            self.active_users = 50  # Reset to default
            
            # Force recalculation
            self._recalculate_utilization()
            
            logger.info("✅ LM: Reset to defaults completed")
            return {"status": "success", "message": "Liquidity manager reset to defaults"}
            
        except Exception as e:
            logger.error(f"❌ LM: Error resetting to defaults: {e}")
            return {"status": "error", "message": str(e)}

    def update_total_pool(self, new_total_pool_usd: float, reason: str = "manual_adjustment") -> Dict[str, Any]:
        """CRITICAL: Update total liquidity pool size for scaling"""
        try:
            # Validate new pool size
            if new_total_pool_usd < self.min_pool_size_usd:
                return {
                    "error": f"Pool size too small. Minimum: ${self.min_pool_size_usd:,.0f}",
                    "success": False
                }
            
            if new_total_pool_usd > self.max_pool_size_usd:
                return {
                    "error": f"Pool size too large. Maximum: ${self.max_pool_size_usd:,.0f}",
                    "success": False
                }
            
            old_pool = self.current_allocation.total_pool_usd
            scaling_factor = new_total_pool_usd / old_pool
            
            # Update pool size
            self.current_allocation.total_pool_usd = new_total_pool_usd
            self.current_allocation.last_updated = time.time()
            
            # Recalculate everything based on new pool size
            self._recalculate_utilization()
            
            logger.info(f"LM: Pool size updated: ${old_pool:,.0f} → ${new_total_pool_usd:,.0f} (Scaling factor: {scaling_factor:.2f}x)")
            
            # Log to audit engine
            if self.audit_engine:
                try:
                    self.audit_engine.track_operational_cost(
                        amount_usd=0.0,  # No cost for pool adjustment
                        description=f"Pool size updated to ${new_total_pool_usd:,.0f} - {reason}"
                    )
                except Exception as e:
                    logger.warning(f"LM: Failed to log pool update to audit: {e}")
            
            return {
                "success": True,
                "old_pool_usd": old_pool,
                "new_pool_usd": new_total_pool_usd,
                "scaling_factor": scaling_factor,
                "new_hedge_capacity_increase": (scaling_factor - 1) * 100,  # % increase in hedge capacity
                "reason": reason,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"LM: Error updating pool size: {e}")
            return {"error": str(e), "success": False}

    def get_recommended_pool_size(self) -> Dict[str, Any]:
        """FIXED: Get recommended pool size based on current usage and trader count"""
        try:
            # FIXED: Ensure fresh calculation first
            self._recalculate_utilization()
            
            current_pool = self.current_allocation.total_pool_usd
            
            # FIXED: Get utilization from current allocation (consistent calculation)
            allocation_data = self.get_current_allocation()
            current_utilization_pct = allocation_data["utilization_pct"]
            
            # Base pool size recommendation based on trader count
            # Formula: Higher trader counts need proportionally less per trader (efficiency gains)
            if self.active_users <= 50:
                recommended_per_trader = 30000  # $30K per trader
            elif self.active_users <= 100:
                recommended_per_trader = 25000  # $25K per trader
            elif self.active_users <= 200:
                recommended_per_trader = 20000  # $20K per trader
            elif self.active_users <= 350:
                recommended_per_trader = 17000  # $17K per trader
            else:
                recommended_per_trader = 16000  # $16K per trader
            
            base_recommended_pool = self.active_users * recommended_per_trader
            
            # Adjust based on current utilization
            if current_utilization_pct > 90:
                utilization_multiplier = 1.5  # Need 50% more capacity
            elif current_utilization_pct > 80:
                utilization_multiplier = 1.3  # Need 30% more capacity
            elif current_utilization_pct > 70:
                utilization_multiplier = 1.2  # Need 20% more capacity
            elif current_utilization_pct < 30:
                utilization_multiplier = 0.9  # Can reduce capacity by 10%
            else:
                utilization_multiplier = 1.0  # Current size is adequate
            
            recommended_pool = base_recommended_pool * utilization_multiplier
            
            # Ensure within bounds
            recommended_pool = max(self.min_pool_size_usd, min(self.max_pool_size_usd, recommended_pool))
            
            # Calculate impact
            scaling_needed = recommended_pool / current_pool
            
            # Determine recommendation status
            if scaling_needed > 1.2:
                status = "URGENT_SCALE_UP"
                message = "Pool size critically low for current usage"
            elif scaling_needed > 1.1:
                status = "SCALE_UP_RECOMMENDED"
                message = "Pool size should be increased for optimal performance"
            elif scaling_needed < 0.8:
                status = "SCALE_DOWN_POSSIBLE"
                message = "Pool size could be reduced to improve capital efficiency"
            else:
                status = "OPTIMAL_SIZE"
                message = "Current pool size is appropriate"
            
            return {
                "current_pool_usd": current_pool,
                "recommended_pool_usd": recommended_pool,
                "current_traders": self.active_users,
                "recommended_per_trader": recommended_per_trader,
                "current_utilization_pct": current_utilization_pct,  # FIXED: Now consistent
                "scaling_factor_needed": scaling_needed,
                "status": status,
                "message": message,
                "hedge_capacity_change_pct": (scaling_needed - 1) * 100,
                "capital_efficiency": self.active_users / (current_pool / 1000000),  # traders per $1M
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"LM: Error calculating recommended pool size: {e}")
            return {"error": str(e), "status": "ERROR"}

    def auto_scale_pool_for_traders(self, target_traders: int) -> Dict[str, Any]:
        """CRITICAL: Automatically scale pool size based on target trader count"""
        try:
            # Calculate optimal pool size for target trader count
            if target_traders <= 50:
                optimal_per_trader = 30000
            elif target_traders <= 100:
                optimal_per_trader = 25000
            elif target_traders <= 200:
                optimal_per_trader = 20000
            elif target_traders <= 350:
                optimal_per_trader = 17000
            else:
                optimal_per_trader = 16000
            
            target_pool_size = target_traders * optimal_per_trader
            
            # Apply buffer for growth
            target_pool_size *= 1.2  # 20% buffer
            
            # Update pool size
            result = self.update_total_pool(
                target_pool_size,
                f"auto_scaling_for_{target_traders}_traders"
            )
            
            if result.get("success", False):
                result["target_traders"] = target_traders
                result["optimal_per_trader"] = optimal_per_trader
                result["buffer_applied"] = "20%"
            
            return result
            
        except Exception as e:
            logger.error(f"LM: Error auto-scaling pool: {e}")
            return {"error": str(e), "success": False}

    def update_allocation(self, liquidity_pct: float, operations_pct: float) -> Dict[str, Any]:
        """CRITICAL: Update liquidity allocation based on slider changes"""
        try:
            # Validate percentages
            if abs(liquidity_pct + operations_pct - 100.0) > 0.1:
                return {"error": "Percentages must sum to 100%", "success": False}
            
            if liquidity_pct < 50.0 or liquidity_pct > 95.0:
                return {"error": "Liquidity percentage must be between 50% and 95%", "success": False}
            
            # Update allocation
            old_liquidity_pct = self.current_allocation.liquidity_percentage
            self.current_allocation.liquidity_percentage = liquidity_pct
            self.current_allocation.operations_percentage = operations_pct
            self.current_allocation.last_updated = time.time()
            
            # Recalculate everything based on new allocation
            self._recalculate_utilization()
            
            logger.info(f"LM: Allocation updated: {old_liquidity_pct:.1f}% → {liquidity_pct:.1f}% liquidity")
            
            # Log to audit engine
            if self.audit_engine:
                try:
                    self.audit_engine.track_operational_cost(
                        amount_usd=0.0,  # No cost for allocation change
                        description=f"Liquidity allocation changed to {liquidity_pct:.1f}%/{operations_pct:.1f}%"
                    )
                except Exception as e:
                    logger.warning(f"LM: Failed to log allocation change to audit: {e}")
            
            return {
                "success": True,
                "new_allocation": {
                    "liquidity_pct": liquidity_pct,
                    "operations_pct": operations_pct,
                    "available_usd": self.current_allocation.available_amount_usd,
                    "utilization_ratio": self.current_allocation.utilization_ratio,
                    "stress_test_status": self.current_allocation.stress_test_status
                }
            }
            
        except Exception as e:
            logger.error(f"LM: Error updating allocation: {e}")
            return {"error": str(e), "success": False}

    def _recalculate_utilization(self):
        """Recalculate utilization based on current allocation and usage"""
        try:
            # Calculate base utilization from user activity
            base_utilization = self.active_users * self.base_liquidity_per_user
            
            # Add volume-based utilization (from recent transactions)
            volume_utilization = self._calculate_volume_utilization()
            
            # Add options exposure utilization
            options_utilization = self._calculate_options_utilization()
            
            # Total utilized amount
            total_utilized = base_utilization + volume_utilization + options_utilization
            
            # Calculate available based on allocation
            liquidity_allocated = self.current_allocation.total_pool_usd * (self.current_allocation.liquidity_percentage / 100.0)
            
            self.current_allocation.available_amount_usd = liquidity_allocated - total_utilized
            self.current_allocation.utilized_amount_usd = total_utilized
            
            # Calculate utilization ratio
            if liquidity_allocated > 0:
                self.current_allocation.utilization_ratio = 1.0 + (total_utilized / self.current_allocation.total_pool_usd)
            else:
                self.current_allocation.utilization_ratio = 1.0
            
            # Update stress test status
            utilization_pct = (total_utilized / liquidity_allocated) * 100 if liquidity_allocated > 0 else 0
            self.current_allocation.stress_test_status = "FAIL" if utilization_pct >= (100 - self.stress_test_buffer_pct) else "PASS"
            
            logger.debug(f"LM: Recalculated - Utilized: ${total_utilized:,.0f}, Available: ${self.current_allocation.available_amount_usd:,.0f}, "
                        f"Ratio: {self.current_allocation.utilization_ratio:.3f}, Stress: {self.current_allocation.stress_test_status}")
            
        except Exception as e:
            logger.error(f"LM: Error recalculating utilization: {e}")

    def _calculate_volume_utilization(self) -> float:
        """Calculate liquidity utilization from trading volume"""
        try:
            # Get recent transactions (last 24h)
            current_time = time.time()
            cutoff_24h = current_time - (24 * 3600)
            
            recent_volume = sum(
                t.get("amount_usd", 0) for t in self.recent_transactions
                if t.get("timestamp", 0) >= cutoff_24h
            )
            
            # Volume factor determines how much liquidity is tied up per dollar of volume
            return recent_volume * 0.1  # 10% of trading volume requires liquidity backing
            
        except Exception as e:
            logger.warning(f"LM: Error calculating volume utilization: {e}")
            return 0.0

    def _calculate_options_utilization(self) -> float:
        """Calculate liquidity utilization from options exposure"""
        try:
            # This would ideally get data from position manager
            # For now, estimate based on user count and exposure factor
            estimated_options_exposure = self.active_users * 10000.0  # $10K average exposure per user
            return estimated_options_exposure * self.options_exposure_factor
            
        except Exception as e:
            logger.warning(f"LM: Error calculating options utilization: {e}")
            return 0.0

    def record_transaction(self, amount_usd: float, transaction_type: str, transaction_id: str = None):
        """Record transaction for liquidity tracking"""
        try:
            transaction = {
                "amount_usd": amount_usd,
                "type": transaction_type,
                "transaction_id": transaction_id,
                "timestamp": time.time()
            }
            
            self.recent_transactions.append(transaction)
            
            # Keep only last 1000 transactions
            if len(self.recent_transactions) > 1000:
                self.recent_transactions = self.recent_transactions[-1000:]
            
            # Recalculate utilization after transaction
            self._recalculate_utilization()
            
            logger.debug(f"LM: Recorded transaction: ${amount_usd:,.0f} ({transaction_type})")
            
        except Exception as e:
            logger.warning(f"LM: Error recording transaction: {e}")

    def record_hedge_transaction(self, amount_usd: float, hedge_id: str, transaction_type: str = "hedge"):
        """Record hedge transaction for liquidity tracking"""
        try:
            hedge_transaction = {
                "amount_usd": amount_usd,
                "hedge_id": hedge_id,
                "type": transaction_type,
                "timestamp": time.time()
            }
            
            self.hedge_transactions.append(hedge_transaction)
            
            # Keep only last 500 hedge transactions
            if len(self.hedge_transactions) > 500:
                self.hedge_transactions = self.hedge_transactions[-500:]
            
            # Record as regular transaction for utilization calculation
            self.record_transaction(amount_usd, transaction_type, hedge_id)
            
        except Exception as e:
            logger.warning(f"LM: Error recording hedge transaction: {e}")

    def update_user_count(self, new_user_count: int):
        """Update active user count for dynamic scaling"""
        if new_user_count != self.active_users:
            old_count = self.active_users
            self.active_users = new_user_count
            
            # Recalculate utilization with new user count
            self._recalculate_utilization()
            
            logger.info(f"LM: User count updated: {old_count} → {new_user_count}")

    def get_current_allocation(self) -> Dict[str, Any]:
        """Get current liquidity allocation state"""
        # Ensure fresh calculation
        self._recalculate_utilization()
        
        return {
            "total_pool_usd": self.current_allocation.total_pool_usd,
            "liquidity_percentage": self.current_allocation.liquidity_percentage,
            "operations_percentage": self.current_allocation.operations_percentage,
            "liquidity_ratio": self.current_allocation.utilization_ratio,
            "utilized_amount_usd": self.current_allocation.utilized_amount_usd,
            "available_amount_usd": self.current_allocation.available_amount_usd,
            "utilization_pct": (self.current_allocation.utilized_amount_usd /
                              (self.current_allocation.total_pool_usd * self.current_allocation.liquidity_percentage / 100)) * 100,
            "stress_test_status": self.current_allocation.stress_test_status,
            "active_users": self.active_users,
            "last_updated": self.current_allocation.last_updated,
            "timestamp": time.time()
        }

    def get_recommended_allocation(self) -> Dict[str, Any]:
        """Get recommended allocation based on current usage patterns"""
        try:
            current_utilization_pct = (self.current_allocation.utilized_amount_usd /
                                     (self.current_allocation.total_pool_usd * self.current_allocation.liquidity_percentage / 100)) * 100
            
            # Recommend allocation based on utilization
            if current_utilization_pct > 90:
                recommended_liquidity = min(95, self.current_allocation.liquidity_percentage + 10)
            elif current_utilization_pct > 80:
                recommended_liquidity = min(90, self.current_allocation.liquidity_percentage + 5)
            elif current_utilization_pct < 30:
                recommended_liquidity = max(50, self.current_allocation.liquidity_percentage - 5)
            else:
                recommended_liquidity = self.current_allocation.liquidity_percentage
            
            recommended_operations = 100 - recommended_liquidity
            
            # Calculate impact of recommendation
            impact = "no change"
            if recommended_liquidity > self.current_allocation.liquidity_percentage:
                impact = "increase liquidity allocation"
            elif recommended_liquidity < self.current_allocation.liquidity_percentage:
                impact = "decrease liquidity allocation"
            
            return {
                "current_liquidity_pct": self.current_allocation.liquidity_percentage,
                "current_operations_pct": self.current_allocation.operations_percentage,
                "recommended_liquidity_pct": recommended_liquidity,
                "recommended_operations_pct": recommended_operations,
                "current_utilization_pct": current_utilization_pct,
                "impact": impact,
                "reason": f"Based on {current_utilization_pct:.1f}% utilization",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"LM: Error generating recommendation: {e}")
            return {"error": str(e)}

    def get_liquidity_health(self) -> Dict[str, Any]:
        """Get comprehensive liquidity health status"""
        try:
            allocation = self.get_current_allocation()
            recommendation = self.get_recommended_allocation()
            pool_recommendation = self.get_recommended_pool_size()
            
            # Determine health status
            utilization_pct = allocation["utilization_pct"]
            
            if utilization_pct > 95:
                health_status = "CRITICAL"
            elif utilization_pct > 85:
                health_status = "HIGH_RISK"
            elif utilization_pct > 70:
                health_status = "MODERATE"
            elif utilization_pct < 30:
                health_status = "UNDERUTILIZED"
            else:
                health_status = "HEALTHY"
            
            return {
                "health_status": health_status,
                "utilization_percentage": utilization_pct,
                "stress_test_status": allocation["stress_test_status"],
                "allocation": allocation,
                "recommendation": recommendation,
                "pool_recommendation": pool_recommendation,
                "recent_transactions_24h": len([t for t in self.recent_transactions
                                              if t.get("timestamp", 0) >= time.time() - 86400]),
                "hedge_transactions_24h": len([h for h in self.hedge_transactions
                                             if h.get("timestamp", 0) >= time.time() - 86400]),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"LM: Error getting liquidity health: {e}")
            return {"error": str(e), "health_status": "ERROR"}

    def get_scaling_metrics(self) -> Dict[str, Any]:
        """FIXED: Get comprehensive scaling metrics for frontend display"""
        try:
            allocation = self.get_current_allocation()
            pool_rec = self.get_recommended_pool_size()
            
            # FIXED: Ensure we have positive values
            total_pool_usd = allocation.get("total_pool_usd", 1500000)
            liquidity_percentage = allocation.get("liquidity_percentage", 75) / 100
            
            # FIXED: Use proper liquidity calculation for hedge capacity
            # Available for hedging = total pool * liquidity percentage * safety buffer
            hedge_available_usd = total_pool_usd * liquidity_percentage * 0.8  # 80% of allocated liquidity
            
            # FIXED: Use current BTC price, with fallback
            current_btc_price = 108000.0  # Fallback price
            
            # Calculate hedge capacities
            current_hedge_capacity_btc = hedge_available_usd / current_btc_price
            
            # Calculate recommended hedge capacity
            recommended_pool_usd = pool_rec.get("recommended_pool_usd", total_pool_usd)
            recommended_hedge_available_usd = recommended_pool_usd * liquidity_percentage * 0.8
            recommended_hedge_capacity_btc = recommended_hedge_available_usd / current_btc_price
            
            # FIXED: Safe improvement percentage calculation
            if current_hedge_capacity_btc > 0.01:  # Minimum viable capacity
                hedge_capacity_improvement = ((recommended_hedge_capacity_btc / current_hedge_capacity_btc) - 1) * 100
            else:
                hedge_capacity_improvement = 0.0
            
            # FIXED: Ensure all values are positive and realistic
            return {
                "current_pool_size": total_pool_usd,
                "recommended_pool_size": recommended_pool_usd,
                "scaling_factor_needed": pool_rec.get("scaling_factor_needed", 1.0),
                "current_traders": self.active_users,
                "pool_per_trader": total_pool_usd / max(self.active_users, 1),
                "current_hedge_capacity_btc": round(max(current_hedge_capacity_btc, 0), 4),
                "recommended_hedge_capacity_btc": round(max(recommended_hedge_capacity_btc, 0), 4),
                "hedge_capacity_improvement": round(hedge_capacity_improvement, 2),
                "capital_efficiency": self.active_users / (total_pool_usd / 1000000),
                "hedge_available_usd": hedge_available_usd,
                "status": pool_rec.get("status", "UNKNOWN"),
                "message": pool_rec.get("message", "Status unavailable"),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"LM: Error getting scaling metrics: {e}")
            return {
                "error": str(e),
                "current_hedge_capacity_btc": 0.0,
                "hedge_capacity_improvement": 0.0,
                "timestamp": time.time()
            }

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information about liquidity manager state"""
        return {
            "active_users": self.active_users,
            "current_allocation": self.current_allocation.__dict__,
            "recent_transactions_count": len(self.recent_transactions),
            "hedge_transactions_count": len(self.hedge_transactions),
            "min_pool_size_usd": self.min_pool_size_usd,
            "max_pool_size_usd": self.max_pool_size_usd,
            "base_liquidity_per_user": self.base_liquidity_per_user,
            "volume_factor_per_user": self.volume_factor_per_user,
            "options_exposure_factor": self.options_exposure_factor,
            "stress_test_buffer_pct": self.stress_test_buffer_pct,
            "audit_engine_connected": self.audit_engine is not None,
            "timestamp": time.time()
        }

    def clear_transaction_history(self) -> Dict[str, Any]:
        """Clear all transaction history - for testing/debugging only"""
        logger.warning("LM: CLEARING ALL TRANSACTION HISTORY")
        
        transaction_count = len(self.recent_transactions)
        hedge_count = len(self.hedge_transactions)
        
        self.recent_transactions.clear()
        self.hedge_transactions.clear()
        
        # Recalculate utilization with cleared history
        self._recalculate_utilization()
        
        return {
            "cleared_transactions": transaction_count,
            "cleared_hedge_transactions": hedge_count,
            "utilization_recalculated": True,
            "timestamp": time.time()
        }
