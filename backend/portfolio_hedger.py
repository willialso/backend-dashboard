# backend/portfolio_hedger.py

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import time # Changed from pandas for timestamp, can revert if pd.Timestamp is preferred
# import pandas as pd # If you prefer pandas Timestamps

from backend import config
from backend.utils import setup_logger
# --- ADD IMPORT FOR TYPE HINTING ---
from backend.advanced_pricing_engine import AdvancedPricingEngine # Assuming this is the correct path and class name
# --- END ADD IMPORT ---

logger = setup_logger(__name__)

@dataclass
class Position: # Assuming this dataclass remains as you provided
    position_id: str; user_id: str; option_type: str; strike: float; expiry_minutes: int
    quantity: int; entry_price_usd: float; entry_timestamp: float
    current_delta: float = 0.0; current_gamma: float = 0.0
    current_theta: float = 0.0; current_vega: float = 0.0
    mark_to_market_usd: float = 0.0

@dataclass
class HedgeExecution: # Assuming this dataclass remains as you provided
    timestamp: float; btc_quantity: float; btc_price: float
    hedge_cost_usd: float; reason: str

@dataclass
class PortfolioRisk: # Assuming this dataclass remains as you provided
    net_delta: float = 0.0; net_gamma: float = 0.0; net_theta: float = 0.0
    net_vega: float = 0.0; total_exposure_usd: float = 0.0; total_positions: int = 0
    hedging_pnl: float = 0.0; option_pnl: float = 0.0
    hedge_executions: List[HedgeExecution] = field(default_factory=list)

class PortfolioHedger:
    """Manages platform's option portfolio and hedging strategy."""

    # --- MODIFIED __init__ ---
    def __init__(self, pricing_engine: AdvancedPricingEngine):
        self.pricing_engine = pricing_engine # Store the pricing engine instance
    # --- END MODIFICATION ---
        self.positions: Dict[str, Position] = {}
        self.hedge_btc_position: float = 0.0 # Net BTC held for hedging
        self.hedge_cost_basis: float = 0.0 # Average cost of hedge BTC
        self.risk_metrics = PortfolioRisk() # Initialize with default values
        self.last_hedge_timestamp: float = 0.0
        logger.info("PortfolioHedger initialized.")

    def add_position(self, position: Position) -> None:
        """Add new position to portfolio."""
        self.positions[position.position_id] = position
        logger.info(f"Added position {position.position_id}: {position.quantity} {position.option_type} @ {position.strike}")
        self.calculate_portfolio_risk() # Recalculate risk on new position

    def remove_position(self, position_id: str) -> Optional[Position]:
        """Remove position (e.g., on expiry or closure)."""
        removed_pos = self.positions.pop(position_id, None)
        if removed_pos:
            logger.info(f"Removed position {position_id}")
            self.calculate_portfolio_risk() # Recalculate risk
        return removed_pos

    def update_position_greeks_and_mark(self, position_id: str, 
                                  delta_per_contract: float, gamma_per_contract: float,
                                  theta_per_contract: float, vega_per_contract: float, 
                                  mark_price_usd_per_contract: float) -> None:
        """Update position's Greeks and mark-to-market price (all per contract)."""
        if position_id in self.positions:
            pos = self.positions[position_id]
            # Greeks are per contract, so multiply by quantity for total position greek
            # Note: pos.quantity is signed (negative for short)
            pos.current_delta = delta_per_contract * pos.quantity
            pos.current_gamma = gamma_per_contract * pos.quantity 
            pos.current_theta = theta_per_contract * pos.quantity
            pos.current_vega = vega_per_contract * pos.quantity
            pos.mark_to_market_usd = mark_price_usd_per_contract # Mark price is per contract
            # logger.debug(f"Updated greeks/mark for position {position_id}")
        else:
            logger.warning(f"Attempted to update non-existent position: {position_id}")

    def calculate_portfolio_risk(self) -> PortfolioRisk:
        """Calculate aggregated portfolio risk metrics."""
        if not self.positions: # If no positions, risk is zero
            self.risk_metrics = PortfolioRisk(hedge_executions=self.risk_metrics.hedge_executions) # Preserve history
            return self.risk_metrics

        current_btc_price = self.pricing_engine.current_price if self.pricing_engine and self.pricing_engine.current_price > 0 else 0
        if current_btc_price == 0:
            logger.warning("Cannot calculate portfolio risk accurately without current BTC price.")
            # Return last known risk or a default, but preserve hedge history
            return PortfolioRisk(total_positions=len(self.positions), hedge_executions=self.risk_metrics.hedge_executions)


        net_delta_val = sum(pos.current_delta for pos in self.positions.values())
        net_gamma_val = sum(pos.current_gamma for pos in self.positions.values())
        net_theta_val = sum(pos.current_theta for pos in self.positions.values())
        net_vega_val = sum(pos.current_vega for pos in self.positions.values())
        
        total_exposure_val = sum(
            abs(pos.strike * pos.quantity * config.STANDARD_CONTRACT_SIZE_BTC) 
            if pos.option_type == "call" else \
            abs(pos.strike * pos.quantity * config.STANDARD_CONTRACT_SIZE_BTC) # Simplified; could be more nuanced
            for pos in self.positions.values()
        )

        option_pnl_val = self._calculate_option_pnl()
        hedging_pnl_val = self._calculate_hedging_pnl(current_btc_price)

        self.risk_metrics = PortfolioRisk(
            net_delta=net_delta_val, net_gamma=net_gamma_val,
            net_theta=net_theta_val, net_vega=net_vega_val,
            total_exposure_usd=total_exposure_val,
            total_positions=len(self.positions),
            hedging_pnl=hedging_pnl_val, option_pnl=option_pnl_val,
            hedge_executions=self.risk_metrics.hedge_executions # Preserve history
        )
        # logger.debug(f"Portfolio risk calculated: Delta={self.risk_metrics.net_delta:.3f}")
        return self.risk_metrics

    def _calculate_option_pnl(self) -> float:
        """Calculate mark-to-market P&L on option positions."""
        total_pnl = 0.0
        for pos in self.positions.values():
            # pos.mark_to_market_usd is PER CONTRACT
            # pos.entry_price_usd is PER CONTRACT
            # pos.quantity is number of contracts (signed: positive for long, negative for short)
            # P&L for long: quantity * (current_mark - entry_price)
            # P&L for short: quantity * (entry_price - current_mark) -> which is -quantity * (current_mark - entry_price)
            # So, it's always: pos.quantity * (pos.mark_to_market_usd - pos.entry_price_usd)
            # However, if entry_price for short positions means "premium received", then:
            if pos.quantity < 0: # Short position (sold options)
                # Premium received = abs(pos.quantity) * pos.entry_price_usd
                # Current value (cost to buy back) = abs(pos.quantity) * pos.mark_to_market_usd
                # P&L = Premium received - Current value
                pnl = (abs(pos.quantity) * pos.entry_price_usd) - (abs(pos.quantity) * pos.mark_to_market_usd)
            else: # Long position (bought options)
                # Cost = pos.quantity * pos.entry_price_usd
                # Current value = pos.quantity * pos.mark_to_market_usd
                # P&L = Current value - Cost
                pnl = (pos.quantity * pos.mark_to_market_usd) - (pos.quantity * pos.entry_price_usd)
            total_pnl += pnl
        return total_pnl

    def _calculate_hedging_pnl(self, current_btc_price: float) -> float:
        """Calculate P&L on BTC hedge position."""
        if self.hedge_btc_position == 0 or current_btc_price <= 0:
            return 0.0
        
        # P&L = Current Value of Hedge - Cost of Hedge
        # Current Value = self.hedge_btc_position * current_btc_price
        # Cost of Hedge = self.hedge_btc_position * self.hedge_cost_basis (if cost_basis is avg price)
        # This simplified P&L assumes cost_basis is the average price paid/received for the net position.
        # For a more accurate realized/unrealized P&L, you'd track individual hedge trades.
        current_value_of_hedge = self.hedge_btc_position * current_btc_price
        cost_of_current_hedge_position = self.hedge_btc_position * self.hedge_cost_basis
        
        return current_value_of_hedge - cost_of_current_hedge_position

    def should_rehedge(self, current_btc_price: float) -> bool:
        """Determine if portfolio should be re-hedged based on time or delta thresholds."""
        if not self.pricing_engine or self.pricing_engine.current_price <= 0: # Need current price for decisions
            return False # Cannot make informed decision

        current_risk = self.calculate_portfolio_risk() # This updates self.risk_metrics

        current_time = time.time() # Use time.time() for consistency
        time_since_last_hedge_minutes = (current_time - self.last_hedge_timestamp) / 60

        if time_since_last_hedge_minutes >= config.DELTA_HEDGE_FREQUENCY_MINUTES:
            logger.info(f"Rehedging due to time threshold ({time_since_last_hedge_minutes:.1f} min).")
            return True

        # Risk-based rehedging thresholds (e.g., if net delta in BTC terms exceeds a limit)
        # Net delta from risk_metrics is total delta exposure of options.
        # Hedge position delta is self.hedge_btc_position.
        # Overall delta = self.risk_metrics.net_delta (from options) + self.hedge_btc_position (from spot BTC)
        effective_portfolio_delta_btc = self.risk_metrics.net_delta + self.hedge_btc_position
        
        delta_rehedge_threshold_btc = getattr(config, 'DELTA_REHEDGE_THRESHOLD_BTC', 0.1) # Example: 0.1 BTC

        if abs(effective_portfolio_delta_btc) > delta_rehedge_threshold_btc:
            logger.info(f"Rehedging due to delta threshold (Net Delta: {effective_portfolio_delta_btc:.3f} BTC, Threshold: {delta_rehedge_threshold_btc} BTC).")
            return True
            
        return False

    def execute_delta_hedge(self, current_btc_price: float) -> Optional[HedgeExecution]:
        """Execute delta hedging strategy to neutralize portfolio delta."""
        if not self.pricing_engine or self.pricing_engine.current_price <= 0 :
            logger.warning("Cannot execute hedge: pricing engine or current price unavailable.")
            return None
        
        # Ensure risk_metrics are up-to-date based on latest marks
        self.calculate_portfolio_risk() # This updates self.risk_metrics
        
        # Target delta for the options portfolio + hedge position should be zero (delta neutral)
        # Net delta of options portfolio is self.risk_metrics.net_delta
        # We want: options_delta + current_hedge_btc + adjustment_btc = 0
        # So, adjustment_btc = - (options_delta + current_hedge_btc)
        
        options_portfolio_delta_btc = self.risk_metrics.net_delta
        target_overall_delta_btc = 0.0 # For delta-neutral hedge
        
        # How much BTC to trade to get from current effective delta to target delta
        current_effective_delta_btc = options_portfolio_delta_btc + self.hedge_btc_position
        btc_adjustment_needed = target_overall_delta_btc - current_effective_delta_btc

        # Minimum trade size to avoid tiny, costly hedges
        min_hedge_trade_size_btc = getattr(config, 'MIN_HEDGE_TRADE_SIZE_BTC', 0.001) # Example: 0.001 BTC

        if abs(btc_adjustment_needed) < min_hedge_trade_size_btc:
            # logger.debug(f"No hedge needed. Adjustment {btc_adjustment_needed:.4f} BTC is below threshold {min_hedge_trade_size_btc:.4f} BTC.")
            return None

        # Simulate hedge execution with slippage from config
        slippage_factor = 1 + (config.HEDGE_SLIPPAGE_BPS / 10000.0) # Convert BPS to factor
        
        execution_price_usd: float
        if btc_adjustment_needed > 0: # Buying BTC to increase positive delta (or reduce negative delta)
            execution_price_usd = current_btc_price * slippage_factor
        else: # Selling BTC to decrease positive delta (or increase negative delta)
            execution_price_usd = current_btc_price / slippage_factor # More favorable price for seller
                                                                  # Or current_btc_price * (1 - slippage_factor_for_sell)

        hedge_cost_usd_val = btc_adjustment_needed * execution_price_usd # Cost is negative if selling BTC

        # Update hedge position and cost basis
        # Simple weighted average for cost basis
        if (self.hedge_btc_position + btc_adjustment_needed) != 0:
            new_total_cost = (self.hedge_cost_basis * self.hedge_btc_position) + (execution_price_usd * btc_adjustment_needed)
            self.hedge_cost_basis = new_total_cost / (self.hedge_btc_position + btc_adjustment_needed)
        elif btc_adjustment_needed == 0 : # No change if adjustment is zero
            pass
        else: # Position becomes zero
            self.hedge_cost_basis = 0.0
            
        self.hedge_btc_position += btc_adjustment_needed


        execution_record = HedgeExecution(
            timestamp=time.time(), # Use time.time() for consistency
            btc_quantity=btc_adjustment_needed,
            btc_price=execution_price_usd,
            hedge_cost_usd=hedge_cost_usd_val, # This is the USD value of the BTC traded
            reason=f"Delta hedge: OptDelta={options_portfolio_delta_btc:.3f}, CurrHedge={self.hedge_btc_position-btc_adjustment_needed:.3f}"
        )
        
        self.risk_metrics.hedge_executions.append(execution_record)
        self.last_hedge_timestamp = execution_record.timestamp
        logger.info(f"Executed delta hedge: {btc_adjustment_needed:.4f} BTC @ ${execution_price_usd:.2f}. New hedge pos: {self.hedge_btc_position:.4f} BTC.")
        return execution_record

    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary for monitoring."""
        # Ensure risk_metrics are current before returning
        self.calculate_portfolio_risk() # This updates self.risk_metrics
        
        return {
            "portfolio_metrics": {
                "net_delta_btc": self.risk_metrics.net_delta, # Delta from options
                "net_gamma_btc_per_usd": self.risk_metrics.net_gamma, # Check units
                "net_theta_usd_per_day": self.risk_metrics.net_theta,
                "net_vega_usd_per_1_pct_vol": self.risk_metrics.net_vega,
                "total_notional_exposure_usd": self.risk_metrics.total_exposure_usd,
                "total_positions": self.risk_metrics.total_positions
            },
            "pnl": {
                "option_pnl_usd": self.risk_metrics.option_pnl,
                "hedging_pnl_usd": self.risk_metrics.hedging_pnl,
                "total_pnl_usd": self.risk_metrics.option_pnl + self.risk_metrics.hedging_pnl
            },
            "hedge_position": {
                "net_btc_spot_position": self.hedge_btc_position,
                "avg_cost_basis_usd_per_btc": self.hedge_cost_basis,
                "executions_today": len([
                    h for h in self.risk_metrics.hedge_executions 
                    if h.timestamp > time.time() - 86400 # Hedges in the last 24 hours
                ])
            },
            "effective_portfolio_delta_btc": self.risk_metrics.net_delta + self.hedge_btc_position
        }

    # Added for RLHedger compatibility, though base class doesn't use it.
    def save_rl_model(self, filepath: Optional[str] = None):
        logger.info("PortfolioHedger (base class): No RL model to save.")
        pass
