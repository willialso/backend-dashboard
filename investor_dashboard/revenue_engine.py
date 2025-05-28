# investor_dashboard/revenue_engine.py

import time
from dataclasses import dataclass
from typing import Dict, List
from core.advanced_pricing_engine import AdvancedPricingEngine
from core.volatility_engine import AdvancedVolatilityEngine
from core.config import *

@dataclass
class RevenueMetrics:
    bsm_fair_value: float
    platform_price: float
    markup_percentage: float
    revenue_per_contract: float
    daily_revenue_estimate: float
    contracts_sold_24h: int
    average_markup: float

class RevenueEngine:
    """Tracks revenue generation from option pricing markup."""
    
    def __init__(self):
        self.vol_engine = AdvancedVolatilityEngine()
        self.pricing_engine = AdvancedPricingEngine(self.vol_engine)
        self.current_btc_price = 0.0
        self.revenue_history = []
        self.base_markup_percentage = 0.035  # 3.5% base markup
        
    def update_price(self, btc_price: float):
        """Update current BTC price for revenue calculations."""
        self.current_btc_price = btc_price
        self.pricing_engine.update_market_data(btc_price, volume=25000)
        
    def calculate_option_revenue(self, strike: float, expiry_minutes: int) -> Dict:
        """Calculate revenue for a specific option."""
        if self.current_btc_price == 0:
            return {}
            
        # Generate option chain for this strike/expiry
        option_chain = self.pricing_engine.generate_option_chain(
            underlying_price=self.current_btc_price,
            expiry_minutes=expiry_minutes
        )
        
        # Find the closest strike
        closest_option = None
        min_diff = float('inf')
        
        for option in option_chain:
            if option.option_type == "call":
                diff = abs(option.strike - strike)
                if diff < min_diff:
                    min_diff = diff
                    closest_option = option
        
        if not closest_option:
            return {}
            
        # Calculate BSM fair value (before markup)
        bsm_fair_value = closest_option.premium_usd / (1 + self.base_markup_percentage)
        
        return {
            "bsm_fair_value": bsm_fair_value,
            "platform_price": closest_option.premium_usd,
            "revenue_per_contract": closest_option.premium_usd - bsm_fair_value,
            "markup_percentage": self.base_markup_percentage * 100,
            "strike": closest_option.strike,
            "expiry_minutes": expiry_minutes
        }
    
    def get_current_metrics(self) -> RevenueMetrics:
        """Get current revenue metrics for dashboard."""
        if self.current_btc_price == 0:
            return RevenueMetrics(0, 0, 0, 0, 0, 0, 0)
            
        # Calculate for ATM 1-hour call option
        atm_strike = round(self.current_btc_price, -2)  # Round to nearest $100
        revenue_data = self.calculate_option_revenue(atm_strike, 60)
        
        if not revenue_data:
            return RevenueMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Simulate daily contract volume (realistic estimates)
        estimated_daily_contracts = 150  # Professional platform volume
        daily_revenue = revenue_data["revenue_per_contract"] * estimated_daily_contracts
        
        return RevenueMetrics(
            bsm_fair_value=revenue_data["bsm_fair_value"],
            platform_price=revenue_data["platform_price"],
            markup_percentage=revenue_data["markup_percentage"],
            revenue_per_contract=revenue_data["revenue_per_contract"],
            daily_revenue_estimate=daily_revenue,
            contracts_sold_24h=estimated_daily_contracts,
            average_markup=self.base_markup_percentage * 100
        )
