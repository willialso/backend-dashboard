# investor_dashboard/revenue_engine.py (FIXED TO USE ACTUAL VOLATILITY ENGINE)

import time
import math
from dataclasses import dataclass
from typing import Dict, List

from backend.advanced_pricing_engine import AdvancedPricingEngine
from backend.volatility_engine import AdvancedVolatilityEngine
from backend.config import *

@dataclass
class RevenueMetrics:
    bsm_fair_value: float
    platform_price: float
    markup_percentage: float
    revenue_per_contract: float
    daily_revenue_estimate: float
    contracts_sold_24h: int
    average_markup: float

class MinimalAlphaSignalGenerator:
    """Minimal alpha signal generator to satisfy AdvancedPricingEngine requirements."""
    
    def __init__(self):
        pass
    
    def generate_signal(self, price: float) -> float:
        """Return neutral signal (no alpha)."""
        return 0.0
    
    def get_alpha(self, price: float) -> float:
        """Alternative method name - return neutral."""
        return 0.0

class RevenueEngine:
    """Revenue engine using REAL volatility engine methods."""
    
    def __init__(self):
        print("🔧 Initializing Revenue Engine with REAL volatility engine...")
        
        # Initialize volatility engine (this works great!)
        self.vol_engine = AdvancedVolatilityEngine()
        
        # Create minimal alpha generator to satisfy pricing engine
        self.alpha_generator = MinimalAlphaSignalGenerator()
        
        # Try to initialize pricing engine, fall back if it fails
        try:
            self.pricing_engine = AdvancedPricingEngine(
                volatility_engine=self.vol_engine,
                alpha_signal_generator=self.alpha_generator
            )
            self.use_advanced_pricing = True
            print("✅ Advanced Pricing Engine initialized")
        except Exception as e:
            print(f"⚠️ Advanced Pricing Engine failed: {e}")
            print("🔄 Using volatility engine + BSM combination")
            self.pricing_engine = None
            self.use_advanced_pricing = False
        
        self.current_btc_price = 0.0
        self.revenue_history = []
        self.base_markup_percentage = 0.035  # 3.5% base markup
        self.last_price_update = 0.0
        self.price_update_count = 0
        self.debug_mode = True
        
        print("✅ Revenue Engine initialized with REAL volatility integration")
        
    def update_price(self, btc_price: float):
        """Update price in BOTH volatility engine AND pricing engine."""
        if btc_price <= 0:
            print(f"⚠️ Revenue Engine: Invalid price received: {btc_price}")
            return
            
        self.price_update_count += 1
        old_price = self.current_btc_price
        self.current_btc_price = btc_price
        self.last_price_update = time.time()
        
        # ← FIX: Update volatility engine with price history
        try:
            self.vol_engine.update_price(btc_price)
            if self.debug_mode and self.price_update_count % 30 == 0:
                print(f"💰 Updated volatility engine with price: ${btc_price:,.2f}")
        except Exception as e:
            print(f"❌ Volatility engine update error: {e}")
        
        # Update pricing engine if available
        if self.use_advanced_pricing and self.pricing_engine:
            try:
                self.pricing_engine.update_market_data(btc_price, volume=25000)
            except Exception as e:
                print(f"❌ Pricing engine update error: {e}")
                
        if self.price_update_count % 30 == 0:
            print(f"💰 Revenue Engine: Price updated ${old_price:,.2f} → ${btc_price:,.2f} (Update #{self.price_update_count})")
    
    def test_volatility_engine(self, test_price: float = None) -> Dict:
        """Test the REAL volatility engine methods."""
        test_price = test_price or self.current_btc_price or 107780.0
        test_results = {}
        
        print(f"🧪 Testing REAL volatility engine with BTC price: ${test_price:,.2f}")
        
        # Update volatility engine with test price
        try:
            self.vol_engine.update_price(test_price)
            test_results["price_update"] = {"status": "success"}
        except Exception as e:
            test_results["price_update"] = {"status": "error", "error": str(e)}
        
        # Test 1: EWMA Volatility
        try:
            ewma_vol = self.vol_engine.calculate_ewma_volatility()
            test_results["ewma_volatility"] = {
                "status": "success",
                "volatility": ewma_vol,
                "annualized_pct": ewma_vol * 100
            }
            print(f"✅ EWMA Volatility: {ewma_vol:.4f} ({ewma_vol*100:.2f}%)")
        except Exception as e:
            test_results["ewma_volatility"] = {"status": "error", "error": str(e)}
        
        # Test 2: Historical Volatility
        try:
            hist_vol = self.vol_engine.calculate_simple_historical_vol()
            test_results["historical_volatility"] = {
                "status": "success", 
                "volatility": hist_vol,
                "annualized_pct": hist_vol * 100
            }
            print(f"✅ Historical Volatility: {hist_vol:.4f} ({hist_vol*100:.2f}%)")
        except Exception as e:
            test_results["historical_volatility"] = {"status": "error", "error": str(e)}
        
        # Test 3: Expiry-Adjusted Volatility (Main Method!)
        try:
            atm_strike = round(test_price, -2)
            expiry_vol = self.vol_engine.get_expiry_adjusted_volatility(
                expiry_minutes=60,
                strike_price=atm_strike,
                underlying_price=test_price
            )
            test_results["expiry_adjusted_volatility"] = {
                "status": "success",
                "volatility": expiry_vol,
                "annualized_pct": expiry_vol * 100,
                "strike": atm_strike,
                "expiry_minutes": 60
            }
            print(f"✅ Expiry-Adjusted Vol (60min, ATM): {expiry_vol:.4f} ({expiry_vol*100:.2f}%)")
        except Exception as e:
            test_results["expiry_adjusted_volatility"] = {"status": "error", "error": str(e)}
        
        # Test 4: Volatility Metrics
        try:
            vol_metrics = self.vol_engine.get_volatility_metrics()
            test_results["volatility_metrics"] = {
                "status": "success",
                "current_vol": vol_metrics.current_vol,
                "regime_vol": vol_metrics.regime_vol,
                "ewma_vol": vol_metrics.ewma_vol,
                "confidence": vol_metrics.confidence,
                "regime": vol_metrics.regime
            }
            print(f"✅ Vol Metrics: Regime={vol_metrics.regime}, Confidence={vol_metrics.confidence:.2f}")
        except Exception as e:
            test_results["volatility_metrics"] = {"status": "error", "error": str(e)}
        
        return test_results
    
    def calculate_advanced_bsm_with_real_volatility(self, S: float, K: float, expiry_minutes: int) -> Dict:
        """BSM calculation using REAL volatility engine data."""
        try:
            from scipy.stats import norm
            
            # ← USE REAL VOLATILITY ENGINE
            volatility = self.vol_engine.get_expiry_adjusted_volatility(
                expiry_minutes=expiry_minutes,
                strike_price=K,
                underlying_price=S
            )
            
            T = expiry_minutes / (365 * 24 * 60)  # Convert to years
            r = 0.05  # 5% risk-free rate
            
            if T <= 0:
                return {
                    "call_price": max(S - K, 0),
                    "volatility_used": volatility,
                    "calculation_type": "intrinsic_value"
                }
            
            # BSM with REAL volatility
            d1 = (math.log(S/K) + (r + 0.5*volatility**2)*T) / (volatility*math.sqrt(T))
            d2 = d1 - volatility*math.sqrt(T)
            
            call_price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
            
            return {
                "call_price": max(call_price, 0),
                "volatility_used": volatility,
                "volatility_pct": volatility * 100,
                "time_to_expiry": T,
                "calculation_type": "advanced_bsm_with_real_volatility"
            }
            
        except Exception as e:
            print(f"❌ Advanced BSM with real volatility error: {e}")
            # Fallback to simple BSM
            return self.calculate_simple_bsm_fallback(S, K, expiry_minutes)
    
    def calculate_simple_bsm_fallback(self, S: float, K: float, expiry_minutes: int) -> Dict:
        """Fallback BSM with fixed volatility."""
        try:
            from scipy.stats import norm
            
            T = expiry_minutes / (365 * 24 * 60)
            r = 0.05
            sigma = 0.80  # Fixed 80% volatility
            
            if T <= 0:
                return {
                    "call_price": max(S - K, 0),
                    "volatility_used": sigma,
                    "calculation_type": "fallback_intrinsic"
                }
            
            d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            
            call_price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
            
            return {
                "call_price": max(call_price, 0),
                "volatility_used": sigma,
                "volatility_pct": sigma * 100,
                "calculation_type": "fallback_bsm"
            }
            
        except Exception as e:
            print(f"❌ Fallback BSM error: {e}")
            return {"call_price": 0, "volatility_used": 0, "calculation_type": "error"}
    
    def calculate_option_revenue(self, strike: float, expiry_minutes: int) -> Dict:
        """Calculate revenue using REAL volatility engine."""
        if self.current_btc_price == 0:
            print("❌ No BTC price available for revenue calculation")
            return {}
            
        try:
            if self.debug_mode:
                print(f"💰 Calculating revenue for ${strike:,.0f} strike using REAL volatility engine")
            
            # ← USE ADVANCED BSM WITH REAL VOLATILITY
            bsm_result = self.calculate_advanced_bsm_with_real_volatility(
                self.current_btc_price, strike, expiry_minutes
            )
            
            bsm_price = bsm_result["call_price"]
            volatility_used = bsm_result["volatility_used"]
            calc_type = bsm_result["calculation_type"]
            
            if bsm_price <= 0:
                if self.debug_mode:
                    print(f"❌ BSM calculation returned invalid price: ${bsm_price}")
                return {}
            
            # Apply markup to get platform price
            platform_price = bsm_price * (1 + self.base_markup_percentage)
            revenue_per_contract = platform_price - bsm_price
            
            if self.debug_mode:
                print(f"💰 Revenue calculation: Platform ${platform_price:.2f}, BSM ${bsm_price:.2f}, Revenue ${revenue_per_contract:.2f}")
                print(f"📊 Volatility used: {volatility_used:.4f} ({volatility_used*100:.2f}%), Method: {calc_type}")
            
            return {
                "bsm_fair_value": bsm_price,
                "platform_price": platform_price,
                "revenue_per_contract": revenue_per_contract,
                "markup_percentage": self.base_markup_percentage * 100,
                "strike": strike,
                "expiry_minutes": expiry_minutes,
                "volatility_used": volatility_used,
                "volatility_pct": volatility_used * 100,
                "calculation_method": calc_type
            }
            
        except Exception as e:
            print(f"❌ Revenue calculation error: {e}")
            return {}
    
    def get_current_metrics(self) -> RevenueMetrics:
        """Get current revenue metrics using REAL volatility."""
        time_since_update = time.time() - self.last_price_update if self.last_price_update > 0 else 999
        
        if self.current_btc_price == 0:
            print(f"⚠️ Revenue Engine: No BTC price available")
            return RevenueMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Calculate for ATM 1-hour call option
        atm_strike = round(self.current_btc_price, -2)
        
        if self.debug_mode:
            print(f"📊 Getting revenue metrics for ATM strike ${atm_strike:,.0f} (BTC: ${self.current_btc_price:,.2f})")
        
        revenue_data = self.calculate_option_revenue(atm_strike, 60)
        
        if not revenue_data:
            print("❌ Revenue Engine: Failed to calculate option revenue")
            return RevenueMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Realistic daily contract volume
        estimated_daily_contracts = 150
        daily_revenue = revenue_data["revenue_per_contract"] * estimated_daily_contracts
        
        metrics = RevenueMetrics(
            bsm_fair_value=revenue_data["bsm_fair_value"],
            platform_price=revenue_data["platform_price"],
            markup_percentage=revenue_data["markup_percentage"],
            revenue_per_contract=revenue_data["revenue_per_contract"],
            daily_revenue_estimate=daily_revenue,
            contracts_sold_24h=estimated_daily_contracts,
            average_markup=self.base_markup_percentage * 100
        )
        
        print(f"💰 Revenue Metrics: Platform ${metrics.platform_price:.2f}, Daily Revenue ${metrics.daily_revenue_estimate:.2f}")
        print(f"📊 Using volatility: {revenue_data.get('volatility_pct', 0):.2f}%")
        
        return metrics
    
    def get_debug_info(self) -> Dict:
        """Get debugging information including volatility engine status."""
        vol_status = "unknown"
        try:
            vol_metrics = self.vol_engine.get_volatility_metrics()
            vol_status = f"regime={vol_metrics.regime}, confidence={vol_metrics.confidence:.2f}"
        except:
            vol_status = "error_getting_metrics"
        
        return {
            "current_btc_price": self.current_btc_price,
            "last_price_update": self.last_price_update,
            "time_since_last_update": time.time() - self.last_price_update if self.last_price_update > 0 else None,
            "price_update_count": self.price_update_count,
            "base_markup_percentage": self.base_markup_percentage,
            "pricing_engine_initialized": self.pricing_engine is not None,
            "vol_engine_initialized": self.vol_engine is not None,
            "vol_engine_status": vol_status,
            "use_advanced_pricing": self.use_advanced_pricing,
            "alpha_generator_type": "minimal_neutral",
            "debug_mode": self.debug_mode,
            "calculation_method": "real_volatility_engine"
        }
    
    def force_price_update(self, btc_price: float) -> Dict:
        """Force update with volatility engine testing."""
        print(f"🔧 Force updating revenue engine with BTC price: ${btc_price:,.2f}")
        self.update_price(btc_price)
        
        # Test volatility engine
        vol_tests = self.test_volatility_engine(btc_price)
        
        # Get metrics
        metrics = self.get_current_metrics()
        
        return {
            **metrics.__dict__,
            "volatility_tests": vol_tests,
            "force_update_timestamp": time.time()
        }
    
    def toggle_debug_mode(self, enabled: bool = None):
        """Toggle debug mode."""
        if enabled is None:
            self.debug_mode = not self.debug_mode
        else:
            self.debug_mode = enabled
        print(f"🔧 Debug mode: {'ENABLED' if self.debug_mode else 'DISABLED'}")
        return self.debug_mode
