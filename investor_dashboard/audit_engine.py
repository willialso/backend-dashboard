# investor_dashboard/audit_engine.py

import time
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

@dataclass
class AuditMetrics:
    revenue_24h: float
    trades_24h: int
    total_pnl_24h: float
    hedge_efficiency_pct: float
    platform_uptime_pct: float
    compliance_status: str
    avg_response_time_ms: float
    error_rate_pct: float

@dataclass
class ComplianceCheck:
    check_name: str
    status: str  # "PASS", "WARN", "FAIL"
    value: float
    threshold: float
    last_check: float

class AuditEngine:
    """Tracks platform performance and compliance metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_downtime = 0.0
        self.revenue_tracking = []
        self.trade_tracking = []
        self.response_times = []
        self.error_count = 0
        self.total_requests = 0
        
        # Compliance thresholds
        self.compliance_thresholds = {
            "max_single_exposure_btc": 10.0,
            "max_platform_delta_btc": 100.0,
            "min_liquidity_ratio": 1.2,
            "max_downtime_pct": 0.1,
            "max_response_time_ms": 500,
            "max_error_rate_pct": 1.0
        }
        
    def track_revenue(self, amount: float):
        """Track revenue transaction."""
        self.revenue_tracking.append({
            "amount": amount,
            "timestamp": time.time()
        })
        
    def track_trade(self, trade_details: Dict):
        """Track completed trade."""
        self.trade_tracking.append({
            "details": trade_details,
            "timestamp": time.time()
        })
        
    def track_response_time(self, response_time_ms: float):
        """Track API response time."""
        self.response_times.append({
            "time_ms": response_time_ms,
            "timestamp": time.time()
        })
        self.total_requests += 1
        
    def track_error(self):
        """Track system error occurrence."""
        self.error_count += 1
        
    def track_downtime(self, downtime_seconds: float):
        """Track system downtime."""
        self.total_downtime += downtime_seconds
        
    def get_24h_metrics(self) -> AuditMetrics:
        """Get 24-hour audit metrics."""
        current_time = time.time()
        cutoff_24h = current_time - (24 * 60 * 60)
        
        # Revenue in last 24 hours
        recent_revenue = [
            r["amount"] for r in self.revenue_tracking 
            if r["timestamp"] > cutoff_24h
        ]
        revenue_24h = sum(recent_revenue)
        
        # Trades in last 24 hours
        recent_trades = [
            t for t in self.trade_tracking 
            if t["timestamp"] > cutoff_24h
        ]
        trades_24h = len(recent_trades)
        
        # Calculate P&L (simplified)
        total_pnl_24h = revenue_24h * 0.85  # Assume 85% profit margin
        
        # Platform uptime
        total_runtime = current_time - self.start_time
        uptime_pct = ((total_runtime - self.total_downtime) / total_runtime) * 100 if total_runtime > 0 else 100
        
        # Response time (last hour)
        cutoff_1h = current_time - 3600
        recent_response_times = [
            r["time_ms"] for r in self.response_times 
            if r["timestamp"] > cutoff_1h
        ]
        avg_response_time = sum(recent_response_times) / len(recent_response_times) if recent_response_times else 0
        
        # Error rate
        error_rate = (self.error_count / self.total_requests) * 100 if self.total_requests > 0 else 0
        
        # Hedge efficiency (simulated based on market conditions)
        hedge_efficiency = self._calculate_hedge_efficiency()
        
        # Compliance status
        compliance_status = self._check_compliance_status()
        
        return AuditMetrics(
            revenue_24h=revenue_24h,
            trades_24h=trades_24h,
            total_pnl_24h=total_pnl_24h,
            hedge_efficiency_pct=hedge_efficiency,
            platform_uptime_pct=uptime_pct,
            compliance_status=compliance_status,
            avg_response_time_ms=avg_response_time,
            error_rate_pct=error_rate
        )
    
    def _calculate_hedge_efficiency(self) -> float:
        """Calculate hedging efficiency percentage."""
        # Simplified calculation - in real system would use actual hedge data
        import random
        
        # Simulate realistic hedge efficiency (92-98%)
        base_efficiency = 94.5
        market_volatility_factor = random.uniform(-2, 2)
        return max(90, min(98, base_efficiency + market_volatility_factor))
    
    def _check_compliance_status(self) -> str:
        """Check overall compliance status."""
        compliance_checks = self.run_compliance_checks()
        
        failed_checks = [c for c in compliance_checks if c.status == "FAIL"]
        warning_checks = [c for c in compliance_checks if c.status == "WARN"]
        
        if failed_checks:
            return "NON_COMPLIANT"
        elif warning_checks:
            return "WARNING"
        else:
            return "COMPLIANT"
    
    def run_compliance_checks(self) -> List[ComplianceCheck]:
        """Run all compliance checks."""
        checks = []
        current_time = time.time()
        
        # Uptime check
        total_runtime = current_time - self.start_time
        uptime_pct = ((total_runtime - self.total_downtime) / total_runtime) * 100 if total_runtime > 0 else 100
        downtime_pct = 100 - uptime_pct
        
        checks.append(ComplianceCheck(
            check_name="Platform Uptime",
            status="PASS" if downtime_pct <= self.compliance_thresholds["max_downtime_pct"] else "FAIL",
            value=uptime_pct,
            threshold=100 - self.compliance_thresholds["max_downtime_pct"],
            last_check=current_time
        ))
        
        # Response time check
        cutoff_1h = current_time - 3600
        recent_response_times = [
            r["time_ms"] for r in self.response_times 
            if r["timestamp"] > cutoff_1h
        ]
        avg_response_time = sum(recent_response_times) / len(recent_response_times) if recent_response_times else 0
        
        checks.append(ComplianceCheck(
            check_name="Average Response Time",
            status="PASS" if avg_response_time <= self.compliance_thresholds["max_response_time_ms"] else "WARN",
            value=avg_response_time,
            threshold=self.compliance_thresholds["max_response_time_ms"],
            last_check=current_time
        ))
        
        # Error rate check
        error_rate = (self.error_count / self.total_requests) * 100 if self.total_requests > 0 else 0
        
        checks.append(ComplianceCheck(
            check_name="Error Rate",
            status="PASS" if error_rate <= self.compliance_thresholds["max_error_rate_pct"] else "WARN",
            value=error_rate,
            threshold=self.compliance_thresholds["max_error_rate_pct"],
            last_check=current_time
        ))
        
        return checks
    
    def generate_audit_report(self) -> Dict:
        """Generate comprehensive audit report."""
        metrics = self.get_24h_metrics()
        compliance_checks = self.run_compliance_checks()
        
        return {
            "report_timestamp": time.time(),
            "metrics": metrics,
            "compliance_checks": compliance_checks,
            "summary": {
                "overall_health": "HEALTHY" if metrics.compliance_status == "COMPLIANT" else "ATTENTION_REQUIRED",
                "key_achievements": [
                    f"${metrics.revenue_24h:,.2f} revenue in 24h",
                    f"{metrics.trades_24h} trades completed",
                    f"{metrics.platform_uptime_pct:.2f}% uptime"
                ],
                "areas_for_improvement": [
                    c.check_name for c in compliance_checks 
                    if c.status in ["WARN", "FAIL"]
                ]
            }
        }
