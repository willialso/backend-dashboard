# investor_dashboard/audit_engine.py

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
import random
import logging

from backend import config # Import your config

logger = logging.getLogger(__name__)

@dataclass
class AuditMetrics:
    gross_option_premiums_24h_usd: float
    net_hedging_pnl_24h_usd: float
    operational_costs_24h_usd: float
    option_trades_executed_24h: int # Number of option trades executed
    net_platform_pnl_24h_usd: float   # Net Platform P&L
    avg_hedge_efficiency_pct: Optional[float] # Optional if hard to calculate
    platform_uptime_pct: float
    compliance_status: str
    avg_api_response_time_ms: float
    api_error_rate_pct: float

@dataclass
class ComplianceCheck:
    check_name: str
    status: str
    value: Optional[Any] # Can be float, str, etc.
    threshold: Optional[Any]
    last_check_timestamp: float
    details: Optional[str] = None

class AuditEngine:
    def __init__(self):
        self.start_time = time.time()
        self.total_downtime_seconds = 0.0
        
        self.option_premium_records: List[Dict[str, Any]] = []
        self.hedging_pnl_records: List[Dict[str, Any]] = []
        self.operational_cost_records: List[Dict[str, Any]] = []
        self.option_trade_execution_records: List[Dict[str, Any]] = []
        self.api_response_time_records: List[Dict[str, Any]] = []
        self.api_error_count = 0
        self.total_api_requests = 0
        logger.info("AuditEngine initialized.")

    def track_option_premium(self, amount: float, trade_id: Optional[str] = None, is_credit: bool = True):
        signed_amount = amount if is_credit else -amount
        self.option_premium_records.append({
            "amount_usd": signed_amount, "trade_id": trade_id, "timestamp": time.time()
        })
        logger.debug(f"AE: Tracked option premium: ${signed_amount:.2f} for trade {trade_id}")

    def track_hedging_activity(self, pnl_usd: float, cost_usd: float, hedge_id: Optional[str] = None, risk_reduction_metric: Optional[float]=None):
        """Tracks P&L and cost of a hedging activity."""
        self.hedging_pnl_records.append({
            "pnl_usd": pnl_usd, # Realized or MTM PNL of the hedge itself
            "cost_usd": cost_usd, # Transaction costs for the hedge
            "net_effect_usd": pnl_usd - cost_usd,
            "hedge_id": hedge_id,
            "risk_reduction_metric": risk_reduction_metric, # e.g., Delta reduced
            "timestamp": time.time()
        })
        logger.debug(f"AE: Tracked hedge activity: PNL=${pnl_usd:.2f}, Cost=${cost_usd:.2f} for hedge {hedge_id}")
    
    def track_operational_cost(self, amount_usd: float, description: Optional[str] = None):
        self.operational_cost_records.append({
            "amount_usd": amount_usd, "description": description, "timestamp": time.time()
        })
        logger.debug(f"AE: Tracked OpCost: ${amount_usd:.2f} for {description}")

    def track_option_trade_executed(self, trade_id: Optional[str] = None):
        self.option_trade_execution_records.append({
            "trade_id": trade_id, "timestamp": time.time()
        })

    def track_api_response(self, response_time_ms: float, endpoint: str, success: bool):
        self.api_response_time_records.append({
            "time_ms": response_time_ms, "endpoint": endpoint, "timestamp": time.time()
        })
        self.total_api_requests += 1
        if not success:
            self.api_error_count += 1

    def track_downtime_event(self, duration_seconds: float):
        self.total_downtime_seconds += duration_seconds

    def get_24h_metrics(self) -> AuditMetrics:
        current_time = time.time()
        cutoff_24h = current_time - (24 * 60 * 60)

        premiums_24h = sum(r["amount_usd"] for r in self.option_premium_records if r["timestamp"] >= cutoff_24h)
        
        hedging_net_effect_24h = sum(r["net_effect_usd"] for r in self.hedging_pnl_records if r["timestamp"] >= cutoff_24h)
        
        op_costs_24h = sum(r["amount_usd"] for r in self.operational_cost_records if r["timestamp"] >= cutoff_24h)
        
        option_trades_24h = len([r for r in self.option_trade_execution_records if r["timestamp"] >= cutoff_24h])

        net_platform_pnl_24h = premiums_24h + hedging_net_effect_24h - op_costs_24h
        
        total_runtime_seconds = current_time - self.start_time
        uptime_pct = ((total_runtime_seconds - self.total_downtime_seconds) / total_runtime_seconds) * 100 if total_runtime_seconds > 0 else 100.0
        
        responses_1h = [r["time_ms"] for r in self.api_response_time_records if r["timestamp"] >= current_time - 3600]
        avg_api_response_ms = sum(responses_1h) / len(responses_1h) if responses_1h else 0.0
        
        api_error_rate_pct = (self.api_error_count / self.total_api_requests) * 100 if self.total_api_requests > 0 else 0.0

        # Hedge efficiency calculation is complex and requires linking hedge cost/P&L to actual risk reduction.
        # For now, a placeholder or simple ratio.
        total_hedge_costs_24h = sum(r["cost_usd"] for r in self.hedging_pnl_records if r["timestamp"] >= cutoff_24h)
        # A true efficiency needs |delta_reduced_value| / |hedge_cost_or_pnl_impact|
        avg_hedge_efficiency = None # Placeholder - needs more data
        if total_hedge_costs_24h > 0 and hedging_net_effect_24h != 0: # Avoid division by zero
             # This is a very rough proxy: PNL relative to costs.
             # If PNL is positive (hedges made money), efficiency is high.
             # If PNL is negative but small relative to costs, still okay if risk reduced.
             # avg_hedge_efficiency = (hedging_net_effect_24h / total_hedge_costs_24h) * 100 # Not quite right
             pass # Needs better metric

        compliance_status_str = self._check_compliance_status_with_placeholders()

        return AuditMetrics(
            gross_option_premiums_24h_usd=round(premiums_24h, 2),
            net_hedging_pnl_24h_usd=round(hedging_net_effect_24h, 2),
            operational_costs_24h_usd=round(op_costs_24h, 2),
            option_trades_executed_24h=option_trades_24h,
            net_platform_pnl_24h_usd=round(net_platform_pnl_24h, 2),
            avg_hedge_efficiency_pct=avg_hedge_efficiency, # Could be None
            platform_uptime_pct=round(max(0, min(100, uptime_pct)), 2),
            compliance_status=compliance_status_str,
            avg_api_response_time_ms=round(avg_api_response_ms, 2),
            api_error_rate_pct=round(api_error_rate_pct, 2)
        )

    def _check_compliance_status_with_placeholders(self) -> str: # Renamed to be clear
        # This should ideally query PositionManager for live delta, LiquidityManager for ratio etc.
        # For now, uses config thresholds and simulates values.
        checks = self.run_compliance_checks_with_placeholders() # Uses config thresholds
        if any(c.status == "FAIL" for c in checks): return "NON_COMPLIANT"
        if any(c.status == "WARN" for c in checks): return "WARNING"
        return "COMPLIANT"

    def run_compliance_checks_with_placeholders(self) -> List[ComplianceCheck]:
        checks = []
        ct = time.time()
        # Actual values should be fetched from relevant managers (PositionManager, LiquidityManager)
        # Simulating values for now:
        sim_delta = random.uniform(-config.MAX_PLATFORM_NET_DELTA_BTC * 1.2, config.MAX_PLATFORM_NET_DELTA_BTC * 1.2)
        sim_liq_ratio = random.uniform(config.MIN_LIQUIDITY_RATIO * 0.8, config.MIN_LIQUIDITY_RATIO * 1.5)
        
        checks.append(ComplianceCheck("Max Platform Delta (BTC)", 
                                      "PASS" if abs(sim_delta) <= config.MAX_PLATFORM_NET_DELTA_BTC else "FAIL",
                                      round(sim_delta,2), config.MAX_PLATFORM_NET_DELTA_BTC, ct))
        checks.append(ComplianceCheck("Min Liquidity Ratio",
                                      "PASS" if sim_liq_ratio >= config.MIN_LIQUIDITY_RATIO else "FAIL",
                                      round(sim_liq_ratio,2), config.MIN_LIQUIDITY_RATIO, ct))
        # ... (add internal checks like uptime, error rate from get_24h_metrics calculations)
        metrics = self.get_24h_metrics() # Get current calculated metrics
        checks.append(ComplianceCheck("Platform Uptime (%)",
                                      "PASS" if metrics.platform_uptime_pct >= (100 - config.MAX_DOWNTIME_PCT) else "FAIL",
                                      metrics.platform_uptime_pct, (100-config.MAX_DOWNTIME_PCT), ct))
        checks.append(ComplianceCheck("API Error Rate (%)",
                                      "PASS" if metrics.api_error_rate_pct <= config.MAX_ERROR_RATE_PCT else "WARN",
                                      metrics.api_error_rate_pct, config.MAX_ERROR_RATE_PCT, ct))
        return checks

    def generate_audit_report(self) -> Dict: # From your Attachment [1]
        metrics = self.get_24h_metrics()
        compliance_checks_list = self.run_compliance_checks_with_placeholders()

        return {
            "report_timestamp": time.time(),
            "metrics_24h": metrics.__dict__,
            "compliance_summary": {
                "overall_status": metrics.compliance_status,
                "checks": [check.__dict__ for check in compliance_checks_list]
            }
        }
