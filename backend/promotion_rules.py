"""
Paper trading promotion rules.
Defines objective thresholds that must be met before allowing live trading.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from metrics import MetricsSnapshot, compute_metrics
from database import Trade, EquityCurve
from utils import fmt, fmt_pct


@dataclass
class PromotionThresholds:
    """Objective thresholds for paper trading promotion."""
    max_drawdown_pct: float = 15.0  # Maximum drawdown % (must be <= threshold)
    sharpe_proxy_min: float = 0.5  # Minimum Sharpe proxy (must be >= threshold)
    min_trade_count: int = 20  # Minimum number of closed trades
    min_win_rate: float = 0.40  # Minimum win rate (40%)
    min_expectancy: float = 0.0  # Minimum expectancy per trade (must be positive)


@dataclass
class PromotionDecision:
    """Result of promotion rule evaluation."""
    promoted: bool  # True if thresholds met, False otherwise
    reasons: list[str]  # List of reasons (failures or successes)
    metrics: Dict  # Current metrics snapshot


class PromotionRules:
    """
    Paper trading promotion rules engine.
    Evaluates paper trading performance against objective thresholds.
    """
    
    def __init__(self, thresholds: Optional[PromotionThresholds] = None):
        """
        Initialize promotion rules.
        
        Args:
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or PromotionThresholds()
    
    def evaluate_promotion(
        self,
        trades: list[Trade],
        equity_curve: list[EquityCurve]
    ) -> PromotionDecision:
        """
        Evaluate if paper trading metrics meet promotion thresholds.
        
        Args:
            trades: List of Trade objects (live trading only, replay_id=None)
            equity_curve: List of EquityCurve objects (live trading only, replay_id=None)
        
        Returns:
            PromotionDecision with promotion status and reasons
        """
        reasons = []
        
        # Compute metrics
        metrics_snapshot = compute_metrics(trades, equity_curve)
        
        # Check each threshold
        all_passed = True
        
        # 1. Trade count threshold
        if metrics_snapshot.trade_count < self.thresholds.min_trade_count:
            all_passed = False
            reasons.append(
                f"Trade count {metrics_snapshot.trade_count} < minimum {self.thresholds.min_trade_count}"
            )
        else:
            reasons.append(
                f"Trade count {metrics_snapshot.trade_count} >= minimum {self.thresholds.min_trade_count} ✓"
            )
        
        # 2. Max drawdown threshold
        if metrics_snapshot.max_drawdown_pct > self.thresholds.max_drawdown_pct:
            all_passed = False
            reasons.append(
                f"Max drawdown {fmt_pct(metrics_snapshot.max_drawdown_pct)} > threshold {fmt_pct(self.thresholds.max_drawdown_pct)}"
            )
        else:
            reasons.append(
                f"Max drawdown {fmt_pct(metrics_snapshot.max_drawdown_pct)} <= threshold {fmt_pct(self.thresholds.max_drawdown_pct)} ✓"
            )
        
        # 3. Sharpe proxy threshold
        import math
        sharpe_valid = (
            metrics_snapshot.sharpe_proxy is not None and
            not math.isnan(metrics_snapshot.sharpe_proxy) and
            not math.isinf(metrics_snapshot.sharpe_proxy)
        )
        
        if not sharpe_valid or (sharpe_valid and metrics_snapshot.sharpe_proxy < self.thresholds.sharpe_proxy_min):
            all_passed = False
            sharpe_display = fmt(metrics_snapshot.sharpe_proxy) if sharpe_valid else "N/A"
            reasons.append(
                f"Sharpe proxy {sharpe_display} < minimum {self.thresholds.sharpe_proxy_min}"
            )
        else:
            reasons.append(
                f"Sharpe proxy {fmt(metrics_snapshot.sharpe_proxy)} >= minimum {self.thresholds.sharpe_proxy_min} ✓"
            )
        
        # 4. Win rate threshold
        if metrics_snapshot.win_rate < self.thresholds.min_win_rate:
            all_passed = False
            reasons.append(
                f"Win rate {fmt_pct(metrics_snapshot.win_rate)} < minimum {fmt_pct(self.thresholds.min_win_rate)}"
            )
        else:
            reasons.append(
                f"Win rate {fmt_pct(metrics_snapshot.win_rate)} >= minimum {fmt_pct(self.thresholds.min_win_rate)} ✓"
            )
        
        # 5. Expectancy threshold
        if metrics_snapshot.expectancy_per_trade < self.thresholds.min_expectancy:
            all_passed = False
            reasons.append(
                f"Expectancy {fmt_currency(metrics_snapshot.expectancy_per_trade)} < minimum {fmt_currency(self.thresholds.min_expectancy)}"
            )
        else:
            reasons.append(
                f"Expectancy {fmt_currency(metrics_snapshot.expectancy_per_trade)} >= minimum {fmt_currency(self.thresholds.min_expectancy)} ✓"
            )
        
        # Log promotion decision
        status = "PROMOTED" if all_passed else "BLOCKED"
        print(f"\n[PROMOTION RULES] {status}: Paper trading promotion evaluation")
        print(f"[PROMOTION RULES]   Trade count: {metrics_snapshot.trade_count}")
        print(f"[PROMOTION RULES]   Max drawdown: {fmt_pct(metrics_snapshot.max_drawdown_pct)}")
        print(f"[PROMOTION RULES]   Sharpe proxy: {fmt(metrics_snapshot.sharpe_proxy)}")
        print(f"[PROMOTION RULES]   Win rate: {fmt_pct(metrics_snapshot.win_rate)}")
        print(f"[PROMOTION RULES]   Expectancy: {fmt_currency(metrics_snapshot.expectancy_per_trade)}")
        print(f"[PROMOTION RULES]   Reasons:")
        for reason in reasons:
            print(f"[PROMOTION RULES]     - {reason}")
        
        return PromotionDecision(
            promoted=all_passed,
            reasons=reasons,
            metrics={
                "trade_count": metrics_snapshot.trade_count,
                "max_drawdown_pct": metrics_snapshot.max_drawdown_pct,
                "sharpe_proxy": metrics_snapshot.sharpe_proxy,
                "win_rate": metrics_snapshot.win_rate,
                "expectancy": metrics_snapshot.expectancy_per_trade,
                "total_return_pct": metrics_snapshot.total_return_pct,
                "profit_factor": metrics_snapshot.profit_factor
            }
        )
    
    def check_trade_allowed(
        self,
        trades: list[Trade],
        equity_curve: list[EquityCurve],
        trade_type: str = "BUY"
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a trade is allowed based on promotion rules.
        
        Args:
            trades: List of Trade objects (live trading only)
            equity_curve: List of EquityCurve objects (live trading only)
            trade_type: Type of trade ("BUY" or "EXIT")
        
        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        # EXIT trades are always allowed (risk management)
        if trade_type == "EXIT":
            return True, None
        
        # Check promotion status for BUY trades
        decision = self.evaluate_promotion(trades, equity_curve)
        
        if decision.promoted:
            return True, None
        else:
            # Block trade and return reason
            failed_checks = [r for r in decision.reasons if "✓" not in r]
            reason = f"Paper trading promotion blocked: {', '.join(failed_checks)}"
            print(f"[PROMOTION RULES] BLOCKED {trade_type} trade: {reason}")
            return False, reason
