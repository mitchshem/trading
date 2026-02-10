"""
Paper trading promotion rules.
Defines objective thresholds that must be met before allowing live trading.

Phase 5 additions:
- Minimum paper trading duration (20 trading days)
- Minimum walk-forward OOS windows with positive return
- Profit factor threshold
- Readiness percentage tracking
- All thresholds must pass on live paper data (not just backtest)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math

from metrics import MetricsSnapshot, compute_metrics
from database import Trade, EquityCurve
from utils import fmt, fmt_pct, fmt_currency


@dataclass
class PromotionThresholds:
    """Objective thresholds for paper trading promotion."""
    max_drawdown_pct: float = 15.0          # Maximum drawdown % (must be <= threshold)
    sharpe_proxy_min: float = 0.5           # Minimum Sharpe proxy
    min_trade_count: int = 20               # Minimum number of closed trades
    min_win_rate: float = 0.40              # Minimum win rate (40%)
    min_expectancy: float = 0.0             # Minimum expectancy per trade (positive)
    min_trading_days: int = 20              # Minimum paper trading days
    min_profit_factor: float = 1.0          # Minimum profit factor (> 1 = profitable)
    min_oos_windows_positive: int = 2       # Minimum walk-forward OOS windows with positive return


@dataclass
class PromotionDecision:
    """Result of promotion rule evaluation."""
    promoted: bool                          # True if ALL thresholds met
    reasons: list[str]                      # List of pass/fail reasons
    metrics: Dict                           # Current metrics snapshot
    checks_passed: int = 0                  # Number of checks that passed
    checks_total: int = 0                   # Total number of checks
    readiness_pct: float = 0.0             # Percentage of checks passed


class PromotionRules:
    """
    Paper trading promotion rules engine.
    Evaluates paper trading performance against objective thresholds.
    All thresholds must pass on LIVE paper data (replay_id=None).
    """

    def __init__(self, thresholds: Optional[PromotionThresholds] = None):
        self.thresholds = thresholds or PromotionThresholds()

    def _count_trading_days(self, trades: list[Trade], equity_curve: list[EquityCurve]) -> int:
        """
        Count unique trading days from trade entries and equity curve.
        A trading day is any calendar date where we have data.
        """
        dates = set()

        for trade in trades:
            if trade.entry_time:
                if isinstance(trade.entry_time, str):
                    try:
                        dt = datetime.fromisoformat(trade.entry_time)
                    except (ValueError, TypeError):
                        continue
                else:
                    dt = trade.entry_time
                dates.add(dt.date() if hasattr(dt, 'date') else dt)

            if trade.exit_time:
                if isinstance(trade.exit_time, str):
                    try:
                        dt = datetime.fromisoformat(trade.exit_time)
                    except (ValueError, TypeError):
                        continue
                else:
                    dt = trade.exit_time
                dates.add(dt.date() if hasattr(dt, 'date') else dt)

        for eq in equity_curve:
            if eq.timestamp:
                if isinstance(eq.timestamp, str):
                    try:
                        dt = datetime.fromisoformat(eq.timestamp)
                    except (ValueError, TypeError):
                        continue
                else:
                    dt = eq.timestamp
                dates.add(dt.date() if hasattr(dt, 'date') else dt)

        return len(dates)

    def evaluate_promotion(
        self,
        trades: list[Trade],
        equity_curve: list[EquityCurve],
        oos_positive_windows: int = 0,
    ) -> PromotionDecision:
        """
        Evaluate if paper trading metrics meet promotion thresholds.

        Args:
            trades: List of Trade objects (live trading only, replay_id=None)
            equity_curve: List of EquityCurve objects (live trading only)
            oos_positive_windows: Number of walk-forward OOS windows with positive return

        Returns:
            PromotionDecision with promotion status and reasons
        """
        reasons = []
        all_passed = True
        checks_passed = 0
        checks_total = 0

        # Compute metrics from live data
        metrics_snapshot = compute_metrics(trades, equity_curve)

        # 1. Trade count threshold
        checks_total += 1
        if metrics_snapshot.trade_count < self.thresholds.min_trade_count:
            all_passed = False
            reasons.append(
                f"Trade count {metrics_snapshot.trade_count} < minimum {self.thresholds.min_trade_count}"
            )
        else:
            checks_passed += 1
            reasons.append(
                f"Trade count {metrics_snapshot.trade_count} >= minimum {self.thresholds.min_trade_count} ✓"
            )

        # 2. Max drawdown threshold
        checks_total += 1
        if metrics_snapshot.max_drawdown_pct > self.thresholds.max_drawdown_pct:
            all_passed = False
            reasons.append(
                f"Max drawdown {fmt_pct(metrics_snapshot.max_drawdown_pct)} > threshold {fmt_pct(self.thresholds.max_drawdown_pct)}"
            )
        else:
            checks_passed += 1
            reasons.append(
                f"Max drawdown {fmt_pct(metrics_snapshot.max_drawdown_pct)} <= threshold {fmt_pct(self.thresholds.max_drawdown_pct)} ✓"
            )

        # 3. Sharpe proxy threshold
        checks_total += 1
        sharpe_valid = (
            metrics_snapshot.sharpe_proxy is not None and
            not math.isnan(metrics_snapshot.sharpe_proxy) and
            not math.isinf(metrics_snapshot.sharpe_proxy)
        )

        if not sharpe_valid or metrics_snapshot.sharpe_proxy < self.thresholds.sharpe_proxy_min:
            all_passed = False
            sharpe_display = fmt(metrics_snapshot.sharpe_proxy) if sharpe_valid else "N/A"
            reasons.append(
                f"Sharpe proxy {sharpe_display} < minimum {self.thresholds.sharpe_proxy_min}"
            )
        else:
            checks_passed += 1
            reasons.append(
                f"Sharpe proxy {fmt(metrics_snapshot.sharpe_proxy)} >= minimum {self.thresholds.sharpe_proxy_min} ✓"
            )

        # 4. Win rate threshold
        checks_total += 1
        if metrics_snapshot.win_rate < self.thresholds.min_win_rate:
            all_passed = False
            reasons.append(
                f"Win rate {fmt_pct(metrics_snapshot.win_rate)} < minimum {fmt_pct(self.thresholds.min_win_rate)}"
            )
        else:
            checks_passed += 1
            reasons.append(
                f"Win rate {fmt_pct(metrics_snapshot.win_rate)} >= minimum {fmt_pct(self.thresholds.min_win_rate)} ✓"
            )

        # 5. Expectancy threshold
        checks_total += 1
        if metrics_snapshot.expectancy_per_trade < self.thresholds.min_expectancy:
            all_passed = False
            reasons.append(
                f"Expectancy {fmt_currency(metrics_snapshot.expectancy_per_trade)} < minimum {fmt_currency(self.thresholds.min_expectancy)}"
            )
        else:
            checks_passed += 1
            reasons.append(
                f"Expectancy {fmt_currency(metrics_snapshot.expectancy_per_trade)} >= minimum {fmt_currency(self.thresholds.min_expectancy)} ✓"
            )

        # 6. Minimum trading days (Phase 5 addition)
        checks_total += 1
        trading_days = self._count_trading_days(trades, equity_curve)
        if trading_days < self.thresholds.min_trading_days:
            all_passed = False
            reasons.append(
                f"Trading days {trading_days} < minimum {self.thresholds.min_trading_days}"
            )
        else:
            checks_passed += 1
            reasons.append(
                f"Trading days {trading_days} >= minimum {self.thresholds.min_trading_days} ✓"
            )

        # 7. Profit factor threshold (Phase 5 addition)
        checks_total += 1
        pf_valid = (
            metrics_snapshot.profit_factor is not None and
            not math.isnan(metrics_snapshot.profit_factor) and
            not math.isinf(metrics_snapshot.profit_factor)
        )
        if not pf_valid or metrics_snapshot.profit_factor < self.thresholds.min_profit_factor:
            all_passed = False
            pf_display = fmt(metrics_snapshot.profit_factor) if pf_valid else "N/A"
            reasons.append(
                f"Profit factor {pf_display} < minimum {self.thresholds.min_profit_factor}"
            )
        else:
            checks_passed += 1
            reasons.append(
                f"Profit factor {fmt(metrics_snapshot.profit_factor)} >= minimum {self.thresholds.min_profit_factor} ✓"
            )

        # 8. Walk-forward OOS positive windows (Phase 5 addition)
        checks_total += 1
        if oos_positive_windows < self.thresholds.min_oos_windows_positive:
            all_passed = False
            reasons.append(
                f"OOS positive windows {oos_positive_windows} < minimum {self.thresholds.min_oos_windows_positive}"
            )
        else:
            checks_passed += 1
            reasons.append(
                f"OOS positive windows {oos_positive_windows} >= minimum {self.thresholds.min_oos_windows_positive} ✓"
            )

        readiness_pct = (checks_passed / checks_total * 100) if checks_total > 0 else 0

        # Log promotion decision
        status = "PROMOTED" if all_passed else "BLOCKED"
        print(f"\n[PROMOTION RULES] {status}: Paper trading promotion evaluation")
        print(f"[PROMOTION RULES]   Readiness: {checks_passed}/{checks_total} ({readiness_pct:.0f}%)")
        print(f"[PROMOTION RULES]   Trade count: {metrics_snapshot.trade_count}")
        print(f"[PROMOTION RULES]   Trading days: {trading_days}")
        print(f"[PROMOTION RULES]   Max drawdown: {fmt_pct(metrics_snapshot.max_drawdown_pct)}")
        print(f"[PROMOTION RULES]   Sharpe proxy: {fmt(metrics_snapshot.sharpe_proxy)}")
        print(f"[PROMOTION RULES]   Win rate: {fmt_pct(metrics_snapshot.win_rate)}")
        print(f"[PROMOTION RULES]   Profit factor: {fmt(metrics_snapshot.profit_factor)}")
        print(f"[PROMOTION RULES]   Expectancy: {fmt_currency(metrics_snapshot.expectancy_per_trade)}")
        print(f"[PROMOTION RULES]   OOS windows: {oos_positive_windows}")
        for reason in reasons:
            print(f"[PROMOTION RULES]     - {reason}")

        return PromotionDecision(
            promoted=all_passed,
            reasons=reasons,
            metrics={
                "trade_count": metrics_snapshot.trade_count,
                "trading_days": trading_days,
                "max_drawdown_pct": metrics_snapshot.max_drawdown_pct,
                "sharpe_proxy": metrics_snapshot.sharpe_proxy,
                "win_rate": metrics_snapshot.win_rate,
                "expectancy": metrics_snapshot.expectancy_per_trade,
                "total_return_pct": metrics_snapshot.total_return_pct,
                "profit_factor": metrics_snapshot.profit_factor,
                "oos_positive_windows": oos_positive_windows,
            },
            checks_passed=checks_passed,
            checks_total=checks_total,
            readiness_pct=readiness_pct,
        )

    def check_trade_allowed(
        self,
        trades: list[Trade],
        equity_curve: list[EquityCurve],
        trade_type: str = "BUY"
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a trade is allowed based on promotion rules.
        EXIT trades are always allowed (risk management).
        """
        # EXIT trades are always allowed (risk management)
        if trade_type == "EXIT":
            return True, None

        # Check promotion status for BUY trades
        decision = self.evaluate_promotion(trades, equity_curve)

        if decision.promoted:
            return True, None
        else:
            failed_checks = [r for r in decision.reasons if "✓" not in r]
            reason = f"Paper trading promotion blocked: {', '.join(failed_checks)}"
            print(f"[PROMOTION RULES] BLOCKED {trade_type} trade: {reason}")
            return False, reason
