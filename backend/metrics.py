"""
Metrics and evaluation engine for trading performance analysis.
Strategy-agnostic metrics computed from persisted trade and equity data.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class MetricsSnapshot:
    """
    Immutable snapshot of trading performance metrics.
    Computed from persisted data at a specific point in time.
    """
    # Metadata
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    trade_count: int
    equity_start: float
    equity_end: float
    
    # Core Metrics
    total_return_pct: float
    net_pnl: float
    win_rate: float
    loss_rate: float
    profit_factor: float
    expectancy_per_trade: float
    average_win: float
    average_loss: float
    
    # Risk Metrics
    max_drawdown_absolute: float
    max_drawdown_pct: float
    max_consecutive_losses: int
    max_consecutive_wins: int
    exposure_pct: float
    
    # Time-Based Metrics
    trades_per_day: float
    avg_trade_duration_hours: float
    profitable_days_pct: float
    
    # Risk-Adjusted Metric
    sharpe_proxy: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
                "trade_count": self.trade_count,
                "equity_start": round(self.equity_start, 2),
                "equity_end": round(self.equity_end, 2),
            },
            "core_metrics": {
                "total_return_pct": round(self.total_return_pct, 2),
                "net_pnl": round(self.net_pnl, 2),
                "win_rate": round(self.win_rate, 2),
                "loss_rate": round(self.loss_rate, 2),
                "profit_factor": round(self.profit_factor, 2) if not math.isinf(self.profit_factor) else None,
                "expectancy_per_trade": round(self.expectancy_per_trade, 2),
                "average_win": round(self.average_win, 2),
                "average_loss": round(self.average_loss, 2),
            },
            "risk_metrics": {
                "max_drawdown_absolute": round(self.max_drawdown_absolute, 2),
                "max_drawdown_pct": round(self.max_drawdown_pct, 2),
                "max_consecutive_losses": self.max_consecutive_losses,
                "max_consecutive_wins": self.max_consecutive_wins,
                "exposure_pct": round(self.exposure_pct, 2),
            },
            "time_metrics": {
                "trades_per_day": round(self.trades_per_day, 2),
                "avg_trade_duration_hours": round(self.avg_trade_duration_hours, 2),
                "profitable_days_pct": round(self.profitable_days_pct, 2),
            },
            "risk_adjusted": {
                "sharpe_proxy": round(self.sharpe_proxy, 2) if not math.isnan(self.sharpe_proxy) and not math.isinf(self.sharpe_proxy) else None,
            }
        }


def compute_metrics(trades: List, equity_curve: List) -> MetricsSnapshot:
    """
    Compute all trading performance metrics from persisted data.
    
    Args:
        trades: List of Trade objects (from database)
        equity_curve: List of EquityCurve objects (from database)
    
    Returns:
        MetricsSnapshot with all computed metrics
    """
    # Filter to closed trades only (pnl is not None)
    closed_trades = [t for t in trades if t.pnl is not None]
    
    # Handle zero trades case
    if len(closed_trades) == 0:
        return _empty_metrics_snapshot(equity_curve)
    
    # Extract P&L values
    pnl_values = [t.pnl for t in closed_trades]
    wins = [pnl for pnl in pnl_values if pnl > 0]
    losses = [pnl for pnl in pnl_values if pnl < 0]
    
    # Core Metrics
    net_pnl = sum(pnl_values)
    total_return_pct = _compute_total_return(equity_curve)
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0.0
    loss_rate = len(losses) / len(closed_trades) * 100 if closed_trades else 0.0
    
    # Profit factor (gross wins / gross losses)
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (float('inf') if gross_wins > 0 else 0.0)
    
    # Expectancy per trade
    expectancy_per_trade = net_pnl / len(closed_trades) if closed_trades else 0.0
    
    # Average win/loss
    average_win = statistics.mean(wins) if wins else 0.0
    average_loss = statistics.mean(losses) if losses else 0.0
    
    # Risk Metrics
    max_drawdown_absolute, max_drawdown_pct = _compute_max_drawdown(equity_curve)
    max_consecutive_losses = _compute_max_consecutive_losses(closed_trades)
    max_consecutive_wins = _compute_max_consecutive_wins(closed_trades)
    exposure_pct = _compute_exposure_pct(closed_trades, equity_curve)
    
    # Time-Based Metrics
    trades_per_day = _compute_trades_per_day(closed_trades)
    avg_trade_duration_hours = _compute_avg_trade_duration(closed_trades)
    profitable_days_pct = _compute_profitable_days_pct(equity_curve)
    
    # Risk-Adjusted Metric
    sharpe_proxy = _compute_sharpe_proxy(equity_curve)
    
    # Metadata
    start_date = min((t.entry_time for t in closed_trades), default=None)
    end_date = max((t.exit_time for t in closed_trades if t.exit_time), default=None)
    equity_start = equity_curve[0].equity if equity_curve else 100000.0
    equity_end = equity_curve[-1].equity if equity_curve else 100000.0
    
    return MetricsSnapshot(
        start_date=start_date,
        end_date=end_date,
        trade_count=len(closed_trades),
        equity_start=equity_start,
        equity_end=equity_end,
        total_return_pct=total_return_pct,
        net_pnl=net_pnl,
        win_rate=win_rate,
        loss_rate=loss_rate,
        profit_factor=profit_factor,
        expectancy_per_trade=expectancy_per_trade,
        average_win=average_win,
        average_loss=average_loss,
        max_drawdown_absolute=max_drawdown_absolute,
        max_drawdown_pct=max_drawdown_pct,
        max_consecutive_losses=max_consecutive_losses,
        max_consecutive_wins=max_consecutive_wins,
        exposure_pct=exposure_pct,
        trades_per_day=trades_per_day,
        avg_trade_duration_hours=avg_trade_duration_hours,
        profitable_days_pct=profitable_days_pct,
        sharpe_proxy=sharpe_proxy
    )


def _empty_metrics_snapshot(equity_curve: List) -> MetricsSnapshot:
    """Return empty metrics snapshot when no trades exist."""
    equity_start = equity_curve[0].equity if equity_curve else 100000.0
    equity_end = equity_curve[-1].equity if equity_curve else 100000.0
    
    return MetricsSnapshot(
        start_date=None,
        end_date=None,
        trade_count=0,
        equity_start=equity_start,
        equity_end=equity_end,
        total_return_pct=0.0,
        net_pnl=0.0,
        win_rate=0.0,
        loss_rate=0.0,
        profit_factor=0.0,
        expectancy_per_trade=0.0,
        average_win=0.0,
        average_loss=0.0,
        max_drawdown_absolute=0.0,
        max_drawdown_pct=0.0,
        max_consecutive_losses=0,
        max_consecutive_wins=0,
        exposure_pct=0.0,
        trades_per_day=0.0,
        avg_trade_duration_hours=0.0,
        profitable_days_pct=0.0,
        sharpe_proxy=0.0
    )


def _compute_total_return(equity_curve: List) -> float:
    """Compute total return percentage from equity curve."""
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    
    equity_start = equity_curve[0].equity
    equity_end = equity_curve[-1].equity
    
    if equity_start == 0:
        return 0.0
    
    return ((equity_end - equity_start) / equity_start) * 100


def _compute_max_drawdown(equity_curve: List) -> Tuple[float, float]:
    """
    Compute maximum drawdown (absolute and percentage).
    Drawdown = peak equity - current equity.
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0
    
    peak_equity = equity_curve[0].equity
    max_drawdown_absolute = 0.0
    max_drawdown_pct = 0.0
    
    for point in equity_curve:
        if point.equity > peak_equity:
            peak_equity = point.equity
        
        drawdown_absolute = peak_equity - point.equity
        drawdown_pct = (drawdown_absolute / peak_equity * 100) if peak_equity > 0 else 0.0
        
        if drawdown_absolute > max_drawdown_absolute:
            max_drawdown_absolute = drawdown_absolute
            max_drawdown_pct = drawdown_pct
    
    return max_drawdown_absolute, max_drawdown_pct


def _compute_max_consecutive_losses(trades: List) -> int:
    """Compute maximum consecutive losing trades."""
    if not trades:
        return 0
    
    max_consecutive = 0
    current_consecutive = 0
    
    for trade in sorted(trades, key=lambda t: t.exit_time or t.entry_time):
        if trade.pnl is None:
            continue
        
        if trade.pnl < 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive


def _compute_max_consecutive_wins(trades: List) -> int:
    """Compute maximum consecutive winning trades."""
    if not trades:
        return 0
    
    max_consecutive = 0
    current_consecutive = 0
    
    for trade in sorted(trades, key=lambda t: t.exit_time or t.entry_time):
        if trade.pnl is None:
            continue
        
        if trade.pnl > 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive


def _compute_exposure_pct(trades: List, equity_curve: List) -> float:
    """
    Compute exposure percentage (% time in market).
    
    AUDIT FIX: Compute true account exposure as union of position intervals, capped at 100%.
    Previous implementation was gross exposure (sum of all durations) which could exceed 100%.
    
    True exposure = (union of all position time intervals) / (total time period) * 100
    
    This represents the percentage of time the account had at least one open position,
    accounting for overlapping positions (e.g., if 2 positions overlap, it counts as 1).
    """
    if not trades or not equity_curve:
        return 0.0
    
    # Get time range
    first_trade = min((t.entry_time for t in trades), default=None)
    last_exit = max((t.exit_time for t in trades if t.exit_time), default=None)
    
    if not first_trade or not last_exit:
        return 0.0
    
    total_time = (last_exit - first_trade).total_seconds()
    if total_time == 0:
        return 0.0
    
    # Build list of position intervals (entry_time, exit_time)
    intervals = []
    for trade in trades:
        if trade.exit_time and trade.entry_time:
            intervals.append((trade.entry_time, trade.exit_time))
    
    if not intervals:
        return 0.0
    
    # Sort intervals by entry time
    intervals.sort(key=lambda x: x[0])
    
    # Merge overlapping intervals to compute union
    merged = []
    for start, end in intervals:
        if not merged or merged[-1][1] < start:
            # No overlap, add new interval
            merged.append([start, end])
        else:
            # Overlap, extend current interval
            merged[-1][1] = max(merged[-1][1], end)
    
    # Sum the duration of merged intervals (union)
    union_time = sum((end - start).total_seconds() for start, end in merged)
    
    # Cap at 100% (shouldn't happen, but safety check)
    exposure_pct = (union_time / total_time) * 100 if total_time > 0 else 0.0
    return min(exposure_pct, 100.0)


def _compute_trades_per_day(trades: List) -> float:
    """Compute average trades per day."""
    if not trades:
        return 0.0
    
    # Get date range
    dates = set()
    for trade in trades:
        if trade.entry_time:
            dates.add(trade.entry_time.date())
        if trade.exit_time:
            dates.add(trade.exit_time.date())
    
    if not dates:
        return 0.0
    
    days = (max(dates) - min(dates)).days + 1
    if days == 0:
        return 0.0
    
    return len(trades) / days


def _compute_avg_trade_duration(trades: List) -> float:
    """Compute average trade duration in hours."""
    if not trades:
        return 0.0
    
    durations = []
    for trade in trades:
        if trade.exit_time and trade.entry_time:
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
            durations.append(duration)
    
    if not durations:
        return 0.0
    
    return statistics.mean(durations)


def _compute_profitable_days_pct(equity_curve: List) -> float:
    """
    Compute percentage of profitable days.
    
    AUDIT FIX: Use end-of-day equity (last point of each day), not intraday max.
    This ensures conservative metrics that match actual daily close values.
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    
    profitable_days = 0
    total_days = 0
    
    # Group equity points by date, keeping only the last point of each day (end-of-day)
    daily_equity_eod = {}
    for point in equity_curve:
        date = point.timestamp.date()
        # Keep only the last (most recent) equity point for each day
        if date not in daily_equity_eod or point.timestamp > daily_equity_eod[date][0]:
            daily_equity_eod[date] = (point.timestamp, point.equity)
    
    # Compute daily returns using end-of-day equity
    dates = sorted(daily_equity_eod.keys())
    for i in range(1, len(dates)):
        prev_equity = daily_equity_eod[dates[i-1]][1]  # End-of-day equity
        curr_equity = daily_equity_eod[dates[i]][1]  # End-of-day equity
        
        if prev_equity > 0:
            daily_return = (curr_equity - prev_equity) / prev_equity
            if daily_return > 0:
                profitable_days += 1
            total_days += 1
    
    return (profitable_days / total_days * 100) if total_days > 0 else 0.0


def _compute_sharpe_proxy(equity_curve: List) -> float:
    """
    Compute Sharpe ratio proxy.
    Sharpe proxy = mean(daily returns) / std(daily returns)
    Assumes risk-free rate = 0
    
    AUDIT FIX: Use end-of-day equity (last point of each day), not intraday max.
    This ensures conservative metrics that match actual daily close values.
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    
    # Group equity points by date, keeping only the last point of each day (end-of-day)
    daily_equity_eod = {}
    for point in equity_curve:
        date = point.timestamp.date()
        # Keep only the last (most recent) equity point for each day
        if date not in daily_equity_eod or point.timestamp > daily_equity_eod[date][0]:
            daily_equity_eod[date] = (point.timestamp, point.equity)
    
    # Compute daily returns using end-of-day equity
    daily_returns = []
    dates = sorted(daily_equity_eod.keys())
    for i in range(1, len(dates)):
        prev_equity = daily_equity_eod[dates[i-1]][1]  # End-of-day equity
        curr_equity = daily_equity_eod[dates[i]][1]  # End-of-day equity
        
        if prev_equity > 0:
            daily_return = (curr_equity - prev_equity) / prev_equity
            daily_returns.append(daily_return)
    
    if not daily_returns or len(daily_returns) < 2:
        return 0.0
    
    mean_return = statistics.mean(daily_returns)
    std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.0
    
    if std_return == 0:
        return 0.0
    
    return mean_return / std_return
