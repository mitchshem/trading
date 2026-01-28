"""
Walk-forward evaluation utilities for preventing overfitting.
Provides date filtering and window-based evaluation.
"""

from typing import List, Dict, Optional
from datetime import datetime, timezone, date
import statistics


def filter_candles_by_date_range(
    candles: List[Dict],
    start_date: str,
    end_date: str,
    candle_type: str = "intraday"
) -> List[Dict]:
    """
    Filter candles to include only those within the specified date range.
    
    Args:
        candles: List of candle dicts with 'time' key (Unix timestamp)
        start_date: Start date in "YYYY-MM-DD" format (inclusive)
        end_date: End date in "YYYY-MM-DD" format (inclusive)
        candle_type: "intraday" or "daily" (for logging)
    
    Returns:
        Filtered list of candles within the date range
    
    Raises:
        ValueError: If no candles fall within the date range
    """
    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        # Make end_date inclusive by setting to end of day
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
    
    filtered = []
    
    for candle in candles:
        candle_time = candle.get("time")
        
        # Convert Unix timestamp to datetime
        if isinstance(candle_time, (int, float)):
            candle_dt = datetime.fromtimestamp(candle_time, tz=timezone.utc)
        elif isinstance(candle_time, datetime):
            candle_dt = candle_time
            if candle_dt.tzinfo is None:
                candle_dt = candle_dt.replace(tzinfo=timezone.utc)
        else:
            continue
        
        # Check if candle falls within date range (inclusive)
        if start_dt <= candle_dt <= end_dt:
            filtered.append(candle)
    
    if not filtered:
        raise ValueError(
            f"No {candle_type} candles found in date range {start_date} to {end_date}. "
            f"Total candles available: {len(candles)}"
        )
    
    return filtered


def compute_window_metrics(
    trades: List,
    equity_curve: List,
    starting_equity: float,
    intraday_candles: List[Dict]
) -> Dict:
    """
    Compute window-level metrics for walk-forward evaluation.
    
    Args:
        trades: List of Trade objects
        equity_curve: List of EquityCurve objects
        starting_equity: Starting equity for this window
        intraday_candles: Intraday candles for diagnostics
    
    Returns:
        Dict with window-level metrics
    """
    from metrics import compute_metrics
    from intraday_diagnostics import compute_intraday_aggregate_metrics, compute_frequency_and_session_metrics
    
    # Compute standard metrics
    # Note: compute_metrics handles empty trades/equity_curve gracefully, always returns MetricsSnapshot
    metrics_snapshot = compute_metrics(trades, equity_curve)
    
    # Compute intraday-specific metrics (handles empty trades gracefully)
    intraday_aggregate_metrics = compute_intraday_aggregate_metrics(trades, intraday_candles)
    frequency_session_metrics = compute_frequency_and_session_metrics(trades)
    
    # Calculate return on capital
    # Zero-trade case: If no trades occurred, metrics_snapshot.equity_end = starting_equity (from _empty_metrics_snapshot)
    final_equity = metrics_snapshot.equity_end
    net_pnl = final_equity - starting_equity
    return_on_capital_pct = (net_pnl / starting_equity * 100) if starting_equity > 0 else 0.0
    
    # Compute daily P&L distribution
    daily_pnl_distribution = {}
    for trade in trades:
        if trade.exit_time:
            trade_date = trade.exit_time.date()
            if trade_date not in daily_pnl_distribution:
                daily_pnl_distribution[trade_date] = 0.0
            daily_pnl_distribution[trade_date] += trade.pnl or 0.0
    
    # Daily P&L summary stats
    daily_pnls = list(daily_pnl_distribution.values())
    daily_pnl_mean = statistics.mean(daily_pnls) if daily_pnls else 0.0
    daily_pnl_std = statistics.stdev(daily_pnls) if len(daily_pnls) > 1 else 0.0
    daily_pnl_worst = min(daily_pnls) if daily_pnls else 0.0
    
    return {
        "trades_executed": len(trades),
        "return_on_capital_pct": round(return_on_capital_pct, 2),
        "max_portfolio_drawdown_pct": round(metrics_snapshot.max_drawdown_pct, 2),
        "expectancy": round(intraday_aggregate_metrics["expectancy"], 2),
        "win_rate": round(intraday_aggregate_metrics["win_rate"], 2),
        "average_trades_per_day": round(frequency_session_metrics["frequency_metrics"]["average_trades_per_day"], 2),
        "daily_pnl_distribution": {
            "mean": round(daily_pnl_mean, 2),
            "std": round(daily_pnl_std, 2),
            "worst_day": round(daily_pnl_worst, 2),
            "days_with_trades": len(daily_pnl_distribution)
        }
    }
