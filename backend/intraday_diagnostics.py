"""
Intraday trade diagnostics module.
Computes per-trade and aggregate metrics for intraday trades.
"""

from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import statistics


def compute_intraday_trade_diagnostics(
    trade,
    intraday_candles: List[Dict]
) -> Dict:
    """
    Compute diagnostics for a single intraday trade.
    
    Args:
        trade: Trade object from database
        intraday_candles: List of 5-minute candle dicts with keys: time, open, high, low, close, volume
                         Candles should be ordered by time ascending
    
    Returns:
        Dict with diagnostics:
        - holding_time_minutes: Number of minutes between entry and exit
        - holding_period_candles: Number of 5-minute candles between entry and exit
        - mfe: Max Favorable Excursion (best price reached - entry_price)
        - mae: Max Adverse Excursion (entry_price - worst price reached)
        - mfe_pct: MFE as percentage of entry price
        - mae_pct: MAE as percentage of entry price
        - entry_to_exit_return_pct: (exit_price - entry_price) / entry_price * 100
        - mfe_given_up_pct: (MFE - realized_pnl) / MFE * 100 (if MFE > 0)
    """
    if not trade.exit_time or not trade.entry_time:
        # Open position - can't compute diagnostics
        return {
            "holding_time_minutes": None,
            "holding_period_candles": None,
            "mfe": None,
            "mae": None,
            "mfe_pct": None,
            "mae_pct": None,
            "entry_to_exit_return_pct": None,
            "mfe_given_up_pct": None
        }
    
    # Convert entry/exit times to Unix timestamps for comparison
    if isinstance(trade.entry_time, datetime):
        entry_timestamp = int(trade.entry_time.timestamp())
    else:
        entry_timestamp = trade.entry_time
    
    if isinstance(trade.exit_time, datetime):
        exit_timestamp = int(trade.exit_time.timestamp())
    else:
        exit_timestamp = trade.exit_time
    
    # Find candles during the trade period (entry_time <= candle_time <= exit_time)
    trade_candles = []
    for candle in intraday_candles:
        candle_time = candle.get("time")
        if isinstance(candle_time, datetime):
            candle_time = int(candle_time.timestamp())
        elif isinstance(candle_time, (int, float)):
            candle_time = int(candle_time)
        else:
            continue  # Skip invalid candle time
        
        if entry_timestamp <= candle_time <= exit_timestamp:
            trade_candles.append(candle)
    
    if not trade_candles:
        # No candles found for this trade period
        return {
            "holding_time_minutes": 0,
            "holding_period_candles": 0,
            "mfe": None,
            "mae": None,
            "mfe_pct": None,
            "mae_pct": None,
            "entry_to_exit_return_pct": None,
            "mfe_given_up_pct": None
        }
    
    # Compute holding time in minutes
    holding_time_minutes = (exit_timestamp - entry_timestamp) / 60.0
    
    # Compute holding period (number of 5-minute candles)
    holding_period_candles = len(trade_candles)
    
    # Compute MFE (Max Favorable Excursion)
    # MFE = max(high - entry_price) during trade period
    entry_price = trade.entry_price
    max_favorable = max(candle["high"] for candle in trade_candles) - entry_price
    mfe = round(max_favorable, 2)
    mfe_pct = round((mfe / entry_price * 100) if entry_price > 0 else 0.0, 2)
    
    # Compute MAE (Max Adverse Excursion)
    # MAE = max(entry_price - low) during trade period
    min_adverse = entry_price - min(candle["low"] for candle in trade_candles)
    mae = round(min_adverse, 2)
    mae_pct = round((mae / entry_price * 100) if entry_price > 0 else 0.0, 2)
    
    # Compute entry_to_exit_return_pct
    exit_price = trade.exit_price
    entry_to_exit_return_pct = round(((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0, 2)
    
    # Compute MFE_given_up_pct (exit quality metric)
    # MFE_given_up_pct = (MFE - realized_pnl_per_share) / MFE * 100
    # This shows how much of the favorable move was captured
    # realized_pnl_per_share = (exit_price - entry_price) per share
    realized_pnl = trade.pnl if trade.pnl is not None else 0.0
    realized_pnl_per_share = (realized_pnl / trade.shares) if trade.shares > 0 else 0.0
    mfe_given_up_pct = None
    if mfe > 0:
        # MFE_given_up = MFE - realized_pnl_per_share
        # This is the amount of favorable move that was not captured
        mfe_given_up = mfe - realized_pnl_per_share
        mfe_given_up_pct = round((mfe_given_up / mfe * 100) if mfe > 0 else 0.0, 2)
    
    return {
        "holding_time_minutes": round(holding_time_minutes, 1),
        "holding_period_candles": holding_period_candles,
        "mfe": mfe,
        "mae": mae,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "entry_to_exit_return_pct": entry_to_exit_return_pct,
        "mfe_given_up_pct": mfe_given_up_pct
    }


def compute_intraday_aggregate_metrics(
    trades: List,
    intraday_candles: List[Dict]
) -> Dict:
    """
    Compute aggregate intraday trade metrics.
    
    Args:
        trades: List of Trade objects (from database)
        intraday_candles: List of 5-minute candle dicts (for computing per-trade diagnostics)
    
    Returns:
        Dict with aggregate metrics:
        - total_trades: Total number of closed trades
        - win_rate: Percentage of winning trades
        - average_win: Average P&L of winning trades
        - average_loss: Average P&L of losing trades
        - expectancy: Expected value per trade
        - profit_factor: Gross wins / gross losses
        - average_holding_time_minutes: Average holding time in minutes
        - average_holding_period_candles: Average holding period in 5-minute candles
        - average_mfe: Average Max Favorable Excursion
        - average_mae: Average Max Adverse Excursion
        - average_mfe_given_up_pct: Average MFE given up percentage
    """
    closed_trades = [t for t in trades if t.exit_time is not None and t.pnl is not None]
    
    if not closed_trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "average_holding_time_minutes": 0.0,
            "average_holding_period_candles": 0.0,
            "average_mfe": 0.0,
            "average_mae": 0.0,
            "average_mfe_given_up_pct": None
        }
    
    # Separate wins and losses
    wins = [t.pnl for t in closed_trades if t.pnl > 0]
    losses = [t.pnl for t in closed_trades if t.pnl < 0]
    
    # Core aggregate metrics
    total_trades = len(closed_trades)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    average_win = statistics.mean(wins) if wins else 0.0
    average_loss = statistics.mean(losses) if losses else 0.0
    
    # Expectancy = (win_rate * average_win) - (loss_rate * abs(average_loss))
    win_rate_decimal = len(wins) / total_trades if total_trades > 0 else 0.0
    loss_rate_decimal = len(losses) / total_trades if total_trades > 0 else 0.0
    expectancy = (win_rate_decimal * average_win) - (loss_rate_decimal * abs(average_loss))
    
    # Profit factor = gross wins / gross losses
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (float('inf') if gross_wins > 0 else 0.0)
    
    # Compute per-trade diagnostics for aggregate stats
    holding_times_minutes = []
    holding_periods_candles = []
    mfe_values = []
    mae_values = []
    mfe_given_up_pcts = []
    
    for trade in closed_trades:
        diagnostics = compute_intraday_trade_diagnostics(trade, intraday_candles)
        
        if diagnostics["holding_time_minutes"] is not None:
            holding_times_minutes.append(diagnostics["holding_time_minutes"])
        if diagnostics["holding_period_candles"] is not None:
            holding_periods_candles.append(diagnostics["holding_period_candles"])
        if diagnostics["mfe"] is not None:
            mfe_values.append(diagnostics["mfe"])
        if diagnostics["mae"] is not None:
            mae_values.append(diagnostics["mae"])
        if diagnostics["mfe_given_up_pct"] is not None:
            mfe_given_up_pcts.append(diagnostics["mfe_given_up_pct"])
    
    average_holding_time_minutes = statistics.mean(holding_times_minutes) if holding_times_minutes else 0.0
    average_holding_period_candles = statistics.mean(holding_periods_candles) if holding_periods_candles else 0.0
    average_mfe = statistics.mean(mfe_values) if mfe_values else 0.0
    average_mae = statistics.mean(mae_values) if mae_values else 0.0
    average_mfe_given_up_pct = statistics.mean(mfe_given_up_pcts) if mfe_given_up_pcts else None
    
    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "average_win": round(average_win, 2),
        "average_loss": round(average_loss, 2),
        "expectancy": round(expectancy, 2),
        "profit_factor": round(profit_factor, 2) if not (isinstance(profit_factor, float) and (profit_factor == float('inf') or profit_factor == float('-inf'))) else None,
        "average_holding_time_minutes": round(average_holding_time_minutes, 1),
        "average_holding_period_candles": round(average_holding_period_candles, 1),
        "average_mfe": round(average_mfe, 2),
        "average_mae": round(average_mae, 2),
        "average_mfe_given_up_pct": round(average_mfe_given_up_pct, 2) if average_mfe_given_up_pct is not None else None
    }


def compute_frequency_and_session_metrics(
    trades: List
) -> Dict:
    """
    Compute frequency and session-level diagnostics for intraday trades.
    
    Args:
        trades: List of Trade objects (from database)
    
    Returns:
        Dict with frequency and session metrics:
        - frequency_metrics: Per-day trade counts and statistics
        - session_metrics: Time-of-day analysis (Market Open, Midday, Power Hour)
        - clustering_metrics: Streak and clustering diagnostics
    """
    closed_trades = [t for t in trades if t.exit_time is not None and t.entry_time is not None]
    
    if not closed_trades:
        return {
            "frequency_metrics": {
                "trades_per_day": [],
                "average_trades_per_day": 0.0,
                "max_trades_in_a_day": 0,
                "number_of_trading_days_with_trades": 0,
                "percentage_of_days_with_trades": 0.0,
                "total_trading_days": 0
            },
            "session_metrics": {
                "market_open": {"trade_count": 0, "win_rate": 0.0, "expectancy": 0.0},
                "midday": {"trade_count": 0, "win_rate": 0.0, "expectancy": 0.0},
                "power_hour": {"trade_count": 0, "win_rate": 0.0, "expectancy": 0.0}
            },
            "clustering_metrics": {
                "consecutive_trades_same_day": 0,
                "average_time_between_trades_minutes": 0.0,
                "days_with_multiple_trades": 0,
                "max_consecutive_trades_same_day": 0
            }
        }
    
    # Group trades by date
    trades_by_date = defaultdict(list)
    for trade in closed_trades:
        entry_date = trade.entry_time.date()
        trades_by_date[entry_date].append(trade)
    
    # Compute per-day trade counts
    trades_per_day = [len(trades) for trades in trades_by_date.values()]
    total_trading_days = len(trades_by_date)
    number_of_trading_days_with_trades = total_trading_days
    
    # Calculate date range for percentage calculation
    if closed_trades:
        first_trade_date = min(t.entry_time.date() for t in closed_trades)
        last_trade_date = max(t.entry_time.date() for t in closed_trades)
        date_range_days = (last_trade_date - first_trade_date).days + 1
        percentage_of_days_with_trades = (total_trading_days / date_range_days * 100) if date_range_days > 0 else 0.0
    else:
        date_range_days = 0
        percentage_of_days_with_trades = 0.0
    
    average_trades_per_day = statistics.mean(trades_per_day) if trades_per_day else 0.0
    max_trades_in_a_day = max(trades_per_day) if trades_per_day else 0
    
    # Time-of-day analysis
    # Market Open: 09:30-10:30 ET (13:30-14:30 UTC)
    # Midday: 10:30-14:30 ET (14:30-18:30 UTC)
    # Power Hour: 14:30-16:00 ET (18:30-20:00 UTC)
    # Note: Assuming UTC times - adjust if needed for different timezones
    session_trades = {
        "market_open": [],  # 09:30-10:30 ET (13:30-14:30 UTC)
        "midday": [],        # 10:30-14:30 ET (14:30-18:30 UTC)
        "power_hour": []     # 14:30-16:00 ET (18:30-20:00 UTC)
    }
    
    for trade in closed_trades:
        entry_time = trade.entry_time
        if isinstance(entry_time, datetime):
            hour = entry_time.hour
            minute = entry_time.minute
            time_minutes = hour * 60 + minute
            
            # Market Open: 13:30-14:30 UTC (09:30-10:30 ET)
            if 13 * 60 + 30 <= time_minutes < 14 * 60 + 30:
                session_trades["market_open"].append(trade)
            # Power Hour: 18:30-20:00 UTC (14:30-16:00 ET)
            elif 18 * 60 + 30 <= time_minutes < 20 * 60:
                session_trades["power_hour"].append(trade)
            # Midday: 14:30-18:30 UTC (10:30-14:30 ET)
            elif 14 * 60 + 30 <= time_minutes < 18 * 60 + 30:
                session_trades["midday"].append(trade)
    
    # Compute session metrics
    session_metrics = {}
    for session_name, session_trade_list in session_trades.items():
        if not session_trade_list:
            session_metrics[session_name] = {
                "trade_count": 0,
                "win_rate": 0.0,
                "expectancy": 0.0
            }
        else:
            wins = [t.pnl for t in session_trade_list if t.pnl and t.pnl > 0]
            losses = [t.pnl for t in session_trade_list if t.pnl and t.pnl < 0]
            
            trade_count = len(session_trade_list)
            win_rate = (len(wins) / trade_count * 100) if trade_count > 0 else 0.0
            
            # Expectancy = (win_rate * average_win) - (loss_rate * abs(average_loss))
            win_rate_decimal = len(wins) / trade_count if trade_count > 0 else 0.0
            loss_rate_decimal = len(losses) / trade_count if trade_count > 0 else 0.0
            average_win = statistics.mean(wins) if wins else 0.0
            average_loss = statistics.mean(losses) if losses else 0.0
            expectancy = (win_rate_decimal * average_win) - (loss_rate_decimal * abs(average_loss))
            
            session_metrics[session_name] = {
                "trade_count": trade_count,
                "win_rate": round(win_rate, 2),
                "expectancy": round(expectancy, 2)
            }
    
    # Clustering diagnostics
    # Sort trades by entry time
    sorted_trades = sorted(closed_trades, key=lambda t: t.entry_time)
    
    # Compute consecutive trades same day
    consecutive_trades_same_day = 0  # Count of pairs of consecutive trades
    max_consecutive_trades_same_day = 0  # Maximum streak length
    days_with_multiple_trades = 0
    
    for date, date_trades in trades_by_date.items():
        if len(date_trades) > 1:
            days_with_multiple_trades += 1
            # Sort trades on this day by entry time
            sorted_date_trades = sorted(date_trades, key=lambda t: t.entry_time)
            # Check for consecutive trades (within reasonable time window, e.g., 1 hour)
            current_streak = 1  # Start streak at 1 (first trade)
            for i in range(len(sorted_date_trades) - 1):
                time_diff = (sorted_date_trades[i+1].entry_time - sorted_date_trades[i].entry_time).total_seconds() / 60
                if time_diff <= 60:  # Within 1 hour
                    consecutive_trades_same_day += 1  # Count this pair
                    current_streak += 1  # Extend streak
                    max_consecutive_trades_same_day = max(max_consecutive_trades_same_day, current_streak)
                else:
                    # Streak broken - update max if needed, then reset
                    max_consecutive_trades_same_day = max(max_consecutive_trades_same_day, current_streak)
                    current_streak = 1  # Reset to 1 (new streak starts)
            # Check final streak after loop
            max_consecutive_trades_same_day = max(max_consecutive_trades_same_day, current_streak)
    
    # Compute average time between trades (minutes)
    time_between_trades = []
    for i in range(len(sorted_trades) - 1):
        time_diff = (sorted_trades[i+1].entry_time - sorted_trades[i].entry_time).total_seconds() / 60
        time_between_trades.append(time_diff)
    
    average_time_between_trades = statistics.mean(time_between_trades) if time_between_trades else 0.0
    
    return {
        "frequency_metrics": {
            "trades_per_day": trades_per_day,
            "average_trades_per_day": round(average_trades_per_day, 2),
            "max_trades_in_a_day": max_trades_in_a_day,
            "number_of_trading_days_with_trades": number_of_trading_days_with_trades,
            "percentage_of_days_with_trades": round(percentage_of_days_with_trades, 2),
            "total_trading_days": date_range_days
        },
        "session_metrics": session_metrics,
        "clustering_metrics": {
            "consecutive_trades_same_day": consecutive_trades_same_day,
            "average_time_between_trades_minutes": round(average_time_between_trades, 1),
            "days_with_multiple_trades": days_with_multiple_trades,
            "max_consecutive_trades_same_day": max_consecutive_trades_same_day
        }
    }
