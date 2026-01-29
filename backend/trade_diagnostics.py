"""
Trade diagnostics module for analyzing trade performance.
Computes MFE (Max Favorable Excursion), MAE (Max Adverse Excursion), and holding periods.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import statistics


def compute_trade_diagnostics(
    trade,
    candles: List[Dict]
) -> Dict:
    """
    Compute diagnostics for a single trade.
    
    Args:
        trade: Trade object from database
        candles: List of candle dicts with keys: time, open, high, low, close, volume
                 Candles should be ordered by time ascending
    
    Returns:
        Dict with diagnostics:
        - holding_period_candles: Number of candles between entry and exit
        - mfe: Max Favorable Excursion (best price reached - entry_price)
        - mae: Max Adverse Excursion (entry_price - worst price reached)
        - mfe_pct: MFE as percentage of entry price
        - mae_pct: MAE as percentage of entry price
    """
    if not trade.exit_time or not trade.entry_time:
        # Open position - can't compute diagnostics
        return {
            "holding_period_candles": None,
            "mfe": None,
            "mae": None,
            "mfe_pct": None,
            "mae_pct": None
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
    for candle in candles:
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
            "holding_period_candles": 0,
            "mfe": None,
            "mae": None,
            "mfe_pct": None,
            "mae_pct": None
        }
    
    # Compute holding period (number of candles)
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
    
    return {
        "holding_period_candles": holding_period_candles,
        "mfe": mfe,
        "mae": mae,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct
    }


def compute_aggregate_diagnostics(
    trades: List,
    candles: List[Dict]
) -> Dict:
    """
    Compute aggregate trade diagnostics.
    
    Args:
        trades: List of Trade objects
        candles: List of candle dicts (for computing per-trade diagnostics)
    
    Returns:
        Dict with aggregate metrics:
        - average_win: Average P&L of winning trades
        - average_loss: Average P&L of losing trades
        - expectancy: Expected value per trade
        - profit_factor: Gross wins / gross losses
        - average_holding_period: Average holding period in candles
        - average_mfe: Average Max Favorable Excursion
        - average_mae: Average Max Adverse Excursion
    """
    closed_trades = [t for t in trades if t.exit_time is not None and t.pnl is not None]
    
    if not closed_trades:
        return {
            "average_win": 0.0,
            "average_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "average_holding_period": 0.0,
            "average_mfe": 0.0,
            "average_mae": 0.0
        }
    
    # Separate wins and losses
    wins = [t.pnl for t in closed_trades if t.pnl > 0]
    losses = [t.pnl for t in closed_trades if t.pnl < 0]
    
    # Compute aggregate metrics
    average_win = statistics.mean(wins) if wins else 0.0
    average_loss = statistics.mean(losses) if losses else 0.0
    
    # Expectancy = (win_rate * average_win) - (loss_rate * abs(average_loss))
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0
    loss_rate = len(losses) / len(closed_trades) if closed_trades else 0.0
    expectancy = (win_rate * average_win) - (loss_rate * abs(average_loss))
    
    # Profit factor = gross wins / gross losses
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (float('inf') if gross_wins > 0 else 0.0)
    
    # Compute per-trade diagnostics for aggregate stats
    holding_periods = []
    mfe_values = []
    mae_values = []
    
    for trade in closed_trades:
        diagnostics = compute_trade_diagnostics(trade, candles)
        if diagnostics["holding_period_candles"] is not None:
            holding_periods.append(diagnostics["holding_period_candles"])
        if diagnostics["mfe"] is not None:
            mfe_values.append(diagnostics["mfe"])
        if diagnostics["mae"] is not None:
            mae_values.append(diagnostics["mae"])
    
    average_holding_period = statistics.mean(holding_periods) if holding_periods else 0.0
    average_mfe = statistics.mean(mfe_values) if mfe_values else 0.0
    average_mae = statistics.mean(mae_values) if mae_values else 0.0
    
    return {
        "average_win": round(average_win, 2),
        "average_loss": round(average_loss, 2),
        "expectancy": round(expectancy, 2),
        "profit_factor": round(profit_factor, 2) if not (isinstance(profit_factor, float) and (profit_factor == float('inf') or profit_factor == float('-inf'))) else None,
        "average_holding_period": round(average_holding_period, 1),
        "average_mfe": round(average_mfe, 2),
        "average_mae": round(average_mae, 2)
    }
