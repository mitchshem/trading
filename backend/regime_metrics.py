"""
Regime-level metrics computation.
Aggregates trade performance by market regime.
"""

from typing import List, Dict, Optional
import statistics
from regime_classifier import get_regime_at_timestamp


def compute_regime_metrics(
    trades: List,
    candles: List[Dict]
) -> Dict[str, Dict]:
    """
    Compute aggregate metrics for each market regime.
    
    Args:
        trades: List of Trade objects
        candles: List of candle dicts (for regime classification)
    
    Returns:
        Dict mapping regime name to metrics:
        {
            "TREND_UP": {
                "trade_count": 10,
                "win_rate": 60.0,
                "average_win": 1234.56,
                "average_loss": -567.89,
                "expectancy": 234.12,
                "max_drawdown_contribution": -500.00
            },
            "TREND_DOWN": {...},
            "CHOP": {...}
        }
    """
    # Group trades by entry regime
    regime_trades = {
        "TREND_UP": [],
        "TREND_DOWN": [],
        "CHOP": []
    }
    
    # Classify each trade's entry and exit regime
    for trade in trades:
        if not trade.entry_time:
            continue
        
        # Get entry regime
        entry_regime = get_regime_at_timestamp(candles, trade.entry_time)
        if entry_regime is None:
            entry_regime = "CHOP"  # Default to CHOP if can't classify
        
        # Store trade with regime info
        regime_trades[entry_regime].append({
            "trade": trade,
            "entry_regime": entry_regime,
            "exit_regime": get_regime_at_timestamp(candles, trade.exit_time) if trade.exit_time else None
        })
    
    # Compute metrics for each regime
    regime_metrics = {}
    
    for regime_name, regime_trade_list in regime_trades.items():
        if not regime_trade_list:
            regime_metrics[regime_name] = {
                "trade_count": 0,
                "win_rate": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "expectancy": 0.0,
                "max_drawdown_contribution": 0.0
            }
            continue
        
        # Get closed trades only
        closed_trades = [
            rt for rt in regime_trade_list
            if rt["trade"].exit_time is not None and rt["trade"].pnl is not None
        ]
        
        if not closed_trades:
            regime_metrics[regime_name] = {
                "trade_count": len(regime_trade_list),
                "win_rate": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "expectancy": 0.0,
                "max_drawdown_contribution": 0.0
            }
            continue
        
        # Separate wins and losses
        wins = [rt["trade"].pnl for rt in closed_trades if rt["trade"].pnl > 0]
        losses = [rt["trade"].pnl for rt in closed_trades if rt["trade"].pnl < 0]
        
        # Compute metrics
        trade_count = len(closed_trades)
        win_rate = (len(wins) / trade_count * 100) if trade_count > 0 else 0.0
        average_win = statistics.mean(wins) if wins else 0.0
        average_loss = statistics.mean(losses) if losses else 0.0
        
        # Expectancy = (win_rate * average_win) - (loss_rate * abs(average_loss))
        win_rate_pct = len(wins) / trade_count if trade_count > 0 else 0.0
        loss_rate_pct = len(losses) / trade_count if trade_count > 0 else 0.0
        expectancy = (win_rate_pct * average_win) - (loss_rate_pct * abs(average_loss))
        
        # Max drawdown contribution: sum of all negative P&L trades
        max_drawdown_contribution = sum([rt["trade"].pnl for rt in closed_trades if rt["trade"].pnl < 0])
        
        regime_metrics[regime_name] = {
            "trade_count": trade_count,
            "win_rate": round(win_rate, 2),
            "average_win": round(average_win, 2),
            "average_loss": round(average_loss, 2),
            "expectancy": round(expectancy, 2),
            "max_drawdown_contribution": round(max_drawdown_contribution, 2)
        }
    
    return regime_metrics


def attach_regime_to_trades(
    trades: List,
    candles: List[Dict]
) -> List[Dict]:
    """
    Attach regime information to each trade.
    
    Args:
        trades: List of Trade objects
        candles: List of candle dicts (for regime classification)
    
    Returns:
        List of trade dicts with regime information added:
        {
            ...trade fields...,
            "entry_regime": "TREND_UP",
            "exit_regime": "TREND_DOWN"
        }
    """
    trades_with_regime = []
    
    for trade in trades:
        trade_dict = {
            "id": trade.id,
            "symbol": trade.symbol,
            "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
            "entry_price": trade.entry_price,
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "exit_price": trade.exit_price,
            "shares": trade.shares,
            "pnl": trade.pnl,
            "reason": trade.reason,
            "entry_regime": None,
            "exit_regime": None
        }
        
        # Get entry regime
        if trade.entry_time:
            entry_regime = get_regime_at_timestamp(candles, trade.entry_time)
            trade_dict["entry_regime"] = entry_regime or "CHOP"
        
        # Get exit regime
        if trade.exit_time:
            exit_regime = get_regime_at_timestamp(candles, trade.exit_time)
            trade_dict["exit_regime"] = exit_regime or "CHOP"
        
        trades_with_regime.append(trade_dict)
    
    return trades_with_regime
