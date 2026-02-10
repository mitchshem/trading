"""
Decision logging module for strategy evaluations.

Logs the full context of every strategy evaluation to the database,
enabling:
- Live vs. backtest comparison (determinism verification)
- Debugging unexpected behavior
- Audit trail for all trading decisions
- Anomaly detection input
"""

import json
from typing import Dict, Optional, List
from datetime import datetime, timezone

from database import DecisionLog


def log_decision(
    db_session,
    *,
    timestamp: datetime,
    symbol: str,
    strategy_name: str,
    candle: Dict,
    indicator_snapshot: Optional[Dict] = None,
    signal: str,
    signal_reason: Optional[str] = None,
    stop_distance: Optional[float] = None,
    has_position: bool = False,
    position_entry_price: Optional[float] = None,
    equity: Optional[float] = None,
    cash: Optional[float] = None,
    daily_pnl: Optional[float] = None,
    trade_blocked: bool = False,
    pending_orders_count: int = 0,
    anomaly_flags: Optional[List[str]] = None,
    broker_action: Optional[str] = None,
    broker_action_reason: Optional[str] = None,
    replay_id: Optional[str] = None,
) -> DecisionLog:
    """
    Log a single strategy evaluation decision.

    Args:
        db_session: SQLAlchemy session
        timestamp: UTC datetime of this evaluation
        symbol: Trading symbol
        strategy_name: Name of the strategy evaluated
        candle: Current candle dict (open, high, low, close, volume)
        indicator_snapshot: Dict of indicator values at evaluation time
        signal: Strategy signal (BUY, EXIT, HOLD)
        signal_reason: Human-readable reason for the signal
        stop_distance: ATR-based stop distance (if BUY)
        has_position: Whether broker has an open position
        position_entry_price: Entry price of current position (if any)
        equity: Broker equity at decision time
        cash: Broker cash at decision time
        daily_pnl: Daily P&L at decision time
        trade_blocked: Whether trading is blocked (kill-switch active)
        pending_orders_count: Number of pending orders
        anomaly_flags: List of detected anomalies (e.g., ["gap_up_5pct", "volume_spike"])
        broker_action: Action taken by broker (may differ from signal)
        broker_action_reason: Why broker action differs from signal
        replay_id: UUID for replay isolation (None for live)

    Returns:
        The created DecisionLog record
    """
    entry = DecisionLog(
        timestamp=timestamp,
        symbol=symbol,
        strategy_name=strategy_name,
        candle_open=candle.get("open"),
        candle_high=candle.get("high"),
        candle_low=candle.get("low"),
        candle_close=candle.get("close"),
        candle_volume=candle.get("volume"),
        indicator_values=json.dumps(indicator_snapshot) if indicator_snapshot else None,
        signal=signal,
        signal_reason=signal_reason,
        stop_distance=stop_distance,
        has_position=1 if has_position else 0,
        position_entry_price=position_entry_price,
        equity=equity,
        cash=cash,
        daily_pnl=daily_pnl,
        trade_blocked=1 if trade_blocked else 0,
        pending_orders_count=pending_orders_count,
        anomaly_flags=json.dumps(anomaly_flags) if anomaly_flags else None,
        broker_action=broker_action,
        broker_action_reason=broker_action_reason,
        replay_id=replay_id,
    )
    db_session.add(entry)
    db_session.flush()  # Get the ID without committing
    return entry


def get_decisions(
    db_session,
    symbol: Optional[str] = None,
    replay_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict]:
    """
    Query decision logs with optional filtering.

    Args:
        db_session: SQLAlchemy session
        symbol: Filter by symbol (optional)
        replay_id: Filter by replay_id (None = live trading only)
        limit: Max records to return
        offset: Records to skip

    Returns:
        List of decision log dicts
    """
    query = db_session.query(DecisionLog)

    if symbol:
        query = query.filter(DecisionLog.symbol == symbol)
    if replay_id is not None:
        query = query.filter(DecisionLog.replay_id == replay_id)
    else:
        query = query.filter(DecisionLog.replay_id.is_(None))

    query = query.order_by(DecisionLog.timestamp.desc())
    query = query.offset(offset).limit(limit)

    return [entry.to_dict() for entry in query.all()]


def count_decisions(
    db_session,
    symbol: Optional[str] = None,
    replay_id: Optional[str] = None,
) -> int:
    """Count decision log entries matching filters."""
    query = db_session.query(DecisionLog)
    if symbol:
        query = query.filter(DecisionLog.symbol == symbol)
    if replay_id is not None:
        query = query.filter(DecisionLog.replay_id == replay_id)
    else:
        query = query.filter(DecisionLog.replay_id.is_(None))
    return query.count()
