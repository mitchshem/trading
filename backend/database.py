"""
Database models and setup for SQLite.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
from typing import Optional

# SQLite database file
DATABASE_URL = "sqlite:///./trading_signals.db"

# Create engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class Signal(Base):
    """
    Signal model for storing trading signals.
    
    FIX 2: EXPLICIT REPLAY ISOLATION
    - replay_id = None: Live trading signals
    - replay_id = UUID: Replay signals (isolated from live trading)
    """
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    # FIX 1: CANONICAL TIME HANDLING - UTC timezone-aware datetime only
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    symbol = Column(String, index=True)
    signal = Column(String)  # "BUY", "EXIT", "HOLD"
    price = Column(Float)
    reason = Column(String)
    # FIX 2: EXPLICIT REPLAY ISOLATION - None for live trading, UUID for replays
    replay_id = Column(String, nullable=True, index=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "symbol": self.symbol,
            "signal": self.signal,
            "price": self.price,
            "reason": self.reason
        }


class Trade(Base):
    """
    Trade model for storing executed trades.
    
    FIX 2: EXPLICIT REPLAY ISOLATION
    - replay_id = None: Live trading trades
    - replay_id = UUID: Replay trades (isolated from live trading)
    """
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    # FIX 1: CANONICAL TIME HANDLING - UTC timezone-aware datetime only
    entry_time = Column(DateTime, index=True)
    entry_price = Column(Float)
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    shares = Column(Integer)
    pnl = Column(Float, nullable=True)  # Realized P&L (null if position still open)
    reason = Column(String, nullable=True)  # Exit reason
    # FIX 2: EXPLICIT REPLAY ISOLATION - None for live trading, UUID for replays
    replay_id = Column(String, nullable=True, index=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "shares": self.shares,
            "pnl": self.pnl,
            "reason": self.reason
        }


class EquityCurve(Base):
    """
    Equity curve model for tracking account equity over time.
    
    FIX 2: EXPLICIT REPLAY ISOLATION
    - replay_id = None: Live trading equity curve
    - replay_id = UUID: Replay equity curve (isolated from live trading)
    """
    __tablename__ = "equity_curve"
    
    id = Column(Integer, primary_key=True, index=True)
    # FIX 1: CANONICAL TIME HANDLING - UTC timezone-aware datetime only
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    equity = Column(Float)
    # FIX 2: EXPLICIT REPLAY ISOLATION - None for live trading, UUID for replays
    replay_id = Column(String, nullable=True, index=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "equity": self.equity
        }


class ReplaySummary(Base):
    """
    Replay summary model for storing completed replay results.
    Used for evaluation, determinism verification, and historical tracking.
    """
    __tablename__ = "replay_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    replay_id = Column(String, unique=True, index=True, nullable=False)  # UUID
    symbol = Column(String, index=True, nullable=False)
    start_date = Column(String, nullable=True)  # YYYY-MM-DD format
    end_date = Column(String, nullable=True)  # YYYY-MM-DD format
    source = Column(String, nullable=True)  # e.g., "yahoo_finance", "manual"
    candle_count = Column(Integer, nullable=False)
    trade_count = Column(Integer, nullable=False)
    final_equity = Column(Float, nullable=False)
    net_pnl = Column(Float, nullable=False)  # Net P&L (final_equity - initial_equity)
    max_drawdown_pct = Column(Float, nullable=False)
    max_drawdown_absolute = Column(Float, nullable=False)
    sharpe_proxy = Column(Float, nullable=True)  # Risk-adjusted metric
    allowed_entry_regimes = Column(String, nullable=True)  # JSON string of allowed regimes, e.g. '["TREND_UP"]' or None for all
    # FIX 1: CANONICAL TIME HANDLING - UTC timezone-aware datetime only
    timestamp_completed = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "replay_id": self.replay_id,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "source": self.source,
            "candle_count": self.candle_count,
            "trade_count": self.trade_count,
            "final_equity": self.final_equity,
            "net_pnl": self.net_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_absolute": self.max_drawdown_absolute,
            "sharpe_proxy": self.sharpe_proxy,
            "allowed_entry_regimes": self.allowed_entry_regimes,  # JSON string or None - shows which regimes were allowed for entries
            "timestamp_completed": self.timestamp_completed.isoformat() if self.timestamp_completed else None
        }


class DecisionLog(Base):
    """
    Decision log for every strategy evaluation.

    Logs the full context of each decision: candle data, indicator values,
    signal produced, broker action, and risk state. Essential for:
    - Debugging live vs. backtest discrepancies
    - Auditing strategy behavior
    - Detecting anomalies or unexpected state transitions
    """
    __tablename__ = "decision_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    symbol = Column(String, index=True)
    strategy_name = Column(String, index=True)

    # Candle data at evaluation time
    candle_open = Column(Float)
    candle_high = Column(Float)
    candle_low = Column(Float)
    candle_close = Column(Float)
    candle_volume = Column(Integer)

    # Indicator snapshot (JSON string for flexibility)
    indicator_values = Column(Text, nullable=True)

    # Strategy output
    signal = Column(String)  # BUY, EXIT, HOLD
    signal_reason = Column(String, nullable=True)
    stop_distance = Column(Float, nullable=True)

    # Broker state at decision time
    has_position = Column(Integer, default=0)  # 0/1 boolean
    position_entry_price = Column(Float, nullable=True)
    equity = Column(Float, nullable=True)
    cash = Column(Float, nullable=True)
    daily_pnl = Column(Float, nullable=True)

    # Risk state
    trade_blocked = Column(Integer, default=0)  # 0/1 boolean
    pending_orders_count = Column(Integer, default=0)

    # Anomaly flags (JSON string)
    anomaly_flags = Column(Text, nullable=True)

    # Broker action taken (may differ from signal due to risk controls)
    broker_action = Column(String, nullable=True)  # BUY, EXIT, BLOCKED, NONE
    broker_action_reason = Column(String, nullable=True)

    # Replay isolation
    replay_id = Column(String, nullable=True, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "candle": {
                "open": self.candle_open,
                "high": self.candle_high,
                "low": self.candle_low,
                "close": self.candle_close,
                "volume": self.candle_volume,
            },
            "indicator_values": self.indicator_values,
            "signal": self.signal,
            "signal_reason": self.signal_reason,
            "stop_distance": self.stop_distance,
            "has_position": bool(self.has_position),
            "position_entry_price": self.position_entry_price,
            "equity": self.equity,
            "cash": self.cash,
            "daily_pnl": self.daily_pnl,
            "trade_blocked": bool(self.trade_blocked),
            "pending_orders_count": self.pending_orders_count,
            "anomaly_flags": self.anomaly_flags,
            "broker_action": self.broker_action,
            "broker_action_reason": self.broker_action_reason,
        }


class NotificationPrefs(Base):
    """
    Notification preferences for the solo trader.
    Single-row table (id=1). Stores SMTP config, channel toggles,
    and per-event-category notification preferences.
    """
    __tablename__ = "notification_prefs"

    id = Column(Integer, primary_key=True, default=1)

    # Master toggles (0/1 booleans)
    email_enabled = Column(Integer, default=0)
    browser_enabled = Column(Integer, default=1)

    # SMTP configuration (nullable = not configured)
    email_address = Column(String, nullable=True)
    smtp_host = Column(String, nullable=True)
    smtp_port = Column(Integer, default=587)
    smtp_user = Column(String, nullable=True)
    smtp_password = Column(String, nullable=True)
    smtp_use_tls = Column(Integer, default=1)

    # Minimum severity filter: "info", "warning", "critical"
    min_severity = Column(String, default="info")

    # Per-category email toggles (0/1 booleans)
    email_trade_executed = Column(Integer, default=1)
    email_kill_switch = Column(Integer, default=1)
    email_anomaly_detected = Column(Integer, default=1)
    email_risk_limit_breached = Column(Integer, default=1)
    email_loop_status_change = Column(Integer, default=0)
    email_daily_summary = Column(Integer, default=1)
    email_system_error = Column(Integer, default=1)

    # Per-category browser toggles (0/1 booleans)
    browser_trade_executed = Column(Integer, default=1)
    browser_kill_switch = Column(Integer, default=1)
    browser_anomaly_detected = Column(Integer, default=1)
    browser_risk_limit_breached = Column(Integer, default=1)
    browser_loop_status_change = Column(Integer, default=1)
    browser_daily_summary = Column(Integer, default=0)
    browser_system_error = Column(Integer, default=1)

    # Timestamp of last update
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        """Serialize preferences. Omits smtp_user/smtp_password for security."""
        return {
            "email_enabled": bool(self.email_enabled),
            "browser_enabled": bool(self.browser_enabled),
            "email_address": self.email_address,
            "smtp_configured": bool(self.smtp_host and self.email_address),
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_use_tls": bool(self.smtp_use_tls),
            "min_severity": self.min_severity,
            "email_categories": {
                "trade_executed": bool(self.email_trade_executed),
                "kill_switch": bool(self.email_kill_switch),
                "anomaly_detected": bool(self.email_anomaly_detected),
                "risk_limit_breached": bool(self.email_risk_limit_breached),
                "loop_status_change": bool(self.email_loop_status_change),
                "daily_summary": bool(self.email_daily_summary),
                "system_error": bool(self.email_system_error),
            },
            "browser_categories": {
                "trade_executed": bool(self.browser_trade_executed),
                "kill_switch": bool(self.browser_kill_switch),
                "anomaly_detected": bool(self.browser_anomaly_detected),
                "risk_limit_breached": bool(self.browser_risk_limit_breached),
                "loop_status_change": bool(self.browser_loop_status_change),
                "daily_summary": bool(self.browser_daily_summary),
                "system_error": bool(self.browser_system_error),
            },
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# Create tables
def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


# Dependency to get DB session
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
