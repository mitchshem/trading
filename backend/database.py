"""
Database models and setup for SQLite.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
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
