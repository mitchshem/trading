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
    """Signal model for storing trading signals."""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    # AUDIT FIX: Use timezone-aware UTC datetime (datetime.utcnow is deprecated)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    symbol = Column(String, index=True)
    signal = Column(String)  # "BUY", "EXIT", "HOLD"
    price = Column(Float)
    reason = Column(String)
    replay_id = Column(String, nullable=True, index=True)  # None for live trading, UUID for replays
    
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
    """Trade model for storing executed trades."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    entry_time = Column(DateTime, index=True)
    entry_price = Column(Float)
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    shares = Column(Integer)
    pnl = Column(Float, nullable=True)  # Realized P&L (null if position still open)
    reason = Column(String, nullable=True)  # Exit reason
    replay_id = Column(String, nullable=True, index=True)  # None for live trading, UUID for replays
    
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
    """Equity curve model for tracking account equity over time."""
    __tablename__ = "equity_curve"
    
    id = Column(Integer, primary_key=True, index=True)
    # AUDIT FIX: Use timezone-aware UTC datetime (datetime.utcnow is deprecated)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    equity = Column(Float)
    replay_id = Column(String, nullable=True, index=True)  # None for live trading, UUID for replays
    
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
