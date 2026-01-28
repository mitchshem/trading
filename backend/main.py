from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import asyncio
import random
import time
import math
import json
import statistics
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import init_db, get_db, Signal, Trade, EquityCurve, ReplaySummary
from indicators import ema, atr
from strategy import ema_trend_v1, PositionState
from paper_broker import PaperBroker
from metrics import compute_metrics
from replay_engine import ReplayEngine
from market_data.yahoo import fetch_yahoo_candles, convert_to_replay_format
from market_data.csv_loader import load_csv_candles, convert_to_replay_format as csv_convert_to_replay_format
from market_data.stooq_loader import load_daily_candles, load_intraday_candles
from trade_diagnostics import compute_trade_diagnostics, compute_aggregate_diagnostics
from regime_metrics import compute_regime_metrics, attach_regime_to_trades
from intraday_replay import IntradayReplayEngine
from intraday_diagnostics import compute_intraday_trade_diagnostics, compute_intraday_aggregate_metrics, compute_frequency_and_session_metrics
from walkforward import filter_candles_by_date_range, compute_window_metrics
from utils import ensure_utc_datetime, unix_to_utc_datetime, fmt, fmt_pct, fmt_currency

# ============================================================================
# STARTUP INSTRUCTIONS
# ============================================================================
# To run this FastAPI application:
#   1. Activate the virtual environment:
#      macOS/Linux: source venv/bin/activate
#      Windows: venv\Scripts\activate
#   2. Start the server:
#      python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# Why use 'python -m uvicorn' instead of just 'uvicorn'?
# - Ensures uvicorn runs from the activated virtual environment
# - Prevents "command not found" errors if uvicorn isn't in system PATH
# - Works consistently across different operating systems
# ============================================================================

app = FastAPI(title="Paper Trading API")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    print("\n" + "=" * 60)
    print("BACKEND STARTED")
    print("=" * 60)
    print("Backend URL: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("\nAvailable Replay Endpoints:")
    print("  POST /replay/start - Start historical replay")
    print("  GET  /replay/history - Get replay summaries")
    print("  GET  /replay/results?replay_id=XXX - Get replay results")
    print("  GET  /replay/status - Get replay status")
    print("\nOther Key Endpoints:")
    print("  GET  /symbols - List available symbols")
    print("  GET  /candles?symbol=XXX - Get historical candles")
    print("  GET  /account - Get account state")
    print("  GET  /trades - Get trade history")
    print("  GET  /metrics - Get performance metrics")
    print("  WebSocket /ws - Real-time candle stream")
    print("=" * 60 + "\n")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dow 30 symbols
DOW_30 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
]

# Nasdaq-100 symbols (top 100 by market cap)
NASDAQ_100 = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "PEP", "ADBE", "CMCSA", "CSCO", "INTC", "INTU", "AMGN", "TXN",
    "QCOM", "ISRG", "AMAT", "BKNG", "VRTX", "GILD", "ADI", "ADP", "LRCX", "REGN",
    "MU", "SNPS", "CDNS", "KLAC", "NXPI", "MELI", "ASML", "FTNT", "DXCM", "CTSH",
    "PAYX", "ODFL", "FAST", "CRWD", "XEL", "CTAS", "ANSS", "TEAM", "PCAR", "IDXX",
    "BKR", "GEHC", "ZS", "ON", "ENPH", "CDW", "MCHP", "MRVL", "ALGN", "ROST",
    "AEP", "DDOG", "CPRT", "TTD", "FANG", "WBD", "ILMN", "VRSK", "EXC", "EA",
    "EBAY", "DLTR", "ZS", "LULU", "CTSH", "FAST", "CSGP", "WDAY", "MNST", "TTWO",
    "CHTR", "NTES", "JD", "BIDU", "PDD", "NIO", "XPEV", "LI", "BILI", "TCOM"
]

# Combine and deduplicate (some symbols appear in both)
# Add SPY (S&P 500 ETF) for reliable daily data testing
# Add AAAU (Gold ETF) - liquid ETF used for research and intraday testing
ALL_SYMBOLS = sorted(list(set(DOW_30 + NASDAQ_100 + ["SPY", "AAAU"])))

# Store active WebSocket connections
active_connections: Dict[str, List[WebSocket]] = {}

# Track position state per symbol
position_states: Dict[str, PositionState] = {}

# Track candle history per symbol (for strategy evaluation)
candle_history: Dict[str, List[Dict]] = {}

# Global paper broker instance
broker = PaperBroker(initial_equity=100000.0)

# Global replay engine instance (for daily replays)
replay_engine = ReplayEngine(initial_equity=100000.0)

# Global intraday replay engine instance (for 5-minute replays)
# Risk parameters can be overridden per replay via API request
intraday_replay_engine = IntradayReplayEngine(
    initial_equity=100000.0,
    risk_per_trade_pct=0.25,
    max_daily_loss_pct=1.0,
    max_concurrent_positions=1
)

# Track if replay is running (prevents concurrent replay/live trading)
replay_running = False


class Candle(BaseModel):
    time: int  # Unix timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int


def generate_synthetic_candles(symbol: str, limit: int = 500) -> List[Candle]:
    """
    Generate deterministic synthetic candle data based on symbol name.
    Uses symbol hash to ensure same symbol always generates same pattern.
    """
    # Use symbol hash for deterministic randomness
    random.seed(hash(symbol) % 10000)
    
    candles = []
    base_price = 100.0 + (hash(symbol) % 1000) / 10.0  # Base price between 100-200
    
    # Start from 500 candles ago (5-minute candles = ~41 hours)
    now = int(time.time())
    start_time = now - (limit * 5 * 60)  # 5 minutes per candle
    
    current_price = base_price
    
    for i in range(limit):
        candle_time = start_time + (i * 5 * 60)
        
        # Generate price movement (random walk with slight upward bias)
        change_pct = (random.random() - 0.45) * 0.02  # -0.45 to 0.55, scaled to 2%
        current_price = current_price * (1 + change_pct)
        
        # Generate OHLC
        open_price = current_price
        volatility = random.random() * 0.01  # 1% volatility
        
        high_price = open_price * (1 + volatility * random.random())
        low_price = open_price * (1 - volatility * random.random())
        close_price = open_price * (1 + (random.random() - 0.5) * volatility)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        volume = int(random.random() * 1000000) + 100000
        
        candles.append(Candle(
            time=candle_time,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=volume
        ))
        
        current_price = close_price
    
    return candles


def generate_next_candle(symbol: str, last_candle: Candle) -> Candle:
    """Generate the next candle after the last one."""
    random.seed(hash(symbol) % 10000 + int(last_candle.time / 300))  # 300 = 5 min
    
    change_pct = (random.random() - 0.45) * 0.02
    current_price = last_candle.close * (1 + change_pct)
    
    open_price = last_candle.close
    volatility = random.random() * 0.01
    
    high_price = max(open_price, current_price) * (1 + volatility * random.random())
    low_price = min(open_price, current_price) * (1 - volatility * random.random())
    close_price = current_price
    
    high_price = max(high_price, open_price, close_price)
    low_price = min(low_price, open_price, close_price)
    
    volume = int(random.random() * 1000000) + 100000
    
    return Candle(
        time=last_candle.time + 300,  # 5 minutes later
        open=round(open_price, 2),
        high=round(high_price, 2),
        low=round(low_price, 2),
        close=round(close_price, 2),
        volume=volume
    )


@app.get("/")
async def root():
    return {"message": "Paper Trading API"}


@app.get("/symbols")
async def get_symbols():
    """Return list of allowed symbols (Dow 30 + Nasdaq-100)."""
    return {"symbols": ALL_SYMBOLS}


@app.get("/candles")
async def get_candles(symbol: str, limit: int = 500):
    """
    Return historical candle data for a symbol.
    Currently returns synthetic data.
    
    BUG FIX: Preserves existing candle history if:
    - An active WebSocket connection is streaming candles for this symbol
    - Candle history already exists and has accumulated candles
    
    This prevents overwriting accumulated candle history that's being used
    by active strategy evaluation, which would cause loss of historical context
    and potentially incorrect signals.
    """
    if symbol not in ALL_SYMBOLS:
        return {"error": f"Symbol {symbol} not in allowed list"}, 400
    
    # BUG FIX: Check if there's an active WebSocket connection for this symbol
    has_active_websocket = symbol in active_connections and len(active_connections[symbol]) > 0
    
    # BUG FIX: Preserve existing candle history if:
    # 1. There's an active WebSocket streaming candles, OR
    # 2. Candle history already exists and has candles
    if has_active_websocket or (symbol in candle_history and len(candle_history[symbol]) > 0):
        # Return existing history without overwriting
        existing_candles = candle_history[symbol]
        return {
            "symbol": symbol,
            "candles": existing_candles[-limit:] if len(existing_candles) > limit else existing_candles,
            "note": "Returning existing candle history (preserved due to active stream or existing history)"
        }
    
    # No active stream and no existing history - generate new candles
    candles = generate_synthetic_candles(symbol, limit)
    
    # Store candle history for strategy evaluation
    candle_history[symbol] = [c.dict() for c in candles]
    
    # Initialize position state if not exists
    if symbol not in position_states:
        position_states[symbol] = PositionState()
    
    return {"symbol": symbol, "candles": candle_history[symbol]}


@app.get("/signals")
async def get_signals(symbol: str, limit: int = 100, db: Session = Depends(get_db)):
    """
    Return trading signals for a symbol.
    Only includes live trading data (replay_id is None).
    """
    if symbol not in ALL_SYMBOLS:
        return {"error": f"Symbol {symbol} not in allowed list"}, 400
    
    signals = db.query(Signal).filter(
        Signal.symbol == symbol,
        Signal.replay_id.is_(None)
    ).order_by(
        Signal.timestamp.desc()
    ).limit(limit).all()
    
    return {
        "symbol": symbol,
        "signals": [s.to_dict() for s in reversed(signals)]  # Reverse to show oldest first
    }


@app.get("/account")
async def get_account():
    """
    Return account summary (equity, daily P&L, open positions).
    
    FIX 2: EXPLICIT REPLAY ISOLATION
    This endpoint returns live paper trading state only.
    Replay state is completely separate (different broker instance).
    """
    # Get current prices for all symbols with open positions
    current_prices = {}
    for symbol in broker.positions.keys():
        if symbol in candle_history and len(candle_history[symbol]) > 0:
            current_prices[symbol] = candle_history[symbol][-1]["close"]
    
    # Also include the symbol being watched if available
    # (This is a simple approach - in production you'd track all symbols)
    account_summary = broker.get_account_summary(current_prices)
    return account_summary


@app.get("/trades")
async def get_trades(symbol: Optional[str] = None, limit: int = 100, db: Session = Depends(get_db)):
    """
    Return executed trades.
    Only includes live trading data (replay_id is None).
    """
    query = db.query(Trade).filter(Trade.replay_id.is_(None))
    
    if symbol:
        if symbol not in ALL_SYMBOLS:
            return {"error": f"Symbol {symbol} not in allowed list"}, 400
        query = query.filter(Trade.symbol == symbol)
    
    trades = query.order_by(
        Trade.entry_time.desc()
    ).limit(limit).all()
    
    return {
        "trades": [t.to_dict() for t in reversed(trades)]  # Reverse to show oldest first
    }


@app.get("/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    """
    Return trading performance metrics computed from persisted data.
    Strategy-agnostic metrics that can be recomputed at any time.
    
    FIX 2: EXPLICIT REPLAY ISOLATION
    Only includes live trading data (replay_id is None).
    Replay metrics are accessed via /replay/results endpoint.
    """
    # Fetch only live trading trades and equity curve data (replay_id is None)
    trades = db.query(Trade).filter(Trade.replay_id.is_(None)).all()
    equity_curve = db.query(EquityCurve).filter(EquityCurve.replay_id.is_(None)).order_by(EquityCurve.timestamp).all()
    
    # Compute metrics
    metrics_snapshot = compute_metrics(trades, equity_curve)
    
    return metrics_snapshot.to_dict()


@app.get("/equity-curve")
async def get_equity_curve(limit: int = 1000, db: Session = Depends(get_db)):
    """
    Return equity curve data for charting.
    Only includes live trading data (replay_id is None).
    """
    equity_curve = db.query(EquityCurve).filter(EquityCurve.replay_id.is_(None)).order_by(EquityCurve.timestamp).limit(limit).all()
    
    return {
        "equity_curve": [
            {
                "timestamp": point.timestamp.isoformat() if point.timestamp else None,
                "equity": point.equity
            }
            for point in equity_curve
        ]
    }


def evaluate_strategy_on_candle_close(symbol: str, new_candle: Dict, db: Session):
    """
    Evaluate strategy when a candle closes, route through paper broker, and store trades.
    This is called when a new candle is generated (representing a closed candle).
    
    AUDIT FIX: Prevent duplicate candle processing by checking if candle already exists.
    """
    # Ensure we have candle history
    if symbol not in candle_history:
        # Initialize with historical candles
        historical_candles = generate_synthetic_candles(symbol, 500)
        candle_history[symbol] = [c.dict() for c in historical_candles]
    
    # AUDIT FIX: Check for duplicate candle (same timestamp) to prevent double processing
    candle_time = new_candle["time"]
    if candle_history[symbol] and candle_history[symbol][-1]["time"] == candle_time:
        # Duplicate candle detected - skip processing
        return None
    
    # Add new candle to history
    candle_history[symbol].append(new_candle)
    
    # Keep only last 500 candles for memory efficiency
    if len(candle_history[symbol]) > 500:
        candle_history[symbol] = candle_history[symbol][-500:]
    
    # Ensure we have position state
    if symbol not in position_states:
        position_states[symbol] = PositionState()
    
    position_state = position_states[symbol]
    
    # Get candles for indicator calculation
    candles = candle_history[symbol]
    
    # Need at least 50 candles for EMA(50)
    if len(candles) < 50:
        return None
    
    # Extract price arrays
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    
    # Calculate indicators
    ema20_values = ema(closes, 20)
    ema50_values = ema(closes, 50)
    atr14_values = atr(highs, lows, closes, 14)
    
    # Get current ATR for position sizing
    current_atr = atr14_values[-1] if atr14_values and atr14_values[-1] is not None else None
    
    # Evaluate strategy
    result = ema_trend_v1(
        candles=candles,
        ema20_values=ema20_values,
        ema50_values=ema50_values,
        atr14_values=atr14_values,
        position_state=position_state
    )
    
    # Store signal if not HOLD
    signal_stored = False
    if result["signal"] != "HOLD":
        # FIX 1: CANONICAL TIME HANDLING
        # Convert Unix timestamp (API boundary) to UTC datetime (internal representation)
        signal_time = unix_to_utc_datetime(new_candle["time"])
        ensure_utc_datetime(signal_time, f"signal timestamp for {symbol}")
        
        # FIX 2: EXPLICIT REPLAY ISOLATION
        # Live trading: replay_id must be None
        existing_signal = db.query(Signal).filter(
            Signal.symbol == symbol,
            Signal.timestamp == signal_time,
            Signal.signal == result["signal"],
            Signal.replay_id.is_(None)  # Only check live trading signals
        ).first()
        
        if not existing_signal:
            # FIX 2: EXPLICIT REPLAY ISOLATION
            # Live trading: replay_id is None (replay data uses UUID)
            signal = Signal(
                timestamp=signal_time,
                symbol=symbol,
                signal=result["signal"],
                price=new_candle["close"],
                reason=result["reason"],
                replay_id=None  # None for live trading, UUID for replays
            )
            db.add(signal)
            db.commit()
            signal_stored = True
    
    # FIX 2: AUTOMATIC STOP-LOSS ENFORCEMENT
    # Check stop-losses on every candle close (before processing signals)
    current_prices_all = {}
    for sym in broker.positions.keys():
        if sym in candle_history and len(candle_history[sym]) > 0:
            current_prices_all[sym] = candle_history[sym][-1]["close"]
    # Add current symbol's price
    current_prices_all[symbol] = new_candle["close"]
    
    stop_loss_exits = broker.check_stop_losses(current_prices_all, new_candle["time"])
    
    # Track which symbols had stop-loss exits to prevent double processing
    stop_loss_symbols = set()
    
    # Update trade records for stop-loss exits
    for exit_trade in stop_loss_exits:
        stop_loss_symbols.add(exit_trade["symbol"])
        # FIX 2: EXPLICIT REPLAY ISOLATION
        # Live trading: only query trades with replay_id=None
        open_trade = db.query(Trade).filter(
            Trade.symbol == exit_trade["symbol"],
            Trade.exit_time.is_(None),
            Trade.replay_id.is_(None)  # Only live trading trades
        ).first()
        
        if open_trade:
            # FIX 1: CANONICAL TIME HANDLING
            # Convert Unix timestamp to UTC datetime
            exit_time = unix_to_utc_datetime(exit_trade["timestamp"])
            ensure_utc_datetime(exit_time, f"stop-loss exit time for {exit_trade['symbol']}")
            open_trade.exit_time = exit_time
            open_trade.exit_price = exit_trade["exit_price"]
            open_trade.pnl = exit_trade["pnl"]
            open_trade.reason = exit_trade["reason"]
            db.commit()
        
        # Update position state if this was the current symbol
        if exit_trade["symbol"] == symbol:
            position_state.has_position = False
            position_state.entry_price = None
            position_state.entry_time = None
    
    # Route signal through paper broker
    trade_executed = None
    kill_switch_triggered = False
    
    # AUDIT FIX: Prevent double EXIT processing - if stop-loss already exited, skip signal EXIT
    if result["signal"] == "BUY" and current_atr is not None and symbol not in stop_loss_symbols:
        # Calculate stop distance (2 * ATR)
        stop_distance = 2 * current_atr
        
        # Execute BUY through broker
        trade_executed = broker.execute_buy(
            symbol=symbol,
            signal_price=new_candle["close"],
            stop_distance=stop_distance,
            timestamp=new_candle["time"]
        )
        
        if trade_executed:
            # FIX 1: CANONICAL TIME HANDLING
            # Convert Unix timestamp to UTC datetime
            entry_time = unix_to_utc_datetime(new_candle["time"])
            ensure_utc_datetime(entry_time, f"BUY entry time for {symbol}")
            
            # FIX 2: EXPLICIT REPLAY ISOLATION
            # Live trading: replay_id is None (replay data uses UUID)
            trade = Trade(
                symbol=symbol,
                entry_time=entry_time,
                entry_price=trade_executed["entry_price"],
                shares=trade_executed["shares"],
                exit_time=None,
                exit_price=None,
                pnl=None,
                reason=None,
                replay_id=None  # None for live trading, UUID for replays
            )
            db.add(trade)
            db.commit()
            
            # Update position state
            position_state.has_position = True
            position_state.entry_price = trade_executed["entry_price"]
            position_state.entry_time = new_candle["time"]
    
    elif result["signal"] == "EXIT" and symbol not in stop_loss_symbols:
        # AUDIT FIX: Only process signal EXIT if stop-loss didn't already exit this symbol
        # Execute EXIT through broker
        trade_executed = broker.execute_exit(
            symbol=symbol,
            signal_price=new_candle["close"],
            timestamp=new_candle["time"],
            reason=result["reason"]
        )
        
        if trade_executed:
            # FIX 2: EXPLICIT REPLAY ISOLATION
            # Live trading: only query trades with replay_id=None
            open_trade = db.query(Trade).filter(
                Trade.symbol == symbol,
                Trade.exit_time.is_(None),
                Trade.replay_id.is_(None)  # Only live trading trades
            ).first()
            
            if open_trade:
                # FIX 1: CANONICAL TIME HANDLING
                # Convert Unix timestamp to UTC datetime
                exit_time = unix_to_utc_datetime(new_candle["time"])
                ensure_utc_datetime(exit_time, f"EXIT time for {symbol}")
                open_trade.exit_time = exit_time
                open_trade.exit_price = trade_executed["exit_price"]
                open_trade.pnl = trade_executed["pnl"]
                open_trade.reason = trade_executed["reason"]
                db.commit()
            
            # Update position state
            position_state.has_position = False
            position_state.entry_price = None
            position_state.entry_time = None
    
    # FIX 4: KILL SWITCH BEHAVIOR (FAIL-SAFE)
    # AUDIT FIX: Ensure equity is updated before risk checks (order matters)
    broker.update_equity(current_prices_all)
    
    # Check and enforce risk controls on every candle close
    # This automatically closes all positions if daily loss limit is breached
    kill_switch_exits = broker.check_and_enforce_risk_controls(current_prices_all, new_candle["time"])
    
    # Update trade records for kill switch exits
    for exit_trade in kill_switch_exits:
        # FIX 2: EXPLICIT REPLAY ISOLATION
        # Live trading: only query trades with replay_id=None
        open_trade = db.query(Trade).filter(
            Trade.symbol == exit_trade["symbol"],
            Trade.exit_time.is_(None),
            Trade.replay_id.is_(None)  # Only live trading trades
        ).first()
        
        if open_trade:
            # FIX 1: CANONICAL TIME HANDLING
            # Convert Unix timestamp to UTC datetime
            exit_time = unix_to_utc_datetime(exit_trade["timestamp"])
            ensure_utc_datetime(exit_time, f"kill switch exit time for {exit_trade['symbol']}")
            open_trade.exit_time = exit_time
            open_trade.exit_price = exit_trade["exit_price"]
            open_trade.pnl = exit_trade["pnl"]
            open_trade.reason = exit_trade["reason"]
            db.commit()
        
        # Update position state if this was the current symbol
        if exit_trade["symbol"] == symbol:
            position_state.has_position = False
            position_state.entry_price = None
            position_state.entry_time = None
    
    if kill_switch_exits:
        kill_switch_triggered = True
    
    # Update equity curve (equity already updated above for risk checks)
    # FIX 1: CANONICAL TIME HANDLING
    # Convert Unix timestamp to UTC datetime
    equity_timestamp = unix_to_utc_datetime(new_candle["time"])
    ensure_utc_datetime(equity_timestamp, f"equity curve timestamp for {symbol}")
    
    # FIX 2: EXPLICIT REPLAY ISOLATION
    # Live trading: replay_id is None (replay data uses UUID)
    equity_point = EquityCurve(
        timestamp=equity_timestamp,
        equity=broker.equity,
        replay_id=None  # None for live trading, UUID for replays
    )
    db.add(equity_point)
    db.commit()
    
    # Return result with trade info
    if signal_stored:
        return {
            "signal": result,
            "trade": trade_executed,
            "kill_switch": kill_switch_triggered
        }
    
    return None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming candle updates."""
    await websocket.accept()
    symbol = None
    
    # Get database session
    db = next(get_db())
    
    try:
        # Wait for symbol selection
        data = await websocket.receive_json()
        symbol = data.get("symbol")
        
        if not symbol or symbol not in ALL_SYMBOLS:
            await websocket.send_json({"error": "Invalid symbol"})
            await websocket.close()
            return
        
        # Add connection to active connections
        if symbol not in active_connections:
            active_connections[symbol] = []
        active_connections[symbol].append(websocket)
        
        # Initialize candle history if not exists
        if symbol not in candle_history:
            candles = generate_synthetic_candles(symbol, 500)
            candle_history[symbol] = [c.dict() for c in candles]
        
        # Initialize position state if not exists
        if symbol not in position_states:
            position_states[symbol] = PositionState()
        
        # Get last candle from history
        last_candle_dict = candle_history[symbol][-1] if candle_history[symbol] else None
        if not last_candle_dict:
            await websocket.close()
            return
        
        # Convert to Candle object for generate_next_candle
        last_candle = Candle(**last_candle_dict)
        
        # Stream new candles every 5 seconds (simulating 5-minute candles)
        while True:
            await asyncio.sleep(5)  # Simulate 5-minute candle updates
            
            # Generate next candle (this represents a closed candle)
            next_candle = generate_next_candle(symbol, last_candle)
            next_candle_dict = next_candle.dict()
            
            # Evaluate strategy on candle close and route through broker
            strategy_result = evaluate_strategy_on_candle_close(symbol, next_candle_dict, db)
            
            # Send candle to client
            try:
                message = {
                    "symbol": symbol,
                    "candle": next_candle_dict
                }
                
                # Include signal and trade info if generated
                if strategy_result:
                    if "signal" in strategy_result:
                        message["signal"] = strategy_result["signal"]
                    if "trade" in strategy_result:
                        message["trade"] = strategy_result["trade"]
                    if "kill_switch" in strategy_result and strategy_result["kill_switch"]:
                        message["kill_switch"] = True
                
                await websocket.send_json(message)
                last_candle = next_candle
            except Exception as e:
                # Connection closed
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Remove connection
        if symbol and symbol in active_connections:
            active_connections[symbol] = [
                conn for conn in active_connections[symbol] if conn != websocket
            ]
        db.close()


# ============================================================================
# REPLAY (BACKTEST) ENDPOINTS
# ============================================================================

class ReplayRequest(BaseModel):
    symbol: str
    candles: Optional[List[Dict]] = None  # Optional: if not provided, will fetch from Yahoo Finance
    start_date: Optional[str] = None  # YYYY-MM-DD format, required if candles not provided
    end_date: Optional[str] = None    # YYYY-MM-DD format, required if candles not provided
    replay_id: Optional[str] = None
    allowed_entry_regimes: Optional[List[str]] = None  # Optional regime gate: ["TREND_UP"], ["TREND_DOWN"], ["CHOP"], or None for all


class ReplayCsvRequest(BaseModel):
    symbol: str
    csv_path: str
    replay_id: Optional[str] = None
    allowed_entry_regimes: Optional[List[str]] = None  # Optional regime gate: ["TREND_UP"], ["TREND_DOWN"], ["CHOP"], or None for all


class IntradayReplayCsvRequest(BaseModel):
    symbol: Optional[str] = None  # Single symbol (for backward compatibility)
    symbols: Optional[List[str]] = None  # List of symbols (for multi-symbol replay)
    interval: str = "5min"  # Data interval (currently only "5min" supported)
    exit_mode: str = "momentum_reversal"  # Exit mode: "momentum_reversal", "time_based", "mfe_trailing"
    exit_params: Optional[Dict] = None  # Optional exit parameters (e.g., {"time_candles": 6, "mfe_retrace_pct": 50})
    allowed_sessions: Optional[List[str]] = None  # Optional session gate: ["market_open", "midday", "power_hour"]
    entry_variant: Optional[str] = None  # Entry variant: "break_prev_high", "close_above_prev_close", "close_above_prev_low"
    diagnostic_mode: Optional[bool] = False  # If True, bypass daily regime and session gating for diagnostic purposes
    bypass_daily_regime_gate: Optional[bool] = False  # If True, bypass only daily regime gate (for gate isolation testing)
    bypass_session_gate: Optional[bool] = False  # If True, bypass only session gate (for gate isolation testing)
    replay_id: Optional[str] = None
    # Risk parameters
    starting_equity: Optional[float] = 100000.0  # Starting portfolio equity
    risk_per_trade_pct: Optional[float] = 0.25  # Risk per trade as % of equity (default: 0.25%)
    max_daily_loss_pct: Optional[float] = 1.0  # Max daily loss as % of equity (default: 1.0%)
    max_concurrent_positions: Optional[int] = 1  # Max concurrent positions (default: 1)
    # Date filtering (optional)
    start_date: Optional[str] = None  # YYYY-MM-DD format (inclusive)
    end_date: Optional[str] = None  # YYYY-MM-DD format (inclusive)


class WalkForwardWindow(BaseModel):
    """A single walk-forward window (train or test period)."""
    label: str  # e.g., "train_1", "test_1"
    start_date: str  # YYYY-MM-DD format (inclusive)
    end_date: str  # YYYY-MM-DD format (inclusive)


class WalkForwardRequest(BaseModel):
    """Walk-forward evaluation request."""
    symbols: List[str]  # List of symbols to evaluate
    interval: str = "5min"  # Data interval (currently only "5min" supported)
    exit_mode: str = "momentum_reversal"  # Exit mode
    exit_params: Optional[Dict] = None  # Exit parameters
    allowed_sessions: Optional[List[str]] = None  # Session gate
    # Risk parameters
    starting_equity: float = 100000.0  # Starting equity (reset for each window)
    risk_per_trade_pct: float = 0.25  # Risk per trade %
    max_daily_loss_pct: float = 1.0  # Max daily loss %
    max_concurrent_positions: int = 1  # Max concurrent positions
    # Walk-forward windows
    windows: List[WalkForwardWindow]  # List of train/test windows


@app.post("/replay/start")
async def start_replay(request: ReplayRequest, db: Session = Depends(get_db)):
    """
    Start a historical replay (backtest) using real market data from Yahoo Finance.
    
    Can be called in two ways:
    1. With pre-fetched candles: Provide 'candles' array
    2. With date range: Provide 'start_date' and 'end_date' to fetch from Yahoo Finance
    
    Safeguards:
    - Prevents concurrent replay/live trading
    - Validates symbol in allowed universe
    - Validates date range (max 1 year)
    - Validates candle ordering
    - Ensures UTC datetime validation throughout
    """
    global replay_running
    
    # Safeguard: Prevent concurrent replay
    if replay_running:
        return {"error": "Replay already in progress"}, 400
    
    # Safeguard: Validate symbol
    if request.symbol not in ALL_SYMBOLS:
        return {"error": f"Symbol {request.symbol} not in allowed list"}, 400
    
    # Determine data source: candles provided OR dates provided
    candles = None
    source = "manual"
    
    if request.candles:
        # Use provided candles
        candles = request.candles
        source = "manual"
    elif request.start_date and request.end_date:
        # Fetch from Yahoo Finance
        try:
            # Validate date format
            start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
        except ValueError as e:
            return {"error": f"Invalid date format. Use YYYY-MM-DD. Error: {e}"}, 400
        
        # Safeguard: Validate date range (max 1 year)
        date_range = (end_dt - start_dt).days
        if date_range > 365:
            return {"error": f"Date range exceeds 1 year limit. Requested: {date_range} days"}, 400
        
        if date_range <= 0:
            return {"error": "End date must be after start date"}, 400
        
        try:
            # Fetch candles from Yahoo Finance
            print(f"[REPLAY] Fetching DAILY candles from Yahoo Finance for {request.symbol}")
            yahoo_candles = fetch_yahoo_candles(
                symbol=request.symbol,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            # PREFLIGHT VALIDATION: Check if candle count == 0
            if not yahoo_candles or len(yahoo_candles) == 0:
                error_msg = f"No DAILY data returned from Yahoo Finance for {request.symbol} between {request.start_date} and {request.end_date}"
                print(f"[REPLAY] ERROR: {error_msg}")
                return {"error": error_msg}, 400
            
            print(f"[REPLAY] Fetched {len(yahoo_candles)} daily candles from Yahoo Finance")
            
            # Convert to replay format (Unix timestamps)
            candles = convert_to_replay_format(yahoo_candles)
            source = "yahoo_finance"
            
            # PREFLIGHT VALIDATION: Check candle count after conversion
            if not candles or len(candles) == 0:
                error_msg = f"No valid candles after conversion for {request.symbol}"
                print(f"[REPLAY] ERROR: {error_msg}")
                return {"error": error_msg}, 400
            
            print(f"[REPLAY] Converted to replay format: {len(candles)} candles")
            
        except ValueError as e:
            error_msg = str(e)
            print(f"[REPLAY] ERROR: {error_msg}")
            return {"error": error_msg}, 400
        except Exception as e:
            error_msg = f"Failed to fetch Yahoo Finance data: {str(e)}"
            print(f"[REPLAY] ERROR: {error_msg}")
            return {"error": error_msg}, 500
    else:
        return {"error": "Must provide either 'candles' array or both 'start_date' and 'end_date'"}, 400
    
    # GUARDRAILS: Validate candles meet minimum thresholds
    # For daily candles, a year has ~252 trading days, so 500 is too high
    # Use lower threshold for daily data
    MIN_CANDLES_FOR_REPLAY_DAILY = 50  # Minimum for daily data (about 2.5 months)
    MIN_CANDLES_FOR_REPLAY_INTRADAY = 500  # Minimum for intraday data
    MIN_CANDLES_FOR_INDICATORS = 50  # Minimum for EMA(50)
    
    # Determine minimum based on data source
    if source == "yahoo_finance":
        # Daily candles: use lower threshold
        MIN_CANDLES_FOR_REPLAY = MIN_CANDLES_FOR_REPLAY_DAILY
    else:
        # Manual/intraday candles: use higher threshold
        MIN_CANDLES_FOR_REPLAY = MIN_CANDLES_FOR_REPLAY_INTRADAY
    
    # PREFLIGHT VALIDATION: Check if candle count == 0 (must be explicit, no silent exit)
    if not candles or len(candles) == 0:
        error_msg = f"No candles provided for replay"
        print(f"[REPLAY] ERROR: {error_msg}")
        return {"error": error_msg}, 400
    
    print(f"[REPLAY] Preflight validation: {len(candles)} candles, minimum required: {MIN_CANDLES_FOR_INDICATORS} for indicators, {MIN_CANDLES_FOR_REPLAY} for replay")
    
    if len(candles) < MIN_CANDLES_FOR_INDICATORS:
        error_msg = f"Need at least {MIN_CANDLES_FOR_INDICATORS} candles for replay (EMA(50) requires {MIN_CANDLES_FOR_INDICATORS} candles). Got {len(candles)}"
        print(f"[REPLAY] ERROR: {error_msg}")
        return {"error": error_msg}, 400
    
    if len(candles) < MIN_CANDLES_FOR_REPLAY:
        error_msg = f"Insufficient candles for meaningful evaluation. Got {len(candles)}, minimum {MIN_CANDLES_FOR_REPLAY} required."
        print(f"[REPLAY] ERROR: {error_msg}")
        return {
            "error": error_msg,
            "warning": f"Replay with fewer than {MIN_CANDLES_FOR_REPLAY} candles may not provide reliable performance metrics"
        }, 400
    
    print(f"[REPLAY] Preflight validation PASSED: {len(candles)} candles ready for replay")
    
    # GUARDRAILS: Validate date range if provided
    if request.start_date and request.end_date:
        try:
            start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
            date_range_days = (end_dt - start_dt).days
            
            MIN_DATE_RANGE_DAYS = 7  # Minimum 1 week for meaningful evaluation
            
            if date_range_days < MIN_DATE_RANGE_DAYS:
                return {
                    "error": f"Date range too short for meaningful evaluation. Got {date_range_days} days, minimum {MIN_DATE_RANGE_DAYS} days required.",
                    "warning": f"Replay with date range shorter than {MIN_DATE_RANGE_DAYS} days may not provide reliable performance metrics"
                }, 400
        except ValueError:
            pass  # Date validation already done above
    
    try:
        print(f"[REPLAY] Starting replay for {request.symbol} with {len(candles)} candles")
        replay_id = replay_engine.start_replay(
            symbol=request.symbol,
            candles=candles,
            replay_id=request.replay_id,
            source=source,
            allowed_entry_regimes=request.allowed_entry_regimes
        )
        
        print(f"[REPLAY] Replay started with replay_id={replay_id}")
        replay_running = True
        
        # MULTI-RUN SAFETY: Ensure replay state is always cleaned up, even on failure
        try:
            # Run replay synchronously (in production, consider background task)
            # Processes candles ONE AT A TIME through full pipeline
            print(f"[REPLAY] Running replay engine...")
            result = replay_engine.run(db)
            print(f"[REPLAY] Replay engine completed: {result['total_candles']} candles processed")
            print(f"[REPLAY] CSV replay complete")
            
            # Fetch trades and equity curve for metrics computation
            trades = db.query(Trade).filter(
                Trade.replay_id == replay_id
            ).order_by(Trade.entry_time).all()
            
            equity_curve = db.query(EquityCurve).filter(
                EquityCurve.replay_id == replay_id
            ).order_by(EquityCurve.timestamp).all()
            
            # Compute metrics for determinism verification and summary
            metrics_snapshot = compute_metrics(trades, equity_curve)
            
            # Calculate net P&L (final_equity - initial_equity)
            initial_equity = replay_engine.initial_equity
            net_pnl = result["final_equity"] - initial_equity
            
            # Generate performance report
            closed_trades = [t for t in trades if t.exit_time is not None]
            stop_loss_trades = [t for t in closed_trades if t.reason and "STOP_LOSS" in t.reason.upper()]
            stop_loss_pct = (len(stop_loss_trades) / len(closed_trades) * 100) if len(closed_trades) > 0 else 0.0
            
            replay_report = {
                "symbol": request.symbol,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "candles_processed": result["total_candles"],
                "trades_executed": len(trades),
                "win_rate": metrics_snapshot.win_rate,
                "net_pnl": net_pnl,
                "max_drawdown_pct": metrics_snapshot.max_drawdown_pct,
                "sharpe_proxy": metrics_snapshot.sharpe_proxy if not (math.isnan(metrics_snapshot.sharpe_proxy) or math.isinf(metrics_snapshot.sharpe_proxy)) else None,
                "stop_loss_pct": round(stop_loss_pct, 2),
                "expectancy": metrics_snapshot.expectancy_per_trade,  # Expected value per trade
                "allowed_entry_regimes": request.allowed_entry_regimes  # Show which regimes were allowed
            }
            
            # Print performance report to console
            print("\n" + "=" * 60)
            print("HISTORICAL REPLAY RESULTS (No Optimization)")
            print("=" * 60)
            print(f"Symbol: {replay_report['symbol']}")
            if replay_report['start_date'] and replay_report['end_date']:
                print(f"Date Range: {replay_report['start_date']} → {replay_report['end_date']}")
            print(f"Candles Processed: {replay_report['candles_processed']:,}")
            print(f"Trades Executed: {replay_report['trades_executed']}")
            print(f"Win Rate: {fmt_pct(replay_report.get('win_rate'))}")
            print(f"Net P&L: {fmt_currency(replay_report.get('net_pnl'))}")
            print(f"Max Drawdown: {fmt_pct(replay_report.get('max_drawdown_pct'))}")
            sharpe_display = fmt(replay_report.get('sharpe_proxy'))
            print(f"Sharpe Proxy: {sharpe_display}")
            print(f"Stop-Loss Exits: {fmt_pct(replay_report.get('stop_loss_pct'))} of trades")
            print("=" * 60 + "\n")
            
            # DETERMINISM VERIFICATION: Check if same inputs produce same outputs
            # Create replay fingerprint: symbol + date range + candle count
            replay_fingerprint = f"{request.symbol}|{request.start_date}|{request.end_date}|{result['total_candles']}"
            
            determinism_status = "no_previous_run"  # Default: no previous run to compare
            determinism_message = None
            determinism_mismatches = None
            
            # Look for previous replays with same symbol and date range
            if request.start_date and request.end_date:
                previous_replays = db.query(ReplaySummary).filter(
                    ReplaySummary.symbol == request.symbol,
                    ReplaySummary.start_date == request.start_date,
                    ReplaySummary.end_date == request.end_date,
                    ReplaySummary.replay_id != replay_id  # Exclude current replay
                ).order_by(ReplaySummary.timestamp_completed.desc()).limit(1).all()
                
                if previous_replays:
                    prev = previous_replays[0]
                    prev_fingerprint = f"{prev.symbol}|{prev.start_date}|{prev.end_date}|{prev.candle_count}"
                    
                    # Verify determinism: same inputs should produce same outputs
                    mismatches = []
                    
                    if prev.candle_count != result["total_candles"]:
                        mismatches.append(f"candle_count: {prev.candle_count} vs {result['total_candles']}")
                    if prev.trade_count != len(trades):
                        mismatches.append(f"trade_count: {prev.trade_count} vs {len(trades)}")
                    if abs(prev.final_equity - result["final_equity"]) > 0.01:  # Allow small floating point differences
                        mismatches.append(f"final_equity: {fmt(prev.final_equity)} vs {fmt(result.get('final_equity'))}")
                    if abs(prev.max_drawdown_pct - metrics_snapshot.max_drawdown_pct) > 0.01:
                        mismatches.append(f"max_drawdown_pct: {fmt_pct(prev.max_drawdown_pct)} vs {fmt_pct(metrics_snapshot.max_drawdown_pct)}")
                    
                    if mismatches:
                        # DETERMINISM VIOLATION: Raise clear error and log both results
                        determinism_status = "mismatch"
                        determinism_message = f"DETERMINISM VIOLATION: Mismatch detected"
                        determinism_mismatches = mismatches
                        
                        # Log both results for inspection
                        print("\n" + "=" * 60)
                        print("[ERROR] DETERMINISM VIOLATION DETECTED")
                        print("=" * 60)
                        print(f"Replay Fingerprint: {replay_fingerprint}")
                        print(f"Previous Replay ID: {prev.replay_id}")
                        print(f"Current Replay ID: {replay_id}")
                        print(f"\nPrevious Run Results:")
                        print(f"  Candle count: {prev.candle_count}")
                        print(f"  Trade count: {prev.trade_count}")
                        print(f"  Final equity: {fmt(prev.final_equity)}")
                        print(f"  Max drawdown: {fmt_pct(prev.max_drawdown_pct)}")
                        print(f"\nCurrent Run Results:")
                        print(f"  Candle count: {result.get('total_candles')}")
                        print(f"  Trade count: {len(trades)}")
                        print(f"  Final equity: {fmt(result.get('final_equity'))}")
                        print(f"  Max drawdown: {fmt_pct(metrics_snapshot.max_drawdown_pct)}")
                        print(f"\nMismatches: {', '.join(mismatches)}")
                        print("=" * 60 + "\n")
                        
                        # Raise error to fail fast
                        raise ValueError(f"DETERMINISM VIOLATION: Same inputs produced different results. Mismatches: {', '.join(mismatches)}")
                    else:
                        # DETERMINISM VERIFIED: Same inputs produced same outputs
                        determinism_status = "verified"
                        determinism_message = "Deterministic replay confirmed"
                        print(f"\n[DETERMINISM VERIFIED] Replay {replay_id} matches previous replay {prev.replay_id}")
                        print(f"Replay Fingerprint: {replay_fingerprint}")
                        print(f"All metrics match: candle_count={result.get('total_candles')}, trade_count={len(trades)}, final_equity={fmt(result.get('final_equity'))}, max_drawdown={fmt_pct(metrics_snapshot.max_drawdown_pct)}\n")
                else:
                    # No previous run to compare
                    determinism_status = "no_previous_run"
                    determinism_message = "No previous run to compare (first run for this input)"
                    print(f"[DETERMINISM] First run for fingerprint: {replay_fingerprint}")
            
            # Log replay fingerprint for tracking
            print(f"[DETERMINISM] Replay fingerprint: {replay_fingerprint}")
            
            # CONSOLE OUTPUT FOR HUMANS: Print concise, copy-pasteable summary
            print("\n" + "=" * 50)
            print("REPLAY COMPLETE")
            print("=" * 50)
            print(f"Symbol: {request.symbol}")
            if request.start_date and request.end_date:
                print(f"Date range: {request.start_date} → {request.end_date}")
            else:
                print(f"Date range: Manual candles")
            print(f"Candles processed: {result['total_candles']:,}")
            print(f"Trades executed: {len(trades)}")
            print(f"Final equity: {fmt_currency(result.get('final_equity'))}")
            print(f"Net P&L: {fmt_currency(net_pnl)}")
            print(f"Max drawdown: {fmt_pct(metrics_snapshot.max_drawdown_pct)}")
            sharpe_display = fmt(metrics_snapshot.sharpe_proxy)
            print(f"Sharpe proxy: {sharpe_display}")
            print(f"Replay ID: {replay_id}")
            print("=" * 50 + "\n")
            
            # Log determinism metrics (for debugging)
            print(f"[DETERMINISM] Replay {replay_id} completed:")
            print(f"  Symbol: {request.symbol}")
            print(f"  Date range: {request.start_date} to {request.end_date}")
            print(f"  Candle count: {result['total_candles']}")
            print(f"  Trade count: {len(trades)}")
            print(f"  Final equity: {fmt(result.get('final_equity'))}")
            print(f"  Max drawdown: {fmt_pct(metrics_snapshot.max_drawdown_pct)}")
            
            # PERSISTENCE GUARANTEE: Log before persistence
            print(f"[REPLAY] Persisting replay summary for replay_id={replay_id}")
            
            # PERSIST REPLAY SUMMARY
            replay_summary = ReplaySummary(
                replay_id=replay_id,
                symbol=request.symbol,
                start_date=request.start_date,
                end_date=request.end_date,
                source=source,
                candle_count=result["total_candles"],
                trade_count=len(trades),
                final_equity=result["final_equity"],
                net_pnl=net_pnl,
                max_drawdown_pct=metrics_snapshot.max_drawdown_pct,
                max_drawdown_absolute=metrics_snapshot.max_drawdown_absolute,
                sharpe_proxy=metrics_snapshot.sharpe_proxy if not (math.isnan(metrics_snapshot.sharpe_proxy) or math.isinf(metrics_snapshot.sharpe_proxy)) else None,
                allowed_entry_regimes=json.dumps(request.allowed_entry_regimes) if request.allowed_entry_regimes else None,
                timestamp_completed=datetime.now(timezone.utc)
            )
            db.add(replay_summary)
            db.commit()
            
            print(f"[REPLAY] Replay summary persisted successfully for replay_id={replay_id}")
            
            replay_running = False
            
            return {
                "status": "completed",
                "replay_id": replay_id,
                "symbol": request.symbol,
                "total_candles": result["total_candles"],
                "final_equity": result["final_equity"],
                "trade_count": len(trades),
                "max_drawdown_pct": metrics_snapshot.max_drawdown_pct,
                "source": source,
                "allowed_entry_regimes": request.allowed_entry_regimes,
                "determinism_status": determinism_status,
                "determinism_message": determinism_message,
                "determinism_mismatches": determinism_mismatches,
                "replay_fingerprint": replay_fingerprint,
                "report": replay_report
            }
        
        except Exception as e:
            # MULTI-RUN SAFETY: Always reset replay_running flag on failure
            error_msg = f"Replay execution failed: {str(e)}"
            print(f"[REPLAY] ERROR: {error_msg}")
            replay_running = False
            # MULTI-RUN SAFETY: Reset replay engine state to prevent corruption
            replay_engine.reset()
            # REPLAY COMPLETION PATH: Always reach SUCCESS or FAILURE state
            raise Exception(error_msg)
    
    except ValueError as e:
        # REPLAY COMPLETION PATH: Explicit error returned (FAILURE state)
        error_msg = str(e)
        print(f"[REPLAY] ERROR (ValueError): {error_msg}")
        replay_running = False
        replay_engine.reset()  # MULTI-RUN SAFETY: Clean up state
        return {"error": error_msg}, 400
    except Exception as e:
        # REPLAY COMPLETION PATH: Explicit error returned (FAILURE state)
        error_msg = f"Replay failed: {str(e)}"
        print(f"[REPLAY] ERROR (Exception): {error_msg}")
        replay_running = False
        replay_engine.reset()  # MULTI-RUN SAFETY: Clean up state
        return {"error": error_msg}, 500


@app.post("/replay/start_csv")
async def start_replay_csv(request: ReplayCsvRequest, db: Session = Depends(get_db)):
    """
    Start a historical replay (backtest) using CSV file data.
    
    This endpoint loads OHLCV candles from a CSV file and runs a replay.
    CSV format must have columns: date, open, high, low, close, volume
    
    Args:
        request: ReplayCsvRequest with symbol and csv_path
    
    Returns:
        Replay results with status, replay_id, trades, metrics, etc.
    """
    global replay_running
    
    # Safeguard: Prevent concurrent replay
    if replay_running:
        return {"error": "Replay already in progress"}, 400
    
    # Safeguard: Validate symbol
    if request.symbol not in ALL_SYMBOLS:
        return {"error": f"Symbol {request.symbol} not in allowed list"}, 400
    
    # Load candles from CSV
    candles = None
    source = "csv"
    
    try:
        print(f"[REPLAY] Starting CSV replay for {request.symbol}")
        print(f"[REPLAY] Loading candles from CSV: {request.csv_path}")
        csv_candles = load_csv_candles(
            csv_path=request.csv_path,
            symbol=request.symbol
        )
        
        # PREFLIGHT VALIDATION: Check if candle count == 0
        if not csv_candles or len(csv_candles) == 0:
            error_msg = f"No valid candles found in CSV file: {request.csv_path}"
            print(f"[REPLAY] ERROR: {error_msg}")
            return {"error": error_msg}, 400
        
        print(f"[REPLAY] Loaded {len(csv_candles)} candles from CSV")
        
        # Convert to replay format (Unix timestamps)
        candles = csv_convert_to_replay_format(csv_candles)
        source = "csv"
        
        # PREFLIGHT VALIDATION: Check candle count after conversion
        if not candles or len(candles) == 0:
            error_msg = f"No valid candles after conversion for {request.symbol}"
            print(f"[REPLAY] ERROR: {error_msg}")
            return {"error": error_msg}, 400
        
        print(f"[REPLAY] Converted to replay format: {len(candles)} candles")
        
    except FileNotFoundError as e:
        error_msg = f"CSV file not found: {request.csv_path}"
        print(f"[REPLAY] ERROR: {error_msg}")
        return {"error": error_msg}, 400
    except ValueError as e:
        error_msg = str(e)
        print(f"[REPLAY] ERROR: {error_msg}")
        return {"error": error_msg}, 400
    except Exception as e:
        error_msg = f"Failed to load CSV file: {str(e)}"
        print(f"[REPLAY] ERROR: {error_msg}")
        return {"error": error_msg}, 500
    
    # GUARDRAILS: Validate candles meet minimum thresholds
    MIN_CANDLES_FOR_REPLAY_DAILY = 50  # Minimum for daily data (about 2.5 months)
    MIN_CANDLES_FOR_INDICATORS = 50  # Minimum for EMA(50)
    
    # CSV data is assumed to be daily
    MIN_CANDLES_FOR_REPLAY = MIN_CANDLES_FOR_REPLAY_DAILY
    
    # PREFLIGHT VALIDATION: Check if candle count == 0 (must be explicit, no silent exit)
    if not candles or len(candles) == 0:
        error_msg = f"No candles provided for replay"
        print(f"[REPLAY] ERROR: {error_msg}")
        return {"error": error_msg}, 400
    
    print(f"[REPLAY] Preflight validation: {len(candles)} candles, minimum required: {MIN_CANDLES_FOR_INDICATORS} for indicators, {MIN_CANDLES_FOR_REPLAY} for replay")
    
    if len(candles) < MIN_CANDLES_FOR_INDICATORS:
        error_msg = f"Need at least {MIN_CANDLES_FOR_INDICATORS} candles for replay (EMA(50) requires {MIN_CANDLES_FOR_INDICATORS} candles). Got {len(candles)}"
        print(f"[REPLAY] ERROR: {error_msg}")
        return {"error": error_msg}, 400
    
    if len(candles) < MIN_CANDLES_FOR_REPLAY:
        error_msg = f"Insufficient candles for meaningful evaluation. Got {len(candles)}, minimum {MIN_CANDLES_FOR_REPLAY} required."
        print(f"[REPLAY] ERROR: {error_msg}")
        return {
            "error": error_msg,
            "warning": f"Replay with fewer than {MIN_CANDLES_FOR_REPLAY} candles may not provide reliable performance metrics"
        }, 400
    
    print(f"[REPLAY] Preflight validation PASSED: {len(candles)} candles ready for replay")
    
    # Extract date range from candles for summary
    first_candle_time = datetime.fromtimestamp(candles[0]["time"], tz=timezone.utc)
    last_candle_time = datetime.fromtimestamp(candles[-1]["time"], tz=timezone.utc)
    start_date = first_candle_time.strftime("%Y-%m-%d")
    end_date = last_candle_time.strftime("%Y-%m-%d")
    
    try:
        print(f"[REPLAY] Starting CSV replay for {request.symbol} with {len(candles)} candles")
        replay_id = replay_engine.start_replay(
            symbol=request.symbol,
            candles=candles,
            replay_id=request.replay_id,
            source=source,
            allowed_entry_regimes=request.allowed_entry_regimes
        )
        
        print(f"[REPLAY] Replay started with replay_id={replay_id}")
        replay_running = True
        
        # MULTI-RUN SAFETY: Ensure replay state is always cleaned up, even on failure
        try:
            # Run replay synchronously (in production, consider background task)
            # Processes candles ONE AT A TIME through full pipeline
            print(f"[REPLAY] Running replay engine...")
            result = replay_engine.run(db)
            print(f"[REPLAY] Replay engine completed: {result['total_candles']} candles processed")
            print(f"[REPLAY] CSV replay complete")
            
            # Fetch trades and equity curve for metrics computation
            trades = db.query(Trade).filter(
                Trade.replay_id == replay_id
            ).order_by(Trade.entry_time).all()
            
            equity_curve = db.query(EquityCurve).filter(
                EquityCurve.replay_id == replay_id
            ).order_by(EquityCurve.timestamp).all()
            
            # Compute metrics for determinism verification and summary
            metrics_snapshot = compute_metrics(trades, equity_curve)
            
            # Calculate net P&L (final_equity - initial_equity)
            initial_equity = replay_engine.initial_equity
            net_pnl = result["final_equity"] - initial_equity
            
            # Generate performance report
            closed_trades = [t for t in trades if t.exit_time is not None]
            stop_loss_trades = [t for t in closed_trades if t.reason and "STOP_LOSS" in t.reason.upper()]
            stop_loss_pct = (len(stop_loss_trades) / len(closed_trades) * 100) if len(closed_trades) > 0 else 0.0
            
            replay_report = {
                "symbol": request.symbol,
                "csv_path": request.csv_path,
                "start_date": start_date,
                "end_date": end_date,
                "candles_processed": result["total_candles"],
                "trades_executed": len(trades),
                "win_rate": metrics_snapshot.win_rate,
                "net_pnl": net_pnl,
                "max_drawdown_pct": metrics_snapshot.max_drawdown_pct,
                "sharpe_proxy": metrics_snapshot.sharpe_proxy if not (math.isnan(metrics_snapshot.sharpe_proxy) or math.isinf(metrics_snapshot.sharpe_proxy)) else None,
                "stop_loss_pct": round(stop_loss_pct, 2),
                "expectancy": metrics_snapshot.expectancy_per_trade,  # Expected value per trade
                "allowed_entry_regimes": request.allowed_entry_regimes  # Show which regimes were allowed
            }
            
            # Print performance report to console
            print("\n" + "=" * 60)
            print("HISTORICAL REPLAY RESULTS (CSV Data)")
            print("=" * 60)
            print(f"Symbol: {replay_report['symbol']}")
            print(f"CSV File: {replay_report['csv_path']}")
            print(f"Date Range: {replay_report['start_date']} → {replay_report['end_date']}")
            print(f"Candles Processed: {replay_report['candles_processed']:,}")
            print(f"Trades Executed: {replay_report['trades_executed']}")
            print(f"Win Rate: {fmt_pct(replay_report.get('win_rate'))}")
            print(f"Net P&L: {fmt_currency(replay_report.get('net_pnl'))}")
            print(f"Max Drawdown: {fmt_pct(replay_report.get('max_drawdown_pct'))}")
            sharpe_display = fmt(replay_report.get('sharpe_proxy'))
            print(f"Sharpe Proxy: {sharpe_display}")
            print(f"Stop-Loss Exits: {fmt_pct(replay_report.get('stop_loss_pct'))} of trades")
            print("=" * 60 + "\n")
            
            # PERSISTENCE GUARANTEE: Log before persistence
            print(f"[REPLAY] Persisting replay summary for replay_id={replay_id}")
            
            # PERSIST REPLAY SUMMARY
            replay_summary = ReplaySummary(
                replay_id=replay_id,
                symbol=request.symbol,
                start_date=start_date,
                end_date=end_date,
                source=source,
                candle_count=result["total_candles"],
                trade_count=len(trades),
                final_equity=result["final_equity"],
                net_pnl=net_pnl,
                max_drawdown_pct=metrics_snapshot.max_drawdown_pct,
                max_drawdown_absolute=metrics_snapshot.max_drawdown_absolute,
                sharpe_proxy=metrics_snapshot.sharpe_proxy if not (math.isnan(metrics_snapshot.sharpe_proxy) or math.isinf(metrics_snapshot.sharpe_proxy)) else None,
                allowed_entry_regimes=json.dumps(request.allowed_entry_regimes) if request.allowed_entry_regimes else None,
                timestamp_completed=datetime.now(timezone.utc)
            )
            db.add(replay_summary)
            db.commit()
            
            print(f"[REPLAY] Replay summary persisted successfully for replay_id={replay_id}")
            
            replay_running = False
            
            return {
                "status": "completed",
                "replay_id": replay_id,
                "symbol": request.symbol,
                "csv_path": request.csv_path,
                "total_candles": result["total_candles"],
                "final_equity": result["final_equity"],
                "trade_count": len(trades),
                "max_drawdown_pct": metrics_snapshot.max_drawdown_pct,
                "source": source,
                "allowed_entry_regimes": request.allowed_entry_regimes,
                "report": replay_report
            }
        
        except Exception as e:
            # MULTI-RUN SAFETY: Always reset replay_running flag on failure
            error_msg = f"Replay execution failed: {str(e)}"
            print(f"[REPLAY] ERROR: {error_msg}")
            replay_running = False
            # MULTI-RUN SAFETY: Reset replay engine state to prevent corruption
            replay_engine.reset()
            # REPLAY COMPLETION PATH: Always reach SUCCESS or FAILURE state
            raise Exception(error_msg)
    
    except ValueError as e:
        # REPLAY COMPLETION PATH: Explicit error returned (FAILURE state)
        error_msg = str(e)
        print(f"[REPLAY] ERROR (ValueError): {error_msg}")
        replay_running = False
        replay_engine.reset()  # MULTI-RUN SAFETY: Clean up state
        return {"error": error_msg}, 400
    except Exception as e:
        # REPLAY COMPLETION PATH: Explicit error returned (FAILURE state)
        error_msg = f"Replay failed: {str(e)}"
        print(f"[REPLAY] ERROR (Exception): {error_msg}")
        replay_running = False
        replay_engine.reset()  # MULTI-RUN SAFETY: Clean up state
        return {"error": error_msg}, 500


@app.post("/replay/start_intraday_csv")
async def start_intraday_replay_csv(request: IntradayReplayCsvRequest, db: Session = Depends(get_db)):
    """
    Start an intraday replay (5-minute candles) with daily regime context.
    
    This endpoint automatically locates and loads:
    - Daily candles from data d/daily/us/{exchange}/{asset_class}/ (for regime classification)
    - 5-minute candles from data 5/5 min/us/{exchange}/{asset_class}/ (for execution)
    
    The system searches the directory structure to find files matching the symbol.
    
    Currently logs only - no trades executed yet (scaffold for future implementation).
    
    Example payload:
    {
      "symbol": "SPY",
      "interval": "5min"
    }
    
    Args:
        request: IntradayReplayCsvRequest with symbol and interval
    
    Returns:
        Replay results with status, replay_id, candles processed, etc.
    """
    global replay_running
    
    # Safeguard: Prevent concurrent replay
    if replay_running:
        return {"error": "Replay already in progress"}, 400
    
    # Determine symbols list (support both single symbol and symbols list)
    if request.symbols:
        symbols = request.symbols
    elif request.symbol:
        symbols = [request.symbol]
    else:
        return {"error": "Must provide either 'symbol' or 'symbols'"}, 400
    
    # Safeguard: Validate all symbols
    invalid_symbols = [s for s in symbols if s not in ALL_SYMBOLS]
    if invalid_symbols:
        return {"error": f"Invalid symbol(s): {invalid_symbols}. Must be in allowed list"}, 400
    
    # Validate interval
    if request.interval != "5min":
        return {"error": f"Only '5min' interval is currently supported. Got: {request.interval}"}, 400
    
    # Validate exit_mode
    valid_exit_modes = ["momentum_reversal", "time_based", "mfe_trailing"]
    if request.exit_mode not in valid_exit_modes:
        return {"error": f"Invalid exit_mode: {request.exit_mode}. Must be one of: {valid_exit_modes}"}, 400
    
    # Validate allowed_sessions
    valid_sessions = ["market_open", "midday", "power_hour"]
    if request.allowed_sessions:
        invalid_sessions = [s for s in request.allowed_sessions if s not in valid_sessions]
        if invalid_sessions:
            return {"error": f"Invalid session(s): {invalid_sessions}. Must be one of: {valid_sessions}"}, 400
    
    # Get risk parameters from request or use defaults
    starting_equity = request.starting_equity if request.starting_equity is not None else 100000.0
    risk_per_trade_pct = request.risk_per_trade_pct if request.risk_per_trade_pct is not None else 0.25
    max_daily_loss_pct = request.max_daily_loss_pct if request.max_daily_loss_pct is not None else 1.0
    max_concurrent_positions = request.max_concurrent_positions if request.max_concurrent_positions is not None else 1
    
    # Multi-symbol replay
    if len(symbols) > 1:
        return await _run_multi_symbol_intraday_replay(
            symbols=symbols,
            interval=request.interval,
            exit_mode=request.exit_mode,
            exit_params=request.exit_params,
            allowed_sessions=request.allowed_sessions,
            starting_equity=starting_equity,
            risk_per_trade_pct=risk_per_trade_pct,
            max_daily_loss_pct=max_daily_loss_pct,
            max_concurrent_positions=max_concurrent_positions,
            db=db
        )
    
    # Single symbol replay (existing logic)
    symbol = symbols[0]
    
    # Load daily candles (for regime classification) - auto-discover from directory structure
    daily_candles = None
    try:
        print(f"[INTRADAY REPLAY] Searching for daily candles for symbol: {symbol}")
        daily_candles_raw = load_daily_candles(symbol)
        daily_candles_all = csv_convert_to_replay_format(daily_candles_raw)
        
        # Apply date filtering if provided
        if request.start_date and request.end_date:
            try:
                daily_candles = filter_candles_by_date_range(
                    daily_candles_all,
                    request.start_date,
                    request.end_date,
                    candle_type="daily"
                )
                print(f"[INTRADAY REPLAY] Filtered to {len(daily_candles)} daily candles ({request.start_date} to {request.end_date})")
            except ValueError as e:
                error_msg = f"Date filtering failed for daily candles: {str(e)}"
                print(f"[INTRADAY REPLAY] ERROR: {error_msg}")
                return {"error": error_msg}, 400
        else:
            daily_candles = daily_candles_all
        
        print(f"[INTRADAY REPLAY] Loaded {len(daily_candles)} daily candles")
    except FileNotFoundError as e:
        error_msg = f"Daily data file not found: {str(e)}"
        print(f"[INTRADAY REPLAY] ERROR: {error_msg}")
        return {"error": error_msg}, 404
    except Exception as e:
        error_msg = f"Failed to load daily candles: {str(e)}"
        print(f"[INTRADAY REPLAY] ERROR: {error_msg}")
        return {"error": error_msg}, 400
    
    # Load intraday candles (5-minute) - auto-discover from directory structure
    intraday_candles = None
    try:
        print(f"[INTRADAY REPLAY] Searching for intraday candles for symbol: {symbol} (interval={request.interval})")
        intraday_candles_raw = load_intraday_candles(symbol, interval=request.interval)
        intraday_candles_all = csv_convert_to_replay_format(intraday_candles_raw)
        
        # Apply date filtering if provided
        if request.start_date and request.end_date:
            try:
                intraday_candles = filter_candles_by_date_range(
                    intraday_candles_all,
                    request.start_date,
                    request.end_date,
                    candle_type="intraday"
                )
                print(f"[INTRADAY REPLAY] Filtered to {len(intraday_candles)} intraday candles ({request.start_date} to {request.end_date})")
            except ValueError as e:
                error_msg = f"Date filtering failed for intraday candles: {str(e)}"
                print(f"[INTRADAY REPLAY] ERROR: {error_msg}")
                return {"error": error_msg}, 400
        else:
            intraday_candles = intraday_candles_all
        
        print(f"[INTRADAY REPLAY] Loaded {len(intraday_candles)} intraday candles")
    except FileNotFoundError as e:
        error_msg = f"Intraday data file not found: {str(e)}"
        print(f"[INTRADAY REPLAY] ERROR: {error_msg}")
        return {"error": error_msg}, 404
    except Exception as e:
        error_msg = f"Failed to load intraday candles: {str(e)}"
        print(f"[INTRADAY REPLAY] ERROR: {error_msg}")
        return {"error": error_msg}, 400
    
    # Validate minimum requirements
    if len(daily_candles) < 200:
        return {"error": f"Need at least 200 daily candles for regime classification (EMA(200)). Got {len(daily_candles)}"}, 400
    
    if len(intraday_candles) < 10:
        return {"error": f"Need at least 10 intraday candles. Got {len(intraday_candles)}"}, 400
    
    try:
        print(f"[INTRADAY REPLAY] Starting intraday replay for {symbol}")
        print(f"[INTRADAY REPLAY]   Exit mode: {request.exit_mode}")
        if request.exit_params:
            print(f"[INTRADAY REPLAY]   Exit params: {request.exit_params}")
        if request.allowed_sessions:
            print(f"[INTRADAY REPLAY]   Allowed sessions: {request.allowed_sessions}")
        
        # Update engine's risk parameters
        intraday_replay_engine.initial_equity = starting_equity
        intraday_replay_engine.risk_per_trade_pct = risk_per_trade_pct
        intraday_replay_engine.max_daily_loss_pct = max_daily_loss_pct
        intraday_replay_engine.max_concurrent_positions = max_concurrent_positions
        
        replay_id = intraday_replay_engine.start_intraday_replay(
            symbol=symbol,
            intraday_candles=intraday_candles,
            daily_candles=daily_candles,
            replay_id=request.replay_id,
            source="csv_intraday",
            exit_mode=request.exit_mode,
            exit_params=request.exit_params,
            allowed_sessions=request.allowed_sessions,
            entry_variant=request.entry_variant,
            diagnostic_mode=request.diagnostic_mode if request.diagnostic_mode is not None else False,
            bypass_daily_regime_gate=request.bypass_daily_regime_gate if request.bypass_daily_regime_gate is not None else False,
            bypass_session_gate=request.bypass_session_gate if request.bypass_session_gate is not None else False,
            risk_per_trade_pct=risk_per_trade_pct,
            max_daily_loss_pct=max_daily_loss_pct,
            max_concurrent_positions=max_concurrent_positions
        )
        
        print(f"[INTRADAY REPLAY] Replay started with replay_id={replay_id}")
        replay_running = True
        
        try:
            # Run intraday replay (executes trades)
            print(f"[INTRADAY REPLAY] Running intraday replay engine...")
            result = intraday_replay_engine.run(db)
            print(f"[INTRADAY REPLAY] Intraday replay complete")
            
            # Fetch executed trades from database
            trades = db.query(Trade).filter(
                Trade.replay_id == replay_id
            ).order_by(Trade.entry_time).all()
            
            # Compute intraday trade diagnostics
            intraday_aggregate_metrics = compute_intraday_aggregate_metrics(trades, intraday_candles)
            
            # Compute frequency and session metrics
            frequency_session_metrics = compute_frequency_and_session_metrics(trades)
            
            # Compute per-trade diagnostics
            trades_with_diagnostics = []
            for trade in trades:
                trade_dict = {
                    "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
                    "entry_price": trade.entry_price,
                    "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                    "exit_price": trade.exit_price,
                    "shares": trade.shares,
                    "pnl": trade.pnl,
                    "reason": trade.reason
                }
                
                # Add intraday diagnostics
                diagnostics = compute_intraday_trade_diagnostics(trade, intraday_candles)
                trade_dict["diagnostics"] = diagnostics
                
                trades_with_diagnostics.append(trade_dict)
            
            replay_running = False
            
            # Determine most active session
            session_metrics = frequency_session_metrics["session_metrics"]
            most_active_session = max(
                session_metrics.items(),
                key=lambda x: x[1]["trade_count"]
            )[0] if session_metrics else None
            
            # Get risk manager metrics (before logging, so variables are defined)
            risk_manager = intraday_replay_engine.risk_manager
            final_equity = result.get("final_equity", starting_equity)
            max_drawdown_pct = result.get("max_drawdown_pct", 0.0)
            max_drawdown_absolute = result.get("max_drawdown_absolute", 0.0)
            
            # Calculate return on capital
            net_pnl = final_equity - starting_equity
            return_on_capital_pct = (net_pnl / starting_equity * 100) if starting_equity > 0 else 0.0
            
            # Log summary
            print("\n" + "=" * 60)
            print("INTRADAY REPLAY SUMMARY")
            print("=" * 60)
            print(f"Symbol: {symbol}")
            print(f"Exit Mode: {request.exit_mode}")
            if request.exit_params:
                print(f"Exit Params: {request.exit_params}")
            if request.allowed_sessions:
                print(f"Allowed Sessions: {request.allowed_sessions}")
            else:
                print(f"Allowed Sessions: All (no session gate)")
            print(f"Total Trades: {intraday_aggregate_metrics['total_trades']}")
            print(f"Win Rate: {fmt_pct(intraday_aggregate_metrics.get('win_rate'))}")
            print(f"Expectancy: {fmt_currency(intraday_aggregate_metrics.get('expectancy'))}")
            print(f"Average Holding Time: {fmt(intraday_aggregate_metrics.get('average_holding_time_minutes'), 1)} minutes")
            print(f"Average MFE: {fmt_currency(intraday_aggregate_metrics.get('average_mfe'))}")
            print(f"Average MAE: {fmt_currency(intraday_aggregate_metrics.get('average_mae'))}")
            if intraday_aggregate_metrics.get('average_mfe_given_up_pct') is not None:
                print(f"Average MFE Given Up: {fmt_pct(intraday_aggregate_metrics.get('average_mfe_given_up_pct'))}")
            print()
            print("RISK METRICS:")
            print(f"  Starting Equity: {fmt_currency(starting_equity)}")
            print(f"  Final Equity: {fmt_currency(final_equity)}")
            print(f"  Net P&L: {fmt_currency(net_pnl)}")
            print(f"  Return on Capital: {fmt_pct(return_on_capital_pct)}")
            print(f"  Max Portfolio Drawdown: {fmt_pct(max_drawdown_pct)}")
            print(f"  Risk per Trade: {fmt_pct(risk_per_trade_pct)}")
            print(f"  Max Daily Loss Limit: {fmt_pct(max_daily_loss_pct)}")
            print()
            print("FREQUENCY METRICS:")
            freq_metrics = frequency_session_metrics.get("frequency_metrics", {})
            print(f"  Average Trades/Day: {fmt(freq_metrics.get('average_trades_per_day'))}")
            print(f"  Max Trades in a Day: {freq_metrics.get('max_trades_in_a_day')}")
            print(f"  % of Days with Trades: {fmt_pct(freq_metrics.get('percentage_of_days_with_trades'))}")
            print(f"  Days with Multiple Trades: {frequency_session_metrics['clustering_metrics']['days_with_multiple_trades']}")
            print()
            print("SESSION METRICS:")
            for session_name, metrics in session_metrics.items():
                print(f"  {session_name.replace('_', ' ').title()}: {metrics.get('trade_count')} trades, "
                      f"win_rate={fmt_pct(metrics.get('win_rate'))}, expectancy={fmt_currency(metrics.get('expectancy'))}")
            if most_active_session:
                print(f"  Most Active Session: {most_active_session.replace('_', ' ').title()}")
            print("=" * 60 + "\n")
            
            # Compute metrics from equity curve
            equity_curve = db.query(EquityCurve).filter(
                EquityCurve.replay_id == replay_id
            ).order_by(EquityCurve.timestamp).all()
            metrics_snapshot = compute_metrics(trades, equity_curve) if equity_curve else None
            
            # Note: final_equity, max_drawdown_pct, net_pnl, return_on_capital_pct already assigned above
            
            # Compute daily P&L distribution (if we have trade data)
            daily_pnl_distribution = {}
            for trade in trades:
                if trade.exit_time:
                    trade_date = trade.exit_time.date()
                    if trade_date not in daily_pnl_distribution:
                        daily_pnl_distribution[trade_date] = 0.0
                    daily_pnl_distribution[trade_date] += trade.pnl or 0.0
            
            # Risk diagnostics
            risk_diagnostics = {
                "starting_equity": starting_equity,
                "final_equity": final_equity,
                "net_pnl": net_pnl,
                "return_on_capital_pct": round(return_on_capital_pct, 2),
                "max_portfolio_drawdown_pct": round(max_drawdown_pct, 2),
                "max_portfolio_drawdown_absolute": round(max_drawdown_absolute, 2),
                "risk_per_trade_pct": risk_per_trade_pct,
                "max_daily_loss_pct": max_daily_loss_pct,
                "max_concurrent_positions": max_concurrent_positions,
                "daily_pnl_distribution": {
                    str(date): round(pnl, 2) for date, pnl in daily_pnl_distribution.items()
                }
            }
            
            return {
                "status": "completed",
                "replay_id": replay_id,
                "symbol": symbol,
                "replay_mode": "intraday",
                "exit_mode": request.exit_mode,
                "exit_params": request.exit_params,
                "allowed_sessions": request.allowed_sessions,
                "total_intraday_candles": result["total_intraday_candles"],
                "total_daily_candles": result["total_daily_candles"],
                "candles_processed": result["candles_processed"],
                "trades_executed": result["trades_executed"],
                "intraday_trade_metrics": intraday_aggregate_metrics,
                "frequency_metrics": frequency_session_metrics["frequency_metrics"],
                "session_metrics": frequency_session_metrics["session_metrics"],
                "clustering_metrics": frequency_session_metrics["clustering_metrics"],
                "max_drawdown_pct": max_drawdown_pct,
                "risk_diagnostics": risk_diagnostics,
                "trades": trades_with_diagnostics,
                "note": f"Intraday execution with risk-based sizing - exit_mode={request.exit_mode}, risk_per_trade={risk_per_trade_pct}%, max_daily_loss={max_daily_loss_pct}%, gated by daily TREND_UP regime"
            }
        
        except Exception as e:
            error_msg = f"Intraday replay execution failed: {str(e)}"
            print(f"[INTRADAY REPLAY] ERROR: {error_msg}")
            replay_running = False
            intraday_replay_engine.reset()
            raise Exception(error_msg)
    
    except ValueError as e:
        error_msg = str(e)
        print(f"[INTRADAY REPLAY] ERROR (ValueError): {error_msg}")
        replay_running = False
        intraday_replay_engine.reset()
        return {"error": error_msg}, 400
    except Exception as e:
        error_msg = f"Intraday replay failed: {str(e)}"
        print(f"[INTRADAY REPLAY] ERROR (Exception): {error_msg}")
        replay_running = False
        intraday_replay_engine.reset()
        return {"error": error_msg}, 500


async def _run_multi_symbol_intraday_replay(
    symbols: List[str],
    interval: str,
    exit_mode: str,
    exit_params: Optional[Dict],
    allowed_sessions: Optional[List[str]],
    starting_equity: Optional[float],
    risk_per_trade_pct: Optional[float],
    max_daily_loss_pct: Optional[float],
    max_concurrent_positions: Optional[int],
    db: Session
):
    """
    Run intraday replay for multiple symbols with identical logic.
    
    Returns per-symbol results and aggregate portfolio summary.
    """
    global replay_running
    
    print(f"\n" + "=" * 60)
    print(f"MULTI-SYMBOL INTRADAY REPLAY")
    print("=" * 60)
    print(f"Symbols: {symbols}")
    print(f"Exit Mode: {exit_mode}")
    if exit_params:
        print(f"Exit Params: {exit_params}")
    if allowed_sessions:
        print(f"Allowed Sessions: {allowed_sessions}")
    print("=" * 60 + "\n")
    
    per_symbol_results = []
    errors = []
    
    # Run replay for each symbol
    for symbol in symbols:
        print(f"\n[SYMBOL: {symbol}] Starting replay...")
        
        try:
            # Load daily candles
            daily_candles_raw = load_daily_candles(symbol)
            daily_candles = csv_convert_to_replay_format(daily_candles_raw)
            
            if len(daily_candles) < 200:
                errors.append({"symbol": symbol, "error": f"Insufficient daily candles: {len(daily_candles)}"})
                continue
            
            # Load intraday candles
            intraday_candles_raw = load_intraday_candles(symbol, interval=interval)
            intraday_candles = csv_convert_to_replay_format(intraday_candles_raw)
            
            if len(intraday_candles) < 10:
                errors.append({"symbol": symbol, "error": f"Insufficient intraday candles: {len(intraday_candles)}"})
                continue
            
            # Use provided risk parameters or defaults
            equity = starting_equity if starting_equity is not None else 100000.0
            risk_pct = risk_per_trade_pct if risk_per_trade_pct is not None else 0.25
            max_loss_pct = max_daily_loss_pct if max_daily_loss_pct is not None else 1.0
            max_positions = max_concurrent_positions if max_concurrent_positions is not None else 1
            
            # Update engine's risk parameters
            intraday_replay_engine.initial_equity = equity
            intraday_replay_engine.risk_per_trade_pct = risk_pct
            intraday_replay_engine.max_daily_loss_pct = max_loss_pct
            intraday_replay_engine.max_concurrent_positions = max_positions
            
            # Run replay
            replay_id = intraday_replay_engine.start_intraday_replay(
                symbol=symbol,
                intraday_candles=intraday_candles,
                daily_candles=daily_candles,
                replay_id=None,  # Generate new ID for each symbol
                source="csv_intraday",
                exit_mode=exit_mode,
                exit_params=exit_params,
                allowed_sessions=allowed_sessions,
                risk_per_trade_pct=risk_pct,
                max_daily_loss_pct=max_loss_pct,
                max_concurrent_positions=max_positions
            )
            
            result = intraday_replay_engine.run(db)
            
            # Fetch trades and compute metrics
            trades = db.query(Trade).filter(
                Trade.replay_id == replay_id
            ).order_by(Trade.entry_time).all()
            
            equity_curve = db.query(EquityCurve).filter(
                EquityCurve.replay_id == replay_id
            ).order_by(EquityCurve.timestamp).all()
            
            # Compute metrics
            metrics_snapshot = compute_metrics(trades, equity_curve) if equity_curve else None
            intraday_aggregate_metrics = compute_intraday_aggregate_metrics(trades, intraday_candles)
            frequency_session_metrics = compute_frequency_and_session_metrics(trades)
            
            # Store per-symbol result
            per_symbol_results.append({
                "symbol": symbol,
                "replay_id": replay_id,
                "trades": len(trades),
                "expectancy": intraday_aggregate_metrics["expectancy"],
                "max_drawdown_pct": metrics_snapshot.max_drawdown_pct if metrics_snapshot else None,
                "average_trades_per_day": frequency_session_metrics["frequency_metrics"]["average_trades_per_day"],
                "win_rate": intraday_aggregate_metrics["win_rate"],
                "total_trades": intraday_aggregate_metrics["total_trades"]
            })
            
            print(f"[SYMBOL: {symbol}] Completed: {len(trades)} trades, expectancy={fmt_currency(intraday_aggregate_metrics.get('expectancy'))}")
            
            # Reset engine for next symbol
            intraday_replay_engine.reset()
            
        except FileNotFoundError as e:
            errors.append({"symbol": symbol, "error": f"Data file not found: {str(e)}"})
            print(f"[SYMBOL: {symbol}] ERROR: Data file not found")
            intraday_replay_engine.reset()
            continue
        except Exception as e:
            errors.append({"symbol": symbol, "error": f"Replay failed: {str(e)}"})
            print(f"[SYMBOL: {symbol}] ERROR: {str(e)}")
            intraday_replay_engine.reset()
            continue
    
    replay_running = False
    
    # Compute portfolio summary
    if not per_symbol_results:
        return {
            "status": "completed",
            "symbols": symbols,
            "per_symbol_results": [],
            "portfolio_summary": {
                "average_expectancy": 0.0,
                "symbols_with_positive_expectancy_pct": 0.0,
                "worst_symbol_drawdown": None,
                "best_symbol_expectancy": None
            },
            "errors": errors,
            "note": "Multi-symbol replay completed with errors"
        }
    
    # Aggregate metrics
    expectancies = [r["expectancy"] for r in per_symbol_results if r["expectancy"] is not None]
    drawdowns = [r["max_drawdown_pct"] for r in per_symbol_results if r["max_drawdown_pct"] is not None]
    
    average_expectancy = statistics.mean(expectancies) if expectancies else 0.0
    symbols_with_positive_expectancy = len([e for e in expectancies if e > 0])
    symbols_with_positive_expectancy_pct = (symbols_with_positive_expectancy / len(per_symbol_results) * 100) if per_symbol_results else 0.0
    
    worst_symbol_drawdown = max(drawdowns) if drawdowns else None
    best_symbol_expectancy = max(expectancies) if expectancies else None
    
    # Log portfolio summary
    print("\n" + "=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    print(f"Symbols Tested: {len(per_symbol_results)}")
    print(f"Average Expectancy: {fmt_currency(average_expectancy)}")
    print(f"Symbols with Positive Expectancy: {symbols_with_positive_expectancy}/{len(per_symbol_results)} ({fmt_pct(symbols_with_positive_expectancy_pct)})")
    if worst_symbol_drawdown is not None:
        print(f"Worst Symbol Drawdown: {fmt_pct(worst_symbol_drawdown)}")
    if best_symbol_expectancy is not None:
        print(f"Best Symbol Expectancy: {fmt_currency(best_symbol_expectancy)}")
    print("=" * 60 + "\n")
    
    return {
        "status": "completed",
        "symbols": symbols,
        "per_symbol_results": per_symbol_results,
        "portfolio_summary": {
            "average_expectancy": round(average_expectancy, 2),
            "symbols_with_positive_expectancy_pct": round(symbols_with_positive_expectancy_pct, 2),
            "worst_symbol_drawdown": round(worst_symbol_drawdown, 2) if worst_symbol_drawdown is not None else None,
            "best_symbol_expectancy": round(best_symbol_expectancy, 2) if best_symbol_expectancy is not None else None
        },
        "errors": errors,
        "note": f"Multi-symbol replay - exit_mode={exit_mode}, allowed_sessions={allowed_sessions or 'All'}, identical logic applied to all symbols"
    }


@app.post("/replay/walkforward_intraday")
async def walkforward_intraday_replay(request: WalkForwardRequest, db: Session = Depends(get_db)):
    """
    Run walk-forward evaluation across multiple time windows.
    
    This endpoint prevents overfitting by evaluating strategy performance on separate
    train/test periods. Each window runs with identical configuration but resets equity
    to starting_equity for clean comparisons.
    
    Example payload:
    {
      "symbols": ["AAAU"],
      "interval": "5min",
      "exit_mode": "mfe_trailing",
      "exit_params": {"mfe_retrace_pct": 50},
      "allowed_sessions": ["market_open", "power_hour"],
      "starting_equity": 100000,
      "risk_per_trade_pct": 0.25,
      "max_daily_loss_pct": 1.0,
      "max_concurrent_positions": 1,
      "windows": [
        {"label": "train_1", "start_date": "2024-01-01", "end_date": "2024-06-30"},
        {"label": "test_1", "start_date": "2024-07-01", "end_date": "2024-12-31"}
      ]
    }
    
    Args:
        request: WalkForwardRequest with symbols, windows, and configuration
    
    Returns:
        Dict with window-level results and aggregate comparison
    """
    global replay_running
    
    # Safeguard: Prevent concurrent replay
    if replay_running:
        return {"error": "Replay already in progress"}, 400
    
    # Validate symbols
    invalid_symbols = [s for s in request.symbols if s not in ALL_SYMBOLS]
    if invalid_symbols:
        return {"error": f"Invalid symbol(s): {invalid_symbols}. Must be in allowed list"}, 400
    
    # Validate interval
    if request.interval != "5min":
        return {"error": f"Only '5min' interval is currently supported. Got: {request.interval}"}, 400
    
    # Validate exit_mode
    valid_exit_modes = ["momentum_reversal", "time_based", "mfe_trailing"]
    if request.exit_mode not in valid_exit_modes:
        return {"error": f"Invalid exit_mode: {request.exit_mode}. Must be one of: {valid_exit_modes}"}, 400
    
    # Validate allowed_sessions
    valid_sessions = ["market_open", "midday", "power_hour"]
    if request.allowed_sessions:
        invalid_sessions = [s for s in request.allowed_sessions if s not in valid_sessions]
        if invalid_sessions:
            return {"error": f"Invalid session(s): {invalid_sessions}. Must be one of: {valid_sessions}"}, 400
    
    # Validate windows
    if not request.windows:
        return {"error": "Must provide at least one window"}, 400
    
    print(f"\n" + "=" * 60)
    print(f"WALK-FORWARD EVALUATION")
    print("=" * 60)
    print(f"Symbols: {request.symbols}")
    print(f"Windows: {len(request.windows)}")
    for window in request.windows:
        print(f"  - {window.label}: {window.start_date} to {window.end_date}")
    print(f"Exit Mode: {request.exit_mode}")
    print(f"Starting Equity: {fmt_currency(request.starting_equity)}")
    print(f"Risk per Trade: {fmt_pct(request.risk_per_trade_pct)}")
    print("=" * 60 + "\n")
    
    window_results = []
    errors = []
    
    # Process each window
    for window in request.windows:
        print(f"\n[WINDOW: {window.label}] Processing {window.start_date} to {window.end_date}...")
        
        window_result = {
            "label": window.label,
            "start_date": window.start_date,
            "end_date": window.end_date,
            "symbols": request.symbols,
            "metrics": {}
        }
        
        # For multi-symbol, aggregate results across symbols
        per_symbol_window_results = []
        
        for symbol in request.symbols:
            try:
                # Load all daily candles (for regime classification)
                daily_candles_raw = load_daily_candles(symbol)
                daily_candles_all = csv_convert_to_replay_format(daily_candles_raw)
                
                # Filter daily candles by window date range
                try:
                    daily_candles = filter_candles_by_date_range(
                        daily_candles_all,
                        window.start_date,
                        window.end_date,
                        candle_type="daily"
                    )
                except ValueError as e:
                    errors.append({
                        "window": window.label,
                        "symbol": symbol,
                        "error": str(e)
                    })
                    print(f"[WINDOW: {window.label}] [SYMBOL: {symbol}] ERROR: {str(e)}")
                    continue
                
                if len(daily_candles) < 200:
                    errors.append({
                        "window": window.label,
                        "symbol": symbol,
                        "error": f"Insufficient daily candles after filtering: {len(daily_candles)}"
                    })
                    continue
                
                # Load all intraday candles
                intraday_candles_raw = load_intraday_candles(symbol, interval=request.interval)
                intraday_candles_all = csv_convert_to_replay_format(intraday_candles_raw)
                
                # Filter intraday candles by window date range
                try:
                    intraday_candles = filter_candles_by_date_range(
                        intraday_candles_all,
                        window.start_date,
                        window.end_date,
                        candle_type="intraday"
                    )
                except ValueError as e:
                    errors.append({
                        "window": window.label,
                        "symbol": symbol,
                        "error": str(e)
                    })
                    print(f"[WINDOW: {window.label}] [SYMBOL: {symbol}] ERROR: {str(e)}")
                    continue
                
                if len(intraday_candles) < 10:
                    errors.append({
                        "window": window.label,
                        "symbol": symbol,
                        "error": f"Insufficient intraday candles after filtering: {len(intraday_candles)}"
                    })
                    continue
                
                # Run replay for this window
                # Reset equity to starting_equity for each window
                intraday_replay_engine.initial_equity = request.starting_equity
                intraday_replay_engine.risk_per_trade_pct = request.risk_per_trade_pct
                intraday_replay_engine.max_daily_loss_pct = request.max_daily_loss_pct
                intraday_replay_engine.max_concurrent_positions = request.max_concurrent_positions
                
                replay_id = intraday_replay_engine.start_intraday_replay(
                    symbol=symbol,
                    intraday_candles=intraday_candles,
                    daily_candles=daily_candles,
                    replay_id=None,  # Generate new ID for each window
                    source="csv_intraday_walkforward",
                    exit_mode=request.exit_mode,
                    exit_params=request.exit_params,
                    allowed_sessions=request.allowed_sessions,
                    risk_per_trade_pct=request.risk_per_trade_pct,
                    max_daily_loss_pct=request.max_daily_loss_pct,
                    max_concurrent_positions=request.max_concurrent_positions
                )
                
                result = intraday_replay_engine.run(db)
                
                # Fetch trades and equity curve
                trades = db.query(Trade).filter(
                    Trade.replay_id == replay_id
                ).order_by(Trade.entry_time).all()
                
                equity_curve = db.query(EquityCurve).filter(
                    EquityCurve.replay_id == replay_id
                ).order_by(EquityCurve.timestamp).all()
                
                # Compute window metrics
                window_metrics = compute_window_metrics(
                    trades=trades,
                    equity_curve=equity_curve,
                    starting_equity=request.starting_equity,
                    intraday_candles=intraday_candles
                )
                
                per_symbol_window_results.append({
                    "symbol": symbol,
                    "replay_id": replay_id,
                    **window_metrics
                })
                
                print(f"[WINDOW: {window.label}] [SYMBOL: {symbol}] Completed: "
                      f"{window_metrics.get('trades_executed')} trades, "
                      f"return={fmt_pct(window_metrics.get('return_on_capital_pct'))}, "
                      f"drawdown={fmt_pct(window_metrics.get('max_portfolio_drawdown_pct'))}")
                
                # Reset engine for next symbol/window
                intraday_replay_engine.reset()
                
            except FileNotFoundError as e:
                errors.append({
                    "window": window.label,
                    "symbol": symbol,
                    "error": f"Data file not found: {str(e)}"
                })
                print(f"[WINDOW: {window.label}] [SYMBOL: {symbol}] ERROR: Data file not found")
                intraday_replay_engine.reset()
                continue
            except Exception as e:
                errors.append({
                    "window": window.label,
                    "symbol": symbol,
                    "error": f"Replay failed: {str(e)}"
                })
                print(f"[WINDOW: {window.label}] [SYMBOL: {symbol}] ERROR: {str(e)}")
                intraday_replay_engine.reset()
                continue
        
        # Aggregate metrics across symbols for this window
        if per_symbol_window_results:
            # Average metrics across symbols
            return_pcts = [r["return_on_capital_pct"] for r in per_symbol_window_results]
            drawdown_pcts = [r["max_portfolio_drawdown_pct"] for r in per_symbol_window_results]
            expectancies = [r["expectancy"] for r in per_symbol_window_results]
            win_rates = [r["win_rate"] for r in per_symbol_window_results]
            trades_counts = [r["trades_executed"] for r in per_symbol_window_results]
            avg_trades_per_day = [r["average_trades_per_day"] for r in per_symbol_window_results]
            
            window_result["metrics"] = {
                "trades_executed": sum(trades_counts),
                "return_on_capital_pct": round(statistics.mean(return_pcts), 2),
                "max_portfolio_drawdown_pct": round(max(drawdown_pcts), 2) if drawdown_pcts else 0.0,
                "expectancy": round(statistics.mean(expectancies), 2),
                "win_rate": round(statistics.mean(win_rates), 2),
                "average_trades_per_day": round(statistics.mean(avg_trades_per_day), 2),
                "daily_pnl_distribution": {
                    # Aggregate daily P&L stats across all symbols
                    "mean": round(statistics.mean([r["daily_pnl_distribution"]["mean"] for r in per_symbol_window_results]), 2),
                    "std": round(statistics.mean([r["daily_pnl_distribution"]["std"] for r in per_symbol_window_results]), 2),
                    "worst_day": round(min([r["daily_pnl_distribution"]["worst_day"] for r in per_symbol_window_results]), 2),
                    "days_with_trades": max([r["daily_pnl_distribution"]["days_with_trades"] for r in per_symbol_window_results])
                }
            }
            window_result["per_symbol"] = per_symbol_window_results
        else:
            window_result["metrics"] = {
                "trades_executed": 0,
                "return_on_capital_pct": 0.0,
                "max_portfolio_drawdown_pct": 0.0,
                "expectancy": 0.0,
                "win_rate": 0.0,
                "average_trades_per_day": 0.0,
                "daily_pnl_distribution": {
                    "mean": 0.0,
                    "std": 0.0,
                    "worst_day": 0.0,
                    "days_with_trades": 0
                }
            }
        
        window_results.append(window_result)
    
    replay_running = False
    
    # Compute aggregate comparison
    train_windows = [w for w in window_results if w["label"].startswith("train")]
    test_windows = [w for w in window_results if w["label"].startswith("test")]
    
    # Aggregate train metrics
    train_returns = [w["metrics"]["return_on_capital_pct"] for w in train_windows if w["metrics"]["trades_executed"] > 0]
    train_drawdowns = [w["metrics"]["max_portfolio_drawdown_pct"] for w in train_windows if w["metrics"]["trades_executed"] > 0]
    
    # Aggregate test metrics
    test_returns = [w["metrics"]["return_on_capital_pct"] for w in test_windows if w["metrics"]["trades_executed"] > 0]
    test_drawdowns = [w["metrics"]["max_portfolio_drawdown_pct"] for w in test_windows if w["metrics"]["trades_executed"] > 0]
    
    # Compute deltas
    avg_train_return = statistics.mean(train_returns) if train_returns else 0.0
    avg_test_return = statistics.mean(test_returns) if test_returns else 0.0
    return_delta = avg_test_return - avg_train_return
    
    avg_train_drawdown = statistics.mean(train_drawdowns) if train_drawdowns else 0.0
    avg_test_drawdown = statistics.mean(test_drawdowns) if test_drawdowns else 0.0
    drawdown_delta = avg_test_drawdown - avg_train_drawdown
    
    # Percentage of windows with positive return
    all_windows_with_trades = [w for w in window_results if w["metrics"]["trades_executed"] > 0]
    windows_with_positive_return = [w for w in all_windows_with_trades if w["metrics"]["return_on_capital_pct"] > 0]
    pct_positive_windows = (len(windows_with_positive_return) / len(all_windows_with_trades) * 100) if all_windows_with_trades else 0.0
    
    # Worst test drawdown
    worst_test_drawdown = max(test_drawdowns) if test_drawdowns else None
    
    aggregate_comparison = {
        "train_vs_test": {
            "avg_train_return_pct": round(avg_train_return, 2),
            "avg_test_return_pct": round(avg_test_return, 2),
            "return_delta_pct": round(return_delta, 2),
            "avg_train_drawdown_pct": round(avg_train_drawdown, 2),
            "avg_test_drawdown_pct": round(avg_test_drawdown, 2),
            "drawdown_delta_pct": round(drawdown_delta, 2)
        },
        "robustness": {
            "windows_with_positive_return_pct": round(pct_positive_windows, 2),
            "total_windows": len(window_results),
            "windows_with_trades": len(all_windows_with_trades),
            "worst_test_drawdown_pct": round(worst_test_drawdown, 2) if worst_test_drawdown is not None else None
        }
    }
    
    # Log summary
    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY")
    print("=" * 60)
    print(f"Total Windows: {len(window_results)}")
    print(f"Train Windows: {len(train_windows)}")
    print(f"Test Windows: {len(test_windows)}")
    print()
    print("TRAIN vs TEST COMPARISON:")
    print(f"  Avg Train Return: {fmt_pct(avg_train_return)}")
    print(f"  Avg Test Return: {fmt_pct(avg_test_return)}")
    print(f"  Return Delta: {fmt_pct(return_delta)}")
    print(f"  Avg Train Drawdown: {fmt_pct(avg_train_drawdown)}")
    print(f"  Avg Test Drawdown: {fmt_pct(avg_test_drawdown)}")
    print(f"  Drawdown Delta: {fmt_pct(drawdown_delta)}")
    print()
    print("ROBUSTNESS:")
    print(f"  Windows with Positive Return: {len(windows_with_positive_return)}/{len(all_windows_with_trades)} ({fmt_pct(pct_positive_windows)})")
    if worst_test_drawdown is not None:
        print(f"  Worst Test Drawdown: {fmt_pct(worst_test_drawdown)}")
    print("=" * 60 + "\n")
    
    return {
        "status": "completed",
        "symbols": request.symbols,
        "windows": window_results,
        "aggregate_comparison": aggregate_comparison,
        "errors": errors,
        "note": f"Walk-forward evaluation - {len(window_results)} windows, identical logic applied to all windows"
    }


@app.get("/replay/status")
async def get_replay_status():
    """Get current replay status."""
    status = replay_engine.get_status()
    return status


@app.get("/replay/results")
async def get_replay_results(replay_id: str, db: Session = Depends(get_db)):
    """
    Get replay results: trades, equity curve, and metrics.
    
    FIX 2: EXPLICIT REPLAY ISOLATION
    This endpoint returns ONLY replay data (replay_id = UUID).
    Live trading data (replay_id = None) is never returned here.
    Replay state never mutates live paper trading state.
    
    Response Structure:
    - Top-level comparison metrics (for easy side-by-side comparison):
      - allowed_entry_regimes: None (baseline) or ["TREND_UP"] (gated)
      - trade_count: Number of trades executed
      - final_equity: Final account equity
      - max_drawdown_pct: Maximum drawdown percentage
      - expectancy: Expected value per trade
    - Detailed metrics: Full nested metrics snapshot
    - report: Performance summary
    - diagnostics: Trade-level diagnostics (MFE, MAE, holding period)
    - regime_summary: Performance breakdown by market regime
    - trades: Full trade list with diagnostics and regime info
    - equity_curve: Equity curve over time
    
    Args:
        replay_id: Replay identifier (UUID)
    """
    # Fetch replay summary for symbol and date range
    summary = db.query(ReplaySummary).filter(
        ReplaySummary.replay_id == replay_id
    ).first()
    
    # Fetch trades for this replay
    trades = db.query(Trade).filter(
        Trade.replay_id == replay_id
    ).order_by(Trade.entry_time).all()
    
    # Fetch equity curve for this replay
    equity_curve = db.query(EquityCurve).filter(
        EquityCurve.replay_id == replay_id
    ).order_by(EquityCurve.timestamp).all()
    
    # Compute metrics
    metrics_snapshot = compute_metrics(trades, equity_curve)
    
    # Load candles for diagnostics computation
    # Try to get candles from replay engine first (if still available)
    candles_for_diagnostics = []
    if replay_engine.replay_id == replay_id and replay_engine.candle_history:
        # Use candles from replay engine if available
        candles_for_diagnostics = replay_engine.candle_history
    elif summary and summary.source == "csv":
        # For CSV replays, try to reload from common CSV paths
        # Try common pattern: data d/{SYMBOL}_daily.csv
        csv_paths_to_try = [
            f"data d/{summary.symbol}_daily.csv",
            f"../data d/{summary.symbol}_daily.csv",
            f"backend/data d/{summary.symbol}_daily.csv",
        ]
        for csv_path in csv_paths_to_try:
            try:
                csv_candles = load_csv_candles(
                    csv_path=csv_path,
                    symbol=summary.symbol
                )
                candles_for_diagnostics = csv_convert_to_replay_format(csv_candles)
                print(f"[DIAGNOSTICS] Loaded {len(candles_for_diagnostics)} candles from {csv_path} for diagnostics")
                break
            except (FileNotFoundError, ValueError) as e:
                continue  # Try next path
        if not candles_for_diagnostics:
            print(f"[DIAGNOSTICS] Warning: Could not reload CSV candles for diagnostics. Tried: {csv_paths_to_try}")
    elif summary and summary.source == "yahoo_finance" and summary.start_date and summary.end_date:
        # Reload candles from Yahoo Finance
        try:
            yahoo_candles = fetch_yahoo_candles(
                symbol=summary.symbol,
                start_date=summary.start_date,
                end_date=summary.end_date
            )
            candles_for_diagnostics = convert_to_replay_format(yahoo_candles)
        except Exception as e:
            print(f"[DIAGNOSTICS] Warning: Could not reload candles for diagnostics: {e}")
            candles_for_diagnostics = []
    
    # Compute trade diagnostics (per-trade and aggregate)
    trade_diagnostics = []
    for trade in trades:
        trade_dict = trade.to_dict()
        diagnostics = compute_trade_diagnostics(trade, candles_for_diagnostics)
        trade_dict["diagnostics"] = diagnostics
        trade_diagnostics.append(trade_dict)
    
    # Compute aggregate diagnostics
    aggregate_diagnostics = compute_aggregate_diagnostics(trades, candles_for_diagnostics) if candles_for_diagnostics else {}
    
    # Compute regime metrics and attach regime to trades
    regime_metrics = {}
    trades_with_regime = []
    if candles_for_diagnostics:
        regime_metrics = compute_regime_metrics(trades, candles_for_diagnostics)
        trades_with_regime = attach_regime_to_trades(trades, candles_for_diagnostics)
        
        # Merge regime info into trade_diagnostics
        for i, trade_dict in enumerate(trade_diagnostics):
            if i < len(trades_with_regime):
                trade_dict["entry_regime"] = trades_with_regime[i]["entry_regime"]
                trade_dict["exit_regime"] = trades_with_regime[i]["exit_regime"]
    else:
        # No candles available, add None regimes
        for trade_dict in trade_diagnostics:
            trade_dict["entry_regime"] = None
            trade_dict["exit_regime"] = None
    
    # Generate performance report
    closed_trades = [t for t in trades if t.exit_time is not None]
    stop_loss_trades = [t for t in closed_trades if t.reason and "STOP_LOSS" in t.reason.upper()]
    stop_loss_pct = (len(stop_loss_trades) / len(closed_trades) * 100) if len(closed_trades) > 0 else 0.0
    
    initial_equity = summary.final_equity - summary.net_pnl if summary else 100000.0
    net_pnl = summary.net_pnl if summary else (metrics_snapshot.equity_end - metrics_snapshot.equity_start)
    
    # Parse allowed_entry_regimes from JSON string if stored
    allowed_regimes = None
    if summary and summary.allowed_entry_regimes:
        try:
            allowed_regimes = json.loads(summary.allowed_entry_regimes)
        except (json.JSONDecodeError, TypeError):
            allowed_regimes = None
    
    replay_report = {
        "symbol": summary.symbol if summary else None,
        "start_date": summary.start_date if summary else None,
        "end_date": summary.end_date if summary else None,
        "candles_processed": summary.candle_count if summary else 0,
        "trades_executed": len(trades),
        "win_rate": metrics_snapshot.win_rate,
        "net_pnl": net_pnl,
        "max_drawdown_pct": metrics_snapshot.max_drawdown_pct,
        "sharpe_proxy": metrics_snapshot.sharpe_proxy if not (math.isnan(metrics_snapshot.sharpe_proxy) or math.isinf(metrics_snapshot.sharpe_proxy)) else None,
        "stop_loss_pct": round(stop_loss_pct, 2),
        "expectancy": metrics_snapshot.expectancy_per_trade,  # Expected value per trade
        "allowed_entry_regimes": allowed_regimes  # Show which regimes were allowed
    }
    
    # Top-level comparison metrics for easy side-by-side comparison
    comparison_metrics = {
        "trade_count": len(trades),
        "final_equity": summary.final_equity if summary else metrics_snapshot.equity_end,
        "max_drawdown_pct": metrics_snapshot.max_drawdown_pct,
        "expectancy": metrics_snapshot.expectancy_per_trade,
        "allowed_entry_regimes": allowed_regimes  # None = baseline, ["TREND_UP"] = gated
    }
    
    return {
        "replay_id": replay_id,
        "symbol": summary.symbol if summary else None,
        "start_date": summary.start_date if summary else None,
        "end_date": summary.end_date if summary else None,
        "source": summary.source if summary else None,
        "allowed_entry_regimes": allowed_regimes,  # Show regime gate setting (top level for visibility)
        # Top-level comparison metrics for easy side-by-side comparison
        "trade_count": len(trades),
        "final_equity": summary.final_equity if summary else metrics_snapshot.equity_end,
        "max_drawdown_pct": metrics_snapshot.max_drawdown_pct,
        "expectancy": metrics_snapshot.expectancy_per_trade,
        # Detailed metrics and diagnostics
        "metrics": metrics_snapshot.to_dict(),  # Full metrics snapshot (nested structure)
        "report": replay_report,  # Performance report (No Optimization)
        "comparison_metrics": comparison_metrics,  # Flattened key metrics for comparison
        "diagnostics": {
            "aggregate": aggregate_diagnostics,
            "per_trade": [t["diagnostics"] for t in trade_diagnostics]  # Just diagnostics, not full trade objects
        },
        "regime_summary": regime_metrics,  # Performance breakdown by regime
        "trades": trade_diagnostics,  # Trades with diagnostics and regime info included
        "equity_curve": [
            {
                "timestamp": point.timestamp.isoformat() if point.timestamp else None,
                "equity": point.equity
            }
            for point in equity_curve
        ]
    }


@app.get("/replay/history")
async def get_replay_history(limit: int = 100, db: Session = Depends(get_db)):
    """
    Get history of completed replays for evaluation and inspection.
    
    Returns JSON array of replay summaries sorted by completion time (newest first).
    Each summary includes all fields needed for easy inspection and sharing:
    - replay_id, symbol, start_date, end_date
    - candle_count, trade_count, final_equity
    - net_pnl, max_drawdown_pct, sharpe_proxy
    - timestamp_completed
    
    Used for:
    - Tracking replay runs
    - Determinism verification (comparing same inputs across runs)
    - Performance evaluation over time
    - Easy inspection and sharing of results
    
    Args:
        limit: Maximum number of summaries to return (default: 100)
    
    Returns:
        JSON object with:
        - replay_summaries: Array of replay summary objects
        - count: Number of summaries returned
    """
    summaries = db.query(ReplaySummary).order_by(
        ReplaySummary.timestamp_completed.desc()
    ).limit(limit).all()
    
    return {
        "replay_summaries": [s.to_dict() for s in summaries],
        "count": len(summaries)
    }


# ============================================================================
# YAHOO FINANCE DATA ENDPOINTS
# ============================================================================

class YahooFetchRequest(BaseModel):
    symbol: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD


@app.post("/data/yahoo/fetch")
async def fetch_yahoo_data(request: YahooFetchRequest):
    """
    Fetch historical daily candles from Yahoo Finance.
    
    Note: Daily candles (1d interval) are used as the default because Yahoo Finance
    has limitations on intraday data availability. For historical backtesting,
    daily candles provide sufficient granularity and better data quality.
    
    Safeguards:
    - Validates symbol is in approved universe (Nasdaq-100, Dow-30)
    - Limits date range to 1 year maximum
    - Returns normalized candles ready for replay
    - Rejects intraday interval requests with clear error message
    
    Args:
        request: YahooFetchRequest with symbol, start_date, end_date
    """
    # Safeguard: Validate symbol
    if request.symbol not in ALL_SYMBOLS:
        return {"error": f"Symbol {request.symbol} not in approved universe (Nasdaq-100, Dow-30)"}, 400
    
    # Safeguard: Validate date range (max 1 year)
    try:
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError as e:
        return {"error": f"Invalid date format. Use YYYY-MM-DD. Error: {e}"}, 400
    
    # Check date range
    date_range = (end_dt - start_dt).days
    if date_range > 365:
        return {"error": f"Date range exceeds 1 year limit. Requested: {date_range} days"}, 400
    
    if date_range <= 0:
        return {"error": "End date must be after start date"}, 400
    
    try:
        # Fetch candles from Yahoo Finance
        candles = fetch_yahoo_candles(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if not candles:
            return {"error": f"No valid candles returned for {request.symbol}"}, 400
        
        # Convert to replay format (Unix timestamps)
        replay_candles = convert_to_replay_format(candles)
        
        # Return response with metadata
        return {
            "symbol": request.symbol,
            "source": "yahoo_finance",
            "candle_count": len(replay_candles),
            "start_timestamp": replay_candles[0]["time"] if replay_candles else None,
            "end_timestamp": replay_candles[-1]["time"] if replay_candles else None,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "sample_first": replay_candles[0] if replay_candles else None,
            "sample_last": replay_candles[-1] if replay_candles else None,
            "candles": replay_candles  # Full candle list for replay
        }
    
    except ValueError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        return {"error": f"Failed to fetch Yahoo Finance data: {str(e)}"}, 500


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
