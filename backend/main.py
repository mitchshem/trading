from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import asyncio
import random
import time
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
from utils import ensure_utc_datetime, unix_to_utc_datetime

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
ALL_SYMBOLS = sorted(list(set(DOW_30 + NASDAQ_100)))

# Store active WebSocket connections
active_connections: Dict[str, List[WebSocket]] = {}

# Track position state per symbol
position_states: Dict[str, PositionState] = {}

# Track candle history per symbol (for strategy evaluation)
candle_history: Dict[str, List[Dict]] = {}

# Global paper broker instance
broker = PaperBroker(initial_equity=100000.0)

# Global replay engine instance
replay_engine = ReplayEngine(initial_equity=100000.0)

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
            yahoo_candles = fetch_yahoo_candles(
                symbol=request.symbol,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            if not yahoo_candles:
                return {"error": f"No valid candles returned for {request.symbol}"}, 400
            
            # Convert to replay format (Unix timestamps)
            candles = convert_to_replay_format(yahoo_candles)
            source = "yahoo_finance"
            
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            return {"error": f"Failed to fetch Yahoo Finance data: {str(e)}"}, 500
    else:
        return {"error": "Must provide either 'candles' array or both 'start_date' and 'end_date'"}, 400
    
    # GUARDRAILS: Validate candles meet minimum thresholds
    MIN_CANDLES_FOR_REPLAY = 500  # Minimum for meaningful evaluation
    MIN_CANDLES_FOR_INDICATORS = 50  # Minimum for EMA(50)
    
    if not candles:
        return {"error": "No candles provided"}, 400
    
    if len(candles) < MIN_CANDLES_FOR_INDICATORS:
        return {"error": f"Need at least {MIN_CANDLES_FOR_INDICATORS} candles for replay (EMA(50) requires {MIN_CANDLES_FOR_INDICATORS} candles)"}, 400
    
    if len(candles) < MIN_CANDLES_FOR_REPLAY:
        return {
            "error": f"Insufficient candles for meaningful evaluation. Got {len(candles)}, minimum {MIN_CANDLES_FOR_REPLAY} required.",
            "warning": f"Replay with fewer than {MIN_CANDLES_FOR_REPLAY} candles may not provide reliable performance metrics"
        }, 400
    
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
        replay_id = replay_engine.start_replay(
            symbol=request.symbol,
            candles=candles,
            replay_id=request.replay_id,
            source=source
        )
        
        replay_running = True
        
        # MULTI-RUN SAFETY: Ensure replay state is always cleaned up, even on failure
        try:
            # Run replay synchronously (in production, consider background task)
            # Processes candles ONE AT A TIME through full pipeline
            result = replay_engine.run(db)
            
            # Fetch trades and equity curve for metrics computation
            trades = db.query(Trade).filter(
                Trade.replay_id == replay_id
            ).order_by(Trade.entry_time).all()
            
            equity_curve = db.query(EquityCurve).filter(
                EquityCurve.replay_id == replay_id
            ).order_by(EquityCurve.timestamp).all()
            
            # Compute metrics for determinism verification and summary
            metrics_snapshot = compute_metrics(trades, equity_curve)
            
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
                "win_rate": metrics_snapshot.core_metrics.win_rate,
                "net_pnl": net_pnl,
                "max_drawdown_pct": metrics_snapshot.risk_metrics.max_drawdown_pct,
                "sharpe_proxy": metrics_snapshot.risk_adjusted.sharpe_proxy if metrics_snapshot.risk_adjusted.sharpe_proxy is not None else None,
                "stop_loss_pct": round(stop_loss_pct, 2)
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
            print(f"Win Rate: {replay_report['win_rate']:.2f}%")
            print(f"Net P&L: ${replay_report['net_pnl']:,.2f}")
            print(f"Max Drawdown: {replay_report['max_drawdown_pct']:.2f}%")
            sharpe_display = f"{replay_report['sharpe_proxy']:.2f}" if replay_report['sharpe_proxy'] is not None else "N/A"
            print(f"Sharpe Proxy: {sharpe_display}")
            print(f"Stop-Loss Exits: {replay_report['stop_loss_pct']:.2f}% of trades")
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
                        mismatches.append(f"final_equity: {prev.final_equity:.2f} vs {result['final_equity']:.2f}")
                    if abs(prev.max_drawdown_pct - metrics_snapshot.risk_metrics.max_drawdown_pct) > 0.01:
                        mismatches.append(f"max_drawdown_pct: {prev.max_drawdown_pct:.2f}% vs {metrics_snapshot.risk_metrics.max_drawdown_pct:.2f}%")
                    
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
                        print(f"  Final equity: {prev.final_equity:.2f}")
                        print(f"  Max drawdown: {prev.max_drawdown_pct:.2f}%")
                        print(f"\nCurrent Run Results:")
                        print(f"  Candle count: {result['total_candles']}")
                        print(f"  Trade count: {len(trades)}")
                        print(f"  Final equity: {result['final_equity']:.2f}")
                        print(f"  Max drawdown: {metrics_snapshot.risk_metrics.max_drawdown_pct:.2f}%")
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
                        print(f"All metrics match: candle_count={result['total_candles']}, trade_count={len(trades)}, final_equity={result['final_equity']:.2f}, max_drawdown={metrics_snapshot.risk_metrics.max_drawdown_pct:.2f}%\n")
                else:
                    # No previous run to compare
                    determinism_status = "no_previous_run"
                    determinism_message = "No previous run to compare (first run for this input)"
                    print(f"[DETERMINISM] First run for fingerprint: {replay_fingerprint}")
            
            # Log replay fingerprint for tracking
            print(f"[DETERMINISM] Replay fingerprint: {replay_fingerprint}")
            
            # Calculate net P&L (final_equity - initial_equity)
            initial_equity = replay_engine.initial_equity
            net_pnl = result["final_equity"] - initial_equity
            
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
            print(f"Final equity: ${result['final_equity']:,.2f}")
            print(f"Net P&L: ${net_pnl:,.2f}")
            print(f"Max drawdown: {metrics_snapshot.risk_metrics.max_drawdown_pct:.2f}%")
            sharpe_display = f"{metrics_snapshot.risk_adjusted.sharpe_proxy:.2f}" if metrics_snapshot.risk_adjusted.sharpe_proxy is not None else "N/A"
            print(f"Sharpe proxy: {sharpe_display}")
            print(f"Replay ID: {replay_id}")
            print("=" * 50 + "\n")
            
            # Log determinism metrics (for debugging)
            print(f"[DETERMINISM] Replay {replay_id} completed:")
            print(f"  Symbol: {request.symbol}")
            print(f"  Date range: {request.start_date} to {request.end_date}")
            print(f"  Candle count: {result['total_candles']}")
            print(f"  Trade count: {len(trades)}")
            print(f"  Final equity: {result['final_equity']:.2f}")
            print(f"  Max drawdown: {metrics_snapshot.risk_metrics.max_drawdown_pct:.2f}%")
            
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
                max_drawdown_pct=metrics_snapshot.risk_metrics.max_drawdown_pct,
                max_drawdown_absolute=metrics_snapshot.risk_metrics.max_drawdown_absolute,
                sharpe_proxy=metrics_snapshot.risk_adjusted.sharpe_proxy,
                timestamp_completed=datetime.now(timezone.utc)
            )
            db.add(replay_summary)
            db.commit()
            
            replay_running = False
            
            return {
                "status": "completed",
                "replay_id": replay_id,
                "symbol": request.symbol,
                "total_candles": result["total_candles"],
                "final_equity": result["final_equity"],
                "trade_count": len(trades),
                "max_drawdown_pct": metrics_snapshot.risk_metrics.max_drawdown_pct,
                "source": source,
                "determinism_status": determinism_status,
                "determinism_message": determinism_message,
                "determinism_mismatches": determinism_mismatches,
                "replay_fingerprint": replay_fingerprint,
                "report": replay_report
            }
        
        except Exception as e:
            # MULTI-RUN SAFETY: Always reset replay_running flag on failure
            replay_running = False
            # MULTI-RUN SAFETY: Reset replay engine state to prevent corruption
            replay_engine.reset()
            raise
    
    except ValueError as e:
        replay_running = False
        replay_engine.reset()  # MULTI-RUN SAFETY: Clean up state
        return {"error": str(e)}, 400
    except Exception as e:
        replay_running = False
        replay_engine.reset()  # MULTI-RUN SAFETY: Clean up state
        return {"error": f"Replay failed: {str(e)}"}, 500


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
    
    Returns:
    - symbol: Trading symbol
    - start_date, end_date: Date range (if available)
    - metrics: Full metrics snapshot (easy to find)
    - trades: List of trades (optional detail)
    - equity_curve: Equity curve data (optional detail)
    
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
    
    return {
        "replay_id": replay_id,
        "symbol": summary.symbol if summary else None,
        "start_date": summary.start_date if summary else None,
        "end_date": summary.end_date if summary else None,
        "source": summary.source if summary else None,
        "metrics": metrics_snapshot.to_dict(),  # Metrics are easy to find at top level
        "trades": [t.to_dict() for t in trades],
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
    Fetch historical 5-minute candles from Yahoo Finance.
    
    Safeguards:
    - Validates symbol is in approved universe (Nasdaq-100, Dow-30)
    - Limits date range to 1 year maximum
    - Returns normalized candles ready for replay
    
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
