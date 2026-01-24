from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import asyncio
import random
import time
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import init_db, get_db, Signal, Trade, EquityCurve
from indicators import ema, atr
from strategy import ema_trend_v1, PositionState
from paper_broker import PaperBroker
from metrics import compute_metrics
from replay_engine import ReplayEngine

app = FastAPI(title="Paper Trading API")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

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
    """
    if symbol not in ALL_SYMBOLS:
        return {"error": f"Symbol {symbol} not in allowed list"}, 400
    
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
    Only includes live trading data (replay_id is None).
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
        # AUDIT FIX: Prevent duplicate signal storage (idempotency check)
        signal_time = datetime.fromtimestamp(new_candle["time"], tz=timezone.utc)
        existing_signal = db.query(Signal).filter(
            Signal.symbol == symbol,
            Signal.timestamp == signal_time,
            Signal.signal == result["signal"]
        ).first()
        
        if not existing_signal:
            # AUDIT FIX: Use UTC explicitly for timezone consistency
            # Live trading: replay_id is None
            signal = Signal(
                timestamp=signal_time,
                symbol=symbol,
                signal=result["signal"],
                price=new_candle["close"],
                reason=result["reason"],
                replay_id=None  # None for live trading
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
        open_trade = db.query(Trade).filter(
            Trade.symbol == exit_trade["symbol"],
            Trade.exit_time.is_(None)
        ).first()
        
        if open_trade:
            # AUDIT FIX: Use UTC explicitly for timezone consistency
            open_trade.exit_time = datetime.fromtimestamp(exit_trade["timestamp"], tz=timezone.utc)
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
            # Create trade record (entry only, exit will be added later)
            # AUDIT FIX: Use UTC explicitly for timezone consistency
            # Live trading: replay_id is None
            trade = Trade(
                symbol=symbol,
                entry_time=datetime.fromtimestamp(new_candle["time"], tz=timezone.utc),
                entry_price=trade_executed["entry_price"],
                shares=trade_executed["shares"],
                exit_time=None,
                exit_price=None,
                pnl=None,
                reason=None,
                replay_id=None  # None for live trading
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
            # Find open trade and update with exit
            open_trade = db.query(Trade).filter(
                Trade.symbol == symbol,
                Trade.exit_time.is_(None)
            ).first()
            
            if open_trade:
                # AUDIT FIX: Use UTC explicitly for timezone consistency
                open_trade.exit_time = datetime.fromtimestamp(new_candle["time"], tz=timezone.utc)
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
        open_trade = db.query(Trade).filter(
            Trade.symbol == exit_trade["symbol"],
            Trade.exit_time.is_(None)
        ).first()
        
        if open_trade:
            # AUDIT FIX: Use UTC explicitly for timezone consistency
            open_trade.exit_time = datetime.fromtimestamp(exit_trade["timestamp"], tz=timezone.utc)
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
    # AUDIT FIX: Use UTC explicitly for timezone consistency
    # Live trading: replay_id is None
    equity_point = EquityCurve(
        timestamp=datetime.fromtimestamp(new_candle["time"], tz=timezone.utc),
        equity=broker.equity,
        replay_id=None  # None for live trading
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
    candles: List[Dict]  # List of candle dicts: time, open, high, low, close, volume
    replay_id: Optional[str] = None


@app.post("/replay/start")
async def start_replay(request: ReplayRequest, db: Session = Depends(get_db)):
    """
    Start a historical replay (backtest).
    
    Safeguards:
    - Prevents concurrent replay/live trading
    - Validates symbol in allowed universe
    - Validates candle ordering
    """
    global replay_running
    
    # Safeguard: Prevent concurrent replay
    if replay_running:
        return {"error": "Replay already in progress"}, 400
    
    # Safeguard: Validate symbol
    if request.symbol not in ALL_SYMBOLS:
        return {"error": f"Symbol {request.symbol} not in allowed list"}, 400
    
    # Safeguard: Validate candles
    if not request.candles or len(request.candles) < 50:
        return {"error": "Need at least 50 candles for replay (EMA(50) requires 50 candles)"}, 400
    
    try:
        replay_id = replay_engine.start_replay(
            symbol=request.symbol,
            candles=request.candles,
            replay_id=request.replay_id
        )
        
        replay_running = True
        
        # Run replay synchronously (in production, consider background task)
        result = replay_engine.run(db)
        
        replay_running = False
        
        return {
            "status": "completed",
            "replay_id": replay_id,
            "symbol": request.symbol,
            "total_candles": result["total_candles"],
            "final_equity": result["final_equity"]
        }
    
    except ValueError as e:
        replay_running = False
        return {"error": str(e)}, 400
    except Exception as e:
        replay_running = False
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
    
    Args:
        replay_id: Replay identifier
    """
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
        "trades": [t.to_dict() for t in trades],
        "equity_curve": [
            {
                "timestamp": point.timestamp.isoformat() if point.timestamp else None,
                "equity": point.equity
            }
            for point in equity_curve
        ],
        "metrics": metrics_snapshot.to_dict()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
