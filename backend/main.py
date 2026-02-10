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
from promotion_rules import PromotionRules, PromotionThresholds
from monitoring import monitor, APITimingMiddleware
from replay_engine import ReplayEngine
from market_data.yahoo import fetch_yahoo_candles, convert_to_replay_format
from market_data.csv_loader import load_csv_candles, convert_to_replay_format as csv_convert_to_replay_format
from market_data.stooq_loader import load_daily_candles, load_intraday_candles
from trade_diagnostics import compute_trade_diagnostics, compute_aggregate_diagnostics
from regime_metrics import compute_regime_metrics, attach_regime_to_trades
from intraday_replay import IntradayReplayEngine
from intraday_diagnostics import compute_intraday_trade_diagnostics, compute_intraday_aggregate_metrics, compute_frequency_and_session_metrics
from walkforward import filter_candles_by_date_range, compute_window_metrics
from walkforward_harness import WalkForwardHarness
from cost_sensitivity import CostSensitivityTester
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
    # Initialize notification service
    from notification_service import notifier
    notifier.load_prefs()
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
    print("\nLive Trading Endpoints:")
    print("  POST /live/start - Start live paper trading loop")
    print("  POST /live/stop - Stop live paper trading loop")
    print("  GET  /live/status - Get live trading status")
    print("  GET  /decisions - Get decision log entries")
    print("  GET  /events - Server-Sent Events (real-time)")
    print("  GET  /strategies - List all strategies with params")
    print("  POST /live/switch-strategy - Switch active strategy")
    print("  PATCH /live/params - Update strategy params")
    print("  PATCH /risk/limits - Update risk limits")
    print("  GET  /data/alpaca/status - Alpaca connection status")
    print("  POST /data/alpaca/test - Test Alpaca connection")
    print("\nNotification Endpoints:")
    print("  GET  /notifications/prefs - Get notification preferences")
    print("  PATCH /notifications/prefs - Update notification preferences")
    print("  POST /notifications/test-email - Test email delivery")
    print("  GET  /notifications/history - Recent notification history")
    print("\nOther Key Endpoints:")
    print("  GET  /symbols - List available symbols")
    print("  GET  /candles?symbol=XXX - Get historical candles")
    print("  GET  /account - Get account state")
    print("  GET  /trades - Get trade history")
    print("  GET  /metrics - Get performance metrics")
    print("  GET  /promotion/status - Promotion readiness check")
    print("  WebSocket /ws - Real-time candle stream")
    print("\nMonitoring Endpoints:")
    print("  GET  /monitoring/status - System health snapshot")
    print("  GET  /monitoring/alerts - Recent alerts")
    print("  GET  /monitoring/api-stats - API performance stats")
    print("=" * 60 + "\n")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API timing middleware (Phase 5: monitoring)
app.add_middleware(APITimingMiddleware)

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

# Global promotion rules instance (for live trading gate)
promotion_rules = PromotionRules(
    thresholds=PromotionThresholds(
        max_drawdown_pct=15.0,  # Max 15% drawdown
        sharpe_proxy_min=0.5,  # Minimum Sharpe proxy of 0.5
        min_trade_count=20,  # Minimum 20 closed trades
        min_win_rate=0.40,  # Minimum 40% win rate
        min_expectancy=0.0  # Positive expectancy required
    )
)

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


@app.get("/promotion/status")
async def get_promotion_status(db: Session = Depends(get_db)):
    """
    Check paper trading promotion status.
    Returns whether live trading is allowed based on objective thresholds.
    
    Only includes live trading data (replay_id is None).
    """
    # Fetch live trading metrics
    live_trades = db.query(Trade).filter(Trade.replay_id.is_(None)).all()
    live_equity = db.query(EquityCurve).filter(EquityCurve.replay_id.is_(None)).order_by(EquityCurve.timestamp).all()
    
    # Evaluate promotion (with 0 OOS windows by default — updated when walk-forward runs)
    decision = promotion_rules.evaluate_promotion(live_trades, live_equity, oos_positive_windows=0)

    return {
        "promoted": decision.promoted,
        "reasons": decision.reasons,
        "metrics": decision.metrics,
        "checks_passed": decision.checks_passed,
        "checks_total": decision.checks_total,
        "readiness_pct": decision.readiness_pct,
        "thresholds": {
            "max_drawdown_pct": promotion_rules.thresholds.max_drawdown_pct,
            "sharpe_proxy_min": promotion_rules.thresholds.sharpe_proxy_min,
            "min_trade_count": promotion_rules.thresholds.min_trade_count,
            "min_win_rate": promotion_rules.thresholds.min_win_rate,
            "min_expectancy": promotion_rules.thresholds.min_expectancy,
            "min_trading_days": promotion_rules.thresholds.min_trading_days,
            "min_profit_factor": promotion_rules.thresholds.min_profit_factor,
            "min_oos_windows_positive": promotion_rules.thresholds.min_oos_windows_positive,
        }
    }


def evaluate_strategy_on_candle_close(symbol: str, new_candle: Dict, db: Session):
    """
    Evaluate strategy when a candle closes, route through paper broker, and store trades.
    This is called when a new candle is generated (representing a closed candle).
    
    Execution Model:
    1. Process pending orders from previous candle at current candle OPEN
    2. Evaluate strategy on current candle CLOSE
    3. Create pending orders for signals (BUY/EXIT)
    4. Check stop-losses (create pending EXIT orders)
    5. Check kill-switch (create pending EXIT orders)
    
    Key guarantee: Signals NEVER execute on the same candle.
    All orders execute at the NEXT candle's OPEN price.
    
    This ensures identical execution semantics with ReplayEngine.
    
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
    
    # Process pending orders from previous candle at current candle open
    # Use open_time for execution timestamp
    open_time = new_candle.get("open_time", new_candle["time"])  # Fallback for backward compatibility
    executed_trades = broker.process_pending_orders(
        current_open_price=new_candle["open"],
        timestamp=open_time
    )
    
    # Update trade records for executed orders
    for trade in executed_trades:
        if trade["action"] == "BUY":
            # FIX 1: CANONICAL TIME HANDLING
            entry_time = unix_to_utc_datetime(trade["timestamp"])
            ensure_utc_datetime(entry_time, f"BUY entry time for {trade['symbol']}")
            
            # FIX 2: EXPLICIT REPLAY ISOLATION
            # Live trading: replay_id is None
            trade_record = Trade(
                symbol=trade["symbol"],
                entry_time=entry_time,
                entry_price=trade["entry_price"],
                shares=trade["shares"],
                exit_time=None,
                exit_price=None,
                pnl=None,
                reason=None,
                replay_id=None
            )
            db.add(trade_record)
            db.commit()
            
            # Update position state if this was the current symbol
            if trade["symbol"] == symbol:
                if symbol not in position_states:
                    position_states[symbol] = PositionState()
                position_states[symbol].has_position = True
                position_states[symbol].entry_price = trade["entry_price"]
                position_states[symbol].entry_time = trade["timestamp"]
        
        elif trade["action"] == "EXIT":
            # Find open trade and update with exit
            open_trade = db.query(Trade).filter(
                Trade.symbol == trade["symbol"],
                Trade.exit_time.is_(None),
                Trade.replay_id.is_(None)  # Only live trading trades
            ).first()
            
            if open_trade:
                # FIX 1: CANONICAL TIME HANDLING
                exit_time = unix_to_utc_datetime(trade["timestamp"])
                ensure_utc_datetime(exit_time, f"EXIT exit time for {trade['symbol']}")
                open_trade.exit_time = exit_time
                open_trade.exit_price = trade["exit_price"]
                open_trade.pnl = trade["pnl"]
                open_trade.reason = trade["reason"]
                db.commit()
            
            # Update position state if this was the current symbol
            if trade["symbol"] == symbol:
                if symbol not in position_states:
                    position_states[symbol] = PositionState()
                position_states[symbol].has_position = False
                position_states[symbol].entry_price = None
                position_states[symbol].entry_time = None
    
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
        position_state=position_state
    )
    
    # Store signal if not HOLD
    signal_stored = False
    if result["signal"] != "HOLD":
        # FIX 1: CANONICAL TIME HANDLING
        # Use close_time for signal timestamp (signals generated on candle close)
        close_time = new_candle.get("close_time", new_candle["time"])  # Fallback for backward compatibility
        signal_time = unix_to_utc_datetime(close_time)
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
    # Stop-losses create pending orders that execute on next candle open
    current_prices_all = {}
    for sym in broker.positions.keys():
        if sym in candle_history and len(candle_history[sym]) > 0:
            current_prices_all[sym] = candle_history[sym][-1]["close"]
    # Add current symbol's price
    current_prices_all[symbol] = new_candle["close"]
    
    # Use close_time for stop-loss checks (checked on candle close)
    close_time = new_candle.get("close_time", new_candle["time"])  # Fallback for backward compatibility
    stop_loss_orders = broker.check_stop_losses(current_prices_all, close_time)
    
    # Track which symbols have pending stop-loss exits to prevent double processing
    stop_loss_symbols = set()
    for exit_order in stop_loss_orders:
        stop_loss_symbols.add(exit_order["symbol"])
        # Note: Orders will execute on next candle open, so we don't update trade records here
    
    # Route signal through paper broker (creates pending orders)
    kill_switch_triggered = False
    
    # AUDIT FIX: Prevent double EXIT processing - if stop-loss already exited, skip signal EXIT
    if result["signal"] == "BUY" and current_atr is not None and symbol not in stop_loss_symbols:
        # PROMOTION RULES: Check if BUY trade is allowed
        # Fetch live trading metrics (replay_id=None)
        live_trades = db.query(Trade).filter(Trade.replay_id.is_(None)).all()
        live_equity = db.query(EquityCurve).filter(EquityCurve.replay_id.is_(None)).order_by(EquityCurve.timestamp).all()
        
        trade_allowed, block_reason = promotion_rules.check_trade_allowed(
            trades=live_trades,
            equity_curve=live_equity,
            trade_type="BUY"
        )
        
        if not trade_allowed:
            # Block trade - log signal but don't create pending order
            print(f"[PROMOTION RULES] BUY signal blocked for {symbol}: {block_reason}")
            # Signal already stored above, but trade is blocked
        else:
            # Calculate stop distance (2 * ATR)
            stop_distance = 2 * current_atr
            
            # Create pending BUY order (executes on next candle open)
            # Signal timestamp uses close_time (signal generated on close)
            # close_time already defined above
            pending_order = broker.execute_buy(
                symbol=symbol,
                signal_price=new_candle["close"],
                stop_distance=stop_distance,
                timestamp=close_time
            )
            
            # Note: Trade record will be created when order executes on next candle open
    
    elif result["signal"] == "EXIT" and symbol not in stop_loss_symbols:
        # AUDIT FIX: Only process signal EXIT if stop-loss didn't already exit this symbol
        # Create pending EXIT order (executes on next candle open)
        # Signal timestamp uses close_time (signal generated on close)
        close_time = new_candle.get("close_time", new_candle["time"])  # Fallback for backward compatibility
        pending_order = broker.execute_exit(
            symbol=symbol,
            signal_price=new_candle["close"],
            timestamp=close_time,
            reason=result["reason"]
        )
        
        # Note: Trade record will be updated when order executes on next candle open
    
    # Check kill switch (creates pending orders)
    # Ensure equity is updated before risk checks
    broker.update_equity(current_prices_all)
    # Use close_time for kill switch checks (checked on candle close)
    close_time = new_candle.get("close_time", new_candle["time"])  # Fallback for backward compatibility
    kill_switch_orders = broker.check_and_enforce_risk_controls(current_prices_all, close_time)
    
    # Note: Orders will execute on next candle open, so we don't update trade records here
    
    if kill_switch_orders:
        kill_switch_triggered = True
    
    # Update equity curve (equity already updated above for risk checks)
    # FIX 1: CANONICAL TIME HANDLING
    # Use close_time for equity curve (equity calculated on candle close)
    close_time = new_candle.get("close_time", new_candle["time"])  # Fallback for backward compatibility
    equity_timestamp = unix_to_utc_datetime(close_time)
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
    
    # Return result with signal info (trades execute on next candle open)
    if signal_stored:
        return {
            "signal": result,
            "trade": None,  # Trades execute on next candle open
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


class WalkForwardDailyRequest(BaseModel):
    """Walk-forward evaluation request for daily replay."""
    symbol: str  # Single symbol to evaluate
    start_date: str  # YYYY-MM-DD format (required for days-based windows)
    end_date: str  # YYYY-MM-DD format (required for days-based windows)
    train_days: Optional[int] = 252  # Number of trading days in training window (~1 year)
    test_days: Optional[int] = 63  # Number of trading days in test window (~1 quarter)
    step_days: Optional[int] = 21  # Number of trading days to step forward (~1 month)
    train_bars: Optional[int] = None  # Number of candles in training window (alternative to train_days)
    test_bars: Optional[int] = None  # Number of candles in test window (alternative to test_days)
    step_bars: Optional[int] = None  # Number of candles to step forward (alternative to step_days)
    initial_equity: float = 100000.0  # Starting equity for each window
    allowed_entry_regimes: Optional[List[str]] = None  # Optional regime gate (e.g., ["TREND_UP"])
    save_artifacts: bool = True  # Whether to save CSV/JSON files
    output_dir: str = "walkforward_results"  # Directory to save artifacts


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


@app.post("/replay/walkforward_daily")
async def walkforward_daily_replay(request: WalkForwardDailyRequest, db: Session = Depends(get_db)):
    """
    Run walk-forward backtest using daily replay with rolling train/test windows.
    
    This endpoint implements rolling walk-forward evaluation:
    - Generates rolling train/test windows automatically
    - Reuses ReplayEngine for identical execution semantics to replay/live
    - Outputs per-window metrics and stitched OOS equity curve
    - Saves artifacts: trades.csv, equity_curve.csv, metrics.json
    
    Example payload:
    {
      "symbol": "SPY",
      "start_date": "2020-01-01",
      "end_date": "2024-12-31",
      "train_days": 252,
      "test_days": 63,
      "step_days": 21,
      "initial_equity": 100000.0,
      "allowed_entry_regimes": ["TREND_UP"],
      "save_artifacts": true,
      "output_dir": "walkforward_results"
    }
    
    Args:
        request: WalkForwardDailyRequest with symbol, dates, and window configuration
    
    Returns:
        Dict with window results, aggregate metrics, and artifact paths
    """
    # Validate symbol
    if request.symbol not in ALL_SYMBOLS:
        return {"error": f"Invalid symbol: {request.symbol}. Must be in allowed list"}, 400
    
    # Validate dates
    try:
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
        if end_dt <= start_dt:
            return {"error": "End date must be after start date"}, 400
    except ValueError as e:
        return {"error": f"Invalid date format. Use YYYY-MM-DD. Error: {e}"}, 400
    
    # Validate window configuration
    if request.train_days <= 0 or request.test_days <= 0 or request.step_days <= 0:
        return {"error": "train_days, test_days, and step_days must be positive"}, 400
    
    if request.initial_equity <= 0:
        return {"error": "initial_equity must be positive"}, 400
    
    # Load daily candles
    try:
        print(f"[WALK-FORWARD] Loading daily candles for {request.symbol}")
        daily_candles_raw = load_daily_candles(request.symbol)
        daily_candles = csv_convert_to_replay_format(daily_candles_raw)
        
        if len(daily_candles) < 200:
            return {"error": f"Insufficient daily candles: {len(daily_candles)}. Need at least 200 for EMA(200)"}, 400
        
        print(f"[WALK-FORWARD] Loaded {len(daily_candles)} daily candles")
    except FileNotFoundError as e:
        return {"error": f"Daily data file not found: {str(e)}"}, 404
    except Exception as e:
        return {"error": f"Failed to load daily candles: {str(e)}"}, 400
    
    # Create walk-forward harness
    harness = WalkForwardHarness(
        initial_equity=request.initial_equity,
        train_days=request.train_days,
        test_days=request.test_days,
        step_days=request.step_days,
        train_bars=request.train_bars,
        test_bars=request.test_bars,
        step_bars=request.step_bars
    )
    
    # Run walk-forward
    try:
        results = harness.run_walkforward(
            symbol=request.symbol,
            candles=daily_candles,
            start_date=request.start_date,
            end_date=request.end_date,
            db_session=db,
            allowed_entry_regimes=request.allowed_entry_regimes
        )
        
        # Save artifacts if requested
        artifact_paths = {}
        if request.save_artifacts:
            try:
                artifact_paths = harness.save_artifacts(
                    output_dir=request.output_dir,
                    symbol=request.symbol
                )
                print(f"[WALK-FORWARD] Saved artifacts to {request.output_dir}")
            except Exception as e:
                print(f"[WALK-FORWARD] WARNING: Failed to save artifacts: {str(e)}")
        
        results["artifact_paths"] = artifact_paths
        
        return results
    
    except Exception as e:
        error_msg = f"Walk-forward backtest failed: {str(e)}"
        print(f"[WALK-FORWARD] ERROR: {error_msg}")
        return {"error": error_msg}, 500


class CostSensitivityRequest(BaseModel):
    """Cost sensitivity testing request."""
    symbol: str
    start_date: str  # YYYY-MM-DD format
    end_date: str  # YYYY-MM-DD format
    initial_equity: float = 100000.0
    base_slippage: float = 0.0002  # Base slippage (0.02%)
    slippage_multipliers: Optional[List[float]] = None  # Default: [0.5, 1.0, 2.0]
    commission_levels: Optional[List[List[float]]] = None  # Default: [[0.0, 0.0], [0.005, 1.0], [0.01, 2.0]]
    allowed_entry_regimes: Optional[List[str]] = None
    save_report: bool = True
    output_dir: str = "cost_sensitivity_results"


@app.post("/replay/cost_sensitivity")
async def cost_sensitivity_test(request: CostSensitivityRequest, db: Session = Depends(get_db)):
    """
    Run cost sensitivity test with different slippage and commission levels.
    
    This endpoint runs the same replay multiple times with different cost parameters
    to assess strategy robustness. Results are ranked by robustness score (return / drawdown).
    
    Example payload:
    {
      "symbol": "SPY",
      "start_date": "2023-01-01",
      "end_date": "2023-12-31",
      "initial_equity": 100000.0,
      "base_slippage": 0.0002,
      "slippage_multipliers": [0.5, 1.0, 2.0],
      "commission_levels": [[0.0, 0.0], [0.005, 1.0], [0.01, 2.0]],
      "save_report": true
    }
    
    Args:
        request: CostSensitivityRequest with symbol, dates, and cost parameters
    
    Returns:
        Dict with test results and summary report
    """
    # Validate symbol
    if request.symbol not in ALL_SYMBOLS:
        return {"error": f"Invalid symbol: {request.symbol}. Must be in allowed list"}, 400
    
    # Validate dates
    try:
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
        if end_dt <= start_dt:
            return {"error": "End date must be after start date"}, 400
    except ValueError as e:
        return {"error": f"Invalid date format. Use YYYY-MM-DD. Error: {e}"}, 400
    
    # Load daily candles
    try:
        print(f"[COST SENSITIVITY] Loading daily candles for {request.symbol}")
        daily_candles_raw = load_daily_candles(request.symbol)
        daily_candles_all = csv_convert_to_replay_format(daily_candles_raw)
        
        # Filter by date range
        daily_candles = filter_candles_by_date_range(
            daily_candles_all,
            request.start_date,
            request.end_date,
            candle_type="daily"
        )
        
        if len(daily_candles) < 200:
            return {"error": f"Insufficient daily candles: {len(daily_candles)}. Need at least 200 for EMA(200)"}, 400
        
        print(f"[COST SENSITIVITY] Loaded {len(daily_candles)} daily candles ({request.start_date} to {request.end_date})")
    except FileNotFoundError as e:
        return {"error": f"Daily data file not found: {str(e)}"}, 404
    except Exception as e:
        return {"error": f"Failed to load daily candles: {str(e)}"}, 400
    
    # Set defaults if not provided
    slippage_multipliers = request.slippage_multipliers or [0.5, 1.0, 2.0]
    commission_levels_raw = request.commission_levels or [[0.0, 0.0], [0.005, 1.0], [0.01, 2.0]]
    commission_levels = [tuple(level) for level in commission_levels_raw]  # Convert to tuples
    
    # Create cost sensitivity tester
    tester = CostSensitivityTester(
        initial_equity=request.initial_equity,
        base_slippage=request.base_slippage,
        slippage_multipliers=slippage_multipliers,
        commission_levels=commission_levels
    )
    
    # Run cost sensitivity test
    try:
        results = tester.run_cost_sensitivity_test(
            symbol=request.symbol,
            candles=daily_candles,
            db_session=db,
            allowed_entry_regimes=request.allowed_entry_regimes
        )
        
        # Save report if requested
        artifact_paths = {}
        if request.save_report:
            try:
                artifact_paths = tester.save_report(
                    output_dir=request.output_dir,
                    symbol=request.symbol
                )
                print(f"[COST SENSITIVITY] Saved report to {request.output_dir}")
            except Exception as e:
                print(f"[COST SENSITIVITY] WARNING: Failed to save report: {str(e)}")
        
        results["artifact_paths"] = artifact_paths
        
        return results
    
    except Exception as e:
        error_msg = f"Cost sensitivity test failed: {str(e)}"
        print(f"[COST SENSITIVITY] ERROR: {error_msg}")
        return {"error": error_msg}, 500


# ============================================================================
# LIVE TRADING LOOP ENDPOINTS (Phase 2)
# ============================================================================

from live_trading_loop import LiveTradingLoop
from decision_log import get_decisions, count_decisions
from database import DecisionLog

# Global live trading loop instance
_live_loop: Optional[LiveTradingLoop] = None


class LiveStartRequest(BaseModel):
    """Request to start the live trading loop."""
    symbol: str = "SPY"
    strategy_name: str = "ema_trend_v1"
    strategy_params: Optional[Dict] = None
    initial_equity: float = 100000.0
    interval_seconds: float = 60.0
    is_daily: bool = True
    data_provider: str = "auto"  # "auto" (Alpaca→Yahoo), "alpaca", or "yahoo"


@app.post("/live/start")
async def live_start(request: LiveStartRequest):
    """Start the live paper trading loop."""
    global _live_loop

    if _live_loop is not None and _live_loop.state.status.value == "running":
        return {"error": "Live trading loop is already running. Stop it first."}

    try:
        _live_loop = LiveTradingLoop(
            symbol=request.symbol,
            strategy_name=request.strategy_name,
            strategy_params=request.strategy_params,
            initial_equity=request.initial_equity,
            interval_seconds=request.interval_seconds,
            is_daily=request.is_daily,
            data_provider=request.data_provider,
        )
        result = await _live_loop.start()
        return {"message": "Live trading loop started", "status": result}
    except Exception as e:
        return {"error": str(e)}


@app.post("/live/stop")
async def live_stop():
    """Stop the live paper trading loop."""
    global _live_loop

    if _live_loop is None:
        return {"error": "No live trading loop running"}

    _live_loop.stop()
    return {"message": "Live trading loop stopped", "status": _live_loop.get_status()}


@app.get("/live/status")
async def live_status():
    """Get the current live trading loop status."""
    global _live_loop

    if _live_loop is None:
        return {"status": "idle", "message": "No live trading loop has been started"}

    return _live_loop.get_status()


@app.get("/decisions")
async def get_decision_logs(
    symbol: Optional[str] = None,
    replay_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    Get decision log entries.

    Query params:
      - symbol: Filter by symbol
      - replay_id: Filter by replay_id (omit for live trading only)
      - limit: Max records (default 100)
      - offset: Skip records (for pagination)
    """
    decisions = get_decisions(db, symbol=symbol, replay_id=replay_id,
                             limit=limit, offset=offset)
    total = count_decisions(db, symbol=symbol, replay_id=replay_id)
    return {
        "decisions": decisions,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# ============================================================================
# Phase 6 Sprint 4: Interactive Controls — Strategy, Params, Risk Limits
# ============================================================================

import strategy_registry as sr


@app.get("/strategies")
async def get_strategies():
    """List all registered strategies with metadata for the UI."""
    strategies = []
    for name in sr.list_strategies():
        config = sr.get_strategy(name)
        strategies.append({
            "name": config.name,
            "description": config.description,
            "default_params": config.default_params,
            "param_ranges": config.param_ranges,
        })
    return {"strategies": strategies}


class SwitchStrategyRequest(BaseModel):
    """Request to switch the live loop's active strategy."""
    strategy_name: str
    strategy_params: Optional[Dict] = None  # None → use strategy defaults
    force: bool = False  # Required if open position exists


@app.post("/live/switch-strategy")
async def live_switch_strategy(request: SwitchStrategyRequest):
    """
    Switch the live loop to a different strategy.
    Stops the current loop, creates a new one with the new strategy,
    and preserves broker state (equity, positions, risk tracking).
    """
    global _live_loop

    if _live_loop is None or _live_loop.state.status.value != "running":
        return {"error": "No live trading loop is currently running"}

    # Validate strategy exists
    try:
        new_config = sr.get_strategy(request.strategy_name)
    except KeyError as e:
        return {"error": str(e)}

    # Check for open positions
    old_broker = _live_loop.broker
    has_positions = len(old_broker.positions) > 0
    warning = None

    if has_positions and not request.force:
        symbols = list(old_broker.positions.keys())
        return {
            "error": f"Open position(s) in {', '.join(symbols)}. Set force=true to switch anyway, or close positions first."
        }

    if has_positions and request.force:
        symbols = list(old_broker.positions.keys())
        warning = f"Open position(s) in {', '.join(symbols)} preserved (not closed)"

    # Snapshot current state
    old_symbol = _live_loop.state.symbol
    old_interval = _live_loop.state.interval_seconds
    old_is_daily = _live_loop.state.is_daily
    old_data_provider = _live_loop.data_provider
    old_candle_history = list(_live_loop.candle_history)
    old_position_state = _live_loop.position_state

    # Stop current loop
    _live_loop.stop()
    # Brief wait for clean shutdown
    import asyncio as _asyncio
    await _asyncio.sleep(0.1)

    # Create new loop with new strategy
    try:
        new_loop = LiveTradingLoop(
            symbol=old_symbol,
            strategy_name=request.strategy_name,
            strategy_params=request.strategy_params,
            initial_equity=old_broker.equity,  # Use current equity as "initial"
            interval_seconds=old_interval,
            is_daily=old_is_daily,
            data_provider=old_data_provider,
        )

        # Transfer broker state
        new_loop.broker.cash = old_broker.cash
        new_loop.broker.equity = old_broker.equity
        new_loop.broker.positions = old_broker.positions
        new_loop.broker.pending_orders = old_broker.pending_orders
        new_loop.broker.high_water_mark = old_broker.high_water_mark
        new_loop.broker.weekly_pnl = old_broker.weekly_pnl
        new_loop.broker.monthly_pnl = old_broker.monthly_pnl
        new_loop.broker.daily_pnl = old_broker.daily_pnl
        new_loop.broker.daily_realized_pnl = old_broker.daily_realized_pnl
        new_loop.broker.consecutive_losing_days = old_broker.consecutive_losing_days
        new_loop.broker.trade_blocked = old_broker.trade_blocked
        new_loop.broker.pause_until_date = old_broker.pause_until_date

        # Transfer risk limit settings
        new_loop.broker.max_daily_loss_pct = old_broker.max_daily_loss_pct
        new_loop.broker.max_weekly_loss_pct = old_broker.max_weekly_loss_pct
        new_loop.broker.max_monthly_loss_pct = old_broker.max_monthly_loss_pct
        new_loop.broker.max_consecutive_losing_days = old_broker.max_consecutive_losing_days
        new_loop.broker.max_drawdown_from_hwm_pct = old_broker.max_drawdown_from_hwm_pct
        new_loop.broker.max_portfolio_exposure_pct = old_broker.max_portfolio_exposure_pct
        new_loop.broker.vol_adjustment_enabled = old_broker.vol_adjustment_enabled

        # Transfer position state and candle history
        new_loop.position_state = old_position_state
        new_loop.candle_history = old_candle_history

        _live_loop = new_loop
        result = await _live_loop.start()

        # Emit SSE event
        broadcaster.publish("strategy_switched", {
            "strategy_name": request.strategy_name,
            "strategy_params": _live_loop.state.strategy_params,
            "warning": warning,
        })

        return {
            "message": f"Strategy switched to {request.strategy_name}",
            "warning": warning,
            "status": result,
        }

    except Exception as e:
        return {"error": f"Failed to switch strategy: {str(e)}"}


class UpdateParamsRequest(BaseModel):
    """Request to update strategy parameters on the running loop."""
    params: Dict[str, Any]


@app.patch("/live/params")
async def live_update_params(request: UpdateParamsRequest):
    """
    Hot-reload strategy parameters on the running loop.
    Takes effect on the next evaluation cycle (no restart needed).
    """
    global _live_loop

    if _live_loop is None or _live_loop.state.status.value != "running":
        return {"error": "No live trading loop is currently running"}

    config = _live_loop._strategy_config
    default_params = config.default_params
    param_ranges = config.param_ranges

    # Validate keys
    invalid_keys = [k for k in request.params if k not in default_params]
    if invalid_keys:
        return {"error": f"Unknown parameter(s): {', '.join(invalid_keys)}. Valid: {', '.join(default_params.keys())}"}

    # Validate ranges
    range_errors = []
    for key, value in request.params.items():
        if key in param_ranges:
            valid_range = param_ranges[key]
            min_val = min(valid_range)
            max_val = max(valid_range)
            if isinstance(value, (int, float)) and not (min_val <= value <= max_val):
                range_errors.append(f"{key}: {value} not in [{min_val}, {max_val}]")
    if range_errors:
        return {"error": f"Parameter(s) out of range: {'; '.join(range_errors)}"}

    # Snapshot and update
    previous_params = dict(_live_loop.state.strategy_params)
    _live_loop.state.strategy_params.update(request.params)
    current_params = dict(_live_loop.state.strategy_params)

    # Emit SSE event
    broadcaster.publish("params_updated", {
        "previous_params": previous_params,
        "current_params": current_params,
    })

    return {
        "message": "Strategy params updated",
        "previous_params": previous_params,
        "current_params": current_params,
        "status": _live_loop.get_status(),
    }


class UpdateRiskLimitsRequest(BaseModel):
    """Request to update risk limits on the live broker."""
    max_daily_loss_pct: Optional[float] = None
    max_weekly_loss_pct: Optional[float] = None
    max_monthly_loss_pct: Optional[float] = None
    max_consecutive_losing_days: Optional[int] = None
    max_drawdown_from_hwm_pct: Optional[float] = None
    max_portfolio_exposure_pct: Optional[float] = None
    vol_adjustment_enabled: Optional[bool] = None


# Validation bounds for risk limits
_RISK_LIMIT_BOUNDS = {
    "max_daily_loss_pct": (0.001, 0.10),
    "max_weekly_loss_pct": (0.005, 0.20),
    "max_monthly_loss_pct": (0.01, 0.30),
    "max_consecutive_losing_days": (1, 20),
    "max_drawdown_from_hwm_pct": (0.01, 0.50),
    "max_portfolio_exposure_pct": (0.10, 1.0),
}


@app.patch("/risk/limits")
async def update_risk_limits(request: UpdateRiskLimitsRequest):
    """
    Update risk limits on the live broker. Only specified fields are updated.
    Changes take effect immediately on the next risk check.
    """
    global _live_loop

    if _live_loop is None:
        return {"error": "No live trading loop running"}

    broker = _live_loop.broker
    updates = request.dict(exclude_none=True)

    if not updates:
        return {"error": "No fields provided to update"}

    # Validate bounds
    errors = []
    for field, value in updates.items():
        if field in _RISK_LIMIT_BOUNDS:
            lo, hi = _RISK_LIMIT_BOUNDS[field]
            if not (lo <= value <= hi):
                errors.append(f"{field}: {value} not in [{lo}, {hi}]")
    if errors:
        return {"error": f"Validation failed: {'; '.join(errors)}"}

    # Snapshot previous
    previous_limits = {
        "max_daily_loss_pct": broker.max_daily_loss_pct,
        "max_weekly_loss_pct": broker.max_weekly_loss_pct,
        "max_monthly_loss_pct": broker.max_monthly_loss_pct,
        "max_consecutive_losing_days": broker.max_consecutive_losing_days,
        "max_drawdown_from_hwm_pct": broker.max_drawdown_from_hwm_pct,
        "max_portfolio_exposure_pct": broker.max_portfolio_exposure_pct,
        "vol_adjustment_enabled": broker.vol_adjustment_enabled,
    }

    # Apply updates
    for field, value in updates.items():
        setattr(broker, field, value)

    current_limits = {
        "max_daily_loss_pct": broker.max_daily_loss_pct,
        "max_weekly_loss_pct": broker.max_weekly_loss_pct,
        "max_monthly_loss_pct": broker.max_monthly_loss_pct,
        "max_consecutive_losing_days": broker.max_consecutive_losing_days,
        "max_drawdown_from_hwm_pct": broker.max_drawdown_from_hwm_pct,
        "max_portfolio_exposure_pct": broker.max_portfolio_exposure_pct,
        "vol_adjustment_enabled": broker.vol_adjustment_enabled,
    }

    # Emit SSE event
    broadcaster.publish("risk_limits_updated", {
        "previous_limits": previous_limits,
        "current_limits": current_limits,
    })

    return {
        "message": "Risk limits updated",
        "previous_limits": previous_limits,
        "current_limits": current_limits,
    }


# ── Alpaca Connection Status ──

@app.get("/data/alpaca/status")
async def alpaca_status():
    """Check if Alpaca API keys are configured and return masked key."""
    from market_data.alpaca_client import alpaca_service
    import os

    configured = alpaca_service.is_configured()
    api_key = os.environ.get("ALPACA_API_KEY", "")
    masked = None
    if api_key and len(api_key) >= 7:
        masked = api_key[:4] + "..." + api_key[-3:]
    elif api_key:
        masked = "***"

    return {
        "configured": configured,
        "api_key_masked": masked,
    }


@app.post("/data/alpaca/test")
async def alpaca_test_connection():
    """Test the Alpaca connection by fetching a SPY snapshot."""
    from market_data.alpaca_client import alpaca_service

    if not alpaca_service.is_configured():
        return {
            "success": False,
            "message": "Alpaca API keys not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.",
        }

    result = alpaca_service.fetch_snapshot("SPY")
    if result is not None:
        return {
            "success": True,
            "message": "Successfully connected to Alpaca and fetched SPY snapshot",
        }
    else:
        return {
            "success": False,
            "message": "Alpaca API returned no data. Check your API keys and network connection.",
        }


# ============================================================================
# Phase 6 Sprint 5: Notification Preferences Endpoints
# ============================================================================

from notification_service import notifier


class NotificationPrefsUpdate(BaseModel):
    """Request to update notification preferences. All fields optional for partial update."""
    email_enabled: Optional[bool] = None
    browser_enabled: Optional[bool] = None
    email_address: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: Optional[bool] = None
    min_severity: Optional[str] = None  # "info", "warning", "critical"
    email_categories: Optional[Dict[str, bool]] = None
    browser_categories: Optional[Dict[str, bool]] = None


@app.get("/notifications/prefs")
async def get_notification_prefs(db: Session = Depends(get_db)):
    """Get current notification preferences."""
    from database import NotificationPrefs
    prefs = db.query(NotificationPrefs).filter_by(id=1).first()
    if prefs is None:
        prefs = NotificationPrefs(id=1)
        db.add(prefs)
        db.commit()
        db.refresh(prefs)
    return prefs.to_dict()


@app.patch("/notifications/prefs")
async def update_notification_prefs(body: NotificationPrefsUpdate, db: Session = Depends(get_db)):
    """Update notification preferences. Partial updates supported."""
    from database import NotificationPrefs
    prefs = db.query(NotificationPrefs).filter_by(id=1).first()
    if prefs is None:
        prefs = NotificationPrefs(id=1)
        db.add(prefs)
        db.commit()
        db.refresh(prefs)

    # Apply scalar fields
    scalar_fields = [
        "email_enabled", "browser_enabled", "email_address",
        "smtp_host", "smtp_port", "smtp_user", "smtp_password",
        "smtp_use_tls", "min_severity",
    ]
    for field_name in scalar_fields:
        value = getattr(body, field_name)
        if value is not None:
            if isinstance(value, bool):
                setattr(prefs, field_name, 1 if value else 0)
            else:
                setattr(prefs, field_name, value)

    # Validate min_severity
    if body.min_severity is not None and body.min_severity not in ("info", "warning", "critical"):
        return {"error": f"Invalid min_severity: {body.min_severity}. Must be info, warning, or critical."}

    # Apply per-category toggles
    if body.email_categories:
        for cat, enabled in body.email_categories.items():
            col_name = f"email_{cat}"
            if hasattr(prefs, col_name):
                setattr(prefs, col_name, 1 if enabled else 0)

    if body.browser_categories:
        for cat, enabled in body.browser_categories.items():
            col_name = f"browser_{cat}"
            if hasattr(prefs, col_name):
                setattr(prefs, col_name, 1 if enabled else 0)

    prefs.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(prefs)

    # Reload cached prefs in notification service
    notifier.load_prefs()

    return {"message": "Notification preferences updated", "prefs": prefs.to_dict()}


@app.post("/notifications/test-email")
async def test_email_notification():
    """Send a test email to verify SMTP configuration."""
    notifier.load_prefs()  # Ensure latest config
    result = await notifier.send_test_email()
    return result


@app.get("/notifications/history")
async def get_notification_history(limit: int = 20):
    """Get recent notification history (from monitoring alerts)."""
    alerts = monitor.get_recent_alerts(limit=limit)
    return {"notifications": alerts, "total": len(alerts)}


# ============================================================================
# Phase 6 Sprint 3: Server-Sent Events (SSE) Endpoint
# ============================================================================

from starlette.responses import StreamingResponse
from sse_broadcaster import broadcaster


async def _sse_event_generator():
    """
    Async generator that yields SSE-formatted events.
    Subscribes to the broadcaster and yields events as they arrive.
    Sends heartbeat every 15 seconds to keep the connection alive.
    """
    queue = broadcaster.subscribe()
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
                yield broadcaster.format_sse(event)
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                hb = broadcaster.heartbeat_event()
                yield broadcaster.format_sse(hb)
    except asyncio.CancelledError:
        pass
    finally:
        broadcaster.unsubscribe(queue)


@app.get("/events")
async def sse_events():
    """
    Server-Sent Events endpoint for real-time frontend updates.

    Event types:
    - equity_update: Equity, PnL, positions after each evaluation
    - trade_executed: BUY/EXIT/STOP_LOSS actions
    - decision_logged: Strategy evaluation decisions
    - alert_fired: System alerts from monitoring
    - loop_status: Live loop start/stop status changes
    - heartbeat: Keep-alive (every 15s)
    """
    return StreamingResponse(
        _sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Phase 5: Monitoring & System Health Endpoints
# ============================================================================

@app.get("/monitoring/status")
async def monitoring_status():
    """Get comprehensive system health snapshot."""
    return monitor.get_system_status()


@app.get("/monitoring/alerts")
async def monitoring_alerts(limit: int = 20):
    """Get recent system alerts."""
    return {"alerts": monitor.get_recent_alerts(limit=limit)}


@app.get("/monitoring/api-stats")
async def monitoring_api_stats(last_minutes: int = 5):
    """Get API performance stats."""
    return monitor.get_api_stats(last_minutes=last_minutes)


# ============================================================================
# Phase 6 Sprint 1: Risk State & Metrics Aggregation Endpoints
# ============================================================================

@app.get("/risk/state")
async def get_risk_state():
    """
    Get comprehensive risk control state from live PaperBroker.
    Exposes weekly_pnl, monthly_pnl, HWM drawdown, consecutive losing days,
    all breach flags, and configured limits.
    """
    global _live_loop

    if _live_loop is None or _live_loop.broker is None:
        # No live loop running — return default state
        return {
            "any_breached": False,
            "daily_breached": False,
            "weekly_breached": False,
            "monthly_breached": False,
            "hwm_drawdown_breached": False,
            "consecutive_days_breached": False,
            "is_paused": False,
            "details": {
                "daily_pnl": 0.0,
                "weekly_pnl": 0.0,
                "monthly_pnl": 0.0,
                "high_water_mark": 100000.0,
                "drawdown_from_hwm_pct": 0.0,
                "consecutive_losing_days": 0,
                "pause_until_date": None,
            },
            "limits": {
                "max_daily_loss_pct": 0.02,
                "max_weekly_loss_pct": 0.05,
                "max_monthly_loss_pct": 0.10,
                "max_drawdown_from_hwm_pct": 0.15,
                "max_consecutive_losing_days": 5,
                "max_portfolio_exposure_pct": 0.80,
            },
            "equity": 100000.0,
            "trade_blocked": False,
        }

    broker = _live_loop.broker
    symbol = _live_loop.state.symbol

    # Get current price for equity calculation
    current_prices = {}
    if symbol and symbol in broker.positions:
        # Use last known price from candle history
        if _live_loop.candle_history:
            current_prices[symbol] = _live_loop.candle_history[-1].get("close", broker.positions[symbol].entry_price)
        else:
            current_prices[symbol] = broker.positions[symbol].entry_price

    import time as _time
    risk_state = broker.check_all_risk_controls(current_prices, int(_time.time()))

    # Add limits and equity info
    risk_state["limits"] = {
        "max_daily_loss_pct": broker.max_daily_loss_pct,
        "max_weekly_loss_pct": broker.max_weekly_loss_pct,
        "max_monthly_loss_pct": broker.max_monthly_loss_pct,
        "max_drawdown_from_hwm_pct": broker.max_drawdown_from_hwm_pct,
        "max_consecutive_losing_days": broker.max_consecutive_losing_days,
        "max_portfolio_exposure_pct": broker.max_portfolio_exposure_pct,
    }
    risk_state["equity"] = round(broker.equity, 2)
    risk_state["trade_blocked"] = broker.trade_blocked

    return risk_state


@app.get("/metrics/monthly-returns")
async def get_monthly_returns(db: Session = Depends(get_db)):
    """
    Aggregate monthly returns from live trading trades.
    Returns a list of {year, month, return_pct, pnl, trade_count}.
    """
    from collections import defaultdict

    live_trades = db.query(Trade).filter(
        Trade.replay_id.is_(None),
        Trade.exit_time.isnot(None),
    ).all()

    monthly: Dict[str, Dict] = {}

    for trade in live_trades:
        if trade.exit_time is None or trade.pnl is None:
            continue
        exit_time = trade.exit_time
        if isinstance(exit_time, str):
            try:
                exit_time = datetime.fromisoformat(exit_time)
            except (ValueError, TypeError):
                continue

        key = f"{exit_time.year}-{exit_time.month:02d}"
        if key not in monthly:
            monthly[key] = {"year": exit_time.year, "month": exit_time.month, "pnl": 0.0, "trade_count": 0}
        monthly[key]["pnl"] += trade.pnl or 0
        monthly[key]["trade_count"] += 1

    # Compute return_pct (approximate: pnl relative to 100k base)
    result = []
    for key in sorted(monthly.keys()):
        data = monthly[key]
        data["return_pct"] = round(data["pnl"] / 100000.0 * 100, 2)
        data["pnl"] = round(data["pnl"], 2)
        result.append(data)

    return {"monthly_returns": result}


@app.get("/metrics/trade-distribution")
async def get_trade_distribution(db: Session = Depends(get_db)):
    """
    Return P&L per closed trade for histogram visualization.
    Returns list of {id, pnl, symbol, exit_time, is_win}.
    """
    live_trades = db.query(Trade).filter(
        Trade.replay_id.is_(None),
        Trade.exit_time.isnot(None),
    ).order_by(Trade.exit_time).all()

    result = []
    for trade in live_trades:
        pnl = trade.pnl or 0
        result.append({
            "id": trade.id,
            "pnl": round(pnl, 2),
            "symbol": trade.symbol,
            "exit_time": str(trade.exit_time) if trade.exit_time else None,
            "is_win": pnl > 0,
        })

    return {"trades": result, "count": len(result)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
