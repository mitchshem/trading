# Strategy Engine Implementation Summary

## Overview

This document describes the implementation of the deterministic indicator and strategy engine that evaluates candle-close events and generates trading signals.

## New Files Created

### Backend

1. **`backend/indicators.py`**
   - Pure Python implementations (no TA-Lib)
   - `ema(prices, period)` - Exponential Moving Average
   - `atr(highs, lows, closes, period)` - Average True Range

2. **`backend/strategy.py`**
   - `PositionState` class - Tracks position state per symbol
   - `ema_trend_v1()` - EMA trend-following strategy function
   - Strategy rules:
     - **BUY**: EMA(20) crosses above EMA(50) AND close > EMA(50) AND no position
     - **EXIT**: close < EMA(50) OR price <= entry_price - (2 * ATR)

3. **`backend/database.py`**
   - SQLite database setup using SQLAlchemy
   - `Signal` model for storing trading signals
   - Database initialized on FastAPI startup

### Modified Files

1. **`backend/main.py`**
   - Added imports for indicators, strategy, and database
   - Added position state tracking per symbol
   - Added candle history tracking per symbol
   - Modified WebSocket to evaluate strategy on candle close
   - Added `evaluate_strategy_on_candle_close()` function
   - Added `GET /signals` endpoint
   - Signals are stored in database when generated

2. **`frontend/app/page.tsx`**
   - Added signals state and fetching
   - Added signals panel below chart
   - Added chart markers for BUY/EXIT signals
   - Signals panel displays: timestamp, signal, price, reason
   - Markers update automatically when new signals are generated

## Database Schema

```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    symbol VARCHAR,
    signal VARCHAR,  -- "BUY", "EXIT", or "HOLD"
    price FLOAT,
    reason VARCHAR
);
```

## Strategy Evaluation Flow

1. **Candle Close Event**: When a new candle is generated (representing a closed 5-minute candle)
2. **Update History**: Add new candle to symbol's candle history
3. **Calculate Indicators**: 
   - EMA(20) and EMA(50) from close prices
   - ATR(14) from highs, lows, and closes
4. **Evaluate Strategy**: Call `ema_trend_v1()` with:
   - Current candle history
   - Indicator values
   - Current position state
5. **Store Signal**: If signal is BUY or EXIT, store in database
6. **Update Position State**: Update position state based on signal
7. **Send to Client**: Include signal in WebSocket message (if generated)

## Example Signal Output

```json
{
  "id": 1,
  "timestamp": "2024-01-15T10:30:00",
  "symbol": "AAPL",
  "signal": "BUY",
  "price": 150.25,
  "reason": "EMA(20) crossed above EMA(50), close 150.25 > EMA(50) 149.80"
}
```

## Key Implementation Details

### Deterministic Behavior
- All indicators use pure Python (no external libraries)
- Strategy rules are fully deterministic
- Same candle data always produces same signals
- Position state is tracked per symbol in memory

### Candle Close Evaluation
- Strategy is evaluated **only** when a new candle arrives (candle close)
- Historical candles are maintained per symbol (last 500)
- Indicators require minimum data (50 candles for EMA(50))

### Position Management
- One position max per symbol (enforced by strategy logic)
- Position state tracked in memory (`position_states` dict)
- Entry price and time stored for stop-loss calculation

### Signal Storage
- All signals (BUY, EXIT) stored in SQLite database
- HOLD signals are not stored (only BUY/EXIT)
- Signals queryable via `GET /signals?symbol=XXX&limit=100`

## Frontend Integration

### Signals Panel
- Fetches signals on symbol change
- Displays in table format with timestamp, signal, price, reason
- Auto-refreshes when new signal arrives via WebSocket

### Chart Markers
- BUY signals: Green arrow pointing up, below bar
- EXIT signals: Red arrow pointing down, above bar
- Markers positioned at signal timestamp
- Markers update automatically when signals change

## Testing Checklist

- [x] Indicators compute correctly (EMA, ATR)
- [x] Strategy evaluates on candle close
- [x] Signals generated for BUY conditions
- [x] Signals generated for EXIT conditions
- [x] Signals stored in database
- [x] GET /signals endpoint works
- [x] Frontend displays signals panel
- [x] Chart markers appear for BUY/EXIT
- [x] No trades executed (signals only)

## Constraints Met

✅ No paper broker yet  
✅ No real trades  
✅ No ML or LLM logic  
✅ Signals fully deterministic and reproducible  
✅ One strategy only (ema_trend_v1)  
✅ Long-only  
✅ One open position max per symbol  
✅ Strategy evaluates only on candle close  
