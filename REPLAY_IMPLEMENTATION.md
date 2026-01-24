# Historical Replay (Backtest) Engine Implementation

## Overview

The replay engine enables deterministic, candle-by-candle replay of historical market data through the full trading pipeline. It reuses all existing components (strategy, paper broker, metrics) to ensure identical behavior to live trading.

## Architecture

### Components

1. **ReplayEngine** (`backend/replay_engine.py`)
   - Orchestrates replay execution
   - Processes candles sequentially
   - Uses same functions as live trading

2. **Database Schema Updates**
   - Added `replay_id` column to `Signal`, `Trade`, and `EquityCurve` tables
   - `replay_id = None` for live trading data
   - `replay_id = UUID` for replay data

3. **API Endpoints**
   - `POST /replay/start` - Start a replay
   - `GET /replay/status` - Get replay status
   - `GET /replay/results` - Get replay results (trades, equity curve, metrics)

4. **Frontend Panel**
   - Symbol selection
   - Start replay button
   - Progress display
   - Results summary

## Key Features

### Deterministic Processing
- Candles processed one at a time, sequentially
- No shortcuts, vectorization, or batch P&L
- Time advances one candle at a time
- Uses accumulated history (no lookahead bias)

### State Management
- Clean reset between replays
- Isolated broker state per replay
- Position state reset for each replay
- Multiple replays can run back-to-back

### Safeguards
- Prevents concurrent replay/live trading
- Validates symbol in allowed universe
- Validates candle ordering and timestamps
- Requires minimum 50 candles (for EMA(50))

### Persistence
- Replay trades stored with `replay_id`
- Replay equity curve stored with `replay_id`
- Replay signals stored with `replay_id`
- Results can be retrieved by `replay_id`

## Usage

### Backend API

**Start Replay:**
```bash
POST /replay/start
{
  "symbol": "AAPL",
  "candles": [
    {"time": 1234567890, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000000},
    ...
  ]
}
```

**Get Results:**
```bash
GET /replay/results?replay_id=<uuid>
```

### Frontend

1. Select symbol from dropdown
2. Click "Start Replay"
3. System fetches historical candles and runs replay
4. Results displayed in panel

## Verification Checklist

- [x] Replay results reproducible (same input = same output)
- [x] Metrics identical across repeated runs
- [x] Stop-loss triggers correctly in replay
- [x] Kill switch triggers correctly in replay
- [x] No lookahead bias (uses only past data)
- [x] Replay data isolated from live trading data
- [x] Multiple replays can run sequentially
- [x] State properly reset between replays

## Implementation Details

### Replay Flow

1. **Initialize**: Reset broker, position state, candle history
2. **Validate**: Check symbol, candle count, ordering
3. **Process**: For each candle:
   - Calculate indicators (using accumulated history)
   - Evaluate strategy
   - Check stop-losses
   - Execute trades through broker
   - Check kill switch
   - Update equity curve
   - Store results with replay_id
4. **Complete**: Return results

### No Lookahead Guarantee

The replay engine ensures no lookahead bias by:
- Using `current_history = self.candle_history[:self.current_candle_index]`
- Only processing candles up to current index
- Indicators calculated from past data only
- Strategy evaluated on past data only

### Component Reuse

All existing components are reused:
- `PaperBroker` - Same execution logic
- `ema_trend_v1` - Same strategy logic
- `compute_metrics` - Same metrics calculation
- `ema`, `atr` - Same indicator calculations

## Files Modified

- `backend/replay_engine.py` - New file
- `backend/database.py` - Added `replay_id` columns
- `backend/main.py` - Added replay endpoints, updated queries to filter by replay_id
- `frontend/app/page.tsx` - Added replay panel

## Future Enhancements

- Background task processing for long replays
- Progress updates via WebSocket
- Replay comparison views
- Replay speed controls
- Multiple symbol replays
- Replay scheduling
