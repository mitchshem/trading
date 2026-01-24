# Paper Trading System - MVP

A scoped MVP for a paper-trading system that mirrors a small subset of TradingView features.

## Project Structure

```
.
├── backend/                    # Python FastAPI backend
│   ├── main.py                # API server with endpoints and WebSocket
│   ├── database.py            # SQLite database models and setup
│   ├── indicators.py          # Technical indicators (EMA, ATR)
│   ├── strategy.py            # Trading strategy (ema_trend_v1)
│   ├── requirements.txt        # Python dependencies
│   ├── trading_signals.db     # SQLite database (created on first run)
│   └── README.md              # Backend-specific docs (if any)
├── frontend/                   # Next.js frontend
│   ├── app/                   # Next.js app directory
│   │   ├── globals.css        # Global styles
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx           # Main chart page with signals panel
│   ├── next.config.js         # Next.js configuration
│   ├── package.json           # Node.js dependencies
│   └── tsconfig.json          # TypeScript configuration
├── run.sh                     # Single-command startup script
└── README.md                  # This file
```

## Quick Start

### Prerequisites
- Python 3.8+ with pip
- Node.js 18+ with npm

### Running the Application

#### Option 1: Single Command Script (Easiest)

From the project root:
```bash
chmod +x run.sh
./run.sh
```

This will:
- Set up Python virtual environment (if needed)
- Install backend dependencies
- Install frontend dependencies
- Start both backend and frontend services

Press `Ctrl+C` to stop both services.

#### Option 2: Run Separately (Recommended for Development)

**Terminal 1 - Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

#### Option 3: Using concurrently

```bash
# Install concurrently globally (if not already installed)
npm install -g concurrently

# From project root
cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cd ..
concurrently "cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000" "cd frontend && npm run dev"
```

### Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Features Implemented

### Backend Endpoints

1. **GET /symbols** - Returns list of allowed symbols (Dow 30 + Nasdaq-100)
2. **GET /candles?symbol=XXX&limit=500** - Returns synthetic historical candle data
3. **GET /signals?symbol=XXX&limit=100** - Returns trading signals for a symbol
4. **WebSocket /ws** - Streams synthetic candle updates every 5 seconds and evaluates strategy on candle close

### Backend Modules

1. **indicators.py** - Technical indicators:
   - `ema(prices, period)` - Exponential Moving Average
   - `atr(highs, lows, closes, period)` - Average True Range

2. **strategy.py** - Trading strategy:
   - `ema_trend_v1()` - EMA trend-following strategy (long-only)
   - `PositionState` - Tracks position state per symbol

3. **database.py** - SQLite database:
   - `Signal` model - Stores trading signals
   - Database initialized on startup

### Strategy Rules (ema_trend_v1)

**ENTRY (BUY):**
- EMA(20) crosses above EMA(50)
- Candle close > EMA(50)
- No open position

**EXIT:**
- Candle close < EMA(50)
- OR Price <= entry_price - (2 * ATR) [Stop loss]

### Frontend

- Symbol dropdown populated from `/symbols` endpoint
- Candlestick chart using TradingView Lightweight Charts
- Fetches historical candles on symbol change
- Subscribes to WebSocket for real-time candle updates
- Connection status indicator
- **Signals panel** below chart showing:
  - Timestamp
  - Signal type (BUY/EXIT)
  - Price
  - Reason
- **Chart markers** for BUY (green arrow up) and EXIT (red arrow down) signals

## Verification Checklist

### Basic Functionality
- [ ] Backend starts successfully on port 8000
- [ ] Frontend starts successfully on port 3000
- [ ] Symbol dropdown shows list of symbols
- [ ] Selecting a symbol loads candlestick chart with historical data
- [ ] Chart displays properly with green/red candles
- [ ] Connection indicator shows "Connected" (green dot)
- [ ] New candles appear on chart every ~5 seconds (synthetic updates)
- [ ] Switching symbols updates the chart and reconnects WebSocket

### Strategy & Signals
- [ ] Candles update in real-time via WebSocket
- [ ] Indicators compute correctly (EMA, ATR)
- [ ] Signals appear in the signals panel below chart
- [ ] BUY signals show green arrow markers on chart
- [ ] EXIT signals show red arrow markers on chart
- [ ] Signals are stored in database (check `backend/trading_signals.db`)
- [ ] GET /signals endpoint returns signal history
- [ ] No trades executed (signals only, no order execution)

## Database Schema

The SQLite database (`trading_signals.db`) contains a `signals` table with:
- `id` (Integer, Primary Key)
- `timestamp` (DateTime)
- `symbol` (String)
- `signal` (String) - "BUY", "EXIT", or "HOLD"
- `price` (Float)
- `reason` (String) - Explanation of the signal

## Next Steps (Not Implemented Yet)

- Paper trading broker with simulated fills
- Trade execution and position management
- Global kill-switch and max daily loss guardrails
- Additional indicators (SMA, RSI, etc.)

## Notes

- All candle data is currently synthetic/deterministic (no real market data)
- WebSocket simulates 5-minute candle updates every 5 seconds for testing
- Symbols are limited to Dow 30 + Nasdaq-100 (deduplicated)
- No authentication or user accounts in MVP
- Strategy evaluates **only on candle close** (when new candle arrives)
- Strategy is **fully deterministic** and reproducible
- **Long-only** strategy (no short positions)
- **One open position max** per symbol
- Signals are generated but **no trades are executed** (paper broker not yet implemented)