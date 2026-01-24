# Browser Verification Checklist

This document helps verify that the trading system runs correctly in a browser.

## Quick Start

1. **Start Backend** (Terminal 1):
   ```bash
   cd backend
   source venv/bin/activate  # or create venv first
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   ✅ **Check**: You should see startup message with backend URL and endpoints listed

2. **Start Frontend** (Terminal 2):
   ```bash
   cd frontend
   npm run dev
   ```
   
   ✅ **Check**: You should see "ready started server on 0.0.0.0:3000"

3. **Open Browser**: http://localhost:3000

## Verification Steps

### 1. Backend Endpoints (Test via Browser or curl)

Open http://localhost:8000/docs in browser to see interactive API docs, or test these endpoints:

- ✅ **GET /symbols** - Should return list of symbols
  ```bash
  curl http://localhost:8000/symbols
  ```

- ✅ **GET /replay/history** - Should return empty array or existing replays
  ```bash
  curl http://localhost:8000/replay/history
  ```

- ✅ **GET /replay/status** - Should return replay status
  ```bash
  curl http://localhost:8000/replay/status
  ```

- ✅ **GET /account** - Should return account state
  ```bash
  curl http://localhost:8000/account
  ```

### 2. Frontend UI Elements

When you open http://localhost:3000, verify:

- ✅ **Symbol Dropdown** - Shows list of symbols (AAPL, MSFT, etc.)
- ✅ **Candlestick Chart** - Renders when symbol is selected
- ✅ **Account Panel** - Shows equity, daily P&L, open positions
- ✅ **Trades Table** - Shows trade history (may be empty initially)
- ✅ **Performance Metrics** - Shows metrics (may show zeros initially)
- ✅ **Replay Panel** - Shows symbol selector, date inputs, "Start Replay" button
- ✅ **Trading Signals Table** - Shows signals (may be empty initially)

### 3. Replay Functionality

1. **Select Symbol** in Replay panel (e.g., AAPL)
2. **Select Start Date** (e.g., 2023-01-01)
3. **Select End Date** (e.g., 2023-12-31)
4. **Click "Start Replay"**

✅ **Check**:
- Button shows "Running..." while replay is in progress
- After completion, replay status appears showing:
  - Status: completed
  - Candles Processed: [number]
  - Final Equity: $[amount]
- Replay Results Summary appears with:
  - Total Return: [%]
  - Net P&L: $[amount]
  - Win Rate: [%]
  - Trades: [number]

### 4. Backend Console Output

When replay completes, check backend terminal for:

```
==================================================
REPLAY COMPLETE
==================================================
Symbol: AAPL
Date range: 2023-01-01 → 2023-12-31
Candles processed: [number]
Trades executed: [number]
Final equity: $[amount]
Net P&L: $[amount]
Max drawdown: [%]
Sharpe proxy: [number]
Replay ID: [uuid]
==================================================
```

### 5. API Endpoints After Replay

After running a replay, test these endpoints:

- ✅ **GET /replay/history** - Should now include the completed replay
  ```bash
  curl http://localhost:8000/replay/history
  ```
  
  Should return JSON with `replay_summaries` array containing:
  - replay_id
  - symbol
  - start_date, end_date
  - candle_count, trade_count
  - final_equity, net_pnl
  - max_drawdown_pct, sharpe_proxy
  - timestamp_completed

- ✅ **GET /replay/results?replay_id=XXX** - Should return full replay results
  ```bash
  curl "http://localhost:8000/replay/results?replay_id=[YOUR_REPLAY_ID]"
  ```
  
  Should return JSON with:
  - symbol, start_date, end_date
  - metrics (full metrics snapshot)
  - trades (array of trades)
  - equity_curve (array of equity points)

## Troubleshooting

### Backend won't start
- Check Python version: `python3 --version` (needs 3.8+)
- Check if port 8000 is in use: `lsof -i :8000`
- Check virtual environment is activated
- Check dependencies installed: `pip list`

### Frontend won't start
- Check Node version: `node --version` (needs 18+)
- Check if port 3000 is in use: `lsof -i :3000`
- Check dependencies installed: `npm list`
- Clear Next.js cache: `rm -rf frontend/.next`

### CORS errors in browser console
- Verify backend CORS is configured for `http://localhost:3000`
- Check backend is running on port 8000
- Check frontend is calling `http://localhost:8000` (not https)

### Replay fails
- Check backend console for error messages
- Verify symbol is in allowed list (Dow-30 or Nasdaq-100)
- Verify date range is valid (start < end, max 1 year)
- Check Yahoo Finance data is available for the date range

### Charts don't render
- Check browser console for JavaScript errors
- Verify WebSocket connection is established (check connection indicator)
- Try refreshing the page
- Check that symbol is selected

## Expected Behavior

- **Charts**: Should render candlestick data with green/red candles
- **Replay**: Should process candles and generate trades
- **Metrics**: Should compute correctly from trades and equity curve
- **Results**: Should be deterministic (same inputs → same outputs)

## Success Criteria

✅ Backend starts and shows startup message
✅ Frontend starts and shows "ready" message
✅ Browser opens http://localhost:3000 without errors
✅ Charts render when symbol is selected
✅ Replay can be started and completes successfully
✅ Replay results appear in UI
✅ API endpoints return expected data
✅ Console output matches persisted database values
