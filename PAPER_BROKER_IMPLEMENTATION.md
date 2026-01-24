# Paper Trading Broker Implementation Summary

## Overview

This document describes the implementation of the paper trading execution engine that converts BUY/EXIT signals into simulated trades with P&L tracking and risk controls.

## New Files Created

### Backend

1. **`backend/paper_broker.py`**
   - `PaperBroker` class - Main broker implementation
   - `Position` class - Tracks open positions
   - Features:
     - Account equity tracking (starts at $100,000)
     - Position management (one per symbol)
     - Realized and unrealized P&L calculation
     - Slippage application (0.02%)
     - Position sizing (0.5% risk per trade)
     - Risk engine with daily loss limit (2% of equity)
     - Kill switch functionality

### Modified Files

1. **`backend/database.py`**
   - Added `Trade` model for storing executed trades
   - Added `EquityCurve` model for tracking equity over time

2. **`backend/main.py`**
   - Integrated paper broker
   - Modified `evaluate_strategy_on_candle_close()` to route signals through broker
   - Added `GET /account` endpoint
   - Added `GET /trades` endpoint
   - Trades and equity curve stored in database

3. **`frontend/app/page.tsx`**
   - Added Account panel showing equity, daily P&L, open positions
   - Added Trades table showing all executed trades
   - Added position entry/exit lines on chart
   - Auto-refreshes account and trades every 5 seconds

## Key Features

### Position Sizing

- **Risk per trade**: 0.5% of current equity
- **Stop distance**: 2 * ATR
- **Shares calculation**: `floor((risk_amount) / stop_distance)`

Example:
- Equity: $100,000
- Risk: $500 (0.5%)
- Stop distance: $2.00 (2 * ATR of $1.00)
- Shares: floor(500 / 2.00) = 250 shares

### Execution Rules

**BUY Signal:**
1. Check no open position exists
2. Check daily loss limit not breached
3. Calculate position size based on risk
4. Apply slippage (0.02%)
5. Open position at fill price
6. Store trade in database

**EXIT Signal:**
1. Check position exists
2. Apply slippage (0.02%)
3. Close position at fill price
4. Calculate and realize P&L
5. Update trade record in database

### Risk Engine

- **Daily P&L tracking**: Realized + Unrealized
- **Max daily loss**: 2% of current equity
- **Kill switch**: When daily loss limit breached:
  - Close all open positions
  - Block new trades for the day
  - Emit "KILL_SWITCH" event

### Database Schema

**trades table:**
```sql
- id (Integer, Primary Key)
- symbol (String)
- entry_time (DateTime)
- entry_price (Float)
- exit_time (DateTime, nullable)
- exit_price (Float, nullable)
- shares (Integer)
- pnl (Float, nullable) - Realized P&L
- reason (String, nullable) - Exit reason
```

**equity_curve table:**
```sql
- id (Integer, Primary Key)
- timestamp (DateTime)
- equity (Float)
```

## Example Trade Lifecycle

1. **Signal Generated**: BUY signal at $150.00
2. **Position Sizing**: 
   - Equity: $100,000
   - Risk: $500 (0.5%)
   - ATR: $1.00
   - Stop distance: $2.00
   - Shares: 250
3. **Execution**:
   - Fill price: $150.03 (with 0.02% slippage)
   - Stop price: $148.03
   - Trade stored in database
4. **Position Open**: 
   - Entry: $150.03
   - Shares: 250
   - Unrealized P&L tracked in real-time
5. **Exit Signal**: EXIT signal at $152.00
6. **Exit Execution**:
   - Fill price: $151.97 (with 0.02% slippage)
   - P&L: ($151.97 - $150.03) * 250 = $485.00
   - Trade updated in database
   - Position closed

## API Endpoints

### GET /account
Returns account summary:
```json
{
  "equity": 100485.00,
  "daily_pnl": 485.00,
  "daily_realized_pnl": 485.00,
  "open_positions": [],
  "trade_blocked": false,
  "max_daily_loss": 2009.70
}
```

### GET /trades?symbol=XXX&limit=100
Returns executed trades:
```json
{
  "trades": [
    {
      "id": 1,
      "symbol": "AAPL",
      "entry_time": "2024-01-15T10:30:00",
      "entry_price": 150.03,
      "exit_time": "2024-01-15T11:00:00",
      "exit_price": 151.97,
      "shares": 250,
      "pnl": 485.00,
      "reason": "Close below EMA(50)"
    }
  ]
}
```

## Frontend Features

### Account Panel
- Equity display
- Daily P&L (color-coded: green/red)
- Max daily loss
- Trading status (ACTIVE/BLOCKED)
- Open positions table with:
  - Symbol, shares, entry price
  - Current price, unrealized P&L
  - Stop price

### Trades Table
- All executed trades
- Entry/exit times and prices
- Shares and P&L
- Exit reason
- Color-coded P&L (green/red)

### Chart Enhancements
- Entry price line (green, horizontal)
- Exit price line (red, horizontal, if closed)
- Lines update automatically when trades execute

## Verification Checklist

- [x] Paper broker module created
- [x] Position sizing logic implemented (0.5% risk, 2*ATR stop)
- [x] Slippage applied (0.02%)
- [x] Risk engine with daily loss limit (2%)
- [x] Kill switch functionality
- [x] Trades stored in database
- [x] Equity curve tracked
- [x] GET /account endpoint
- [x] GET /trades endpoint
- [x] Signals routed through broker
- [x] Account panel in frontend
- [x] Trades table in frontend
- [x] Position lines on chart
- [x] No trades executed when blocked
- [x] Kill switch closes all positions

## Constraints Met

✅ Paper trading only (no real broker APIs)  
✅ Deterministic fills (slippage is fixed)  
✅ No leverage  
✅ Long-only  
✅ One position per symbol  
✅ Risk controls enforced in code  
✅ Daily loss limit (2% of equity)  
✅ Kill switch functionality  

## Testing Scenarios

1. **Normal Trade Flow**:
   - BUY signal → Position opened → EXIT signal → Position closed → P&L realized

2. **Daily Loss Limit**:
   - Multiple losing trades → Daily P&L reaches -2% → Kill switch triggered → All positions closed → Trading blocked

3. **Position Sizing**:
   - Verify shares calculated correctly based on risk and stop distance
   - Verify position size adjusts with equity changes

4. **Slippage**:
   - Verify buy fills at price * 1.0002
   - Verify sell fills at price * 0.9998

5. **One Position Per Symbol**:
   - Verify second BUY signal for same symbol is ignored if position exists
