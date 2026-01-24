# Replay Results Inspection Examples

This document shows example outputs from the replay inspection endpoints.

## 1. Console Output Example

When a replay completes, you'll see output like this in the console:

```
==================================================
REPLAY COMPLETE
==================================================
Symbol: AAPL
Date range: 2022-01-01 → 2022-12-31
Candles processed: 19,584
Trades executed: 42
Final equity: $108,342.17
Net P&L: $8,342.17
Max drawdown: -12.40%
Sharpe proxy: 1.21
Replay ID: a3c2f9c1-8b4e-4f3a-9d2c-1e5f6a7b8c9d
==================================================
```

This output is deterministic and matches the persisted database values.

## 2. GET /replay/history Example

**Request:**
```bash
curl http://localhost:8000/replay/history?limit=10
```

**Response:**
```json
{
  "replay_summaries": [
    {
      "id": 1,
      "replay_id": "a3c2f9c1-8b4e-4f3a-9d2c-1e5f6a7b8c9d",
      "symbol": "AAPL",
      "start_date": "2022-01-01",
      "end_date": "2022-12-31",
      "source": "yahoo_finance",
      "candle_count": 19584,
      "trade_count": 42,
      "final_equity": 108342.17,
      "net_pnl": 8342.17,
      "max_drawdown_pct": -12.40,
      "max_drawdown_absolute": -12450.00,
      "sharpe_proxy": 1.21,
      "timestamp_completed": "2024-01-15T10:30:45.123456+00:00"
    },
    {
      "id": 2,
      "replay_id": "b4d3e0d2-9c5f-5g4b-0e3d-2f6g7b8c9d0e",
      "symbol": "MSFT",
      "start_date": "2022-06-01",
      "end_date": "2022-12-31",
      "source": "yahoo_finance",
      "candle_count": 15672,
      "trade_count": 28,
      "final_equity": 95678.50,
      "net_pnl": -4321.50,
      "max_drawdown_pct": -8.75,
      "max_drawdown_absolute": -8750.00,
      "sharpe_proxy": 0.45,
      "timestamp_completed": "2024-01-15T09:15:22.654321+00:00"
    }
  ],
  "count": 2
}
```

## 3. GET /replay/results Example

**Request:**
```bash
curl "http://localhost:8000/replay/results?replay_id=a3c2f9c1-8b4e-4f3a-9d2c-1e5f6a7b8c9d"
```

**Response:**
```json
{
  "replay_id": "a3c2f9c1-8b4e-4f3a-9d2c-1e5f6a7b8c9d",
  "symbol": "AAPL",
  "start_date": "2022-01-01",
  "end_date": "2022-12-31",
  "source": "yahoo_finance",
  "metrics": {
    "metadata": {
      "start_date": "2022-01-01T00:00:00+00:00",
      "end_date": "2022-12-31T23:59:59+00:00",
      "trade_count": 42,
      "equity_start": 100000.00,
      "equity_end": 108342.17
    },
    "core_metrics": {
      "total_return_pct": 8.34,
      "net_pnl": 8342.17,
      "win_rate": 0.62,
      "loss_rate": 0.38,
      "profit_factor": 1.85,
      "expectancy_per_trade": 198.62,
      "average_win": 1250.50,
      "average_loss": -675.25
    },
    "risk_metrics": {
      "max_drawdown_absolute": -12450.00,
      "max_drawdown_pct": -12.40,
      "max_consecutive_losses": 4,
      "max_consecutive_wins": 7,
      "exposure_pct": 65.50
    },
    "time_based_metrics": {
      "trades_per_day": 0.12,
      "avg_trade_duration_hours": 48.5,
      "profitable_days_pct": 52.30
    },
    "risk_adjusted": {
      "sharpe_proxy": 1.21
    }
  },
  "trades": [
    {
      "id": 1,
      "symbol": "AAPL",
      "entry_time": "2022-01-05T14:30:00+00:00",
      "entry_price": 175.50,
      "exit_time": "2022-01-10T16:00:00+00:00",
      "exit_price": 182.25,
      "shares": 100,
      "pnl": 675.00,
      "reason": "EXIT_SIGNAL",
      "replay_id": "a3c2f9c1-8b4e-4f3a-9d2c-1e5f6a7b8c9d"
    }
    // ... more trades ...
  ],
  "equity_curve": [
    {
      "timestamp": "2022-01-01T00:00:00+00:00",
      "equity": 100000.00
    },
    {
      "timestamp": "2022-01-05T14:30:00+00:00",
      "equity": 100000.00
    }
    // ... more equity points ...
  ]
}
```

## Verification Checklist

✅ **Replay Summary Endpoint**
- Returns all required fields: replay_id, symbol, start_date, end_date, candle_count, trade_count, final_equity, net_pnl, max_drawdown_pct, sharpe_proxy, timestamp_completed
- Sorted by timestamp_completed descending (newest first)
- Easy to copy/paste and share

✅ **Single Replay Result Endpoint**
- Includes symbol and date range at top level
- Metrics snapshot is easy to find at top level
- Includes full trade history and equity curve for detailed analysis

✅ **Console Output**
- Human-readable format with clear separators
- Shows all key metrics: symbol, date range, candles, trades, equity, P&L, drawdown, Sharpe, replay_id
- Deterministic and matches persisted database values

✅ **Database Persistence**
- ReplaySummary table stores all metrics
- One row per completed replay
- net_pnl and sharpe_proxy are persisted

## Usage

1. **Run a replay** via `POST /replay/start` with symbol and date range
2. **View console output** - summary appears automatically when replay completes
3. **Query history** - `GET /replay/history` to see all completed replays
4. **Get details** - `GET /replay/results?replay_id=XXX` for full metrics and trades

All results are deterministic and can be shared externally for review.
