# Metrics and Evaluation Engine Implementation

## Overview

This document describes the implementation of a professional-grade trading performance metrics engine that computes strategy-agnostic statistics from persisted trade and equity data.

## Architecture

### Design Principles

1. **Strategy-Agnostic**: Metrics are computed from data only, not strategy logic
2. **Persisted Data Only**: All metrics computed from database tables (trades, equity_curve)
3. **Recomputable**: Metrics can be recomputed at any time from historical data
4. **Immutable Snapshots**: MetricsSnapshot is a frozen dataclass, immutable once created
5. **Future-Proof**: Works with paper trading, real brokers, multiple strategies, AI supervision

### Data Sources

- **trades table**: Closed trades with entry/exit times, prices, P&L
- **equity_curve table**: Historical equity values over time

## Metrics Implemented

### Core Metrics

1. **Total Return (%)**: `(equity_end - equity_start) / equity_start * 100`
2. **Net P&L**: Sum of all realized P&L
3. **Win Rate**: `(winning trades / total trades) * 100`
4. **Loss Rate**: `(losing trades / total trades) * 100`
5. **Profit Factor**: `gross_wins / gross_losses` (infinity if no losses)
6. **Expectancy per Trade**: `net_pnl / trade_count`
7. **Average Win**: Mean of all winning trades
8. **Average Loss**: Mean of all losing trades

### Risk Metrics

1. **Max Drawdown (Absolute)**: Largest peak-to-trough decline in equity
2. **Max Drawdown (%)**: Max drawdown as percentage of peak equity
3. **Max Consecutive Losses**: Longest streak of losing trades
4. **Max Consecutive Wins**: Longest streak of winning trades
5. **Exposure (%)**: Percentage of time in market

### Time-Based Metrics

1. **Trades per Day**: Average number of trades per trading day
2. **Avg Trade Duration**: Mean duration of closed trades in hours
3. **Profitable Days (%)**: Percentage of days with positive returns

### Risk-Adjusted Metric

1. **Sharpe Proxy**: `mean(daily_returns) / std(daily_returns)`
   - Assumes risk-free rate = 0
   - Proxy for true Sharpe ratio

## Implementation Details

### Metrics Module (`backend/metrics.py`)

**MetricsSnapshot Class**:
- Frozen dataclass (immutable)
- Contains all computed metrics
- `to_dict()` method for JSON serialization
- Handles edge cases (zero trades, zero losses, etc.)

**compute_metrics() Function**:
- Main entry point
- Takes trades and equity_curve lists
- Returns MetricsSnapshot
- Handles empty data gracefully

**Helper Functions**:
- `_compute_total_return()`: From equity curve
- `_compute_max_drawdown()`: Peak-to-trough analysis
- `_compute_max_consecutive_losses()`: Streak analysis
- `_compute_max_consecutive_wins()`: Streak analysis
- `_compute_exposure_pct()`: Time-weighted exposure
- `_compute_trades_per_day()`: Date-based calculation
- `_compute_avg_trade_duration()`: Duration statistics
- `_compute_profitable_days_pct()`: Daily return analysis
- `_compute_sharpe_proxy()`: Risk-adjusted return

### API Endpoints

**GET /metrics**:
- Returns all computed metrics
- Includes metadata (start_date, end_date, trade_count, equity_start, equity_end)
- Returns empty metrics if no trades exist

**GET /equity-curve**:
- Returns equity curve data for charting
- Ordered by timestamp
- Limit parameter for performance

### Frontend Implementation

**Performance Panel**:
- Three sections: Returns, Risk, Consistency
- Equity curve line chart using Lightweight Charts
- Color-coded metrics (green for positive, red for negative)
- Auto-refreshes every 10 seconds

**Equity Curve Chart**:
- Line chart showing equity over time
- Updates automatically when new data arrives
- Responsive design

## Edge Case Handling

### Zero Trades
- Returns empty MetricsSnapshot with all zeros
- No division errors
- Graceful degradation

### Zero Losses
- Profit factor = infinity (handled as null in JSON)
- Average loss = 0.0
- No division by zero errors

### Zero Wins
- Profit factor = 0.0
- Average win = 0.0
- Win rate = 0.0

### Insufficient Data
- Sharpe proxy returns 0.0 if < 2 data points
- All time-based metrics handle empty data
- No NaNs or infinities in output

## Example Metrics Output

```json
{
  "metadata": {
    "start_date": "2024-01-15T10:30:00",
    "end_date": "2024-01-20T15:45:00",
    "trade_count": 25,
    "equity_start": 100000.00,
    "equity_end": 105250.00
  },
  "core_metrics": {
    "total_return_pct": 5.25,
    "net_pnl": 5250.00,
    "win_rate": 64.00,
    "loss_rate": 36.00,
    "profit_factor": 2.15,
    "expectancy_per_trade": 210.00,
    "average_win": 450.00,
    "average_loss": -200.00
  },
  "risk_metrics": {
    "max_drawdown_absolute": 1500.00,
    "max_drawdown_pct": 1.50,
    "max_consecutive_losses": 3,
    "max_consecutive_wins": 5,
    "exposure_pct": 45.20
  },
  "time_metrics": {
    "trades_per_day": 5.00,
    "avg_trade_duration_hours": 2.50,
    "profitable_days_pct": 60.00
  },
  "risk_adjusted": {
    "sharpe_proxy": 1.25
  }
}
```

## Verification Checklist

- [x] Metrics computed from persisted data only
- [x] Strategy-agnostic implementation
- [x] Handles zero trades gracefully
- [x] Handles zero-loss cases without division errors
- [x] No NaNs or infinities in output
- [x] Metrics match trade history
- [x] Drawdown reacts correctly to losses
- [x] Metrics stable across restarts
- [x] Equity curve renders correctly
- [x] All metrics displayed in frontend

## Files Created/Modified

**New Files**:
- `backend/metrics.py`: Metrics computation module
- `METRICS_IMPLEMENTATION.md`: This document

**Modified Files**:
- `backend/main.py`: Added GET /metrics and GET /equity-curve endpoints
- `frontend/app/page.tsx`: Added Performance panel with equity curve chart

## Future Extensibility

The metrics engine is designed to work with:
- Real broker execution (same data structure)
- Multiple strategies (aggregate metrics)
- AI supervision (same metrics, different interpretation)
- Additional metrics (extend MetricsSnapshot dataclass)

No changes to metrics computation logic needed when switching execution modes.
