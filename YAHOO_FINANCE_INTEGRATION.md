# Yahoo Finance Historical Data Integration

## Overview

Yahoo Finance integration provides historical 5-minute OHLCV candle data for replay/backtesting. The data is normalized and validated before being consumed by the ReplayEngine.

## Architecture

### Components

1. **market_data/yahoo.py**
   - `fetch_yahoo_candles()` - Fetches historical data from Yahoo Finance
   - `normalize_yahoo_candles()` - Converts DataFrame to canonical format
   - `validate_candles()` - Validates ordering, OHLC relationships, data quality
   - `convert_to_replay_format()` - Converts to ReplayEngine input format

2. **API Endpoint**
   - `POST /data/yahoo/fetch` - Fetches and returns normalized candles

3. **Integration**
   - ReplayEngine accepts candles from Yahoo Finance
   - Source logging: `source="yahoo_finance"` tracked in replay runs

## Data Format

### Input
- Symbol (e.g., "AAPL")
- Start date: "YYYY-MM-DD"
- End date: "YYYY-MM-DD"

### Output (Normalized Candle)
```python
{
    "timestamp": datetime(timezone.utc),  # UTC timezone-aware
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": int
}
```

### ReplayEngine Format
```python
{
    "time": int,  # Unix timestamp (seconds)
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": int
}
```

## Validation Rules

1. **Ordering**: Candles must be strictly ordered by timestamp (ascending)
2. **Missing Values**: Rows with NaN OHLCV values are dropped
3. **OHLC Relationships**: 
   - `low <= open <= high`
   - `low <= close <= high`
4. **Data Types**: OHLC are floats, volume is int
5. **Empty Data**: Raises error if no valid candles returned

## Safeguards

1. **Symbol Validation**: Only Nasdaq-100 and Dow-30 symbols allowed
2. **Date Range Limit**: Maximum 1 year (365 days)
3. **Source Logging**: All replays track data source (`yahoo_finance`)

## Usage

### API Endpoint

```bash
POST /data/yahoo/fetch
{
  "symbol": "AAPL",
  "start_date": "2023-01-01",
  "end_date": "2023-06-01"
}
```

### Response

```json
{
  "symbol": "AAPL",
  "source": "yahoo_finance",
  "candle_count": 1234,
  "start_timestamp": 1672531200,
  "end_timestamp": 1685577600,
  "start_date": "2023-01-01",
  "end_date": "2023-06-01",
  "sample_first": {...},
  "sample_last": {...},
  "candles": [...]  // Full list for replay
}
```

### Integration with Replay

1. Fetch candles from Yahoo Finance:
   ```python
   response = POST /data/yahoo/fetch
   candles = response["candles"]
   ```

2. Run replay with fetched candles:
   ```python
   POST /replay/start
   {
     "symbol": "AAPL",
     "candles": candles,
     "source": "yahoo_finance"  // Optional, logged automatically
   }
   ```

## Example Workflow

1. **Fetch Data**:
   ```bash
   curl -X POST http://localhost:8000/data/yahoo/fetch \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "AAPL",
       "start_date": "2023-01-01",
       "end_date": "2023-01-31"
     }'
   ```

2. **Run Replay**:
   ```bash
   curl -X POST http://localhost:8000/replay/start \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "AAPL",
       "candles": [/* from step 1 */]
     }'
   ```

3. **Get Results**:
   ```bash
   curl http://localhost:8000/replay/results?replay_id=<uuid>
   ```

## Verification Checklist

- [x] Candles ordered by timestamp (ascending)
- [x] Timestamps in UTC timezone
- [x] Replay produces trades from Yahoo Finance data
- [x] Metrics compute without error
- [x] Symbol validation enforced
- [x] Date range limit enforced (1 year max)
- [x] Source logging implemented
- [x] Missing values filtered
- [x] OHLC relationships validated

## Dependencies

- `yfinance==0.2.28` - Yahoo Finance data fetching
- `pandas==2.1.3` - DataFrame handling

## Files Modified

- `backend/market_data/yahoo.py` - New file
- `backend/market_data/__init__.py` - New file
- `backend/main.py` - Added `/data/yahoo/fetch` endpoint
- `backend/replay_engine.py` - Added source tracking
- `backend/requirements.txt` - Added yfinance and pandas

## Notes

- Yahoo Finance 5-minute data has limitations:
  - Typically available for last 60 days
  - Market hours only (no pre-market/after-hours by default)
  - May have gaps during weekends/holidays
- Timezone handling: yfinance returns timezone-naive timestamps in market time (EST/EDT). We normalize to UTC.
- No caching: Each request fetches fresh data from Yahoo Finance.
