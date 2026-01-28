# Historical Data CSV Files

This directory contains CSV files with historical OHLCV candle data for backtesting.

## CSV Format

Each CSV file must have the following columns:
- `date`: Date in YYYY-MM-DD format (e.g., "2023-01-03")
- `open`: Opening price (float)
- `high`: High price (float)
- `low`: Low price (float)
- `close`: Closing price (float)
- `volume`: Volume (integer)

## Example

```csv
date,open,high,low,close,volume
2023-01-03,130.28,130.90,124.17,125.07,112117500
2023-01-04,126.89,128.66,125.08,126.36,89113600
```

## Usage

Use the `POST /replay/start_csv` endpoint to run a replay with CSV data:

```json
{
  "symbol": "AAPL",
  "csv_path": "data d/AAPL_daily.csv"
}
```

The CSV path can be:
- Relative to the `backend/` directory (e.g., `"../data d/AAPL_daily.csv"`)
- Absolute path (e.g., `"/path/to/data d/AAPL_daily.csv"`)

**Note:** Daily CSV files are located in the `data d/` folder. For 5-minute candles, use the `data 5/` folder.

## Sample Files

- `AAPL_daily.csv`: Sample AAPL daily data (62 candles, Jan-Mar 2023)
- `SPY_daily.csv`: SPY daily data from Stooq (5,260 candles, 2005-2026)

## Data Sources

- **Raw data**: Stored in `../../raw_data/` (untouched source files)
- **Daily CSV files**: Stored in `../../data d/` folder (for daily replays)
- **5-minute CSV files**: Stored in `../../data 5/` folder (for intraday replays)
- Files in these directories are ready for direct use by the replay system

## Folder Structure

- `data d/`: Contains daily OHLCV CSV files
- `data 5/`: Contains 5-minute OHLCV CSV files

## Notes

- Dates must be in ascending order
- All OHLCV values must be valid numbers
- Missing values will be skipped
- Minimum 50 candles required for replay (EMA(50) needs 50 candles)
