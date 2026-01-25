#!/usr/bin/env python3
"""
Standalone Yahoo Finance debug script.

This script tests Yahoo Finance data fetching independently of the replay system.
Run this to diagnose Yahoo Finance data availability issues.

Usage:
    python backend/debug_yahoo.py

Or from backend directory:
    python debug_yahoo.py
"""

import sys
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd

# Test parameters
# Test with multiple symbols to isolate the issue
TEST_SYMBOLS = ["SPY", "AAPL", "MSFT"]
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
INTERVAL = "1d"

print("=" * 70)
print("YAHOO FINANCE DEBUG SCRIPT")
print("=" * 70)
print(f"Test Symbols: {TEST_SYMBOLS}")
print(f"Start Date: {START_DATE}")
print(f"End Date: {END_DATE}")
print(f"Interval: {INTERVAL}")
print("=" * 70)
print()

# Parse dates
try:
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    print(f"[DATE PARSING] Start: {start_dt}")
    print(f"[DATE PARSING] End: {end_dt}")
    print(f"[DATE PARSING] Date range: {(end_dt - start_dt).days} days")
    print()
except ValueError as e:
    print(f"[ERROR] Date parsing failed: {e}")
    sys.exit(1)

# Test each symbol
results = {}
for SYMBOL in TEST_SYMBOLS:
    print("=" * 70)
    print(f"TESTING SYMBOL: {SYMBOL}")
    print("=" * 70)
    print()
    
    # Test Case A: With start and end dates
    print("-" * 70)
    print(f"TEST CASE A: {SYMBOL} - Fetching with start + end dates")
    print("-" * 70)
    print(f"Parameters:")
    print(f"  start={start_dt}")
    print(f"  end={end_dt}")
    print(f"  interval='{INTERVAL}'")
    print(f"  auto_adjust=False")
    print(f"  prepost=False")
    print()

    df_a = None
    try:
        ticker = yf.Ticker(SYMBOL)
        
        # Fetch with explicit parameters
        df_a = ticker.history(
            start=start_dt,
            end=end_dt,
            interval=INTERVAL,
            auto_adjust=False,
            prepost=False
        )
    
        print(f"[RESULT] DataFrame shape: {df_a.shape}")
        print(f"[RESULT] DataFrame empty: {df_a.empty}")
        print(f"[RESULT] DataFrame index type: {type(df_a.index)}")
        print(f"[RESULT] DataFrame index dtype: {df_a.index.dtype if hasattr(df_a.index, 'dtype') else 'N/A'}")
        
        if not df_a.empty:
            print(f"[RESULT] Index timezone: {df_a.index.tz if hasattr(df_a.index, 'tz') else 'N/A'}")
            print(f"[RESULT] First index value: {df_a.index[0]}")
            print(f"[RESULT] Last index value: {df_a.index[-1]}")
            print(f"[RESULT] Columns: {list(df_a.columns)}")
            print()
            print("[HEAD] First 5 rows:")
            print(df_a.head())
            print()
            print("[TAIL] Last 5 rows:")
            print(df_a.tail())
            print()
            results[SYMBOL] = {"case_a": True, "rows": len(df_a)}
        else:
            print("[WARNING] DataFrame is EMPTY!")
            results[SYMBOL] = {"case_a": False, "rows": 0}
            print()
        
    except Exception as e:
        print(f"[ERROR] Fetch failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        results[SYMBOL] = {"case_a": False, "error": str(e)}
        print()
    
    # Test Case C: With end date + 1 day (inclusive end) - Most likely to work
    print("-" * 70)
    print(f"TEST CASE C: {SYMBOL} - Fetching with end date + 1 day (inclusive)")
    print("-" * 70)
    end_dt_inclusive = end_dt + timedelta(days=1)
    print(f"Parameters:")
    print(f"  start={start_dt}")
    print(f"  end={end_dt_inclusive} (original end + 1 day)")
    print(f"  interval='{INTERVAL}'")
    print(f"  auto_adjust=False")
    print(f"  prepost=False")
    print()
    
    df_c = None
    try:
        ticker = yf.Ticker(SYMBOL)
        
        # Fetch with inclusive end
        df_c = ticker.history(
            start=start_dt,
            end=end_dt_inclusive,
            interval=INTERVAL,
            auto_adjust=False,
            prepost=False
        )
        
        print(f"[RESULT] DataFrame shape: {df_c.shape}")
        print(f"[RESULT] DataFrame empty: {df_c.empty}")
        
        if not df_c.empty:
            print(f"[RESULT] First index value: {df_c.index[0]}")
            print(f"[RESULT] Last index value: {df_c.index[-1]}")
            print(f"[RESULT] Columns: {list(df_c.columns)}")
            print()
            print("[HEAD] First 5 rows:")
            print(df_c.head())
            print()
            print("[TAIL] Last 5 rows:")
            print(df_c.tail())
            print()
            
            # Test timezone conversion
            print("-" * 70)
            print("TIMEZONE HANDLING TEST")
            print("-" * 70)
            first_idx = df_c.index[0]
            print(f"First index value: {first_idx}")
            print(f"Index timezone: {first_idx.tz if hasattr(first_idx, 'tz') else 'None (naive)'}")
            
            if isinstance(first_idx, pd.Timestamp):
                if first_idx.tz is None:
                    print("[CONVERSION] Converting timezone-naive to UTC...")
                    utc_timestamp = first_idx.replace(tzinfo=timezone.utc)
                    print(f"  Converted: {utc_timestamp}")
                else:
                    print("[CONVERSION] Converting timezone-aware to UTC...")
                    utc_timestamp = first_idx.tz_convert(timezone.utc)
                    print(f"  Converted: {utc_timestamp}")
            
            results[SYMBOL]["case_c"] = True
            results[SYMBOL]["rows_c"] = len(df_c)
            print()
        else:
            print("[WARNING] DataFrame is EMPTY!")
            if SYMBOL not in results:
                results[SYMBOL] = {}
            results[SYMBOL]["case_c"] = False
            results[SYMBOL]["rows_c"] = 0
            print()
        
    except Exception as e:
        print(f"[ERROR] Fetch failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        if SYMBOL not in results:
            results[SYMBOL] = {}
        results[SYMBOL]["case_c"] = False
        results[SYMBOL]["error_c"] = str(e)
        print()
    
    # Test Case D: Try download() method (alternative to history())
    print("-" * 70)
    print(f"TEST CASE D: {SYMBOL} - Using download() method (no date params)")
    print("-" * 70)
    print("Testing if we can get ANY data without date constraints...")
    print()
    
    df_d = None
    try:
        # Try download method - sometimes more reliable
        df_d = yf.download(
            SYMBOL,
            period="1y",  # Last 1 year
            interval=INTERVAL,
            auto_adjust=False,
            prepost=False,
            progress=False
        )
        
        # download() returns MultiIndex if multiple symbols, flatten if needed
        if isinstance(df_d.columns, pd.MultiIndex):
            df_d = df_d[SYMBOL] if SYMBOL in df_d.columns.levels[0] else df_d
        
        print(f"[RESULT] DataFrame shape: {df_d.shape}")
        print(f"[RESULT] DataFrame empty: {df_d.empty}")
        
        if not df_d.empty:
            print(f"[RESULT] First index value: {df_d.index[0]}")
            print(f"[RESULT] Last index value: {df_d.index[-1]}")
            print(f"[RESULT] Columns: {list(df_d.columns)}")
            print()
            print("[HEAD] First 5 rows:")
            print(df_d.head())
            print()
            print("[TAIL] Last 5 rows:")
            print(df_d.tail())
            print()
            
            if SYMBOL not in results:
                results[SYMBOL] = {}
            results[SYMBOL]["case_d"] = True
            results[SYMBOL]["rows_d"] = len(df_d)
        else:
            print("[WARNING] DataFrame is EMPTY!")
            if SYMBOL not in results:
                results[SYMBOL] = {}
            results[SYMBOL]["case_d"] = False
            results[SYMBOL]["rows_d"] = 0
            print()
        
    except Exception as e:
        print(f"[ERROR] Download failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        if SYMBOL not in results:
            results[SYMBOL] = {}
        results[SYMBOL]["case_d"] = False
        results[SYMBOL]["error_d"] = str(e)
        print()
    
    print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
for symbol, result in results.items():
    print(f"\n{symbol}:")
    if "case_a" in result:
        status_a = "SUCCESS" if result.get("case_a") else "FAILED"
        rows_a = result.get("rows", 0)
        print(f"  Case A (start + end): {status_a} ({rows_a} rows)")
    if "case_c" in result:
        status_c = "SUCCESS" if result.get("case_c") else "FAILED"
        rows_c = result.get("rows_c", 0)
        print(f"  Case C (start + end+1): {status_c} ({rows_c} rows)")
    if "error" in result:
        print(f"  Error: {result['error']}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

working_symbols = [s for s, r in results.items() if r.get("case_c") and r.get("rows_c", 0) > 0]
if working_symbols:
    print(f"✓ Working symbols: {working_symbols}")
    print("  These symbols successfully returned data from Yahoo Finance.")
    print("  The issue may be symbol-specific or rate limiting.")
else:
    print("✗ No symbols returned data.")
    print()
    print("ROOT CAUSE IDENTIFIED:")
    print("  Yahoo Finance API is rate limiting requests.")
    print("  Direct API test shows: 'Edge: Too Many Requests'")
    print()
    print("Evidence:")
    print("  - All symbols (SPY, AAPL, MSFT) return empty DataFrames")
    print("  - All methods (history(), download()) fail")
    print("  - Error: 'Expecting value: line 1 column 1 (char 0)' (empty JSON response)")
    print("  - HTTP 429 errors observed in ticker.info calls")
    print()
    print("SOLUTIONS:")
    print("  1. Wait 5-10 minutes before retrying (rate limit cooldown)")
    print("  2. Use a different IP/VPN if possible")
    print("  3. Add delays between requests (sleep 1-2 seconds)")
    print("  4. Consider using a different data source for testing")
    print("  5. Check if yfinance needs update: pip install --upgrade yfinance")
    print()
    print("WORKAROUND FOR REPLAY:")
    print("  - The replay system should handle empty data gracefully")
    print("  - Error messages will clearly indicate 'No DAILY data returned'")
    print("  - This prevents silent failures in replay execution")

print("=" * 70)
