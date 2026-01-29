#!/usr/bin/env python3
"""
Run parameter sensitivity harness end-to-end test.
Fetches SPY candles and runs sensitivity analysis.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import csv
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from market_data.stooq_loader import load_daily_candles
from parameter_sensitivity_harness import ParameterSensitivityHarness
from database import get_db, Base, engine
from utils import calculate_candle_timestamps

# Initialize database
Base.metadata.create_all(bind=engine)


def main():
    print("=" * 80)
    print("PARAMETER SENSITIVITY HARNESS - END-TO-END TEST")
    print("=" * 80)
    
    # Step 1: Fetch candles for SPY (5+ years)
    print("\n[STEP 1] Loading SPY daily candles...")
    symbol = "SPY"
    
    try:
        candles_raw = load_daily_candles(symbol)
        print(f"[STEP 1] Loaded {len(candles_raw)} candles")
        
        if len(candles_raw) < 1000:
            print(f"[STEP 1] WARNING: Only {len(candles_raw)} candles loaded. Need at least 1000 for 5+ years.")
        
        # Convert to replay format (ensure time is Unix timestamp)
        # The loader returns candles with 'timestamp' key (datetime) or 'time' key
        candles = []
        for candle in candles_raw:
            # Handle timestamp field (datetime object)
            timestamp_val = candle.get("timestamp") or candle.get("time")
            
            if isinstance(timestamp_val, datetime):
                time_ts = int(timestamp_val.timestamp())
                # Calculate open_time and close_time
                open_time_dt, close_time_dt = calculate_candle_timestamps(timestamp_val, is_daily=True)
                open_time_ts = int(open_time_dt.timestamp())
                close_time_ts = int(close_time_dt.timestamp())
            elif isinstance(timestamp_val, (int, float)):
                time_ts = int(timestamp_val)
                open_time_ts = int(candle.get("open_time", time_ts))
                close_time_ts = int(candle.get("close_time", time_ts))
            else:
                # Skip if no valid timestamp
                continue
            
            candles.append({
                "time": time_ts,
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": int(candle.get("volume", 0)),
                "open_time": open_time_ts,
                "close_time": close_time_ts
            })
        
        print(f"[STEP 1] Converted {len(candles)} candles to replay format")
        
        # Determine date range
        if candles:
            first_ts = candles[0]["time"]
            last_ts = candles[-1]["time"]
            first_date = datetime.fromtimestamp(first_ts, tz=timezone.utc).date()
            last_date = datetime.fromtimestamp(last_ts, tz=timezone.utc).date()
            print(f"[STEP 1] Date range: {first_date} to {last_date}")
            
            # Use a 5-year range (or available data)
            # Start from a date that gives us at least 5 years
            # But ensure we have enough data for walk-forward windows
            end_date = last_date
            # Start earlier to ensure we have enough data for multiple windows
            # Need: train_days (252) + test_days (63) + some buffer = ~350 days minimum per window
            # For multiple windows, start 6 years back
            start_date = datetime(end_date.year - 6, 1, 1).date()
            
            # Filter candles to date range
            filtered_candles = []
            for candle in candles:
                candle_date = datetime.fromtimestamp(candle["time"], tz=timezone.utc).date()
                if start_date <= candle_date <= end_date:
                    filtered_candles.append(candle)
            
            print(f"[STEP 1] Filtered to {len(filtered_candles)} candles from {start_date} to {end_date}")
            candles = filtered_candles
            
    except Exception as e:
        print(f"[STEP 1] ERROR: Failed to load candles: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    if not candles or len(candles) < 500:
        print(f"[STEP 1] ERROR: Insufficient candles ({len(candles)}). Need at least 500.")
        return 1
    
    # Step 2: Run parameter sensitivity harness
    print("\n[STEP 2] Running parameter sensitivity harness...")
    print(f"[STEP 2] Configuration:")
    print(f"  - train_days: 252")
    print(f"  - test_days: 63")
    print(f"  - step_days: 21")
    print(f"  - initial_equity: 100000")
    print(f"  - Total combinations: 81 (3×3×3×3)")
    
    harness = ParameterSensitivityHarness(
        initial_equity=100000.0,
        train_days=252,
        test_days=63,
        step_days=21
    )
    
    # Get database session
    db = next(get_db())
    
    try:
        results = harness.run_sensitivity_test(
            symbol=symbol,
            candles=candles,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            db_session=db
        )
        
        print(f"\n[STEP 2] Sensitivity test completed:")
        print(f"  - Status: {results.get('status')}")
        print(f"  - Total combinations: {results.get('total_combinations')}")
        print(f"  - Completed: {results.get('completed_combinations')}")
        print(f"  - Failed: {results.get('failed_combinations')}")
        print(f"  - Elapsed time: {results.get('elapsed_time_seconds')} seconds")
        
        if results.get('determinism_violations'):
            print(f"\n[STEP 2] WARNING: {len(results['determinism_violations'])} determinism violations detected!")
            for violation in results['determinism_violations']:
                print(f"  - {violation}")
        else:
            print(f"\n[STEP 2] ✓ No determinism violations")
        
    except Exception as e:
        print(f"[STEP 2] ERROR: Sensitivity test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        db.close()
        return 1
    
    # Step 3: Save results
    print("\n[STEP 3] Saving results...")
    output_dir = "parameter_sensitivity_results"
    
    try:
        artifact_paths = harness.save_results(output_dir)
        print(f"[STEP 3] Results saved:")
        print(f"  - CSV: {artifact_paths.get('results_csv')}")
        print(f"  - JSON: {artifact_paths.get('summary_json')}")
    except Exception as e:
        print(f"[STEP 3] ERROR: Failed to save results: {str(e)}")
        import traceback
        traceback.print_exc()
        db.close()
        return 1
    
    # Step 4: Sanity checks
    print("\n[STEP 4] Running sanity checks...")
    
    csv_path = Path(output_dir) / "parameter_sensitivity_results.csv"
    json_path = Path(output_dir) / "parameter_sensitivity_summary.json"
    
    # Check CSV exists and has correct row count
    if not csv_path.exists():
        print(f"[STEP 4] ERROR: CSV file not found: {csv_path}")
        db.close()
        return 1
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"[STEP 4] CSV row count: {len(rows)}")
    if len(rows) != 81:
        print(f"[STEP 4] WARNING: Expected 81 rows, got {len(rows)}")
    else:
        print(f"[STEP 4] ✓ CSV has correct row count (81)")
    
    # Check JSON exists
    if not json_path.exists():
        print(f"[STEP 4] ERROR: JSON file not found: {json_path}")
        db.close()
        return 1
    
    with open(json_path, 'r') as f:
        summary = json.load(f)
    
    print(f"[STEP 4] ✓ JSON file exists")
    
    # Check determinism violations
    violations = summary.get('determinism_violations', [])
    if violations:
        print(f"[STEP 4] WARNING: {len(violations)} determinism violations in summary")
    else:
        print(f"[STEP 4] ✓ No determinism violations in summary")
    
    # Check OOS trade counts
    trade_counts = [int(row.get('num_oos_trades', 0)) for row in rows]
    non_zero_count = sum(1 for count in trade_counts if count > 0)
    print(f"[STEP 4] Combinations with trades: {non_zero_count}/81 ({non_zero_count/81*100:.1f}%)")
    
    if non_zero_count < 20:
        print(f"[STEP 4] WARNING: Only {non_zero_count} combinations produced trades (expected more)")
    else:
        print(f"[STEP 4] ✓ Meaningful portion of combinations produced trades")
    
    # Step 5: Print top 10 and bottom 10
    print("\n[STEP 5] Top 10 and Bottom 10 combinations by OOS return_pct:")
    print("=" * 80)
    
    # Sort by return_pct
    sorted_rows = sorted(rows, key=lambda r: float(r.get('return_pct', 0) or 0), reverse=True)
    
    print("\nTOP 10 COMBINATIONS:")
    print("-" * 80)
    print(f"{'Rank':<6} {'EMA':<12} {'Stop':<8} {'Risk':<8} {'Return%':<10} {'DD%':<10} {'Trades':<8}")
    print("-" * 80)
    
    for i, row in enumerate(sorted_rows[:10], 1):
        ema_fast = row.get('ema_fast', '')
        ema_slow = row.get('ema_slow', '')
        stop = row.get('stop_loss_atr_multiplier', '')
        risk = row.get('risk_per_trade_pct', '')
        return_pct = row.get('return_pct', '0')
        dd_pct = row.get('max_drawdown_pct', '0')
        trades = row.get('num_oos_trades', '0')
        
        print(f"{i:<6} {ema_fast}/{ema_slow:<10} {stop:<8} {risk}%{'':<5} {return_pct:<10} {dd_pct:<10} {trades:<8}")
    
    print("\nBOTTOM 10 COMBINATIONS:")
    print("-" * 80)
    print(f"{'Rank':<6} {'EMA':<12} {'Stop':<8} {'Risk':<8} {'Return%':<10} {'DD%':<10} {'Trades':<8}")
    print("-" * 80)
    
    for i, row in enumerate(sorted_rows[-10:], len(sorted_rows) - 9):
        ema_fast = row.get('ema_fast', '')
        ema_slow = row.get('ema_slow', '')
        stop = row.get('stop_loss_atr_multiplier', '')
        risk = row.get('risk_per_trade_pct', '')
        return_pct = row.get('return_pct', '0')
        dd_pct = row.get('max_drawdown_pct', '0')
        trades = row.get('num_oos_trades', '0')
        
        print(f"{i:<6} {ema_fast}/{ema_slow:<10} {stop:<8} {risk}%{'':<5} {return_pct:<10} {dd_pct:<10} {trades:<8}")
    
    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY TEST COMPLETE")
    print("=" * 80)
    
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
