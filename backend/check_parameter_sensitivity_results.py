#!/usr/bin/env python3
"""Check parameter sensitivity results and print summary."""

import csv
import json
from pathlib import Path

results_dir = Path("parameter_sensitivity_results")
csv_path = results_dir / "parameter_sensitivity_results.csv"
json_path = results_dir / "parameter_sensitivity_summary.json"

if not csv_path.exists():
    print("ERROR: Results CSV not found. Run parameter sensitivity test first.")
    exit(1)

# Read CSV
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("=" * 80)
print("PARAMETER SENSITIVITY RESULTS SUMMARY")
print("=" * 80)

# Sanity checks
print(f"\n[Sanity Checks]")
print(f"  Total rows in CSV: {len(rows)}")
print(f"  Expected: 81")
if len(rows) == 81:
    print("  ✓ Row count correct")
else:
    print(f"  ✗ Row count mismatch (expected 81, got {len(rows)})")

# Check determinism violations
if json_path.exists():
    with open(json_path, 'r') as f:
        summary = json.load(f)
    violations = summary.get('determinism_violations', [])
    print(f"\n  Determinism violations: {len(violations)}")
    if len(violations) == 0:
        print("  ✓ No determinism violations")
    else:
        print(f"  ✗ {len(violations)} determinism violations detected:")
        for v in violations[:5]:
            print(f"    - {v}")

# Check trade counts
trade_counts = [int(r.get('num_oos_trades', 0) or 0) for r in rows]
non_zero_count = sum(1 for count in trade_counts if count > 0)
print(f"\n  Combinations with trades > 0: {non_zero_count}/81 ({non_zero_count/81*100:.1f}%)")
if non_zero_count >= 20:
    print("  ✓ Meaningful portion of combinations produced trades")
else:
    print(f"  ⚠ Only {non_zero_count} combinations produced trades (may indicate small test windows)")

# Filter completed rows with valid return_pct
completed = [r for r in rows if r.get('return_pct') and r.get('return_pct') != '']
# Sort: non-zero returns first (descending), then zero returns
sorted_rows = sorted(
    completed,
    key=lambda r: (float(r.get('return_pct', 0) or 0) != 0.0, float(r.get('return_pct', 0) or 0)),
    reverse=True
)

# Get non-zero returns
non_zero_returns = [r for r in sorted_rows if float(r.get('return_pct', 0) or 0) != 0.0]

print(f"\n  Combinations with non-zero returns: {len(non_zero_returns)}/81")

# Top 10 and Bottom 10
print("\n" + "=" * 80)
print("TOP 10 COMBINATIONS BY OOS RETURN%")
print("=" * 80)
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

print("\n" + "=" * 80)
print("BOTTOM 10 COMBINATIONS BY OOS RETURN%")
print("=" * 80)
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
