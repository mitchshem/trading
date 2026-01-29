#!/usr/bin/env python3
"""
Intraday backtest sanity check script.
Runs controlled replays to verify system behavior.
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def run_intraday_replay(
    symbol: str,
    entry_variant: str,
    exit_mode: str = "time_based",
    exit_params: Dict = None,
    allowed_sessions: list = None,
    starting_equity: float = 100000.0,
    risk_per_trade_pct: float = 0.25,
    max_daily_loss_pct: float = 1.0,
    max_concurrent_positions: int = 1,
    diagnostic_mode: bool = False,
    bypass_daily_regime_gate: bool = False,
    bypass_session_gate: bool = False,
    test_label: str = ""
) -> Dict[str, Any]:
    """
    Run an intraday replay and extract key metrics.
    """
    if exit_params is None:
        exit_params = {"time_candles": 6}
    if allowed_sessions is None:
        allowed_sessions = ["market_open"]
    
    payload = {
        "symbol": symbol,
        "interval": "5min",
        "entry_variant": entry_variant,
        "exit_mode": exit_mode,
        "exit_params": exit_params,
        "allowed_sessions": allowed_sessions,
        "diagnostic_mode": diagnostic_mode,
        "bypass_daily_regime_gate": bypass_daily_regime_gate,
        "bypass_session_gate": bypass_session_gate,
        "starting_equity": starting_equity,
        "risk_per_trade_pct": risk_per_trade_pct,
        "max_daily_loss_pct": max_daily_loss_pct,
        "max_concurrent_positions": max_concurrent_positions
    }
    
    print(f"\n{'='*70}")
    label = test_label if test_label else f"entry_variant={entry_variant}"
    print(f"Running replay: {label}")
    print(f"{'='*70}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/replay/start_intraday_csv",
            json=payload,
            timeout=300  # 5 minute timeout
        )
        print(f"[DEBUG] Response status code: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERROR] Non-200 status: {response.status_code}")
            try:
                error_body = response.json()
                print(f"[ERROR] Error body: {json.dumps(error_body, indent=2)}")
            except:
                print(f"[ERROR] Error text: {response.text}")
            return None
        response.raise_for_status()
        result = response.json()
        
        # Debug: print response type and structure
        print(f"[DEBUG] Response type: {type(result)}")
        if isinstance(result, list):
            print(f"[DEBUG] Response is a list with {len(result)} items")
            if len(result) > 0:
                print(f"[DEBUG] First item type: {type(result[0])}")
                if isinstance(result[0], dict):
                    print(f"[DEBUG] First item keys: {list(result[0].keys())}")
                    # Check for error
                    if "error" in result[0]:
                        error_msg = result[0].get('error', 'Unknown error')
                        print(f"[ERROR] API returned error: {error_msg}")
                        print(f"[ERROR] Full error response: {json.dumps(result[0], indent=2)}")
                        return None
            # If it's a list, take the first item (should be the result for single symbol)
            if len(result) > 0 and isinstance(result[0], dict) and "error" not in result[0]:
                result = result[0]
            else:
                print(f"[ERROR] Unexpected list structure in response or error present")
                return None
        
        # Extract key metrics
        risk_diagnostics = result.get("risk_diagnostics", {})
        metrics = {
            "test_label": test_label if test_label else entry_variant,
            "total_trades": result.get("trades_executed", 0),
            "final_equity": risk_diagnostics.get("final_equity") if risk_diagnostics else result.get("final_equity"),
            "max_drawdown_pct": result.get("max_drawdown_pct"),
            "return_on_capital_pct": risk_diagnostics.get("return_on_capital_pct") if risk_diagnostics else None,
        }
        
        # Extract from intraday_trade_metrics if available (note: it's "intraday_trade_metrics" not "intraday_aggregate_metrics")
        if "intraday_trade_metrics" in result:
            agg = result["intraday_trade_metrics"]
            metrics["expectancy"] = agg.get("expectancy")
            metrics["average_holding_time_minutes"] = agg.get("average_holding_time_minutes")
            metrics["average_mfe"] = agg.get("average_mfe")
            metrics["average_mae"] = agg.get("average_mae")
            metrics["average_mfe_given_up_pct"] = agg.get("average_mfe_given_up_pct")
        
        # Extract from frequency_metrics if available
        if "frequency_metrics" in result:
            freq = result["frequency_metrics"]
            metrics["average_trades_per_day"] = freq.get("average_trades_per_day")
        
        return metrics
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_response = e.response.json()
                print(f"Error response: {json.dumps(error_response, indent=2)}")
            except:
                print(f"Error response text: {e.response.text}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_comparison_table(results: list):
    """
    Print a side-by-side comparison table.
    """
    print(f"\n{'='*70}")
    print("SANITY CHECK RESULTS - SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")
    
    # Header
    header = f"{'Entry Variant':<25} {'Trades':<10} {'Trades/Day':<12} {'Expectancy':<15} {'Final Equity':<15} {'Max DD %':<12}"
    print(header)
    print("-" * 70)
    
    # Data rows
    for r in results:
        if r is None:
            print(f"{'ERROR':<25} {'N/A':<10} {'N/A':<12} {'N/A':<15} {'N/A':<15} {'N/A':<12}")
            continue
        
        entry_variant = r.get("entry_variant", "N/A")
        total_trades = r.get("total_trades", 0)
        avg_trades_per_day = r.get("average_trades_per_day", 0)
        expectancy = r.get("expectancy", 0)
        final_equity = r.get("final_equity", 0)
        max_dd_pct = r.get("max_drawdown_pct", 0)
        
        # Format values safely
        from utils import fmt, fmt_currency, fmt_pct
        
        print(f"{entry_variant:<25} "
              f"{total_trades:<10} "
              f"{fmt(avg_trades_per_day):<12} "
              f"{fmt_currency(expectancy):<15} "
              f"{fmt_currency(final_equity):<15} "
              f"{fmt_pct(max_dd_pct):<12}")
    
    print(f"{'='*70}\n")


def print_gate_comparison_table(results: list):
    """
    Print a comparison table sorted by expectancy (best first).
    """
    from utils import fmt, fmt_pct, fmt_currency
    
    # Sort by expectancy (descending, best first)
    sorted_results = sorted(results, key=lambda x: x.get("expectancy", 0) if x and x.get("expectancy") is not None else float('-inf'), reverse=True)
    
    print(f"\n{'='*90}")
    print("GATE ISOLATION COMPARISON (Sorted by Expectancy)")
    print(f"{'='*90}")
    
    # Header
    header = f"{'Test Case':<20} {'Trades':<10} {'Trades/Day':<12} {'Expectancy':<15} {'Return %':<12} {'Max DD %':<12} {'Final Equity':<15}"
    print(header)
    print("-" * 90)
    
    # Data rows
    for r in sorted_results:
        if r is None:
            print(f"{'ERROR':<20} {'N/A':<10} {'N/A':<12} {'N/A':<15} {'N/A':<12} {'N/A':<12} {'N/A':<15}")
            continue
        
        test_label = r.get("test_label", "N/A")
        total_trades = r.get("total_trades", 0)
        avg_trades_per_day = r.get("average_trades_per_day", 0)
        expectancy = r.get("expectancy", 0)
        return_pct = r.get("return_on_capital_pct", 0)
        max_dd_pct = r.get("max_drawdown_pct", 0)
        final_equity = r.get("final_equity", 0)
        
        print(f"{test_label:<20} "
              f"{total_trades:<10} "
              f"{fmt(avg_trades_per_day):<12} "
              f"{fmt_currency(expectancy):<15} "
              f"{fmt_pct(return_pct):<12} "
              f"{fmt_pct(max_dd_pct):<12} "
              f"{fmt_currency(final_equity):<15}")
    
    print(f"{'='*90}\n")


def print_entry_variant_comparison_table(results: list):
    """
    Print a comparison table sorted by expectancy (best first).
    """
    from utils import fmt, fmt_pct, fmt_currency
    
    # Sort by expectancy (descending, best first)
    sorted_results = sorted(results, key=lambda x: x.get("expectancy", 0) if x and x.get("expectancy") is not None else float('-inf'), reverse=True)
    
    print(f"\n{'='*100}")
    print("ENTRY VARIANT COMPARISON (Sorted by Expectancy)")
    print(f"{'='*100}")
    
    # Header
    header = f"{'Entry Variant':<25} {'Trades':<10} {'Expectancy':<15} {'Avg MFE':<15} {'Avg MAE':<15} {'MFE/MAE':<12} {'MFE Given Up':<15} {'Max DD %':<12} {'Final Equity':<15}"
    print(header)
    print("-" * 100)
    
    # Data rows
    for r in sorted_results:
        if r is None:
            print(f"{'ERROR':<25} {'N/A':<10} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<12} {'N/A':<15} {'N/A':<12} {'N/A':<15}")
            continue
        
        test_label = r.get("test_label", "N/A")
        total_trades = r.get("total_trades", 0)
        expectancy = r.get("expectancy", 0)
        avg_mfe = r.get("average_mfe", 0)
        avg_mae = r.get("average_mae", 0)
        mfe_given_up_pct = r.get("average_mfe_given_up_pct", 0)
        max_dd_pct = r.get("max_drawdown_pct", 0)
        final_equity = r.get("final_equity", 0)
        
        # Calculate MFE/MAE ratio
        mfe_mae_ratio = (avg_mfe / abs(avg_mae)) if avg_mae != 0 else 0
        
        print(f"{test_label:<25} "
              f"{total_trades:<10} "
              f"{fmt_currency(expectancy):<15} "
              f"{fmt_currency(avg_mfe):<15} "
              f"{fmt_currency(avg_mae):<15} "
              f"{fmt(mfe_mae_ratio):<12} "
              f"{fmt_pct(mfe_given_up_pct):<15} "
              f"{fmt_pct(max_dd_pct):<12} "
              f"{fmt_currency(final_equity):<15}")
    
    print(f"{'='*100}\n")


def main():
    """
    Run entry variant comparison test: three replays with different entry variants.
    """
    print("="*100)
    print("ENTRY VARIANT COMPARISON - Improving Expectancy by Reducing MAE")
    print("="*100)
    print("\nThis will run three test cases with identical parameters except for entry variant:")
    print("1) close_above_prev_low")
    print("2) break_prev_high")
    print("3) break_prev_high_atr (volatility-filtered)")
    print("\nBase config:")
    print("  - symbol: SPY")
    print("  - exit_mode: time_based")
    print("  - exit_params: {time_candles: 6}")
    print("  - allowed_sessions: [market_open, power_hour]")
    print("  - starting_equity: 100000")
    print("  - risk_per_trade_pct: 0.25")
    print("="*100)
    
    results = []
    
    # BASE CONFIG (same for all tests)
    base_config = {
        "symbol": "SPY",
        "exit_mode": "time_based",
        "exit_params": {"time_candles": 6},
        "allowed_sessions": ["market_open", "power_hour"],
        "starting_equity": 100000.0,
        "risk_per_trade_pct": 0.25,
        "max_daily_loss_pct": 1.0,
        "max_concurrent_positions": 1,
        "diagnostic_mode": False,
        "bypass_daily_regime_gate": False,
        "bypass_session_gate": False
    }
    
    # TEST 1: close_above_prev_low
    print("\n" + "="*100)
    print("TEST 1: close_above_prev_low")
    print("="*100)
    result1 = run_intraday_replay(
        **base_config,
        entry_variant="close_above_prev_low",
        test_label="1) close_above_prev_low"
    )
    if result1:
        from utils import fmt, fmt_pct, fmt_currency
        print(f"\n[TEST 1 RESULTS]")
        print(f"  Total Trades: {result1.get('total_trades', 0)}")
        print(f"  Expectancy: {fmt_currency(result1.get('expectancy', 0))}")
        print(f"  Avg MFE: {fmt_currency(result1.get('average_mfe', 0))}")
        print(f"  Avg MAE: {fmt_currency(result1.get('average_mae', 0))}")
        print(f"  MFE Given Up: {fmt_pct(result1.get('average_mfe_given_up_pct', 0))}")
        print(f"  Max Drawdown: {fmt_pct(result1.get('max_drawdown_pct', 0))}")
        print(f"  Final Equity: {fmt_currency(result1.get('final_equity', 0))}")
    results.append(result1)
    
    # TEST 2: break_prev_high
    print("\n" + "="*100)
    print("TEST 2: break_prev_high")
    print("="*100)
    result2 = run_intraday_replay(
        **base_config,
        entry_variant="break_prev_high",
        test_label="2) break_prev_high"
    )
    if result2:
        from utils import fmt, fmt_pct, fmt_currency
        print(f"\n[TEST 2 RESULTS]")
        print(f"  Total Trades: {result2.get('total_trades', 0)}")
        print(f"  Expectancy: {fmt_currency(result2.get('expectancy', 0))}")
        print(f"  Avg MFE: {fmt_currency(result2.get('average_mfe', 0))}")
        print(f"  Avg MAE: {fmt_currency(result2.get('average_mae', 0))}")
        print(f"  MFE Given Up: {fmt_pct(result2.get('average_mfe_given_up_pct', 0))}")
        print(f"  Max Drawdown: {fmt_pct(result2.get('max_drawdown_pct', 0))}")
        print(f"  Final Equity: {fmt_currency(result2.get('final_equity', 0))}")
    results.append(result2)
    
    # TEST 3: break_prev_high_atr
    print("\n" + "="*100)
    print("TEST 3: break_prev_high_atr (volatility-filtered)")
    print("="*100)
    result3 = run_intraday_replay(
        **base_config,
        entry_variant="break_prev_high_atr",
        test_label="3) break_prev_high_atr"
    )
    if result3:
        from utils import fmt, fmt_pct, fmt_currency
        print(f"\n[TEST 3 RESULTS]")
        print(f"  Total Trades: {result3.get('total_trades', 0)}")
        print(f"  Expectancy: {fmt_currency(result3.get('expectancy', 0))}")
        print(f"  Avg MFE: {fmt_currency(result3.get('average_mfe', 0))}")
        print(f"  Avg MAE: {fmt_currency(result3.get('average_mae', 0))}")
        print(f"  MFE Given Up: {fmt_pct(result3.get('average_mfe_given_up_pct', 0))}")
        print(f"  Max Drawdown: {fmt_pct(result3.get('max_drawdown_pct', 0))}")
        print(f"  Final Equity: {fmt_currency(result3.get('final_equity', 0))}")
    results.append(result3)
    
    # Print comparison table sorted by expectancy
    print_entry_variant_comparison_table(results)
    
    # Analysis
    print("="*100)
    print("ANALYSIS")
    print("="*100)
    
    valid_results = [r for r in results if r is not None]
    if len(valid_results) < 3:
        print("❌ Some tests failed to complete")
        return
    
    # Find best and worst by expectancy
    best = max(valid_results, key=lambda x: x.get("expectancy", float('-inf')) if x.get("expectancy") is not None else float('-inf'))
    worst = min(valid_results, key=lambda x: x.get("expectancy", float('inf')) if x.get("expectancy") is not None else float('inf'))
    
    from utils import fmt_currency, fmt_pct
    print(f"\nBest Expectancy: {best.get('test_label')} = {fmt_currency(best.get('expectancy', 0))}")
    print(f"Worst Expectancy: {worst.get('test_label')} = {fmt_currency(worst.get('expectancy', 0))}")
    print(f"\nExpectancy Delta: {fmt_currency(best.get('expectancy', 0) - worst.get('expectancy', 0))}")
    
    # Compare MAE and MFE
    for r in valid_results:
        label = r.get("test_label", "Unknown")
        avg_mfe = r.get("average_mfe", 0)
        avg_mae = r.get("average_mae", 0)
        mfe_mae_ratio = (avg_mfe / abs(avg_mae)) if avg_mae != 0 else 0
        print(f"\n{label}:")
        print(f"  Avg MFE: {fmt_currency(avg_mfe)}")
        print(f"  Avg MAE: {fmt_currency(avg_mae)}")
        print(f"  MFE/MAE Ratio: {fmt(mfe_mae_ratio)}")
        if avg_mae < 0:
            print(f"  MAE Improvement: {fmt_currency(abs(avg_mae) - abs(worst.get('average_mae', 0)))} vs worst")
    
    # Check if ATR filter improved MAE
    if result3:
        result1_mae = abs(result1.get("average_mae", 0)) if result1 else 0
        result3_mae = abs(result3.get("average_mae", 0))
        if result3_mae < result1_mae:
            mae_improvement = result1_mae - result3_mae
            print(f"\n✅ ATR filter REDUCED MAE by {fmt_currency(mae_improvement)}")
        elif result3_mae > result1_mae:
            mae_worsening = result3_mae - result1_mae
            print(f"\n❌ ATR filter INCREASED MAE by {fmt_currency(mae_worsening)}")
        else:
            print(f"\n➖ ATR filter had NO EFFECT on MAE")
    
    print("="*100)


if __name__ == "__main__":
    main()
