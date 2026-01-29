"""
Cost sensitivity testing module.
Runs the same replay with different slippage and commission levels to assess robustness.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from replay_engine import ReplayEngine
from database import Trade, EquityCurve
from metrics import compute_metrics
from utils import fmt, fmt_pct, fmt_currency


class CostSensitivityTester:
    """
    Cost sensitivity tester that runs replays with different cost parameters.
    Ensures deterministic results by using the same candles and execution path.
    """
    
    def __init__(
        self,
        initial_equity: float = 100000.0,
        base_slippage: float = 0.0002,  # Base slippage (0.02%)
        slippage_multipliers: List[float] = [0.5, 1.0, 2.0],
        commission_levels: List[Tuple[float, float]] = [
            (0.0, 0.0),  # No commission
            (0.005, 1.0),  # $0.005/share + $1/trade
            (0.01, 2.0),  # $0.01/share + $2/trade
        ]
    ):
        """
        Initialize cost sensitivity tester.
        
        Args:
            initial_equity: Starting equity for all replays
            base_slippage: Base slippage rate (default: 0.02% = 0.0002)
            slippage_multipliers: List of multipliers to apply to base slippage
            commission_levels: List of (commission_per_share, commission_per_trade) tuples
        """
        self.initial_equity = initial_equity
        self.base_slippage = base_slippage
        self.slippage_multipliers = slippage_multipliers
        self.commission_levels = commission_levels
        
        # Results storage
        self.test_results: List[Dict] = []
    
    def run_cost_sensitivity_test(
        self,
        symbol: str,
        candles: List[Dict],
        db_session,
        allowed_entry_regimes: Optional[List[str]] = None
    ) -> Dict:
        """
        Run cost sensitivity test with multiple slippage and commission combinations.
        
        Args:
            symbol: Trading symbol
            candles: Ordered list of candle dicts
            db_session: Database session
            allowed_entry_regimes: Optional regime gate for entries
        
        Returns:
            Dict with test results for all cost combinations
        """
        print(f"\n[COST SENSITIVITY] Starting cost sensitivity test for {symbol}")
        print(f"[COST SENSITIVITY] Base slippage: {self.base_slippage:.4f} ({self.base_slippage*100:.2f}%)")
        print(f"[COST SENSITIVITY] Slippage multipliers: {self.slippage_multipliers}")
        print(f"[COST SENSITIVITY] Commission levels: {self.commission_levels}")
        print(f"[COST SENSITIVITY] Total test combinations: {len(self.slippage_multipliers) * len(self.commission_levels)}\n")
        
        self.test_results = []
        test_id = 0
        
        # Run all combinations
        for slippage_mult in self.slippage_multipliers:
            slippage = self.base_slippage * slippage_mult
            
            for comm_per_share, comm_per_trade in self.commission_levels:
                test_id += 1
                test_name = f"slippage_{slippage_mult}x_comm_{comm_per_share}_{comm_per_trade}"
                
                print(f"[COST SENSITIVITY] Test {test_id}: {test_name}")
                print(f"  Slippage: {slippage:.4f} ({slippage*100:.2f}%)")
                print(f"  Commission: ${comm_per_share:.3f}/share + ${comm_per_trade:.2f}/trade")
                
                # Create replay engine with specific cost parameters
                replay_engine = ReplayEngine(
                    initial_equity=self.initial_equity,
                    commission_per_share=comm_per_share,
                    commission_per_trade=comm_per_trade,
                    slippage=slippage
                )
                
                # Generate unique replay_id for this test
                import uuid
                replay_id = str(uuid.uuid4())
                
                try:
                    # Start and run replay
                    replay_engine.start_replay(
                        symbol=symbol,
                        candles=candles.copy(),  # Copy to ensure no mutation
                        replay_id=replay_id,
                        source="cost_sensitivity",
                        allowed_entry_regimes=allowed_entry_regimes
                    )
                    
                    replay_result = replay_engine.run(db_session)
                    
                    # Fetch trades and equity curve
                    trades = db_session.query(Trade).filter(
                        Trade.replay_id == replay_id,
                        Trade.symbol == symbol
                    ).all()
                    
                    equity_curve = db_session.query(EquityCurve).filter(
                        EquityCurve.replay_id == replay_id
                    ).order_by(EquityCurve.timestamp).all()
                    
                    # Compute metrics
                    metrics_snapshot = compute_metrics(trades, equity_curve)
                    
                    # Calculate total commissions paid
                    # Commission = (shares * commission_per_share) + commission_per_trade
                    total_commissions = sum(
                        (trade.shares * comm_per_share + comm_per_trade) +  # Entry commission
                        (trade.shares * comm_per_share + comm_per_trade)    # Exit commission
                        for trade in trades
                        if trade.exit_time and trade.pnl is not None
                    )
                    
                    # Store result
                    result = {
                        "test_id": test_id,
                        "test_name": test_name,
                        "replay_id": replay_id,
                        "slippage_multiplier": slippage_mult,
                        "slippage": slippage,
                        "commission_per_share": comm_per_share,
                        "commission_per_trade": comm_per_trade,
                        "total_commissions": round(total_commissions, 2),
                        "trade_count": len(trades),
                        "final_equity": metrics_snapshot.equity_end,
                        "net_pnl": metrics_snapshot.net_pnl,
                        "return_pct": metrics_snapshot.total_return_pct,
                        "max_drawdown_pct": metrics_snapshot.max_drawdown_pct,
                        "max_drawdown_absolute": metrics_snapshot.max_drawdown_absolute,
                        "win_rate": metrics_snapshot.win_rate,
                        "expectancy": metrics_snapshot.expectancy_per_trade,
                        "profit_factor": metrics_snapshot.profit_factor,
                        "sharpe_proxy": metrics_snapshot.sharpe_proxy,
                        "status": "completed"
                    }
                    
                    self.test_results.append(result)
                    
                    print(f"  Result: Return={fmt_pct(metrics_snapshot.total_return_pct)}, "
                          f"MaxDD={fmt_pct(metrics_snapshot.max_drawdown_pct)}, "
                          f"Trades={len(trades)}, "
                          f"Commissions=${fmt_currency(total_commissions)}")
                    print()
                
                except Exception as e:
                    error_msg = str(e)
                    print(f"  ERROR: {error_msg}\n")
                    self.test_results.append({
                        "test_id": test_id,
                        "test_name": test_name,
                        "slippage_multiplier": slippage_mult,
                        "slippage": slippage,
                        "commission_per_share": comm_per_share,
                        "commission_per_trade": comm_per_trade,
                        "status": "error",
                        "error": error_msg
                    })
        
        # Generate summary report
        summary = self._generate_summary_report()
        
        return {
            "status": "completed",
            "symbol": symbol,
            "total_tests": len(self.test_results),
            "completed_tests": len([r for r in self.test_results if r.get("status") == "completed"]),
            "test_results": self.test_results,
            "summary": summary
        }
    
    def _calculate_cost_impact(self, completed_results: List[Dict]) -> Dict:
        """Calculate cost impact by comparing lowest vs highest cost scenarios."""
        if not completed_results:
            return {
                "lowest_cost_return": None,
                "highest_cost_return": None,
                "cost_sensitivity_pct": None
            }
        
        # Find lowest and highest cost scenarios (by total cost, not robustness)
        # Lower cost = lower slippage + lower commission
        lowest_cost_result = min(completed_results, key=lambda x: (
            x["slippage_multiplier"] * self.base_slippage + 
            x["commission_per_share"] * 100 + x["commission_per_trade"]
        ))
        highest_cost_result = max(completed_results, key=lambda x: (
            x["slippage_multiplier"] * self.base_slippage + 
            x["commission_per_share"] * 100 + x["commission_per_trade"]
        ))
        
        lowest_return = lowest_cost_result["return_pct"]
        highest_return = highest_cost_result["return_pct"]
        
        cost_sensitivity_pct = None
        if lowest_return != 0:
            cost_sensitivity_pct = round(
                (lowest_return - highest_return) / abs(lowest_return) * 100, 
                2
            )
        
        return {
            "lowest_cost_return": round(lowest_return, 2),
            "highest_cost_return": round(highest_return, 2),
            "cost_sensitivity_pct": cost_sensitivity_pct
        }
    
    def _generate_summary_report(self) -> Dict:
        """Generate summary report ranking robustness."""
        completed_results = [r for r in self.test_results if r.get("status") == "completed"]
        
        if not completed_results:
            return {
                "error": "No completed tests to analyze"
            }
        
        # Calculate robustness score: return_pct / abs(max_drawdown_pct) (higher is better)
        # This measures return per unit of drawdown risk
        for result in completed_results:
            max_dd_abs = abs(result["max_drawdown_pct"])
            if max_dd_abs > 0.01:  # If drawdown > 0.01%
                result["robustness_score"] = result["return_pct"] / max_dd_abs
            elif result["return_pct"] > 0:
                # If no significant drawdown and positive return, use a high score
                result["robustness_score"] = result["return_pct"] * 100
            else:
                # Negative return with no drawdown (edge case) - use return as score
                result["robustness_score"] = result["return_pct"]
        
        # Sort by robustness score (descending)
        sorted_results = sorted(completed_results, key=lambda x: x["robustness_score"], reverse=True)
        
        # Calculate statistics
        returns = [r["return_pct"] for r in completed_results]
        drawdowns = [r["max_drawdown_pct"] for r in completed_results]
        expectancies = [r["expectancy"] for r in completed_results]
        win_rates = [r["win_rate"] for r in completed_results]
        
        return {
            "total_tests": len(completed_results),
            "ranked_results": [
                {
                    "rank": i + 1,
                    "test_name": r["test_name"],
                    "slippage_multiplier": r["slippage_multiplier"],
                    "commission_per_share": r["commission_per_share"],
                    "commission_per_trade": r["commission_per_trade"],
                    "return_pct": r["return_pct"],
                    "max_drawdown_pct": r["max_drawdown_pct"],
                    "robustness_score": round(r["robustness_score"], 2),
                    "trade_count": r["trade_count"],
                    "expectancy": r["expectancy"],
                    "win_rate": r["win_rate"]
                }
                for i, r in enumerate(sorted_results)
            ],
            "statistics": {
                "return_pct": {
                    "min": round(min(returns), 2),
                    "max": round(max(returns), 2),
                    "mean": round(sum(returns) / len(returns), 2),
                    "range": round(max(returns) - min(returns), 2)
                },
                "max_drawdown_pct": {
                    "min": round(min(drawdowns), 2),
                    "max": round(max(drawdowns), 2),
                    "mean": round(sum(drawdowns) / len(drawdowns), 2),
                    "range": round(max(drawdowns) - min(drawdowns), 2)
                },
                "expectancy": {
                    "min": round(min(expectancies), 2),
                    "max": round(max(expectancies), 2),
                    "mean": round(sum(expectancies) / len(expectancies), 2)
                },
                "win_rate": {
                    "min": round(min(win_rates), 2),
                    "max": round(max(win_rates), 2),
                    "mean": round(sum(win_rates) / len(win_rates), 2)
                }
            },
            "cost_impact": self._calculate_cost_impact(completed_results)
        }
    
    def save_report(self, output_dir: str, symbol: str) -> Dict[str, str]:
        """
        Save cost sensitivity report to files.
        
        Args:
            output_dir: Directory to save report
            symbol: Trading symbol (for filename)
        
        Returns:
            Dict with paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results JSON
        results_path = output_path / f"{symbol}_cost_sensitivity_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "test_results": self.test_results,
                "summary": self._generate_summary_report()
            }, f, indent=2, default=str)
        
        # Save summary report as text
        summary_path = output_path / f"{symbol}_cost_sensitivity_summary.txt"
        self._save_text_summary(summary_path, symbol)
        
        return {
            "results_json": str(results_path),
            "summary_txt": str(summary_path)
        }
    
    def _save_text_summary(self, filepath: Path, symbol: str):
        """Save human-readable summary report."""
        summary = self._generate_summary_report()
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"COST SENSITIVITY TEST REPORT: {symbol}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"  Base slippage: {self.base_slippage:.4f} ({self.base_slippage*100:.2f}%)\n")
            f.write(f"  Slippage multipliers: {self.slippage_multipliers}\n")
            f.write(f"  Commission levels: {self.commission_levels}\n")
            f.write(f"  Initial equity: ${fmt_currency(self.initial_equity)}\n\n")
            
            f.write("RANKED RESULTS (by Robustness Score = Return / Max Drawdown):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<6} {'Slippage':<12} {'Commission':<20} {'Return%':<10} {'MaxDD%':<10} {'Score':<10} {'Trades':<8}\n")
            f.write("-" * 80 + "\n")
            
            for result in summary["ranked_results"]:
                comm_str = f"${result['commission_per_share']:.3f}/sh + ${result['commission_per_trade']:.2f}"
                f.write(
                    f"{result['rank']:<6} "
                    f"{result['slippage_multiplier']:.1f}x{'':<6} "
                    f"{comm_str:<20} "
                    f"{result['return_pct']:>8.2f}% "
                    f"{result['max_drawdown_pct']:>8.2f}% "
                    f"{result['robustness_score']:>8.2f} "
                    f"{result['trade_count']:>6}\n"
                )
            
            f.write("\n")
            f.write("STATISTICS:\n")
            f.write("-" * 80 + "\n")
            stats = summary["statistics"]
            f.write(f"Return %:      Min={stats['return_pct']['min']:>8.2f}, Max={stats['return_pct']['max']:>8.2f}, "
                   f"Mean={stats['return_pct']['mean']:>8.2f}, Range={stats['return_pct']['range']:>8.2f}\n")
            f.write(f"Max Drawdown %: Min={stats['max_drawdown_pct']['min']:>8.2f}, Max={stats['max_drawdown_pct']['max']:>8.2f}, "
                   f"Mean={stats['max_drawdown_pct']['mean']:>8.2f}, Range={stats['max_drawdown_pct']['range']:>8.2f}\n")
            f.write(f"Expectancy:     Min={stats['expectancy']['min']:>8.2f}, Max={stats['expectancy']['max']:>8.2f}, "
                   f"Mean={stats['expectancy']['mean']:>8.2f}\n")
            f.write(f"Win Rate:       Min={stats['win_rate']['min']:>8.2f}, Max={stats['win_rate']['max']:>8.2f}, "
                   f"Mean={stats['win_rate']['mean']:>8.2f}\n")
            
            f.write("\n")
            f.write("COST IMPACT:\n")
            f.write("-" * 80 + "\n")
            cost_impact = summary["cost_impact"]
            f.write(f"Lowest cost return:   {cost_impact['lowest_cost_return']:>8.2f}%\n")
            f.write(f"Highest cost return:  {cost_impact['highest_cost_return']:>8.2f}%\n")
            f.write(f"Cost sensitivity:     {cost_impact['cost_sensitivity_pct']:>8.2f}% (performance degradation from lowest to highest cost)\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
