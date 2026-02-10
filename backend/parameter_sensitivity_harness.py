"""
Parameter Sensitivity Harness for Robustness Testing

PURPOSE:
This module tests strategy robustness to small parameter changes. This is NOT optimization.
The goal is to identify which parameters are STABLE (small changes have minimal impact) vs
DANGEROUS (small changes cause sign flips or large variance).

KEY DIFFERENCE FROM OPTIMIZATION:
- Optimization: Find the "best" parameters for historical data (risks overfitting)
- Sensitivity: Test if small parameter changes cause large performance swings (identifies fragility)

INTERPRETATION:
- STABLE parameters: Small changes produce consistent results → strategy is robust
- DANGEROUS parameters: Small changes cause sign flips or high variance → strategy is fragile
- Sign flips: Profitable → unprofitable (or vice versa) indicates parameter sensitivity
- Large variance: High standard deviation suggests unstable performance

USAGE:
    harness = ParameterSensitivityHarness()
    results = harness.run_sensitivity_test(
        symbol="SPY",
        candles=all_candles,
        start_date="2020-01-01",
        end_date="2024-12-31",
        db_session=db_session
    )
    harness.save_results("sensitivity_results")
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import json
import csv
import time
import statistics
from itertools import product

from walkforward_harness import WalkForwardHarness
from replay_engine import ReplayEngine
from database import Trade, EquityCurve
from metrics import compute_metrics
from indicators import ema, atr
from strategy import PositionState
import uuid


class ParameterSensitivityHarness:
    """
    Parameter sensitivity testing harness.
    Tests robustness to small parameter changes using walk-forward evaluation.
    """
    
    def __init__(
        self,
        initial_equity: float = 100000.0,
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 21
    ):
        """
        Initialize parameter sensitivity harness.
        
        Args:
            initial_equity: Starting equity for each walk-forward window
            train_days: Training window size (days)
            test_days: Test window size (days)
            step_days: Step size between windows (days)
        """
        self.initial_equity = initial_equity
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        
        # Parameter ranges to test
        self.ema_fast_values = [15, 20, 25]
        self.ema_slow_values = [40, 50, 60]
        self.stop_loss_atr_multipliers = [1.5, 2.0, 2.5]
        self.risk_per_trade_pcts = [0.25, 0.5, 0.75]  # Percentage values
        
        # Results storage
        self.results: List[Dict] = []
        self.determinism_violations: List[str] = []
    
    def run_sensitivity_test(
        self,
        symbol: str,
        candles: List[Dict],
        start_date: str,
        end_date: str,
        db_session,
        allowed_entry_regimes: Optional[List[str]] = None
    ) -> Dict:
        """
        Run parameter sensitivity test across all parameter combinations.
        
        Args:
            symbol: Trading symbol
            candles: All available candles
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            db_session: Database session
            allowed_entry_regimes: Optional regime gate for entries
        
        Returns:
            Dict with sensitivity test results and summary
        """
        # Generate all parameter combinations
        combinations = list(product(
            self.ema_fast_values,
            self.ema_slow_values,
            self.stop_loss_atr_multipliers,
            self.risk_per_trade_pcts
        ))
        
        total_combinations = len(combinations)
        print(f"[PARAMETER SENSITIVITY] Testing {total_combinations} parameter combinations")
        print(f"[PARAMETER SENSITIVITY] EMA fast: {self.ema_fast_values}")
        print(f"[PARAMETER SENSITIVITY] EMA slow: {self.ema_slow_values}")
        print(f"[PARAMETER SENSITIVITY] Stop-loss ATR multiplier: {self.stop_loss_atr_multipliers}")
        print(f"[PARAMETER SENSITIVITY] Risk per trade: {self.risk_per_trade_pcts}%")
        
        start_time = time.time()
        self.results = []
        
        for idx, (ema_fast, ema_slow, stop_atr_mult, risk_pct) in enumerate(combinations):
            print(f"[PARAMETER SENSITIVITY] Running combination {idx + 1}/{total_combinations}: "
                  f"EMA({ema_fast}/{ema_slow}), Stop={stop_atr_mult}xATR, Risk={risk_pct}%")
            
            try:
                result = self._run_walkforward_with_params(
                    symbol=symbol,
                    candles=candles,
                    start_date=start_date,
                    end_date=end_date,
                    db_session=db_session,
                    ema_fast=ema_fast,
                    ema_slow=ema_slow,
                    stop_loss_atr_multiplier=stop_atr_mult,
                    risk_per_trade_pct=risk_pct,
                    allowed_entry_regimes=allowed_entry_regimes
                )
                
                result["ema_fast"] = ema_fast
                result["ema_slow"] = ema_slow
                result["stop_loss_atr_multiplier"] = stop_atr_mult
                result["risk_per_trade_pct"] = risk_pct
                result["combination_id"] = idx + 1
                
                self.results.append(result)
                
            except Exception as e:
                error_msg = f"Combination {idx + 1} failed: {str(e)}"
                print(f"[PARAMETER SENSITIVITY] ERROR: {error_msg}")
                self.determinism_violations.append(error_msg)
                # Continue with next combination
        
        elapsed_time = time.time() - start_time
        
        # Compute sensitivity analysis
        sensitivity_summary = self._compute_sensitivity_summary()
        
        return {
            "status": "completed",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "total_combinations": total_combinations,
            "completed_combinations": len(self.results),
            "failed_combinations": len(self.determinism_violations),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "sensitivity_summary": sensitivity_summary,
            "determinism_violations": self.determinism_violations
        }
    
    def _run_walkforward_with_params(
        self,
        symbol: str,
        candles: List[Dict],
        start_date: str,
        end_date: str,
        db_session,
        ema_fast: int,
        ema_slow: int,
        stop_loss_atr_multiplier: float,
        risk_per_trade_pct: float,
        allowed_entry_regimes: Optional[List[str]] = None
    ) -> Dict:
        """
        Run walk-forward test with specific parameters.
        
        This creates a custom ReplayEngine that uses the specified parameters.
        """
        # Create walk-forward harness
        harness = WalkForwardHarness(
            initial_equity=self.initial_equity,
            train_days=self.train_days,
            test_days=self.test_days,
            step_days=self.step_days
        )
        
        # Generate windows
        windows = harness.generate_windows(candles, start_date, end_date)
        
        if not windows:
            return {
                "status": "skipped",
                "reason": "No valid windows generated"
            }
        
        # Run each window with custom parameters
        all_oos_trades = []
        all_oos_equity = []
        
        print(f"[PARAMETER SENSITIVITY] Running {len(windows)} walk-forward windows...")
        
        for window in windows:
            window_result = self._run_window_with_params(
                symbol=symbol,
                candles=candles,
                window=window,
                db_session=db_session,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                stop_loss_atr_multiplier=stop_loss_atr_multiplier,
                risk_per_trade_pct=risk_per_trade_pct,
                allowed_entry_regimes=allowed_entry_regimes
            )
            
            if window_result["status"] == "completed":
                # Collect OOS trades and equity (test period only)
                test_replay_id = window_result["test_replay_id"]
                window_trades = db_session.query(Trade).filter(
                    Trade.replay_id == test_replay_id,
                    Trade.symbol == symbol
                ).all()
                window_equity = db_session.query(EquityCurve).filter(
                    EquityCurve.replay_id == test_replay_id
                ).order_by(EquityCurve.timestamp).all()
                
                all_oos_trades.extend(window_trades)
                all_oos_equity.extend(window_equity)
        
        # Filter to closed trades only (for accurate metrics)
        closed_oos_trades = [t for t in all_oos_trades if t.pnl is not None]
        
        # Compute aggregate metrics from all OOS data (closed trades only)
        metrics_snapshot = compute_metrics(closed_oos_trades, all_oos_equity)
        
        # Calculate additional metrics
        final_equity = metrics_snapshot.equity_end
        net_pnl = final_equity - self.initial_equity
        return_pct = (net_pnl / self.initial_equity * 100) if self.initial_equity > 0 else 0.0
        
        return {
            "status": "completed",
            "final_oos_equity": round(final_equity, 2),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(return_pct, 2),
            "max_drawdown_pct": round(metrics_snapshot.max_drawdown_pct, 2),
            "max_drawdown_absolute": round(metrics_snapshot.max_drawdown_absolute, 2),
            "win_rate": round(metrics_snapshot.win_rate, 2),
            "num_oos_trades": len(closed_oos_trades),  # Count only closed trades
            "num_total_trades": len(all_oos_trades),  # Total including open
            "sharpe_proxy": round(metrics_snapshot.sharpe_proxy, 2) if metrics_snapshot.sharpe_proxy is not None else None,
            "expectancy": round(metrics_snapshot.expectancy_per_trade, 2)
        }
    
    def _run_window_with_params(
        self,
        symbol: str,
        candles: List[Dict],
        window: Dict,
        db_session,
        ema_fast: int,
        ema_slow: int,
        stop_loss_atr_multiplier: float,
        risk_per_trade_pct: float,
        allowed_entry_regimes: Optional[List[str]] = None
    ) -> Dict:
        """
        Run a single test window with custom parameters.
        Creates a custom ReplayEngine that uses the specified parameters.
        """
        from walkforward import filter_candles_by_date_range
        
        # Filter candles for test period (OOS)
        test_start = str(window.get("test_start", ""))
        test_end = str(window.get("test_end", ""))
        test_candles = filter_candles_by_date_range(
            candles,
            test_start,
            test_end,
            candle_type="daily"
        )
        
        if not test_candles:
            return {
                "window_id": window["window_id"],
                "status": "skipped",
                "reason": "No candles in test period"
            }
        
        # Skip if test window doesn't have enough candles for EMA calculation
        if len(test_candles) < ema_slow:
            return {
                "window_id": window["window_id"],
                "status": "skipped",
                "reason": f"Test window has only {len(test_candles)} candles, need at least {ema_slow} for EMA({ema_slow})"
            }
        
        # Create custom replay engine with parameter overrides
        replay_engine = CustomReplayEngine(
            initial_equity=self.initial_equity,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            stop_loss_atr_multiplier=stop_loss_atr_multiplier,
            risk_per_trade_pct=risk_per_trade_pct
        )
        
        import uuid
        replay_id = str(uuid.uuid4()) + "_test"
        
        try:
            replay_engine.start_replay(
                symbol=symbol,
                candles=test_candles,
                replay_id=replay_id,
                source="parameter_sensitivity",
                allowed_entry_regimes=allowed_entry_regimes
            )
        except ValueError as e:
            # Handle validation errors (e.g., insufficient candles)
            return {
                "window_id": window["window_id"],
                "status": "skipped",
                "reason": str(e)
            }
        
        # Run replay
        replay_engine.run(db_session)
        
        # Close any open positions at end of window (for accurate metrics)
        # This ensures positions don't carry over to next window and are included in metrics
        if replay_engine.symbol in replay_engine.broker.positions:
            position = replay_engine.broker.positions[replay_engine.symbol]
            # Get final candle close price
            if test_candles:
                final_price = test_candles[-1]["close"]
                final_timestamp = test_candles[-1].get("close_time", test_candles[-1]["time"])
                
                # Create pending exit order
                replay_engine.broker.execute_exit(
                    symbol=replay_engine.symbol,
                    signal_price=final_price,
                    timestamp=final_timestamp,
                    reason="WINDOW_END"
                )
                
                # Process the exit order immediately (using final price as open price)
                executed_trades = replay_engine.broker.process_pending_orders(
                    current_open_price=final_price,
                    timestamp=final_timestamp
                )
                
                # Update trade record for executed exit
                from utils import unix_to_utc_datetime, ensure_utc_datetime
                for trade_dict in executed_trades:
                    if trade_dict.get("action") == "EXIT" and trade_dict.get("symbol") == replay_engine.symbol:
                        open_trade = db_session.query(Trade).filter(
                            Trade.symbol == replay_engine.symbol,
                            Trade.exit_time.is_(None),
                            Trade.replay_id == replay_id
                        ).first()
                        if open_trade:
                            exit_time = unix_to_utc_datetime(trade_dict["timestamp"])
                            ensure_utc_datetime(exit_time, f"window end EXIT for {replay_engine.symbol}")
                            open_trade.exit_time = exit_time
                            open_trade.exit_price = trade_dict["exit_price"]
                            open_trade.pnl = trade_dict.get("pnl")
                            open_trade.reason = "WINDOW_END"
                            db_session.commit()
                            
                            # Update equity curve with final state
                            from database import EquityCurve
                            final_equity_point = EquityCurve(
                                timestamp=exit_time,
                                equity=replay_engine.broker.equity,
                                replay_id=replay_id
                            )
                            db_session.add(final_equity_point)
                            db_session.commit()
                            break
        
        return {
            "window_id": window["window_id"],
            "status": "completed",
            "test_replay_id": replay_id
        }
    
    def _compute_sensitivity_summary(self) -> Dict:
        """Compute sensitivity analysis summary for each parameter."""
        if not self.results:
            return {}
        
        completed_results = [r for r in self.results if r.get("status") == "completed"]
        
        if not completed_results:
            return {}
        
        # Group results by parameter
        summary = {}
        
        # Analyze EMA fast sensitivity
        ema_fast_groups = {}
        for result in completed_results:
            ema_fast = result["ema_fast"]
            if ema_fast not in ema_fast_groups:
                ema_fast_groups[ema_fast] = []
            ema_fast_groups[ema_fast].append(result)
        
        summary["ema_fast"] = self._analyze_parameter_sensitivity(
            "ema_fast", ema_fast_groups, completed_results
        )
        
        # Analyze EMA slow sensitivity
        ema_slow_groups = {}
        for result in completed_results:
            ema_slow = result["ema_slow"]
            if ema_slow not in ema_slow_groups:
                ema_slow_groups[ema_slow] = []
            ema_slow_groups[ema_slow].append(result)
        
        summary["ema_slow"] = self._analyze_parameter_sensitivity(
            "ema_slow", ema_slow_groups, completed_results
        )
        
        # Analyze stop-loss ATR multiplier sensitivity
        stop_atr_groups = {}
        for result in completed_results:
            stop_atr = result["stop_loss_atr_multiplier"]
            if stop_atr not in stop_atr_groups:
                stop_atr_groups[stop_atr] = []
            stop_atr_groups[stop_atr].append(result)
        
        summary["stop_loss_atr_multiplier"] = self._analyze_parameter_sensitivity(
            "stop_loss_atr_multiplier", stop_atr_groups, completed_results
        )
        
        # Analyze risk per trade sensitivity
        risk_groups = {}
        for result in completed_results:
            risk = result["risk_per_trade_pct"]
            if risk not in risk_groups:
                risk_groups[risk] = []
            risk_groups[risk].append(result)
        
        summary["risk_per_trade_pct"] = self._analyze_parameter_sensitivity(
            "risk_per_trade_pct", risk_groups, completed_results
        )
        
        return summary
    
    def _analyze_parameter_sensitivity(
        self,
        param_name: str,
        param_groups: Dict,
        all_results: List[Dict]
    ) -> Dict:
        """Analyze sensitivity for a single parameter."""
        # Extract metrics for each parameter value
        return_pcts = {}
        drawdown_pcts = {}
        win_rates = {}
        sharpe_proxies = {}
        
        for param_value, group_results in param_groups.items():
            return_pcts[param_value] = [r["return_pct"] for r in group_results]
            drawdown_pcts[param_value] = [r["max_drawdown_pct"] for r in group_results]
            win_rates[param_value] = [r["win_rate"] for r in group_results]
            sharpe_proxies[param_value] = [
                r["sharpe_proxy"] for r in group_results 
                if r["sharpe_proxy"] is not None
            ]
        
        # Compute statistics
        def compute_stats(values_dict: Dict) -> Dict:
            stats = {}
            for param_value, values in values_dict.items():
                if values:
                    stats[param_value] = {
                        "mean": round(statistics.mean(values), 2),
                        "std": round(statistics.stdev(values) if len(values) > 1 else 0.0, 2),
                        "min": round(min(values), 2),
                        "max": round(max(values), 2)
                    }
            return stats
        
        # Check for sign flips (profitable → unprofitable)
        sign_flips = []
        return_values = [r["return_pct"] for r in all_results]
        profitable_count = sum(1 for r in return_values if r > 0)
        unprofitable_count = sum(1 for r in return_values if r < 0)
        
        # Check if any parameter value causes sign flip
        for param_value, group_results in param_groups.items():
            group_returns = [r["return_pct"] for r in group_results]
            group_profitable = sum(1 for r in group_returns if r > 0)
            group_unprofitable = sum(1 for r in group_returns if r < 0)
            
            # Sign flip if this parameter value causes different profitability than overall
            if profitable_count > 0 and unprofitable_count > 0:
                if group_profitable == 0 or group_unprofitable == 0:
                    sign_flips.append({
                        "parameter_value": param_value,
                        "all_profitable": group_profitable == len(group_returns),
                        "all_unprofitable": group_unprofitable == len(group_returns)
                    })
        
        # Determine stability
        return_stds = [stats["std"] for stats in compute_stats(return_pcts).values()]
        avg_return_std = statistics.mean(return_stds) if return_stds else 0.0
        
        # High variance threshold: std > 10% of mean absolute value
        all_return_abs = [abs(r) for r in return_values]
        mean_abs_return = statistics.mean(all_return_abs) if all_return_abs else 0.0
        variance_threshold = mean_abs_return * 0.1 if mean_abs_return > 0 else 10.0
        
        is_stable = avg_return_std < variance_threshold and len(sign_flips) == 0
        is_dangerous = avg_return_std > variance_threshold * 2 or len(sign_flips) > 0
        
        return {
            "parameter_name": param_name,
            "return_pct_stats": compute_stats(return_pcts),
            "drawdown_pct_stats": compute_stats(drawdown_pcts),
            "win_rate_stats": compute_stats(win_rates),
            "sharpe_proxy_stats": compute_stats(sharpe_proxies),
            "sign_flips": sign_flips,
            "stability": {
                "is_stable": is_stable,
                "is_dangerous": is_dangerous,
                "avg_return_std": round(avg_return_std, 2),
                "variance_threshold": round(variance_threshold, 2)
            }
        }
    
    def save_results(self, output_dir: str) -> Dict[str, str]:
        """
        Save parameter sensitivity results to files.
        
        Args:
            output_dir: Directory to save results
        
        Returns:
            Dict with paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_path / "parameter_sensitivity_results.csv"
        self._save_results_csv(csv_path)
        
        # Save JSON summary
        json_path = output_path / "parameter_sensitivity_summary.json"
        self._save_summary_json(json_path)
        
        return {
            "results_csv": str(csv_path),
            "summary_json": str(json_path)
        }
    
    def _save_results_csv(self, filepath: Path):
        """Save parameter sensitivity results to CSV."""
        completed_results = [r for r in self.results if r.get("status") == "completed"]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "combination_id",
                "ema_fast",
                "ema_slow",
                "stop_loss_atr_multiplier",
                "risk_per_trade_pct",
                "final_oos_equity",
                "net_pnl",
                "return_pct",
                "max_drawdown_pct",
                "max_drawdown_absolute",
                "win_rate",
                "num_oos_trades",
                "sharpe_proxy",
                "expectancy"
            ])
            
            for result in completed_results:
                writer.writerow([
                    result.get("combination_id", ""),
                    result.get("ema_fast", ""),
                    result.get("ema_slow", ""),
                    result.get("stop_loss_atr_multiplier", ""),
                    result.get("risk_per_trade_pct", ""),
                    result.get("final_oos_equity", ""),
                    result.get("net_pnl", ""),
                    result.get("return_pct", ""),
                    result.get("max_drawdown_pct", ""),
                    result.get("max_drawdown_absolute", ""),
                    result.get("win_rate", ""),
                    result.get("num_oos_trades", ""),
                    result.get("sharpe_proxy", ""),
                    result.get("expectancy", "")
                ])
    
    def _save_summary_json(self, filepath: Path):
        """Save parameter sensitivity summary to JSON."""
        sensitivity_summary = self._compute_sensitivity_summary()
        
        completed_results = [r for r in self.results if r.get("status") == "completed"]
        
        summary_data = {
            "total_combinations": len(self.results),
            "completed_combinations": len(completed_results),
            "failed_combinations": len(self.determinism_violations),
            "parameter_ranges": {
                "ema_fast": self.ema_fast_values,
                "ema_slow": self.ema_slow_values,
                "stop_loss_atr_multiplier": self.stop_loss_atr_multipliers,
                "risk_per_trade_pct": self.risk_per_trade_pcts
            },
            "sensitivity_analysis": sensitivity_summary,
            "determinism_violations": self.determinism_violations
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)


class CustomReplayEngine(ReplayEngine):
    """
    Custom ReplayEngine that accepts parameter overrides.
    Extends ReplayEngine to use custom EMA periods, stop-loss multiplier, and risk per trade.
    """
    
    def __init__(
        self,
        initial_equity: float = 100000.0,
        ema_fast: int = 20,
        ema_slow: int = 50,
        stop_loss_atr_multiplier: float = 2.0,
        risk_per_trade_pct: float = 0.5,
        commission_per_share: float = 0.005,
        commission_per_trade: float = 0.0,
        slippage: float = 0.0002
    ):
        """
        Initialize custom replay engine with parameter overrides.
        
        Args:
            initial_equity: Starting equity
            ema_fast: Fast EMA period (default: 20)
            ema_slow: Slow EMA period (default: 50)
            stop_loss_atr_multiplier: Stop-loss as multiple of ATR (default: 2.0)
            risk_per_trade_pct: Risk per trade as percentage (default: 0.5%)
            commission_per_share: Commission per share
            commission_per_trade: Commission per trade
            slippage: Slippage factor
        """
        # Initialize parent first
        super().__init__(
            initial_equity=initial_equity,
            commission_per_share=commission_per_share,
            commission_per_trade=commission_per_trade,
            slippage=slippage
        )
        
        # Store parameter overrides
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.risk_per_trade_pct = risk_per_trade_pct
    
    def start_replay(
        self,
        symbol: str,
        candles: List[Dict],
        replay_id: Optional[str] = None,
        source: Optional[str] = None,
        allowed_entry_regimes: Optional[List[str]] = None
    ) -> str:
        """
        Override start_replay to use custom ema_slow for validation.
        """
        if self.status == "running":
            raise ValueError("Replay already in progress")
        
        # Validate candles - need at least ema_slow candles (not hardcoded 50)
        if not candles or len(candles) < self.ema_slow:
            raise ValueError(f"Need at least {self.ema_slow} candles for replay (EMA({self.ema_slow}) requires {self.ema_slow} candles)")
        
        # Validate candle ordering
        for i in range(1, len(candles)):
            if candles[i]["time"] <= candles[i-1]["time"]:
                raise ValueError(f"Candles must be ordered by time. Candle {i} has timestamp {candles[i]['time']} <= previous {candles[i-1]['time']}")
        
        # Generate replay_id if not provided
        if replay_id is None:
            replay_id = str(uuid.uuid4())
        
        # Reset state
        self.reset()
        
        # Initialize
        self.replay_id = replay_id
        self.symbol = symbol
        self.candle_history = candles.copy()
        self.total_candles = len(candles)
        self.status = "running"
        self.source = source or "unknown"
        self.allowed_entry_regimes = allowed_entry_regimes
        
        if source:
            regime_gate_info = f", regime_gate: {allowed_entry_regimes}" if allowed_entry_regimes else ""
            print(f"Replay {replay_id} started with source: {source}, symbol: {symbol}, candles: {len(candles)}{regime_gate_info}")
        
        return replay_id
    
    def reset(self):
        """Override reset to use custom broker with parameter overrides."""
        from paper_broker import PaperBroker
        import math
        
        # Create custom broker with overridden calculate_position_size
        broker = PaperBroker(
            initial_equity=self.initial_equity,
            commission_per_share=self.commission_per_share,
            commission_per_trade=self.commission_per_trade,
            slippage=self.slippage
        )
        
        # Override calculate_position_size with custom risk_per_trade_pct
        risk_pct = self.risk_per_trade_pct
        
        def custom_calculate_position_size(entry_price: float, stop_distance: float, current_equity: float) -> int:
            """Custom position sizing with configurable risk per trade."""
            if stop_distance <= 0:
                return 0
            risk_amount = current_equity * (risk_pct / 100.0)
            shares = math.floor(risk_amount / stop_distance)
            return max(0, shares)
        
        broker.calculate_position_size = custom_calculate_position_size
        self.broker = broker
        
        # Call parent reset for other state
        super().reset()
    
    def process_candle(self, db_session, candle: Dict) -> Optional[Dict]:
        """
        Process a candle with custom parameters.
        Overrides parent method to use custom EMA periods and stop-loss multiplier.
        """
        # Process pending orders from previous candle at current candle open
        open_time = candle.get("open_time", candle["time"])
        executed_trades = self.broker.process_pending_orders(
            current_open_price=candle["open"],
            timestamp=open_time
        )
        
        # Update trade records (same as parent)
        for trade in executed_trades:
            if trade["action"] == "BUY":
                from utils import unix_to_utc_datetime, ensure_utc_datetime
                entry_time = unix_to_utc_datetime(trade["timestamp"])
                ensure_utc_datetime(entry_time, f"replay BUY entry time for {trade['symbol']}")
                
                if self.replay_id is None:
                    raise ValueError("Replay ID must be set for replay trades")
                
                trade_record = Trade(
                    symbol=trade["symbol"],
                    entry_time=entry_time,
                    entry_price=trade["entry_price"],
                    shares=trade["shares"],
                    exit_time=None,
                    exit_price=None,
                    pnl=None,
                    reason=None,
                    replay_id=self.replay_id
                )
                db_session.add(trade_record)
                db_session.commit()
                
                if trade["symbol"] == self.symbol:
                    self.position_state.has_position = True
                    self.position_state.entry_price = trade["entry_price"]
                    self.position_state.entry_time = trade["timestamp"]
            
            elif trade["action"] == "EXIT":
                from utils import unix_to_utc_datetime, ensure_utc_datetime
                open_trade = db_session.query(Trade).filter(
                    Trade.symbol == trade["symbol"],
                    Trade.exit_time.is_(None),
                    Trade.replay_id == self.replay_id
                ).first()
                
                if open_trade:
                    exit_time = unix_to_utc_datetime(trade["timestamp"])
                    ensure_utc_datetime(exit_time, f"replay EXIT exit time for {trade['symbol']}")
                    open_trade.exit_time = exit_time
                    open_trade.exit_price = trade["exit_price"]
                    open_trade.pnl = trade["pnl"]
                    open_trade.reason = trade["reason"]
                    db_session.commit()
                
                if trade["symbol"] == self.symbol:
                    self.position_state.has_position = False
                    self.position_state.entry_price = None
                    self.position_state.entry_time = None
        
        # Get history for indicator calculation
        current_history = self.candle_history[:self.current_candle_index]
        
        # Need at least ema_slow candles
        if len(current_history) < self.ema_slow:
            return None
        
        # Keep only last 500 candles for memory efficiency
        if len(current_history) > 500:
            current_history = current_history[-500:]
        
        # Extract price arrays
        closes = [c["close"] for c in current_history]
        highs = [c["high"] for c in current_history]
        lows = [c["low"] for c in current_history]
        
        # Calculate indicators with custom periods
        ema_fast_values = ema(closes, self.ema_fast)
        ema_slow_values = ema(closes, self.ema_slow)
        atr14_values = atr(highs, lows, closes, 14)
        
        # Get current ATR for position sizing
        current_atr = atr14_values[-1] if atr14_values and atr14_values[-1] is not None else None
        
        # Evaluate strategy with custom EMA periods
        from strategy import ema_trend_v1
        
        # Create custom strategy wrapper that uses custom EMA periods
        def custom_ema_trend_v1(candles, position_state):
            """Wrapper that uses custom EMA periods."""
            return ema_trend_v1(
                candles=candles,
                ema20_values=ema_fast_values,  # Use fast EMA
                ema50_values=ema_slow_values,   # Use slow EMA
                position_state=position_state
            )
        
        result = custom_ema_trend_v1(current_history, self.position_state)
        
        # Check kill switch
        current_prices_all = {self.symbol: candle["close"]}
        kill_switch_orders = self.broker.check_and_enforce_risk_controls(
            current_prices=current_prices_all,
            timestamp=candle.get("close_time", candle["time"])
        )
        
        # Process kill switch orders
        if kill_switch_orders:
            for order_dict in kill_switch_orders:
                self.broker.execute_exit(
                    symbol=order_dict["symbol"],
                    signal_price=order_dict["exit_price"],
                    timestamp=order_dict["timestamp"],
                    reason=order_dict["reason"]
                )
        
        # Handle BUY signal
        buy_allowed = True
        if self.allowed_entry_regimes:
            from regime_classifier import classify_regime
            current_regime = classify_regime(current_history, len(current_history) - 1)
            buy_allowed = current_regime in self.allowed_entry_regimes
        
        # Track stop-loss symbols (local variable like parent)
        stop_loss_symbols = set()
        for exit_order in kill_switch_orders:
            if exit_order.get("reason") == "STOP_LOSS":
                stop_loss_symbols.add(exit_order["symbol"])
        
        if result["signal"] == "BUY" and current_atr is not None and self.symbol not in stop_loss_symbols and buy_allowed:
            # Calculate stop distance with custom multiplier
            stop_distance = self.stop_loss_atr_multiplier * current_atr
            
            self.broker.execute_buy(
                symbol=self.symbol,
                signal_price=candle["close"],
                stop_distance=stop_distance,
                timestamp=candle.get("close_time", candle["time"])
            )
        
        # Handle EXIT signal
        elif result["signal"] == "EXIT":
            if self.symbol in self.broker.positions:
                self.broker.execute_exit(
                    symbol=self.symbol,
                    signal_price=candle["close"],
                    timestamp=candle.get("close_time", candle["time"]),
                    reason=result["reason"]
                )
        
        # Update equity before risk checks
        current_prices_all = {self.symbol: candle["close"]}
        self.broker.update_equity(current_prices_all)
        
        # Check stop-losses
        self.broker.check_stop_losses(
            current_prices=current_prices_all,
            timestamp=candle.get("close_time", candle["time"])
        )
        
        # Check and enforce risk controls (kill switch)
        self.broker.check_and_enforce_risk_controls(
            current_prices=current_prices_all,
            timestamp=candle.get("close_time", candle["time"])
        )
        
        return result
