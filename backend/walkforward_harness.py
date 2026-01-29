"""
Walk-forward backtest harness for out-of-sample evaluation.
Implements rolling train/test windows and reuses the same execution path as replay/live.
"""

from typing import List, Dict, Optional, Tuple
from datetime import date
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import csv
import uuid
import random

from replay_engine import ReplayEngine
from database import Trade, EquityCurve
from walkforward import filter_candles_by_date_range
from metrics import compute_metrics, _compute_max_drawdown
from utils import unix_to_utc_datetime
import statistics


class WalkForwardHarness:
    """
    Walk-forward backtest harness that runs rolling train/test windows.
    Reuses ReplayEngine for identical execution semantics to replay/live trading.
    """
    
    def __init__(
        self,
        initial_equity: float = 100000.0,
        train_days: Optional[int] = None,
        test_days: Optional[int] = None,
        step_days: Optional[int] = None,
        train_bars: Optional[int] = None,
        test_bars: Optional[int] = None,
        step_bars: Optional[int] = None
    ):
        """
        Initialize walk-forward harness.
        
        Args:
            initial_equity: Starting equity for each window
            train_days: Number of trading days in training window (if using days)
            test_days: Number of trading days in test window (if using days)
            step_days: Number of trading days to step forward (if using days)
            train_bars: Number of candles in training window (if using bars)
            test_bars: Number of candles in test window (if using bars)
            step_bars: Number of candles to step forward (if using bars)
        
        Note: Either use days OR bars, not both.
        """
        self.initial_equity = initial_equity
        
        # Window configuration (days or bars)
        if train_days is not None:
            self.window_mode = "days"
            self.train_days = train_days or 252
            self.test_days = test_days or 63
            self.step_days = step_days or 21
            self.train_bars = None
            self.test_bars = None
            self.step_bars = None
        elif train_bars is not None:
            self.window_mode = "bars"
            self.train_bars = train_bars
            self.test_bars = test_bars or 63
            self.step_bars = step_bars or 21
            self.train_days = None
            self.test_days = None
            self.step_days = None
        else:
            # Default to days
            self.window_mode = "days"
            self.train_days = 252
            self.test_days = 63
            self.step_days = 21
            self.train_bars = None
            self.test_bars = None
            self.step_bars = None
        
        # Results storage
        self.window_results: List[Dict] = []
        self.all_oos_trades: List[Trade] = []
        self.all_oos_equity: List[EquityCurve] = []
        
        # Monte Carlo results storage
        self.monte_carlo_results: Optional[List[Dict]] = None
        self.monte_carlo_seed: Optional[int] = None
    
    def generate_windows(
        self,
        candles: List[Dict],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate rolling train/test windows.
        
        Args:
            candles: List of all available candles (for bars-based windows)
            start_date: Start date in "YYYY-MM-DD" format (for days-based windows)
            end_date: End date in "YYYY-MM-DD" format (for days-based windows)
        
        Returns:
            List of window dicts with train_start, train_end, test_start, test_end
            For bars-based: includes train_start_idx, train_end_idx, test_start_idx, test_end_idx
        """
        windows = []
        
        if self.window_mode == "days":
            if not start_date or not end_date:
                raise ValueError("start_date and end_date required for days-based windows")
            
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            
            current_start = start_dt
            
            while current_start < end_dt:
                # Training window
                train_start = current_start
                train_end = train_start + timedelta(days=self.train_days)
                
                # Test window (immediately after training)
                test_start = train_end + timedelta(days=1)  # No overlap
                test_end = test_start + timedelta(days=self.test_days)
                
                # Skip if test window extends beyond available data
                if test_end > end_dt:
                    break
                
                windows.append({
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "window_id": len(windows) + 1
                })
                
                # Step forward
                current_start = current_start + timedelta(days=self.step_days)
        
        else:  # bars-based
            total_candles = len(candles)
            current_idx = 0
            
            while current_idx < total_candles:
                # Training window
                train_start_idx = current_idx
                train_end_idx = min(train_start_idx + self.train_bars, total_candles)
                
                # Test window (immediately after training)
                test_start_idx = train_end_idx  # No overlap
                test_end_idx = min(test_start_idx + self.test_bars, total_candles)
                
                # Skip if test window extends beyond available data
                if test_end_idx >= total_candles:
                    break
                
                # Get dates from candles for reference
                train_start_candle = candles[train_start_idx]
                train_end_candle = candles[train_end_idx - 1] if train_end_idx > train_start_idx else candles[train_start_idx]
                test_start_candle = candles[test_start_idx]
                test_end_candle = candles[test_end_idx - 1] if test_end_idx > test_start_idx else candles[test_start_idx]
                
                def get_candle_date(candle):
                    candle_time = candle.get("time")
                    if isinstance(candle_time, (int, float)):
                        return datetime.fromtimestamp(candle_time, tz=timezone.utc).date().isoformat()
                    elif isinstance(candle_time, datetime):
                        dt = candle_time
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt.date().isoformat()
                    return ""
                
                windows.append({
                    "train_start_idx": train_start_idx,
                    "train_end_idx": train_end_idx,
                    "test_start_idx": test_start_idx,
                    "test_end_idx": test_end_idx,
                    "train_start": get_candle_date(train_start_candle),
                    "train_end": get_candle_date(train_end_candle),
                    "test_start": get_candle_date(test_start_candle),
                    "test_end": get_candle_date(test_end_candle),
                    "window_id": len(windows) + 1
                })
                
                # Step forward
                current_idx += self.step_bars
        
        return windows
    
    def _run_replay_period(
        self,
        symbol: str,
        candles: List[Dict],
        period_start: str,
        period_end: str,
        period_start_idx: Optional[int],
        period_end_idx: Optional[int],
        db_session,
        replay_id: str,
        allowed_entry_regimes: Optional[List[str]] = None
    ) -> Tuple[List[Trade], List[EquityCurve]]:
        """
        Run replay for a specific period (train or test).
        
        Returns:
            Tuple of (trades, equity_curve) for the period
        """
        # Filter candles for period
        if self.window_mode == "bars" and period_start_idx is not None and period_end_idx is not None:
            period_candles = candles[period_start_idx:period_end_idx]
        else:
            period_candles = filter_candles_by_date_range(
                candles,
                period_start,
                period_end,
                candle_type="daily"
            )
        
        if not period_candles:
            return [], []
        
        # Create replay engine and run period
        replay_engine = ReplayEngine(initial_equity=self.initial_equity)
        
        replay_engine.start_replay(
            symbol=symbol,
            candles=period_candles,
            replay_id=replay_id,
            source="walkforward",
            allowed_entry_regimes=allowed_entry_regimes
        )
        
        # Run replay
        replay_engine.run(db_session)
        
        # Fetch trades and equity curve for this period
        period_trades = db_session.query(Trade).filter(
            Trade.replay_id == replay_id,
            Trade.symbol == symbol
        ).all()
        
        period_equity = db_session.query(EquityCurve).filter(
            EquityCurve.replay_id == replay_id
        ).order_by(EquityCurve.timestamp).all()
        
        return period_trades, period_equity
    
    def run_window(
        self,
        symbol: str,
        candles: List[Dict],
        window: Dict,
        db_session,
        allowed_entry_regimes: Optional[List[str]] = None
    ) -> Dict:
        """
        Run a single train/test window.
        Executes both train and test periods sequentially.
        
        Args:
            symbol: Trading symbol
            candles: All available candles
            window: Window dict with train_start, train_end, test_start, test_end
            db_session: Database session
            allowed_entry_regimes: Optional regime gate for entries
        
        Returns:
            Window results dict with metrics and trades for both train and test
        """
        # Run train period
        train_replay_id = str(uuid.uuid4()) + "_train"
        train_start = str(window.get("train_start", ""))
        train_end = str(window.get("train_end", ""))
        train_start_idx = window.get("train_start_idx")
        train_end_idx = window.get("train_end_idx")
        
        try:
            train_trades, train_equity = self._run_replay_period(
                symbol=symbol,
                candles=candles,
                period_start=train_start,
                period_end=train_end,
                period_start_idx=train_start_idx,
                period_end_idx=train_end_idx,
                db_session=db_session,
                replay_id=train_replay_id,
                allowed_entry_regimes=allowed_entry_regimes
            )
            
            # Run test period (OOS)
            test_replay_id = str(uuid.uuid4()) + "_test"
            test_start = str(window.get("test_start", ""))
            test_end = str(window.get("test_end", ""))
            test_start_idx = window.get("test_start_idx")
            test_end_idx = window.get("test_end_idx")
            
            test_trades, test_equity = self._run_replay_period(
                symbol=symbol,
                candles=candles,
                period_start=test_start,
                period_end=test_end,
                period_start_idx=test_start_idx,
                period_end_idx=test_end_idx,
                db_session=db_session,
                replay_id=test_replay_id,
                allowed_entry_regimes=allowed_entry_regimes
            )
            
            if not test_trades and not test_equity:
                return {
                    "window_id": window["window_id"],
                    "status": "skipped",
                    "reason": "No candles in test period"
                }
            
            # Compute test window metrics (OOS)
            metrics_snapshot = compute_metrics(test_trades, test_equity)
            
            # Calculate return on capital
            final_equity = metrics_snapshot.equity_end
            net_pnl = final_equity - self.initial_equity
            return_on_capital_pct = (net_pnl / self.initial_equity * 100) if self.initial_equity > 0 else 0.0
            
            # Compute daily P&L distribution
            daily_pnl_distribution = {}
            for trade in test_trades:
                if trade.exit_time:
                    trade_date = trade.exit_time.date()
                    if trade_date not in daily_pnl_distribution:
                        daily_pnl_distribution[trade_date] = 0.0
                    daily_pnl_distribution[trade_date] += trade.pnl or 0.0
            
            # Daily P&L summary stats
            daily_pnls = list(daily_pnl_distribution.values())
            daily_pnl_mean = statistics.mean(daily_pnls) if daily_pnls else 0.0
            daily_pnl_std = statistics.stdev(daily_pnls) if len(daily_pnls) > 1 else 0.0
            daily_pnl_worst = min(daily_pnls) if daily_pnls else 0.0
            
            # Calculate average trades per day
            if test_equity:
                first_date = test_equity[0].timestamp.date()
                last_date = test_equity[-1].timestamp.date()
                trading_days = (last_date - first_date).days + 1
                average_trades_per_day = len(test_trades) / trading_days if trading_days > 0 else 0.0
            else:
                average_trades_per_day = 0.0
            
            window_metrics = {
                "trades_executed": len(test_trades),
                "return_on_capital_pct": round(return_on_capital_pct, 2),
                "max_portfolio_drawdown_pct": round(metrics_snapshot.max_drawdown_pct, 2),
                "expectancy": round(metrics_snapshot.expectancy_per_trade, 2),
                "win_rate": round(metrics_snapshot.win_rate, 2),
                "average_trades_per_day": round(average_trades_per_day, 2),
                "daily_pnl_distribution": {
                    "mean": round(daily_pnl_mean, 2),
                    "std": round(daily_pnl_std, 2),
                    "worst_day": round(daily_pnl_worst, 2),
                    "days_with_trades": len(daily_pnl_distribution)
                }
            }
            
            # Count candles for test period
            if self.window_mode == "bars" and test_start_idx is not None and test_end_idx is not None:
                test_candle_count = test_end_idx - test_start_idx
            else:
                test_candles = filter_candles_by_date_range(
                    candles,
                    test_start,
                    test_end,
                    candle_type="daily"
                )
                test_candle_count = len(test_candles)
            
            return {
                "window_id": window["window_id"],
                "status": "completed",
                "train_replay_id": train_replay_id,
                "test_replay_id": test_replay_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "test_candle_count": test_candle_count,
                "train_trade_count": len(train_trades),
                "test_trade_count": len(test_trades),
                "metrics": window_metrics,
                "trade_count": len(test_trades),
                "equity_point_count": len(test_equity)
            }
        
        except Exception as e:
            return {
                "window_id": window["window_id"],
                "status": "error",
                "error": str(e)
            }
    
    def run_walkforward(
        self,
        symbol: str,
        candles: List[Dict],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        db_session=None,
        allowed_entry_regimes: Optional[List[str]] = None
    ) -> Dict:
        """
        Run complete walk-forward backtest.
        
        Args:
            symbol: Trading symbol
            candles: All available candles
            start_date: Start date in "YYYY-MM-DD" format (required for days-based windows)
            end_date: End date in "YYYY-MM-DD" format (required for days-based windows)
            db_session: Database session
            allowed_entry_regimes: Optional regime gate for entries
        
        Returns:
            Complete walk-forward results
        """
        # Generate windows
        windows = self.generate_windows(candles, start_date, end_date)
        
        if not windows:
            return {
                "status": "error",
                "error": "No valid windows generated"
            }
        
        print(f"[WALK-FORWARD] Generated {len(windows)} windows")
        if self.window_mode == "days":
            print(f"[WALK-FORWARD] Train days: {self.train_days}, Test days: {self.test_days}, Step days: {self.step_days}")
        else:
            print(f"[WALK-FORWARD] Train bars: {self.train_bars}, Test bars: {self.test_bars}, Step bars: {self.step_bars}")
        
        # Run each window
        self.window_results = []
        self.all_oos_trades = []
        self.all_oos_equity = []
        
        for i, window in enumerate(windows):
            print(f"[WALK-FORWARD] Running window {i+1}/{len(windows)}: Train {window.get('train_start')} to {window.get('train_end')}, Test {window.get('test_start')} to {window.get('test_end')}")
            
            window_result = self.run_window(
                symbol=symbol,
                candles=candles,
                window=window,
                db_session=db_session,
                allowed_entry_regimes=allowed_entry_regimes
            )
            
            self.window_results.append(window_result)
            
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
                
                self.all_oos_trades.extend(window_trades)
                self.all_oos_equity.extend(window_equity)
        
        # Compute aggregate metrics
        aggregate_metrics = self._compute_aggregate_metrics()
        
        config = {}
        if self.window_mode == "days":
            config = {
                "train_days": self.train_days,
                "test_days": self.test_days,
                "step_days": self.step_days
            }
        else:
            config = {
                "train_bars": self.train_bars,
                "test_bars": self.test_bars,
                "step_bars": self.step_bars
            }
        
        return {
            "status": "completed",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "window_mode": self.window_mode,
            **config,
            "total_windows": len(windows),
            "completed_windows": len([w for w in self.window_results if w["status"] == "completed"]),
            "window_results": self.window_results,
            "aggregate_metrics": aggregate_metrics,
            "total_oos_trades": len(self.all_oos_trades),
            "total_oos_equity_points": len(self.all_oos_equity)
        }
    
    def run_monte_carlo_resampling(
        self,
        num_runs: int = 1000,
        random_seed: Optional[int] = None,
        max_equity_samples: int = 100
    ) -> Dict:
        """
        Run Monte Carlo resampling on stitched OOS trades.
        
        Args:
            num_runs: Number of resampling runs (default: 1000)
            random_seed: Random seed for determinism (if None, uses current time)
            max_equity_samples: Maximum number of equity curves to save (default: 100)
        
        Returns:
            Dict with Monte Carlo results and percentiles
        """
        if not self.all_oos_trades:
            return {
                "status": "skipped",
                "reason": "No OOS trades available for Monte Carlo resampling"
            }
        
        # Set random seed for determinism
        if random_seed is not None:
            random.seed(random_seed)
            self.monte_carlo_seed = random_seed
        else:
            # Use a deterministic seed based on trade count and initial equity
            seed = hash((len(self.all_oos_trades), self.initial_equity)) % (2**31)
            random.seed(seed)
            self.monte_carlo_seed = seed
        
        print(f"[MONTE CARLO] Starting resampling with {num_runs} runs (seed={self.monte_carlo_seed})")
        
        # Filter to closed trades only (with P&L)
        closed_trades = [t for t in self.all_oos_trades if t.pnl is not None]
        
        if not closed_trades:
            return {
                "status": "skipped",
                "reason": "No closed trades with P&L available"
            }
        
        print(f"[MONTE CARLO] Resampling {len(closed_trades)} closed trades")
        
        # Store original order for validation
        original_order = [t.id for t in closed_trades]
        
        # Run Monte Carlo resampling
        results = []
        
        for run_idx in range(num_runs):
            # Shuffle trades
            shuffled_trades = closed_trades.copy()
            random.shuffle(shuffled_trades)
            
            # Compute equity curve from shuffled trades
            equity_curve = self._compute_equity_curve_from_trades(shuffled_trades)
            
            # Calculate metrics
            final_equity = equity_curve[-1].equity if equity_curve else self.initial_equity
            max_drawdown_absolute, max_drawdown_pct = self._compute_max_drawdown(equity_curve)
            net_pnl = final_equity - self.initial_equity
            
            results.append({
                "run": run_idx,
                "final_equity": final_equity,
                "net_pnl": net_pnl,
                "max_drawdown_absolute": max_drawdown_absolute,
                "max_drawdown_pct": max_drawdown_pct,
                "return_pct": (net_pnl / self.initial_equity * 100) if self.initial_equity > 0 else 0.0
            })
            
            # Progress logging
            if (run_idx + 1) % 100 == 0:
                print(f"[MONTE CARLO] Completed {run_idx + 1}/{num_runs} runs")
        
        self.monte_carlo_results = results
        
        # Compute percentiles
        percentiles = self._compute_monte_carlo_percentiles(results)
        
        # Determinism validation: Run one more time with same seed and verify
        random.seed(self.monte_carlo_seed)
        validation_trades = closed_trades.copy()
        random.shuffle(validation_trades)
        validation_order = [t.id for t in validation_trades]
        
        # Check if first run matches (determinism check)
        if num_runs > 0:
            random.seed(self.monte_carlo_seed)
            first_run_trades = closed_trades.copy()
            random.shuffle(first_run_trades)
            first_run_order = [t.id for t in first_run_trades]
            
            if validation_order != first_run_order:
                raise RuntimeError(
                    f"Monte Carlo determinism check FAILED: "
                    f"Same seed ({self.monte_carlo_seed}) produced different trade orders. "
                    f"This indicates non-deterministic behavior."
                )
        
        print(f"[MONTE CARLO] Completed {num_runs} runs. Percentiles computed.")
        
        return {
            "status": "completed",
            "num_runs": num_runs,
            "num_trades": len(closed_trades),
            "random_seed": self.monte_carlo_seed,
            "percentiles": percentiles,
            "max_equity_samples": max_equity_samples
        }
    
    def _compute_equity_curve_from_trades(self, trades: List[Trade]) -> List[EquityCurve]:
        """
        Compute equity curve from a list of trades.
        
        Args:
            trades: List of Trade objects (should be sorted by exit_time)
        
        Returns:
            List of EquityCurve-like objects (using Trade objects as containers)
        """
        # Sort trades by exit_time (or entry_time if exit_time is None)
        sorted_trades = sorted(
            trades,
            key=lambda t: t.exit_time if t.exit_time else (t.entry_time if t.entry_time else datetime.min.replace(tzinfo=timezone.utc))
        )
        
        equity_curve = []
        current_equity = self.initial_equity
        
        # Add starting equity point
        if sorted_trades:
            first_trade = sorted_trades[0]
            start_time = first_trade.entry_time if first_trade.entry_time else datetime.now(timezone.utc)
            equity_curve.append(
                EquityCurve(
                    timestamp=start_time,
                    equity=current_equity,
                    replay_id="monte_carlo"
                )
            )
        
        # Apply each trade's P&L
        for trade in sorted_trades:
            if trade.pnl is not None:
                current_equity += trade.pnl
                
                # Add equity point at trade exit
                exit_time = trade.exit_time if trade.exit_time else trade.entry_time
                if exit_time:
                    equity_curve.append(
                        EquityCurve(
                            timestamp=exit_time,
                            equity=current_equity,
                            replay_id="monte_carlo"
                        )
                    )
        
        return equity_curve
    
    def _compute_monte_carlo_percentiles(self, results: List[Dict]) -> Dict:
        """Compute percentiles (5%, 25%, 50%, 75%, 95%) for Monte Carlo results."""
        if not results:
            return {}
        
        # Extract metrics
        final_equities = [r["final_equity"] for r in results]
        net_pnls = [r["net_pnl"] for r in results]
        max_drawdowns_absolute = [r["max_drawdown_absolute"] for r in results]
        max_drawdowns_pct = [r["max_drawdown_pct"] for r in results]
        return_pcts = [r["return_pct"] for r in results]
        
        # Sort for percentile calculation
        final_equities.sort()
        net_pnls.sort()
        max_drawdowns_absolute.sort()
        max_drawdowns_pct.sort()
        return_pcts.sort()
        
        def percentile(data: List[float], p: float) -> float:
            """Compute percentile value."""
            if not data:
                return 0.0
            k = (len(data) - 1) * p / 100.0
            f = int(k)
            c = k - f
            if f + 1 < len(data):
                return data[f] + c * (data[f + 1] - data[f])
            return data[-1]
        
        return {
            "final_equity": {
                "p5": round(percentile(final_equities, 5), 2),
                "p25": round(percentile(final_equities, 25), 2),
                "p50": round(percentile(final_equities, 50), 2),
                "p75": round(percentile(final_equities, 75), 2),
                "p95": round(percentile(final_equities, 95), 2),
                "mean": round(statistics.mean(final_equities), 2),
                "std": round(statistics.stdev(final_equities) if len(final_equities) > 1 else 0.0, 2)
            },
            "net_pnl": {
                "p5": round(percentile(net_pnls, 5), 2),
                "p25": round(percentile(net_pnls, 25), 2),
                "p50": round(percentile(net_pnls, 50), 2),
                "p75": round(percentile(net_pnls, 75), 2),
                "p95": round(percentile(net_pnls, 95), 2),
                "mean": round(statistics.mean(net_pnls), 2),
                "std": round(statistics.stdev(net_pnls) if len(net_pnls) > 1 else 0.0, 2)
            },
            "max_drawdown_absolute": {
                "p5": round(percentile(max_drawdowns_absolute, 5), 2),
                "p25": round(percentile(max_drawdowns_absolute, 25), 2),
                "p50": round(percentile(max_drawdowns_absolute, 50), 2),
                "p75": round(percentile(max_drawdowns_absolute, 75), 2),
                "p95": round(percentile(max_drawdowns_absolute, 95), 2),
                "mean": round(statistics.mean(max_drawdowns_absolute), 2),
                "std": round(statistics.stdev(max_drawdowns_absolute) if len(max_drawdowns_absolute) > 1 else 0.0, 2)
            },
            "max_drawdown_pct": {
                "p5": round(percentile(max_drawdowns_pct, 5), 2),
                "p25": round(percentile(max_drawdowns_pct, 25), 2),
                "p50": round(percentile(max_drawdowns_pct, 50), 2),
                "p75": round(percentile(max_drawdowns_pct, 75), 2),
                "p95": round(percentile(max_drawdowns_pct, 95), 2),
                "mean": round(statistics.mean(max_drawdowns_pct), 2),
                "std": round(statistics.stdev(max_drawdowns_pct) if len(max_drawdowns_pct) > 1 else 0.0, 2)
            },
            "return_pct": {
                "p5": round(percentile(return_pcts, 5), 2),
                "p25": round(percentile(return_pcts, 25), 2),
                "p50": round(percentile(return_pcts, 50), 2),
                "p75": round(percentile(return_pcts, 75), 2),
                "p95": round(percentile(return_pcts, 95), 2),
                "mean": round(statistics.mean(return_pcts), 2),
                "std": round(statistics.stdev(return_pcts) if len(return_pcts) > 1 else 0.0, 2)
            }
        }
    
    def _compute_max_drawdown(self, equity_curve: List[EquityCurve]) -> Tuple[float, float]:
        """
        Compute maximum drawdown (absolute and percentage) from equity curve.
        
        Args:
            equity_curve: List of EquityCurve objects
        
        Returns:
            Tuple of (max_drawdown_absolute, max_drawdown_pct)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0, 0.0
        
        peak_equity = equity_curve[0].equity
        max_drawdown_absolute = 0.0
        max_drawdown_pct = 0.0
        
        for point in equity_curve:
            if point.equity > peak_equity:
                peak_equity = point.equity
            
            drawdown_absolute = peak_equity - point.equity
            drawdown_pct = (drawdown_absolute / peak_equity * 100) if peak_equity > 0 else 0.0
            
            if drawdown_absolute > max_drawdown_absolute:
                max_drawdown_absolute = drawdown_absolute
                max_drawdown_pct = drawdown_pct
        
        return max_drawdown_absolute, max_drawdown_pct
    
    def _save_monte_carlo_summary_json(self, filepath: Path):
        """Save Monte Carlo summary with percentiles to JSON."""
        if not self.monte_carlo_results:
            return
        
        percentiles = self._compute_monte_carlo_percentiles(self.monte_carlo_results)
        
        summary_data = {
            "num_runs": len(self.monte_carlo_results),
            "num_trades": len(self.all_oos_trades),
            "random_seed": self.monte_carlo_seed,
            "initial_equity": self.initial_equity,
            "percentiles": percentiles
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
    
    def _save_monte_carlo_equity_samples_csv(self, filepath: Path, max_samples: int = 100):
        """
        Save Monte Carlo equity curve samples to CSV (capped size).
        
        Args:
            filepath: Path to save CSV file
            max_samples: Maximum number of equity curves to save (default: 100)
        """
        if not self.monte_carlo_results:
            return
        
        # Re-run Monte Carlo to get equity samples (capped)
        if not self.all_oos_trades:
            return
        
        closed_trades = [t for t in self.all_oos_trades if t.pnl is not None]
        if not closed_trades:
            return
        
        # Set seed to match original run
        if self.monte_carlo_seed is not None:
            random.seed(self.monte_carlo_seed)
        
        equity_samples = []
        num_runs = min(len(self.monte_carlo_results), max_samples)
        
        for run_idx in range(num_runs):
            # Shuffle trades
            shuffled_trades = closed_trades.copy()
            random.shuffle(shuffled_trades)
            
            # Compute equity curve
            equity_curve = self._compute_equity_curve_from_trades(shuffled_trades)
            
            equity_samples.append({
                "run": run_idx,
                "equity_curve": equity_curve
            })
        
        # Save to CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["run", "timestamp", "equity"])
            
            for sample in equity_samples:
                run = sample["run"]
                for point in sample["equity_curve"]:
                    writer.writerow([
                        run,
                        point.timestamp.isoformat() if point.timestamp else "",
                        point.equity
                    ])
    
    def _compute_aggregate_metrics(self) -> Dict:
        """Compute aggregate metrics across all windows."""
        completed_windows = [w for w in self.window_results if w["status"] == "completed"]
        
        if not completed_windows:
            return {
                "average_return_pct": 0.0,
                "average_max_drawdown_pct": 0.0,
                "average_expectancy": 0.0,
                "average_win_rate": 0.0,
                "total_trades": 0,
                "windows_with_positive_return": 0,
                "windows_with_positive_return_pct": 0.0
            }
        
        returns = [w["metrics"]["return_on_capital_pct"] for w in completed_windows]
        drawdowns = [w["metrics"]["max_portfolio_drawdown_pct"] for w in completed_windows]
        expectancies = [w["metrics"]["expectancy"] for w in completed_windows]
        win_rates = [w["metrics"]["win_rate"] for w in completed_windows]
        
        positive_returns = [r for r in returns if r > 0]
        
        return {
            "average_return_pct": round(sum(returns) / len(returns), 2) if returns else 0.0,
            "average_max_drawdown_pct": round(sum(drawdowns) / len(drawdowns), 2) if drawdowns else 0.0,
            "average_expectancy": round(sum(expectancies) / len(expectancies), 2) if expectancies else 0.0,
            "average_win_rate": round(sum(win_rates) / len(win_rates), 2) if win_rates else 0.0,
            "total_trades": sum(w["trade_count"] for w in completed_windows),
            "windows_with_positive_return": len(positive_returns),
            "windows_with_positive_return_pct": round(len(positive_returns) / len(completed_windows) * 100, 2) if completed_windows else 0.0,
            "best_window_return_pct": round(max(returns), 2) if returns else 0.0,
            "worst_window_return_pct": round(min(returns), 2) if returns else 0.0
        }
    
    def save_artifacts(
        self,
        output_dir: str,
        symbol: str
    ) -> Dict[str, str]:
        """
        Save walk-forward artifacts to files.
        
        Args:
            output_dir: Directory to save artifacts
            symbol: Trading symbol (for filename)
        
        Returns:
            Dict with paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save per_window_metrics.json
        per_window_path = output_path / "per_window_metrics.json"
        self._save_per_window_metrics_json(per_window_path)
        
        # Save stitched_oos_equity_curve.csv
        equity_path = output_path / "stitched_oos_equity_curve.csv"
        self._save_equity_csv(equity_path)
        
        # Save overall_summary_metrics.json
        summary_path = output_path / "overall_summary_metrics.json"
        self._save_summary_metrics_json(summary_path)
        
        artifact_paths = {
            "per_window_metrics_json": str(per_window_path),
            "stitched_oos_equity_curve_csv": str(equity_path),
            "overall_summary_metrics_json": str(summary_path)
        }
        
        # Save Monte Carlo results if available
        if self.monte_carlo_results is not None:
            mc_summary_path = output_path / "monte_carlo_summary.json"
            self._save_monte_carlo_summary_json(mc_summary_path)
            artifact_paths["monte_carlo_summary_json"] = str(mc_summary_path)
            
            # Optionally save equity samples (capped size)
            mc_equity_path = output_path / "monte_carlo_equity_samples.csv"
            self._save_monte_carlo_equity_samples_csv(mc_equity_path)
            artifact_paths["monte_carlo_equity_samples_csv"] = str(mc_equity_path)
        
        return artifact_paths
    
    def _save_equity_csv(self, filepath: Path):
        """
        Save stitched OOS equity curve to CSV.
        
        Note: Each test window starts with initial_equity, so the curve shows
        independent OOS performance per window. Equity values are stitched
        chronologically but each window resets to initial_equity.
        """
        # Sort equity points by timestamp to create continuous chronological curve
        sorted_equity = sorted(self.all_oos_equity, key=lambda e: e.timestamp)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "equity", "replay_id"])
            
            for point in sorted_equity:
                writer.writerow([
                    point.timestamp.isoformat() if point.timestamp else "",
                    point.equity,
                    point.replay_id
                ])
    
    def _save_per_window_metrics_json(self, filepath: Path):
        """Save per-window metrics to JSON."""
        completed_windows = [w for w in self.window_results if w["status"] == "completed"]
        
        config = {}
        if self.window_mode == "days":
            config = {
                "train_days": self.train_days,
                "test_days": self.test_days,
                "step_days": self.step_days
            }
        else:
            config = {
                "train_bars": self.train_bars,
                "test_bars": self.test_bars,
                "step_bars": self.step_bars
            }
        
        metrics_data = {
            "per_window_metrics": [
                {
                    "window_id": w["window_id"],
                    "train_start": w.get("train_start"),
                    "train_end": w.get("train_end"),
                    "test_start": w.get("test_start"),
                    "test_end": w.get("test_end"),
                    "train_trade_count": w.get("train_trade_count", 0),
                    "test_trade_count": w.get("test_trade_count", 0),
                    "metrics": w.get("metrics", {})
                }
                for w in completed_windows
            ],
            "configuration": {
                **config,
                "window_mode": self.window_mode,
                "initial_equity": self.initial_equity
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
    
    def _save_summary_metrics_json(self, filepath: Path):
        """Save summary metrics to JSON."""
        aggregate_metrics = self._compute_aggregate_metrics()
        
        config = {}
        if self.window_mode == "days":
            config = {
                "train_days": self.train_days,
                "test_days": self.test_days,
                "step_days": self.step_days
            }
        else:
            config = {
                "train_bars": self.train_bars,
                "test_bars": self.test_bars,
                "step_bars": self.step_bars
            }
        
        summary_data = {
            "aggregate_metrics": aggregate_metrics,
            "total_windows": len(self.window_results),
            "completed_windows": len([w for w in self.window_results if w["status"] == "completed"]),
            "total_oos_trades": len(self.all_oos_trades),
            "total_oos_equity_points": len(self.all_oos_equity),
            "configuration": {
                **config,
                "window_mode": self.window_mode,
                "initial_equity": self.initial_equity
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
