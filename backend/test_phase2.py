"""
Tests for Phase 2: Execution Engine Hardening.

Tests:
- Decision logging (log_decision, get_decisions, count_decisions)
- Anomaly detection (gap, volume spike, price outlier, staleness, invalid OHLC)
- Live trading loop (evaluation pipeline, anomaly pausing, status tracking)
"""

import unittest
import json
import math
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

from database import Base, DecisionLog, init_db, engine, SessionLocal
from decision_log import log_decision, get_decisions, count_decisions
from anomaly_detector import (
    detect_anomalies,
    AnomalyConfig,
    AnomalyResult,
)
from live_trading_loop import LiveTradingLoop, LoopStatus


class TestDecisionLog(unittest.TestCase):
    """Test decision logging module."""

    def setUp(self):
        """Create a fresh in-memory database for each test."""
        # Drop and recreate all tables
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    def test_log_decision_creates_record(self):
        """log_decision should create a DecisionLog record with all fields."""
        now = datetime.now(timezone.utc)
        candle = {"open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 1000000}

        entry = log_decision(
            self.db,
            timestamp=now,
            symbol="SPY",
            strategy_name="ema_trend_v1",
            candle=candle,
            indicator_snapshot={"ema20": 101.5, "ema50": 100.2},
            signal="BUY",
            signal_reason="EMA crossover",
            stop_distance=2.5,
            has_position=False,
            equity=100000.0,
            cash=100000.0,
            daily_pnl=0.0,
            trade_blocked=False,
            pending_orders_count=0,
            broker_action="BUY",
        )
        self.db.commit()

        self.assertIsNotNone(entry.id)
        self.assertEqual(entry.symbol, "SPY")
        self.assertEqual(entry.signal, "BUY")
        self.assertEqual(entry.candle_close, 103.0)
        self.assertEqual(entry.candle_volume, 1000000)
        self.assertEqual(entry.has_position, 0)
        self.assertEqual(entry.broker_action, "BUY")

        # Check indicator snapshot stored as JSON
        indicators = json.loads(entry.indicator_values)
        self.assertEqual(indicators["ema20"], 101.5)

    def test_log_decision_with_anomalies(self):
        """Decision log should record anomaly flags."""
        now = datetime.now(timezone.utc)
        candle = {"open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 1000000}

        entry = log_decision(
            self.db,
            timestamp=now,
            symbol="SPY",
            strategy_name="ema_trend_v1",
            candle=candle,
            signal="HOLD",
            anomaly_flags=["GAP", "VOLUME_SPIKE"],
            broker_action="BLOCKED",
            broker_action_reason="Anomaly detected",
        )
        self.db.commit()

        flags = json.loads(entry.anomaly_flags)
        self.assertIn("GAP", flags)
        self.assertIn("VOLUME_SPIKE", flags)

    def test_get_decisions_filters_by_symbol(self):
        """get_decisions should filter by symbol."""
        now = datetime.now(timezone.utc)
        candle = {"open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 1000000}

        log_decision(self.db, timestamp=now, symbol="SPY", strategy_name="ema_trend_v1",
                     candle=candle, signal="HOLD")
        log_decision(self.db, timestamp=now, symbol="AAPL", strategy_name="ema_trend_v1",
                     candle=candle, signal="BUY")
        self.db.commit()

        spy_decisions = get_decisions(self.db, symbol="SPY")
        self.assertEqual(len(spy_decisions), 1)
        self.assertEqual(spy_decisions[0]["symbol"], "SPY")

    def test_count_decisions(self):
        """count_decisions should return correct count."""
        now = datetime.now(timezone.utc)
        candle = {"open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 1000000}

        for _ in range(5):
            log_decision(self.db, timestamp=now, symbol="SPY", strategy_name="ema_trend_v1",
                         candle=candle, signal="HOLD")
        self.db.commit()

        self.assertEqual(count_decisions(self.db, symbol="SPY"), 5)

    def test_replay_isolation(self):
        """Decision logs with replay_id should be isolated from live."""
        now = datetime.now(timezone.utc)
        candle = {"open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 1000000}

        # Live decision
        log_decision(self.db, timestamp=now, symbol="SPY", strategy_name="ema_trend_v1",
                     candle=candle, signal="HOLD", replay_id=None)
        # Replay decision
        log_decision(self.db, timestamp=now, symbol="SPY", strategy_name="ema_trend_v1",
                     candle=candle, signal="BUY", replay_id="test-replay-123")
        self.db.commit()

        # Live only
        live = get_decisions(self.db, replay_id=None)
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]["signal"], "HOLD")

        # Replay only
        replay = get_decisions(self.db, replay_id="test-replay-123")
        self.assertEqual(len(replay), 1)
        self.assertEqual(replay[0]["signal"], "BUY")


class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detection module."""

    def _make_candle(self, close, volume=1000000, open_=None, high=None, low=None, close_time=None):
        """Helper to create a candle dict."""
        o = open_ or close * 0.999
        h = high or close * 1.001
        l = low or close * 0.998
        return {
            "open": o, "high": h, "low": l, "close": close,
            "volume": volume, "close_time": close_time,
        }

    def _make_history(self, n, base_price=100.0, base_volume=1000000):
        """Create n candles with slight random variation."""
        candles = []
        for i in range(n):
            price = base_price + (i * 0.1)
            candles.append(self._make_candle(price, base_volume))
        return candles

    def test_no_anomalies_normal_candle(self):
        """Normal candle should not trigger any anomalies."""
        history = self._make_history(50)
        current = self._make_candle(105.1)  # Normal continuation

        result = detect_anomalies(current, history)
        self.assertFalse(result.is_anomalous)
        self.assertEqual(result.flags, [])
        self.assertEqual(result.severity, "none")

    def test_gap_detection(self):
        """Gap > threshold should be detected."""
        history = self._make_history(5, base_price=100.0)
        # Gap up 10% from last close (~100.4)
        current = self._make_candle(115.0, open_=110.5, high=116.0, low=110.0)

        config = AnomalyConfig(gap_threshold_pct=5.0)
        result = detect_anomalies(current, history, config=config)

        self.assertTrue(result.is_anomalous)
        self.assertIn("GAP", result.flags)
        self.assertEqual(result.details["gap"]["direction"], "up")

    def test_gap_down_detection(self):
        """Gap down > threshold should be detected."""
        history = self._make_history(5, base_price=100.0)
        current = self._make_candle(90.0, open_=90.5, high=91.0, low=89.5)

        config = AnomalyConfig(gap_threshold_pct=5.0)
        result = detect_anomalies(current, history, config=config)

        self.assertTrue(result.is_anomalous)
        self.assertIn("GAP", result.flags)
        self.assertEqual(result.details["gap"]["direction"], "down")

    def test_normal_gap_not_flagged(self):
        """Small gap (< threshold) should not be flagged."""
        history = self._make_history(5, base_price=100.0)
        # 2% gap — under 5% threshold
        current = self._make_candle(102.5, open_=102.4, high=103.0, low=102.0)

        config = AnomalyConfig(gap_threshold_pct=5.0)
        result = detect_anomalies(current, history, config=config)

        self.assertNotIn("GAP", result.flags)

    def test_volume_spike_detection(self):
        """Volume > N × average should be detected."""
        history = self._make_history(20, base_volume=1000000)
        current = self._make_candle(105.0, volume=15000000)  # 15× average

        config = AnomalyConfig(volume_spike_multiplier=10.0)
        result = detect_anomalies(current, history, config=config)

        self.assertTrue(result.is_anomalous)
        self.assertIn("VOLUME_SPIKE", result.flags)

    def test_normal_volume_not_flagged(self):
        """Normal volume should not be flagged."""
        history = self._make_history(20, base_volume=1000000)
        current = self._make_candle(105.0, volume=2000000)  # 2× average

        config = AnomalyConfig(volume_spike_multiplier=10.0)
        result = detect_anomalies(current, history, config=config)

        self.assertNotIn("VOLUME_SPIKE", result.flags)

    def test_price_outlier_detection(self):
        """Price outside 3σ should be detected."""
        # Create stable history around 100
        history = [self._make_candle(100.0 + (i % 3) * 0.1) for i in range(50)]
        # Sudden jump to 120 (way outside 3σ)
        current = self._make_candle(120.0, open_=119.0, high=121.0, low=118.0)

        config = AnomalyConfig(price_sigma=3.0, price_lookback=50)
        result = detect_anomalies(current, history, config=config)

        self.assertTrue(result.is_anomalous)
        self.assertIn("PRICE_OUTLIER", result.flags)

    def test_stale_data_detection(self):
        """Candle older than threshold should be detected as stale."""
        ts = int((datetime.now(timezone.utc) - timedelta(hours=48)).timestamp())
        current = self._make_candle(100.0, close_time=ts)

        config = AnomalyConfig(staleness_hours=26.0)
        now = datetime.now(timezone.utc)
        result = detect_anomalies(current, [], config=config, current_time=now, is_daily=True)

        self.assertTrue(result.is_anomalous)
        self.assertIn("STALE_DATA", result.flags)
        self.assertEqual(result.severity, "critical")

    def test_invalid_ohlc_detection(self):
        """Invalid OHLC relationships should be detected."""
        # Low > High
        current = {"open": 100.0, "high": 99.0, "low": 101.0, "close": 100.0, "volume": 1000000}

        result = detect_anomalies(current, [])
        self.assertTrue(result.is_anomalous)
        self.assertIn("INVALID_OHLC", result.flags)
        self.assertEqual(result.severity, "critical")

    def test_zero_volume_detection(self):
        """Zero volume should be detected."""
        current = self._make_candle(100.0, volume=0)

        result = detect_anomalies(current, [])
        self.assertTrue(result.is_anomalous)
        self.assertIn("ZERO_VOLUME", result.flags)

    def test_multiple_anomalies(self):
        """Multiple anomalies can be detected simultaneously."""
        history = self._make_history(50, base_price=100.0, base_volume=1000000)
        # Gap + volume spike + zero volume is impossible, but gap + volume spike works
        current = {
            "open": 112.0,  # 12% gap from ~105 prev close
            "high": 113.0,
            "low": 111.0,
            "close": 112.5,
            "volume": 50000000,  # 50× average
        }

        config = AnomalyConfig(gap_threshold_pct=5.0, volume_spike_multiplier=10.0)
        result = detect_anomalies(current, history, config=config)

        self.assertTrue(result.is_anomalous)
        self.assertIn("GAP", result.flags)
        self.assertIn("VOLUME_SPIKE", result.flags)

    def test_severity_warning_for_non_critical(self):
        """Non-critical anomalies should have 'warning' severity."""
        history = self._make_history(20, base_volume=1000000)
        current = self._make_candle(105.0, volume=15000000)

        result = detect_anomalies(current, history)
        self.assertEqual(result.severity, "warning")

    def test_severity_critical_for_invalid_ohlc(self):
        """INVALID_OHLC should have 'critical' severity."""
        current = {"open": 100.0, "high": 99.0, "low": 101.0, "close": 100.0, "volume": 1000}
        result = detect_anomalies(current, [])
        self.assertEqual(result.severity, "critical")


class TestLiveTradingLoop(unittest.TestCase):
    """Test live trading loop."""

    def setUp(self):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    def test_initialization(self):
        """Loop should initialize with correct defaults."""
        loop = LiveTradingLoop(symbol="SPY", strategy_name="ema_trend_v1")

        self.assertEqual(loop.state.symbol, "SPY")
        self.assertEqual(loop.state.strategy_name, "ema_trend_v1")
        self.assertEqual(loop.state.status, LoopStatus.IDLE)
        self.assertIsNotNone(loop.broker)
        self.assertEqual(loop.broker.equity, 100000.0)

    def test_invalid_strategy_raises(self):
        """Initializing with unknown strategy should raise."""
        with self.assertRaises(KeyError):
            LiveTradingLoop(strategy_name="nonexistent_strategy")

    def test_get_status(self):
        """get_status should return complete state info."""
        loop = LiveTradingLoop(symbol="AAPL", strategy_name="ema_trend_v1")
        status = loop.get_status()

        self.assertEqual(status["status"], "idle")
        self.assertEqual(status["symbol"], "AAPL")
        self.assertEqual(status["strategy_name"], "ema_trend_v1")
        self.assertIn("broker", status)
        self.assertEqual(status["broker"]["equity"], 100000.0)
        self.assertEqual(status["evaluations_count"], 0)

    def test_load_history(self):
        """load_history should populate candle history."""
        loop = LiveTradingLoop(symbol="SPY", strategy_name="ema_trend_v1")

        candles = [{"open": 100 + i, "high": 101 + i, "low": 99 + i,
                    "close": 100.5 + i, "volume": 1000000,
                    "time": 1000000 + i * 86400, "open_time": 1000000 + i * 86400 - 3600,
                    "close_time": 1000000 + i * 86400}
                   for i in range(100)]

        loop.load_history(candles)
        self.assertEqual(len(loop.candle_history), 100)

    def test_load_history_caps_at_500(self):
        """load_history should keep only last 500 candles."""
        loop = LiveTradingLoop(symbol="SPY", strategy_name="ema_trend_v1")

        candles = [{"open": 100 + i, "high": 101 + i, "low": 99 + i,
                    "close": 100.5 + i, "volume": 1000000,
                    "time": 1000000 + i * 86400, "open_time": 1000000 + i * 86400 - 3600,
                    "close_time": 1000000 + i * 86400}
                   for i in range(700)]

        loop.load_history(candles)
        self.assertEqual(len(loop.candle_history), 500)
        # Should be the last 500
        self.assertEqual(loop.candle_history[0]["close"], 100.5 + 200)

    def test_stop_sets_status(self):
        """stop() should set status to STOPPED."""
        loop = LiveTradingLoop(symbol="SPY", strategy_name="ema_trend_v1")
        loop.stop()
        self.assertEqual(loop.state.status, LoopStatus.STOPPED)

    def test_evaluate_and_execute_with_history(self):
        """_evaluate_and_execute should work with sufficient history."""
        # Disable staleness check — test candles use arbitrary timestamps
        config = AnomalyConfig(staleness_hours=999999.0)
        loop = LiveTradingLoop(
            symbol="SPY", strategy_name="ema_trend_v1", anomaly_config=config,
        )

        # Build sufficient history (need 50+ candles for EMA)
        # Use recent-ish timestamps so staleness check doesn't block
        base_ts = int(datetime.now(timezone.utc).timestamp()) - 200 * 86400
        candles = []
        for i in range(100):
            price = 100.0 + i * 0.5
            candles.append({
                "open": price - 0.1,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 1000000,
                "time": base_ts + i * 86400,
                "open_time": base_ts + i * 86400 - 23400,
                "close_time": base_ts + i * 86400,
            })
        loop.load_history(candles)

        # Evaluate on a new candle
        new_candle = {
            "open": 150.0,
            "high": 151.0,
            "low": 149.0,
            "close": 150.5,
            "volume": 1200000,
            "time": base_ts + 100 * 86400,
            "open_time": base_ts + 100 * 86400 - 23400,
            "close_time": base_ts + 100 * 86400,
        }
        loop.candle_history.append(new_candle)

        now = datetime.now(timezone.utc)
        asyncio.get_event_loop().run_until_complete(
            loop._evaluate_and_execute(new_candle, now)
        )

        self.assertEqual(loop.state.evaluations_count, 1)
        self.assertIsNotNone(loop.state.last_evaluation_time)

        # Verify decision was logged
        db = SessionLocal()
        count = db.query(DecisionLog).count()
        db.close()
        self.assertGreaterEqual(count, 1)

    def test_anomaly_blocks_trading(self):
        """Critical anomaly should block trading and log the decision."""
        loop = LiveTradingLoop(symbol="SPY", strategy_name="ema_trend_v1")

        # Build history
        candles = []
        for i in range(100):
            candles.append({
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                "volume": 1000000,
                "time": 1000000 + i * 86400,
                "open_time": 1000000 + i * 86400 - 23400,
                "close_time": 1000000 + i * 86400,
            })
        loop.load_history(candles)

        # Create candle with invalid OHLC (critical anomaly)
        bad_candle = {
            "open": 100.0, "high": 95.0, "low": 105.0, "close": 100.0,
            "volume": 1000000,
            "time": 1000000 + 100 * 86400,
            "open_time": 1000000 + 100 * 86400 - 23400,
            "close_time": 1000000 + 100 * 86400,
        }
        loop.candle_history.append(bad_candle)

        now = datetime.now(timezone.utc)
        asyncio.get_event_loop().run_until_complete(
            loop._evaluate_and_execute(bad_candle, now)
        )

        # Evaluation count should NOT increase (blocked)
        self.assertEqual(loop.state.evaluations_count, 0)
        self.assertEqual(loop.state.anomalies_detected, 1)

        # But a decision should still be logged with BLOCKED action
        db = SessionLocal()
        entry = db.query(DecisionLog).first()
        db.close()
        self.assertIsNotNone(entry)
        self.assertEqual(entry.broker_action, "BLOCKED")
        self.assertEqual(entry.signal, "HOLD")


if __name__ == "__main__":
    unittest.main()
