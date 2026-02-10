"""
Unit tests for execution timing.
Verifies that signals and stop-losses execute on the NEXT candle open, not the same candle.

These tests will FAIL if same-candle execution occurs, ensuring the pending order system
works correctly: signals generated on candle close execute at the next candle open.

Test 1: BUY signal at candle close fills at next candle open
Test 2: Stop-loss detected at candle close exits at next candle open
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_broker import PaperBroker
from replay_engine import ReplayEngine
from database import Trade, EquityCurve, Signal
from utils import utc_datetime_to_unix


class TestExecutionTiming(unittest.TestCase):
    """Test execution timing: signals and stop-losses execute on next candle open."""
    
    def setUp(self):
        """Set up test fixtures.

        Creates history candles with a decline-then-rise pattern so that
        EMA(20) starts below EMA(50) and then crosses above it, triggering
        a valid BUY signal from ema_trend_v1.

        Price pattern (100 candle series, indices 0-99):
          Candles 0-49:  Declining prices (120 -> 95.5)  — EMA(20) drops below EMA(50)
          Candles 50-99: Rising prices   (95.5 -> 120.0) — EMA(20) crosses above EMA(50) at index 78

        History candles use indices 0-77 (78 candles, where EMA20 < EMA50).
        candle1 (signal) = index 78 in the series — this is the crossover candle.
        candle2 (execution) = index 79.
        candle3 (stop detection) = index 80.
        candle4 (stop execution) = index 81.
        """
        self.initial_equity = 100000.0
        self.symbol = "TEST"

        base_date = datetime(2022, 12, 1, 13, 30, 0, tzinfo=timezone.utc)
        self.history_candles = []

        # Build the full 100-candle price series
        all_closes = []
        # Phase 1: Declining (indices 0-49)
        for i in range(50):
            all_closes.append(120.0 - i * 0.5)  # 120.0 -> 95.5
        # Phase 2: Rising (indices 50-99)
        for i in range(50):
            all_closes.append(95.5 + i * 0.5)   # 95.5 -> 120.0

        # History candles: indices 0-77 (where EMA20 < EMA50 at index 77)
        for i in range(78):
            candle_date = base_date + timedelta(days=i)
            open_time = utc_datetime_to_unix(candle_date)
            close_time = utc_datetime_to_unix(candle_date.replace(hour=20, minute=0))
            c = all_closes[i]

            self.history_candles.append({
                "time": close_time,
                "open_time": open_time,
                "close_time": close_time,
                "open": c + (0.2 if i < 50 else -0.2),
                "high": c + 1.0,
                "low": c - 1.0,
                "close": c,
                "volume": 1000000
            })

        # Candle 1 (signal): index 78 in the series — EMA(20) crosses above EMA(50)
        # close=109.5, EMA20=105.30 > EMA50=105.09, prev EMA20=104.86 < prev EMA50=104.91
        signal_date = base_date + timedelta(days=78)
        self.candle1 = {
            "time": utc_datetime_to_unix(signal_date.replace(hour=20, minute=0)),
            "open_time": utc_datetime_to_unix(signal_date),
            "close_time": utc_datetime_to_unix(signal_date.replace(hour=20, minute=0)),
            "open": 109.0,
            "high": 110.5,
            "low": 108.5,
            "close": all_closes[78],  # 109.5
            "volume": 1000000
        }

        # Candle 2 (execution): index 79 — pending BUY fills at this open
        exec_date = base_date + timedelta(days=79)
        self.candle2 = {
            "time": utc_datetime_to_unix(exec_date.replace(hour=20, minute=0)),
            "open_time": utc_datetime_to_unix(exec_date),
            "close_time": utc_datetime_to_unix(exec_date.replace(hour=20, minute=0)),
            "open": 110.0,
            "high": 113.0,
            "low": 109.5,
            "close": 112.0,
            "volume": 1000000
        }

        # Candle 3 (stop-loss detection): index 80
        # Entry ~110.0 (candle2 open + slippage), ATR ~2.0, stop = entry - 2*ATR ~106.0
        # Close at 95.0 is well below stop → pending EXIT created
        stop_date = base_date + timedelta(days=80)
        self.candle3 = {
            "time": utc_datetime_to_unix(stop_date.replace(hour=20, minute=0)),
            "open_time": utc_datetime_to_unix(stop_date),
            "close_time": utc_datetime_to_unix(stop_date.replace(hour=20, minute=0)),
            "open": 112.0,
            "high": 112.5,
            "low": 94.0,
            "close": 95.0,
            "volume": 1000000
        }

        # Candle 4 (stop-loss execution): index 81 — pending EXIT fills at this open
        exit_date = base_date + timedelta(days=81)
        self.candle4 = {
            "time": utc_datetime_to_unix(exit_date.replace(hour=20, minute=0)),
            "open_time": utc_datetime_to_unix(exit_date),
            "close_time": utc_datetime_to_unix(exit_date.replace(hour=20, minute=0)),
            "open": 96.0,
            "high": 99.0,
            "low": 95.0,
            "close": 98.0,
            "volume": 1000000
        }
    
    def test_signal_at_close_fills_at_next_open(self):
        """
        Test that a BUY signal generated on candle close executes at the NEXT candle open.
        
        This test will FAIL if execution happens on the same candle.
        """
        # Create mock database session
        db_session = Mock()
        db_session.add = Mock()
        db_session.commit = Mock()
        db_session.query = Mock(return_value=Mock(filter=Mock(return_value=Mock(first=Mock(return_value=None)))))
        
        # Create replay engine
        replay_engine = ReplayEngine(initial_equity=self.initial_equity)
        
        # Start replay with history candles + signal candles
        all_candles = self.history_candles + [self.candle1, self.candle2]
        replay_engine.start_replay(
            symbol=self.symbol,
            candles=all_candles,
            replay_id="test_signal_timing"
        )
        
        # Process history candles first (to build indicator history)
        for i, hist_candle in enumerate(self.history_candles):
            replay_engine.current_candle_index = i + 1
            replay_engine.process_candle(db_session, hist_candle)
        
        # Process candle 1 (signal candle) - index after history
        signal_candle_idx = len(self.history_candles)
        replay_engine.current_candle_index = signal_candle_idx + 1
        signal_result = replay_engine.process_candle(db_session, self.candle1)
        
        # Verify signal was generated
        self.assertIsNotNone(signal_result)
        self.assertEqual(signal_result["signal"]["signal"], "BUY")
        
        # Verify NO trade was executed on candle 1 (same candle)
        # Check that broker has pending order but no position
        self.assertEqual(len(replay_engine.broker.pending_orders), 1, 
                        "FAIL: Pending order should exist but not executed yet")
        self.assertEqual(len(replay_engine.broker.positions), 0,
                        "FAIL: Position should NOT exist on same candle as signal")
        
        # Count how many Trade objects were added (should be 0, only Signal objects)
        trade_adds = [call[0][0] for call in db_session.add.call_args_list if isinstance(call[0][0], Trade)]
        self.assertEqual(len(trade_adds), 0,
                        f"FAIL: {len(trade_adds)} Trade records created on signal candle. "
                        f"Should be 0 - trades execute on NEXT candle!")
        
        # Process candle 2 (execution candle) - index after signal candle
        exec_candle_idx = len(self.history_candles) + 1
        replay_engine.current_candle_index = exec_candle_idx + 1
        execution_result = replay_engine.process_candle(db_session, self.candle2)
        
        # Verify trade was executed on candle 2 (next candle)
        self.assertEqual(len(replay_engine.broker.positions), 1,
                        "FAIL: Position should exist AFTER next candle open")
        self.assertEqual(len(replay_engine.broker.pending_orders), 0,
                        "Pending order should be cleared after execution")
        
        # Verify trade record was created with candle 2's open_time
        # Find the Trade object among all db_session.add calls (Signal and EquityCurve also added)
        db_session.add.assert_called()
        trade_records = [call[0][0] for call in db_session.add.call_args_list if isinstance(call[0][0], Trade)]
        self.assertEqual(len(trade_records), 1, "Exactly one Trade record should be created")
        trade_record = trade_records[0]
        self.assertEqual(trade_record.symbol, self.symbol)
        
        # CRITICAL: Verify entry_time matches candle 2's open_time (not candle 1's close_time)
        # This test WILL FAIL if execution happens on same candle
        entry_time_unix = utc_datetime_to_unix(trade_record.entry_time)
        self.assertEqual(entry_time_unix, self.candle2["open_time"],
                        f"FAIL: Entry time {entry_time_unix} should match candle 2 open_time {self.candle2['open_time']}, "
                        f"not candle 1 close_time {self.candle1['close_time']}. "
                        f"Same-candle execution detected!")
        
        # Explicit check: entry_time must NOT equal candle 1's close_time
        self.assertNotEqual(entry_time_unix, self.candle1["close_time"],
                           f"FAIL: Entry time {entry_time_unix} must NOT equal signal candle close_time {self.candle1['close_time']}. "
                           f"This indicates same-candle execution!")
        
        # Verify entry price is candle 2's open price (with slippage)
        expected_fill_price = self.candle2["open"] * (1 + 0.0002)  # With slippage
        self.assertAlmostEqual(trade_record.entry_price, expected_fill_price, places=1,
                              msg=f"FAIL: Entry price should be candle 2 open with slippage, not candle 1 close")
    
    def test_stop_loss_detected_on_close_exits_at_next_open(self):
        """
        Test that a stop-loss detected on candle close exits at the NEXT candle open.

        This test will FAIL if exit happens on the same candle.
        """
        # Create mock database session
        db_session = Mock()
        created_trade = None

        # Track trade records that get created
        def mock_add(obj):
            nonlocal created_trade
            if isinstance(obj, Trade):
                created_trade = obj

        db_session.add = Mock(side_effect=mock_add)
        db_session.commit = Mock()

        # Query mock: returns created_trade for Trade queries (once it exists),
        # returns None for Signal queries (no duplicates)
        def query_side_effect(model):
            if model == Trade:
                filter_mock = Mock()
                def first_side_effect():
                    return created_trade  # Returns None before trade created, trade object after
                filter_mock.first = Mock(side_effect=first_side_effect)
                query_mock = Mock(filter=Mock(return_value=filter_mock))
                return query_mock
            # For Signal queries
            return Mock(filter=Mock(return_value=Mock(first=Mock(return_value=None))))

        db_session.query = Mock(side_effect=query_side_effect)
        
        # Create replay engine
        replay_engine = ReplayEngine(initial_equity=self.initial_equity)
        
        # Start replay with history + signal candles
        all_candles = self.history_candles + [self.candle1, self.candle2, self.candle3, self.candle4]
        replay_engine.start_replay(
            symbol=self.symbol,
            candles=all_candles,
            replay_id="test_stop_timing"
        )
        
        # Process history candles first
        for i, hist_candle in enumerate(self.history_candles):
            replay_engine.current_candle_index = i + 1
            replay_engine.process_candle(db_session, hist_candle)
        
        # Process candle 1 (entry signal)
        signal_idx = len(self.history_candles)
        replay_engine.current_candle_index = signal_idx + 1
        replay_engine.process_candle(db_session, self.candle1)
        
        # Process candle 2 (entry execution - creates position)
        exec_idx = len(self.history_candles) + 1
        replay_engine.current_candle_index = exec_idx + 1
        replay_engine.process_candle(db_session, self.candle2)
        
        # Verify position exists
        self.assertEqual(len(replay_engine.broker.positions), 1,
                        "Position should exist after candle 2")
        
        # Get the created trade record
        self.assertIsNotNone(created_trade, "Trade record should be created on candle 2")
        self.assertEqual(created_trade.symbol, self.symbol)
        self.assertIsNone(created_trade.exit_time, "Trade should be open (no exit_time)")
        
        # Get position stop price
        position = replay_engine.broker.positions[self.symbol]
        stop_price = position.stop_price
        
        # Verify stop-loss would be triggered by candle 3's close
        self.assertLess(self.candle3["close"], stop_price,
                       f"Stop-loss should be triggered: close {self.candle3['close']} < stop {stop_price}")
        
        # Process candle 3 (stop-loss detection candle)
        # This should detect stop-loss breach but NOT execute exit
        stop_idx = len(self.history_candles) + 2
        replay_engine.current_candle_index = stop_idx + 1
        detection_result = replay_engine.process_candle(db_session, self.candle3)
        
        # Verify stop-loss was detected (pending order created)
        # Check that broker has pending EXIT order(s) but position still exists
        # Note: Both stop-loss AND kill-switch may fire, creating multiple pending EXITs
        pending_exits = [o for o in replay_engine.broker.pending_orders if o.order_type.value == "EXIT"]
        self.assertGreaterEqual(len(pending_exits), 1,
                        "FAIL: At least one pending EXIT order should exist but not executed yet")
        self.assertEqual(len(replay_engine.broker.positions), 1,
                        "FAIL: Position should still exist on same candle as stop-loss detection")
        
        # Verify position was NOT closed on candle 3
        self.assertIn(self.symbol, replay_engine.broker.positions,
                     "FAIL: Position should NOT be closed on same candle as stop-loss detection")
        
        # Process candle 4 (stop-loss execution candle)
        # This should execute the pending EXIT order from candle 3
        exit_idx = len(self.history_candles) + 3
        replay_engine.current_candle_index = exit_idx + 1
        execution_result = replay_engine.process_candle(db_session, self.candle4)
        
        # Verify position was closed on candle 4 (next candle)
        self.assertEqual(len(replay_engine.broker.positions), 0,
                        "FAIL: Position should be closed AFTER next candle open")
        self.assertEqual(len(replay_engine.broker.pending_orders), 0,
                        "Pending order should be cleared after execution")
        
        # Verify trade record was updated
        db_session.commit.assert_called()
        
        # CRITICAL: Verify exit_time matches candle 4's open_time (not candle 3's close_time)
        # This test WILL FAIL if exit happens on same candle
        self.assertIsNotNone(created_trade.exit_time, "Trade should have exit_time after stop-loss execution")
        exit_time_unix = utc_datetime_to_unix(created_trade.exit_time)
        
        self.assertEqual(exit_time_unix, self.candle4["open_time"],
                        f"FAIL: Exit time {exit_time_unix} should match candle 4 open_time {self.candle4['open_time']}, "
                        f"not candle 3 close_time {self.candle3['close_time']}. "
                        f"Same-candle execution detected!")
        
        # Explicit check: exit_time must NOT equal candle 3's close_time
        self.assertNotEqual(exit_time_unix, self.candle3["close_time"],
                           f"FAIL: Exit time {exit_time_unix} must NOT equal stop-loss detection candle close_time {self.candle3['close_time']}. "
                           f"This indicates same-candle execution!")
        
        # Verify position was closed on candle 4 (next candle), not candle 3 (same candle)
        self.assertNotIn(self.symbol, replay_engine.broker.positions,
                        "FAIL: Position should be closed AFTER next candle open, not on same candle as stop-loss detection")
        
        # Verify pending orders were cleared after execution
        self.assertEqual(len(replay_engine.broker.pending_orders), 0,
                        "Pending orders should be cleared after execution on next candle")
        
        # Verify exit price is candle 4's open price (with slippage)
        expected_exit_price = self.candle4["open"] * (1 - 0.0002)  # With slippage (sell receives less)
        self.assertAlmostEqual(created_trade.exit_price, expected_exit_price, places=1,
                              msg=f"FAIL: Exit price should be candle 4 open with slippage, not candle 3 close")


if __name__ == "__main__":
    unittest.main()
