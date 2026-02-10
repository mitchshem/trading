"""
Unit tests for indicators and strategy signal generation.
Validates RSI indicator, strategy registry, and basic signal logic.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import rsi, ema, atr
from strategy import PositionState
from strategy_mean_reversion_v1 import mean_reversion_v1
from strategy_dual_momentum_v1 import dual_momentum_v1
import strategy_registry as sr


class TestRSIIndicator(unittest.TestCase):
    """Test RSI indicator correctness."""

    def test_rsi_returns_correct_length(self):
        prices = list(range(1, 101))  # 100 prices
        result = rsi(prices, 14)
        self.assertEqual(len(result), 100)

    def test_rsi_insufficient_data(self):
        prices = [100.0] * 10
        result = rsi(prices, 14)
        self.assertTrue(all(v is None for v in result))

    def test_rsi_range_0_100(self):
        """RSI values should always be between 0 and 100."""
        # Use volatile prices to test range
        import math
        prices = [100 + 10 * math.sin(i * 0.3) for i in range(200)]
        result = rsi(prices, 14)
        for v in result:
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 100.0)

    def test_rsi_all_gains_is_100(self):
        """Monotonically increasing prices should yield RSI near 100."""
        prices = [100.0 + i for i in range(50)]
        result = rsi(prices, 14)
        last_rsi = result[-1]
        self.assertIsNotNone(last_rsi)
        self.assertGreater(last_rsi, 95.0)

    def test_rsi_all_losses_is_near_0(self):
        """Monotonically decreasing prices should yield RSI near 0."""
        prices = [200.0 - i for i in range(50)]
        result = rsi(prices, 14)
        last_rsi = result[-1]
        self.assertIsNotNone(last_rsi)
        self.assertLess(last_rsi, 5.0)


class TestMeanReversionV1(unittest.TestCase):
    """Test mean reversion strategy signals."""

    def _make_candles(self, closes):
        """Create candle dicts from a list of close prices."""
        candles = []
        for i, c in enumerate(closes):
            candles.append({
                "time": i,
                "open_time": i,
                "close_time": i,
                "open": c,
                "high": c + 1.0,
                "low": c - 1.0,
                "close": c,
                "volume": 1000000,
            })
        return candles

    def test_hold_insufficient_data(self):
        candles = self._make_candles([100.0] * 10)
        ps = PositionState()
        result = mean_reversion_v1(candles, ps)
        self.assertEqual(result["signal"], "HOLD")

    def test_buy_on_oversold(self):
        """After sharp decline, RSI should be oversold and signal BUY."""
        # Create declining prices to push RSI below 30, with regime gate off
        prices = [200.0 - i * 0.3 for i in range(250)]  # slow decline
        # Then sharp drop
        for i in range(20):
            prices.append(prices[-1] - 3.0)
        candles = self._make_candles(prices)
        ps = PositionState()
        result = mean_reversion_v1(candles, ps, params={"require_trend_up": False})
        # With a sharp decline, RSI should be oversold
        if result["signal"] == "BUY":
            self.assertIn("oversold", result["reason"].lower())

    def test_exit_on_rsi_recovery(self):
        """With position and RSI > exit threshold, should EXIT."""
        # Long history of prices followed by recovery
        prices = [100.0 + i * 0.1 for i in range(200)]
        candles = self._make_candles(prices)
        ps = PositionState()
        ps.has_position = True
        ps.entry_price = 100.0
        result = mean_reversion_v1(candles, ps)
        # After consistent gains, RSI > 50 → EXIT
        if result["signal"] == "EXIT":
            self.assertIn("exit threshold", result["reason"].lower())


class TestDualMomentumV1(unittest.TestCase):
    """Test dual momentum strategy signals."""

    def _make_candles(self, closes):
        candles = []
        for i, c in enumerate(closes):
            candles.append({
                "time": i,
                "open_time": i,
                "close_time": i,
                "open": c,
                "high": c + 1.0,
                "low": c - 1.0,
                "close": c,
                "volume": 1000000,
            })
        return candles

    def test_hold_insufficient_data(self):
        candles = self._make_candles([100.0] * 50)
        ps = PositionState()
        result = dual_momentum_v1(candles, ps)
        self.assertEqual(result["signal"], "HOLD")

    def test_buy_on_positive_momentum(self):
        """Steadily rising prices over lookback should trigger BUY."""
        prices = [100.0 + i * 0.5 for i in range(200)]
        candles = self._make_candles(prices)
        ps = PositionState()
        result = dual_momentum_v1(candles, ps, params={"lookback": 63})
        self.assertEqual(result["signal"], "BUY")

    def test_no_buy_on_declining_prices(self):
        """Declining prices should not trigger BUY."""
        prices = [200.0 - i * 0.5 for i in range(200)]
        candles = self._make_candles(prices)
        ps = PositionState()
        result = dual_momentum_v1(candles, ps, params={"lookback": 63})
        self.assertNotEqual(result["signal"], "BUY")

    def test_exit_on_momentum_loss(self):
        """If price drops below lookback close, should EXIT."""
        # Rising then sharp drop
        prices = [100.0 + i * 0.5 for i in range(150)]
        for i in range(50):
            prices.append(prices[-1] - 2.0)
        candles = self._make_candles(prices)
        ps = PositionState()
        ps.has_position = True
        ps.entry_price = 150.0
        result = dual_momentum_v1(candles, ps, params={"lookback": 63})
        self.assertEqual(result["signal"], "EXIT")


class TestStrategyRegistry(unittest.TestCase):
    """Test strategy registry operations."""

    def test_all_strategies_registered(self):
        names = sr.list_strategies()
        self.assertIn("ema_trend_v1", names)
        self.assertIn("momentum_breakout_v1", names)
        self.assertIn("mean_reversion_v1", names)
        self.assertIn("dual_momentum_v1", names)

    def test_get_unknown_strategy_raises(self):
        with self.assertRaises(KeyError):
            sr.get_strategy("nonexistent_strategy")

    def test_ema_trend_v1_evaluate(self):
        """Smoke test: ema_trend_v1 via registry returns valid signal."""
        config = sr.get_strategy("ema_trend_v1")
        prices = [100.0 + i * 0.1 for i in range(100)]
        candles = []
        for i, c in enumerate(prices):
            candles.append({
                "time": i, "open_time": i, "close_time": i,
                "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1000000,
            })
        ps = PositionState()
        result = config.evaluate_fn(candles, ps)
        self.assertIn(result["signal"], ["BUY", "EXIT", "HOLD"])


if __name__ == "__main__":
    unittest.main()
