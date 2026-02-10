"""
Unit tests for partial exit functionality in PaperBroker.
Verifies that partial exits correctly split positions, update cash, and record trades.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_broker import PaperBroker, OrderType


class TestPartialExit(unittest.TestCase):
    """Test partial exit functionality."""

    def setUp(self):
        self.broker = PaperBroker(
            initial_equity=100000.0,
            commission_per_share=0.005,
            commission_per_trade=1.0,
            slippage=0.0
        )

    def _open_position(self, symbol="TEST", price=100.0, shares=100, stop_distance=4.0):
        """Helper: directly open a position for testing."""
        from paper_broker import Position
        commission = self.broker.calculate_commission(shares)
        cost = shares * price + commission
        self.broker.cash -= cost
        position = Position(
            symbol=symbol,
            entry_time=1000,
            entry_price=price,
            shares=shares,
            stop_price=price - stop_distance,
            entry_commission=commission
        )
        self.broker.positions[symbol] = position
        return position

    def test_partial_exit_splits_position(self):
        """Partial exit of 50% should sell half the shares and keep the rest."""
        self._open_position(shares=100, price=100.0)

        # Create and process partial exit
        self.broker.execute_partial_exit(
            symbol="TEST", signal_price=110.0, exit_pct=0.5,
            timestamp=2000, reason="Target 1 hit"
        )

        trades = self.broker.process_pending_orders(
            current_open_price=110.0, timestamp=2000
        )

        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade["action"], "PARTIAL_EXIT")
        self.assertEqual(trade["shares"], 50)
        self.assertEqual(trade["remaining_shares"], 50)

        # Position should still exist with 50 shares
        self.assertIn("TEST", self.broker.positions)
        self.assertEqual(self.broker.positions["TEST"].shares, 50)

    def test_partial_exit_updates_cash(self):
        """Cash should increase by proceeds minus exit commission."""
        self._open_position(shares=100, price=100.0)
        cash_before = self.broker.cash

        self.broker.execute_partial_exit(
            symbol="TEST", signal_price=110.0, exit_pct=0.5,
            timestamp=2000, reason="Target 1"
        )

        trades = self.broker.process_pending_orders(
            current_open_price=110.0, timestamp=2000
        )

        trade = trades[0]
        expected_proceeds = 50 * 110.0
        expected_commission = (50 * 0.005) + 1.0  # per-share + per-trade
        expected_cash = cash_before + expected_proceeds - expected_commission

        self.assertAlmostEqual(self.broker.cash, expected_cash, places=2)

    def test_partial_exit_pnl(self):
        """P&L should reflect profit on exited shares minus pro-rated commissions."""
        self._open_position(shares=100, price=100.0)

        self.broker.execute_partial_exit(
            symbol="TEST", signal_price=110.0, exit_pct=0.5,
            timestamp=2000, reason="Target 1"
        )

        trades = self.broker.process_pending_orders(
            current_open_price=110.0, timestamp=2000
        )

        trade = trades[0]
        # Gross P&L: (110 - 100) * 50 = 500
        # Entry commission pro-rated: (100 * 0.005 + 1.0) * 0.5 = 0.75
        # Exit commission: 50 * 0.005 + 1.0 = 1.25
        # Net P&L: 500 - 0.75 - 1.25 = 498.0
        self.assertGreater(trade["net_pnl"], 0)
        self.assertAlmostEqual(trade["net_pnl"], 498.0, places=2)

    def test_partial_exit_preserves_stop_price(self):
        """Stop price should remain unchanged after partial exit."""
        pos = self._open_position(shares=100, price=100.0, stop_distance=4.0)
        original_stop = pos.stop_price

        self.broker.execute_partial_exit(
            symbol="TEST", signal_price=110.0, exit_pct=0.5,
            timestamp=2000, reason="Target 1"
        )
        self.broker.process_pending_orders(current_open_price=110.0, timestamp=2000)

        self.assertEqual(self.broker.positions["TEST"].stop_price, original_stop)

    def test_full_exit_if_pct_too_high(self):
        """exit_pct >= 1.0 should do a full exit (no remaining position)."""
        self._open_position(shares=100, price=100.0)

        self.broker.execute_partial_exit(
            symbol="TEST", signal_price=110.0, exit_pct=1.0,
            timestamp=2000, reason="Full exit"
        )

        trades = self.broker.process_pending_orders(
            current_open_price=110.0, timestamp=2000
        )

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["action"], "EXIT")  # Falls back to full exit
        self.assertNotIn("TEST", self.broker.positions)

    def test_partial_exit_no_position(self):
        """Partial exit with no position should return None."""
        result = self.broker.execute_partial_exit(
            symbol="TEST", signal_price=110.0, exit_pct=0.5,
            timestamp=2000, reason="No position"
        )
        self.assertIsNone(result)

    def test_partial_exit_reduces_entry_commission(self):
        """Remaining position entry_commission should be reduced proportionally."""
        pos = self._open_position(shares=100, price=100.0)
        original_commission = pos.entry_commission

        self.broker.execute_partial_exit(
            symbol="TEST", signal_price=110.0, exit_pct=0.5,
            timestamp=2000, reason="Target 1"
        )
        self.broker.process_pending_orders(current_open_price=110.0, timestamp=2000)

        remaining_commission = self.broker.positions["TEST"].entry_commission
        self.assertAlmostEqual(remaining_commission, original_commission * 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
