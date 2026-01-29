"""
Unit tests for commission model in PaperBroker.
Verifies commissions are calculated, deducted, and included in P&L correctly.
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import Mock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_broker import PaperBroker
from utils import utc_datetime_to_unix


class TestCommissionModel(unittest.TestCase):
    """Test commission model implementation."""
    
    def test_commission_calculation(self):
        """Test calculate_commission helper."""
        broker = PaperBroker(
            initial_equity=100000.0,
            commission_per_share=0.005,
            commission_per_trade=1.0
        )
        
        # Test commission calculation
        commission = broker.calculate_commission(100)
        expected = (100 * 0.005) + 1.0  # $0.50 + $1.00 = $1.50
        self.assertEqual(commission, expected)
        
        # Test with zero shares
        commission_zero = broker.calculate_commission(0)
        expected_zero = (0 * 0.005) + 1.0  # $0.00 + $1.00 = $1.00
        self.assertEqual(commission_zero, expected_zero)
    
    def test_buy_commission_deduction(self):
        """Test that commission is deducted from cash on BUY."""
        broker = PaperBroker(
            initial_equity=100000.0,
            commission_per_share=0.005,
            commission_per_trade=0.0
        )
        
        initial_cash = broker.cash
        
        # Execute BUY (position size calculated automatically)
        fill_price = 50.0
        stop_distance = 1.0
        timestamp = utc_datetime_to_unix(datetime.now(timezone.utc))
        
        trade = broker._execute_buy_internal("TEST", fill_price, stop_distance, timestamp)
        
        self.assertIsNotNone(trade)
        shares = trade["shares"]
        self.assertGreater(shares, 0, "Should have shares")
        
        # Verify commission was deducted
        expected_commission = (shares * 0.005) + 0.0
        expected_cost = (shares * fill_price) + expected_commission
        expected_cash = initial_cash - expected_cost
        
        self.assertAlmostEqual(broker.cash, expected_cash, places=2)
        self.assertEqual(trade["entry_commission"], expected_commission)
        self.assertEqual(trade["total_commission"], expected_commission)
        self.assertIsNone(trade["net_pnl"])  # No P&L until exit
    
    def test_exit_commission_and_net_pnl(self):
        """Test that exit commission is deducted and net_pnl includes both commissions."""
        broker = PaperBroker(
            initial_equity=100000.0,
            commission_per_share=0.005,
            commission_per_trade=1.0
        )
        
        # First, create a position (position size calculated automatically)
        entry_price = 50.0
        stop_distance = 1.0
        entry_timestamp = utc_datetime_to_unix(datetime.now(timezone.utc))
        
        buy_trade = broker._execute_buy_internal("TEST", entry_price, stop_distance, entry_timestamp)
        self.assertIsNotNone(buy_trade)
        
        shares = buy_trade["shares"]
        entry_commission = buy_trade["entry_commission"]
        initial_cash_after_buy = broker.cash
        
        # Now exit the position
        exit_price = 52.0  # $2 profit per share
        exit_timestamp = utc_datetime_to_unix(datetime.now(timezone.utc))
        
        exit_trade = broker._execute_exit_internal("TEST", exit_price, exit_timestamp, "TEST_EXIT")
        
        self.assertIsNotNone(exit_trade)
        
        # Verify exit commission
        exit_commission = exit_trade["exit_commission"]
        expected_exit_commission = (shares * 0.005) + 1.0
        self.assertEqual(exit_commission, expected_exit_commission)
        
        # Verify total commission
        total_commission = exit_trade["total_commission"]
        expected_total = entry_commission + exit_commission
        self.assertEqual(total_commission, expected_total)
        
        # Verify net P&L
        gross_pnl = (exit_price - entry_price) * shares
        net_pnl = exit_trade["net_pnl"]
        expected_net_pnl = gross_pnl - total_commission
        self.assertEqual(net_pnl, expected_net_pnl)
        self.assertEqual(exit_trade["pnl"], net_pnl)  # pnl should equal net_pnl
        
        # Verify cash was updated correctly
        # Cash should increase by (proceeds - exit_commission)
        proceeds = shares * exit_price
        net_proceeds = proceeds - exit_commission
        expected_cash_after_exit = initial_cash_after_buy + net_proceeds
        self.assertAlmostEqual(broker.cash, expected_cash_after_exit, places=2)
        
        # Verify daily_realized_pnl was updated with net_pnl
        self.assertEqual(broker.daily_realized_pnl, net_pnl)
    
    def test_equity_includes_commissions(self):
        """Test that equity calculation includes commissions via cash reduction."""
        broker = PaperBroker(
            initial_equity=100000.0,
            commission_per_share=0.005,
            commission_per_trade=0.0
        )
        
        initial_equity = broker.equity
        
        # Execute BUY with commission (position size calculated automatically)
        fill_price = 50.0
        stop_distance = 1.0
        timestamp = utc_datetime_to_unix(datetime.now(timezone.utc))
        
        trade = broker._execute_buy_internal("TEST", fill_price, stop_distance, timestamp)
        self.assertIsNotNone(trade)
        shares = trade["shares"]
        
        # Update equity
        broker.update_equity({"TEST": fill_price})
        
        # Equity should reflect commission deduction (via cash reduction)
        # Commission reduces cash, which reduces equity
        commission = (shares * 0.005) + 0.0
        cost = shares * fill_price
        total_cost = cost + commission
        
        expected_cash = initial_equity - total_cost
        expected_equity = expected_cash + 0.0  # No unrealized P&L at entry price
        
        self.assertAlmostEqual(broker.cash, expected_cash, places=2)
        self.assertAlmostEqual(broker.equity, expected_equity, places=2)
    
    def test_daily_pnl_includes_commissions(self):
        """Test that daily P&L includes commissions via net P&L."""
        broker = PaperBroker(
            initial_equity=100000.0,
            commission_per_share=0.005,
            commission_per_trade=1.0
        )
        
        # Execute BUY and EXIT
        entry_price = 50.0
        exit_price = 52.0
        shares = 100
        stop_distance = 1.0
        
        entry_timestamp = utc_datetime_to_unix(datetime.now(timezone.utc))
        broker._execute_buy_internal("TEST", entry_price, stop_distance, entry_timestamp)
        
        exit_timestamp = utc_datetime_to_unix(datetime.now(timezone.utc))
        exit_trade = broker._execute_exit_internal("TEST", exit_price, exit_timestamp, "TEST")
        
        # Update equity to calculate daily_pnl
        broker.update_equity({})
        
        # daily_realized_pnl should equal net_pnl (includes commissions)
        self.assertEqual(broker.daily_realized_pnl, exit_trade["net_pnl"])
        
        # daily_pnl should include realized P&L (with commissions)
        self.assertEqual(broker.daily_pnl, broker.daily_realized_pnl)


if __name__ == "__main__":
    unittest.main()
