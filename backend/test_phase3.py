"""
Tests for Phase 3: Risk Management Hardening.

Tests:
- Weekly/monthly loss limits
- Consecutive losing days pause
- Drawdown from high-water mark
- Session boundary (end-of-day close)
- Portfolio exposure limits
- Volatility-adjusted position sizing
- Comprehensive risk control check
"""

import unittest
import math
from datetime import date, timedelta

from paper_broker import PaperBroker, Position


class TestEnhancedRiskLimits(unittest.TestCase):
    """Test enhanced kill-switch limits (weekly, monthly, HWM, streak)."""

    def setUp(self):
        self.broker = PaperBroker(initial_equity=100000.0)

    def test_weekly_loss_limit_triggers(self):
        """Weekly loss > 5% of HWM should be detected."""
        # Simulate losing $5001 in a week (> 5% of $100k HWM)
        self.broker.weekly_pnl = -5001.0
        self.assertTrue(self.broker.check_weekly_loss_limit())

    def test_weekly_loss_limit_within(self):
        """Weekly loss within 5% should NOT trigger."""
        self.broker.weekly_pnl = -4000.0
        self.assertFalse(self.broker.check_weekly_loss_limit())

    def test_monthly_loss_limit_triggers(self):
        """Monthly loss > 10% of HWM should be detected."""
        self.broker.monthly_pnl = -10001.0
        self.assertTrue(self.broker.check_monthly_loss_limit())

    def test_monthly_loss_limit_within(self):
        """Monthly loss within 10% should NOT trigger."""
        self.broker.monthly_pnl = -8000.0
        self.assertFalse(self.broker.check_monthly_loss_limit())

    def test_hwm_drawdown_triggers(self):
        """Drawdown > 15% from HWM should be detected."""
        self.broker.high_water_mark = 100000.0
        self.broker.equity = 84000.0  # 16% drawdown
        self.assertTrue(self.broker.check_drawdown_from_hwm())

    def test_hwm_drawdown_within(self):
        """Drawdown < 15% from HWM should NOT trigger."""
        self.broker.high_water_mark = 100000.0
        self.broker.equity = 90000.0  # 10% drawdown
        self.assertFalse(self.broker.check_drawdown_from_hwm())

    def test_hwm_updates_on_new_high(self):
        """HWM should update when equity exceeds it."""
        self.broker.equity = 110000.0
        self.broker.update_high_water_mark()
        self.assertEqual(self.broker.high_water_mark, 110000.0)

    def test_hwm_no_update_on_decline(self):
        """HWM should NOT decrease when equity drops."""
        self.broker.equity = 90000.0
        self.broker.update_high_water_mark()
        self.assertEqual(self.broker.high_water_mark, 100000.0)  # Initial equity


class TestConsecutiveLosingDays(unittest.TestCase):
    """Test consecutive losing days tracking and pause."""

    def setUp(self):
        self.broker = PaperBroker(initial_equity=100000.0)

    def test_consecutive_losing_days_increments(self):
        """Each losing day should increment the counter."""
        for i in range(3):
            d = date(2024, 1, 1 + i)
            self.broker.daily_pnl = -100.0
            self.broker.track_daily_pnl_for_streak(d)

        self.assertEqual(self.broker.consecutive_losing_days, 3)

    def test_winning_day_resets_streak(self):
        """A winning day should reset the counter to 0."""
        # 3 losing days
        for i in range(3):
            d = date(2024, 1, 1 + i)
            self.broker.daily_pnl = -100.0
            self.broker.track_daily_pnl_for_streak(d)

        # 1 winning day
        self.broker.daily_pnl = 100.0
        self.broker.track_daily_pnl_for_streak(date(2024, 1, 4))

        self.assertEqual(self.broker.consecutive_losing_days, 0)

    def test_five_losing_days_triggers_limit(self):
        """5 consecutive losing days should trigger the check."""
        for i in range(5):
            d = date(2024, 1, 1 + i)
            self.broker.daily_pnl = -100.0
            self.broker.track_daily_pnl_for_streak(d)

        self.assertTrue(self.broker.check_consecutive_losing_days())

    def test_same_day_not_double_counted(self):
        """Tracking the same day twice should not increment."""
        d = date(2024, 1, 1)
        self.broker.daily_pnl = -100.0
        self.broker.track_daily_pnl_for_streak(d)
        self.broker.track_daily_pnl_for_streak(d)  # Same day again

        self.assertEqual(self.broker.consecutive_losing_days, 1)

    def test_pause_blocks_trading(self):
        """is_paused should return True when before pause_until_date."""
        self.broker.pause_until_date = date(2024, 1, 5)
        self.assertTrue(self.broker.is_paused(date(2024, 1, 4)))

    def test_pause_expires(self):
        """is_paused should return False and clear pause when date reached."""
        self.broker.pause_until_date = date(2024, 1, 5)
        self.broker.consecutive_losing_days = 5

        self.assertFalse(self.broker.is_paused(date(2024, 1, 5)))
        self.assertIsNone(self.broker.pause_until_date)
        self.assertEqual(self.broker.consecutive_losing_days, 0)  # Reset


class TestWeeklyMonthlyResets(unittest.TestCase):
    """Test weekly and monthly P&L resets."""

    def setUp(self):
        self.broker = PaperBroker(initial_equity=100000.0)

    def test_weekly_pnl_resets_on_new_week(self):
        """Weekly P&L should reset when ISO week changes."""
        self.broker.weekly_pnl = -3000.0
        self.broker.last_reset_week = 1  # Week 1

        # Simulate new week
        new_date = date(2024, 1, 8)  # Week 2
        self.broker.reset_weekly_stats_if_needed(new_date)

        self.assertEqual(self.broker.weekly_pnl, 0.0)

    def test_weekly_pnl_persists_same_week(self):
        """Weekly P&L should NOT reset within the same week."""
        self.broker.weekly_pnl = -3000.0
        d = date(2024, 1, 3)  # Wednesday of week 1
        self.broker.last_reset_week = d.isocalendar()[1]

        self.broker.reset_weekly_stats_if_needed(date(2024, 1, 5))  # Friday same week
        self.assertEqual(self.broker.weekly_pnl, -3000.0)

    def test_monthly_pnl_resets_on_new_month(self):
        """Monthly P&L should reset when month changes."""
        self.broker.monthly_pnl = -7000.0
        self.broker.last_reset_month = 1  # January

        self.broker.reset_monthly_stats_if_needed(date(2024, 2, 1))  # February
        self.assertEqual(self.broker.monthly_pnl, 0.0)


class TestPortfolioExposure(unittest.TestCase):
    """Test portfolio-level exposure controls."""

    def setUp(self):
        self.broker = PaperBroker(initial_equity=100000.0)

    def test_exposure_with_no_positions(self):
        """Exposure should be 0 with no positions."""
        exposure = self.broker.check_portfolio_exposure({"SPY": 500.0})
        self.assertEqual(exposure, 0.0)

    def test_exposure_with_position(self):
        """Exposure should reflect position value / equity."""
        self.broker.positions["SPY"] = Position(
            symbol="SPY", entry_time=0, entry_price=500.0,
            shares=100, stop_price=490.0,
        )
        # Position value = 100 × 500 = $50,000 / $100,000 equity = 50%
        exposure = self.broker.check_portfolio_exposure({"SPY": 500.0})
        self.assertAlmostEqual(exposure, 0.5, places=2)

    def test_can_open_position_within_limits(self):
        """Should allow position within 80% exposure limit."""
        result = self.broker.can_open_position({"SPY": 500.0}, proposed_value=50000.0)
        self.assertTrue(result["allowed"])

    def test_can_open_position_exceeds_exposure(self):
        """Should block position that would exceed 80% exposure."""
        # Already 70% exposed
        self.broker.positions["SPY"] = Position(
            symbol="SPY", entry_time=0, entry_price=500.0,
            shares=140, stop_price=490.0,
        )
        # Trying to add another $20k would bring to 90%
        result = self.broker.can_open_position({"SPY": 500.0}, proposed_value=20000.0)
        self.assertFalse(result["allowed"])
        self.assertIn("exposure", result["reason"])

    def test_can_open_position_blocked_by_kill_switch(self):
        """Kill-switch should block new positions."""
        self.broker.trade_blocked = True
        result = self.broker.can_open_position({"SPY": 500.0}, proposed_value=10000.0)
        self.assertFalse(result["allowed"])
        self.assertIn("kill-switch", result["reason"])

    def test_can_open_position_blocked_by_pause(self):
        """Pause should block new positions."""
        self.broker.pause_until_date = date(2024, 12, 31)
        result = self.broker.can_open_position(
            {"SPY": 500.0}, proposed_value=10000.0,
            current_date=date(2024, 12, 30),
        )
        self.assertFalse(result["allowed"])
        self.assertIn("paused", result["reason"])


class TestVolatilityAdjustedSizing(unittest.TestCase):
    """Test volatility-adjusted position sizing."""

    def setUp(self):
        self.broker = PaperBroker(initial_equity=100000.0)

    def test_normal_volatility_no_reduction(self):
        """Same ATR as historical → no reduction."""
        base = self.broker.calculate_position_size(100.0, 2.0, 100000.0)
        adjusted = self.broker.calculate_vol_adjusted_position_size(
            100.0, 2.0, 100000.0,
            current_atr=2.0, historical_atr=2.0,
        )
        self.assertEqual(adjusted, base)

    def test_high_volatility_reduces_size(self):
        """2× ATR → half the position size."""
        base = self.broker.calculate_position_size(100.0, 2.0, 100000.0)
        adjusted = self.broker.calculate_vol_adjusted_position_size(
            100.0, 2.0, 100000.0,
            current_atr=4.0, historical_atr=2.0,
        )
        self.assertEqual(adjusted, math.floor(base * 0.5))

    def test_low_volatility_capped_at_base(self):
        """0.5× ATR → capped at base (never increases beyond base)."""
        base = self.broker.calculate_position_size(100.0, 2.0, 100000.0)
        adjusted = self.broker.calculate_vol_adjusted_position_size(
            100.0, 2.0, 100000.0,
            current_atr=1.0, historical_atr=2.0,
        )
        self.assertEqual(adjusted, base)  # Capped at 1.0

    def test_no_atr_falls_back_to_base(self):
        """Missing ATR data → uses base position size."""
        base = self.broker.calculate_position_size(100.0, 2.0, 100000.0)
        adjusted = self.broker.calculate_vol_adjusted_position_size(
            100.0, 2.0, 100000.0,
            current_atr=None, historical_atr=None,
        )
        self.assertEqual(adjusted, base)

    def test_disabled_vol_adjustment(self):
        """Disabled vol adjustment → uses base size even with high ATR."""
        self.broker.vol_adjustment_enabled = False
        base = self.broker.calculate_position_size(100.0, 2.0, 100000.0)
        adjusted = self.broker.calculate_vol_adjusted_position_size(
            100.0, 2.0, 100000.0,
            current_atr=4.0, historical_atr=2.0,
        )
        self.assertEqual(adjusted, base)


class TestEndOfDay(unittest.TestCase):
    """Test end-of-day session boundary enforcement."""

    def setUp(self):
        self.broker = PaperBroker(initial_equity=100000.0)

    def test_eod_closes_positions(self):
        """End-of-day should close all open positions."""
        # Create a position
        self.broker.positions["SPY"] = Position(
            symbol="SPY", entry_time=0, entry_price=500.0,
            shares=100, stop_price=490.0,
        )
        self.broker.cash = 50000.0  # Remaining cash after position

        result = self.broker.end_of_day(
            current_prices={"SPY": 505.0},
            timestamp=1000000,
            current_date=date(2024, 1, 5),
        )

        self.assertEqual(result["positions_closed"], 1)
        # Position should be gone (pending exit created)
        # Note: actual close happens on next process_pending_orders call
        self.assertTrue(len(self.broker.pending_orders) > 0 or len(self.broker.positions) == 0)

    def test_eod_tracks_losing_day(self):
        """End-of-day should track daily P&L for losing streak."""
        today = date.today()
        self.broker.last_reset_date = today  # Prevent reset
        # Set realized P&L (update_equity recalculates daily_pnl from realized + unrealized)
        self.broker.daily_realized_pnl = -500.0
        self.broker.daily_pnl = -500.0

        self.broker.end_of_day(
            current_prices={},
            timestamp=1000000,
            current_date=today,
        )

        self.assertEqual(self.broker.consecutive_losing_days, 1)

    def test_eod_updates_weekly_monthly(self):
        """End-of-day should update weekly and monthly P&L."""
        today = date.today()
        self.broker.last_reset_date = today  # Prevent reset
        self.broker.daily_realized_pnl = -500.0
        self.broker.daily_pnl = -500.0

        result = self.broker.end_of_day(
            current_prices={},
            timestamp=1000000,
            current_date=today,
        )

        self.assertEqual(result["weekly_pnl"], -500.0)
        self.assertEqual(result["monthly_pnl"], -500.0)


class TestComprehensiveRiskCheck(unittest.TestCase):
    """Test the comprehensive check_all_risk_controls method."""

    def setUp(self):
        self.broker = PaperBroker(initial_equity=100000.0)

    def test_all_clear(self):
        """No breaches → all clear."""
        result = self.broker.check_all_risk_controls(
            current_prices={},
            timestamp=1000000,
            current_date=date(2024, 1, 5),
        )

        self.assertFalse(result["any_breached"])
        self.assertFalse(result["daily_breached"])
        self.assertFalse(result["weekly_breached"])
        self.assertFalse(result["monthly_breached"])
        self.assertFalse(result["hwm_drawdown_breached"])
        self.assertFalse(result["consecutive_days_breached"])
        self.assertFalse(result["is_paused"])

    def test_daily_breach_detected(self):
        """Daily loss should be detected in comprehensive check."""
        today = date.today()
        self.broker.last_reset_date = today  # Prevent reset
        # Set realized P&L so update_equity computes correct daily_pnl
        self.broker.daily_realized_pnl = -3000.0
        self.broker.daily_pnl = -3000.0  # > 2% of $100k
        result = self.broker.check_all_risk_controls(
            current_prices={},
            timestamp=1000000,
            current_date=today,
        )

        self.assertTrue(result["any_breached"])
        self.assertTrue(result["daily_breached"])

    def test_hwm_breach_detected(self):
        """HWM drawdown should be detected in comprehensive check."""
        self.broker.equity = 84000.0  # 16% drawdown from $100k HWM
        self.broker.cash = 84000.0
        result = self.broker.check_all_risk_controls(
            current_prices={},
            timestamp=1000000,
            current_date=date(2024, 1, 5),
        )

        self.assertTrue(result["any_breached"])
        self.assertTrue(result["hwm_drawdown_breached"])


if __name__ == "__main__":
    unittest.main()
