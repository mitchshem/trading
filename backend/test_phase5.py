"""
Phase 5 Tests: Integration & Live Readiness

Tests for:
- Strengthened promotion rules (trading days, profit factor, OOS windows)
- System monitoring (API tracking, data freshness, alerts, health)
- End-to-end integration (promotion + monitoring working together)
"""

import pytest
import time
from datetime import datetime, timezone, timedelta, date
from unittest.mock import MagicMock, patch

from promotion_rules import PromotionRules, PromotionThresholds, PromotionDecision
from monitoring import SystemMonitor, AlertSeverity, APITimingMiddleware


# ============================================================================
# Helper: Create mock trades and equity curve for promotion tests
# ============================================================================

def make_mock_trade(entry_date_str, exit_date_str, pnl, shares=100, entry_price=500.0):
    """Create a mock Trade object."""
    trade = MagicMock()
    trade.entry_time = datetime.fromisoformat(entry_date_str).replace(tzinfo=timezone.utc)
    trade.exit_time = datetime.fromisoformat(exit_date_str).replace(tzinfo=timezone.utc)
    trade.entry_price = entry_price
    trade.exit_price = entry_price + (pnl / shares)
    trade.shares = shares
    trade.pnl = pnl
    trade.symbol = "SPY"
    trade.reason = "test"
    trade.replay_id = None
    return trade


def make_mock_equity(date_str, equity):
    """Create a mock EquityCurve object."""
    eq = MagicMock()
    eq.timestamp = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    eq.equity = equity
    eq.replay_id = None
    return eq


def make_passing_trades_and_equity(num_trades=25, num_days=25):
    """
    Create a set of trades and equity curve that should pass
    all promotion thresholds.
    """
    trades = []
    equity_points = []
    base_equity = 100000.0
    current_equity = base_equity

    for i in range(num_trades):
        day_offset = i
        entry_date = f"2025-01-{(day_offset % 28) + 1:02d}T14:30:00"
        exit_date = f"2025-01-{(day_offset % 28) + 1:02d}T19:30:00"

        # 60% winners, 40% losers — creates a positive edge
        if i % 5 < 3:  # 60% win rate
            pnl = 200.0 + (i * 10)  # Wins are bigger
        else:
            pnl = -150.0  # Losses are smaller

        current_equity += pnl
        trades.append(make_mock_trade(entry_date, exit_date, pnl))
        equity_points.append(make_mock_equity(entry_date, current_equity))

    # Add extra equity points for more trading days if needed
    for i in range(num_trades, num_days):
        day_offset = i
        eq_date = f"2025-02-{(day_offset % 28) + 1:02d}T20:00:00"
        equity_points.append(make_mock_equity(eq_date, current_equity))

    return trades, equity_points


# ============================================================================
# Promotion Rules Tests
# ============================================================================

class TestPromotionRulesPhase5:
    """Test the strengthened promotion rules."""

    def test_trading_days_insufficient(self):
        """Promotion fails if trading days < 20."""
        thresholds = PromotionThresholds(
            min_trade_count=1,
            min_trading_days=20,
            min_oos_windows_positive=0,
        )
        rules = PromotionRules(thresholds)

        # Only 5 days of data
        trades = [make_mock_trade("2025-01-01T14:30:00", "2025-01-01T19:30:00", 100.0)]
        equity = [make_mock_equity("2025-01-01T20:00:00", 100100.0)]

        decision = rules.evaluate_promotion(trades, equity, oos_positive_windows=5)
        assert not decision.promoted
        assert any("Trading days" in r and "✓" not in r for r in decision.reasons)

    def test_trading_days_sufficient(self):
        """Trading days check passes with enough unique dates."""
        thresholds = PromotionThresholds(
            min_trading_days=5,
            min_trade_count=1,
            min_oos_windows_positive=0,
        )
        rules = PromotionRules(thresholds)

        trades = []
        equity = []
        for i in range(10):
            date_str = f"2025-01-{i + 1:02d}T14:30:00"
            trades.append(make_mock_trade(date_str, date_str, 100.0))
            equity.append(make_mock_equity(date_str, 100000.0 + (i * 100)))

        decision = rules.evaluate_promotion(trades, equity, oos_positive_windows=5)
        # Check the trading days rule specifically passed
        assert any("Trading days" in r and "✓" in r for r in decision.reasons)

    def test_profit_factor_insufficient(self):
        """Promotion fails if profit factor < 1.0."""
        thresholds = PromotionThresholds(
            min_trade_count=2,
            min_profit_factor=1.5,
            min_trading_days=1,
            min_oos_windows_positive=0,
        )
        rules = PromotionRules(thresholds)

        # More losses than wins → profit factor < 1
        trades = [
            make_mock_trade("2025-01-01T14:30:00", "2025-01-01T19:30:00", -200.0),
            make_mock_trade("2025-01-01T14:30:00", "2025-01-01T19:30:00", 100.0),
        ]
        equity = [make_mock_equity("2025-01-01T20:00:00", 99900.0)]

        decision = rules.evaluate_promotion(trades, equity, oos_positive_windows=5)
        assert not decision.promoted
        assert any("Profit factor" in r and "✓" not in r for r in decision.reasons)

    def test_oos_windows_insufficient(self):
        """Promotion fails if OOS positive windows < threshold."""
        thresholds = PromotionThresholds(
            min_trade_count=1,
            min_trading_days=1,
            min_oos_windows_positive=3,
        )
        rules = PromotionRules(thresholds)

        trades = [make_mock_trade("2025-01-01T14:30:00", "2025-01-01T19:30:00", 100.0)]
        equity = [make_mock_equity("2025-01-01T20:00:00", 100100.0)]

        # Only 1 OOS window, need 3
        decision = rules.evaluate_promotion(trades, equity, oos_positive_windows=1)
        assert not decision.promoted
        assert any("OOS positive windows" in r and "✓" not in r for r in decision.reasons)

    def test_oos_windows_sufficient(self):
        """OOS windows check passes when enough positive windows exist."""
        thresholds = PromotionThresholds(
            min_trade_count=1,
            min_trading_days=1,
            min_oos_windows_positive=2,
        )
        rules = PromotionRules(thresholds)

        trades = [make_mock_trade("2025-01-01T14:30:00", "2025-01-01T19:30:00", 100.0)]
        equity = [make_mock_equity("2025-01-01T20:00:00", 100100.0)]

        decision = rules.evaluate_promotion(trades, equity, oos_positive_windows=3)
        assert any("OOS positive windows" in r and "✓" in r for r in decision.reasons)

    def test_readiness_percentage(self):
        """Readiness percentage is calculated correctly."""
        thresholds = PromotionThresholds(
            min_trade_count=100,  # Will fail — not enough trades
            min_trading_days=1,
            min_oos_windows_positive=0,
        )
        rules = PromotionRules(thresholds)

        trades = [make_mock_trade("2025-01-01T14:30:00", "2025-01-01T19:30:00", 100.0)]
        equity = [make_mock_equity("2025-01-01T20:00:00", 100100.0)]

        decision = rules.evaluate_promotion(trades, equity, oos_positive_windows=5)
        assert decision.checks_total == 8  # 8 checks total
        assert decision.checks_passed < decision.checks_total
        assert 0 <= decision.readiness_pct <= 100

    def test_all_checks_pass(self):
        """All 8 checks pass with good data."""
        thresholds = PromotionThresholds(
            min_trade_count=5,
            min_trading_days=5,
            min_oos_windows_positive=1,
            sharpe_proxy_min=0.0,
            min_win_rate=0.3,
            min_expectancy=-1000,
            min_profit_factor=0.5,
            max_drawdown_pct=50.0,
        )
        rules = PromotionRules(thresholds)

        trades, equity = make_passing_trades_and_equity(num_trades=10, num_days=10)
        decision = rules.evaluate_promotion(trades, equity, oos_positive_windows=3)

        assert decision.checks_total == 8
        assert decision.checks_passed == 8
        assert decision.readiness_pct == 100.0
        assert decision.promoted is True

    def test_exit_trades_always_allowed(self):
        """EXIT trades are never blocked by promotion rules."""
        rules = PromotionRules(PromotionThresholds(min_trade_count=99999))
        allowed, reason = rules.check_trade_allowed([], [], trade_type="EXIT")
        assert allowed is True
        assert reason is None

    def test_buy_blocked_when_not_promoted(self):
        """BUY trades are blocked when promotion criteria not met."""
        rules = PromotionRules(PromotionThresholds(min_trade_count=99999))
        trades = [make_mock_trade("2025-01-01T14:30:00", "2025-01-01T19:30:00", 100.0)]
        equity = [make_mock_equity("2025-01-01T20:00:00", 100100.0)]
        allowed, reason = rules.check_trade_allowed(trades, equity, trade_type="BUY")
        assert allowed is False
        assert "promotion blocked" in reason

    def test_count_trading_days_with_string_dates(self):
        """Trading day counter handles string timestamps."""
        rules = PromotionRules()

        trade = MagicMock()
        trade.entry_time = "2025-01-15T14:30:00"
        trade.exit_time = "2025-01-16T19:30:00"
        trade.pnl = 100.0
        trade.shares = 100
        trade.entry_price = 500.0
        trade.exit_price = 501.0
        trade.symbol = "SPY"
        trade.replay_id = None

        days = rules._count_trading_days([trade], [])
        assert days == 2  # Jan 15 and Jan 16


# ============================================================================
# System Monitor Tests
# ============================================================================

class TestSystemMonitor:
    """Test the SystemMonitor class."""

    def test_initial_state(self):
        """Monitor starts in a healthy state."""
        mon = SystemMonitor()
        status = mon.get_system_status()
        assert status["health"] == "healthy"
        assert status["live_loop"]["evaluations"] == 0
        assert status["errors"]["total"] == 0

    def test_record_api_call(self):
        """API calls are recorded and stats computed."""
        mon = SystemMonitor()
        mon.record_api_call("/account", "GET", 200, 15.5)
        mon.record_api_call("/trades", "GET", 200, 22.3)

        stats = mon.get_api_stats(last_minutes=5)
        assert stats["total_calls"] == 2
        assert stats["avg_response_ms"] > 0
        assert stats["error_rate"] == 0

    def test_api_error_tracking(self):
        """API errors are counted and generate alerts."""
        mon = SystemMonitor()
        mon.record_api_call("/bad", "GET", 500, 100.0)

        stats = mon.get_api_stats(last_minutes=5)
        assert stats["error_rate"] == 1.0

        alerts = mon.get_recent_alerts()
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"

    def test_data_freshness_no_data(self):
        """Data freshness reports 'no_data' initially."""
        mon = SystemMonitor()
        freshness = mon.get_data_freshness()
        assert freshness["status"] == "no_data"
        assert freshness["fetch_count"] == 0

    def test_data_freshness_after_fetch(self):
        """Data freshness reports 'fresh' after a candle fetch."""
        mon = SystemMonitor()
        mon.record_candle_fetch("SPY")

        freshness = mon.get_data_freshness()
        assert freshness["status"] == "fresh"
        assert freshness["last_symbol"] == "SPY"
        assert freshness["fetch_count"] == 1

    def test_evaluation_tracking(self):
        """Strategy evaluations are counted."""
        mon = SystemMonitor()
        mon.record_evaluation()
        mon.record_evaluation()
        mon.record_evaluation()

        status = mon.get_system_status()
        assert status["live_loop"]["evaluations"] == 3
        assert status["live_loop"]["last_evaluation"] is not None

    def test_trade_tracking(self):
        """Trades are counted."""
        mon = SystemMonitor()
        mon.record_trade()

        status = mon.get_system_status()
        assert status["live_loop"]["trades"] == 1

    def test_anomaly_tracking(self):
        """Anomalies are counted and generate alerts."""
        mon = SystemMonitor()
        mon.record_anomaly("GAP", "Gap up 6.2%")

        status = mon.get_system_status()
        assert status["live_loop"]["anomalies"] == 1

        alerts = mon.get_recent_alerts()
        assert len(alerts) == 1
        assert "GAP" in alerts[0]["message"]

    def test_error_recording(self):
        """Errors are tracked and generate alerts."""
        mon = SystemMonitor()
        mon.record_error("data_fetch", "Yahoo Finance timeout")

        status = mon.get_system_status()
        assert status["errors"]["total"] == 1

        alerts = mon.get_recent_alerts()
        assert any("Yahoo Finance timeout" in a["message"] for a in alerts)

    def test_custom_alerts(self):
        """Custom alerts can be added."""
        mon = SystemMonitor()
        mon.add_alert(AlertSeverity.INFO, "system", "Trading started")
        mon.add_alert(AlertSeverity.WARNING, "risk", "Kill switch near threshold")

        alerts = mon.get_recent_alerts()
        assert len(alerts) == 2

    def test_websocket_tracking(self):
        """WebSocket connections are tracked."""
        mon = SystemMonitor()
        mon.ws_connected()
        mon.ws_connected()
        mon.ws_message_sent()

        status = mon.get_system_status()
        assert status["websocket"]["active_connections"] == 2
        assert status["websocket"]["messages_sent"] == 1

        mon.ws_disconnected()
        status = mon.get_system_status()
        assert status["websocket"]["active_connections"] == 1

    def test_ws_disconnect_floor_zero(self):
        """WebSocket count doesn't go below zero."""
        mon = SystemMonitor()
        mon.ws_disconnected()
        mon.ws_disconnected()

        status = mon.get_system_status()
        assert status["websocket"]["active_connections"] == 0

    def test_health_degrades_on_high_error_rate(self):
        """Health degrades when API error rate exceeds 10%."""
        mon = SystemMonitor()
        # 5 errors out of 10 calls = 50% error rate
        for _ in range(5):
            mon.record_api_call("/ok", "GET", 200, 10.0)
        for _ in range(5):
            mon.record_api_call("/bad", "GET", 500, 10.0)

        status = mon.get_system_status()
        assert status["health"] == "degraded"
        assert "High API error rate" in status["health_issues"]

    def test_api_stats_per_endpoint(self):
        """API stats break down by endpoint."""
        mon = SystemMonitor()
        mon.record_api_call("/account", "GET", 200, 10.0)
        mon.record_api_call("/account", "GET", 200, 20.0)
        mon.record_api_call("/trades", "GET", 200, 5.0)

        stats = mon.get_api_stats(last_minutes=5)
        assert "GET /account" in stats["endpoints"]
        assert stats["endpoints"]["GET /account"]["calls"] == 2
        assert "GET /trades" in stats["endpoints"]

    def test_uptime_tracked(self):
        """System uptime is tracked from creation."""
        mon = SystemMonitor()
        time.sleep(0.15)  # Ensure measurable uptime
        status = mon.get_system_status()
        assert status["uptime_seconds"] >= 0.1
        assert status["started_at"] is not None

    def test_bounded_history(self):
        """History deques don't grow unbounded."""
        mon = SystemMonitor(max_history=5, max_alerts=3)

        for i in range(10):
            mon.record_api_call(f"/test-{i}", "GET", 200, 10.0)

        stats = mon.get_api_stats(last_minutes=5)
        assert stats["total_calls"] == 5  # Only last 5 retained

        for i in range(10):
            mon.add_alert(AlertSeverity.INFO, "test", f"Alert {i}")

        alerts = mon.get_recent_alerts(limit=100)
        assert len(alerts) == 3  # Only last 3 retained

    def test_empty_api_stats(self):
        """API stats handle empty state gracefully."""
        mon = SystemMonitor()
        stats = mon.get_api_stats(last_minutes=5)
        assert stats["total_calls"] == 0
        assert stats["avg_response_ms"] == 0

    def test_thread_safety(self):
        """Monitor operations are thread-safe."""
        import threading
        mon = SystemMonitor()

        def record_calls():
            for i in range(100):
                mon.record_api_call("/test", "GET", 200, 10.0)

        threads = [threading.Thread(target=record_calls) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = mon.get_api_stats(last_minutes=5)
        assert stats["total_calls"] == 500


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase5Integration:
    """Test Phase 5 components working together."""

    def test_promotion_decision_metrics_dict(self):
        """Promotion decision includes all Phase 5 metrics."""
        rules = PromotionRules(PromotionThresholds(
            min_trade_count=1,
            min_trading_days=1,
            min_oos_windows_positive=0,
        ))
        trades = [make_mock_trade("2025-01-01T14:30:00", "2025-01-01T19:30:00", 100.0)]
        equity = [make_mock_equity("2025-01-01T20:00:00", 100100.0)]

        decision = rules.evaluate_promotion(trades, equity, oos_positive_windows=3)

        # Verify all Phase 5 additions are in metrics dict
        assert "trading_days" in decision.metrics
        assert "oos_positive_windows" in decision.metrics
        assert "profit_factor" in decision.metrics
        assert decision.metrics["oos_positive_windows"] == 3

    def test_monitor_status_structure(self):
        """Monitor status has expected top-level structure."""
        mon = SystemMonitor()
        status = mon.get_system_status()

        required_keys = [
            "health", "health_issues", "uptime_seconds",
            "started_at", "api", "data_freshness",
            "live_loop", "errors", "websocket",
        ]
        for key in required_keys:
            assert key in status, f"Missing key: {key}"

    def test_monitor_singleton_import(self):
        """The module-level monitor singleton is available."""
        from monitoring import monitor as imported_monitor
        assert isinstance(imported_monitor, SystemMonitor)

    def test_alert_ordering(self):
        """Alerts are returned most-recent-first."""
        mon = SystemMonitor()
        mon.add_alert(AlertSeverity.INFO, "test", "First")
        time.sleep(0.01)
        mon.add_alert(AlertSeverity.INFO, "test", "Second")
        time.sleep(0.01)
        mon.add_alert(AlertSeverity.INFO, "test", "Third")

        alerts = mon.get_recent_alerts(limit=10)
        assert alerts[0]["message"] == "Third"
        assert alerts[2]["message"] == "First"

    def test_eight_promotion_checks(self):
        """Promotion evaluates exactly 8 checks."""
        rules = PromotionRules()
        trades = [make_mock_trade("2025-01-01T14:30:00", "2025-01-01T19:30:00", 100.0)]
        equity = [make_mock_equity("2025-01-01T20:00:00", 100100.0)]

        decision = rules.evaluate_promotion(trades, equity, oos_positive_windows=0)
        assert decision.checks_total == 8
        assert len(decision.reasons) == 8
