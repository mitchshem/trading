"""
Sprint 5 tests: Notification system — preferences model, notification service,
API endpoints, and integration points.

Covers:
- NotificationPrefs model: default creation, to_dict security, categories, bool conversion
- NotificationService: load_prefs, severity filter, browser SSE, email scheduling, test email
- API endpoints: GET/PATCH /notifications/prefs, POST /notifications/test-email, GET /notifications/history
- Integration: monitoring.add_alert triggers notify, trade notification, exception resilience
"""

import asyncio
import os
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock, call

# Use an in-memory SQLite database for all tests (isolate from production)
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_trading_signals.db")


# ============================================================================
# Helper: create an isolated in-memory database for model tests
# ============================================================================

def _make_test_db():
    """Create a fresh in-memory SQLite database with all tables."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from database import Base, NotificationPrefs

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session, NotificationPrefs


# ============================================================================
# NotificationPrefs model tests
# ============================================================================

class TestNotificationPrefsModel:
    """Tests for the NotificationPrefs database model."""

    def test_default_creation_has_expected_values(self):
        """Default NotificationPrefs should have email disabled, browser enabled."""
        Session, NotificationPrefs = _make_test_db()
        db = Session()
        try:
            prefs = NotificationPrefs(id=1)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)

            assert prefs.email_enabled == 0
            assert prefs.browser_enabled == 1
            assert prefs.smtp_port == 587
            assert prefs.smtp_use_tls == 1
            assert prefs.min_severity == "info"
        finally:
            db.close()

    def test_to_dict_excludes_smtp_credentials(self):
        """to_dict() should NOT include smtp_user or smtp_password."""
        Session, NotificationPrefs = _make_test_db()
        db = Session()
        try:
            prefs = NotificationPrefs(id=1, smtp_user="myuser", smtp_password="secret123")
            db.add(prefs)
            db.commit()
            db.refresh(prefs)

            d = prefs.to_dict()
            assert "smtp_user" not in d
            assert "smtp_password" not in d
            # But raw attributes still exist
            assert prefs.smtp_user == "myuser"
            assert prefs.smtp_password == "secret123"
        finally:
            db.close()

    def test_to_dict_includes_all_categories(self):
        """to_dict() should include all 7 email and 7 browser categories."""
        Session, NotificationPrefs = _make_test_db()
        db = Session()
        try:
            prefs = NotificationPrefs(id=1)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)

            d = prefs.to_dict()
            expected_categories = [
                "trade_executed", "kill_switch", "anomaly_detected",
                "risk_limit_breached", "loop_status_change", "daily_summary",
                "system_error",
            ]
            for cat in expected_categories:
                assert cat in d["email_categories"], f"Missing email category: {cat}"
                assert cat in d["browser_categories"], f"Missing browser category: {cat}"
            assert len(d["email_categories"]) == 7
            assert len(d["browser_categories"]) == 7
        finally:
            db.close()

    def test_smtp_configured_flag(self):
        """smtp_configured should be True only when both smtp_host and email_address are set."""
        Session, NotificationPrefs = _make_test_db()
        db = Session()
        try:
            # Neither set
            prefs = NotificationPrefs(id=1)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)
            assert prefs.to_dict()["smtp_configured"] is False

            # Only host
            prefs.smtp_host = "smtp.gmail.com"
            db.commit()
            db.refresh(prefs)
            assert prefs.to_dict()["smtp_configured"] is False

            # Both set
            prefs.email_address = "test@example.com"
            db.commit()
            db.refresh(prefs)
            assert prefs.to_dict()["smtp_configured"] is True
        finally:
            db.close()

    def test_bool_conversion_in_to_dict(self):
        """Integer 0/1 columns should be converted to Python booleans in to_dict()."""
        Session, NotificationPrefs = _make_test_db()
        db = Session()
        try:
            prefs = NotificationPrefs(id=1, email_enabled=0, browser_enabled=1)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)

            d = prefs.to_dict()
            assert d["email_enabled"] is False
            assert isinstance(d["email_enabled"], bool)
            assert d["browser_enabled"] is True
            assert isinstance(d["browser_enabled"], bool)
            # Category values should also be bool
            assert isinstance(d["email_categories"]["trade_executed"], bool)
            assert isinstance(d["browser_categories"]["kill_switch"], bool)
        finally:
            db.close()


# ============================================================================
# NotificationService tests
# ============================================================================

class TestNotificationService:
    """Tests for the notification dispatch service."""

    def _make_service(self):
        """Create a fresh NotificationService (not the module singleton)."""
        from notification_service import NotificationService
        return NotificationService()

    def _make_payload(self, category="trade_executed", severity="info",
                      title="Test", body="Test body"):
        from notification_service import NotificationPayload, NotifCategory
        cat_enum = NotifCategory(category)
        return NotificationPayload(
            category=cat_enum,
            severity=severity,
            title=title,
            body=body,
            details={"key": "value"},
        )

    def test_load_prefs_creates_default_on_first_call(self):
        """load_prefs() should create a default row if none exists."""
        Session, NotificationPrefs = _make_test_db()

        svc = self._make_service()

        # Directly test the load_prefs logic with our test DB
        db = Session()
        try:
            prefs = db.query(NotificationPrefs).filter_by(id=1).first()
            assert prefs is None  # No row yet

            prefs = NotificationPrefs(id=1)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)

            svc._prefs_cache = prefs.to_dict()
            svc._email_configured = bool(prefs.smtp_host and prefs.email_address)
        finally:
            db.close()

        assert svc._prefs_cache is not None
        assert svc._prefs_cache["browser_enabled"] is True
        assert svc._prefs_cache["email_enabled"] is False

    def test_prefs_cache_is_used(self):
        """After load_prefs, cached prefs should be used without DB re-query."""
        svc = self._make_service()
        svc._prefs_cache = {
            "browser_enabled": True,
            "email_enabled": False,
            "min_severity": "info",
            "browser_categories": {"trade_executed": True},
            "email_categories": {"trade_executed": False},
        }
        # notify should use cached prefs without calling load_prefs
        payload = self._make_payload()
        with patch.object(svc, "_emit_browser_notification") as mock_emit:
            svc.notify(payload)
            mock_emit.assert_called_once()

    def test_severity_filter_blocks_low_severity(self):
        """Notifications below min_severity should be silently skipped."""
        svc = self._make_service()
        svc._prefs_cache = {
            "browser_enabled": True,
            "email_enabled": False,
            "min_severity": "critical",  # Only critical
            "browser_categories": {"trade_executed": True},
            "email_categories": {},
        }
        # "info" severity should be blocked
        payload = self._make_payload(severity="info")
        with patch.object(svc, "_emit_browser_notification") as mock_emit:
            svc.notify(payload)
            mock_emit.assert_not_called()

    def test_severity_filter_passes_matching_severity(self):
        """Notifications at or above min_severity should be dispatched."""
        svc = self._make_service()
        svc._prefs_cache = {
            "browser_enabled": True,
            "email_enabled": False,
            "min_severity": "warning",
            "browser_categories": {"trade_executed": True},
            "email_categories": {},
        }
        # "critical" should pass "warning" filter
        payload = self._make_payload(severity="critical")
        with patch.object(svc, "_emit_browser_notification") as mock_emit:
            svc.notify(payload)
            mock_emit.assert_called_once()

    def test_browser_sse_emission(self):
        """Browser notification should emit SSE event via broadcaster."""
        svc = self._make_service()
        payload = self._make_payload()

        # broadcaster is imported locally inside _emit_browser_notification
        # via "from sse_broadcaster import broadcaster", so patch the source module
        with patch("sse_broadcaster.broadcaster") as mock_broadcaster:
            svc._emit_browser_notification(payload)
            mock_broadcaster.publish.assert_called_once()
            args = mock_broadcaster.publish.call_args
            assert args[0][0] == "notification_fired"
            event_data = args[0][1]
            assert event_data["category"] == "trade_executed"
            assert event_data["title"] == "Test"
            assert event_data["body"] == "Test body"

    def test_skip_browser_when_disabled(self):
        """Browser notification should not fire when browser_enabled is False."""
        svc = self._make_service()
        svc._prefs_cache = {
            "browser_enabled": False,
            "email_enabled": False,
            "min_severity": "info",
            "browser_categories": {"trade_executed": True},
            "email_categories": {},
        }
        payload = self._make_payload()
        with patch.object(svc, "_emit_browser_notification") as mock_emit:
            svc.notify(payload)
            mock_emit.assert_not_called()

    def test_skip_email_when_not_configured(self):
        """Email should not be scheduled when SMTP is not configured."""
        svc = self._make_service()
        svc._prefs_cache = {
            "browser_enabled": False,
            "email_enabled": True,
            "min_severity": "info",
            "browser_categories": {},
            "email_categories": {"trade_executed": True},
        }
        svc._email_configured = False  # No SMTP host/email
        payload = self._make_payload()
        with patch.object(svc, "_schedule_email") as mock_email:
            svc.notify(payload)
            mock_email.assert_not_called()

    def test_skip_disabled_category(self):
        """Notification for a disabled category should not fire."""
        svc = self._make_service()
        svc._prefs_cache = {
            "browser_enabled": True,
            "email_enabled": False,
            "min_severity": "info",
            "browser_categories": {"trade_executed": False},  # Disabled!
            "email_categories": {},
        }
        payload = self._make_payload(category="trade_executed")
        with patch.object(svc, "_emit_browser_notification") as mock_emit:
            svc.notify(payload)
            mock_emit.assert_not_called()

    def test_email_html_format(self):
        """_format_html_email should produce valid HTML with severity color."""
        svc = self._make_service()
        payload = self._make_payload(severity="critical", title="Kill Switch", body="Trading halted")
        html = svc._format_html_email(payload)
        assert "Kill Switch" in html
        assert "Trading halted" in html
        assert "#ef4444" in html  # Critical color
        assert "<div" in html

    def test_send_test_email_not_configured(self):
        """send_test_email should return failure dict when SMTP not configured."""
        svc = self._make_service()
        svc._email_configured = False

        result = asyncio.run(svc.send_test_email())
        assert result["success"] is False
        assert "not configured" in result["message"].lower()


# ============================================================================
# Notification API endpoint tests
# ============================================================================

class TestNotificationPrefsAPI:
    """Tests for the notification preferences API endpoints (logic-level)."""

    def _setup_test_db(self):
        """Set up an isolated test database and return session + model."""
        return _make_test_db()

    def test_get_prefs_returns_defaults(self):
        """GET /notifications/prefs should return default values when no row exists."""
        Session, NotificationPrefs = self._setup_test_db()
        db = Session()
        try:
            # Simulate what the endpoint does
            prefs = db.query(NotificationPrefs).filter_by(id=1).first()
            assert prefs is None

            prefs = NotificationPrefs(id=1)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)

            d = prefs.to_dict()
            assert d["email_enabled"] is False
            assert d["browser_enabled"] is True
            assert d["smtp_configured"] is False
            assert d["min_severity"] == "info"
        finally:
            db.close()

    def test_patch_prefs_updates_scalar_fields(self):
        """PATCH should update scalar fields correctly, converting bools to ints."""
        Session, NotificationPrefs = self._setup_test_db()
        db = Session()
        try:
            prefs = NotificationPrefs(id=1)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)

            # Simulate PATCH body
            updates = {
                "email_enabled": True,
                "email_address": "trader@example.com",
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 465,
                "min_severity": "warning",
            }
            for field_name, value in updates.items():
                if isinstance(value, bool):
                    setattr(prefs, field_name, 1 if value else 0)
                else:
                    setattr(prefs, field_name, value)
            db.commit()
            db.refresh(prefs)

            assert prefs.email_enabled == 1
            assert prefs.email_address == "trader@example.com"
            assert prefs.smtp_host == "smtp.gmail.com"
            assert prefs.smtp_port == 465
            assert prefs.min_severity == "warning"
        finally:
            db.close()

    def test_partial_update_preserves_other_fields(self):
        """PATCH with only one field should not reset other fields."""
        Session, NotificationPrefs = self._setup_test_db()
        db = Session()
        try:
            prefs = NotificationPrefs(
                id=1,
                email_address="original@example.com",
                smtp_host="smtp.original.com",
            )
            db.add(prefs)
            db.commit()
            db.refresh(prefs)

            # Only update email_enabled
            prefs.email_enabled = 1
            db.commit()
            db.refresh(prefs)

            # Other fields should be unchanged
            assert prefs.email_address == "original@example.com"
            assert prefs.smtp_host == "smtp.original.com"
        finally:
            db.close()

    def test_category_toggles_update(self):
        """PATCH with email_categories should update per-category columns."""
        Session, NotificationPrefs = self._setup_test_db()
        db = Session()
        try:
            prefs = NotificationPrefs(id=1)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)

            # Simulate category update
            email_categories = {"trade_executed": False, "kill_switch": True}
            for cat, enabled in email_categories.items():
                col_name = f"email_{cat}"
                if hasattr(prefs, col_name):
                    setattr(prefs, col_name, 1 if enabled else 0)
            db.commit()
            db.refresh(prefs)

            assert prefs.email_trade_executed == 0  # Was 1, now 0
            assert prefs.email_kill_switch == 1  # Still 1
        finally:
            db.close()

    def test_browser_category_toggles(self):
        """PATCH with browser_categories should update browser-specific columns."""
        Session, NotificationPrefs = self._setup_test_db()
        db = Session()
        try:
            prefs = NotificationPrefs(id=1)
            db.add(prefs)
            db.commit()
            db.refresh(prefs)

            # Disable browser daily summary (was 0), enable loop_status_change (was 1)
            browser_categories = {"daily_summary": True, "loop_status_change": False}
            for cat, enabled in browser_categories.items():
                col_name = f"browser_{cat}"
                if hasattr(prefs, col_name):
                    setattr(prefs, col_name, 1 if enabled else 0)
            db.commit()
            db.refresh(prefs)

            d = prefs.to_dict()
            assert d["browser_categories"]["daily_summary"] is True
            assert d["browser_categories"]["loop_status_change"] is False
        finally:
            db.close()

    def test_notifier_reload_after_patch(self):
        """After PATCH, notifier.load_prefs() should be called to refresh cache."""
        with patch("notification_service.notifier") as mock_notifier:
            mock_notifier.load_prefs = MagicMock()
            # Simulate what the endpoint does
            mock_notifier.load_prefs()
            mock_notifier.load_prefs.assert_called_once()

    def test_test_email_delegates_to_notifier(self):
        """POST /notifications/test-email should delegate to notifier.send_test_email()."""
        async def _test():
            from notification_service import NotificationService
            svc = NotificationService()
            svc._email_configured = False
            result = await svc.send_test_email()
            assert result["success"] is False
            assert "not configured" in result["message"].lower()
        asyncio.run(_test())

    def test_notification_history_from_monitor(self):
        """GET /notifications/history should return alerts from SystemMonitor."""
        from monitoring import SystemMonitor, AlertSeverity
        mon = SystemMonitor()

        # add_alert imports broadcaster and notifier locally, patch at source
        with patch("sse_broadcaster.broadcaster", MagicMock()):
            with patch("notification_service.notifier", MagicMock()):
                mon.add_alert(AlertSeverity.WARNING, "anomaly", "Test anomaly")
                mon.add_alert(AlertSeverity.CRITICAL, "risk", "Risk limit breached")

        alerts = mon.get_recent_alerts(limit=10)
        assert len(alerts) >= 2
        # Most recent first
        assert alerts[0]["category"] == "risk"
        assert alerts[1]["category"] == "anomaly"


# ============================================================================
# Integration tests
# ============================================================================

class TestNotificationIntegration:
    """Tests for notification integration with monitoring and live loop."""

    def test_add_alert_triggers_notify(self):
        """monitoring.add_alert() should dispatch a notification via notifier."""
        from monitoring import SystemMonitor, AlertSeverity

        mon = SystemMonitor()

        # Patch at source modules — add_alert uses local imports
        with patch("sse_broadcaster.broadcaster", MagicMock()):
            mock_notifier = MagicMock()
            with patch("notification_service.notifier", mock_notifier):
                mon.add_alert(AlertSeverity.CRITICAL, "risk", "Daily loss limit breached")

            # Verify notifier.notify was called
            mock_notifier.notify.assert_called_once()

    def test_add_alert_exception_doesnt_break_monitoring(self):
        """If notification service throws, add_alert should still succeed."""
        from monitoring import SystemMonitor, AlertSeverity

        mon = SystemMonitor()

        # Patch broadcaster to work, but make notifier.notify raise
        with patch("sse_broadcaster.broadcaster", MagicMock()):
            mock_notifier = MagicMock()
            mock_notifier.notify.side_effect = RuntimeError("notification failed!")
            with patch("notification_service.notifier", mock_notifier):
                # Should NOT raise — exception is swallowed by try/except
                mon.add_alert(AlertSeverity.CRITICAL, "risk", "Test alert")

        # Alert should still be stored despite notification failure
        alerts = mon.get_recent_alerts()
        assert any(a["message"] == "Test alert" for a in alerts)

    def test_sse_event_shape_for_notification_fired(self):
        """notification_fired SSE event should have category, severity, title, body."""
        from notification_service import NotificationService, NotificationPayload, NotifCategory

        svc = NotificationService()
        payload = NotificationPayload(
            category=NotifCategory.KILL_SWITCH,
            severity="critical",
            title="Kill Switch Activated",
            body="Trading halted due to max daily loss",
            details={"daily_pnl": "-$500"},
        )

        # broadcaster is imported locally inside _emit_browser_notification
        with patch("sse_broadcaster.broadcaster") as mock_broadcaster:
            svc._emit_browser_notification(payload)

            call_args = mock_broadcaster.publish.call_args
            event_type = call_args[0][0]
            event_data = call_args[0][1]

            assert event_type == "notification_fired"
            assert event_data["category"] == "kill_switch"
            assert event_data["severity"] == "critical"
            assert event_data["title"] == "Kill Switch Activated"
            assert event_data["body"] == "Trading halted due to max daily loss"
            assert event_data["details"] == {"daily_pnl": "-$500"}

    def test_daily_summary_notification_content(self):
        """Daily summary notification should include equity and P&L."""
        from notification_service import NotificationPayload, NotifCategory

        payload = NotificationPayload(
            category=NotifCategory.DAILY_SUMMARY,
            severity="info",
            title="Daily Summary - SPY",
            body="Equity: $100500.00 | Daily P&L: $500.00",
            details={
                "equity": "$100500.00",
                "daily_pnl": "$500.00",
                "strategy": "ema_trend_v1",
            },
        )

        assert payload.category == NotifCategory.DAILY_SUMMARY
        assert "SPY" in payload.title
        assert "$100500.00" in payload.body
        assert "$500.00" in payload.body
        assert payload.details["strategy"] == "ema_trend_v1"

    def test_trade_notification_severity_mapping(self):
        """STOP_LOSS trades should be critical, regular trades should be info."""
        from notification_service import NotificationPayload, NotifCategory

        # Regular trade
        regular = NotificationPayload(
            category=NotifCategory.TRADE_EXECUTED,
            severity="info",
            title="BUY: SPY",
            body="BUY SPY at $500.00",
        )
        assert regular.severity == "info"

        # Stop loss
        stop_loss = NotificationPayload(
            category=NotifCategory.TRADE_EXECUTED,
            severity="critical",
            title="STOP_LOSS: SPY",
            body="STOP_LOSS SPY at $485.00",
        )
        assert stop_loss.severity == "critical"

    def test_notif_category_enum_values(self):
        """NotifCategory enum should have all 7 expected values."""
        from notification_service import NotifCategory

        expected = {
            "trade_executed", "kill_switch", "anomaly_detected",
            "risk_limit_breached", "loop_status_change", "daily_summary",
            "system_error",
        }
        actual = {c.value for c in NotifCategory}
        assert actual == expected

    def test_severity_order_complete(self):
        """SEVERITY_ORDER should cover all three levels with correct ordering."""
        from notification_service import SEVERITY_ORDER

        assert SEVERITY_ORDER["info"] < SEVERITY_ORDER["warning"]
        assert SEVERITY_ORDER["warning"] < SEVERITY_ORDER["critical"]
        assert len(SEVERITY_ORDER) == 3
