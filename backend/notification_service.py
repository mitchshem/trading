"""
Notification service for the trading system.

Dispatches notifications via:
- Email (aiosmtplib SMTP)
- Browser push (SSE event → frontend Notification API)

Designed to degrade gracefully: no SMTP configured = email silently skipped.
No third-party services required.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NotifCategory(str, Enum):
    """Notification event categories."""
    TRADE_EXECUTED = "trade_executed"
    KILL_SWITCH = "kill_switch"
    ANOMALY_DETECTED = "anomaly_detected"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    LOOP_STATUS_CHANGE = "loop_status_change"
    DAILY_SUMMARY = "daily_summary"
    SYSTEM_ERROR = "system_error"


SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class NotificationPayload:
    """Payload for a notification dispatch."""
    category: NotifCategory
    severity: str           # "info", "warning", "critical"
    title: str
    body: str
    details: Optional[Dict] = field(default=None)


class NotificationService:
    """
    Singleton notification dispatcher.

    Loads preferences from the database on startup and caches them.
    Dispatches to browser (via SSE) and email (via aiosmtplib) channels.
    Never raises exceptions — all failures are logged and swallowed.
    """

    def __init__(self):
        self._prefs_cache: Optional[Dict] = None
        self._email_configured = False
        self._smtp_user: Optional[str] = None
        self._smtp_password: Optional[str] = None
        self._smtp_host: Optional[str] = None
        self._smtp_port: int = 587
        self._smtp_use_tls: bool = True
        self._email_address: Optional[str] = None

    def load_prefs(self):
        """Load notification preferences from database. Called on startup and after updates."""
        try:
            from database import SessionLocal, NotificationPrefs
            db = SessionLocal()
            try:
                prefs = db.query(NotificationPrefs).filter_by(id=1).first()
                if prefs is None:
                    # Create default prefs row
                    prefs = NotificationPrefs(id=1)
                    db.add(prefs)
                    db.commit()
                    db.refresh(prefs)
                self._prefs_cache = prefs.to_dict()
                # Cache raw SMTP creds separately (not in to_dict for security)
                self._smtp_user = prefs.smtp_user
                self._smtp_password = prefs.smtp_password
                self._smtp_host = prefs.smtp_host
                self._smtp_port = prefs.smtp_port or 587
                self._smtp_use_tls = bool(prefs.smtp_use_tls)
                self._email_address = prefs.email_address
                self._email_configured = bool(prefs.smtp_host and prefs.email_address)
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to load notification prefs: {e}")

    def notify(self, payload: NotificationPayload):
        """
        Dispatch a notification through enabled channels.
        Non-blocking: email is sent in a background task.
        """
        if self._prefs_cache is None:
            self.load_prefs()

        if self._prefs_cache is None:
            return  # Still None after load attempt — skip

        prefs = self._prefs_cache
        category = payload.category.value
        severity = payload.severity

        # Check minimum severity filter
        min_sev = prefs.get("min_severity", "info")
        if SEVERITY_ORDER.get(severity, 0) < SEVERITY_ORDER.get(min_sev, 0):
            return

        # Browser notification (always immediate via SSE)
        if prefs.get("browser_enabled") and prefs.get("browser_categories", {}).get(category):
            self._emit_browser_notification(payload)

        # Email notification (async fire-and-forget)
        if (prefs.get("email_enabled") and self._email_configured
                and prefs.get("email_categories", {}).get(category)):
            self._schedule_email(payload)

    def _emit_browser_notification(self, payload: NotificationPayload):
        """Emit SSE event for browser Notification API."""
        try:
            from sse_broadcaster import broadcaster
            broadcaster.publish("notification_fired", {
                "category": payload.category.value,
                "severity": payload.severity,
                "title": payload.title,
                "body": payload.body,
                "details": payload.details,
            })
        except Exception as e:
            logger.debug(f"Failed to emit browser notification: {e}")

    def _schedule_email(self, payload: NotificationPayload):
        """Schedule async email send. Fire-and-forget."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send_email(payload))
        except RuntimeError:
            # No event loop running (e.g., during tests) — skip
            logger.debug("No event loop for email send — skipping")

    async def _send_email(self, payload: NotificationPayload):
        """Send email via aiosmtplib."""
        try:
            import aiosmtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[Trading] {payload.title}"
            msg["From"] = self._email_address
            msg["To"] = self._email_address

            # Plain text body
            text_body = f"{payload.title}\n\n{payload.body}"
            if payload.details:
                text_body += "\n\nDetails:\n"
                for k, v in payload.details.items():
                    text_body += f"  {k}: {v}\n"

            # Simple HTML body
            html_body = self._format_html_email(payload)

            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            await aiosmtplib.send(
                msg,
                hostname=self._smtp_host,
                port=self._smtp_port,
                username=self._smtp_user,
                password=self._smtp_password,
                use_tls=self._smtp_use_tls,
                start_tls=not self._smtp_use_tls,
            )
            logger.info(f"Email sent: {payload.title}")

        except ImportError:
            logger.warning("aiosmtplib not installed — email disabled")
        except Exception as e:
            logger.error(f"Email send failed: {e}")

    def _format_html_email(self, payload: NotificationPayload) -> str:
        """Format a minimal HTML email body with dark theme styling."""
        severity_color = {
            "info": "#3b82f6",
            "warning": "#eab308",
            "critical": "#ef4444",
        }.get(payload.severity, "#6b7280")

        details_html = ""
        if payload.details:
            rows = "".join(
                f"<tr><td style='padding:4px 8px;color:#9ca3af'>{k}</td>"
                f"<td style='padding:4px 8px;color:#fff'>{v}</td></tr>"
                for k, v in payload.details.items()
            )
            details_html = f"<table style='margin-top:12px'>{rows}</table>"

        return f"""
        <div style="background:#1a1a1a;padding:24px;font-family:sans-serif;color:#fff;max-width:500px">
            <div style="border-left:4px solid {severity_color};padding-left:12px;margin-bottom:16px">
                <h2 style="margin:0;font-size:16px">{payload.title}</h2>
                <span style="color:{severity_color};font-size:12px;text-transform:uppercase">{payload.severity}</span>
            </div>
            <p style="color:#d1d5db;font-size:14px;line-height:1.5">{payload.body}</p>
            {details_html}
            <hr style="border-color:#333;margin-top:20px">
            <p style="color:#6b7280;font-size:11px">Trading System Notification &bull; {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
        </div>
        """

    async def send_test_email(self) -> Dict:
        """Send a test email to verify SMTP configuration. Returns success/failure dict."""
        if not self._email_configured:
            return {"success": False, "message": "SMTP not configured. Set email address and SMTP host in notification preferences."}
        try:
            test_payload = NotificationPayload(
                category=NotifCategory.SYSTEM_ERROR,
                severity="info",
                title="Test Notification",
                body="This is a test email from your trading system. If you received this, email notifications are working correctly.",
            )
            await self._send_email(test_payload)
            return {"success": True, "message": "Test email sent successfully"}
        except Exception as e:
            return {"success": False, "message": str(e)}


# Module-level singleton
notifier = NotificationService()
