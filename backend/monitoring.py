"""
Performance monitoring module for the trading system.

Phase 5: Tracks system health, API performance, data freshness,
and error rates to ensure production readiness.
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from collections import deque
from enum import Enum


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A system alert."""
    timestamp: datetime
    severity: AlertSeverity
    category: str
    message: str
    details: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class APICallMetric:
    """Tracks a single API call."""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    timestamp: datetime


class SystemMonitor:
    """
    Centralized system monitoring.

    Tracks:
    - API response times per endpoint
    - Data freshness (last candle fetch)
    - Error rates and recent errors
    - Live loop health (evaluations, anomalies)
    - System uptime
    """

    def __init__(self, max_history: int = 1000, max_alerts: int = 200):
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)

        # API call history (bounded deque)
        self._api_calls: deque[APICallMetric] = deque(maxlen=max_history)

        # Error tracking
        self._errors: deque[Dict] = deque(maxlen=max_history)
        self._error_count: int = 0

        # Alerts
        self._alerts: deque[Alert] = deque(maxlen=max_alerts)

        # Data freshness
        self._last_candle_fetch: Optional[datetime] = None
        self._last_candle_symbol: Optional[str] = None
        self._candle_fetch_count: int = 0

        # Live loop tracking
        self._evaluation_count: int = 0
        self._trade_count: int = 0
        self._anomaly_count: int = 0
        self._last_evaluation_time: Optional[datetime] = None
        self._last_trade_time: Optional[datetime] = None

        # WebSocket tracking
        self._ws_connections_active: int = 0
        self._ws_messages_sent: int = 0

    # ── API Tracking ──

    def record_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
    ) -> None:
        """Record an API call for performance tracking."""
        metric = APICallMetric(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(timezone.utc),
        )
        with self._lock:
            self._api_calls.append(metric)
            if status_code >= 400:
                self._error_count += 1
                self._errors.append({
                    "timestamp": metric.timestamp.isoformat(),
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": status_code,
                })
                if status_code >= 500:
                    self._alerts.append(Alert(
                        timestamp=metric.timestamp,
                        severity=AlertSeverity.CRITICAL,
                        category="api_error",
                        message=f"Server error {status_code} on {method} {endpoint}",
                        details={"status_code": status_code},
                    ))

    def get_api_stats(self, last_minutes: int = 5) -> Dict:
        """Get API performance stats for the last N minutes."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=last_minutes)
        with self._lock:
            recent = [c for c in self._api_calls if c.timestamp >= cutoff]

        if not recent:
            return {
                "total_calls": 0,
                "avg_response_ms": 0,
                "p95_response_ms": 0,
                "error_rate": 0,
                "endpoints": {},
            }

        response_times = [c.response_time_ms for c in recent]
        response_times.sort()
        errors = sum(1 for c in recent if c.status_code >= 400)

        # Per-endpoint stats
        endpoints: Dict[str, List[float]] = {}
        for c in recent:
            key = f"{c.method} {c.endpoint}"
            endpoints.setdefault(key, []).append(c.response_time_ms)

        endpoint_stats = {}
        for key, times in endpoints.items():
            times.sort()
            endpoint_stats[key] = {
                "calls": len(times),
                "avg_ms": round(sum(times) / len(times), 1),
                "p95_ms": round(times[int(len(times) * 0.95)] if times else 0, 1),
            }

        p95_idx = int(len(response_times) * 0.95)

        return {
            "total_calls": len(recent),
            "avg_response_ms": round(sum(response_times) / len(response_times), 1),
            "p95_response_ms": round(response_times[p95_idx] if response_times else 0, 1),
            "error_rate": round(errors / len(recent), 4) if recent else 0,
            "endpoints": endpoint_stats,
        }

    # ── Data Freshness ──

    def record_candle_fetch(self, symbol: str) -> None:
        """Record that a candle was fetched."""
        with self._lock:
            self._last_candle_fetch = datetime.now(timezone.utc)
            self._last_candle_symbol = symbol
            self._candle_fetch_count += 1

    def get_data_freshness(self) -> Dict:
        """Get data freshness status."""
        with self._lock:
            last_fetch = self._last_candle_fetch
            symbol = self._last_candle_symbol
            count = self._candle_fetch_count

        now = datetime.now(timezone.utc)

        if last_fetch is None:
            return {
                "status": "no_data",
                "last_fetch": None,
                "last_symbol": None,
                "age_seconds": None,
                "fetch_count": 0,
            }

        age = (now - last_fetch).total_seconds()

        # For daily strategy: data older than 25 hours is stale
        status = "fresh" if age < 90000 else "stale"

        return {
            "status": status,
            "last_fetch": last_fetch.isoformat(),
            "last_symbol": symbol,
            "age_seconds": round(age, 1),
            "fetch_count": count,
        }

    # ── Live Loop Tracking ──

    def record_evaluation(self) -> None:
        """Record a strategy evaluation."""
        with self._lock:
            self._evaluation_count += 1
            self._last_evaluation_time = datetime.now(timezone.utc)

    def record_trade(self) -> None:
        """Record a trade execution."""
        with self._lock:
            self._trade_count += 1
            self._last_trade_time = datetime.now(timezone.utc)

    def record_anomaly(self, anomaly_type: str, details: Optional[str] = None) -> None:
        """Record an anomaly detection."""
        now = datetime.now(timezone.utc)
        with self._lock:
            self._anomaly_count += 1
            self._alerts.append(Alert(
                timestamp=now,
                severity=AlertSeverity.WARNING,
                category="anomaly",
                message=f"Anomaly detected: {anomaly_type}",
                details={"type": anomaly_type, "details": details},
            ))

    # ── Error Tracking ──

    def record_error(self, category: str, message: str, details: Optional[Dict] = None) -> None:
        """Record a system error."""
        now = datetime.now(timezone.utc)
        with self._lock:
            self._error_count += 1
            self._errors.append({
                "timestamp": now.isoformat(),
                "category": category,
                "message": message,
                "details": details,
            })
            self._alerts.append(Alert(
                timestamp=now,
                severity=AlertSeverity.CRITICAL,
                category=category,
                message=message,
                details=details,
            ))

    # ── Alert Management ──

    def add_alert(self, severity: AlertSeverity, category: str, message: str, details: Optional[Dict] = None) -> None:
        """Add a custom alert. Also emits SSE event for real-time notification."""
        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            category=category,
            message=message,
            details=details,
        )
        with self._lock:
            self._alerts.append(alert)

        # Emit SSE event for real-time delivery to frontend
        try:
            from sse_broadcaster import broadcaster
            broadcaster.publish("alert_fired", alert.to_dict())
        except Exception:
            pass  # Don't fail monitoring if SSE isn't available

        # Dispatch notification (email + browser push)
        try:
            from notification_service import notifier, NotificationPayload, NotifCategory
            cat_map = {
                "api_error": NotifCategory.SYSTEM_ERROR,
                "anomaly": NotifCategory.ANOMALY_DETECTED,
                "risk": NotifCategory.RISK_LIMIT_BREACHED,
                "kill_switch": NotifCategory.KILL_SWITCH,
                "system": NotifCategory.LOOP_STATUS_CHANGE,
            }
            notif_cat = cat_map.get(category, NotifCategory.SYSTEM_ERROR)
            notifier.notify(NotificationPayload(
                category=notif_cat,
                severity=severity.value,
                title=message,
                body=message,
                details=details,
            ))
        except Exception:
            pass  # Don't fail monitoring if notification service isn't available

    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts."""
        with self._lock:
            alerts = list(self._alerts)

        # Return most recent first
        alerts.reverse()
        return [a.to_dict() for a in alerts[:limit]]

    # ── WebSocket Tracking ──

    def ws_connected(self) -> None:
        """Record a WebSocket connection."""
        with self._lock:
            self._ws_connections_active += 1

    def ws_disconnected(self) -> None:
        """Record a WebSocket disconnection."""
        with self._lock:
            self._ws_connections_active = max(0, self._ws_connections_active - 1)

    def ws_message_sent(self) -> None:
        """Record a WebSocket message sent."""
        with self._lock:
            self._ws_messages_sent += 1

    # ── Comprehensive Status ──

    def get_system_status(self) -> Dict:
        """
        Get a comprehensive system health snapshot.
        Returns all monitoring data in a single call.
        """
        now = datetime.now(timezone.utc)
        uptime = (now - self._started_at).total_seconds()

        api_stats = self.get_api_stats()
        data_freshness = self.get_data_freshness()

        with self._lock:
            last_eval = self._last_evaluation_time
            last_trade = self._last_trade_time
            eval_count = self._evaluation_count
            trade_count = self._trade_count
            anomaly_count = self._anomaly_count
            error_count = self._error_count
            recent_errors = list(self._errors)[-5:]
            ws_active = self._ws_connections_active
            ws_messages = self._ws_messages_sent

        # Determine overall health
        health = "healthy"
        health_issues = []

        if api_stats["error_rate"] > 0.1:
            health = "degraded"
            health_issues.append("High API error rate")

        if data_freshness["status"] == "stale":
            health = "degraded"
            health_issues.append("Data is stale")

        if error_count > 0 and recent_errors:
            last_error_time = recent_errors[-1].get("timestamp", "")
            if last_error_time:
                try:
                    error_dt = datetime.fromisoformat(last_error_time)
                    if (now - error_dt).total_seconds() < 300:  # Error in last 5 min
                        health = "degraded"
                        health_issues.append("Recent errors detected")
                except (ValueError, TypeError):
                    pass

        return {
            "health": health,
            "health_issues": health_issues,
            "uptime_seconds": round(uptime, 1),
            "started_at": self._started_at.isoformat(),
            "api": api_stats,
            "data_freshness": data_freshness,
            "live_loop": {
                "evaluations": eval_count,
                "trades": trade_count,
                "anomalies": anomaly_count,
                "last_evaluation": last_eval.isoformat() if last_eval else None,
                "last_trade": last_trade.isoformat() if last_trade else None,
            },
            "errors": {
                "total": error_count,
                "recent": recent_errors,
            },
            "websocket": {
                "active_connections": ws_active,
                "messages_sent": ws_messages,
            },
        }


# Singleton instance for application-wide monitoring
monitor = SystemMonitor()


class APITimingMiddleware:
    """
    ASGI middleware that records API response times to the monitor.

    Usage:
        app.add_middleware(APITimingMiddleware)
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = 200

        # Capture the response status code
        original_send = send
        async def capture_send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
            await original_send(message)

        try:
            await self.app(scope, receive, capture_send)
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            path = scope.get("path", "unknown")
            method = scope.get("method", "GET")
            monitor.record_api_call(path, method, status_code, elapsed_ms)
