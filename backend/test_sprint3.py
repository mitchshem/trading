"""
Sprint 3 tests: Alpaca data client, SSE broadcaster, LiveTradingLoop data provider.

Covers:
- AlpacaDataService: configuration, bar conversion, error handling
- SSEBroadcaster: subscribe/unsubscribe, publish, event format
- LiveTradingLoop: data_provider param, Alpaca/Yahoo fallback
"""

import asyncio
import os
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock


# ============================================================================
# AlpacaDataService tests
# ============================================================================

class TestAlpacaDataService:
    """Tests for the Alpaca market data client."""

    def _make_service(self, api_key="test-key", secret_key="test-secret"):
        """Create a service with optional env vars."""
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": api_key,
            "ALPACA_SECRET_KEY": secret_key,
        }, clear=False):
            from market_data.alpaca_client import AlpacaDataService
            return AlpacaDataService()

    def test_is_configured_true(self):
        """Returns True when both keys are set."""
        svc = self._make_service("key123", "secret456")
        assert svc.is_configured() is True

    def test_is_configured_false_no_key(self):
        """Returns False when API key is missing."""
        svc = self._make_service("", "secret456")
        assert svc.is_configured() is False

    def test_is_configured_false_no_secret(self):
        """Returns False when secret key is missing."""
        svc = self._make_service("key123", "")
        assert svc.is_configured() is False

    def test_is_configured_false_both_missing(self):
        """Returns False when both keys are missing."""
        svc = self._make_service("", "")
        assert svc.is_configured() is False

    def test_fetch_latest_bar_not_configured_returns_none(self):
        """fetch_latest_bar returns None when not configured."""
        svc = self._make_service("", "")
        result = svc.fetch_latest_bar("SPY")
        assert result is None

    def test_fetch_historical_bars_not_configured_returns_none(self):
        """fetch_historical_bars returns None when not configured."""
        svc = self._make_service("", "")
        result = svc.fetch_historical_bars("SPY")
        assert result is None

    def test_fetch_latest_quote_not_configured_returns_none(self):
        """fetch_latest_quote returns None when not configured."""
        svc = self._make_service("", "")
        result = svc.fetch_latest_quote("SPY")
        assert result is None

    def test_fetch_snapshot_not_configured_returns_none(self):
        """fetch_snapshot returns None when not configured."""
        svc = self._make_service("", "")
        result = svc.fetch_snapshot("SPY")
        assert result is None

    def test_bar_to_replay_format(self):
        """_bar_to_replay_format produces correct replay dict from mock bar."""
        svc = self._make_service()

        # Create a mock bar object mimicking Alpaca Bar
        mock_bar = MagicMock()
        mock_bar.timestamp = datetime(2026, 1, 15, 14, 30, tzinfo=timezone.utc)
        mock_bar.open = 500.0
        mock_bar.high = 510.0
        mock_bar.low = 495.0
        mock_bar.close = 505.0
        mock_bar.volume = 1000000

        result = svc._bar_to_replay_format(mock_bar, is_daily=True)

        assert result is not None
        assert result["open"] == 500.0
        assert result["high"] == 510.0
        assert result["low"] == 495.0
        assert result["close"] == 505.0
        assert result["volume"] == 1000000
        assert "time" in result
        assert "open_time" in result
        assert "close_time" in result
        # Timestamps should be ints (Unix)
        assert isinstance(result["time"], int)
        assert isinstance(result["open_time"], int)
        assert isinstance(result["close_time"], int)

    def test_bar_to_replay_format_intraday(self):
        """_bar_to_replay_format handles intraday bars correctly."""
        svc = self._make_service()

        mock_bar = MagicMock()
        mock_bar.timestamp = datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc)
        mock_bar.open = 500.0
        mock_bar.high = 501.0
        mock_bar.low = 499.0
        mock_bar.close = 500.5
        mock_bar.volume = 50000

        result = svc._bar_to_replay_format(mock_bar, is_daily=False)

        assert result is not None
        # For intraday, close_time = open_time + 1 minute
        assert result["close_time"] - result["open_time"] == 60

    def test_candle_format_matches_yahoo_format(self):
        """Alpaca candle output has same keys as Yahoo replay format."""
        svc = self._make_service()

        mock_bar = MagicMock()
        mock_bar.timestamp = datetime(2026, 1, 15, 14, 30, tzinfo=timezone.utc)
        mock_bar.open = 500.0
        mock_bar.high = 510.0
        mock_bar.low = 495.0
        mock_bar.close = 505.0
        mock_bar.volume = 1000000

        result = svc._bar_to_replay_format(mock_bar, is_daily=True)
        expected_keys = {"time", "open_time", "close_time", "open", "high", "low", "close", "volume"}
        assert set(result.keys()) == expected_keys

    def test_bar_to_replay_format_handles_error(self):
        """_bar_to_replay_format returns None on error."""
        svc = self._make_service()

        # A bar that will cause an error (missing timestamp)
        mock_bar = MagicMock()
        mock_bar.timestamp = None  # Will cause AttributeError

        result = svc._bar_to_replay_format(mock_bar, is_daily=True)
        assert result is None


# ============================================================================
# SSEBroadcaster tests
# ============================================================================

class TestSSEBroadcaster:
    """Tests for the SSE event broadcaster."""

    def _make_broadcaster(self):
        from sse_broadcaster import SSEBroadcaster
        return SSEBroadcaster()

    def test_subscribe_unsubscribe(self):
        """subscribe adds a queue, unsubscribe removes it."""
        b = self._make_broadcaster()
        assert b.subscriber_count == 0

        q = b.subscribe()
        assert b.subscriber_count == 1
        assert isinstance(q, asyncio.Queue)

        b.unsubscribe(q)
        assert b.subscriber_count == 0

    def test_unsubscribe_nonexistent(self):
        """unsubscribe with unknown queue doesn't error."""
        b = self._make_broadcaster()
        q = asyncio.Queue()
        b.unsubscribe(q)  # Should not raise
        assert b.subscriber_count == 0

    def test_publish_all_subscribers(self):
        """publish delivers event to all subscriber queues."""
        async def _test():
            b = self._make_broadcaster()
            q1 = b.subscribe()
            q2 = b.subscribe()

            await b.publish_async("test_event", {"value": 42})

            e1 = q1.get_nowait()
            e2 = q2.get_nowait()

            assert e1["type"] == "test_event"
            assert e1["payload"]["value"] == 42
            assert "timestamp" in e1

            assert e2["type"] == "test_event"
            assert e2["payload"]["value"] == 42

        asyncio.run(_test())

    def test_publish_no_subscribers(self):
        """publish with no subscribers doesn't error."""
        b = self._make_broadcaster()
        # Should not raise
        b.publish("test_event", {"value": 1})

    def test_event_has_type_and_timestamp(self):
        """Published events contain type, timestamp, and payload."""
        async def _test():
            b = self._make_broadcaster()
            q = b.subscribe()

            await b.publish_async("equity_update", {"equity": 100500.0})

            event = q.get_nowait()
            assert "type" in event
            assert "timestamp" in event
            assert "payload" in event
            assert event["type"] == "equity_update"
            assert event["payload"]["equity"] == 100500.0

            # Timestamp should be valid ISO format
            ts = datetime.fromisoformat(event["timestamp"])
            assert ts.tzinfo is not None

        asyncio.run(_test())

    def test_multiple_event_types(self):
        """Different event types are published correctly."""
        async def _test():
            b = self._make_broadcaster()
            q = b.subscribe()

            await b.publish_async("equity_update", {"equity": 100000})
            await b.publish_async("trade_executed", {"action": "BUY", "symbol": "SPY"})
            await b.publish_async("alert_fired", {"severity": "warning"})

            e1 = q.get_nowait()
            e2 = q.get_nowait()
            e3 = q.get_nowait()

            assert e1["type"] == "equity_update"
            assert e2["type"] == "trade_executed"
            assert e3["type"] == "alert_fired"

        asyncio.run(_test())

    def test_full_queue_drops_oldest(self):
        """When queue is full, oldest event is dropped to make room."""
        async def _test():
            b = self._make_broadcaster()
            q = b.subscribe()

            # Fill the queue (maxsize=100)
            for i in range(100):
                await b.publish_async("fill", {"i": i})

            # Queue should be full
            assert q.qsize() == 100

            # Publish one more — should drop oldest and add new
            await b.publish_async("new_event", {"i": 999})

            assert q.qsize() == 100

            # The first event should now be i=1 (i=0 was dropped)
            first = q.get_nowait()
            assert first["payload"]["i"] == 1

        asyncio.run(_test())

    def test_format_sse(self):
        """format_sse produces valid SSE text format."""
        b = self._make_broadcaster()
        event = {
            "type": "equity_update",
            "timestamp": "2026-01-15T14:30:00+00:00",
            "payload": {"equity": 100500.0},
        }

        result = b.format_sse(event)

        assert result.startswith("event: equity_update\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_heartbeat_event(self):
        """heartbeat_event returns a valid heartbeat dict."""
        b = self._make_broadcaster()
        hb = b.heartbeat_event()

        assert hb["type"] == "heartbeat"
        assert "timestamp" in hb
        assert hb["payload"] == {}


# ============================================================================
# LiveTradingLoop data provider tests
# ============================================================================

class TestLiveTradingLoopProvider:
    """Tests for the data_provider parameter in LiveTradingLoop."""

    def test_default_data_provider_auto(self):
        """Default data_provider is 'auto'."""
        async def _test():
            from live_trading_loop import LiveTradingLoop
            loop = LiveTradingLoop(symbol="SPY", strategy_name="ema_trend_v1")
            assert loop.data_provider == "auto"
        asyncio.run(_test())

    def test_data_provider_yahoo(self):
        """data_provider='yahoo' is stored correctly."""
        async def _test():
            from live_trading_loop import LiveTradingLoop
            loop = LiveTradingLoop(
                symbol="SPY",
                strategy_name="ema_trend_v1",
                data_provider="yahoo",
            )
            assert loop.data_provider == "yahoo"
        asyncio.run(_test())

    def test_data_provider_alpaca(self):
        """data_provider='alpaca' is stored correctly."""
        async def _test():
            from live_trading_loop import LiveTradingLoop
            loop = LiveTradingLoop(
                symbol="SPY",
                strategy_name="ema_trend_v1",
                data_provider="alpaca",
            )
            assert loop.data_provider == "alpaca"
        asyncio.run(_test())

    def test_yahoo_fallback_when_no_alpaca(self):
        """With auto provider and no Alpaca keys, falls back to Yahoo."""
        async def _test():
            from live_trading_loop import LiveTradingLoop

            loop = LiveTradingLoop(
                symbol="SPY",
                strategy_name="ema_trend_v1",
                data_provider="auto",
            )

            # Mock Alpaca as not configured
            with patch("live_trading_loop.LiveTradingLoop._fetch_daily_alpaca", new_callable=AsyncMock) as mock_alpaca, \
                 patch("live_trading_loop.LiveTradingLoop._fetch_daily_yahoo", new_callable=AsyncMock) as mock_yahoo:

                mock_alpaca.return_value = None  # Alpaca fails
                mock_yahoo.return_value = {
                    "time": 1705363200,
                    "open_time": 1705329000,
                    "close_time": 1705363200,
                    "open": 500.0, "high": 510.0, "low": 495.0, "close": 505.0,
                    "volume": 1000000,
                }

                result = await loop._fetch_latest_daily_candle()

                # Alpaca was tried first
                mock_alpaca.assert_called_once()
                # Then Yahoo fallback
                mock_yahoo.assert_called_once()
                # Got a result from Yahoo
                assert result is not None
                assert result["close"] == 505.0

        asyncio.run(_test())

    def test_alpaca_only_no_fallback(self):
        """With alpaca-only provider, no fallback to Yahoo on failure."""
        async def _test():
            from live_trading_loop import LiveTradingLoop

            loop = LiveTradingLoop(
                symbol="SPY",
                strategy_name="ema_trend_v1",
                data_provider="alpaca",
            )

            with patch("live_trading_loop.LiveTradingLoop._fetch_daily_alpaca", new_callable=AsyncMock) as mock_alpaca, \
                 patch("live_trading_loop.LiveTradingLoop._fetch_daily_yahoo", new_callable=AsyncMock) as mock_yahoo:

                mock_alpaca.return_value = None

                result = await loop._fetch_latest_daily_candle()

                mock_alpaca.assert_called_once()
                mock_yahoo.assert_not_called()  # No fallback
                assert result is None

        asyncio.run(_test())

    def test_yahoo_only_skips_alpaca(self):
        """With yahoo-only provider, Alpaca is never tried."""
        async def _test():
            from live_trading_loop import LiveTradingLoop

            loop = LiveTradingLoop(
                symbol="SPY",
                strategy_name="ema_trend_v1",
                data_provider="yahoo",
            )

            with patch("live_trading_loop.LiveTradingLoop._fetch_daily_alpaca", new_callable=AsyncMock) as mock_alpaca, \
                 patch("live_trading_loop.LiveTradingLoop._fetch_daily_yahoo", new_callable=AsyncMock) as mock_yahoo:

                mock_yahoo.return_value = {
                    "time": 1705363200,
                    "open_time": 1705329000,
                    "close_time": 1705363200,
                    "open": 500.0, "high": 510.0, "low": 495.0, "close": 505.0,
                    "volume": 1000000,
                }

                result = await loop._fetch_latest_daily_candle()

                mock_alpaca.assert_not_called()  # Alpaca skipped
                mock_yahoo.assert_called_once()
                assert result is not None

        asyncio.run(_test())


# ============================================================================
# SSE emission from monitoring tests
# ============================================================================

class TestMonitoringSSEEmission:
    """Test that monitoring.add_alert emits SSE events."""

    def test_add_alert_emits_sse(self):
        """add_alert should publish an alert_fired SSE event."""
        from monitoring import SystemMonitor, AlertSeverity

        mon = SystemMonitor()

        with patch("sse_broadcaster.broadcaster") as mock_broadcaster:
            mon.add_alert(
                severity=AlertSeverity.WARNING,
                category="risk",
                message="Daily loss limit approaching",
                details={"pct_used": 85},
            )

            # broadcaster.publish is called at least once for alert_fired
            # (Sprint 5 notification integration may add a second call for notification_fired)
            assert mock_broadcaster.publish.call_count >= 1
            first_call = mock_broadcaster.publish.call_args_list[0]
            assert first_call[0][0] == "alert_fired"
            payload = first_call[0][1]
            assert payload["severity"] == "warning"
            assert payload["category"] == "risk"
            assert payload["message"] == "Daily loss limit approaching"
