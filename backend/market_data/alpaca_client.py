"""
Alpaca Markets data provider.

Provides historical bars, latest quotes, and snapshots via the Alpaca REST API.
Uses alpaca-py SDK. Configured via environment variables:
  ALPACA_API_KEY, ALPACA_SECRET_KEY

Design decisions:
- Data only — execution stays in PaperBroker (preserves all existing tests)
- All methods return None on failure (no exceptions) — callers handle fallback
- Output format matches convert_to_replay_format(): {time, open_time, close_time, open, high, low, close, volume}
- Free tier: IEX feed, 200 req/min
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class AlpacaDataService:
    """
    Alpaca Markets data service.

    Singleton usage: import alpaca_service from this module.
    All methods return None on error (graceful fallback).
    """

    def __init__(self):
        self._client = None
        self._api_key = os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

    def is_configured(self) -> bool:
        """Return True if Alpaca API keys are present in environment."""
        return bool(self._api_key and self._secret_key)

    def _get_client(self):
        """Lazy-initialize the Alpaca data client."""
        if self._client is not None:
            return self._client

        if not self.is_configured():
            return None

        try:
            from alpaca.data import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
            logger.info("Alpaca data client initialized")
            return self._client
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            return None

    def fetch_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 500,
    ) -> Optional[List[Dict]]:
        """
        Fetch historical bars from Alpaca.

        Args:
            symbol: Trading symbol (e.g., "SPY")
            timeframe: "1Min", "5Min", "1Hour", "1Day"
            start: Start date ISO string (e.g., "2025-01-01")
            end: End date ISO string
            limit: Max number of bars

        Returns:
            List of candle dicts in replay format, or None on failure.
        """
        client = self._get_client()
        if client is None:
            return None

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrame.Minute.unit) if hasattr(TimeFrame, "Minute") else TimeFrame.Minute,
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day,
            }
            tf = tf_map.get(timeframe, TimeFrame.Day)

            # Build request
            request_params = {
                "symbol_or_symbols": symbol,
                "timeframe": tf,
                "limit": limit,
            }

            if start:
                request_params["start"] = datetime.fromisoformat(start).replace(
                    tzinfo=timezone.utc
                )
            if end:
                request_params["end"] = datetime.fromisoformat(end).replace(
                    tzinfo=timezone.utc
                )

            request = StockBarsRequest(**request_params)
            bars = client.get_stock_bars(request)

            # bars[symbol] returns a list of Bar objects
            bar_list = bars[symbol] if symbol in bars else []
            if not bar_list:
                logger.warning(f"No bars returned from Alpaca for {symbol}")
                return None

            is_daily = timeframe == "1Day"
            candles = []
            for bar in bar_list:
                candle = self._bar_to_replay_format(bar, is_daily)
                if candle:
                    candles.append(candle)

            logger.info(
                f"Alpaca: fetched {len(candles)} {timeframe} bars for {symbol}"
            )
            return candles if candles else None

        except Exception as e:
            logger.error(f"Alpaca fetch_historical_bars failed: {e}")
            return None

    def fetch_latest_bar(self, symbol: str, timeframe: str = "1Day") -> Optional[Dict]:
        """
        Fetch the latest bar for a symbol.

        Returns:
            Single candle dict in replay format, or None.
        """
        client = self._get_client()
        if client is None:
            return None

        try:
            from alpaca.data.requests import StockLatestBarRequest

            request = StockLatestBarRequest(symbol_or_symbols=symbol)
            bars = client.get_stock_latest_bar(request)

            bar = bars.get(symbol)
            if bar is None:
                logger.warning(f"No latest bar from Alpaca for {symbol}")
                return None

            is_daily = timeframe == "1Day"
            return self._bar_to_replay_format(bar, is_daily)

        except Exception as e:
            logger.error(f"Alpaca fetch_latest_bar failed: {e}")
            return None

    def fetch_latest_quote(self, symbol: str) -> Optional[Dict]:
        """
        Fetch the latest quote (bid/ask/last) for a symbol.

        Returns:
            Dict with {bid, ask, last, timestamp}, or None.
        """
        client = self._get_client()
        if client is None:
            return None

        try:
            from alpaca.data.requests import StockLatestQuoteRequest

            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = client.get_stock_latest_quote(request)

            quote = quotes.get(symbol)
            if quote is None:
                return None

            return {
                "bid": float(quote.bid_price),
                "ask": float(quote.ask_price),
                "last": float(quote.ask_price),  # Alpaca doesn't always have 'last'
                "timestamp": quote.timestamp.isoformat()
                if quote.timestamp
                else None,
            }

        except Exception as e:
            logger.error(f"Alpaca fetch_latest_quote failed: {e}")
            return None

    def fetch_snapshot(self, symbol: str) -> Optional[Dict]:
        """
        Fetch a snapshot for a symbol (daily bar + latest quote + minute bar).

        Returns:
            Dict with {daily_bar, latest_quote, minute_bar}, or None.
        """
        client = self._get_client()
        if client is None:
            return None

        try:
            from alpaca.data.requests import StockSnapshotRequest

            request = StockSnapshotRequest(symbol_or_symbols=symbol)
            snapshots = client.get_stock_snapshot(request)

            snap = snapshots.get(symbol)
            if snap is None:
                return None

            result = {}

            if snap.daily_bar:
                result["daily_bar"] = self._bar_to_replay_format(
                    snap.daily_bar, is_daily=True
                )

            if snap.minute_bar:
                result["minute_bar"] = self._bar_to_replay_format(
                    snap.minute_bar, is_daily=False
                )

            if snap.latest_quote:
                result["latest_quote"] = {
                    "bid": float(snap.latest_quote.bid_price),
                    "ask": float(snap.latest_quote.ask_price),
                    "timestamp": snap.latest_quote.timestamp.isoformat()
                    if snap.latest_quote.timestamp
                    else None,
                }

            return result if result else None

        except Exception as e:
            logger.error(f"Alpaca fetch_snapshot failed: {e}")
            return None

    def _bar_to_replay_format(self, bar, is_daily: bool = True) -> Optional[Dict]:
        """
        Convert an Alpaca Bar object to replay format.

        Replay format:
            {time: int, open_time: int, close_time: int,
             open: float, high: float, low: float, close: float, volume: int}
        """
        try:
            from utils import calculate_candle_timestamps, ensure_utc_datetime

            # Alpaca bar.timestamp is the bar open time (timezone-aware)
            bar_ts = bar.timestamp
            if bar_ts.tzinfo is None:
                bar_ts = bar_ts.replace(tzinfo=timezone.utc)
            else:
                bar_ts = bar_ts.astimezone(timezone.utc)

            bar_ts = ensure_utc_datetime(bar_ts, "Alpaca bar timestamp")

            if is_daily:
                open_time, close_time = calculate_candle_timestamps(
                    bar_ts, is_daily=True
                )
            else:
                # For intraday, bar.timestamp is open time
                open_time = bar_ts
                close_time = bar_ts + timedelta(minutes=1)

            open_time_unix = int(open_time.timestamp())
            close_time_unix = int(close_time.timestamp())

            return {
                "time": close_time_unix,
                "open_time": open_time_unix,
                "close_time": close_time_unix,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            }

        except Exception as e:
            logger.error(f"Failed to convert Alpaca bar to replay format: {e}")
            return None


# Module-level singleton
alpaca_service = AlpacaDataService()
