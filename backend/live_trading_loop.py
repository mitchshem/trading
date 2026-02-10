"""
Live paper trading loop.

Autonomous trading loop that:
1. Fetches latest market data at configurable intervals
2. Detects anomalies before evaluation
3. Evaluates the active strategy
4. Routes signals through PaperBroker
5. Logs every decision to the database
6. Enforces trading hours (9:30 AM – 4:00 PM ET)

Designed for daily or intraday operation. For daily mode, evaluates once
per trading day after market close.

Usage:
    loop = LiveTradingLoop(
        symbol="SPY",
        strategy_name="ema_trend_v1",
    )
    await loop.start()    # Begin autonomous trading
    loop.stop()           # Graceful shutdown
    loop.get_status()     # Current state
"""

import asyncio
import json
import logging
from typing import Optional, Dict, List
from datetime import datetime, timezone, timedelta, time as dt_time
from dataclasses import dataclass, field
from enum import Enum

from paper_broker import PaperBroker
from strategy import PositionState
from anomaly_detector import detect_anomalies, AnomalyConfig, AnomalyResult
from decision_log import log_decision
from database import SessionLocal
from sse_broadcaster import broadcaster
import strategy_registry as sr

logger = logging.getLogger(__name__)


class LoopStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"        # Paused due to anomaly or outside market hours
    STOPPED = "stopped"
    ERROR = "error"


# US Eastern Time offset from UTC (handles DST manually)
# ET = UTC-5 (EST) or UTC-4 (EDT)
# For simplicity, we use pytz-free approach with explicit offsets
MARKET_OPEN_ET = dt_time(9, 30)   # 9:30 AM ET
MARKET_CLOSE_ET = dt_time(16, 0)   # 4:00 PM ET
NO_ENTRY_CUTOFF_ET = dt_time(15, 45)  # No new entries after 3:45 PM ET


@dataclass
class LoopState:
    """Internal state of the live trading loop."""
    status: LoopStatus = LoopStatus.IDLE
    symbol: str = "SPY"
    strategy_name: str = "ema_trend_v1"
    strategy_params: Dict = field(default_factory=dict)
    interval_seconds: float = 60.0  # Check interval (seconds)
    is_daily: bool = True
    last_evaluation_time: Optional[datetime] = None
    last_candle_time: Optional[datetime] = None
    evaluations_count: int = 0
    trades_count: int = 0
    anomalies_detected: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None


class LiveTradingLoop:
    """
    Autonomous live paper trading loop.

    For daily mode: evaluates once per day after market close using
    the latest daily candle from Yahoo Finance.

    For intraday mode: evaluates on each new candle at the configured interval.
    """

    def __init__(
        self,
        symbol: str = "SPY",
        strategy_name: str = "ema_trend_v1",
        strategy_params: Optional[Dict] = None,
        initial_equity: float = 100000.0,
        commission_per_share: float = 0.005,
        commission_per_trade: float = 1.0,
        slippage: float = 0.0002,
        interval_seconds: float = 60.0,
        is_daily: bool = True,
        anomaly_config: Optional[AnomalyConfig] = None,
        data_provider: str = "auto",
    ):
        """
        Initialize the live trading loop.

        Args:
            symbol: Trading symbol
            strategy_name: Name from strategy registry
            strategy_params: Override default params (or None for defaults)
            initial_equity: Starting equity for paper broker
            commission_per_share: Per-share commission
            commission_per_trade: Per-trade commission
            slippage: Slippage as decimal (0.0002 = 0.02%)
            interval_seconds: Seconds between evaluation checks
            is_daily: True for daily candles, False for intraday
            anomaly_config: Custom anomaly detection thresholds
            data_provider: "auto" (Alpaca→Yahoo fallback), "alpaca", or "yahoo"
        """
        # Validate strategy exists
        self._strategy_config = sr.get_strategy(strategy_name)

        self.broker = PaperBroker(
            initial_equity=initial_equity,
            commission_per_share=commission_per_share,
            commission_per_trade=commission_per_trade,
            slippage=slippage,
        )
        self.position_state = PositionState()
        self.anomaly_config = anomaly_config or AnomalyConfig()

        self.data_provider = data_provider  # "auto", "alpaca", or "yahoo"

        self.state = LoopState(
            symbol=symbol,
            strategy_name=strategy_name,
            strategy_params=strategy_params or self._strategy_config.default_params,
            interval_seconds=interval_seconds,
            is_daily=is_daily,
        )

        # Candle history (maintained across evaluations)
        self.candle_history: List[Dict] = []

        # Internal task handle
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        """Start the live trading loop."""
        if self.state.status == LoopStatus.RUNNING:
            raise RuntimeError("Loop is already running")

        self.state.status = LoopStatus.RUNNING
        self.state.started_at = datetime.now(timezone.utc)
        self.state.stopped_at = None
        self.state.error_message = None
        self._stop_event.clear()

        logger.info(
            f"Live trading loop started: {self.state.symbol} / {self.state.strategy_name} "
            f"(daily={self.state.is_daily}, interval={self.state.interval_seconds}s, "
            f"provider={self.data_provider})"
        )

        broadcaster.publish("loop_status", {
            "status": "running",
            "symbol": self.state.symbol,
            "strategy_name": self.state.strategy_name,
            "data_provider": self.data_provider,
        })

        self._task = asyncio.create_task(self._run_loop())
        return self.get_status()

    def stop(self):
        """Signal the loop to stop gracefully."""
        self._stop_event.set()
        self.state.status = LoopStatus.STOPPED
        self.state.stopped_at = datetime.now(timezone.utc)
        logger.info("Live trading loop stop requested")

        broadcaster.publish("loop_status", {
            "status": "stopped",
            "symbol": self.state.symbol,
            "strategy_name": self.state.strategy_name,
        })

    def get_status(self) -> Dict:
        """Get current loop status."""
        return {
            "status": self.state.status.value,
            "symbol": self.state.symbol,
            "strategy_name": self.state.strategy_name,
            "strategy_params": self.state.strategy_params,
            "is_daily": self.state.is_daily,
            "interval_seconds": self.state.interval_seconds,
            "evaluations_count": self.state.evaluations_count,
            "trades_count": self.state.trades_count,
            "anomalies_detected": self.state.anomalies_detected,
            "last_evaluation_time": (
                self.state.last_evaluation_time.isoformat()
                if self.state.last_evaluation_time
                else None
            ),
            "broker": {
                "equity": self.broker.equity,
                "cash": self.broker.cash,
                "positions": {
                    sym: {
                        "shares": pos.shares,
                        "entry_price": pos.entry_price,
                        "stop_price": pos.stop_price,
                    }
                    for sym, pos in self.broker.positions.items()
                },
                "pending_orders": len(self.broker.pending_orders),
                "trade_blocked": self.broker.trade_blocked,
            },
            "started_at": (
                self.state.started_at.isoformat() if self.state.started_at else None
            ),
            "stopped_at": (
                self.state.stopped_at.isoformat() if self.state.stopped_at else None
            ),
            "error_message": self.state.error_message,
        }

    async def _run_loop(self):
        """Main loop: check for new candles and evaluate strategy."""
        try:
            while not self._stop_event.is_set():
                try:
                    now = datetime.now(timezone.utc)

                    if self.state.is_daily:
                        await self._daily_evaluation_cycle(now)
                    else:
                        await self._intraday_evaluation_cycle(now)

                except Exception as e:
                    logger.error(f"Error in trading loop iteration: {e}", exc_info=True)
                    self.state.error_message = str(e)
                    # Continue running — don't crash on transient errors

                # Wait for next check interval (or stop signal)
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.state.interval_seconds,
                    )
                    # If wait_for returns, stop was requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout — continue loop
                    pass

        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}", exc_info=True)
            self.state.status = LoopStatus.ERROR
            self.state.error_message = str(e)
        finally:
            if self.state.status != LoopStatus.ERROR:
                self.state.status = LoopStatus.STOPPED
            self.state.stopped_at = datetime.now(timezone.utc)

    async def _daily_evaluation_cycle(self, now: datetime):
        """
        Daily mode: evaluate once per trading day after market close.
        Fetches the latest daily candle and evaluates the strategy.
        """
        # Check if we already evaluated today
        if self.state.last_evaluation_time:
            last_date = self.state.last_evaluation_time.date()
            if last_date == now.date():
                return  # Already evaluated today

        # Check if it's after market close (~20:15 UTC / 4:15 PM ET)
        # We wait 15 minutes after close for data to settle
        # Approximate: use UTC hour > 20 (covers both EST and EDT)
        if now.hour < 20:
            self.state.status = LoopStatus.PAUSED
            return

        self.state.status = LoopStatus.RUNNING

        # Fetch latest candle
        candle = await self._fetch_latest_daily_candle()
        if candle is None:
            return

        # Check if this is a new candle (not already processed)
        if self.state.last_candle_time:
            candle_time = candle.get("close_time") or candle.get("time")
            if isinstance(candle_time, (int, float)):
                candle_dt = datetime.fromtimestamp(candle_time, tz=timezone.utc)
            elif isinstance(candle_time, datetime):
                candle_dt = candle_time
            else:
                candle_dt = None

            if candle_dt and candle_dt <= self.state.last_candle_time:
                return  # Already processed this candle

        await self._evaluate_and_execute(candle, now)

    async def _intraday_evaluation_cycle(self, now: datetime):
        """
        Intraday mode: evaluate on each new candle.
        """
        # Check if within market hours (approximate using UTC)
        # Market open: ~13:30 UTC (9:30 ET), close: ~20:00 UTC (4:00 ET)
        if now.hour < 13 or (now.hour == 13 and now.minute < 30) or now.hour >= 20:
            self.state.status = LoopStatus.PAUSED
            return

        self.state.status = LoopStatus.RUNNING

        candle = await self._fetch_latest_intraday_candle()
        if candle is None:
            return

        await self._evaluate_and_execute(candle, now)

    async def _fetch_latest_daily_candle(self) -> Optional[Dict]:
        """
        Fetch the latest daily candle.
        Tries Alpaca first (if provider is "auto" or "alpaca"), falls back to Yahoo.
        Returns None if all sources fail.
        """
        candle = None

        if self.data_provider in ("auto", "alpaca"):
            candle = await self._fetch_daily_alpaca()
            if candle is None and self.data_provider == "alpaca":
                logger.error("Alpaca-only mode but fetch failed — no fallback")
                return None

        if candle is None and self.data_provider in ("auto", "yahoo"):
            candle = await self._fetch_daily_yahoo()

        if candle is None:
            return None

        # Update candle history
        self._update_candle_history(candle)
        return self.candle_history[-1]

    async def _fetch_daily_alpaca(self) -> Optional[Dict]:
        """Fetch latest daily candle from Alpaca. Returns None on failure."""
        try:
            from market_data.alpaca_client import alpaca_service

            if not alpaca_service.is_configured():
                logger.debug("Alpaca not configured, skipping")
                return None

            bar = alpaca_service.fetch_latest_bar(self.state.symbol)
            if bar is not None:
                logger.info(f"Fetched daily candle from Alpaca for {self.state.symbol}")
                return bar

            logger.warning(f"Alpaca returned no data for {self.state.symbol}")
            return None

        except Exception as e:
            logger.error(f"Alpaca daily fetch failed: {e}")
            return None

    async def _fetch_daily_yahoo(self) -> Optional[Dict]:
        """
        Fetch the latest daily candle from Yahoo Finance.
        Returns None if fetch fails or no new data available.
        """
        try:
            from market_data.yahoo import fetch_yahoo_candles, convert_to_replay_format

            # Fetch last 5 days to account for weekends/holidays
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

            candles = fetch_yahoo_candles(
                symbol=self.state.symbol,
                start_date=start_date,
                end_date=end_date,
            )

            if not candles:
                logger.warning(f"Yahoo returned no candles for {self.state.symbol}")
                return None

            # Convert to replay format (Unix timestamps)
            replay_candles = convert_to_replay_format(candles)

            if replay_candles:
                # If we had no history, seed it from Yahoo's multi-day fetch
                if not self.candle_history:
                    self.candle_history = replay_candles[:-1]  # All but last (returned separately)
                return replay_candles[-1]

            return None

        except Exception as e:
            logger.error(f"Yahoo daily fetch failed: {e}")
            return None

    def _update_candle_history(self, candle: Dict):
        """Append a candle to history if it's new, keep last 500."""
        if not self.candle_history:
            self.candle_history = [candle]
        else:
            last_time = self.candle_history[-1].get("close_time", 0)
            candle_time = candle.get("close_time", 0)
            if candle_time > last_time:
                self.candle_history.append(candle)
            elif candle_time == last_time:
                # Update in place (same candle, possibly newer data)
                self.candle_history[-1] = candle
            # else: older candle, skip

        if len(self.candle_history) > 500:
            self.candle_history = self.candle_history[-500:]

    async def _fetch_latest_intraday_candle(self) -> Optional[Dict]:
        """
        Fetch the latest intraday (minute) candle.
        Uses Alpaca if configured, otherwise returns None.
        """
        if self.data_provider == "yahoo":
            logger.debug("Yahoo doesn't support intraday — skipping")
            return None

        try:
            from market_data.alpaca_client import alpaca_service

            if not alpaca_service.is_configured():
                logger.debug("Alpaca not configured — no intraday data available")
                return None

            bar = alpaca_service.fetch_latest_bar(self.state.symbol, timeframe="1Min")
            if bar is not None:
                self._update_candle_history(bar)
                return bar

            return None

        except Exception as e:
            logger.error(f"Intraday candle fetch failed: {e}")
            return None

    async def _evaluate_and_execute(self, candle: Dict, now: datetime):
        """
        Core evaluation pipeline:
        1. Detect anomalies
        2. Evaluate strategy
        3. Process pending orders
        4. Route signal through broker
        5. Update equity and risk controls
        6. Log decision
        """
        db = SessionLocal()
        try:
            symbol = self.state.symbol

            # 1. ANOMALY DETECTION
            history_for_anomaly = self.candle_history[:-1] if len(self.candle_history) > 1 else []
            anomaly = detect_anomalies(
                current_candle=candle,
                history=history_for_anomaly,
                config=self.anomaly_config,
                current_time=now,
                is_daily=self.state.is_daily,
            )

            if anomaly.severity == "critical":
                self.state.anomalies_detected += 1
                logger.warning(f"CRITICAL anomaly detected: {anomaly.flags}")
                # Log the blocked decision
                log_decision(
                    db,
                    timestamp=now,
                    symbol=symbol,
                    strategy_name=self.state.strategy_name,
                    candle=candle,
                    signal="HOLD",
                    signal_reason="Anomaly detected — trading paused",
                    has_position=self.position_state.has_position,
                    position_entry_price=self.position_state.entry_price,
                    equity=self.broker.equity,
                    cash=self.broker.cash,
                    daily_pnl=self.broker.daily_pnl,
                    trade_blocked=True,
                    pending_orders_count=len(self.broker.pending_orders),
                    anomaly_flags=anomaly.flags,
                    broker_action="BLOCKED",
                    broker_action_reason=f"Critical anomaly: {', '.join(anomaly.flags)}",
                )
                db.commit()
                return

            # 2. PROCESS PENDING ORDERS at candle open
            executed = self.broker.process_pending_orders(
                current_open_price=candle["open"],
                timestamp=candle.get("open_time", candle.get("time")),
            )
            for t in executed:
                if t["action"] == "BUY":
                    self.position_state.has_position = True
                    self.position_state.entry_price = t["entry_price"]
                    self.position_state.entry_time = t["timestamp"]
                    self.state.trades_count += 1
                elif t["action"] == "EXIT":
                    self.position_state.has_position = False
                    self.position_state.entry_price = None
                    self.position_state.entry_time = None
                    self.state.trades_count += 1

            # 3. EVALUATE STRATEGY
            if len(self.candle_history) < 50:
                signal_result = {"signal": "HOLD", "reason": "Insufficient history"}
            else:
                # Use last 500 candles for evaluation
                history_for_eval = self.candle_history[-500:]
                signal_result = self._strategy_config.evaluate_fn(
                    history_for_eval,
                    self.position_state,
                    self.state.strategy_params,
                )

            signal = signal_result.get("signal", "HOLD")
            stop_distance = signal_result.get("stop_distance")
            signal_reason = signal_result.get("reason", "")

            # 4. CHECK STOP-LOSSES
            current_prices = {symbol: candle["close"]}
            close_time = candle.get("close_time", candle.get("time"))
            stop_orders = self.broker.check_stop_losses(current_prices, close_time)
            stop_symbols = {o["symbol"] for o in stop_orders}

            # 5. ROUTE SIGNAL THROUGH BROKER
            broker_action = "NONE"
            broker_action_reason = None

            if signal == "BUY" and stop_distance and symbol not in stop_symbols:
                if self.broker.trade_blocked:
                    broker_action = "BLOCKED"
                    broker_action_reason = "Kill-switch active"
                else:
                    self.broker.execute_buy(
                        symbol=symbol,
                        signal_price=candle["close"],
                        stop_distance=stop_distance,
                        timestamp=close_time,
                    )
                    broker_action = "BUY"
            elif signal == "EXIT" and symbol not in stop_symbols:
                self.broker.execute_exit(
                    symbol=symbol,
                    signal_price=candle["close"],
                    timestamp=close_time,
                    reason=signal_reason,
                )
                broker_action = "EXIT"
            elif symbol in stop_symbols:
                broker_action = "STOP_LOSS"
                broker_action_reason = "Stop-loss triggered"

            # 6. UPDATE EQUITY AND RISK CONTROLS
            self.broker.update_equity(current_prices)
            self.broker.check_and_enforce_risk_controls(current_prices, close_time)

            # 6.5. SSE: Emit equity update and trade events
            broadcaster.publish("equity_update", {
                "equity": round(self.broker.equity, 2),
                "cash": round(self.broker.cash, 2),
                "daily_pnl": round(self.broker.daily_pnl, 2),
                "positions": {
                    sym: {
                        "shares": pos.shares,
                        "entry_price": pos.entry_price,
                        "unrealized_pnl": round(
                            (current_prices.get(sym, pos.entry_price) - pos.entry_price) * pos.shares, 2
                        ),
                    }
                    for sym, pos in self.broker.positions.items()
                },
                "trade_blocked": self.broker.trade_blocked,
            })

            if broker_action in ("BUY", "EXIT", "STOP_LOSS"):
                broadcaster.publish("trade_executed", {
                    "action": broker_action,
                    "symbol": symbol,
                    "signal": signal,
                    "price": candle["close"],
                    "equity": round(self.broker.equity, 2),
                    "reason": broker_action_reason or signal_reason,
                })

                # Dispatch trade notification
                try:
                    from notification_service import notifier, NotificationPayload, NotifCategory
                    sev = "critical" if broker_action == "STOP_LOSS" else "info"
                    notifier.notify(NotificationPayload(
                        category=NotifCategory.TRADE_EXECUTED,
                        severity=sev,
                        title=f"{broker_action}: {symbol}",
                        body=f"{broker_action} {symbol} at ${candle['close']:.2f}",
                        details={
                            "symbol": symbol,
                            "action": broker_action,
                            "price": f"${candle['close']:.2f}",
                            "equity": f"${self.broker.equity:.2f}",
                            "reason": broker_action_reason or signal_reason or "",
                        },
                    ))
                except Exception:
                    pass

            # 7. LOG DECISION
            log_decision(
                db,
                timestamp=now,
                symbol=symbol,
                strategy_name=self.state.strategy_name,
                candle=candle,
                signal=signal,
                signal_reason=signal_reason,
                stop_distance=stop_distance,
                has_position=self.position_state.has_position,
                position_entry_price=self.position_state.entry_price,
                equity=self.broker.equity,
                cash=self.broker.cash,
                daily_pnl=self.broker.daily_pnl,
                trade_blocked=self.broker.trade_blocked,
                pending_orders_count=len(self.broker.pending_orders),
                anomaly_flags=anomaly.flags if anomaly.flags else None,
                broker_action=broker_action,
                broker_action_reason=broker_action_reason,
            )

            db.commit()

            # SSE: Emit decision logged event
            broadcaster.publish("decision_logged", {
                "signal": signal,
                "symbol": symbol,
                "strategy_name": self.state.strategy_name,
                "broker_action": broker_action,
                "reason": signal_reason,
                "equity": round(self.broker.equity, 2),
                "evaluation_number": self.state.evaluations_count + 1,
            })

            # Update state
            self.state.evaluations_count += 1
            self.state.last_evaluation_time = now
            candle_time = candle.get("close_time") or candle.get("time")
            if isinstance(candle_time, (int, float)):
                self.state.last_candle_time = datetime.fromtimestamp(candle_time, tz=timezone.utc)
            elif isinstance(candle_time, datetime):
                self.state.last_candle_time = candle_time

            if anomaly.flags:
                self.state.anomalies_detected += 1

            logger.info(
                f"Evaluation #{self.state.evaluations_count}: "
                f"signal={signal} broker_action={broker_action} "
                f"equity=${self.broker.equity:.2f}"
            )

            # Dispatch daily summary notification (fires every evaluation cycle)
            try:
                from notification_service import notifier as _notifier
                from notification_service import NotificationPayload as _NP
                from notification_service import NotifCategory as _NC
                _notifier.notify(_NP(
                    category=_NC.DAILY_SUMMARY,
                    severity="info",
                    title=f"Daily Summary - {self.state.symbol}",
                    body=f"Equity: ${self.broker.equity:.2f} | Daily P&L: ${self.broker.daily_pnl:.2f}",
                    details={
                        "symbol": self.state.symbol,
                        "equity": f"${self.broker.equity:.2f}",
                        "daily_pnl": f"${self.broker.daily_pnl:.2f}",
                        "strategy": self.state.strategy_name,
                        "evaluation": str(self.state.evaluations_count),
                    },
                ))
            except Exception:
                pass

        except Exception as e:
            db.rollback()
            raise
        finally:
            db.close()

    def load_history(self, candles: List[Dict]):
        """
        Pre-load candle history (e.g., from a backtest or CSV).
        Useful for warming up indicators before going live.

        Args:
            candles: List of candle dicts in replay format (Unix timestamps)
        """
        self.candle_history = candles[-500:]  # Keep last 500
        logger.info(f"Loaded {len(self.candle_history)} candles into history")
