"""
Anomaly detection module for market data.

Detects abnormal market conditions before strategy evaluation to prevent
trading on suspect data. Designed to be called by the live trading loop.

Anomaly Types:
- GAP: Price gap > threshold (default 5% from previous close)
- VOLUME_SPIKE: Volume > N × average (default 10×)
- PRICE_OUTLIER: Price outside 3σ of recent distribution
- STALE_DATA: Candle timestamp older than expected
- ZERO_VOLUME: No trading activity
- INVALID_OHLC: Invalid OHLC relationships (H < L, C > H, etc.)

When anomalies are detected, the live loop should pause trading and alert.
"""

from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
import math


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection thresholds."""
    gap_threshold_pct: float = 5.0       # Max gap from prev close (%)
    volume_spike_multiplier: float = 10.0  # Volume spike detection multiplier
    volume_lookback: int = 20             # Candles for volume average
    price_sigma: float = 3.0             # Standard deviations for outlier
    price_lookback: int = 50             # Candles for price distribution
    staleness_hours: float = 26.0        # Max hours since last candle (allows for weekends buffer)
    staleness_minutes: float = 10.0      # For intraday: max minutes between candles


@dataclass
class AnomalyResult:
    """Result of anomaly detection on a single candle."""
    is_anomalous: bool = False
    flags: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)
    severity: str = "none"  # none, warning, critical

    def to_dict(self) -> Dict:
        return {
            "is_anomalous": self.is_anomalous,
            "flags": self.flags,
            "details": self.details,
            "severity": self.severity,
        }


def detect_anomalies(
    current_candle: Dict,
    history: List[Dict],
    config: Optional[AnomalyConfig] = None,
    current_time: Optional[datetime] = None,
    is_daily: bool = True,
) -> AnomalyResult:
    """
    Detect anomalies in the current candle given recent history.

    Args:
        current_candle: The candle to check (OHLCV dict)
        history: Recent candle history (for comparison). Should NOT include current_candle.
        config: Detection thresholds (uses defaults if None)
        current_time: Current wall clock time (for staleness check). Uses utcnow if None.
        is_daily: True for daily candles, False for intraday.

    Returns:
        AnomalyResult with flags and details
    """
    if config is None:
        config = AnomalyConfig()

    result = AnomalyResult()

    # 1. INVALID OHLC
    _check_invalid_ohlc(current_candle, result)

    # 2. ZERO VOLUME
    _check_zero_volume(current_candle, result)

    # 3. GAP detection (needs at least 1 history candle)
    if history:
        _check_gap(current_candle, history[-1], config, result)

    # 4. VOLUME SPIKE (needs enough history)
    if len(history) >= config.volume_lookback:
        _check_volume_spike(current_candle, history, config, result)

    # 5. PRICE OUTLIER (needs enough history)
    if len(history) >= config.price_lookback:
        _check_price_outlier(current_candle, history, config, result)

    # 6. STALE DATA
    if current_time is not None:
        _check_staleness(current_candle, current_time, config, is_daily, result)

    # Determine overall severity
    if result.flags:
        result.is_anomalous = True
        critical_flags = {"INVALID_OHLC", "STALE_DATA"}
        if any(f in critical_flags for f in result.flags):
            result.severity = "critical"
        else:
            result.severity = "warning"

    return result


def _check_invalid_ohlc(candle: Dict, result: AnomalyResult):
    """Check for invalid OHLC relationships."""
    o, h, l, c = candle.get("open", 0), candle.get("high", 0), candle.get("low", 0), candle.get("close", 0)

    if l > h:
        result.flags.append("INVALID_OHLC")
        result.details["invalid_ohlc"] = f"low ({l}) > high ({h})"

    if o > h or o < l:
        result.flags.append("INVALID_OHLC")
        result.details["invalid_ohlc_open"] = f"open ({o}) outside low-high range ({l}-{h})"

    if c > h or c < l:
        result.flags.append("INVALID_OHLC")
        result.details["invalid_ohlc_close"] = f"close ({c}) outside low-high range ({l}-{h})"

    if any(v <= 0 for v in [o, h, l, c]):
        result.flags.append("INVALID_OHLC")
        result.details["invalid_ohlc_zero"] = f"Non-positive price: O={o} H={h} L={l} C={c}"


def _check_zero_volume(candle: Dict, result: AnomalyResult):
    """Check for zero trading volume."""
    vol = candle.get("volume", 0)
    if vol == 0:
        result.flags.append("ZERO_VOLUME")
        result.details["zero_volume"] = "No trading activity"


def _check_gap(candle: Dict, prev_candle: Dict, config: AnomalyConfig, result: AnomalyResult):
    """Check for price gap from previous close."""
    prev_close = prev_candle.get("close", 0)
    current_open = candle.get("open", 0)

    if prev_close <= 0:
        return

    gap_pct = abs(current_open - prev_close) / prev_close * 100

    if gap_pct > config.gap_threshold_pct:
        direction = "up" if current_open > prev_close else "down"
        result.flags.append("GAP")
        result.details["gap"] = {
            "gap_pct": round(gap_pct, 2),
            "direction": direction,
            "prev_close": prev_close,
            "current_open": current_open,
        }


def _check_volume_spike(candle: Dict, history: List[Dict], config: AnomalyConfig, result: AnomalyResult):
    """Check for abnormal volume relative to recent average."""
    current_vol = candle.get("volume", 0)
    lookback = config.volume_lookback

    recent_volumes = [c.get("volume", 0) for c in history[-lookback:] if c.get("volume", 0) > 0]
    if not recent_volumes:
        return

    avg_volume = sum(recent_volumes) / len(recent_volumes)
    if avg_volume <= 0:
        return

    volume_ratio = current_vol / avg_volume

    if volume_ratio >= config.volume_spike_multiplier:
        result.flags.append("VOLUME_SPIKE")
        result.details["volume_spike"] = {
            "current_volume": current_vol,
            "avg_volume": round(avg_volume, 0),
            "ratio": round(volume_ratio, 2),
        }


def _check_price_outlier(candle: Dict, history: List[Dict], config: AnomalyConfig, result: AnomalyResult):
    """Check if current price is outside N standard deviations of recent prices."""
    current_close = candle.get("close", 0)
    lookback = config.price_lookback

    recent_closes = [c.get("close", 0) for c in history[-lookback:] if c.get("close", 0) > 0]
    if len(recent_closes) < 10:
        return

    mean_price = sum(recent_closes) / len(recent_closes)
    variance = sum((p - mean_price) ** 2 for p in recent_closes) / len(recent_closes)
    std_dev = math.sqrt(variance)

    if std_dev <= 0:
        return

    z_score = (current_close - mean_price) / std_dev

    if abs(z_score) > config.price_sigma:
        result.flags.append("PRICE_OUTLIER")
        result.details["price_outlier"] = {
            "close": current_close,
            "mean": round(mean_price, 2),
            "std_dev": round(std_dev, 2),
            "z_score": round(z_score, 2),
        }


def _check_staleness(candle: Dict, current_time: datetime, config: AnomalyConfig, is_daily: bool, result: AnomalyResult):
    """Check if candle data is stale (too old relative to current time)."""
    candle_time = candle.get("close_time") or candle.get("time")

    if candle_time is None:
        result.flags.append("STALE_DATA")
        result.details["stale_data"] = "No timestamp on candle"
        return

    # Convert Unix timestamp to datetime if needed
    if isinstance(candle_time, (int, float)):
        candle_dt = datetime.fromtimestamp(candle_time, tz=timezone.utc)
    elif isinstance(candle_time, datetime):
        candle_dt = candle_time if candle_time.tzinfo else candle_time.replace(tzinfo=timezone.utc)
    else:
        result.flags.append("STALE_DATA")
        result.details["stale_data"] = f"Unrecognized timestamp type: {type(candle_time)}"
        return

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    age = current_time - candle_dt

    if is_daily:
        max_age = timedelta(hours=config.staleness_hours)
    else:
        max_age = timedelta(minutes=config.staleness_minutes)

    if age > max_age:
        result.flags.append("STALE_DATA")
        result.details["stale_data"] = {
            "candle_time": candle_dt.isoformat(),
            "current_time": current_time.isoformat(),
            "age_hours": round(age.total_seconds() / 3600, 2),
            "max_age_hours": round(max_age.total_seconds() / 3600, 2),
        }
