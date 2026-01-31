# Momentum Breakout with Volume Confirmation Strategy

## Overview

Strategy 1: Exploits consolidation breakouts with volume confirmation. Targets continuation after consolidation with asymmetric profit-taking.

## Core Components

### 1. `consolidation_detector.py`

Detects consolidation periods:
- `detect_consolidation()`: Identifies low-volatility, sideways movement periods
- Conditions:
  - High-low range over last N candles <= range_pct * average price
  - ATR(14) flat or declining (current <= previous)

### 2. `volume_analyzer.py`

Detects volume spikes:
- `detect_volume_spike()`: Identifies volume spikes relative to rolling average
- Condition: current_volume >= volume_multiplier * rolling_avg_volume
- `get_volume_percentile()`: Calculates volume percentile (for analysis)

### 3. `breakout_strategy.py`

Main strategy module containing:
- `momentum_breakout_v1()`: Main strategy function
- `momentum_breakout_wrapper()`: Integration wrapper for ReplayEngine
- `BreakoutStrategyState`: Extended state tracking (entry ATR, targets hit, entry index)
- `is_close_in_top_quarter()`: Helper to check if close is in top 25% of range

## Entry Logic

1. **Consolidation Detection**:
   - Range over last 20 candles <= 2% of average price
   - ATR(14) flat or declining

2. **Breakout Confirmation**:
   - Close > highest high of consolidation window
   - Volume >= 1.5x 20-period average volume
   - Close in top 25% of candle range

3. **Regime Gate**:
   - Only TREND_UP allowed
   - Price > EMA(200) AND EMA(200) rising

4. **Entry Execution**:
   - Signal generated on candle CLOSE
   - Entry executes on NEXT candle OPEN
   - Stop-loss: 1.0x ATR
   - Risk per trade: 0.25% (configurable)

## Exit Logic (Asymmetric)

**Note**: PaperBroker doesn't support partial exits, so exits are full:
- **Target 1**: +2.0x ATR → Exit fully (captures ~50% of intended profit)
- **Target 2**: +3.5x ATR → Exit fully (if hit before Target 1)
- **Stop-loss**: -1.0x ATR → Exit fully
- **Trailing stop**: After Target 1 hit, stop moves to entry + 1.0x ATR
- **Time stop**: Exit fully if Target 1 not hit within 10 candles

## Integration with ReplayEngine

The strategy requires:
1. EMA(200) calculation (existing `indicators.ema()`)
2. ATR(14) calculation (existing `indicators.atr()`)
3. Regime classification (existing `regime_classifier.classify_regime()`)
4. Position state tracking (existing `strategy.PositionState`)
5. Strategy-specific state (`BreakoutStrategyState`)

### Example Integration (pseudo-code):

```python
# In ReplayEngine.__init__:
from breakout_strategy import momentum_breakout_wrapper, BreakoutStrategyState
self.strategy_state = BreakoutStrategyState()

# In ReplayEngine.process_candle:
ema200_values = ema(closes, 200)
atr14_values = atr(highs, lows, closes, 14)
entry_atr = None  # Store from previous BUY signal

# After processing pending orders (entry executed):
if position_just_entered:
    entry_atr = stored_entry_atr_from_buy_signal

result = momentum_breakout_wrapper(
    candles=current_history,
    ema200_values=ema200_values,
    atr14_values=atr14_values,
    position_state=self.position_state,
    strategy_state=self.strategy_state,
    current_index=self.current_candle_index - 1,
    entry_atr_from_signal=entry_atr
)

# Store entry_atr from BUY signal for next candle:
if result["signal"] == "BUY":
    stored_entry_atr_from_buy_signal = result.get("entry_atr")
```

## Key Features

- **Deterministic**: All logic is rule-based, no ML
- **No lookahead**: Uses only past data
- **Regime-gated**: Only trades in TREND_UP with price > EMA(200) and EMA(200) rising
- **Volume confirmation**: Requires volume spike to validate breakout
- **Asymmetric exits**: Targets favor profit-taking over stop-losses
- **Time-based exit**: Prevents holding losing positions too long
- **Compatible**: Works with existing ReplayEngine, PaperBroker, walk-forward, Monte Carlo, sensitivity harness

## Limitations

1. **Partial Exits**: PaperBroker doesn't support partial exits, so Target 1 exits fully instead of 50%
2. **Consolidation Dependency**: Strategy requires consolidation periods to generate signals
3. **Regime Dependency**: Only trades in TREND_UP regime (may miss opportunities in other regimes)
4. **Time Tracking**: Uses candle count (approximate for daily data, exact for intraday)

## Testing

The strategy can be tested with:
- `ReplayEngine`: Single symbol replay
- `WalkForwardHarness`: Out-of-sample testing
- `MonteCarloResampling`: Robustness testing
- `ParameterSensitivityHarness`: Parameter robustness

## Parameters (Not Optimized)

- Consolidation window: 20 candles (fixed)
- Consolidation range: 2% of average price (fixed)
- Volume multiplier: 1.5x average (fixed)
- Volume period: 20 candles (fixed)
- ATR period: 14 (fixed)
- Target 1: +2.0x ATR
- Target 2: +3.5x ATR
- Stop-loss: -1.0x ATR
- Trailing stop: +1.0x ATR (after Target 1)
- Time stop: 10 candles
- Risk per trade: 0.25%
- Regime: TREND_UP only
