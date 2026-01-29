"""
Intraday risk manager for portfolio-level risk controls and position sizing.
Provides risk-based position sizing and portfolio risk constraints.
"""

from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone, date
import math
from utils import fmt_pct


class IntradayRiskManager:
    """
    Risk manager for intraday trading with portfolio-level controls.
    
    Features:
    - Risk-based position sizing (configurable risk per trade %)
    - Portfolio equity tracking
    - Daily loss limits
    - Max concurrent positions
    - Stop distance calculation
    """
    
    def __init__(
        self,
        initial_equity: float = 100000.0,
        risk_per_trade_pct: float = 0.25,
        max_daily_loss_pct: float = 1.0,
        max_concurrent_positions: int = 1
    ):
        """
        Initialize risk manager.
        
        Args:
            initial_equity: Starting portfolio equity
            risk_per_trade_pct: Risk per trade as % of equity (default: 0.25%)
            max_daily_loss_pct: Max daily loss as % of equity (default: 1.0%)
            max_concurrent_positions: Max concurrent positions (default: 1)
        """
        self.initial_equity = initial_equity
        self.cash = initial_equity  # Available cash
        self.current_equity = initial_equity  # Total equity (cash + unrealized P&L)
        self.risk_per_trade_pct = risk_per_trade_pct / 100.0  # Convert to decimal
        self.max_daily_loss_pct = max_daily_loss_pct / 100.0
        self.max_concurrent_positions = max_concurrent_positions
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.last_reset_date: Optional[date] = None
        self.trade_blocked = False  # Daily loss limit kill switch
        
        # Position tracking
        self.open_positions: Dict[str, Dict] = {}  # symbol -> position info
        
        # Equity curve tracking
        self.equity_curve: List[Dict] = []  # List of {timestamp, equity} dicts
        
        # Slippage model
        self.slippage = 0.0002  # 0.02% slippage
    
    def reset_daily_stats_if_needed(self, current_date: date):
        """Reset daily stats if it's a new day."""
        if self.last_reset_date is None or current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            self.trade_blocked = False  # Reset kill switch for new day
    
    def calculate_stop_distance(
        self,
        entry_price: float,
        previous_candle: Optional[Dict],
        recent_candles: Optional[List[Dict]] = None
    ) -> float:
        """
        Calculate stop distance using recent price volatility.
        
        Uses the range of the previous candle as a proxy for volatility.
        Falls back to a percentage-based stop if no previous candle.
        
        Args:
            entry_price: Entry price
            previous_candle: Previous candle dict with high, low
            recent_candles: Optional list of recent candles for ATR-like calculation
        
        Returns:
            Stop distance in price units
        """
        if previous_candle:
            # Use previous candle range as volatility proxy
            candle_range = previous_candle["high"] - previous_candle["low"]
            # Use 1.5x the candle range as stop distance
            stop_distance = candle_range * 1.5
            # Ensure minimum stop distance (0.5% of entry price)
            min_stop = entry_price * 0.005
            return max(stop_distance, min_stop)
        else:
            # Fallback: percentage-based stop (1% of entry price)
            return entry_price * 0.01
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_distance: float
    ) -> int:
        """
        Calculate position size based on risk.
        
        Position size is calculated as:
        shares = floor((equity * risk_per_trade_pct) / stop_distance)
        
        Args:
            entry_price: Entry price
            stop_distance: Stop loss distance in price units
        
        Returns:
            Number of shares (floored, minimum 0)
        """
        if stop_distance <= 0:
            return 0
        
        # Risk amount = current equity * risk_per_trade_pct
        risk_amount = self.current_equity * self.risk_per_trade_pct
        
        # Shares = floor(risk_amount / stop_distance)
        shares = math.floor(risk_amount / stop_distance)
        
        return max(0, shares)
    
    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to fill price."""
        if is_buy:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)
    
    def can_enter_position(self, symbol: str, current_date: date) -> Tuple[bool, str]:
        """
        Check if a new position can be entered.
        
        Args:
            symbol: Trading symbol
            current_date: Current date
        
        Returns:
            Tuple of (can_enter: bool, reason: str)
        """
        self.reset_daily_stats_if_needed(current_date)
        
        # Check kill switch
        if self.trade_blocked:
            return False, "Daily loss limit reached - trading blocked"
        
        # Check if position already exists
        if symbol in self.open_positions:
            return False, f"Position already open for {symbol}"
        
        # Check max concurrent positions
        if len(self.open_positions) >= self.max_concurrent_positions:
            return False, f"Max concurrent positions ({self.max_concurrent_positions}) reached"
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl / self.initial_equity) if self.initial_equity > 0 else 0.0
        if daily_loss_pct >= self.max_daily_loss_pct:
            self.trade_blocked = True
            return False, f"Daily loss limit ({fmt_pct(self.max_daily_loss_pct * 100)}) reached"
        
        return True, "OK"
    
    def enter_position(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
        stop_distance: float,
        timestamp: int,
        reason: str
    ) -> Optional[Dict]:
        """
        Enter a new position.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price (before slippage)
            shares: Number of shares
            stop_distance: Stop loss distance
            timestamp: Entry timestamp
            reason: Entry reason
        
        Returns:
            Dict with position info if successful, None if blocked
        """
        # Convert timestamp to date for daily tracking
        if isinstance(timestamp, (int, float)):
            entry_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        else:
            entry_dt = timestamp
        current_date = entry_dt.date()
        
        # Check if position can be entered
        can_enter, reason_msg = self.can_enter_position(symbol, current_date)
        if not can_enter:
            return None
        
        # Apply slippage
        fill_price = self.apply_slippage(entry_price, is_buy=True)
        
        # Calculate cost
        cost = shares * fill_price
        
        # Check if we have enough cash
        if cost > self.cash:
            return None
        
        # Deduct cost from cash
        self.cash -= cost
        
        # Calculate stop price
        stop_price = fill_price - stop_distance
        
        # Store position
        self.open_positions[symbol] = {
            "entry_price": fill_price,
            "entry_time": timestamp,
            "shares": shares,
            "stop_price": stop_price,
            "stop_distance": stop_distance,
            "cost": cost,
            "reason": reason
        }
        
        # Update equity: equity = cash + position_value
        # At entry, position_value = shares * entry_price = cost (approximately, ignoring slippage)
        # So equity stays approximately the same (cash decreased, position value increased)
        # We'll update equity more accurately on exit when we know the realized P&L
        # For now, equity = cash + (shares * entry_price) ≈ initial_equity (if no previous trades)
        position_value = shares * fill_price
        self.current_equity = self.cash + position_value
        
        # Note: Equity curve updated on exit (when P&L is realized)
        # Entry updates equity but doesn't add a curve point (no realized P&L yet)
        
        return {
            "symbol": symbol,
            "entry_price": fill_price,
            "entry_time": timestamp,
            "shares": shares,
            "stop_price": stop_price,
            "cost": cost,
            "reason": reason
        }
    
    def exit_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: int,
        reason: str
    ) -> Optional[Dict]:
        """
        Exit an existing position.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price (before slippage)
            timestamp: Exit timestamp
            reason: Exit reason
        
        Returns:
            Dict with exit info if successful, None if position not found
        """
        if symbol not in self.open_positions:
            return None
        
        position = self.open_positions[symbol]
        
        # Apply slippage
        fill_price = self.apply_slippage(exit_price, is_buy=False)
        
        # Calculate P&L
        entry_price = position["entry_price"]
        shares = position["shares"]
        pnl = (fill_price - entry_price) * shares
        
        # Update daily P&L
        if isinstance(timestamp, (int, float)):
            exit_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        else:
            exit_dt = timestamp
        current_date = exit_dt.date()
        self.reset_daily_stats_if_needed(current_date)
        self.daily_pnl += pnl
        
        # Add proceeds from sale to cash
        proceeds = shares * fill_price
        self.cash += proceeds
        
        # Update current equity: equity = cash (no open positions after exit)
        self.current_equity = self.cash
        
        # Update equity curve (realized P&L)
        self._update_equity_curve(timestamp, self.current_equity)
        
        # Remove position
        del self.open_positions[symbol]
        
        return {
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": fill_price,
            "shares": shares,
            "pnl": pnl,
            "reason": reason
        }
    
    def _update_equity_curve(self, timestamp: int, equity: float):
        """Update equity curve with new data point."""
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": equity
        })
    
    def get_current_equity(self) -> float:
        """Get current portfolio equity."""
        return self.current_equity
    
    def get_daily_pnl(self) -> float:
        """Get today's realized P&L."""
        return self.daily_pnl
    
    def get_daily_pnl_pct(self) -> float:
        """Get today's realized P&L as % of initial equity."""
        if self.initial_equity <= 0:
            return 0.0
        return (self.daily_pnl / self.initial_equity) * 100.0
    
    def get_open_positions_count(self) -> int:
        """Get number of open positions."""
        return len(self.open_positions)
    
    def get_equity_curve(self) -> List[Dict]:
        """Get equity curve data."""
        return self.equity_curve.copy()
    
    def get_max_drawdown(self) -> Dict:
        """
        Calculate maximum drawdown from equity curve.
        
        Returns:
            Dict with max_drawdown_pct, max_drawdown_absolute, peak_equity, trough_equity
        """
        if not self.equity_curve:
            return {
                "max_drawdown_pct": 0.0,
                "max_drawdown_absolute": 0.0,
                "peak_equity": self.initial_equity,
                "trough_equity": self.initial_equity
            }
        
        peak = self.initial_equity
        max_drawdown_absolute = 0.0
        max_drawdown_pct = 0.0
        trough_equity = self.initial_equity
        
        for point in self.equity_curve:
            equity = point["equity"]
            if equity > peak:
                peak = equity
            
            drawdown = peak - equity
            if drawdown > max_drawdown_absolute:
                max_drawdown_absolute = drawdown
                trough_equity = equity
                max_drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0.0
        
        return {
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_absolute": max_drawdown_absolute,
            "peak_equity": peak,
            "trough_equity": trough_equity
        }
