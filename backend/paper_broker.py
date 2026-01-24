"""
Paper trading broker module.
Simulates trade execution with deterministic fills and risk controls.
"""

from typing import Dict, Optional, List
from datetime import datetime, date
import math


class Position:
    """Represents an open position."""
    def __init__(self, symbol: str, entry_time: int, entry_price: float, shares: int, stop_price: float):
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.shares = shares
        self.stop_price = stop_price
    
    def to_dict(self):
        return {
            "symbol": self.symbol,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "shares": self.shares,
            "stop_price": self.stop_price
        }


class PaperBroker:
    """
    Paper trading broker that executes trades based on signals.
    Tracks account equity, positions, and enforces risk controls.
    """
    
    def __init__(self, initial_equity: float = 100000.0):
        # FIX 1: CASH-BASED EQUITY ACCOUNTING
        # Initialize cash and equity to initial_equity
        self.cash = initial_equity
        self.equity = initial_equity
        self.initial_equity = initial_equity  # Keep for reference only
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.daily_pnl = 0.0
        self.last_reset_date = date.today()
        self.trade_blocked = False  # Kill switch flag
        self.slippage = 0.0002  # 0.02% slippage
        
        # Track realized P&L for the day
        self.daily_realized_pnl = 0.0
    
    def reset_daily_stats_if_needed(self):
        """Reset daily stats if it's a new day."""
        today = date.today()
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_realized_pnl = 0.0
            self.last_reset_date = today
            self.trade_blocked = False  # Reset kill switch for new day
    
    def calculate_position_size(self, entry_price: float, stop_distance: float, current_equity: float) -> int:
        """
        Calculate position size based on risk.
        
        Args:
            entry_price: Entry price for the trade
            stop_distance: Stop loss distance (in price units)
            current_equity: Current account equity
        
        Returns:
            Number of shares to trade (floored)
        """
        if stop_distance <= 0:
            return 0
        
        # Risk per trade = 0.5% of current equity
        risk_amount = current_equity * 0.005
        
        # Shares = floor(risk_amount / stop_distance)
        shares = math.floor(risk_amount / stop_distance)
        
        return max(0, shares)  # Ensure non-negative
    
    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Apply slippage to fill price.
        
        Args:
            price: Base price
            is_buy: True for buy, False for sell
        
        Returns:
            Fill price with slippage applied
        """
        if is_buy:
            # Buy: pay slightly more
            return price * (1 + self.slippage)
        else:
            # Sell: receive slightly less
            return price * (1 - self.slippage)
    
    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss limit is breached.
        
        Returns:
            True if daily loss limit is breached (2% of equity)
        """
        self.reset_daily_stats_if_needed()
        max_daily_loss = self.equity * 0.02
        return self.daily_pnl <= -max_daily_loss
    
    def calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """Calculate unrealized P&L for a position."""
        return (current_price - position.entry_price) * position.shares
    
    def update_equity(self, current_prices: Dict[str, float]):
        """
        Update account equity based on current positions and prices.
        
        FIX 1: CASH-BASED EQUITY ACCOUNTING
        Equity = cash + sum(unrealized P&L of open positions)
        
        Args:
            current_prices: Dict of symbol -> current price
        """
        self.reset_daily_stats_if_needed()
        
        # Calculate unrealized P&L from open positions
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                unrealized_pnl += self.calculate_unrealized_pnl(position, current_price)
        
        # FIX 1: Equity = cash + unrealized P&L (not initial_equity + realized + unrealized)
        self.equity = self.cash + unrealized_pnl
        
        # AUDIT FIX: Assert equity is valid (fail fast > silent failure)
        assert self.equity >= 0, f"Equity became negative: {self.equity} (cash: {self.cash}, unrealized: {unrealized_pnl})"
        
        # Update daily P&L (realized + unrealized)
        self.daily_pnl = self.daily_realized_pnl + unrealized_pnl
    
    def execute_buy(self, symbol: str, signal_price: float, stop_distance: float, timestamp: int) -> Optional[Dict]:
        """
        Execute a BUY order.
        
        Args:
            symbol: Trading symbol
            signal_price: Price from signal
            stop_distance: Stop loss distance (2 * ATR)
            timestamp: Current timestamp
        
        Returns:
            Dict with trade info if executed, None if blocked
        """
        self.reset_daily_stats_if_needed()
        
        # Check if trade is blocked (kill switch)
        if self.trade_blocked:
            return None
        
        # Check if position already exists
        if symbol in self.positions:
            return None  # Already have position
        
        # FIX 3: MAX OPEN POSITIONS LIMIT
        # Enforce hard limit of 3 open positions
        if len(self.positions) >= 3:
            return None  # Max positions reached
        
        # Check daily loss limit
        if self.check_daily_loss_limit():
            self.trigger_kill_switch()
            return None
        
        # Calculate position size
        shares = self.calculate_position_size(signal_price, stop_distance, self.equity)
        
        if shares == 0:
            return None  # Position too small
        
        # Apply slippage
        fill_price = self.apply_slippage(signal_price, is_buy=True)
        
        # FIX 1: CASH-BASED EQUITY ACCOUNTING
        # Deduct (shares * fill_price) from cash
        cost = shares * fill_price
        if cost > self.cash:
            return None  # Insufficient cash
        
        self.cash -= cost
        
        # AUDIT FIX: Assert cash never goes negative (fail fast > silent failure)
        assert self.cash >= 0, f"Cash became negative: {self.cash} after BUY of {shares} shares @ {fill_price}"
        
        # Calculate stop price
        stop_price = fill_price - stop_distance
        
        # Create position
        position = Position(
            symbol=symbol,
            entry_time=timestamp,
            entry_price=fill_price,
            shares=shares,
            stop_price=stop_price
        )
        
        self.positions[symbol] = position
        
        return {
            "action": "BUY",
            "symbol": symbol,
            "shares": shares,
            "entry_price": fill_price,
            "stop_price": stop_price,
            "timestamp": timestamp
        }
    
    def execute_exit(self, symbol: str, signal_price: float, timestamp: int, reason: str) -> Optional[Dict]:
        """
        Execute an EXIT order.
        
        Args:
            symbol: Trading symbol
            signal_price: Price from signal
            timestamp: Current timestamp
            reason: Exit reason
        
        Returns:
            Dict with trade info if executed, None if no position
        """
        self.reset_daily_stats_if_needed()
        
        # Check if position exists
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Apply slippage
        fill_price = self.apply_slippage(signal_price, is_buy=False)
        
        # Calculate P&L
        pnl = (fill_price - position.entry_price) * position.shares
        
        # FIX 1: CASH-BASED EQUITY ACCOUNTING
        # Add (shares * fill_price) to cash
        proceeds = position.shares * fill_price
        self.cash += proceeds
        
        # AUDIT FIX: Assert cash is valid (fail fast > silent failure)
        assert self.cash >= 0, f"Cash became negative: {self.cash} after EXIT of {position.shares} shares @ {fill_price}"
        
        # Update realized P&L
        self.daily_realized_pnl += pnl
        
        # Remove position
        del self.positions[symbol]
        
        return {
            "action": "EXIT",
            "symbol": symbol,
            "shares": position.shares,
            "entry_price": position.entry_price,
            "exit_price": fill_price,
            "pnl": pnl,
            "timestamp": timestamp,
            "reason": reason
        }
    
    def trigger_kill_switch(self):
        """
        Trigger kill switch: close all positions and block new trades.
        
        FIX 4: KILL SWITCH BEHAVIOR (FAIL-SAFE)
        Immediately block trades. Positions will be closed by check_and_enforce_risk_controls.
        """
        self.trade_blocked = True
    
    def close_all_positions(self, current_prices: Dict[str, float], timestamp: int) -> List[Dict]:
        """
        Close all open positions (for kill switch).
        
        FIX 4: KILL SWITCH BEHAVIOR (FAIL-SAFE)
        This is called automatically when kill switch is triggered.
        
        Args:
            current_prices: Current prices for all symbols
            timestamp: Current timestamp
        
        Returns:
            List of exit trade dicts
        """
        exits = []
        symbols_to_close = list(self.positions.keys())
        
        for symbol in symbols_to_close:
            if symbol in current_prices:
                exit_trade = self.execute_exit(
                    symbol=symbol,
                    signal_price=current_prices[symbol],
                    timestamp=timestamp,
                    reason="Kill switch triggered"
                )
                if exit_trade:
                    exits.append(exit_trade)
        
        return exits
    
    def check_stop_losses(self, current_prices: Dict[str, float], timestamp: int) -> List[Dict]:
        """
        FIX 2: AUTOMATIC STOP-LOSS ENFORCEMENT
        Check all open positions for stop-loss breaches on candle close.
        If current price <= stop_price, force EXIT with reason "STOP_LOSS".
        
        Args:
            current_prices: Dict of symbol -> current price (candle close)
            timestamp: Current timestamp
        
        Returns:
            List of exit trade dicts for stop-loss exits
        """
        exits = []
        symbols_to_check = list(self.positions.keys())
        
        for symbol in symbols_to_check:
            if symbol in current_prices:
                position = self.positions[symbol]
                current_price = current_prices[symbol]
                
                # FIX 2: If current candle close <= stop_price, force EXIT
                if current_price <= position.stop_price:
                    exit_trade = self.execute_exit(
                        symbol=symbol,
                        signal_price=current_price,
                        timestamp=timestamp,
                        reason="STOP_LOSS"
                    )
                    if exit_trade:
                        exits.append(exit_trade)
        
        return exits
    
    def check_and_enforce_risk_controls(self, current_prices: Dict[str, float], timestamp: int) -> List[Dict]:
        """
        FIX 4: KILL SWITCH BEHAVIOR (FAIL-SAFE)
        Check daily loss limit and automatically close all positions if breached.
        This must be called on every candle close to ensure fail-safe behavior.
        
        Args:
            current_prices: Dict of symbol -> current price
            timestamp: Current timestamp
        
        Returns:
            List of exit trade dicts if kill switch triggered, empty list otherwise
        """
        # Update equity first to get current daily P&L
        self.update_equity(current_prices)
        
        # Check if daily loss limit is breached
        if self.check_daily_loss_limit() and not self.trade_blocked:
            # FIX 4: Immediately trigger kill switch and close all positions
            self.trigger_kill_switch()
            exits = self.close_all_positions(current_prices, timestamp)
            
            # AUDIT FIX: Assert all positions are closed after kill switch (fail fast)
            assert len(self.positions) == 0, f"Kill switch triggered but {len(self.positions)} positions remain open"
            
            return exits
        
        return []
    
    def get_account_summary(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get account summary.
        
        Args:
            current_prices: Current prices for all symbols
        
        Returns:
            Dict with account info
        """
        self.update_equity(current_prices)
        
        open_positions = []
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.entry_price)
            unrealized_pnl = self.calculate_unrealized_pnl(position, current_price)
            
            open_positions.append({
                "symbol": symbol,
                "shares": position.shares,
                "entry_price": position.entry_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "stop_price": position.stop_price
            })
        
        return {
            "equity": round(self.equity, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_realized_pnl": round(self.daily_realized_pnl, 2),
            "open_positions": open_positions,
            "trade_blocked": self.trade_blocked,
            "max_daily_loss": round(self.equity * 0.02, 2)
        }
