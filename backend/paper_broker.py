"""
Paper trading broker module.
Simulates trade execution with deterministic fills and risk controls.
"""

from typing import Dict, Optional, List
from datetime import datetime, date
from enum import Enum
import math


class OrderType(Enum):
    """Order types."""
    BUY = "BUY"
    EXIT = "EXIT"


class PendingOrder:
    """Represents a pending order that will execute on the next candle open."""
    def __init__(self, order_type: OrderType, symbol: str, signal_price: float, timestamp: int, 
                 stop_distance: Optional[float] = None, reason: Optional[str] = None):
        self.order_type = order_type
        self.symbol = symbol
        self.signal_price = signal_price  # Price from signal (for reference)
        self.timestamp = timestamp  # Timestamp when order was created
        self.stop_distance = stop_distance  # For BUY orders only
        self.reason = reason  # For EXIT orders only
    
    def to_dict(self):
        return {
            "order_type": self.order_type.value,
            "symbol": self.symbol,
            "signal_price": self.signal_price,
            "timestamp": self.timestamp,
            "stop_distance": self.stop_distance,
            "reason": self.reason
        }


class Position:
    """Represents an open position."""
    def __init__(self, symbol: str, entry_time: int, entry_price: float, shares: int, stop_price: float, entry_commission: float = 0.0):
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.shares = shares
        self.stop_price = stop_price
        self.entry_commission = entry_commission  # Commission paid on entry
    
    def to_dict(self):
        return {
            "symbol": self.symbol,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "shares": self.shares,
            "stop_price": self.stop_price,
            "entry_commission": self.entry_commission
        }


class PaperBroker:
    """
    Paper trading broker that manages pending orders and executes trades.
    
    Execution Model:
    - Strategies generate signals on candle close
    - Signals create pending orders (no immediate execution)
    - Pending orders execute on the next candle open price
    - Stop-loss and kill-switch exits also create pending orders
    - All orders execute at candle open with slippage and commissions applied
    
    Commission Model:
    - Configurable per-share commission (e.g., $0.005 per share)
    - Configurable per-trade commission (e.g., $1.00 per trade)
    - Commission = (shares * commission_per_share) + commission_per_trade
    - Commissions deducted on both BUY and EXIT
    - Commissions reduce realized P&L (included in net P&L calculation)
    - Commissions deducted from cash on BUY (reduces available cash)
    - Exit commissions deducted from proceeds on EXIT (reduces cash received)
    - Equity calculation includes commissions (via cash reduction)
    - Daily P&L includes commissions (via net P&L calculation)
    - Kill-switch logic includes commissions (via daily P&L)
    
    This ensures:
    - No lookahead bias (signals on close, execution on next open)
    - Deterministic execution (same inputs produce same outputs)
    - Identical execution path for replay and live trading
    - Realistic trading costs included in P&L
    
    Tracks account equity, positions, and enforces risk controls.
    """
    
    def __init__(self, initial_equity: float = 100000.0, commission_per_share: float = 0.005, commission_per_trade: float = 0.0, slippage: float = 0.0002):
        # FIX 1: CASH-BASED EQUITY ACCOUNTING
        # Initialize cash and equity to initial_equity
        self.cash = initial_equity
        self.equity = initial_equity
        self.initial_equity = initial_equity  # Keep for reference only
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.daily_pnl = 0.0
        self.last_reset_date = date.today()
        self.trade_blocked = False  # Kill switch flag
        self.slippage = slippage  # Configurable slippage (default: 0.02% = 0.0002)
        
        # Commission model: per-share and per-trade commissions
        self.commission_per_share = commission_per_share  # e.g., 0.005 = $0.005 per share
        self.commission_per_trade = commission_per_trade  # e.g., 1.0 = $1.00 per trade
        
        # Track realized P&L for the day
        self.daily_realized_pnl = 0.0
        
        # Pending orders queue - orders execute on next candle open
        self.pending_orders: List[PendingOrder] = []
    
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
    
    def _execute_buy_internal(self, symbol: str, fill_price: float, stop_distance: float, timestamp: int) -> Optional[Dict]:
        """
        Internal method to execute a BUY order immediately.
        Called by process_pending_orders when order executes.
        
        Args:
            symbol: Trading symbol
            fill_price: Fill price (next candle open with slippage)
            stop_distance: Stop loss distance
            timestamp: Execution timestamp
        
        Returns:
            Dict with trade info if executed, None if blocked
        """
        # Calculate position size
        shares = self.calculate_position_size(fill_price, stop_distance, self.equity)
        
        if shares == 0:
            return None  # Position too small
        
        # Calculate commission
        commission = self.calculate_commission(shares)
        
        # FIX 1: CASH-BASED EQUITY ACCOUNTING
        # Deduct (shares * fill_price + commission) from cash
        cost = shares * fill_price
        total_cost = cost + commission
        if total_cost > self.cash:
            return None  # Insufficient cash
        
        self.cash -= total_cost
        
        # AUDIT FIX: Assert cash never goes negative (fail fast > silent failure)
        assert self.cash >= 0, f"Cash became negative: {self.cash} after BUY of {shares} shares @ {fill_price} + commission {commission}"
        
        # Calculate stop price
        stop_price = fill_price - stop_distance
        
        # Create position (store entry commission for P&L calculation on exit)
        position = Position(
            symbol=symbol,
            entry_time=timestamp,
            entry_price=fill_price,
            shares=shares,
            stop_price=stop_price,
            entry_commission=commission
        )
        
        self.positions[symbol] = position
        
        return {
            "action": "BUY",
            "symbol": symbol,
            "shares": shares,
            "entry_price": fill_price,
            "stop_price": stop_price,
            "entry_commission": commission,
            "exit_commission": 0.0,  # No exit commission yet
            "total_commission": commission,  # Only entry commission at this point
            "net_pnl": None,  # No P&L until exit
            "timestamp": timestamp
        }
    
    def _execute_exit_internal(self, symbol: str, fill_price: float, timestamp: int, reason: str) -> Optional[Dict]:
        """
        Internal method to execute an EXIT order immediately.
        Called by process_pending_orders when order executes.
        
        Args:
            symbol: Trading symbol
            fill_price: Fill price (next candle open with slippage)
            timestamp: Execution timestamp
            reason: Exit reason
        
        Returns:
            Dict with trade info if executed, None if no position
        """
        position = self.positions[symbol]
        
        # Calculate exit commission
        exit_commission = self.calculate_commission(position.shares)
        total_commission = position.entry_commission + exit_commission
        
        # Calculate P&L (gross P&L minus both entry and exit commissions)
        gross_pnl = (fill_price - position.entry_price) * position.shares
        net_pnl = gross_pnl - total_commission  # Both entry and exit commissions reduce P&L
        
        # FIX 1: CASH-BASED EQUITY ACCOUNTING
        # Add (shares * fill_price - exit_commission) to cash
        # Note: Entry commission was already deducted on BUY
        proceeds = position.shares * fill_price
        net_proceeds = proceeds - exit_commission
        self.cash += net_proceeds
        
        # AUDIT FIX: Assert cash is valid (fail fast > silent failure)
        assert self.cash >= 0, f"Cash became negative: {self.cash} after EXIT of {position.shares} shares @ {fill_price} - exit commission {exit_commission}"
        
        # Update realized P&L (net P&L includes commission impact)
        self.daily_realized_pnl += net_pnl
        
        # Remove position
        del self.positions[symbol]
        
        return {
            "action": "EXIT",
            "symbol": symbol,
            "shares": position.shares,
            "entry_price": position.entry_price,
            "exit_price": fill_price,
            "pnl": net_pnl,  # Net P&L (after both entry and exit commissions)
            "net_pnl": net_pnl,  # Explicit net_pnl field (same as pnl)
            "entry_commission": position.entry_commission,
            "exit_commission": exit_commission,
            "total_commission": total_commission,
            "timestamp": timestamp,
            "reason": reason
        }
    
    def process_pending_orders(self, current_open_price: float, timestamp: int) -> List[Dict]:
        """
        Process all pending orders at the current candle open price.
        Orders execute at the open price with slippage applied.
        
        Args:
            current_open_price: Current candle open price
            timestamp: Current timestamp
        
        Returns:
            List of executed trade dicts
        """
        self.reset_daily_stats_if_needed()
        
        executed_trades = []
        orders_to_process = list(self.pending_orders)  # Copy list
        self.pending_orders.clear()  # Clear queue
        
        for order in orders_to_process:
            if order.order_type == OrderType.BUY:
                # Check if trade is still allowed
                if self.trade_blocked:
                    continue
                
                # Check if position already exists (may have been created by another order)
                if order.symbol in self.positions:
                    continue
                
                # Check max positions limit
                if len(self.positions) >= 3:
                    continue
                
                # Check daily loss limit
                if self.check_daily_loss_limit():
                    self.trigger_kill_switch()
                    continue
                
                # Apply slippage to open price
                fill_price = self.apply_slippage(current_open_price, is_buy=True)
                
                # Execute BUY
                trade = self._execute_buy_internal(
                    symbol=order.symbol,
                    fill_price=fill_price,
                    stop_distance=order.stop_distance,
                    timestamp=timestamp
                )
                
                if trade:
                    executed_trades.append(trade)
            
            elif order.order_type == OrderType.EXIT:
                # Check if position still exists (may have been closed by another order)
                if order.symbol not in self.positions:
                    continue
                
                # Apply slippage to open price
                fill_price = self.apply_slippage(current_open_price, is_buy=False)
                
                # Execute EXIT
                trade = self._execute_exit_internal(
                    symbol=order.symbol,
                    fill_price=fill_price,
                    timestamp=timestamp,
                    reason=order.reason or "EXIT"
                )
                
                if trade:
                    executed_trades.append(trade)
        
        return executed_trades
    
    def calculate_commission(self, shares: int) -> float:
        """
        Calculate commission for a trade.
        
        Commission = (shares * commission_per_share) + commission_per_trade
        
        Args:
            shares: Number of shares traded
        
        Returns:
            Total commission amount
        """
        return (shares * self.commission_per_share) + self.commission_per_trade
    
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
        
        Commission Integration:
        - Uses daily_pnl which includes daily_realized_pnl (net P&L with commissions)
        - Commissions reduce realized P&L, so they're included in kill-switch calculation
        - Kill-switch triggers based on net losses (after commissions)
        
        Returns:
            True if daily loss limit is breached (2% of equity)
        """
        self.reset_daily_stats_if_needed()
        max_daily_loss = self.equity * 0.02
        # daily_pnl includes commissions via daily_realized_pnl (net P&L)
        return self.daily_pnl <= -max_daily_loss
    
    def calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """Calculate unrealized P&L for a position."""
        return (current_price - position.entry_price) * position.shares
    
    def update_equity(self, current_prices: Dict[str, float]):
        """
        Update account equity based on current positions and prices.
        
        FIX 1: CASH-BASED EQUITY ACCOUNTING
        Equity = cash + sum(unrealized P&L of open positions)
        
        Commission Integration:
        - Commissions are deducted from cash on BUY and EXIT
        - Since equity = cash + unrealized_pnl, commissions are automatically included
        - Entry commissions reduce cash immediately on BUY
        - Exit commissions reduce cash on EXIT (via net_proceeds)
        
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
        # Commissions are included via cash reduction (entry commissions deducted on BUY,
        # exit commissions deducted on EXIT via net_proceeds)
        self.equity = self.cash + unrealized_pnl
        
        # AUDIT FIX: Assert equity is valid (fail fast > silent failure)
        assert self.equity >= 0, f"Equity became negative: {self.equity} (cash: {self.cash}, unrealized: {unrealized_pnl})"
        
        # Update daily P&L (realized + unrealized)
        # daily_realized_pnl includes net P&L (gross P&L - commissions) from closed trades
        # Commissions are included in daily P&L via daily_realized_pnl
        self.daily_pnl = self.daily_realized_pnl + unrealized_pnl
    
    def execute_buy(self, symbol: str, signal_price: float, stop_distance: float, timestamp: int) -> Optional[Dict]:
        """
        Create a pending BUY order that will execute on the next candle open.
        
        Args:
            symbol: Trading symbol
            signal_price: Price from signal (for reference, actual fill at next open)
            stop_distance: Stop loss distance (2 * ATR)
            timestamp: Current timestamp
        
        Returns:
            Dict with pending order info if created, None if blocked
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
        
        # Create pending BUY order
        pending_order = PendingOrder(
            order_type=OrderType.BUY,
            symbol=symbol,
            signal_price=signal_price,
            timestamp=timestamp,
            stop_distance=stop_distance
        )
        
        self.pending_orders.append(pending_order)
        
        return {
            "action": "BUY_PENDING",
            "symbol": symbol,
            "signal_price": signal_price,
            "stop_distance": stop_distance,
            "timestamp": timestamp
        }
    
    def execute_exit(self, symbol: str, signal_price: float, timestamp: int, reason: str) -> Optional[Dict]:
        """
        Create a pending EXIT order that will execute on the next candle open.
        
        Args:
            symbol: Trading symbol
            signal_price: Price from signal (for reference, actual fill at next open)
            timestamp: Current timestamp
            reason: Exit reason
        
        Returns:
            Dict with pending order info if created, None if no position
        """
        self.reset_daily_stats_if_needed()
        
        # Check if position exists
        if symbol not in self.positions:
            return None
        
        # Create pending EXIT order
        pending_order = PendingOrder(
            order_type=OrderType.EXIT,
            symbol=symbol,
            signal_price=signal_price,
            timestamp=timestamp,
            reason=reason
        )
        
        self.pending_orders.append(pending_order)
        
        return {
            "action": "EXIT_PENDING",
            "symbol": symbol,
            "signal_price": signal_price,
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
        Create pending EXIT orders for all open positions (for kill switch).
        Orders will execute on next candle open.
        
        FIX 4: KILL SWITCH BEHAVIOR (FAIL-SAFE)
        This is called automatically when kill switch is triggered.
        
        Args:
            current_prices: Current prices for all symbols
            timestamp: Current timestamp
        
        Returns:
            List of pending exit order dicts
        """
        pending_exits = []
        symbols_to_close = list(self.positions.keys())
        
        for symbol in symbols_to_close:
            if symbol in current_prices:
                exit_order = self.execute_exit(
                    symbol=symbol,
                    signal_price=current_prices[symbol],
                    timestamp=timestamp,
                    reason="Kill switch triggered"
                )
                if exit_order:
                    pending_exits.append(exit_order)
        
        return pending_exits
    
    def check_stop_losses(self, current_prices: Dict[str, float], timestamp: int) -> List[Dict]:
        """
        FIX 2: AUTOMATIC STOP-LOSS ENFORCEMENT
        Check all open positions for stop-loss breaches on candle close.
        If current price <= stop_price, create pending EXIT order with reason "STOP_LOSS".
        Order will execute on next candle open.
        
        Args:
            current_prices: Dict of symbol -> current price (candle close)
            timestamp: Current timestamp
        
        Returns:
            List of pending exit order dicts for stop-loss exits
        """
        pending_exits = []
        symbols_to_check = list(self.positions.keys())
        
        for symbol in symbols_to_check:
            if symbol in current_prices:
                position = self.positions[symbol]
                current_price = current_prices[symbol]
                
                # FIX 2: If current candle close <= stop_price, create pending EXIT
                if current_price <= position.stop_price:
                    exit_order = self.execute_exit(
                        symbol=symbol,
                        signal_price=current_price,
                        timestamp=timestamp,
                        reason="STOP_LOSS"
                    )
                    if exit_order:
                        pending_exits.append(exit_order)
        
        return pending_exits
    
    def check_and_enforce_risk_controls(self, current_prices: Dict[str, float], timestamp: int) -> List[Dict]:
        """
        FIX 4: KILL SWITCH BEHAVIOR (FAIL-SAFE)
        Check daily loss limit and automatically create pending EXIT orders for all positions if breached.
        Orders will execute on next candle open.
        This must be called on every candle close to ensure fail-safe behavior.
        
        Args:
            current_prices: Dict of symbol -> current price
            timestamp: Current timestamp
        
        Returns:
            List of pending exit order dicts if kill switch triggered, empty list otherwise
        """
        # Update equity first to get current daily P&L
        self.update_equity(current_prices)
        
        # Check if daily loss limit is breached
        if self.check_daily_loss_limit() and not self.trade_blocked:
            # FIX 4: Trigger kill switch and create pending EXIT orders for all positions
            self.trigger_kill_switch()
            pending_exits = self.close_all_positions(current_prices, timestamp)
            
            return pending_exits
        
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
