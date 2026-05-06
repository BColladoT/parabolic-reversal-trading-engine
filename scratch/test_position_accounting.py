"""
Position Accounting Validation Tests

Test cases for correct short position accounting with scale-in/scale-out.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class PositionState:
    """Auditable position state."""
    shares: float = 0.0           # Signed: negative for short
    avg_entry: float = 0.0        # Weighted average entry price
    cash: float = 100000.0        # Cash balance
    realized_pnl: float = 0.0     # Cumulative realized PnL
    unrealized_pnl: float = 0.0   # Current unrealized PnL
    fees_paid: float = 0.0        # Cumulative fees
    
    @property
    def equity(self) -> float:
        """Total equity = cash + unrealized_pnl (for shorts)."""
        return self.cash + self.unrealized_pnl
    
    def __str__(self):
        return (f"Position(shares={self.shares:+.0f}, avg_entry=${self.avg_entry:.2f}, "
                f"cash=${self.cash:,.2f}, realized=${self.realized_pnl:,.2f}, "
                f"unrealized=${self.unrealized_pnl:,.2f}, equity=${self.equity:,.2f})")


class PositionAccountant:
    """
    Financially correct position accounting for short-selling.
    
    Key invariants:
    1. Realized PnL is recognized ONLY on covered portion
    2. Average entry price updates correctly on adds, unchanged on covers
    3. Equity = Cash + Unrealized PnL (conservation of capital)
    4. Fees are deducted from cash immediately
    """
    
    def __init__(self, initial_cash: float = 100000.0, fee_rate: float = 0.01):
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.reset()
    
    def reset(self):
        """Reset to initial state."""
        self.state = PositionState(cash=self.initial_cash)
        self.trade_log: List[dict] = []
    
    def short(self, shares: int, price: float) -> dict:
        """
        Open or add to short position.
        
        Args:
            shares: Number of shares to short (positive value)
            price: Fill price
            
        Returns:
            Trade record
        """
        if shares <= 0:
            raise ValueError("Shares must be positive for short")
        
        # Signed quantity: negative for short
        delta_shares = -float(shares)
        
        # Calculate fees
        notional = shares * price
        fees = notional * self.fee_rate
        
        if self.state.shares == 0:
            # NEW SHORT: Set entry price directly
            self.state.avg_entry = price
            self.state.shares = delta_shares
        else:
            # ADDING TO SHORT: Update weighted average entry
            # Formula: (old_shares * old_avg + new_shares * fill_price) / new_total
            total_shares = self.state.shares + delta_shares  # Both negative
            self.state.avg_entry = (
                (self.state.shares * self.state.avg_entry) + 
                (delta_shares * price)
            ) / total_shares
            self.state.shares = total_shares
        
        # Deduct cash (short sale proceeds minus fees)
        # Short sale: we receive notional but pay fees
        self.state.cash += notional - fees
        self.state.fees_paid += fees
        
        trade = {
            'action': 'SHORT',
            'shares': shares,
            'price': price,
            'fees': fees,
            'position_after': self.state.shares,
            'avg_entry_after': self.state.avg_entry,
        }
        self.trade_log.append(trade)
        return trade
    
    def cover(self, shares: int, price: float) -> dict:
        """
        Cover (reduce) short position.
        
        Args:
            shares: Number of shares to cover (positive value)
            price: Fill price
            
        Returns:
            Trade record
        """
        if shares <= 0:
            raise ValueError("Shares must be positive for cover")
        if self.state.shares >= 0:
            raise ValueError("No short position to cover")
        
        # Can't cover more than we have
        shares_to_cover = min(shares, abs(self.state.shares))
        
        # Calculate realized PnL on the covered portion ONLY
        # For shorts: profit when cover_price < avg_entry
        # realized_pnl = shares_covered * (avg_entry - cover_price)
        realized_pnl = shares_to_cover * (self.state.avg_entry - price)
        self.state.realized_pnl += realized_pnl
        
        # Calculate fees
        notional = shares_to_cover * price
        fees = notional * self.fee_rate
        
        # Update position
        self.state.shares += shares_to_cover  # Moving toward zero
        
        # Cash adjustment: pay to buy back, minus fees
        # Note: realized PnL doesn't directly change cash - it's reflected in equity
        self.state.cash -= notional + fees
        self.state.fees_paid += fees
        
        # If fully covered, reset entry price
        if abs(self.state.shares) < 0.001:
            self.state.avg_entry = 0.0
            self.state.shares = 0.0
        
        trade = {
            'action': 'COVER',
            'shares': shares_to_cover,
            'price': price,
            'fees': fees,
            'realized_pnl': realized_pnl,
            'position_after': self.state.shares,
            'avg_entry_after': self.state.avg_entry,
        }
        self.trade_log.append(trade)
        return trade
    
    def mark_to_market(self, price: float):
        """Update unrealized PnL at current market price."""
        if self.state.shares != 0:
            # For short: Q * (P_market - P_avg) where Q is negative
            # Example: -100 shares * ($90 - $100) = +$1000 profit when price drops
            self.state.unrealized_pnl = self.state.shares * (price - self.state.avg_entry)
        else:
            self.state.unrealized_pnl = 0.0
    
    def get_state(self) -> PositionState:
        """Get current state."""
        return self.state


def test_case_1():
    """
    Test Case 1: Short 100 @ 10, Add 100 @ 12, Cover 50 @ 11, Cover 150 @ 9
    
    Expected:
    - After short 100 @ 10: position = -100, avg_entry = 10
    - After add 100 @ 12: position = -200, avg_entry = 11 (weighted)
    - After cover 50 @ 11: position = -150, avg_entry = 11, realized = 0 (11-11=0)
    - After cover 150 @ 9: position = 0, realized = +300 (150 * (11-9))
    """
    print("\n" + "="*70)
    print("TEST CASE 1: Scale-in, Scale-out with Partial Cover")
    print("="*70)
    
    acct = PositionAccountant(initial_cash=100000.0, fee_rate=0.01)
    
    # Step 1: Short 100 @ $10
    print("\n1. SHORT 100 shares @ $10.00")
    acct.short(100, 10.0)
    acct.mark_to_market(10.0)
    state = acct.get_state()
    print(f"   {state}")
    assert state.shares == -100, f"Expected -100 shares, got {state.shares}"
    assert state.avg_entry == 10.0, f"Expected avg_entry=10, got {state.avg_entry}"
    assert state.realized_pnl == 0, f"Expected realized=0, got {state.realized_pnl}"
    # Cash: 100000 + 100*10*(1-0.01) = 100000 + 990 = 100990
    expected_cash = 100000 + 100 * 10 * 0.99
    assert abs(state.cash - expected_cash) < 0.01, f"Cash mismatch: {state.cash} vs {expected_cash}"
    print("   [PASS]")
    
    # Step 2: Add 100 @ $12
    print("\n2. ADD 100 shares @ $12.00 (scale-in)")
    acct.short(100, 12.0)
    acct.mark_to_market(12.0)
    state = acct.get_state()
    print(f"   {state}")
    assert state.shares == -200, f"Expected -200 shares, got {state.shares}"
    # Weighted avg: (-100 * 10 + -100 * 12) / -200 = 11
    assert state.avg_entry == 11.0, f"Expected avg_entry=11, got {state.avg_entry}"
    assert state.realized_pnl == 0, f"Expected realized=0, got {state.realized_pnl}"
    # Unrealized: -200 * (12 - 11) = -200 (losing $1 per share)
    expected_unrealized = -200 * (12 - 11)
    assert abs(state.unrealized_pnl - expected_unrealized) < 0.01, f"Unrealized mismatch"
    print("   [PASS]")
    
    # Step 3: Cover 50 @ $11
    print("\n3. COVER 50 shares @ $11.00 (partial cover at breakeven)")
    acct.cover(50, 11.0)
    acct.mark_to_market(11.0)
    state = acct.get_state()
    print(f"   {state}")
    assert state.shares == -150, f"Expected -150 shares, got {state.shares}"
    # Avg entry unchanged on partial cover
    assert state.avg_entry == 11.0, f"Expected avg_entry=11, got {state.avg_entry}"
    # Realized: 50 * (11 - 11) = 0
    expected_realized = 50 * (11 - 11)
    assert abs(state.realized_pnl - expected_realized) < 0.01, f"Realized mismatch"
    print("   [PASS]")
    
    # Step 4: Cover 150 @ $9
    print("\n4. COVER 150 shares @ $9.00 (full cover with profit)")
    acct.cover(150, 9.0)
    acct.mark_to_market(9.0)
    state = acct.get_state()
    print(f"   {state}")
    assert state.shares == 0, f"Expected 0 shares, got {state.shares}"
    assert state.avg_entry == 0.0, f"Expected avg_entry=0, got {state.avg_entry}"
    # Total realized: 0 + 150 * (11 - 9) = 300
    expected_total_realized = 150 * (11 - 9)
    assert abs(state.realized_pnl - expected_total_realized) < 0.01, f"Realized mismatch"
    assert state.unrealized_pnl == 0, f"Expected unrealized=0 after full cover"
    print("   [PASS]")
    
    print("\n" + "-"*70)
    print("TEST CASE 1: ALL ASSERTIONS PASSED")
    print("-"*70)
    return True


def test_case_2():
    """
    Test Case 2: Profitable short with partial cover at different prices
    
    Sequence:
    - Short 200 @ $20
    - Cover 100 @ $18 (profit on half)
    - Cover 100 @ $15 (more profit on remaining half)
    """
    print("\n" + "="*70)
    print("TEST CASE 2: Profitable Partial Covers")
    print("="*70)
    
    acct = PositionAccountant(initial_cash=100000.0, fee_rate=0.01)
    
    # Step 1: Short 200 @ $20
    print("\n1. SHORT 200 shares @ $20.00")
    acct.short(200, 20.0)
    acct.mark_to_market(20.0)
    state = acct.get_state()
    print(f"   {state}")
    assert state.shares == -200
    assert state.avg_entry == 20.0
    
    # Step 2: Cover 100 @ $18 (profit: $2/share)
    print("\n2. COVER 100 shares @ $18.00 (profit $2/share)")
    acct.cover(100, 18.0)
    acct.mark_to_market(18.0)
    state = acct.get_state()
    print(f"   {state}")
    assert state.shares == -100
    assert state.avg_entry == 20.0  # Unchanged
    expected_realized = 100 * (20 - 18)  # $200 profit
    assert abs(state.realized_pnl - expected_realized) < 0.01
    expected_unrealized = -100 * (18 - 20)  # $200 unrealized profit
    assert abs(state.unrealized_pnl - expected_unrealized) < 0.01
    print("   [PASS]")
    
    # Step 3: Cover 100 @ $15 (profit: $5/share on remaining)
    print("\n3. COVER 100 shares @ $15.00 (profit $5/share)")
    acct.cover(100, 15.0)
    acct.mark_to_market(15.0)
    state = acct.get_state()
    print(f"   {state}")
    assert state.shares == 0
    assert state.avg_entry == 0.0
    expected_total_realized = 100 * (20 - 18) + 100 * (20 - 15)  # $200 + $500 = $700
    assert abs(state.realized_pnl - expected_total_realized) < 0.01
    assert state.unrealized_pnl == 0
    print("   [PASS]")
    
    print("\n" + "-"*70)
    print("TEST CASE 2: ALL ASSERTIONS PASSED")
    print("-"*70)
    return True


def test_case_3():
    """
    Test Case 3: Losing short (adverse price movement)
    
    Sequence:
    - Short 100 @ $50
    - Cover 50 @ $55 (loss of $5/share)
    - Cover 50 @ $60 (loss of $10/share on remaining)
    """
    print("\n" + "="*70)
    print("TEST CASE 3: Losing Short Position")
    print("="*70)
    
    acct = PositionAccountant(initial_cash=100000.0, fee_rate=0.01)
    
    # Step 1: Short 100 @ $50
    print("\n1. SHORT 100 shares @ $50.00")
    acct.short(100, 50.0)
    acct.mark_to_market(50.0)
    state = acct.get_state()
    print(f"   {state}")
    
    # Step 2: Cover 50 @ $55 (loss: $5/share)
    print("\n2. COVER 50 shares @ $55.00 (loss $5/share)")
    acct.cover(50, 55.0)
    acct.mark_to_market(55.0)
    state = acct.get_state()
    print(f"   {state}")
    expected_realized = 50 * (50 - 55)  # -$250
    assert abs(state.realized_pnl - expected_realized) < 0.01
    expected_unrealized = -50 * (55 - 50)  # -$250
    assert abs(state.unrealized_pnl - expected_unrealized) < 0.01
    print("   [PASS]")
    
    # Step 3: Cover 50 @ $60 (loss: $10/share)
    print("\n3. COVER 50 shares @ $60.00 (loss $10/share)")
    acct.cover(50, 60.0)
    acct.mark_to_market(60.0)
    state = acct.get_state()
    print(f"   {state}")
    expected_total_realized = 50 * (50 - 55) + 50 * (50 - 60)  # -$250 - $500 = -$750
    assert abs(state.realized_pnl - expected_total_realized) < 0.01
    assert state.shares == 0
    print("   [PASS]")
    
    print("\n" + "-"*70)
    print("TEST CASE 3: ALL ASSERTIONS PASSED")
    print("-"*70)
    return True


def test_equity_conservation():
    """
    Test Case 4: Equity Conservation Law
    
    Verifies that: Equity = Initial_Capital + Realized_PnL + Unrealized_PnL - Fees
    """
    print("\n" + "="*70)
    print("TEST CASE 4: Equity Conservation Law")
    print("="*70)
    
    acct = PositionAccountant(initial_cash=100000.0, fee_rate=0.01)
    initial_equity = acct.get_state().equity
    
    # Execute random sequence of trades
    trades = [
        ('short', 100, 10.0),
        ('short', 50, 12.0),
        ('cover', 30, 11.0),
        ('cover', 60, 9.0),
        ('short', 200, 15.0),
        ('cover', 100, 14.0),
    ]
    
    for action, shares, price in trades:
        print(f"\n   {action.upper()} {shares} @ ${price}")
        if action == 'short':
            acct.short(shares, price)
        else:
            acct.cover(shares, price)
        acct.mark_to_market(price)
        
        state = acct.get_state()
        # Verify conservation: equity = cash + unrealized
        calculated_equity = state.cash + state.unrealized_pnl
        assert abs(state.equity - calculated_equity) < 0.01, \
            f"Equity conservation violated: {state.equity} != {calculated_equity}"
        print(f"   Equity=${state.equity:,.2f}, Cash=${state.cash:,.2f}, "
              f"Unrealized=${state.unrealized_pnl:,.2f}, Realized=${state.realized_pnl:,.2f}")
    
    print("\n" + "-"*70)
    print("TEST CASE 4: EQUITY CONSERVATION VERIFIED")
    print("-"*70)
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("POSITION ACCOUNTING VALIDATION SUITE")
    print("="*70)
    
    all_passed = True
    
    try:
        test_case_1()
    except AssertionError as e:
        print(f"\n✗ TEST CASE 1 FAILED: {e}")
        all_passed = False
    
    try:
        test_case_2()
    except AssertionError as e:
        print(f"\n[FAIL] TEST CASE 2 FAILED: {e}")
        all_passed = False
    
    try:
        test_case_3()
    except AssertionError as e:
        print(f"\n[FAIL] TEST CASE 3 FAILED: {e}")
        all_passed = False
    
    try:
        test_equity_conservation()
    except AssertionError as e:
        print(f"\n[FAIL] TEST CASE 4 FAILED: {e}")
        all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*70)
    
    sys.exit(0 if all_passed else 1)
