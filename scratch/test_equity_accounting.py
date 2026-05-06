"""
Comprehensive Equity Accounting Test Suite

Validates correct separation of cash vs equity for short-selling strategy.

Key Invariants:
1. Equity = Cash + Position_Value (position_value is negative for shorts)
2. Short sale proceeds increase cash but don't change equity (offset by negative position)
3. Realized PnL is recognized only on covered portion
4. Weighted average entry updates correctly on adds
5. Equity is used for position sizing, drawdown, returns
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.rl.env import ParabolicReversalEnv, EnvironmentConfig


def check_invariants(env, test_name):
    """Verify accounting invariants."""
    errors = []
    
    # Invariant 1: Equity = Cash + Position Value
    expected_equity = env.cash + env.current_position_value
    if abs(env.current_capital - expected_equity) > 0.01:
        errors.append(f"  [{test_name}] Equity mismatch: {env.current_capital} != {expected_equity}")
    
    # Invariant 2: Position Value = Position * Price
    if env.current_position != 0 and env.current_price > 0:
        expected_pos_value = env.current_position * env.current_price
        if abs(env.current_position_value - expected_pos_value) > 0.01:
            errors.append(f"  [{test_name}] Position value mismatch: {env.current_position_value} != {expected_pos_value}")
    
    # Invariant 3: Unrealized = Position * (Price - Entry)
    if env.current_position != 0 and env.entry_price > 0:
        expected_unrealized = env.current_position * (env.current_price - env.entry_price)
        if abs(env.unrealized_pnl - expected_unrealized) > 0.01:
            errors.append(f"  [{test_name}] Unrealized mismatch: {env.unrealized_pnl} != {expected_unrealized}")
    
    if errors:
        for e in errors:
            print(e)
        return False
    return True


def test_short_open_equity_conservation():
    """
    Test A: Opening a short should NOT increase equity.
    
    Short 100 shares @ $10:
    - Cash: +$1000 (proceeds) - fees
    - Position: -100 shares
    - Position Value: -$1000 (at $10)
    - Equity: ($1000 - fees) + (-$1000) = -fees
    
    Equity should DECREASE by fees only, not increase by proceeds.
    """
    print("\n" + "="*70)
    print("TEST A: Short Open - Equity Conservation")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    env.current_price = 10.0
    
    initial_equity = env.current_capital
    
    print(f"\nInitial: Cash=${env.cash:,.2f}, Equity=${env.current_capital:,.2f}")
    
    # Short $1000 worth @ $10 = 100 shares
    env._execute_position_change(-1000.0)
    env._update_portfolio_metrics()  # Update equity
    
    print(f"After short 100 @ $10:")
    print(f"  Cash: ${env.cash:,.2f}")
    print(f"  Position: {env.current_position:+.0f} shares")
    print(f"  Position Value: ${env.current_position_value:,.2f}")
    print(f"  Equity: ${env.current_capital:,.2f}")
    print(f"  Unrealized: ${env.unrealized_pnl:,.2f}")
    
    # Check: Position should be -100 shares
    assert abs(env.current_position - (-100)) < 0.1, f"Position: {env.current_position}"
    
    # Check: Equity should be ~$100,000 - fees (NOT $101,000)
    fees = 1000 * 0.01  # $10
    expected_equity = initial_equity - fees
    assert abs(env.current_capital - expected_equity) < 0.1, \
        f"Equity should be {expected_equity}, got {env.current_capital}"
    
    # Check: Cash should be $100,000 + $1000 - fees = $100,990
    expected_cash = 100000 + 1000 - fees
    assert abs(env.cash - expected_cash) < 0.1, f"Cash should be {expected_cash}, got {env.cash}"
    
    # Check invariants
    assert check_invariants(env, "Short Open")
    
    print(f"\n[PASS] Equity correctly decreased by fees only (${fees:.2f})")
    print(f"       Cash increased by proceeds, but offset by position liability")
    return True


def test_add_to_short_equity_conservation():
    """
    Test B: Adding to short should NOT increase equity.
    
    Start: Short 100 @ $10
    Add: Short 100 more @ $12
    
    Weighted avg entry should be $11.
    Equity should decrease by fees only.
    """
    print("\n" + "="*70)
    print("TEST B: Add to Short - Equity Conservation")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    env.current_price = 10.0
    
    # Initial short
    env._execute_position_change(-1000.0)
    env._update_portfolio_metrics()
    equity_after_first = env.current_capital
    
    print(f"\nAfter first short 100 @ $10:")
    print(f"  Cash: ${env.cash:,.2f}, Equity: ${env.current_capital:,.2f}")
    
    # Add to short at $12
    env.current_price = 12.0
    env._execute_position_change(-2400.0)  # $2400 total = 200 shares
    env._update_portfolio_metrics()
    
    print(f"\nAfter adding 100 @ $12 (total 200 shares):")
    print(f"  Cash: ${env.cash:,.2f}")
    print(f"  Position: {env.current_position:+.0f} shares")
    print(f"  Avg Entry: ${env.entry_price:.2f}")
    print(f"  Position Value: ${env.current_position_value:,.2f}")
    print(f"  Equity: ${env.current_capital:,.2f}")
    
    # Check weighted average: (100*10 + 100*12) / 200 = 11
    assert abs(env.entry_price - 11.0) < 0.01, f"Avg entry: {env.entry_price}"
    
    # Check position: 200 shares
    assert abs(env.current_position - (-200)) < 0.1, f"Position: {env.current_position}"
    
    # Equity should have decreased by fees only
    # Fees are calculated on delta_value in _execute_position_change
    # First trade: fees = $1000 * 0.01 = $10
    # Add trade: delta = -$1400, fees = $14
    # Total fees = $24
    assert abs(env.current_capital - 99776) < 1.0, \
        f"Equity should be ~99776, got {env.current_capital}"
    
    assert check_invariants(env, "Add Short")
    
    print(f"\n[PASS] Equity correctly calculated")
    print(f"       Weighted avg entry correct: ${env.entry_price:.2f}")
    return True


def test_partial_cover_equity_conservation():
    """
    Test C: Partial cover should realize PnL on covered portion only.
    
    Start: Short 200 @ $11 (avg entry)
    Cover 50 @ $9 (profit $2/share = $100)
    
    Realized PnL: $100
    Remaining position: 150 shares
    Avg entry: still $11
    """
    print("\n" + "="*70)
    print("TEST C: Partial Cover - Realized PnL on Covered Portion Only")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    
    # Setup: Short 200 @ $11
    env.current_price = 11.0
    env._execute_position_change(-2200.0)  # 200 shares
    env.entry_price = 11.0  # Force avg entry
    env._update_portfolio_metrics()
    
    print(f"\nSetup: Short 200 @ $11")
    print(f"  Cash: ${env.cash:,.2f}, Equity: ${env.current_capital:,.2f}")
    print(f"  Position: {env.current_position:+.0f} shares")
    print(f"  Unrealized: ${env.unrealized_pnl:,.2f}")
    
    initial_realized = env.realized_pnl_session
    initial_equity = env.current_capital
    
    # Cover 50 shares @ $9
    env.current_price = 9.0
    env._execute_position_change(-1350.0)  # 150 shares remaining = $1350 exposure
    env._update_portfolio_metrics()
    
    print(f"\nAfter covering 50 @ $9:")
    print(f"  Cash: ${env.cash:,.2f}")
    print(f"  Position: {env.current_position:+.0f} shares")
    print(f"  Avg Entry: ${env.entry_price:.2f} (should be unchanged)")
    print(f"  Realized PnL: ${env.realized_pnl_session:,.2f}")
    print(f"  Unrealized: ${env.unrealized_pnl:,.2f}")
    print(f"  Equity: ${env.current_capital:,.2f}")
    
    # Check realized PnL: 50 * ($11 - $9) = $100 profit
    expected_realized = 50 * (11.0 - 9.0)
    assert abs(env.realized_pnl_session - expected_realized) < 0.1, \
        f"Realized should be {expected_realized}, got {env.realized_pnl_session}"
    
    # Check avg entry unchanged
    assert abs(env.entry_price - 11.0) < 0.01, f"Avg entry should be 11, got {env.entry_price}"
    
    # Check remaining position
    assert abs(env.current_position - (-150)) < 0.1, f"Position should be -150, got {env.current_position}"
    
    # Equity should have increased (profitable cover)
    # Just verify equity increased from initial
    assert env.current_capital > initial_equity, \
        f"Equity should increase after profitable cover: {env.current_capital} vs {initial_equity}"
    
    assert check_invariants(env, "Partial Cover")
    
    print(f"\n[PASS] Realized PnL correct: ${env.realized_pnl_session:,.2f}")
    print(f"       Avg entry unchanged: ${env.entry_price:.2f}")
    print(f"       Equity increased by profit minus fees")
    return True


def test_full_cover_equity_conservation():
    """
    Test D: Full cover should realize all remaining PnL.
    
    Start: Short 150 @ $11
    Cover all 150 @ $9 (profit $2/share = $300)
    
    Realized PnL: $300 total
    Position: 0
    Equity: Cash only (no position liability)
    """
    print("\n" + "="*70)
    print("TEST D: Full Cover - All PnL Realized")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    
    # Setup: Short 150 @ $11
    env.current_price = 11.0
    env._execute_position_change(-1650.0)  # 150 shares
    env.entry_price = 11.0
    env._update_portfolio_metrics()
    
    print(f"\nSetup: Short 150 @ $11")
    print(f"  Cash: ${env.cash:,.2f}, Equity: ${env.current_capital:,.2f}")
    
    initial_equity = env.current_capital
    
    # Cover all @ $9
    env.current_price = 9.0
    env._execute_position_change(0.0)  # Full cover
    env._update_portfolio_metrics()
    
    print(f"\nAfter covering all 150 @ $9:")
    print(f"  Cash: ${env.cash:,.2f}")
    print(f"  Position: {env.current_position:+.0f} shares")
    print(f"  Avg Entry: ${env.entry_price:.2f}")
    print(f"  Realized PnL: ${env.realized_pnl_session:,.2f}")
    print(f"  Unrealized: ${env.unrealized_pnl:,.2f}")
    print(f"  Equity: ${env.current_capital:,.2f}")
    
    # Check realized PnL: 150 * ($11 - $9) = $300
    expected_realized = 150 * (11.0 - 9.0)
    assert abs(env.realized_pnl_session - expected_realized) < 0.1, \
        f"Realized should be {expected_realized}, got {env.realized_pnl_session}"
    
    # Check position closed
    assert abs(env.current_position) < 0.001, f"Position should be 0, got {env.current_position}"
    assert env.entry_price == 0.0, f"Entry should be 0, got {env.entry_price}"
    
    # Equity should equal cash (no position)
    assert abs(env.current_capital - env.cash) < 0.01, \
        f"Equity ({env.current_capital}) should equal cash ({env.cash})"
    
    assert check_invariants(env, "Full Cover")
    
    print(f"\n[PASS] Full cover realized all PnL: ${env.realized_pnl_session:,.2f}")
    print(f"       Equity equals cash: ${env.current_capital:,.2f}")
    return True


def test_complete_trade_sequence():
    """
    Test E: Complete sequence from user requirements.
    
    1. Short 100 @ $10
    2. Add 100 @ $12 (avg entry = $11)
    3. Cover 50 @ $11 (breakeven on covered portion)
    4. Cover 150 @ $9 (profit $2/share on remaining = $300)
    
    Total realized: $0 + $300 = $300
    """
    print("\n" + "="*70)
    print("TEST E: Complete Trade Sequence")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    
    initial_equity = env.current_capital
    
    # Step 1: Short 100 @ $10
    print("\n1. Short 100 @ $10")
    env.current_price = 10.0
    env._execute_position_change(-1000.0)
    env._update_portfolio_metrics()
    print(f"   Cash: ${env.cash:,.2f}, Equity: ${env.current_capital:,.2f}")
    print(f"   Position: {env.current_position:+.0f}, Avg: ${env.entry_price:.2f}")
    assert check_invariants(env, "Step 1")
    
    # Step 2: Add 100 @ $12
    print("\n2. Add 100 @ $12")
    env.current_price = 12.0
    env._execute_position_change(-2400.0)  # Total $2400 = 200 shares
    env._update_portfolio_metrics()
    print(f"   Cash: ${env.cash:,.2f}, Equity: ${env.current_capital:,.2f}")
    print(f"   Position: {env.current_position:+.0f}, Avg: ${env.entry_price:.2f}")
    assert abs(env.entry_price - 11.0) < 0.01, f"Avg should be $11, got {env.entry_price}"
    assert check_invariants(env, "Step 2")
    
    # Step 3: Cover 50 @ $11
    print("\n3. Cover 50 @ $11")
    env.current_price = 11.0
    env._execute_position_change(-1650.0)  # 150 shares = $1650
    env._update_portfolio_metrics()
    print(f"   Cash: ${env.cash:,.2f}, Equity: ${env.current_capital:,.2f}")
    print(f"   Position: {env.current_position:+.0f}, Avg: ${env.entry_price:.2f}")
    print(f"   Realized: ${env.realized_pnl_session:,.2f}")
    # Cover at breakeven: 50 * ($11 - $11) = $0
    assert abs(env.realized_pnl_session - 0.0) < 0.1, f"Realized should be $0, got {env.realized_pnl_session}"
    assert check_invariants(env, "Step 3")
    
    # Step 4: Cover 150 @ $9
    print("\n4. Cover 150 @ $9")
    env.current_price = 9.0
    env._execute_position_change(0.0)  # Full cover
    env._update_portfolio_metrics()
    print(f"   Cash: ${env.cash:,.2f}, Equity: ${env.current_capital:,.2f}")
    print(f"   Position: {env.current_position:+.0f}, Realized: ${env.realized_pnl_session:,.2f}")
    
    # Total realized: $0 + 150 * ($11 - $9) = $300
    expected_total_realized = 150 * (11.0 - 9.0)
    assert abs(env.realized_pnl_session - expected_total_realized) < 0.1, \
        f"Total realized should be {expected_total_realized}, got {env.realized_pnl_session}"
    assert check_invariants(env, "Step 4")
    
    # Summary
    print(f"\n" + "-"*70)
    print(f"Summary:")
    print(f"  Initial Equity: ${initial_equity:,.2f}")
    print(f"  Final Equity: ${env.current_capital:,.2f}")
    print(f"  Total Realized PnL: ${env.realized_pnl_session:,.2f}")
    print(f"  Total Fees Paid: ${100000 - env.cash + env.realized_pnl_session:,.2f}")
    print(f"  Net Profit: ${env.current_capital - initial_equity:,.2f}")
    print("-"*70)
    
    print(f"\n[PASS] Complete sequence validated")
    print(f"       Total realized PnL: ${env.realized_pnl_session:,.2f}")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EQUITY ACCOUNTING VALIDATION SUITE")
    print("="*70)
    print("\nVerifying correct cash/equity separation for short-selling.")
    
    tests = [
        ("Short Open Equity Conservation", test_short_open_equity_conservation),
        ("Add to Short Equity Conservation", test_add_to_short_equity_conservation),
        ("Partial Cover Equity Conservation", test_partial_cover_equity_conservation),
        ("Full Cover Equity Conservation", test_full_cover_equity_conservation),
        ("Complete Trade Sequence", test_complete_trade_sequence),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n[FAIL] {name}")
        except AssertionError as e:
            failed += 1
            print(f"\n[FAIL] {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"\n[ERROR] {name}: {e}")
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    sys.exit(0 if failed == 0 else 1)
