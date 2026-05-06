"""
Test environment position accounting in env.py

Tests the dollar-exposure based position tracking with correct financial accounting.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.rl.env import ParabolicReversalEnv, EnvironmentConfig


def test_position_accounting():
    """Test the actual env.py position accounting with dollar-based targets."""
    print("\n" + "="*70)
    print("TESTING env.py POSITION ACCOUNTING (Dollar-Exposure Based)")
    print("="*70)
    print("\nNote: Environment tracks dollar exposure, not fixed share counts.")
    print("When price changes, same dollar target = different share count.\n")
    
    # Create environment
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    
    # Set initial state manually
    env.current_capital = 100000.0
    env.current_price = 10.0
    
    print("1. Opening short: $1000 exposure @ $10.00")
    print("   Expected: -100 shares (1000/10)")
    env._execute_position_change(-1000.0)
    print(f"   Position: {env.current_position:+.0f} shares")
    print(f"   Avg Entry: ${env.entry_price:.2f}")
    print(f"   Capital: ${env.current_capital:,.2f}")
    print(f"   Realized Session: ${env.realized_pnl_session:,.2f}")
    
    assert abs(env.current_position - (-100)) < 0.1, f"Expected -100, got {env.current_position}"
    assert abs(env.entry_price - 10.0) < 0.01, f"Expected 10.0, got {env.entry_price}"
    print("   [PASS]\n")
    
    print("2. Adding to short: $2000 total exposure @ $12.00")
    print("   Expected: -167 shares (2000/12), avg entry = 10.80 weighted")
    env.current_price = 12.0
    env._execute_position_change(-2000.0)
    print(f"   Position: {env.current_position:+.0f} shares")
    print(f"   Avg Entry: ${env.entry_price:.2f}")
    print(f"   Capital: ${env.current_capital:,.2f}")
    
    # At $12, $2000 exposure = 166.67 shares
    # Weighted avg: (-100 * 10 + -66.67 * 12) / -166.67 = 10.80
    assert abs(env.current_position - (-166.67)) < 1.0, f"Expected ~-167, got {env.current_position}"
    assert abs(env.entry_price - 10.80) < 0.1, f"Expected ~10.80, got {env.entry_price}"
    print("   [PASS]\n")
    
    print("3. Partial cover: $1500 exposure @ $11.00 (covering ~$500)")
    print("   Expected: -136 shares (1500/11), avg entry unchanged")
    env.current_price = 11.0
    prev_realized = env.realized_pnl_session
    env._execute_position_change(-1500.0)
    print(f"   Position: {env.current_position:+.0f} shares")
    print(f"   Avg Entry: ${env.entry_price:.2f}")
    print(f"   Realized Session: ${env.realized_pnl_session:,.2f}")
    print(f"   Capital: ${env.current_capital:,.2f}")
    
    # Shares covered: 166.67 - 136.36 = 30.3 shares
    # Realized PnL: 30.3 * (10.80 - 11.00) = -6.06 (small loss)
    assert abs(env.current_position - (-136.36)) < 2.0, f"Expected ~-136, got {env.current_position}"
    assert abs(env.entry_price - 10.80) < 0.1, f"Avg entry should be unchanged, got {env.entry_price}"
    print("   [PASS]\n")
    
    print("4. Full cover: $0 exposure @ $9.00")
    print("   Expected: 0 shares, realized PnL updated")
    env.current_price = 9.0
    prev_realized = env.realized_pnl_session
    env._execute_position_change(0.0)
    print(f"   Position: {env.current_position:+.0f} shares")
    print(f"   Avg Entry: ${env.entry_price:.2f}")
    print(f"   Realized Session: ${env.realized_pnl_session:,.2f}")
    print(f"   Capital: ${env.current_capital:,.2f}")
    
    assert abs(env.current_position) < 0.1, f"Expected ~0, got {env.current_position}"
    assert env.entry_price == 0.0, f"Expected 0.0, got {env.entry_price}"
    # Should have realized remaining PnL
    assert env.realized_pnl_session != prev_realized, "Realized PnL should have changed"
    print("   [PASS]\n")
    
    print("="*70)
    print("ALL env.py ACCOUNTING TESTS PASSED")
    print("="*70)
    return True


def test_weighted_average_calculation():
    """Test the weighted average entry price calculation explicitly."""
    print("\n" + "="*70)
    print("TESTING WEIGHTED AVERAGE ENTRY CALCULATION")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.current_capital = 100000.0
    
    # Test case: Short 100 @ $10, Add 100 @ $12
    # Expected avg entry: $11.00
    print("\nManual weighted average test:")
    print("  Short 100 shares @ $10, then add 100 shares @ $12")
    
    env.current_price = 10.0
    env._execute_position_change(-1000.0)  # $1000 @ $10 = 100 shares
    
    env.current_price = 12.0
    env._execute_position_change(-2400.0)  # $2400 @ $12 = 200 shares total
    
    print(f"  Result: {env.current_position:+.0f} shares @ ${env.entry_price:.2f}")
    print(f"  Expected: -200 shares @ $11.00")
    
    # Weighted avg: (100*10 + 100*12) / 200 = 11
    assert abs(env.current_position - (-200)) < 0.1
    assert abs(env.entry_price - 11.0) < 0.01
    print("  [PASS]")
    
    print("\n" + "="*70)
    print("WEIGHTED AVERAGE TEST PASSED")
    print("="*70)
    return True


def test_realized_pnl_on_partial_cover():
    """Test that realized PnL is calculated only on covered portion."""
    print("\n" + "="*70)
    print("TESTING REALIZED PnL ON PARTIAL COVER")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.current_capital = 100000.0
    
    # Setup: Short 200 @ $10 (avg entry = $10)
    env.current_price = 10.0
    env._execute_position_change(-2000.0)
    
    # Cover 100 @ $8 (profit of $2/share on covered portion)
    env.current_price = 8.0
    initial_realized = env.realized_pnl_session
    env._execute_position_change(-800.0)  # Remaining $800 exposure = 100 shares
    
    # Expected realized: 100 shares * ($10 - $8) = $200
    expected_pnl = 100 * (10.0 - 8.0)
    actual_pnl = env.realized_pnl_session - initial_realized
    
    print(f"\n  Covered 100 shares @ $8 (avg entry $10)")
    print(f"  Expected realized PnL: ${expected_pnl:.2f}")
    print(f"  Actual realized PnL: ${actual_pnl:.2f}")
    print(f"  Remaining position: {env.current_position:+.0f} shares")
    print(f"  Avg entry (should be $10): ${env.entry_price:.2f}")
    
    assert abs(actual_pnl - expected_pnl) < 1.0, f"PnL mismatch: {actual_pnl} vs {expected_pnl}"
    assert abs(env.entry_price - 10.0) < 0.01, "Avg entry should not change on partial cover"
    print("  [PASS]")
    
    print("\n" + "="*70)
    print("PARTIAL COVER PnL TEST PASSED")
    print("="*70)
    return True


if __name__ == "__main__":
    all_passed = True
    
    try:
        test_position_accounting()
    except AssertionError as e:
        print(f"\n[FAIL] test_position_accounting: {e}")
        all_passed = False
    
    try:
        test_weighted_average_calculation()
    except AssertionError as e:
        print(f"\n[FAIL] test_weighted_average_calculation: {e}")
        all_passed = False
    
    try:
        test_realized_pnl_on_partial_cover()
    except AssertionError as e:
        print(f"\n[FAIL] test_realized_pnl_on_partial_cover: {e}")
        all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*70)
    
    sys.exit(0 if all_passed else 1)
