"""
Regression test for action-semantics bug fix.

Validates that:
1. HOLD action (|action| <= 0.1) preserves current position
2. INCREASE-SHORT action (action < -0.1) makes position more negative
3. COVER action (action > 0.1) makes position less negative or flat
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.rl.env import ParabolicReversalEnv


def test_hold_preserves_position():
    """Test that HOLD action (|action| <= 0.1) truly preserves current short position."""
    print("=" * 60)
    print("TEST 1: HOLD action preserves current short position")
    print("=" * 60)
    
    env = ParabolicReversalEnv()
    
    # Mock a scenario with an existing short position
    env.current_position = -100.0  # Short 100 shares
    env.current_position_value = -100.0 * 10.0  # At price $10
    env.entry_price = 10.0
    env.current_price = 10.0
    env.cash = 110000.0  # Received proceeds from short
    env.initial_capital = 100000.0
    env.current_capital = 100000.0
    env.prev_capital = 100000.0
    env.peak_capital = 100000.0
    env.current_drawdown = 0.0
    env.circuit_breaker_triggered = False
    env.in_entry_window = True
    env.vwap_deviation = 30.0  # Valid entry condition
    env.episode_pnl = 0.0
    env.episode_trades = 0
    
    # Set up minimal data provider state to avoid errors
    class MockDataProvider:
        def __init__(self):
            self.current_bar_idx = 1
            self.current_data = None
        def get_current_bar(self):
            class Bar:
                open = high = low = close = vwap = 10.0
                vwap_deviation = 30.0
                timestamp = None
                volume = 1000
            return Bar()
        def advance(self):
            class Bar:
                open = high = low = close = vwap = 10.0
                vwap_deviation = 30.0
                timestamp = None
                volume = 1000
            return Bar()
        def is_done(self):
            return False
    
    env.data_provider = MockDataProvider()
    
    # Record position before
    position_before = env.current_position
    position_value_before = env.current_position_value
    cash_before = env.cash
    
    # Test various HOLD actions within the hold band
    hold_actions = [0.0, 0.05, -0.05, 0.1, -0.1]
    
    for action_val in hold_actions:
        action = np.array([action_val])
        
        # Store state before step
        pos_before = env.current_position
        
        # Take step with HOLD action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify position unchanged
        if env.current_position != pos_before:
            print(f"  FAIL: HOLD action {action_val} changed position from {pos_before} to {env.current_position}")
            return False
        else:
            print(f"  PASS: HOLD action {action_val} preserved position at {env.current_position}")
        
        # Reset for next test
        env.current_position = -100.0
        env.current_position_value = -1000.0
        env.cash = cash_before
    
    print()
    return True


def test_increase_short_makes_more_negative():
    """Test that INCREASE-SHORT action makes position more negative."""
    print("=" * 60)
    print("TEST 2: INCREASE-SHORT makes position more negative")
    print("=" * 60)
    
    env = ParabolicReversalEnv()
    
    # Start with small short position
    env.current_position = -50.0  # Short 50 shares
    env.current_position_value = -50.0 * 10.0
    env.entry_price = 10.0
    env.current_price = 10.0
    env.cash = 105000.0
    env.initial_capital = 100000.0
    env.current_capital = 100000.0
    env.prev_capital = 100000.0
    env.peak_capital = 100000.0
    env.current_drawdown = 0.0
    env.circuit_breaker_triggered = False
    env.in_entry_window = True
    env.vwap_deviation = 30.0
    env.episode_pnl = 0.0
    env.episode_trades = 0
    
    class MockDataProvider:
        def __init__(self):
            self.current_bar_idx = 1
            self.current_data = None
        def get_current_bar(self):
            class Bar:
                open = high = low = close = vwap = 10.0
                vwap_deviation = 30.0
                timestamp = None
                volume = 1000
            return Bar()
        def advance(self):
            class Bar:
                open = high = low = close = vwap = 10.0
                vwap_deviation = 30.0
                timestamp = None
                volume = 1000
            return Bar()
        def is_done(self):
            return False
    
    env.data_provider = MockDataProvider()
    
    # Test with strong increase-short action
    action = np.array([-0.5])  # 50% of max leverage
    pos_before = env.current_position
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if env.current_position < pos_before:
        print(f"  PASS: Action -0.5 changed position from {pos_before} to {env.current_position} (more negative)")
        return True
    else:
        print(f"  FAIL: Action -0.5 did not make position more negative: {pos_before} -> {env.current_position}")
        return False


def test_cover_makes_less_negative():
    """Test that COVER action makes position less negative or flat."""
    print("=" * 60)
    print("TEST 3: COVER makes position less negative or flat")
    print("=" * 60)
    
    env = ParabolicReversalEnv()
    
    # Start with short position
    env.current_position = -100.0  # Short 100 shares
    env.current_position_value = -100.0 * 10.0
    env.entry_price = 10.0
    env.current_price = 10.0
    env.cash = 110000.0
    env.initial_capital = 100000.0
    env.current_capital = 100000.0
    env.prev_capital = 100000.0
    env.peak_capital = 100000.0
    env.current_drawdown = 0.0
    env.circuit_breaker_triggered = False
    env.in_entry_window = True
    env.vwap_deviation = 30.0
    env.episode_pnl = 0.0
    env.episode_trades = 0
    
    class MockDataProvider:
        def __init__(self):
            self.current_bar_idx = 1
            self.current_data = None
        def get_current_bar(self):
            class Bar:
                open = high = low = close = vwap = 10.0
                vwap_deviation = 30.0
                timestamp = None
                volume = 1000
            return Bar()
        def advance(self):
            class Bar:
                open = high = low = close = vwap = 10.0
                vwap_deviation = 30.0
                timestamp = None
                volume = 1000
            return Bar()
        def is_done(self):
            return False
    
    env.data_provider = MockDataProvider()
    
    # Test with cover action (partial)
    action = np.array([0.3])  # 30% cover
    pos_before = env.current_position
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Position should be less negative (closer to zero)
    if env.current_position > pos_before and env.current_position <= 0:
        print(f"  PASS: Action 0.3 changed position from {pos_before} to {env.current_position} (less negative)")
        return True
    elif env.current_position == 0:
        print(f"  PASS: Action 0.3 closed position from {pos_before} to flat")
        return True
    else:
        print(f"  FAIL: Action 0.3 did not reduce short: {pos_before} -> {env.current_position}")
        return False


def test_boundary_values():
    """Test that boundary values -0.1 and 0.1 are classified as HOLD."""
    print("=" * 60)
    print("TEST 4: Boundary values -0.1 and 0.1 are HOLD")
    print("=" * 60)
    
    env = ParabolicReversalEnv()
    
    # Mock a scenario with an existing short position
    env.current_position = -100.0  # Short 100 shares
    env.current_position_value = -100.0 * 10.0  # At price $10
    env.entry_price = 10.0
    env.current_price = 10.0
    env.cash = 110000.0
    env.initial_capital = 100000.0
    env.current_capital = 100000.0
    env.prev_capital = 100000.0
    env.peak_capital = 100000.0
    env.current_drawdown = 0.0
    env.circuit_breaker_triggered = False
    env.in_entry_window = True
    env.vwap_deviation = 30.0
    env.episode_pnl = 0.0
    env.episode_trades = 0
    
    class MockDataProvider:
        def __init__(self):
            self.current_bar_idx = 1
            self.current_data = None
        def get_current_bar(self):
            class Bar:
                open = high = low = close = vwap = 10.0
                vwap_deviation = 30.0
                timestamp = None
                volume = 1000
            return Bar()
        def advance(self):
            class Bar:
                open = high = low = close = vwap = 10.0
                vwap_deviation = 30.0
                timestamp = None
                volume = 1000
            return Bar()
        def is_done(self):
            return False
    
    env.data_provider = MockDataProvider()
    cash_before = env.cash
    
    # Test boundary values explicitly
    boundary_values = [-0.1, 0.1]
    all_passed = True
    
    for action_val in boundary_values:
        # Reset position
        env.current_position = -100.0
        env.current_position_value = -1000.0
        env.cash = cash_before
        
        action = np.array([action_val])
        pos_before = env.current_position
        
        # Verify discretization classifies as HOLD (type 2)
        action_type = env._discretize_action(action_val)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if action_type != 2:
            print(f"  FAIL: action={action_val} classified as type={action_type}, expected type=2 (HOLD)")
            all_passed = False
        elif env.current_position != pos_before:
            print(f"  FAIL: Boundary action {action_val} changed position from {pos_before} to {env.current_position}")
            all_passed = False
        else:
            print(f"  PASS: Boundary action {action_val} is HOLD (type=2), position unchanged at {env.current_position}")
    
    print()
    return all_passed


def test_discretize_action():
    """Test the discretize action function directly."""
    print("=" * 60)
    print("TEST 5: Action discretization boundaries")
    print("=" * 60)
    
    env = ParabolicReversalEnv()
    
    # Test boundaries
    test_cases = [
        (-1.0, 0, "INCREASE_SHORT"),
        (-0.5, 0, "INCREASE_SHORT"),
        (-0.11, 0, "INCREASE_SHORT"),
        (-0.1, 2, "HOLD"),
        (-0.05, 2, "HOLD"),
        (0.0, 2, "HOLD"),
        (0.05, 2, "HOLD"),
        (0.1, 2, "HOLD"),
        (0.11, 1, "DECREASE_SHORT/COVER"),
        (0.5, 1, "DECREASE_SHORT/COVER"),
        (1.0, 1, "DECREASE_SHORT/COVER"),
    ]
    
    all_passed = True
    for action_val, expected_type, expected_name in test_cases:
        result = env._discretize_action(action_val)
        status = "PASS" if result == expected_type else "FAIL"
        if result != expected_type:
            all_passed = False
        print(f"  {status}: action={action_val:+.2f} -> type={result} (expected {expected_type}={expected_name})")
    
    print()
    return all_passed


if __name__ == "__main__":
    print("\nAction Semantics Regression Tests")
    print("=" * 60)
    print()
    
    results = []
    
    # Test discretization first
    results.append(("Discretization", test_discretize_action()))
    
    # Test boundary values -0.1 and 0.1 explicitly
    results.append(("Boundary values (-0.1, 0.1)", test_boundary_values()))
    
    # Test HOLD behavior
    results.append(("HOLD preserves position", test_hold_preserves_position()))
    
    # Test INCREASE-SHORT behavior
    results.append(("INCREASE-SHORT", test_increase_short_makes_more_negative()))
    
    # Test COVER behavior
    results.append(("COVER", test_cover_makes_less_negative()))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    
    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
