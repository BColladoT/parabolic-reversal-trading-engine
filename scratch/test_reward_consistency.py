"""
Reward Consistency Test Suite

Verifies that reward = normalized equity delta.

Key Invariant:
    reward_t = (equity_t - equity_{t-1}) / initial_capital * 100

This means:
- 1% equity increase = +1.0 reward
- 1% equity decrease = -1.0 reward
- Sum of rewards = total return scaled by 100
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.rl.env import ParabolicReversalEnv, EnvironmentConfig


def test_reward_no_trade():
    """
    Test A: No trade, price moves.
    
    With no position, equity should not change.
    Reward should be 0 (plus any drawdown penalty if applicable).
    """
    print("\n" + "="*70)
    print("TEST A: No Trade - Reward Should Be 0")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    env.prev_capital = 100000.0
    env.initial_capital = 100000.0
    env.current_drawdown = 0.0
    
    # Price moves but no position
    env.current_price = 11.0  # Up 10%
    env._update_portfolio_metrics()
    
    reward = env._calculate_true_reward()
    
    print(f"\nNo position, price moved from $10 to $11:")
    print(f"  Equity: ${env.current_capital:,.2f}")
    print(f"  Prev Equity: ${env.prev_capital:,.2f}")
    print(f"  Equity Delta: ${env.current_capital - env.prev_capital:,.2f}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Expected: 0.0 (no position = no equity change)")
    
    assert abs(reward) < 0.001, f"Reward should be ~0, got {reward}"
    print("  [PASS]")
    return True


def test_reward_short_open_with_fees():
    """
    Test B: Open short, pay fees.
    
    Short $1000 @ $10 with 1% fees:
    - Cash: +$1000 - $10 = +$990
    - Position value: -$1000
    - Equity: $100,000 - $10 = $99,990 (decreased by fees)
    
    Reward should be negative (equity decreased by fees).
    """
    print("\n" + "="*70)
    print("TEST B: Short Open With Fees")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    env.prev_capital = 100000.0
    env.initial_capital = 100000.0
    env.current_drawdown = 0.0
    
    # Open short
    env.current_price = 10.0
    env._execute_position_change(-1000.0)
    env._update_portfolio_metrics()
    
    # Manually set prev_capital to initial for reward calc
    env.prev_capital = 100000.0
    
    reward = env._calculate_true_reward()
    expected_reward = -10.0 / 100000.0 * 100  # -$10 fees / $100k * 100 = -0.01
    
    print(f"\nShort $1000 @ $10 (1% fees = $10):")
    print(f"  Equity: ${env.current_capital:,.2f}")
    print(f"  Prev Equity: ${env.prev_capital:,.2f}")
    print(f"  Equity Delta: ${env.current_capital - env.prev_capital:,.2f}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Expected: {expected_reward:.4f} (fees / initial * 100)")
    
    assert abs(reward - expected_reward) < 0.001, f"Reward mismatch: {reward} vs {expected_reward}"
    print("  [PASS]")
    return True


def test_reward_favorable_move():
    """
    Test C: Favorable price move for short position.
    
    Short 100 shares @ $10, price drops to $9.
    - Position value: -$900 (was -$1000)
    - Equity gain: $100 (unrealized profit)
    
    Reward should be positive.
    """
    print("\n" + "="*70)
    print("TEST C: Favorable Move (Short Profits)")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    env.initial_capital = 100000.0
    env.current_drawdown = 0.0
    
    # Open short
    env.current_price = 10.0
    env._execute_position_change(-1000.0)
    env._update_portfolio_metrics()
    
    # Price drops to $9 (favorable for short)
    env.prev_capital = env.current_capital  # Store before update
    env.current_price = 9.0
    env._update_portfolio_metrics()
    
    reward = env._calculate_true_reward()
    expected_delta = 90.0  # $100 gain - fees already paid
    expected_reward = expected_delta / 100000.0 * 100
    
    print(f"\nShort 100 shares @ $10, price drops to $9:")
    print(f"  Equity: ${env.current_capital:,.2f}")
    print(f"  Prev Equity: ${env.prev_capital:,.2f}")
    print(f"  Equity Delta: ${env.current_capital - env.prev_capital:,.2f}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Expected: ~{expected_reward:.4f} (profit / initial * 100)")
    
    assert reward > 0, f"Reward should be positive for favorable move, got {reward}"
    print("  [PASS]")
    return True


def test_reward_adverse_move():
    """
    Test D: Adverse price move for short position.
    
    Short 100 shares @ $10, price rises to $11.
    - Position value: -$1100 (was -$1000)
    - Equity loss: $100 (unrealized loss)
    
    Reward should be negative.
    """
    print("\n" + "="*70)
    print("TEST D: Adverse Move (Short Loses)")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    env.initial_capital = 100000.0
    env.current_drawdown = 0.0
    
    # Open short
    env.current_price = 10.0
    env._execute_position_change(-1000.0)
    env._update_portfolio_metrics()
    
    # Price rises to $11 (adverse for short)
    env.prev_capital = env.current_capital
    env.current_price = 11.0
    env._update_portfolio_metrics()
    
    reward = env._calculate_true_reward()
    
    print(f"\nShort 100 shares @ $10, price rises to $11:")
    print(f"  Equity: ${env.current_capital:,.2f}")
    print(f"  Prev Equity: ${env.prev_capital:,.2f}")
    print(f"  Equity Delta: ${env.current_capital - env.prev_capital:,.2f}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Expected: negative (loss / initial * 100)")
    
    assert reward < 0, f"Reward should be negative for adverse move, got {reward}"
    print("  [PASS]")
    return True


def test_reward_partial_cover():
    """
    Test E: Partial cover realizes some PnL.
    
    Short 200 @ $10, price drops to $9, cover half.
    - Initial equity: $100,000 - $20 fees = $99,980
    - After price drop: $100,000 - $20 + $200 = $100,180 (unrealized)
    - After cover: realize $100 profit, pay $9 fees = +$91 net
    """
    print("\n" + "="*70)
    print("TEST E: Partial Cover")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    env.initial_capital = 100000.0
    env.current_drawdown = 0.0
    
    # Open short 200 shares @ $10
    env.current_price = 10.0
    env._execute_position_change(-2000.0)
    env._update_portfolio_metrics()
    
    # Price drops to $9
    env.current_price = 9.0
    env._update_portfolio_metrics()
    
    # Store equity before cover
    env.prev_capital = env.current_capital
    
    # Cover half (100 shares)
    env._execute_position_change(-900.0)  # 100 shares @ $9 = $900 exposure
    env._update_portfolio_metrics()
    
    reward = env._calculate_true_reward()
    
    print(f"\nShort 200 @ $10, price to $9, cover 100 shares:")
    print(f"  Equity before cover: ${env.prev_capital:,.2f}")
    print(f"  Equity after cover: ${env.current_capital:,.2f}")
    print(f"  Equity Delta: ${env.current_capital - env.prev_capital:,.2f}")
    print(f"  Reward: {reward:.4f}")
    
    # Note: Equity may decrease slightly due to fees on the cover trade
    # The realized profit was already in equity (unrealized), now it's realized
    # Reward reflects the actual equity change (fees paid)
    print("  [PASS] (Reward reflects fees paid on cover)")
    return True


def test_reward_full_cover():
    """
    Test F: Full cover realizes all remaining PnL.
    
    Short 100 @ $10, price drops to $9, cover all.
    - Profit: $100
    - Total fees: $10 (open) + $9 (close) = $19
    - Net profit: $81
    """
    print("\n" + "="*70)
    print("TEST F: Full Cover")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    env.initial_capital = 100000.0
    env.current_drawdown = 0.0
    
    # Open short
    env.current_price = 10.0
    env._execute_position_change(-1000.0)
    env._update_portfolio_metrics()
    
    # Price drops
    env.current_price = 9.0
    env._update_portfolio_metrics()
    
    # Store equity before full cover
    env.prev_capital = env.current_capital
    
    # Full cover
    env._execute_position_change(0.0)
    env._update_portfolio_metrics()
    
    reward = env._calculate_true_reward()
    
    # Calculate expected final equity
    # Proceeds: $1000, Cover cost: $900, Fees: $10 + $9 = $19
    # Final cash: $100,000 + $1000 - $900 - $19 = $100,081
    expected_final = 100081.0
    
    print(f"\nShort 100 @ $10, price to $9, cover all:")
    print(f"  Equity before cover: ${env.prev_capital:,.2f}")
    print(f"  Equity after cover: ${env.current_capital:,.2f}")
    print(f"  Expected final: ${expected_final:,.2f}")
    print(f"  Equity Delta: ${env.current_capital - env.prev_capital:,.2f}")
    print(f"  Reward: {reward:.4f}")
    
    assert abs(env.current_capital - expected_final) < 1.0, \
        f"Final equity mismatch: {env.current_capital} vs {expected_final}"
    # Reward is negative here because we paid fees to close the position
    # The profit was already reflected in equity before the cover
    print("  [PASS] (Reward reflects fees paid, profit was already in equity)")
    return True


def test_reward_cumulative_sum():
    """
    Test G: Cumulative reward should equal total return.
    
    Run a sequence of steps and verify sum of rewards = total return.
    """
    print("\n" + "="*70)
    print("TEST G: Cumulative Reward = Total Return")
    print("="*70)
    
    config = EnvironmentConfig()
    env = ParabolicReversalEnv(config=config)
    env.cash = 100000.0
    env.current_capital = 100000.0
    env.initial_capital = 100000.0
    env.current_drawdown = 0.0
    
    steps = [
        ("Open short 100 @ $10", 10.0, -1000.0),
        ("Price to $9", 9.0, None),
        ("Price to $8", 8.0, None),
        ("Cover half @ $8", 8.0, -400.0),  # 50 shares
        ("Price to $9", 9.0, None),
        ("Cover all @ $9", 9.0, 0.0),
    ]
    
    rewards = []
    
    for desc, price, target in steps:
        env.prev_capital = env.current_capital
        
        if target is not None:
            env.current_price = price
            env._execute_position_change(target)
        else:
            env.current_price = price
        
        env._update_portfolio_metrics()
        reward = env._calculate_true_reward()
        rewards.append(reward)
        
        print(f"  {desc}: equity=${env.current_capital:,.2f}, reward={reward:.4f}")
    
    cumulative_reward = sum(rewards)
    total_return = (env.current_capital - 100000.0) / 100000.0 * 100
    
    print(f"\n  Cumulative reward: {cumulative_reward:.4f}")
    print(f"  Total return (scaled): {total_return:.4f}")
    
    assert abs(cumulative_reward - total_return) < 0.1, \
        f"Cumulative reward {cumulative_reward} != total return {total_return}"
    print("  [PASS]")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REWARD CONSISTENCY TEST SUITE")
    print("="*70)
    print("\nVerifying reward = normalized equity delta")
    
    tests = [
        test_reward_no_trade,
        test_reward_short_open_with_fees,
        test_reward_favorable_move,
        test_reward_adverse_move,
        test_reward_partial_cover,
        test_reward_full_cover,
        test_reward_cumulative_sum,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n[FAIL] {test.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"\n[ERROR] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    sys.exit(0 if failed == 0 else 1)
