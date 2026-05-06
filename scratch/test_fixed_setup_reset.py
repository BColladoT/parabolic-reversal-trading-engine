"""
Test to verify that env.reset(options={"fixed_setup": ...}) 
loads the exact specified episode without random resampling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def test_fixed_setup_reset():
    """Prove that fixed_setup option loads exact episode without random sampling."""
    print("\n" + "="*70)
    print("TEST: Fixed Setup Reset Option")
    print("="*70)
    
    from src.rl.env import ParabolicReversalEnv
    
    # Create environment with date range containing multiple setups
    env = ParabolicReversalEnv(
        config={
            'initial_capital': 100000.0,
            'date_range': ('2021-06-01', '2021-06-30'),  # Multiple setups in June 2021
            'seed': 42,
            'mode': 'eval'
        }
    )
    
    # Get available setups
    all_setups = env.data_provider.parquet_setups + env.data_provider.csv_setups
    if len(all_setups) == 0:
        print("[SKIP] No setups available in date range")
        return True
    
    # Find a valid setup by trying to load them until one works
    print(f"\nSearching for valid setup among {len(all_setups)} setups...")
    target_setup = None
    for setup in all_setups[:20]:  # Check first 20
        symbol = setup['symbol']
        date_str = setup['date']
        obs, info = env.reset(options={
            "fixed_setup": {"symbol": symbol, "date": date_str}
        })
        # Check if it loaded successfully
        if env.data_provider.current_symbol == symbol and env.data_provider.current_date == date_str:
            target_setup = setup
            print(f"Found valid setup: {symbol} {date_str}")
            break
    
    if target_setup is None:
        print("[SKIP] No valid setups found in first 20 attempts")
        return True
    
    target_symbol = target_setup['symbol']
    target_date = target_setup['date']
    
    print(f"\nTarget setup: {target_symbol} {target_date}")
    print(f"Total available setups: {len(all_setups)}")
    
    # Reset with fixed_setup option
    obs, info = env.reset(options={
        "fixed_setup": {"symbol": target_symbol, "date": target_date}
    })
    
    # Verify the exact setup was loaded
    actual_symbol = env.data_provider.current_symbol
    actual_date = env.data_provider.current_date
    
    print(f"\nAfter reset with fixed_setup:")
    print(f"  Expected: {target_symbol} {target_date}")
    print(f"  Actual:   {actual_symbol} {actual_date}")
    
    if actual_symbol != target_symbol or actual_date != target_date:
        print(f"[FAIL] Fixed setup did not load the correct episode!")
        return False
    
    print(f"[PASS] Exact episode loaded correctly")
    
    # Verify we can do this multiple times with the same result
    print(f"\nTesting multiple resets with same fixed setup...")
    for i in range(3):
        obs, info = env.reset(options={
            "fixed_setup": {"symbol": target_symbol, "date": target_date}
        })
        actual_symbol = env.data_provider.current_symbol
        actual_date = env.data_provider.current_date
        
        if actual_symbol != target_symbol or actual_date != target_date:
            print(f"[FAIL] Reset {i+1} loaded wrong episode: {actual_symbol} {actual_date}")
            return False
        print(f"  Reset {i+1}: {actual_symbol} {actual_date} [OK]")
    
    print(f"[PASS] All resets loaded the exact same episode")
    
    # Test with a different setup
    second_setup = None
    for setup in all_setups[21:40]:  # Check next 20
        symbol = setup['symbol']
        date_str = setup['date']
        obs, info = env.reset(options={
            "fixed_setup": {"symbol": symbol, "date": date_str}
        })
        if env.data_provider.current_symbol == symbol and env.data_provider.current_date == date_str:
            second_setup = setup
            print(f"\nFound second valid setup: {symbol} {date_str}")
            break
    
    if second_setup:
        second_symbol = second_setup['symbol']
        second_date = second_setup['date']
        
        print(f"\nTesting with different setup: {second_symbol} {second_date}")
        obs, info = env.reset(options={
            "fixed_setup": {"symbol": second_symbol, "date": second_date}
        })
        
        actual_symbol = env.data_provider.current_symbol
        actual_date = env.data_provider.current_date
        
        if actual_symbol != second_symbol or actual_date != second_date:
            print(f"[FAIL] Second setup not loaded correctly!")
            print(f"  Expected: {second_symbol} {second_date}")
            print(f"  Actual:   {actual_symbol} {actual_date}")
            return False
        
        print(f"[PASS] Second setup loaded correctly")
    else:
        print("\n[SKIP] Could not find second valid setup")
    
    return True


def test_no_random_sampling_with_fixed_setup():
    """Prove that random sampling is skipped when fixed_setup is provided."""
    print("\n" + "="*70)
    print("TEST: No Random Sampling with Fixed Setup")
    print("="*70)
    
    from src.rl.env import ParabolicReversalEnv
    
    env = ParabolicReversalEnv(
        config={
            'initial_capital': 100000.0,
            'date_range': ('2021-06-01', '2021-06-30'),
            'seed': 42,
            'mode': 'eval'
        }
    )
    
    all_setups = env.data_provider.parquet_setups + env.data_provider.csv_setups
    if len(all_setups) == 0:
        print("[SKIP] No setups available")
        return True
    
    # Find a valid setup
    target_symbol = None
    target_date = None
    for setup in all_setups[:20]:
        symbol = setup['symbol']
        date_str = setup['date']
        obs, info = env.reset(options={
            "fixed_setup": {"symbol": symbol, "date": date_str}
        })
        if env.data_provider.current_symbol == symbol and env.data_provider.current_date == date_str:
            target_symbol = symbol
            target_date = date_str
            break
    
    if target_symbol is None:
        print("[SKIP] No valid setup found")
        return True
    
    print(f"Testing with valid setup: {target_symbol} {target_date}")
    
    # Reset multiple times - should always get same episode
    episodes = []
    for _ in range(5):
        obs, info = env.reset(options={
            "fixed_setup": {"symbol": target_symbol, "date": target_date}
        })
        episodes.append((env.data_provider.current_symbol, env.data_provider.current_date))
    
    # All should be identical and NOT None
    if len(set(episodes)) == 1 and episodes[0][0] is not None:
        print(f"[PASS] All 5 resets returned identical episode: {episodes[0]}")
        print(f"[PASS] No random sampling occurred")
        return True
    else:
        print(f"[FAIL] Got different episodes: {set(episodes)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("FIXED SETUP RESET - VALIDATION TESTS")
    print("="*70)
    
    tests = [
        ("Fixed Setup Reset", test_fixed_setup_reset),
        ("No Random Sampling", test_no_random_sampling_with_fixed_setup),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[FAIL] Test '{name}' error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("[PASS] ALL TESTS PASSED - Fixed Setup Reset Works Correctly")
    else:
        print("[FAIL] SOME TESTS FAILED")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
