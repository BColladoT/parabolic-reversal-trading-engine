"""
Complete Backtest Comparison - Master Orchestrator

Runs the full comparison backtest with monitoring and report generation.

Usage:
    python run_complete_comparison.py [--limit N] [--quick-test]
    
Options:
    --limit N      Limit to N setups (for testing)
    --quick-test   Run quick test on 50 setups
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def run_backtest(limit=None):
    """Run the backtest comparison."""
    print("="*80)
    print("STEP 1: RUNNING BACKTEST")
    print("="*80)
    
    cmd = [sys.executable, "run_full_comparison_backtest.py"]
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_monitor():
    """Run the monitor in parallel."""
    print("\n" + "="*80)
    print("STEP 2: STARTING MONITOR (in parallel)")
    print("="*80)
    print("The monitor will run alongside the backtest.")
    print("Press Ctrl+C in the monitor window to stop monitoring.")
    print("\nTo monitor separately, run: python monitor_backtest.py")
    print("="*80 + "\n")


def generate_report():
    """Generate the comparison report."""
    print("\n" + "="*80)
    print("STEP 3: GENERATING REPORT")
    print("="*80)
    
    result = subprocess.run([sys.executable, "generate_comparison_report.py"])
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Complete comparison backtest')
    parser.add_argument('--limit', type=int, default=None, help='Limit setups')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode')
    parser.add_argument('--skip-backtest', action='store_true', help='Skip backtest, just generate report')
    args = parser.parse_args()
    
    print("="*80)
    print("COMPLETE BACKTEST COMPARISON")
    print("V5 Relaxed Scanner vs V5 Institutional ML")
    print("="*80)
    print(f"Start Time: {datetime.now()}")
    
    if args.quick_test:
        args.limit = 50
        print("\n[QUICK TEST MODE] Running on 50 setups only\n")
    
    # Step 1: Run backtest
    if not args.skip_backtest:
        print("\nStarting backtest...")
        print("This will take approximately 1-2 minutes per 50 setups")
        print(f"Estimated time: {(args.limit or 909) * 2 / 60:.0f} minutes\n")
        
        success = run_backtest(args.limit)
        
        if not success:
            print("\n[ERROR] Backtest failed!")
            return 1
    else:
        print("\n[SKIP] Skipping backtest, using existing results")
    
    # Step 2: Generate report
    print("\nGenerating report...")
    success = generate_report()
    
    if not success:
        print("\n[ERROR] Report generation failed!")
        return 1
    
    # Final summary
    print("\n" + "="*80)
    print("ALL STEPS COMPLETE!")
    print("="*80)
    print("\nResults available in:")
    print("  - reports/comparison_backtest_results.csv (raw data)")
    print("  - reports/comparison_summary.json (summary stats)")
    print("  - reports/comparison_report.html (interactive dashboard)")
    print("  - reports/comparison_charts/ (individual charts)")
    print("\nOpen reports/comparison_report.html in your browser to view results.")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
