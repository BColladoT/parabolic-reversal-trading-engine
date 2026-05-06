"""
Auto-start backtest without confirmation
"""
import subprocess
import sys
from pathlib import Path

print("Starting comprehensive backtest in background...")
print("This will run for 2-3 hours")
print("\nTo monitor progress, run in another terminal:")
print("  python monitor_backtest.py")
print("\nTo view results when complete:")
print("  python generate_comparison_report.py")
print("  start reports/comparison_report.html")

# Run with auto-yes
cmd = [sys.executable, "run_comprehensive_backtest.py"]

# Create a simple input wrapper
import runpy
import io
import contextlib

# Monkey-patch input to auto-confirm
original_input = input
def auto_input(prompt):
    print(f"{prompt} yes")
    return "yes"

# Run the backtest
import run_comprehensive_backtest
