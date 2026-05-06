#!/usr/bin/env python3
"""
Parabolic Reversal Trading Engine - Launcher
Simple entry point for running the trading system.
"""
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main_engine import main

if __name__ == "__main__":
    main()
