#!/bin/bash
cd /mnt/c/quant_trading
source venv_wsl/bin/activate
rm -f src/scripts/data/cache/hybrid_index.pkl
PYTHONPATH=/mnt/c/quant_trading/src:$PYTHONPATH timeout 1800 python src/scripts/train_wfo_quick_test.py
