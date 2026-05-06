#!/bin/bash
cd /mnt/c/quant_trading
source venv_wsl/bin/activate
export PYTHONPATH=/mnt/c/quant_trading/src
setsid python -u src/scripts/train_wfo.py > /tmp/wfo_full_train.log 2>&1 &
echo "PID: $!"
disown $!
