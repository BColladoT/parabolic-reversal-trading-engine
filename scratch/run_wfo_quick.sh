#!/bin/bash
cd /mnt/c/quant_trading
source venv_wsl/bin/activate
export PYTHONPATH=/mnt/c/quant_trading/src
python src/scripts/train_wfo_quick_test.py > /tmp/wfo_quicktest.log 2>&1
echo "EXIT_CODE: $?" >> /tmp/wfo_quicktest.log
