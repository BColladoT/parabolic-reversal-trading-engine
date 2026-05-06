#!/bin/bash
# Walk-Forward Optimization Training Script
# Execute in WSL2 from /mnt/c/quant_trading

cd /mnt/c/quant_trading

# Activate virtual environment
source venv_wsl/bin/activate

# Set Python path to include src directory
export PYTHONPATH=/mnt/c/quant_trading/src:$PYTHONPATH

# Clean old cache to ensure fresh data provider initialization
rm -f src/scripts/data/cache/hybrid_index.pkl

# Run WFO training with quick test (1 fold, ~1-2 hours)
# For full training use: python src/scripts/train_wfo.py
echo "Starting WFO Quick Test Training..."
echo "This will run 1 fold with 20,000 timesteps"
echo "Expected duration: 1-2 hours"
echo "=========================================="

python src/scripts/train_wfo_quick_test.py 2>&1 | tee logs/wfo_training_$(date +%Y%m%d_%H%M%S).log

echo "=========================================="
echo "Training complete. Check logs/ directory for results."
