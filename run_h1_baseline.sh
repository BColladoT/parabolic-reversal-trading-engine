#!/usr/bin/env bash
# Third data point for window-generalization study: 2024 H1 (Jan-Jun).
set -e
cd /c/quant_trading
PY="venv_ray310/Scripts/python.exe"
ARGS="--algo ppo --action-space discrete --total-steps 25000 --test-start-date 2024-01-01 --test-end-date 2024-06-30"
mkdir -p logs
for SEED in 42 43 44; do
  echo "===== seed=$SEED start $(date +%H:%M:%S) ====="
  $PY src/scripts/train_wfo_quick_test.py $ARGS --seed $SEED --output-dir models/ppo_discrete_h1_s$SEED 2>&1 | tee logs/ppo_discrete_h1_s${SEED}.log
  echo "===== seed=$SEED done $(date +%H:%M:%S) ====="
done
echo "H1 3-SEED BASELINE COMPLETE"
