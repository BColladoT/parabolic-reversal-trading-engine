#!/usr/bin/env bash
# Phase 6-style 10-seed validation of the Q4 RL result.
# Tests whether +$1,958 holds with proper sample size.
set -e
cd /c/quant_trading
PY="venv_ray310/Scripts/python.exe"
ARGS="--algo ppo --action-space discrete --total-steps 25000 --test-start-date 2024-10-01 --test-end-date 2024-12-30"
mkdir -p logs models/ppo_q4_10seed
for SEED in 42 43 44 45 46 47 48 49 50 51; do
  if [ -f "models/ppo_q4_10seed/s${SEED}/quick_test_results.json" ]; then
    echo "===== seed=${SEED} ALREADY DONE, skipping ====="
    continue
  fi
  echo "===== seed=${SEED} start $(date +%H:%M:%S) ====="
  $PY src/scripts/train_wfo_quick_test.py $ARGS --seed $SEED --output-dir models/ppo_q4_10seed/s${SEED} 2>&1 | tee logs/ppo_q4_10seed_s${SEED}.log
  echo "===== seed=${SEED} done $(date +%H:%M:%S) ====="
done
echo "Q4 10-SEED VALIDATION COMPLETE"
