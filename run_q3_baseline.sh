#!/usr/bin/env bash
# Multi-window generalization check: 3 Discrete PPO seeds on 2024 Q3.
set -e
cd /c/quant_trading
PY="venv_ray310/Scripts/python.exe"
ARGS="--algo ppo --action-space discrete --total-steps 25000 --test-start-date 2024-07-01 --test-end-date 2024-09-30"
mkdir -p logs
for SEED in 42 43 44; do
  echo "===== seed=$SEED start $(date +%H:%M:%S) ====="
  $PY src/scripts/train_wfo_quick_test.py $ARGS --seed $SEED --output-dir models/ppo_discrete_q3_s$SEED 2>&1 | tee logs/ppo_discrete_q3_s${SEED}.log
  echo "===== seed=$SEED done $(date +%H:%M:%S) ====="
done
echo "Q3 3-SEED BASELINE COMPLETE"
