#!/usr/bin/env bash
# Test whether 12-month training window reduces overfit / improves generalization.
# 3 seeds * 3 OOS windows = 9 runs.
set -e
cd /c/quant_trading
PY="venv_ray310/Scripts/python.exe"
ARGS_BASE="--algo ppo --action-space discrete --total-steps 25000 --train-months 12"
mkdir -p logs

# Define (window_name, test_start, test_end) tuples
declare -a WINDOWS=(
  "q4 2024-10-01 2024-12-30"
  "q3 2024-07-01 2024-09-30"
  "h1 2024-01-01 2024-06-30"
)

for W in "${WINDOWS[@]}"; do
  read -ra PARTS <<<"$W"
  NAME=${PARTS[0]}; TS=${PARTS[1]}; TE=${PARTS[2]}
  for SEED in 42 43 44; do
    echo "===== ${NAME} seed=${SEED} (train=12mo, test=${TS} -> ${TE}) start $(date +%H:%M:%S) ====="
    $PY src/scripts/train_wfo_quick_test.py $ARGS_BASE \
      --test-start-date $TS --test-end-date $TE \
      --seed $SEED --output-dir models/ppo_12mo_${NAME}_s${SEED} 2>&1 \
      | tee logs/ppo_12mo_${NAME}_s${SEED}.log
    echo "===== ${NAME} seed=${SEED} done $(date +%H:%M:%S) ====="
  done
done
echo "12-MONTH SWEEP COMPLETE (9 runs)"
