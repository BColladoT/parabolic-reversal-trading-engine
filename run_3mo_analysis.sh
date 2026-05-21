#!/usr/bin/env bash
# Runs after the 3 PPO seeds complete. Does aggregation, rule baseline,
# and loss attribution on the new 3-month window. One-off.
set -e
cd /c/quant_trading
PY="venv_ray310/Scripts/python.exe"
mkdir -p logs reports

echo "===== 1/3 Aggregate 3-seed Discrete PPO results ====="
$PY src/scripts/aggregate_3mo_baseline.py 2>&1 | tee logs/3mo_aggregate.log

echo "===== 2/3 Phase 0 rule baseline on new window ====="
# Use seed 42's quick_test_results.json to define the test setups list
$PY src/scripts/compare_rl_vs_rule.py \
  --rl-results models/ppo_discrete_3mo_s42/quick_test_results.json \
                models/ppo_discrete_3mo_s43/quick_test_results.json \
                models/ppo_discrete_3mo_s44/quick_test_results.json \
  --run-baseline \
  --output reports/rl_vs_rule_baseline_2026-05-21-3mo.json 2>&1 | tee logs/3mo_rule_baseline.log

echo "===== 3/3 Loss attribution analysis on new window ====="
$PY -c "
import json
from pathlib import Path
import numpy as np

# Load 3 seed results
seeds = {}
for seed in (42, 43, 44):
    p = Path(f'models/ppo_discrete_3mo_s{seed}/quick_test_results.json')
    if not p.exists():
        print(f'WARN missing {p}')
        continue
    r = json.loads(p.read_text())
    folds = r.get('folds') or []
    fm = folds[0] if folds else r.get('fold_metrics', r)
    per_ep = fm.get('per_episode_results', [])
    seeds[seed] = {(e.get('symbol'), e.get('date')): e.get('pnl') for e in per_ep}

# Per-setup average across seeds
all_keys = set().union(*[set(d.keys()) for d in seeds.values()])
per_setup_mean = []
for k in all_keys:
    pnls = [seeds[s].get(k) for s in seeds if k in seeds[s]]
    if pnls:
        per_setup_mean.append({'setup': k, 'mean_pnl': float(np.mean(pnls)), 'n_seeds_seen': len(pnls), 'per_seed': pnls})

per_setup_mean.sort(key=lambda x: x['mean_pnl'])

# Headline stats
all_pnls = [s['mean_pnl'] for s in per_setup_mean]
total = sum(all_pnls)
abs_total = sum(abs(p) for p in all_pnls)

# Loss concentration
losers = sorted([s for s in per_setup_mean if s['mean_pnl'] < 0], key=lambda x: x['mean_pnl'])
total_loss = sum(s['mean_pnl'] for s in losers)
top3_loss = sum(s['mean_pnl'] for s in losers[:3]) if len(losers) >= 3 else None

print(f'\\n  Setups (n={len(per_setup_mean)}, total pnl=\${total:.0f}, abs sum=\${abs_total:.0f})')
print(f'  Top-3 worst losers contribute \${top3_loss:.0f} of \${total_loss:.0f} total loss ({(top3_loss/total_loss*100 if total_loss else 0):.1f}% of losses)' if top3_loss else '  No 3-worst computation possible')
print(f'\\n  Worst 5 setups (mean PnL):')
for s in losers[:5]:
    print(f'    {s[\"setup\"][0]} {s[\"setup\"][1]}: \${s[\"mean_pnl\"]:.0f} (seeds: {[round(p,0) for p in s[\"per_seed\"]]})')

# MDE estimate
n = len(per_setup_mean)
std_per_setup = float(np.std(all_pnls, ddof=1)) if n >= 2 else None
mde_per_setup = 2.776 * std_per_setup / (n ** 0.5) if std_per_setup else None
mde_total = mde_per_setup * n if mde_per_setup else None
print(f'\\n  Per-setup std: \${std_per_setup:.0f}')
print(f'  MDE per setup at alpha=0.05 (paired): \${mde_per_setup:.0f}')
print(f'  MDE total: \${mde_total:.0f}')

# Save for doc
Path('reports/loss_attribution_3mo.json').write_text(json.dumps({
    'n_setups': n,
    'total_pnl': total,
    'std_per_setup': std_per_setup,
    'mde_per_setup': mde_per_setup,
    'mde_total': mde_total,
    'worst_5': losers[:5],
    'all_setups_ranked': per_setup_mean,
}, indent=2, default=str))
print(f'\\n  Wrote: reports/loss_attribution_3mo.json')
" 2>&1 | tee logs/3mo_loss_attribution.log

echo "ALL 3 ANALYSIS STEPS COMPLETE"
