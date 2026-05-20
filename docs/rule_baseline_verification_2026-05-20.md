# Rule Baseline Verification — Same OOS as Discrete PPO (2026-05-20)

**Headline:** **Rule baseline OOS test_pnl = -$2,160.19** (vs Discrete PPO -$1,609.58 mean across 3 seeds). The rule baseline LOSES money on this OOS test set and is **worse than Discrete PPO by $550.61**. The previously cited "+2.19 R / 78% win rate" rule-baseline number is **refuted** on this exact test window — observed win rate is 35.7% with a $251 negative R-multiple equivalent (mean trade PnL / |mean loser| = -$166 / $302 = -0.55 R). Phase 0 of the RL tuning plan now has a verified benchmark that is itself in the red, which materially changes the decision gate downstream: "beat the rule baseline" is no longer "beat a +2.19 R reference"; it is "lose less than -$2,160 on this 14-setup OOS window."

---

## Test setup window

- **Source:** `models/ppo_discrete_test/w0.00_s{1,2,3}/quick_test_results.json` (PR #17 Discrete(7) PPO).
- **Window:** 2024-11-30 → 2024-12-30 (1-month OOS test, fold 1 of quick-test WFO).
- **Setups:** 14 (symbol, date) pairs, identical across all 3 PPO seeds; the rule baseline is deterministic and runs once on the shared list.
- **Setups list:** PRFX 2024-12-19, SES 2024-12-26, GGRP 2024-12-18, PLRZ 2024-12-18, INTZ 2024-12-30, BAOS 2024-12-24, DFNS 2024-12-17, GXAI 2024-12-06, CTEV 2024-12-24, GXAI 2024-12-10, WKEY 2024-12-13, PPBT 2024-12-02, SNTI 2024-12-02, HOUR 2024-12-24.
- **Cost model:** 30 bps/leg transaction cost (matches `src/rl/env.py` — see `src/baselines/rule_baseline.run_baseline`).
- **Evaluation harness:** `src/baselines/evaluate_baseline.py::evaluate_baseline_on_fold` — same `ParabolicReversalEnv` instance, same `fixed_setup` reset path, same accounting as the RL eval loop in `train_wfo_quick_test.py`.

## Headline comparison table

| Metric | Rule baseline (V5 Relaxed, fixed shares) | Discrete PPO (3-seed mean) |
|---|---:|---:|
| total_test_pnl | **-$2,160.19** | -$1,609.58 |
| win_rate | 35.7% (5W / 8L / 1 zero-trade) | 21.4% (3W / 11L) |
| mean_winner | +$51.18 | (categorical head — not directly comparable) |
| mean_loser | -$302.01 | (see `algorithm_comparison_smoke_2026-05-19.md`) |
| n_trades total | 13 | ~14 (1 per setup, 1 setup is zero-trade) |
| n_setups evaluated | 14/14 | 14/14 |
| mean_trades_per_episode | 0.93 | 0.93 |

(`fixed_fraction_of_equity` sizing gave identical aggregate numbers — same `-$2,160.19` total — because in this 1-month window each rule-baseline episode opened at most one position before VWAP-reversion exit or 15:25 flatten, so both sizing modes resolve to the same trade-level outcomes.)

## Per-setup PnL: rule baseline vs RL mean

```
Symbol   Date          Rule PnL    RL mean PnL    rule - rl
─────────────────────────────────────────────────────────────
PRFX     2024-12-19      +73.34         -72.84      +146.18
SES      2024-12-26       +2.19         -35.35       +37.54
GGRP     2024-12-18      -23.84          -7.92       -15.91
PLRZ     2024-12-18       +0.00          +0.00        +0.00
INTZ     2024-12-30     +134.34         +46.84       +87.50
BAOS     2024-12-24     -236.38          +2.54      -238.91
DFNS     2024-12-17   -1,330.10        -816.57      -513.53
GXAI     2024-12-06      +20.78         +39.64       -18.86
CTEV     2024-12-24     -310.92        -147.00      -163.93
GXAI     2024-12-10      +25.26         -26.55       +51.81
WKEY     2024-12-13     -158.43        -156.17        -2.26
PPBT     2024-12-02     -214.65        -221.99        +7.33
SNTI     2024-12-02      -37.37         -56.00       +18.63
HOUR     2024-12-24     -104.41        -158.21       +53.80
─────────────────────────────────────────────────────────────
TOTAL                 -2,160.19      -1,609.58      -550.61
```

Rule baseline wins on 7 setups, RL wins on 6, 1 tie. DFNS 2024-12-17 is the dominant loser for both (-$1,330 rule, -$817 RL) — that one setup explains -$514 of the $551 gap. Without DFNS, totals are rule -$830 vs RL -$793 — essentially even.

## Does this support or refute the "+2.19 R / 78%" prior claim?

**Refutes — clearly and on this exact OOS window.** The +2.19 R-multiple / 78% win-rate claim was a number that lived only in user conversation, never in a doc or test, and was the implicit benchmark Phase 0 was meant to verify. On the same 14-setup OOS window that PR #17 trained Discrete PPO against, the V5 Relaxed rule (entry VWAP-dev > 20%, exit at VWAP touch or 4% stop, 30 bps/leg costs, 100 fixed shares) wins only 5 of 14 setups (35.7%) and posts a mean per-trade R-multiple of approximately **-0.55 R** (mean_episode_pnl -$154 against mean_loser magnitude $302). The previously cited +2.19 R was probably computed on the much broader 909-setup historical winners CSV (`reports/relaxed_909_backtest.csv`) without the 30 bps/leg cost charge that the RL environment imposes — that is a different evaluation regime than the WFO OOS slice and should not be used as the benchmark RL has to beat. Phase 0 closes with an unambiguous and uncomfortable result: **the rule baseline is unprofitable on this WFO OOS window. The RL beat-the-rule gate is now "lose less than $2,160," not "beat a profitable baseline."** That has direct implications for Phase 6 of the tuning plan (decision gate): even if Discrete PPO never reaches profitability, it has already beaten the rule by $551 on this window, which makes the "pivot to rule paper trading" branch of the decision gate substantially less attractive than the user's prior framing assumed. The DFNS 2024-12-17 episode (-$1,330 rule, -$817 RL) is doing a disproportionate amount of the damage in both directions; the decision gate should explicitly examine robustness to that single setup.

## Artifacts

- Output JSON: `reports/rl_vs_rule_baseline_2026-05-20.json`
  - `summary_fixed_shares.rl.total_test_pnl` = -$1,609.58
  - `summary_fixed_shares.rule_baseline.total_test_pnl` = -$2,160.19
  - `summary_fixed_shares.rule_baseline.per_setup_pnls` — full 14-element list for Phase 6 per-setup breakdown
  - `summary_fixed_shares.delta.rule_minus_rl_total_pnl` = -$550.61 (negative means rule loses MORE than RL)

## What changed in `compare_rl_vs_rule.py`

- Added a `_extract_test_setups` helper that reads `per_episode_results` from either the quick-test JSON shape (fold-level `per_episode_results`) or the full WFO JSON shape (`fold['test_metrics']['per_episode_results']`). The previous code only handled the latter, so it silently returned zero setups against `quick_test_results.json` — meaning the script had **never been executed end-to-end against the Discrete PPO results** until this run.
- Added `--rl-results` comma-separated support so the same invocation can average across 3 PPO seeds (each contributes one fold-1 result; rule baseline runs once on the shared deterministic setup list).
- Added a top-level `summary_fixed_shares` / `summary_fraction_of_equity` block with `rl.total_test_pnl`, `rule_baseline.total_test_pnl`, `delta.rule_minus_rl_total_pnl`, `rule_baseline.mean_winner` / `mean_loser`, `rule_baseline.per_setup_pnls`, and the (symbol, date) setup list.
- Added `import numpy as np` at module top (the existing `run_statistical_benchmarks` used `np` without importing it — latent crash if anyone called it).

No changes to `src/baselines/rule_baseline.py`, `src/baselines/evaluate_baseline.py`, `src/rl/env.py`, `src/utils/metrics.py`, or `src/scripts/train_wfo_quick_test.py` (the env_config allowlist fix in `src/rl/env.py:376-379` is untouched).
