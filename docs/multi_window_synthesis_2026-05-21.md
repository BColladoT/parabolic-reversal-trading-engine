# Multi-Window OOS Generalization Study — Final Synthesis (2026-05-21)

## Verdict

**CONTINUE_RESEARCH.** Discrete(7) PPO at 25K steps produced a strikingly profitable result on Q4 2024 but does NOT generalize across windows. The investigation's MIT-level conclusion: a single profitable OOS window is not evidence the policy works; reproducibility across multiple windows is the bar.

## Results across three OOS windows

| Window | Range | n_setups | seed42 | seed43 | seed44 | Mean | Std | Win rate | Action dist (top) |
|---|---|---|---|---|---|---|---|---|---|
| **Q4 2024** | 2024-10-01 → 12-30 | 50 | +$1,980 | +$1,914 | +$1,980 | **+$1,958** | **$38** | 40% | bin 2 (ENTRY-50%) 63% |
| **Q3 2024** | 2024-07-01 → 09-30 | 50 | +$379 | -$288 | -$354 | **-$88** | **$405** | 14-28% | bin 1 (ENTRY-100%) 93% |
| **H1 2024** | 2024-01-01 → 06-30 | 50 | +$149 | +$149 | +$149 | **+$149** | $0 | n/a (0 trades) | bin 1 67%, bin 0 30% |

All runs: Discrete(7) PPO, 25K steps, default hyperparameters, 6-month preceding train window.

## Per-window interpretation

### Q4 2024 — "Looks great"
- All 3 seeds converge to the same trades (2 of 3 seeds = $1,979.72 to the cent)
- Bin 2 (ENTRY-50%) dominates; agent rarely COVERs, lets env's stops/EOD-flatten handle exits
- 20 winners / 6 losers / 24 abstentions
- Top 3 winners contribute 60% of wins; top 3 losers contribute 95% of losses
- Outlier-robust: drop NVNI (worst loss) → +$2,401. Drop top 3 winners → +$294 (still positive)

### Q3 2024 — "Doesn't work"
- High seed variance ($405 vs Q4's $38)
- Different action distribution: bin 1 (ENTRY-100%) at 93%, vs Q4's bin 2 at 63%
- Lower win rates: 14% / 26% / 28% (vs Q4's stable 40%)
- Mean essentially zero (-$88) with one seed barely positive (+$379), two negative
- The policy is unstable here — different seeds learn meaningfully different strategies

### H1 2024 — "Agent couldn't trade"
- **Every episode has trades=0**
- Action distribution shows agent issued ENTRY actions (67% bin 1, 3% bin 2), but the action mask blocked them
- The mask gates entries when `vwap_deviation < min_vwap_deviation_entry` (default 15.0). H1 setups largely didn't qualify
- The $149 "PnL" is mark-to-market noise from position holding/closing, not from agent trade decisions
- 25 of 50 episodes have exactly zero PnL — confirms the no-trade pattern

## Why Q4 worked and Q3 didn't (hypotheses)

1. **Different training data per window.** Each OOS run uses the 6 months preceding `test_start` as training. Q4's train was 2024-03-30 → 09-26; Q3's train was 2024-01-04 → 07-01. The Q4 train window includes the H1 setups Q3 trained on, but also includes Q2 (which Q3 doesn't see). Different train data → different policy.
2. **Different volatility regime.** Q3 2024 includes the August 2024 sell-off (a regime shift); Q4 includes the post-election rally. Parabolic-reversal strategies behave differently across these regimes.
3. **The action mask filters different setups.** Q3 may have setups closer to the entry threshold; Q4's setups may be more extreme. Causes the policy to engage on different proportions of setups.
4. **Overfit hypothesis.** The Q4 train window ends just before Q4 OOS starts. There may be subtle leakage or learned correlations to recent market state that don't carry into Q3.

## What this means for the tuning plan

The original 6-phase plan was designed around "push Discrete PPO from -$1,610 toward break-even." That premise is gone — RL is sometimes profitable, sometimes not. The new problem is **generalization, not optimization**.

### Recommended next moves (in priority order)

1. **Run more seeds per window** to characterize variance properly. 3 seeds is too few for high-variance windows like Q3. Suggest 10 seeds per window. ~5 hr GPU per window.

2. **Train on a longer history.** Currently 6-month train; try 12 or 24 months. More training data should reduce regime sensitivity. Validate the same multi-window protocol.

3. **Symbol/regime conditioning features** in the observation. Currently the env's 10 explicit features are mostly price-action; adding regime/volatility-of-volatility features could help the policy distinguish regimes.

4. **Recurrent PPO (LSTM)** — see `docs/recurrent_ppo_scoping_2026-05-20.md`. Trade-state memory may help, but is unlikely to fix a generalization failure rooted in data shift.

5. **Walk-forward with rolling re-train.** Instead of evaluating one model on multiple OOS windows, retrain at each window's boundary. This is the standard quant approach. Real-world deployment would do this.

### What NOT to do

- Don't run Phase 3 (budget sweep) on the Q4 window alone — it'll just optimize for Q4-specific features.
- Don't run Phase 4 (bin sweep) — the action distribution shifted across windows, the bin layout isn't the bottleneck.
- Don't run Phase 5 (HP sweep) on Q4 alone — same overfitting concern.
- Don't paper-trade based on the Q4 result.

## Rule baseline parallel finding

The rule baseline on the same new 3-month Q4 window: **+$1,666** (18% win rate, mean_winner $297, mean_loser -$168). Also profitable on Q4. Worth running rule on Q3 and H1 to see if rule's generalization is better/worse than RL's. Cheap (~10 min CPU each).

If rule generalizes consistently across windows but RL doesn't, that argues for paper-trading the rule, not the RL.

## Numerical artifacts

- `models/ppo_discrete_3mo_s{42,43,44}/` — Q4 raw outputs
- `models/ppo_discrete_q3_s{42,43,44}/`  — Q3 raw outputs
- `models/ppo_discrete_h1_s{42,43,44}/`  — H1 raw outputs
- `reports/ppo_discrete_3mo_3seed_summary.json` — Q4 aggregate
- `reports/rl_vs_rule_baseline_2026-05-21-3mo.json` — Q4 RL vs rule comparison
- `reports/loss_attribution_3mo.json` — Q4 per-setup attribution

## Honest assessment

The investigation produced a real but narrow finding:
- ✓ The methodology fix (14 → 50 setups) was correct and necessary
- ✓ Discrete PPO CAN be profitable on this domain (Q4 demonstrated)
- ✗ The current architecture/training doesn't generalize across regimes
- ✗ The original "push to break-even" tuning plan is the wrong frame

The work product is a much sharper diagnostic than where we started. We now know:
1. The OOS evaluation protocol that's actually statistically discriminating (50+ setups, multi-window)
2. That the model has a real signal on at least one window
3. That the model fails to generalize, with clear failure modes (regime change, action mask gating)

This is closer to the truth than -$1,610 was, and points to the right next experiments. But "SHIP_RL" is premature.
