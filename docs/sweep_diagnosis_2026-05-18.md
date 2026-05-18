# RL Agent Sweep Diagnosis — 2026-05-18

**Question we set out to answer:** Would a longer training run (100K+ steps, 3-6 GPU-days) help, or are we hitting a structural problem the training budget can't fix?

**Short answer:** Structural problem. Three independent diagnostics — pooled across 11,650 trades, 63 training iterations × 12 runs, and 5,059 mask-violation events — all point to **issues that more training will not fix and one that more training actively makes worse.**

Source data: `models/sweep_2026-05-18/` (the 4-weight × 3-seed × 25K-step sweep from PR #9).

---

## Executive summary

1. **The agent has structural negative expectancy** (mean winner / |mean loser| ≈ **0.73** across all weights). Winners are 27% smaller than losers. Even at the observed 54% win rate, this loses money: `0.54 × $33 − 0.46 × $47 ≈ −$3.81/trade`. The trade-size distribution itself is the bug, and it's identical across all reward weights.

2. **Mask violations RISE through training** (1.49 → 2.79 per episode, all 12 runs). The Gaussian policy gets more confident over training and pushes *harder* into invalid action regions. More steps = more violations = noisier gradient, not less.

3. **Training was still improving at cutoff** (median late-quartile slope **+0.078**), with **low across-seed variance** (max stddev 1.76). Convergence is reproducible. But the *thing* it's converging to has a -$3/trade expectancy — converging better doesn't make a -EV strategy profitable.

**Recommendation: Do not run the 100K-step sweep yet.** Spend the budget on (a) understanding *why* losers run deep (-$73 mean MAE vs only -$47 final pnl) and (b) re-thinking the mask architecture so violation rate *decreases* with training. After those fixes, a longer sweep becomes worth running.

---

## Detail 1: Trade distribution (A1: `diagnose_trades`)

```
| weight | n_trades | win_rate | mean_pnl  | mean_winner | mean_loser | expectancy_ratio | median_bars |
|--------|---------:|---------:|----------:|------------:|-----------:|-----------------:|------------:|
| 0.00   |    2,900 |    54.9% |   $-2.62  |    $+32.47  |   $-45.33  |             0.72 |           2 |
| 0.10   |    2,832 |    52.4% |   $-4.95  |    $+32.41  |   $-46.13  |             0.70 |           2 |
| 0.20   |    2,902 |    54.6% |   $-1.60  |    $+38.89  |   $-50.33  |             0.77 |           2 |
| 0.50   |    3,016 |    53.5% |   $-3.66  |    $+34.02  |   $-47.03  |             0.72 |           2 |
```

**MFE/MAE asymmetry tells the story:**

```
| weight | winner_MFE | winner_MAE | loser_MFE | loser_MAE  |
|--------|-----------:|-----------:|----------:|-----------:|
| 0.00   |    $+41.01 |    $-18.76 |    $+5.87 |    $-73.88 |
| 0.10   |    $+41.41 |    $-17.09 |    $+5.81 |    $-72.56 |
| 0.20   |    $+48.84 |    $-15.57 |    $+5.51 |    $-76.70 |
| 0.50   |    $+41.68 |    $-19.20 |    $+5.80 |    $-71.61 |
```

What this says:
- **Winners are released early.** Avg winner has MFE ~$42 but is closed at $33 (78% capture). 22% of the move is left on the table.
- **Losers are held until they hemorrhage.** Avg loser has MAE -$73 but is closed at -$47. The agent watches losers dig $73 underwater before cutting — and the final loss is "only" -$47 because some recover partially. This is **the opposite of "cut losers fast, let winners run."**
- **Median hold = 2 bars.** Winners and losers both close fast on average. But losers visit -$73 mid-trade. The exit policy reacts to *something*, just not the right thing.

**Per-symbol scorecard pooled across all weights:**

```
Top 5 (best mean pnl over 5+ trades):
  - SFWL: +$459.74 over 8 trades
  - MPU:  +$90.51  over 7 trades
  - SEV:  +$67.42  over 11 trades
  - RGS:  +$63.66  over 41 trades
  - GCT:  +$54.73  over 62 trades

Bottom 5:
  - GLMD: -$169.95 over 15 trades
  - BETR: -$98.08  over 17 trades
  - MNPR: -$65.47  over 43 trades  <- statistically substantial
  - DCOY: -$54.81  over 8 trades
  - ZJYL: -$49.78  over 16 trades
```

Diagnosis: **the agent has clearly different competence per symbol**, and at least two of the bottom-5 (MNPR n=43, BETR n=17) have enough trades to be more than noise. Worth investigating whether those symbols have characteristics that the policy systematically misreads.

---

## Detail 2: Training convergence (A2: `diagnose_training`)

```
| weight | n_seeds | mean_final | stddev_final | mean_best | mean_late_slope |
|--------|--------:|-----------:|-------------:|----------:|----------------:|
| 0.00   |       3 |     -1.47  |        1.76  |   +26.00  |          -0.095 |
| 0.10   |       3 |     -5.45  |        0.71  |    +0.17  |          +0.128 |
| 0.20   |       3 |     -4.18  |        1.31  |   +11.95  |          +0.064 |
| 0.50   |       3 |     -4.25  |        0.73  |    +3.82  |          +0.093 |
```

Interpretation that fired automatically:
- **Still improving at end of training** (median late-quartile slope +0.078). Cutoff was premature.
- **Low across-seed variance** (max 1.76). Training is reproducible; weight differences are real signal.

But look at `mean_best`: w=0.00 hit a peak of +26 before drifting back to -1.47. **All other weights either never get high (w=0.10) or get only slightly positive (w=0.20, w=0.50).** The baseline (no shaping) bursts higher than any shaped variant but is also the only one with a negative late slope (`-0.095`).

This suggests the shaped reward is *damping* exploration. The baseline finds better policies but can't hold them. The shaped variants find worse policies but hold them more stably. Neither is profitable.

**Single seed call-outs:**
- `w=0.00, seed 2`: best +40.81, final -0.64. Massive bust late.
- `w=0.10, all 3 seeds`: best rewards are -0.37, -0.90, +1.78. Two of three NEVER hit positive in 63 iterations.
- `w=0.50`: all 3 hit positive by iter 7-9 with stable positive late slope. The most "convergent" weight, just to a mediocre policy.

---

## Detail 3: Mask violations (A3: `diagnose_masks`)

```
| weight | n_seeds | mean_per_1000_steps | mean_early | mean_late | learning? |
|--------|--------:|--------------------:|-----------:|----------:|:---------:|
| 0.00   |       3 |              16.91  |       1.42 |     2.66  |    no     |
| 0.10   |       3 |              16.78  |       1.61 |     2.87  |    no     |
| 0.20   |       3 |              16.93  |       1.48 |     2.74  |    no     |
| 0.50   |       3 |              17.10  |       1.45 |     2.90  |    no     |
```

Every single run shows `trend=rise`. The early/late ratios cluster around 1.9× — almost a doubling. **Mask violation rate is roughly proportional to training progress** (in the wrong direction).

Hypothesis: the action mask is enforced as a post-hoc override (`action_type overridden to HOLD`), not as a hard constraint on the policy distribution. The continuous Gaussian policy is unaware of the mask; it learns to be confident, alpha decays, exploration shrinks, the mean of the Gaussian lands in invalid regions more reliably, and the env catches every attempt and overrides. The "mask penalty" reward term (if any) is insufficient to keep the mean of the distribution inside the valid region.

This won't fix itself. Either:
- **(a)** the policy needs to be aware of the mask at distribution-construction time (filtered Gaussian, or discrete-action head); or
- **(b)** the mask penalty needs to be much stronger; or
- **(c)** the masking should be removed and the env should accept invalid actions as costly-but-real (let the policy learn the cost organically).

---

## Recommendations (opinionated)

### 1. Don't run the 100K-step sweep yet
Spending 3-6 GPU-days to converge a strategy with -$3 EV per trade is converging to the wrong destination more precisely. The win rate is already 53-55% — close enough to break-even that small expectancy fixes have large impact, but training alone won't get there.

### 2. Fix the exit policy first (highest-leverage single change)
The `winner_MAE = -$18, loser_MAE = -$73` asymmetry says the agent doesn't know how to cut losers. Two concrete things worth trying:

- **Hard stop at -$30 mid-trade** (somewhere between -$18 winner-MAE and -$73 loser-MAE). Forces early exit on losers without sacrificing the winners that go red first. This is an env change in `src/rl/env.py` — `intra_step_stop_loss` already exists as a CLI flag at -$2000; tighten it.
- **Per-symbol blacklist of the worst-5** (GLMD, BETR, MNPR, DCOY, ZJYL). At n≥15-43 trades each, these aren't noise — the strategy systematically loses on them. Either filter pre-entry or weight down via the trade journal.

Either fix should push expectancy_ratio toward 1.0 and would be visible in a 25K smoke test (no need to spend the GPU-days).

### 3. Re-think the masking before more training
The mask is making training *harder*, not easier. Options:
- **Disable masking** in env.py and retrain at the same step budget. See if the underlying policy is actually trying to do something sensible or if the mask is the only thing preventing total chaos.
- **Switch to a constrained policy** (filtered Gaussian via `MaskedGaussianPolicy` — which already exists in `src/rl/agent.py` per the docstring). The current `MaskedSACRLModule` was deliberately disabled in the PR #8 patches because of new-Ray-API incompatibility — re-enabling it with proper `RLModuleSpec` wrapping might be the right move.

### 4. After (2) and (3): do a 25K-step smoke sweep on the fixes
Cheap (~30 min on GPU). If expectancy_ratio crosses 1.0 and violation rate goes flat-or-down with training, **then** commit to the 100K-step sweep with confidence that you're scaling something that works.

---

## What we did NOT diagnose (out of scope for this batch)

- **Action distribution** (HOLD/ENTER/EXIT %). Not in trades.jsonl. Would require env.py instrumentation. Worth doing if you want to know "is the agent essentially HOLD-ing 95% of steps?"
- **Reward decomposition** (Sortino / drawdown / r_multiple component magnitudes). Same — needs env.py instrumentation.
- **Trade-level comparison to the rule baseline.** We compared aggregate PnL, but not "on the same episode, where did rule enter and where did RL enter?" That comparison would require re-running `rule_baseline.py` on the same test episodes.

Each is its own follow-up. None block the recommendations above.

---

## How to reproduce this report

```
python -m src.scripts.diagnose_sweep models/sweep_2026-05-18/
```

Or against any future sweep:

```
python -m src.scripts.sweep_reward_weights --weights ... --seeds ... --output-root models/sweep_NEW/
python -m src.scripts.diagnose_sweep models/sweep_NEW/
```

The diagnostic modules (`diagnose_trades`, `diagnose_training`, `diagnose_masks`) are read-only and run in seconds against any sweep directory matching the `models/sweep_*/w*_s*/` layout.
