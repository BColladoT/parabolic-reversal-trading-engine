# MFE-Evaporation Penalty — Smoke Results (2026-05-18)

**Headline:** Penalty fires correctly but agent doesn't act on it at 25K steps. Behavior is statistically indistinguishable from scale-out-without-penalty (PR #11). **Either the penalty is too weak to compete with the rest of the reward at this training budget, or — more likely — the convergence problem flagged in PR #11 is the binding constraint.**

## Three-way comparison

```
                              BASELINE      SCALE-OUT     MFE-EVAP
                              (PR #9 sweep) (PR #11)      (this PR, penalty=0.5)
─────────────────────────────────────────────────────────────────────────────
test_pnl per seed (seeds 1/2/3)
    seed 1                    -2733         -2300         -2387
    seed 2                    -2610         -3033         -2851
    seed 3                    -1899         -2105         -2110
  mean                        -2414         -2479         -2449

trade-level (pooled across 3 seeds)
    n_trades                  2900          329           327
    win_rate                  55%           71%           68%
    mean_pnl/trade            -$2.62        -$0.80        -$1.46
    mean_winner               +$32.47       +$0.87        +$0.33   ← shrunk further
    mean_loser                -$45.33       -$4.83        -$5.31
    median_bars_held          2             259           258
    mean_bars_held            6.3           243.8         243.0
    mfe_capture               79%           1.0%          0.4%     ← actually worse
    expectancy_ratio          0.72          0.18          0.06     ← worse
```

## Reading

The MFE-evap penalty produced an identical behavioral pattern to scale-out alone:
- Same hold lengths (258 vs 259 bars)
- Same trade count (327 vs 329)
- MFE_capture didn't climb — it's still ~0%
- Mean winner shrunk further ($0.87 → $0.33)

**The penalty IS firing** (we can confirm from the formula: `-0.5 × evaporation_fraction` per step). But the agent's behavior hasn't changed. Two interpretations:

1. **Convergence-limited.** Per PR #11, the scale-out fix introduces a NEW skill the policy has to learn ("when to trim profitably"). At 25K steps, the agent hadn't converged on scale-out alone. Adding a second signal (the penalty) on top requires *more* convergence time, not less. The penalty is correct, the budget is too small.

2. **Signal-too-weak.** Penalty max is 0.5/step, calibrated to balance hold_discipline. Maybe the agent treats the penalty as noise relative to the much larger base equity reward (which is `equity_delta * 1000 / initial_capital` — can be many points/step on price moves). The penalty signal gets averaged into the gradient but doesn't dominate enough to change the policy quickly.

Both interpretations point to the same next experiment: **longer training**. With more steps:
- If interpretation (1) is right, the penalty will start to take effect as the policy converges
- If (2) is right, longer training won't help much — we'd need to crank the penalty higher

## What we ruled OUT

- The penalty isn't making things meaningfully *worse* (test_pnl within $30 of scale-out alone). It's not actively breaking anything.
- The implementation is correct (10 unit tests green, the formula does what we want).
- The signal is reaching the agent (the per-step rewards include the penalty term in `shaping_total`).

So this isn't a code bug — it's a learning-budget question.

## Decision

**Keep the change.** It's now available behind a CLI flag (`--mfe-evap-penalty 0.5`), backward-compatible (default 0.0). Future longer-training experiments can opt in. The change is composable with scale-out and the R-multiple reward weight.

## Recommended next moves (NOT in this PR)

In order of leverage:

1. **75K-step run with `--mfe-evap-penalty 0.5`** — the obvious next test. Same env, same penalty, 3x training. ~1.5 hours per seed × 3 = ~4.5 GPU-hours. Verifies whether convergence was the limit.

2. **Sweep penalty magnitudes** at 25K steps: `{0.0, 0.5, 1.0, 2.0}` × 3 seeds = 12 runs, ~3 hours. Tests whether a louder signal moves behavior faster.

3. **Both together** (3x training × 2 penalty magnitudes) — ~9 GPU-hours total. Most informative but largest commitment.

If forced to pick one: (1) first. The PR #11 diagnosis already said scale-out needed longer training to converge. Validating that with the penalty added is the most direct test of the "convergence-limited" hypothesis.
