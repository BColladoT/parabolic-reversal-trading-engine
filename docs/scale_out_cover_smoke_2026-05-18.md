# Scale-Out Cover — Smoke Results (2026-05-18)

**Headline:** Fix works **mechanically** — positions no longer killed by policy noise — but exposes a different learning problem the agent didn't have time to solve at 25K steps. Bottom-line PnL is statistically indistinguishable from baseline. Keep the code; need longer training or additional reward shaping to convert the mechanic into PnL.

## Setup

- Branch: `feat/scale-out-cover` (commit `56d6388`)
- 3 seeds × 25K training steps × weight=0.0 (no R-multiple shaping)
- Run dir: `models/scaleout_test/w0.00_s{1,2,3}/`
- Baseline for comparison: the same configuration from PR #9 sweep (`models/sweep_2026-05-18/w0.00_s{1,2,3}/`)

## Test PnL (mean across 3 seeds)

```
                          per-seed PnL              mean
baseline (full-close)    -2733, -2610, -1899      -$2,414
scale-out cover          -2300, -3033, -2105      -$2,479
delta                                              -$65/episode (within noise)
```

Per-seed variance is much higher than the means' difference. The new env is **not statistically distinguishable** from the old one at the bottom-line PnL level.

## Trade-level structure: dramatically different

```
Metric              Baseline      Scale-out     Direction      Reading
──────────────────────────────────────────────────────────────────────
n_trades            2,900         329           -88%           Agent commits to fewer setups
win_rate            54.9%         70.8%         +16pp          Wins are more reliable
median_bars_held    2             259           ~130x          Positions actually hold now
mean_bars_held      6.3           243.8         39x            Same story
mean_winner         +$32          +$0.87        -97%           Winners released near entry
mean_loser          -$45          -$4.83        -89%           Losers don't run as far either
winner_MFE          +$41          +$77          +88%           Bigger profitable excursions
winner_MFE_capture  79%           1.1%          --             Lets nearly all profit slip away
winner_MAE          -$19          -$220         12x            Holds through huge drawdowns
loser_MAE           -$74          -$188         2.5x           Same
expectancy_ratio    0.72          0.18          -75%           Much worse
```

## What this means

The fix is mechanically correct: policy noise no longer fully closes positions. The agent now holds. **But it holds too long, indecisively.**

The picture is consistent across all 3 seeds:
- Agent enters a setup
- Trims tiny fractions throughout (matches scale-out semantics)
- Position swings widely (MFE +$77, MAE -$220 on winners)
- Eventually exits at near-zero PnL (1.1% MFE capture)

The agent went from "scared of holding (closes too fast)" to "afraid to commit to exit (drifts through MFE peaks)." Both are failure modes of an under-trained policy. **Neither is a failure of the env change.**

Two contributing factors, in order of likely impact:

1. **Training-budget limited.** Per PR #10 training-curve diagnosis, the median late-quartile slope at 25K steps was already `+0.078` (still improving). The scale-out change introduces a *new* skill the policy has to learn: "when to commit to a trim vs hold." 25K steps wasn't enough to converge the original policy; with a behavioral change this dramatic, it's certainly not enough.

2. **Reward shaping mismatch.** The reward function rewards holding profitable shorts (`hold_discipline` +0.1-0.5/step, `participation_bonus` +0.15/step) but doesn't penalize letting MFE evaporate. The agent learns "holding pays" but has no signal pushing toward "exit at peak." The `trade_completion_bonus` includes an MFE-capture term but only fires once per trade — dominated by the per-step holding rewards for any long hold.

## Decision

**Keep the change.** It's diagnostically valuable (proves the original problem was real) and unblocks future fixes. Without it, no amount of reward shaping or training time would have produced the >2-bar holds the strategy needs.

**Don't conclude scale-out is a failure based on this smoke.** The test ran for 25K steps on a policy parameterization the agent had never encountered. That's the equivalent of teaching a new skill while measuring against an already-undertrained baseline.

## Recommended next steps (NOT in this PR)

In rough order of leverage:

1. **Re-run with 75K-100K training steps** at weight=0.0. If median_bars_held settles to a strategically sensible value (say, 20-60 bars, not 2 and not 259), and MFE_capture climbs above 50%, then we know the fix works given enough training.

2. **Add an MFE-evaporation penalty** to the reward. Something like: `-0.5 × max(0, MFE_peak - current_unrealized_pnl) / MFE_peak` per step once a position has been profitable. Pushes the agent to take profit at peaks instead of riding back through them.

3. **Sweep reward weights again on the new env**. With `r_multiple_reward_weight ∈ {0.0, 0.1, 0.3, 0.5}` × 3 seeds × 25K steps. The original sweep declared "no clear winner" but that was on an env where the policy couldn't realize the signal. With scale-out, shaping may now matter.

Each is ~3-6 GPU-hours. Item (1) is the most direct test of the hypothesis.
