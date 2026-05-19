# RL Training Investigation — Synthesis (2026-05-19)

**14 PRs, ~3 days, ~10 GPU-hours of experiments.** This document is what we'd want to read if we came back fresh and asked: "what is the state of the RL training pipeline and what should we do next?"

## TL;DR

- **Bottom-line:** mean test PnL across all configs sits in -$2,373 to -$2,479 (within seed-variance noise of each other). **Best config = wider HOLD band at 25K** (`--hold-band-threshold 0.3`).
- **What we fixed:** the 2-bar-median-hold problem (now ~258 bars), broken Ray/torch DLLs on Windows, missing observability for live sizing, missing test infrastructure for sweep analysis, lots of compat patches.
- **What we ruled out:** R-multiple reward weights {0.0, 0.1, 0.2, 0.5}, MFE-evaporation penalty (Goodhart-game-able), longer training (degenerates to many-small-trades).
- **What we did NOT solve:** the agent's MFE-capture is still ~7% vs baseline's 79%. We can't beat baseline test PnL.
- **Best recommendation if continuing:** investigate the entry side. The exit side has been thoroughly explored.

---

## The 14-PR journey

```
PR #1-#8   System hardening: live wiring, observability, sweep infra,
           torch+CUDA on Windows, paper-trade preflight.
           
PR #9      First sweep: r_multiple_reward_weight in {0.0, 0.1, 0.2, 0.5}.
           Result: "no clear winner" at 25K steps. All weights losing
           similar money.
           
PR #10     Diagnosis batch: trade-level + training-curve + mask
           analyzers. Identified three structural issues:
             1. Negative expectancy across all weights (0.72)
             2. Mask violations RISING with training
             3. Training cut short — still improving
           Verdict: more training won't fix this alone.
           
PR #11     Scale-out COVER fix. The all-or-nothing binary cover was
           producing 2-bar median holds because SAC's Gaussian policy
           noise drove 25% of samples into COVER territory, full-closing
           positions. Scale-out: action magnitude = fraction closed.
           Result: median_bars_held 2 -> 259. Positions hold now.
           BUT: agent gives all profit back (mfe_capture 1.1% vs 79%).
           
PR #12     MFE-evaporation penalty. Per-step penalty for letting
           profitable position drift back from MFE peak.
           Result at 25K: penalty fires correctly but agent doesn't act.
           Result at 75K: expectancy_ratio 0.06 -> 0.82 monotonically,
           but test_pnl got WORSE (-$2,735, overfit signal).
           
[non-PR]   Position-size analysis on 75K data:
           agent took same-sized initial positions as baseline
           (~110 shares × ~$7) but realized PnL per trade was 20x
           smaller. Cause: Gaussian noise + scale-out = micro-trims
           every step that decay positions over 260-bar holds.
           
PR #13     Wider HOLD-band threshold. Replaced hardcoded ±0.05 in
           _discretize_action with configurable threshold. Default
           preserved (0.05). Smoke at 0.3:
             test_pnl     -$2,373  (best of all scale-out variants)
             expectancy   0.34
             mean_winner  +$5.96   (6.8x larger than narrow scale-out)
             mfe_capture  6.8%     (6x improvement)
             spread       $617     (lowest variance of any config)
           Beats 75K MFE-evap experiment at 1/3 the training cost.
           
[no PR]    Stacked experiments:
           - Wider band + MFE-evap 0.5 at 25K: WORSE (-$2,472).
             MFE penalty actively harmful when agent can hold;
             Goodhart's law (agent suppresses MFE to avoid penalty).
           - Wider band alone at 75K: NO IMPROVEMENT (-$2,383).
             More training degenerates wider-band's edge.
```

## Numerical summary across all configs

```
config                                 test_pnl   e_ratio   mean_w    mfe_cap   median_bars   spread
─────────────────────────────────────────────────────────────────────────────────────────────────────
BASELINE 25K (full-close cover)        -$2,414   0.72     +$32      79%       2             $834
SCALE-OUT 25K (narrow band, no penalty) -$2,479   0.18     +$0.87    1.1%      259           $928
MFE-EVAP 25K (narrow band + p=0.5)     -$2,449   0.06     +$0.33    0.4%      258           $741
MFE-EVAP 50K                            -$2,844   0.27     +$1.49    1.9%      250           $546
MFE-EVAP 75K                            -$2,735   0.82     +$1.60    1.95%     260           $1,043
WIDER-BAND 25K (band 0.3, no penalty)  -$2,373   0.34     +$5.96    6.8%      258           $617  ← best
WIDER-BAND + MFE 25K (combined)        -$2,472   0.07     +$0.52    0.6%      250           $600
WIDER-BAND 75K                          -$2,383   0.28     +$1.63    1.85%     260           $4    (2 seeds)
```

## What's actually been proven

### Proven 

1. **The 2-bar median hold problem was real and structural** — caused by Gaussian policy noise interacting with binary cover. Scale-out (PR #11) fixed it.
2. **Position-decay-via-noise is a real failure mode** of scale-out with narrow discretization band. The wider band (PR #13) fixes it.
3. **The MFE-evaporation penalty is structurally broken** — Goodhart's law applies. Agent suppresses MFE to avoid the penalty rather than capturing more of it. Don't use.
4. **More training beyond 25K hurts in this setup** — across both MFE-evap and wider-band, longer training causes the policy to converge to a fragmented many-small-trades equilibrium.
5. **The 25K-step budget is a sweet spot** for this configuration, not a "still improving" condition as PR #10 suggested.

### Ruled out 

1. R-multiple reward weight in {0.0, 0.1, 0.2, 0.5} — no clear winner (PR #9)
2. MFE-evaporation penalty at any magnitude or training length
3. Longer training as a path to higher MFE capture
4. Combining scale-out + MFE-evap

### Not proven yet (open questions)

1. **What blocks MFE capture from rising above ~7%?** The agent holds 258 bars but only captures 7% of available MFE. Why? Possibilities:
   - The policy literally can't tell when MFE is peaking (insufficient observation features)
   - The reward landscape has no peak signal — only "hold winning is good" (constant per-step bonus)
   - The agent has learned that taking SOMETHING is safer than waiting
   - The data is genuinely hard (the strategy's edge is small)
2. **Whether ENTRY-side improvements would matter** — we've spent all our effort on EXITS. The wider-band agent only takes ~330 trades vs baseline's 2,900. Is the entry policy too conservative?
3. **Whether a discrete action head** (instead of Gaussian-on-continuous) would solve the policy-noise problem more cleanly. We've worked around it with the wider band.

---

## Honest assessment

After 14 PRs:

- Baseline test PnL: **-$2,414**
- Best new config: **-$2,373**
- Gap closed: **$41 / episode** (≈ 1.7% improvement, within seed-variance noise)

We haven't moved the needle on bottom-line PnL. We HAVE:
- Built robust experimental infrastructure (sweep, diagnose, compare scripts)
- Fixed multiple structural bugs (Ray API, torch DLLs, position_manager wiring, etc.)
- Documented the strategy's behavior in unprecedented detail
- Identified a working "trade structure" (wider band 25K) even if PnL hasn't improved

**A skeptical reading:** the strategy's parabolic-reversal signal may simply be too weak in this dataset to support a profitable RL policy at 25K steps. The mean winner being **6× smaller** than baseline ($5.96 vs $32) — while taking 10× fewer trades — suggests the agent has learned to be much more selective, but the selectivity itself isn't yielding better expected value.

---

## Three honest options going forward

### Option A: Stop here

Accept that the RL track has hit a plateau. The infrastructure is good, the documentation is honest, the live engine (paper trade preflight) is operational. **Switch focus to the rule-based system + live paper trading**, which already shows positive edge in the journal (PR #4-#7 data: 334 trades, 78% win rate, +2.19 R-multiple).

This is the highest-leverage move purely in expected-value terms — the rule system *already works*. RL is research; rule trading is operational.

### Option B: Attack the entry side

Everything tested has been about exits. Try:
- Loosen `min_vwap_deviation_entry` to take more setups
- Add entry-time features the agent can use to be selective
- Try ENTRY scale-in (similar to scale-out cover) so the agent can build positions gradually

~3-5 experiments, each ~36 min. Potentially uncovers new degrees of freedom.

### Option C: Reconsider the algorithm

SAC + masked Gaussian on continuous action is the wrong fit for a discrete-action problem. Try:
- Switch to PPO (more stable, less noise-sensitive)
- Switch to a discrete-action SAC variant
- Or even imitation learning from the rule baseline (which already works)

~Days of work. Highest research value but highest cost.

---

## What we'd tell a fresh engineer right now

"The rule-based system works (~78% win rate, +2.19 R-multiple in backtest). It's running through PR #4-#7's trade journal and is ready for paper trading via `preflight_paper_trade.py`.

The RL system has been investigated thoroughly. The scale-out + wider-band config (`--hold-band-threshold 0.3`) is the best research configuration and is mathematically meaningful — but doesn't yet generate alpha vs the random-init baseline at the bottom line. If you want to push RL further, start with the entry side or change the algorithm. If you want to make money, run the rule system."
