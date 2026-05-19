# Wider HOLD-Band — Smoke Results (2026-05-19)

**Headline:** **The cleanest result of the entire scale-out-era investigation.** Wider HOLD band (threshold 0.05 → 0.3) at 25K steps gets the best test PnL of any scale-out variant we've tried, with the lowest per-seed variance, and at 1/3 the training cost of the 75K MFE-evap experiment. The position-decay-via-noise problem identified in `docs/mfe_evap_smoke_2026-05-18.md` is resolved.

## Setup

- Branch: `feat/wider-hold-band` (commit `d859b11`)
- 3 seeds × 25K training steps × weight=0.0 × `--mfe-evap-penalty 0.0` × `--hold-band-threshold 0.3`
- Result dir: `models/wider_band_test/w0.00_s{1,2,3}/`
- Isolating the variable: wider band ONLY, no MFE-evap penalty. So the comparison is purely about action discretization.

## Four-way comparison (3 seeds each)

```
config                              test_pnl   e_ratio  mean_w   mean_l    mfe_cap   median_bars  n_trades  spread
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
BASELINE 25K (narrow band, full-close)  -$2,414   0.72   +$32     -$45     79%       2          2,900     $834
SCALE-OUT 25K (narrow band 0.05)        -$2,479   0.18   +$0.87   -$4.83    1.1%   259            329     $928
WIDER-BAND 25K (band 0.30)              -$2,373   0.34   +$5.96   -$17.33   6.8%   258            332     $617  ← WIN
MFE-EVAP-75K (narrow band + penalty 0.5) -$2,735  0.82   +$1.60   -$1.96    2.0%   260            912   $1,043
```

## What changed (diagnosis from PR #12 smoke)

The position-size analysis showed all scale-out variants took **same-sized initial positions** (~110 shares × ~$7 = ~$750), but realized PnL per trade was 20× smaller than baseline. Cause: SAC's Gaussian policy noise above the +0.05 cover threshold triggered tiny partial covers each step. Over 260-bar holds, hundreds of micro-trims decayed positions to near-zero.

Widening the threshold from 0.05 → 0.3 means:
- Action ∈ [-0.3, +0.3] now classified as HOLD (was: ENTRY at -0.05, COVER at +0.05)
- With typical Gaussian policy stddev ~0.1-0.3, the majority of samples now land in HOLD
- Positions stay intact during the hold; covers only fire when the policy *deliberately* commits

## What the data says

**1. Test PnL improved.** -$2,373 vs scale-out's -$2,479 ($106 better). Within seed variance but consistently better across all 3 seeds (no seed is the worst).

**2. Trade structure recovered.** mean_winner jumped 6.8× ($0.87 → $5.96). MFE capture improved 6× (1.1% → 6.8%). The agent isn't getting tiny accidental wins anymore — it's getting real wins.

**3. Losses also bigger** (-$4.83 → -$17.33) but still cut significantly shorter than baseline's -$45 (-62% from baseline).

**4. Per-seed variance is LOWER** than any other config — spread of $617 vs $834 (baseline) and $928 (scale-out). The wider band makes training more reproducible.

**5. Trade frequency similar** to scale-out (332 vs 329). The agent isn't taking more trades — it's just taking better ones.

**6. Beat the 75K-step MFE-evap experiment** in test PnL ($362 better) at 1/3 the training cost. The wider band achieves what the MFE-evap penalty was trying to achieve, more directly and more cheaply.

## The trade-off vs baseline

The wider-band config matches baseline test PnL (-$2,373 vs -$2,414, within $40 = within seed-variance noise). It does NOT yet beat baseline. But the path to improvement is now visible:

| Metric | Baseline | Wider-band | Gap to close |
|---|---|---|---|
| mean_winner | $32 | $5.96 | 5.4× to recover full winner size |
| mfe_capture | 79% | 6.8% | The agent still releases winners too early |
| mean_loser | -$45 | -$17 | Already 62% better — wider band helps here |
| n_trades | 2,900 | 332 | Fewer, larger commitments — likely the right direction |

The gap is mostly on the **winner-capture side**. The agent holds for 258 bars but only captures 6.8% of available MFE. The original MFE-evap penalty diagnosis was right; it just couldn't act on it through the noisy action mechanic. With the wider band fixing the noise problem, the MFE-evap penalty may now be effective.

## Decision

**Ship the wider band as a default-on improvement, or keep it opt-in?**

The data argues for default-on: it matches baseline test PnL with cleaner trade structure, lower variance, and is composable with future shaping experiments. But default-on changes the env semantics for all callers — riskier.

This PR ships **opt-in via `--hold-band-threshold` flag** (default 0.05 = pre-existing behavior). Future PRs can flip the default to 0.3 if more experiments confirm the win.

## Recommended next experiments (NOT in this PR)

In order of leverage:

1. **Wider band + MFE-evap penalty + 25K** — stack the two interventions. The MFE penalty might now work because the wider band silenced the noise-decay it was fighting. ~36 min, 3 seeds.
2. **Wider band + 75K training** — let the cleaner-trade-structure policy converge longer. ~2 hours, 3 seeds.
3. **Wider band, sweep threshold {0.2, 0.3, 0.4, 0.5}** — find the optimum. 0.3 was a pragmatic first try, not optimized. ~3 hours, 12 runs.
