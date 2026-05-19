# Asymmetric ENTRY/COVER Thresholds — Smoke Results (2026-05-19)

**Headline:** **Hypothesis falsified.** Asymmetric thresholds (entry=0.05, cover=0.3) did NOT increase trade count and DID degrade trade structure. PR #13's symmetric wider band remains the best config.

## Setup

- Branch: `feat/asymmetric-thresholds` (commit `[new]`)
- 3 seeds × 25K training steps × weight=0.0 × `--entry-threshold 0.05` × `--cover-threshold 0.3`
- Result dir: `models/asymmetric_test/w0.00_s{1,2,3}/`
- Variable being isolated: entry vs cover threshold asymmetry. All other settings match PR #13's wider-band smoke.

## Four-way comparison (3 seeds each)

```
config                              test_pnl   e_ratio   mean_w     mean_l   mfe_cap   n_trades  median_bars  spread
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
BASELINE 25K (narrow 0.05, full-close) -$2,414   0.72   +$32.47    -$45.33   79%      2,900     2            $834
SCALE-OUT 25K (narrow 0.05, scale-out) -$2,479   0.18   +$0.87     -$4.83    1.1%       329     259          $928
WIDER-BAND 25K (symmetric 0.30)        -$2,373   0.34   +$5.96     -$17.33   6.8%       332     258          $617
ASYMMETRIC 25K (entry=0.05, cover=0.30) -$2,369   0.14   +$0.73     -$5.35    0.8%       338     250          $650  ← this
```

## What we predicted vs what happened

**Predicted** (per the plan):
- Trade count climbs from 332 → 800-1,500 (entry restriction lifted)
- mean_winner stays around $5.96 (cover side preserved)
- Test PnL improves $100-$300 vs PR #13

**Actual**:
- Trade count: 332 → 338 (+6, barely moved)
- mean_winner: $5.96 → $0.73 (88% DROP — back to scale-out degenerate)
- Test PnL: -$2,373 → -$2,369 (essentially same)

The prediction was wrong on every count.

## Real diagnosis: the wider band was doing double duty

The asymmetric setup made entries easier (narrow threshold 0.05) but also made them **smaller**:

- With ENTRY threshold = 0.05, any action < -0.05 triggers an entry
- A Gaussian policy with mean ≈ -0.1 and noise stddev ~0.2 produces many entry actions, but most are in the range -0.05 to -0.20
- Position size = `action × max_leverage × capital`, so small actions = small positions
- Small initial positions = small per-trade dollar magnitude regardless of price moves

**PR #13's symmetric wider band (0.3) wasn't just suppressing noise-covers — it was enforcing a minimum entry commitment.** To enter, action must be < -0.3, which is a "committed" entry. The resulting positions are large enough to produce meaningful winners.

This is a meaningful structural insight we hadn't seen before. The wider band's symmetric structure does TWO things at once:
1. Suppress Gaussian-noise-driven micro-covers (the original PR #13 finding)
2. Enforce a minimum commitment on entry (the new finding here)

## What this tells us about "the entry side"

The entry side **isn't the bottleneck** for trade count or PnL. The wider band already does what we wanted (give entries a commitment threshold). The trade-count drop from 2,900 → 332 isn't a bug — it's the **agent learning to be selective**.

Per-seed test_pnl variance for asymmetric is $650 — close to wider-band's $617. Both configurations are stable; the asymmetric just leads to a different (and worse) equilibrium.

## Decision

**Keep the code, ship as opt-in.** The asymmetric threshold infrastructure is correct and tested. It's available behind `--entry-threshold` and `--cover-threshold` flags (default None = falls back to `--hold-band-threshold`, preserving PR #13 behavior). Future experiments can use the asymmetric knob if motivated by a different hypothesis.

**Don't change the recommended config.** PR #13's symmetric `--hold-band-threshold 0.3` remains the best config we've found.

## What this adds to the synthesis (`docs/rl_investigation_synthesis_2026-05-19.md`)

A new "ruled out" item: **lowering the entry threshold while preserving cover threshold does NOT improve PnL or trade frequency.** The wider symmetric band's role is dual (noise suppression + entry commitment), and breaking the symmetry undermines the entry commitment.

## What's still untested on the entry side

- **Lowering `min_vwap_deviation_entry`** (currently 15) — would let setups with weaker VWAP signals through the entry mask. Different mechanism than threshold.
- **Adding entry-quality features to the observation** — observation may not have the right signals to discriminate good vs bad entries.
- **Higher minimum-commitment entry threshold** (e.g., -0.5 entry, 0.3 cover) — opposite asymmetry, forces even larger initial positions.

But honestly: after this experiment, the **entry side feels well-explored**. The bottleneck for further improvement is probably elsewhere (algorithm choice, observation features, or different problem framing).
