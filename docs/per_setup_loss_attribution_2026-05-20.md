# Per-Setup Loss Attribution — Discrete PPO (2026-05-20)

**Status:** read-only research, untracked draft.
**Question:** is the Phase 3-6 sweep tuning to genuine signal, or to noise from 1-2 outlier setups on the 14-setup OOS window?

## Headline

**1 of 14 OOS setups (DFNS 2024-12-17) accounts for 48% of Discrete PPO's total losses and 46% of total |PnL| magnitude across the 3-seed average.** It is the only IQR outlier (mean PnL -$816.57 vs lower fence -$381.72). The top 3 worst setups (DFNS, PPBT 2024-12-02, HOUR 2024-12-24) account for **70.5% of losses, 66.9% of total |PnL|**. **The sweep is effectively measuring policy behavior on ~3 setups.**

## Per-setup table (sorted by mean PnL ascending)

| Symbol | Date | Seed 42 | Seed 43 | Seed 44 | Mean | Rule | Δ (RL-Rule) | Per-seed agreement | Trade structure |
|--------|------|---------|---------|---------|------|------|-------------|---------------------|------------------|
| DFNS | 2024-12-17 | -5 | -1222 | -1222 | **-817** | -1330 | +513 | s2/s3 identical, s1 escaped (smaller size) | Entry 9:51 @ 2.60; held to **16:52 @ 13.01** — 348 bars, **400% adverse, never covered**, forced flat at close. MFE=295 then MAE=-698. |
| PPBT | 2024-12-02 | -172 | -172 | -322 | -222 | -215 | -7 | All negative; s3 worse | Entry 10:02 @ 5.59; held to 15:59 @ 8.27 — 221 bars, MFE=153 then MAE=-861, never covered. |
| HOUR | 2024-12-24 | -294 | -181 | 0 | -158 | -104 | -54 | s3 doesn't enter; s1/s2 lose | Entry 9:48 @ 2.36, forced flat at 12:59 @ 4.39 (CB), MFE=119 then -182. |
| WKEY | 2024-12-13 | -312 | -157 | 0 | -156 | -158 | +2 | s3 abstains; s1/s2 lose | s1: 1 trade held to close, MFE=0. s2: 2 trades, second tiny winner. |
| CTEV | 2024-12-24 | -298 | -143 | 0 | -147 | -311 | +164 | s3 abstains | Booked MFE=112 then -310 (s1); s2 split into 2 trades, net loss. |
| PRFX | 2024-12-19 | -450 | +231 | 0 | -73 | +73 | -146 | **Wildest disagreement** ($681 range) | s1 sizes large and loses, s2 wins, s3 abstains. |
| SNTI | 2024-12-02 | -35 | -133 | 0 | -56 | -37 | -19 | s3 abstains | Held to 16:40, MFE=0. |
| SES | 2024-12-26 | -37 | -37 | -32 | -35 | +2 | -38 | All consistent | All 3 seeds enter, hold to close, lose small. Rule baseline scratched. |
| GXAI | 2024-12-10 | -103 | -54 | +78 | -27 | +25 | -52 | s3 wins, others lose | MFE 12-91, MAE -185 to -215. |
| GGRP | 2024-12-18 | +0 | -24 | 0 | -8 | -24 | +16 | s1/s3 ~scratch | 1-2 trades, near-zero P&L. |
| PLRZ | 2024-12-18 | 0 | 0 | 0 | 0 | 0 | 0 | All abstain | Never entered. |
| BAOS | 2024-12-24 | -62 | -128 | +198 | +3 | -236 | +239 | s3 wins big | s2 entered 9:45 lost; s3 entered later @ 7.84 won. |
| GXAI | 2024-12-06 | +49 | +49 | +21 | +40 | +21 | +19 | All winners | s1/s2 identical, smaller win on s3. |
| INTZ | 2024-12-30 | +70 | +70 | 0 | +47 | +134 | -87 | s3 abstains | Single trade @ 9:47, MFE=151 then exits with MAE=-64. |

Totals: s1=-1649, s2=-1901, s3=-1279, **mean=-1610**, rule=-2160.

## Outlier-robustness analysis

| Scenario | Per-seed totals | Mean total PnL |
|----------|-----------------|----------------|
| Full 14 setups | [-1649, -1901, -1279] | **-1610** |
| Drop DFNS (worst-1 by mean) | [-1644, -678, -57] | **-793** |
| Drop DFNS + PPBT (worst-2) | [-1472, -506, +265] | **-571** |
| Drop top-3 worst (DFNS, PPBT, HOUR) | [-1178, -326, +265] | **-413** |
| Drop IQR outliers (DFNS only) | identical to drop-w1 | **-793** |

**Mean test PnL improves by $817 (51%) just by dropping DFNS.** Seed 44 goes positive when the top-2 worst are dropped, suggesting the policy isn't structurally bad on the other 11 setups — it just decisively bleeds on the 1-3 problem cases. **The sweep "improvements" the user is observing in Phase 3 are likely measuring behavior on this same DFNS-style failure rather than improving the broader policy.**

### Does "Discrete PPO wins" survive?

| Comparison | Full | Disc minus DFNS-only |
|------------|------|----------------------|
| Disc vs SAC (-2373) | +763 (Welch t=3.01, df≈4) | +1580 if SAC unchanged (caveat below) |
| Disc vs PPO-cont (-2145) | +535 (Welch t=2.36, df≈4) | unknown (need per-setup) |

**Caveat:** We don't have per-setup data for the SAC and PPO-cont smokes. **If DFNS is similarly catastrophic for them**, removing it would flatten their totals too, possibly preserving the ranking. From the rule baseline (-1330 on DFNS) it's plausible *all* policies lose ~$1k+ on that setup — DFNS is a 400% intraday squeeze that's unfair to any shorting policy.

With n=14 and observed per-setup mean-PnL std of $217.73, the **minimum detectable effect at α=0.05 two-sided (~50% power) is $126 per setup ≈ $1,760 in total PnL**. Phase 3-6 is sweeping configs hoping to detect changes of *~$200-500 per seed* (often closer to noise-level). **The OOS sample size is statistically inadequate for the granularity of differentiation the sweep is targeting.**

## Cross-seed variance per setup (top disagreement)

| Symbol | Date | Range | Std | Comment |
|--------|------|-------|-----|---------|
| PRFX | 2024-12-19 | $681 | $342 | s1 enters large and loses -$450, s2 enters and wins +$231, s3 abstains. Policy behavior on this setup is **completely seed-dependent**. |
| DFNS | 2024-12-17 | $1,217 | $703 | s1 enters tiny ($-5), s2/s3 enter at full size (-$1,222 identical). Same trade decisions but different sizing. Note the trade record (entry 9:51) is identical across seeds — the divergence is at the in-flight sizing/cover decisions, not entry. |
| CTEV | 2024-12-24 | $298 | $156 | s3 abstains; s1/s2 enter and lose. |
| WKEY | 2024-12-13 | $312 | $156 | s3 abstains; s1/s2 enter and lose. |
| HOUR | 2024-12-24 | $294 | $156 | s3 abstains; s1/s2 enter and lose. |
| BAOS | 2024-12-24 | $326 | $174 | s3 wins +$198, others lose. |

**Key pattern:** seed 44 abstains from 8 of 14 setups (no trade). It is essentially a half-trading policy. This is consistent with the headline algorithm-comparison doc's note that variance is the cost of categorical exploration — but it also means **at n=3 seeds we cannot distinguish "the bin layout matters" from "this seed happened to abstain on the bad setups."**

## Failure-mode hypothesis (from `trades.jsonl` evaluation rows)

**Primary failure: policy fails to cover at mean reversion — enters early then holds to close.** Evidence:

- **DFNS** (s2/s3, -$1,222 each): entry 9:51 @ $2.60, **exit 16:52 @ $13.01** (forced flat at close), 348 bars, 400% adverse. Policy never covered.
- **PPBT s1+s2**: entry 10:02 @ $5.59, exit 15:59 @ $8.27 (close), MFE=$153 reached then surrendered through MAE=-$861.
- **WKEY s1**: 289 bars to close, MFE=$0.
- **HOUR s1, CTEV s1**: forced flat at 12:59 via per-trade circuit breaker (~$19K) after MFE peaks of $112-120 were given back.

The 348-bar holds and identical exits at session-close-flat or circuit-breaker times point at a **HOLD bias in the Discrete(7) categorical head**: policy biased toward HOLD over COVER once long-running adverse movement is in progress. 25K timesteps appears insufficient to train this out.

## Implications for the tuning plan

1. **Phase 6's verdict should be reported with-and-without DFNS.** The "Discrete PPO wins by $763" headline rests partly on DFNS being equally bad across all configs — we don't know that, because per-setup data for SAC/PPO-cont smokes wasn't saved. Phase 6 should re-save per-setup PnL per config so an "ex-DFNS" delta is computable.
2. **n=14 OOS setups is statistically inadequate for $200-500/seed differentiation.** MDE at α=0.05 is ~$126/setup ≈ $1,760 total. Phase 6 should require either effect sizes > $1,500 total to claim a winner, or expand the OOS window (50+ setups) to discriminate config-level signal from setup noise.
3. **Audit HOLD-vs-COVER frequency in policy logs on adverse-MFE setups** — the never-covers failure mode is a separate workstream from bin sweeps. A reward-shaping or curriculum change targeting the cover decision may be higher-leverage than bin/budget tuning.
4. **PRFX (s1=-$450, s2=+$231, s3=$0) is being sampled, not learned.** Inspect training-data coverage of similar setups.

## Method caveats

- `trades.jsonl` trade-level pnl differs from episode pnl (DFNS s2: trade -$31, episode -$1,222) — likely trade pnl is one fill's realized basis vs episode total. Episode-level used as ground truth; trade-level only for entry/exit-time inference.
- Seed 44 abstains on 8/14 setups, dragging mean improvements when DFNS is removed.
- All numbers computed via polars, 3 seeds, 14 OOS setups, 2024-11-30 to 2024-12-30.
