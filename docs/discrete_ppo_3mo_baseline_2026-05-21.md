# Discrete PPO on the Expanded OOS Window — Initial Result + Window-Generalization Test (2026-05-21)

## ⚠️ POST-WRITE UPDATE: The Q4 result does NOT robustly generalize

A follow-up run on Q3 2024 (2024-07-01 -> 09-30) produced:
- Mean: **-$88** (std $405 across 3 seeds, vs Q4's $38)
- Win rates: 28% / 26% / 14% (vs Q4's stable 40% across all 3)
- Action distribution shifts: Q3 uses bin 1 (ENTRY-100%) at 93% vs Q4's bin 2 (ENTRY-50%) at 63%

**Interpretation:** the Q4 +$1,958 is window-specific, not a robust "RL works" signal. The wide std on Q3 (3 seeds disagreeing by hundreds of dollars) suggests the policy learns different strategies depending on training-data drift. Currently running H1 2024 (Jan-Jun) as a third data point.

Verdict (provisional, pending H1): **CONTINUE_RESEARCH** — RL is not yet trustworthy enough to ship. The Q4 result remains real but is a single-window artifact, not a generalization claim.

See bottom of doc for the H1 result when it lands.

---


## Headline

**Discrete(7) PPO 3-seed mean OOS test_pnl on 2024-10-01 -> 2024-12-30: +$1,958** (std $38, n=50 episodes per seed).

| Window | n setups | RL mean | RL std (seeds) | Rule baseline | RL beats rule? |
|---|---|---|---|---|---|
| Old (2024-11-30 -> 12-30, 14 setups) | 14 | **-$1,610** | $311 | **-$2,160** | yes (by $551) |
| **New (2024-10-01 -> 12-30, 50 setups)** | **50** | **+$1,958** | **$38** | **+$1,666** | **yes (by $292)** |

The prior "RL loses money" finding was an artifact of the tiny OOS window. The DFNS 2024-12-17 catastrophic loss (48% of total) was an outlier; on a representative window both RL AND the rule baseline are profitable.

## Per-seed table (Discrete PPO, 25K steps, new window)

| Seed | test_pnl | win_rate | n_episodes |
|---|---|---|---|
| 42 | +$1,980 | 40% | 50 |
| 43 | +$1,914 | 40% | 50 |
| 44 | +$1,980 | 40% | 50 |
| **Mean** | **+$1,958** | **40%** | 50 |
| Std (ddof=1) | $38 | - | - |

Extremely tight seed variance ($38 across 3 seeds) — the policy is converged.

## Action distribution (averaged across 3 seeds)

| Bin | Action | Probability |
|---|---|---|
| 0 | HOLD | 0.00% |
| 1 | ENTRY -100% | 5.83% |
| 2 | ENTRY -50%  | 63.07% |
| 3 | ENTRY -25%  | 30.38% |
| 4 | COVER 25%   | 0.00% |
| 5 | COVER 50%   | 0.72% |
| 6 | COVER 100%  | 0.00% |

The policy almost exclusively uses ENTRY actions (bin 2 = ENTRY-50% is dominant). HOLD and COVER actions are essentially unused. This is a "stay short until forced exit" strategy — the env's circuit breakers, intra-step stops, and EOD flatten handle position closes.

## Loss attribution

- **Winners: 20 / 50 (40%)**, total +$2,777, top 3 contribute 60% (LXEO 2024-11-13 +$644, SOS 2024-11-27 +$606, SUNE 2024-10-18 +$415).
- **Losers: 6 / 50 (12%)**, total -$819, top 3 contribute 95% (NVNI 2024-12-19 -$443, WKEY 2024-12-13 -$312, LICN 2024-12-06 -$26).
- **No-trade: 24 / 50 (48%)** — agent abstains on roughly half of setups.

Outlier-robustness check:
- Drop worst 1 setup (NVNI): +$1,958 -> +$2,401
- Drop top 3 winners: +$1,958 -> +$294 (still positive!)
- Drop top 3 losers AND top 3 winners: +$1,958 -> +$1,075 (positive)

The result is robust to outlier removal.

## Methodology fix recap

The expansion from 14 -> 50 setups required:
1. Adding `--test-start-date` / `--test-end-date` CLI flags to `train_wfo_quick_test.py` (commit b442897)
2. Rebuilding `src/scripts/data/cache/hybrid_index.pkl` with `skip_parquet_scan=False` (previously cached with 0 parquet_setups, capping the eval pool to CSV winners only)
3. Recovering broken venv dependencies: polars-lts-cpu, torch, scipy, scikit-image (see memory `rl_next_moves.md` Finding 2a)

Pool sizes for the new 3-month window:
- 34 CSV winners (pnl > 100)
- 858 volatile parquet days (VWAP > 15%)
- ~892 total candidates; trainer samples up to 50 (the `--eval-episodes` default)

## Statistical power

- Per-setup PnL std on new window: $165 (vs $126 on old)
- MDE per setup at alpha=0.05: $65 (vs $126 on old)
- MDE total ($/50 setups): $3,238 — *technically larger than old* because the new window includes higher-variance parquet days
- HOWEVER per-seed mean variance is now $38 (vs $311 on old), so config-to-config sweep discrimination IS much sharper

## Implications for the tuning plan

The prior plan assumed RL was losing and needed to be tuned toward break-even. **That premise is now false.** RL is already profitable at 25K steps with default hyperparameters on this window.

Open questions:
1. **Does this generalize?** The new window is 2024 Q4. Need to test on 2024 Q3 (2024-07-01 -> 09-30) and 2024 Q1+Q2 (2024-01-01 -> 06-30) to confirm regime robustness before going to paper trading.
2. **Is 25K steps optimal?** Could be undertrained (1 seed at 25K is profitable; more steps could be even better) or overtrained (lucky 25K snapshot). Phase 3's budget sweep would resolve this.
3. **Is bin=7 optimal?** Phase 4's bin sweep would resolve.
4. **Does the policy hold up at 10 seeds?** Current is 3 seeds with very tight variance ($38). Phase 6's 10-seed run would solidify the CI.
5. **What about the rule baseline's profitability?** Now also positive ($1,666). With paper-trade infrastructure already in place, this could potentially be deployed RIGHT NOW — but the same generalization question applies.

## Recommendations

Given the new state, two parallel tracks:

**Track 1: Confirm generalization on more OOS windows**
- Re-run Discrete PPO 3 seeds on 2024 Q3 (2024-07-01 -> 09-30, ~3 months)
- Re-run on 2024 H1 (2024-01-01 -> 06-30, ~6 months)
- If results are consistently positive across windows, Phase 6's decision gate verdict is SHIP_RL.
- Estimated GPU: ~3 hr (3 seeds × 3 windows + already done)

**Track 2: Compress Phase 3-5 sweeps**
- Phase 3 (budget sweep) becomes "is 25K already the sweet spot or can we push further?" — only Phase 3 makes sense.
- Skip Phase 4 (bin sweep) — the action distribution shows the policy converged to using only ENTRY bins anyway; bin granularity is irrelevant.
- Skip Phase 5 (HP sweep) — at default HPs we already see profitability with tight variance.
- Phase 6 (10-seed final) is still valuable for tighter CIs.
- Estimated GPU: ~30 GPU-hr total (down from ~96).

Combined Track 1 + Track 2 = ~33 GPU-hr. Net savings vs original plan: 63 GPU-hr.

## Files referenced

- `models/ppo_discrete_3mo_s{42,43,44}/quick_test_results.json` — per-seed RL results
- `reports/ppo_discrete_3mo_3seed_summary.json` — sweep-summary-shaped aggregate
- `reports/rl_vs_rule_baseline_2026-05-21-3mo.json` — Phase 0 re-run on new window
- `reports/loss_attribution_3mo.json` — per-setup loss attribution
- `docs/oos_methodology_fix_2026-05-20.md` — methodology plan
- `src/scripts/data/cache/hybrid_index.pkl` — rebuilt cache (31MB, gitignored)
