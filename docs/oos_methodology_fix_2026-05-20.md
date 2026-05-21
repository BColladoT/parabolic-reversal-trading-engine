# OOS Methodology Fix — Widen Test Window from 14 to 50+ Setups (2026-05-20)

## 1. Current state diagnosis

**Current OOS window:** `2024-11-30 → 2024-12-30` (1 calendar month, 14 setups).

Verified from `models/ppo_discrete_test/w0.00_s{1,2,3}/quick_test_results.json` — all three Discrete PPO seeds use the identical window (`test_start: 2024-11-30T00:00:00`, `test_end: 2024-12-30T00:00:00`, `test_episodes_evaluated: 14`). `models/wfo_test/quick_test_results.json` (SAC baseline) uses the same window.

**Where the window is computed:** `src/scripts/train_wfo_quick_test.py:933-1012`. There are no `--test-start-date / --test-end-date` flags. The window is derived from:

- `--train-months` (default `6`, line 1206)
- `--test-months` (default `1`, line 1207) — **this is the parameter that caps the OOS at 14 setups**
- `--purge-days` (default `5`, line 1208)
- `--n-folds` (default `1`, line 1209)

The trainer (a) auto-detects the data range from `reports/all_setups_backtest.csv` via `_detect_data_range()` (lines 63-92), (b) anchors `end_date = data_max` (line 963), and (c) walks back `train_months + purge + test_months = 6 + 5/30 + 1 = ~7.17` months from `end_date`. With `data_max = 2024-12-30`, this yields train=`2024-05-29 → 2024-11-25`, purge=`2024-11-25 → 2024-11-30`, test=`2024-11-30 → 2024-12-30` — exactly what the JSON records.

**Original invocation (inferred):** `python src/scripts/train_wfo_quick_test.py --algo ppo --action-space discrete --discrete-action-bins 7 --seed {1,2,3} --output-dir models/ppo_discrete_test/w0.00_s{N}` with defaults for all date/window flags.

**Why only 14 setups?** The eval pool comes from `_get_test_setups` (lines 767-832). It builds a `HybridDataProvider(date_range=(test_start, test_end))`, gathers `dp.csv_setups` + VWAP-filtered `dp.parquet_setups`, shuffles, and loads up to `--eval-episodes` (default 50). The cached `hybrid_index.pkl` has **0 parquet_setups** (built with `skip_parquet_scan=True` in 2026-05-06), so the eval pool degenerates to "winners from `reports/relaxed_909_backtest.csv` with `pnl > 100` falling in the test window." For 2024-11-30 → 2024-12-30 that pool is 15 candidates; 14 load successfully.

## 2. Recommended window

**Use `2024-10-01 → 2024-12-30` (≈3 months, ending where the data ends).**

| Candidate window           | CSV winners (pnl>100) | Notes                                        |
|----------------------------|----------------------:|----------------------------------------------|
| 2024-11-30 → 2024-12-30 (current) | 15            | Status quo. MDE ≈$1,760.                     |
| 2024-10-01 → 2024-12-30 (3 mo)    | **34**        | **Recommended.** ~2.4× setups, MDE ≈$700–800.|
| 2024-09-01 → 2024-11-30 (3 mo shifted) | 30        | Disjoint from current; loses Dec data.       |
| 2024-07-01 → 2024-12-30 (6 mo)    | 59            | Even more power, but eats half the train window. |
| 2024-01-01 → 2024-12-30 (12 mo)   | 89            | Maximum power, but no train data left.       |

Counts use the `pnl > 100` filter `HybridDataProvider._load_csv_setups` applies (`src/rl/data_provider_hybrid.py:224`). Adding parquet-scanned high-vol days (currently zero in cache; would require rebuilding `hybrid_index.pkl`) could push 34 → 50+ but is a separate workstream.

**Rationale for 3-month recommendation:**
- ≥34 CSV winners > 14 → MDE drops materially. With 50 episodes target, full $500 MDE per the prompt requires either rebuilding the parquet index or accepting ~$700–800 MDE; still ~2.5× tighter than today.
- Doesn't overlap the train window `2024-03-01 → 2024-08-25` (under `--train-months 6 --purge-days 5 --test-months 3`).
- Recent (calendar-Q4 2024), so closest in regime to live trading.
- Mix of CSV winners and structurally diverse days (Oct/Nov/Dec each contribute ≥9 winners).
- Not cherry-picked: includes the DFNS 2024-12-17 catastrophic loss setup that anchors the loss-attribution finding.

**Stretch option:** Rebuild `hybrid_index.pkl` with full parquet scan, then 3-month window likely yields 50–80 candidates including non-winner high-vol days (the agent's "learn-when-NOT-to-trade" universe). Document this as Phase 2 of the methodology fix.

## 3. Cascade re-run plan

| # | Item                                            | Action      | Wall time est. |
|---|-------------------------------------------------|-------------|----------------|
| 1 | Phase 0 rule baseline on new window             | **MUST**    | ~5 min CPU     |
| 2 | Discrete PPO 3-seed baseline on new window      | **MUST**    | 3 × 15 min = 45 min GPU |
| 3 | SAC + PPO-continuous on new window (re-confirm PR #17 ranking) | **NICE** | 6 runs × 30 min = 3 hr GPU |
| 4 | Action-distribution diagnostics (Phase 1.2)     | **AUTO**    | included in #2 (re-emitted by trainer) |
| 5 | Re-run prior bin-count sweep (3/5/9/11) on new window | **NICE** | 4 bins × 3 seeds × 15 min = 3 hr GPU |
| 6 | Loss-attribution analysis on new window         | **MUST**    | ~10 min CPU (script over JSON) |
| 7 | NaiveShort / Random benchmarks                  | **AUTO**    | included in trainer's `_run_benchmarks` |

Items 1, 2, 6 are required to make the new window usable. Items 3, 5 establish whether prior conclusions hold; defer if compute is constrained but flag any conclusions drawn from them as provisional.

**Total wall time:** ~50 min (must) + ~6 hr (nice) ≈ **7 hr GPU + 15 min CPU** for a fully re-baselined methodology.

## 4. Exact code changes needed

All changes scoped to `src/scripts/train_wfo_quick_test.py`. **Do not** modify `train_wfo.py` (production WFO) or `data_provider_hybrid.py` (its `date_range` filter already handles arbitrary windows).

1. **Change `--test-months` default from `1` to `3`** (line 1207). One-line edit. Preserves backward-compat for anyone passing `--test-months 1` explicitly.

   Alternative (more explicit, no behavioral surprise for old invocations): **add two new flags** `--test-start-date` and `--test-end-date` that, when set, override the auto-computed window. Inside `run()` (around lines 962-967), branch on `args.test_start_date is not None`. The flag wins; otherwise the existing `--train-months/--test-months` computation runs unchanged.

   Recommendation: **add the new flags**, leave defaults of `--test-months 1` in place. Document the new flags in the docstring. Future sweep invocations pass `--test-start-date 2024-10-01 --test-end-date 2024-12-30`. This avoids breaking the existing scripts and CLI examples in PRs #16-#17.

2. **No change to `_get_test_setups`** (lines 767-832). It already takes a date range and produces up to `--eval-episodes` setups. The 50-cap default is fine.

3. **No change to `HybridDataProvider`**. Its `date_range` filter (`_filter_by_date_range`, line 154) is the WFO data-leakage barrier and already handles arbitrary contiguous ranges.

4. **`reports/rule_baseline_2024Q4.json`** — new artifact path for Phase 0 re-run. Invocation:
   `python -m src.scripts.compare_rl_vs_rule --rl-results models/ppo_discrete_test/w0.00_s{1,2,3}/quick_test_results.json --run-baseline --output reports/rl_vs_rule_baseline_2026-05-20-3mo.json` — but only AFTER re-running the 3 PPO seeds with the new window (so the `per_episode_results` setup list matches).

5. **`hybrid_index.pkl` rebuild (OPTIONAL, Phase 2):** Delete `src/scripts/data/cache/hybrid_index.pkl`. Next `HybridDataProvider` instantiation will rebuild via `_load_parquet_setups` (scans 3,082 parquet files; ~10–20 min). This unlocks the 50+ setup target.

## 5. Risk register

- **R1 — Train window collision.** With `--test-months 3`, `--train-months 6 --purge-days 5` plus 3-month test = 9.17 months total. `data_min ≈ 2020-08-12`. The auto-anchored window starts `~2024-03-22`. If any sweep uses `--train-months 12` or larger, the test window shifts back and may no longer be 2024-Q4. Mitigation: pin via new `--test-start-date / --test-end-date` flags (see §4 #1).
- **R2 — `HybridDataProvider` cache is stale.** `built_at: 2026-05-06`. `parquet_setups: 0`. Eval pool is currently CSV-only. Widening the window from 1→3 mo *does* increase CSV winners 15→34, but does NOT reach 50 without rebuilding the cache. If 50 is a hard requirement, schedule the cache rebuild before re-runs.
- **R3 — DFNS 2024-12-17 still dominates.** That single setup is 48% of current loss magnitude. In the 3-mo window it falls to ~20% (still outsized). The new MDE estimate assumes setup-level PnLs are roughly i.i.d. — if a handful of setups continue to dominate, real MDE will be worse than the formula suggests. Mitigation: report `total_test_pnl` *and* `total_test_pnl ex top-N losers` in §6 of the eval doc.
- **R4 — PR #17 "PPO beats SAC by $763" may flip sign.** That delta lives on the 14-setup window. On the 3-mo window the variance composition changes (more setups → smaller seed-to-seed PnL spread, but also exposes algorithms to more regime variance). If the SAC vs PPO ordering flips on the wider window, the "Discrete PPO is the new strongest baseline" conclusion in `docs/algorithm_comparison_smoke_2026-05-19.md` needs revision.
- **R5 — `_detect_data_range` reads `reports/all_setups_backtest.csv`, NOT `reports/relaxed_909_backtest.csv`** (line 71). These files are different. If the former's date range diverges from the latter's, the anchor will shift unexpectedly. Worth a one-line audit before re-running.
- **R6 — Trainer assumes `--test-months` is an int and uses `30 * test_months` days arithmetic.** Sub-month windows (e.g. 2 weeks) would require a different flag. The recommended 3-month value is safe.
- **R7 — Concurrent sweep shutdown.** Another agent is shutting down a running sweep. Any new artifacts in `models/ppo_discrete_test/` from in-flight runs may collide with the new 3-month runs. Use a clean output directory (`models/ppo_discrete_test_3mo/` or `models/ppo_discrete_test_v2/`).

## 6. Summary table

| Action                              | Owner       | Wall time | Output                                    |
|-------------------------------------|-------------|-----------|-------------------------------------------|
| Add `--test-start-date/--end-date` flags | code change | minutes   | `src/scripts/train_wfo_quick_test.py`     |
| Re-run 3 Discrete PPO seeds         | training    | 45 min    | `models/ppo_discrete_test_3mo/w0.00_s*/`  |
| Re-run rule baseline on new window  | benchmark   | 5 min     | `reports/rl_vs_rule_baseline_2026-05-20-3mo.json` |
| Re-run loss attribution             | analysis    | 10 min    | doc update                                |
| Optional: rebuild `hybrid_index.pkl`| data        | 10–20 min | `src/scripts/data/cache/hybrid_index.pkl` |
| Optional: SAC/PPO-continuous re-runs| training    | 3 hr      | algo-comparison doc revision              |
| Optional: bin-count sweep re-run    | training    | 3 hr      | `models/ppo_bin_sweep_3mo/`               |
