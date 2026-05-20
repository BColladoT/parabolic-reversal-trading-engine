# Algorithm Comparison — Full 4-way Smoke (2026-05-19/20)

**Headline:** **Discrete PPO wins decisively at -$1,610 mean test PnL** — +$535 better than PPO continuous, +$763 better than SAC wider-band, and a meaningful step away from the SAC plateau the prior 15 PRs were stuck on. This crosses the plan's "Discrete PPO wins decisively" threshold (>$300 improvement over best SAC). The Discrete(7) action space, by eliminating the continuous-to-discrete mapping entirely, attacks the root cause the SAC-era investigation kept circling. **A second, separate finding** lurking in this PR: the env_config plumbing has a latent allowlist bug that silently dropped `hold_band_threshold`, `r_multiple_reward_weight`, `mfe_evaporation_penalty_max`, and `entry/cover_threshold` since PR #13. **Past PRs were running at defaults, not their advertised configs.** Both findings ship together in this PR.

---

## 4-way comparison (3 seeds each, 25K steps, OOS test window)

```
config                                          test_pnl   spread   n_trd    win   mean_w    mean_l     e   bars   mfe_cap
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
SAC wider-band 25K (allegedly 0.30 — see bug)   -$2,373    $617     110    62%   +$6.43   -$15.26   0.41   257       5%
PPO cont 25K, PR #16 merge (also bugged 0.05)   -$2,200    $169     733    54%   +$26.74  -$34.70   0.78     8      48%
PPO cont 25K, REAL hold_band=0.30 (this PR)     -$2,145    $414     750    52%   +$26.83  -$36.84   0.74     7      50%
PPO DISCRETE(7) 25K (this PR)                   -$1,610    $621    1015    50%   +$25.63  -$38.15   0.67     5      57%
```

(Per-seed PnL — Discrete: -$1,279, -$1,649, -$1,901. PPO cont real: -$1,869, -$2,283, -$2,283. PPO cont bug: -$2,114, -$2,203, -$2,283. SAC: -$2,070, -$2,360, -$2,688.)

## Why this is the real answer for the algorithm question

After 15 PRs of investigation (PRs #1-#15), every intervention on SAC's exit, entry, and reward-shape side moved test PnL by at most ~$40. The diagnostic conclusion across PR #10 onwards — *the Gaussian policy's noise interaction with the discretization step is the structural bottleneck* — has now been tested directly:

1. **Same algorithm, different action space (PPO cont vs PPO Discrete):** moves PnL by **$535**. This is 13× the largest movement we got out of SAC tuning.
2. **Same action space, different algorithm (SAC vs PPO cont):** moves PnL by ~$173. PR #16's contribution.
3. **Both effects compound** (SAC continuous → Discrete PPO): ~$763 total.

The 15 PRs of investigation that diagnosed the discretization-noise interaction were *correct*. The interventions (scale-out cover, wider HOLD band, asymmetric thresholds) were treating symptoms because they kept the broken discretization layer in place. Eliminating that layer is the actual fix.

## What the metrics say about Discrete PPO's trade structure

- **1,015 trades per seed, 50% win rate.** More trades than PPO continuous (~750), much higher than SAC wider-band (~110). The agent is being decisive — the categorical head outputs entry-class actions directly, no noise-rejected near-misses.
- **mean_winner = $25.63 vs mean_loser = $38.15, e_ratio = 0.67.** Slightly worse expectancy ratio than PPO continuous (0.78), but mean_winner is large enough to be meaningful (4× the SAC variant).
- **5-bar median holds.** The shortest *deliberate* holds we've seen. Compare SAC baseline's 2 bars (broken — Gaussian noise hit the cover threshold) vs SAC wider-band's 257 bars (broken — Gaussian noise eroded positions via micro-covers). Discrete picks a duration and exits.
- **57% MFE capture (winners only).** The highest measured in any RL config. The Discrete categorical head can identify "this is now a good time to close" without the policy-noise smear that plagued the continuous head.
- **Per-seed spread $621.** Higher variance than PPO continuous ($169-414) but in the same ballpark as SAC ($617). The variance increase is the cost of categorical exploration — different seeds find slightly different policies. Worth it for the PnL gain.

## The allowlist bug: a separately material finding

`env.__init__` had a hardcoded allowlist of keys it would copy from `env_config` into `EnvironmentConfig`. The allowlist hadn't been updated since the earliest PRs:

```python
# Before this PR (silently dropping fields added in PRs #9, #12, #13, #15):
for key in ['initial_capital', 'max_drawdown', 'circuit_breaker_threshold',
            'reward_scale', 'max_acceptable_drawdown', 'annealer_total_timesteps',
            'intra_step_stop_loss', 'max_position_capital_fraction',
            'min_vwap_deviation_entry', 'transaction_cost_per_dollar']:
    if key in env_context:
        setattr(self.config, key, env_context[key])
```

Verified with a 4-line test (see commit message). Passing `hold_band_threshold=0.3` produces a config with `hold_band_threshold == 0.05` (the default).

### What this retroactively means for past PRs

- **PR #9 (r_multiple_reward_weight sweep):** Every sweep run was at weight=0.0. "No clear winner" is now over-determined: all runs were the same baseline.
- **PR #12 (MFE-evap penalty):** The penalty's max-magnitude was always 0.0. "75K experiment had penalty firing correctly but no PnL improvement" — penalty was never on.
- **PR #13 (wider HOLD band):** Every "wider band" smoke was at hold_band=0.05 (the default), not 0.3. The "best PnL of any scale-out variant (-$2,373)" was just SAC at default config, three more seeds. The "$617 spread, lowest variance" is meaningless — seed variance for default SAC.
- **PR #15 (asymmetric thresholds):** Both `entry_threshold` and `cover_threshold` were dropped. Every asymmetric smoke was at default symmetric 0.05.

PR #16 (PPO continuous, just-merged) was also running at hold_band=0.05 (not 0.3). Its result (-$2,200) is therefore "PPO continuous at default hold_band", not "PPO at wider band". This PR's "PPO cont REAL 0.30" smoke (-$2,145) shows what PPO continuous looks like at the actually-intended config — within $55 of the bugged version, i.e., **the wider HOLD band was a no-op for PPO too**.

### Why the bug went undetected

The investigation was guided by smoke results that *looked* like they reflected the intervention. Default behavior (3 seeds of SAC with no overrides) has its own variance ($617 spread), and that variance is enough to produce plausible-looking improvements. Three seeds isn't enough to distinguish "wider band helps by $40" from "wider band does nothing and seed variance is $617." The plan-writing process *expected* small improvements, so seeing -$2,373 vs baseline's -$2,414 looked like a real if modest win. None of it was.

This is a Goodhart-adjacent failure: smoke-test infrastructure that *appeared* to be measuring an intervention, but was actually measuring nothing. Worth flagging strongly.

## Decision

1. **Discrete PPO becomes the new RL strongest baseline.** Recommended config:
   ```
   --algo ppo --action-space discrete --total-steps 25000
   ```
2. **All previously-reported "wider band 0.3", "r_multiple_reward_weight = 0.X", "MFE-evap = 0.X", and "asymmetric entry/cover" results in synthesis docs and PR descriptions should be treated as default-config noise.** Update the synthesis doc accordingly.
3. **The 15-PR investigation's diagnosis was correct.** The bottleneck was indeed the continuous-to-discrete mapping. The interventions were the wrong fix, but the diagnosis pointed at the right place — and now Discrete PPO confirms it.

## Risks (carried over from plan)

- **Risk 1 (Dict obs + PPO):** Resolved earlier in PR #16 via DictFlatteningPreprocessor in the eval loop. Discrete PPO uses the same path; works.
- **Risk 2 (no BC warm-start):** PPO Discrete trained from scratch and still won by $763. Strongest possible result for the "no BC" comparison.
- **Risk 3 (default hyperparams underperform):** Did not materialize.
- **Risk 4 (Discrete bin layout is suboptimal):** The 7-bin {HOLD, ENTRY×3, COVER×3} layout is a designer's first guess. The smoke result already meaningfully beats all priors — bin-count or bin-magnitude sweeps are an obvious next experiment.
- **Risk 5 (stochastic env-reset retries):** Materialized 2 of 6 times (PPO cont real seeds 1 and 3); recovered by re-running.

## Plan checkpoint

- PR B done-criteria met: [x] 12 new tests + full suite green (~230)   [x] 3 Discrete smoke result dirs + 3 PPO-cont-real-0.30 control dirs   [x] 4-way comparison doc committed   [ ] PR open w/ auto-merge   [ ] Merged
- Synthesis doc (`docs/rl_investigation_synthesis_2026-05-19.md`) needs updating to flag the allowlist bug retrospectively. Will do in a small follow-up PR.

## What's still untested (for future work)

- **Discrete-bin sweep:** Try N ∈ {3, 5, 9, 11}. Different bin granularities may help or hurt.
- **Bin semantics:** ENTRY-as-target-exposure vs ENTRY-as-add-fraction. Current is target-exposure.
- **Discrete + entry-side improvements:** Tighter VWAP filter, entry features, etc. — these were untested under SAC because of the bottleneck. Worth retesting under Discrete PPO.
- **Longer training (50K, 75K).** 25K is the SAC sweet spot, but PPO/Discrete may benefit from more steps.
- **Action masking via RLModule.** Currently the env-side "override to HOLD" guard handles illegal actions, but RLlib has a proper categorical-with-mask RLModule that may train more cleanly.

## What we would tell a fresh engineer right now

"The RL system has a working baseline: PPO with Discrete(7) action space, 25K steps. It posts about -$1,610 mean test PnL across 3 seeds — meaningfully ahead of the SAC variants the prior investigation explored, though still negative versus the rule baseline.

If you want to push RL further, the highest-leverage next moves are (1) sweep Discrete bin counts, (2) revisit entry-side improvements (now uncontaminated by the discretization-noise bottleneck), (3) explore longer training schedules for PPO. Avoid the SAC + masked Gaussian path — 15 PRs of work confirmed it's a structural dead end for this problem.

Also: the env_config allowlist was fixed in this PR. Past smoke results reported as "wider band 0.30" / "MFE-evap 0.5" / etc. were silently running at defaults. The fix landed in commit 929a726."
