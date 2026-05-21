# Discrete PPO Tuning — Final Results

> **Status: TEMPLATE.** Replace `<...>` placeholders with values from
> `docs/discrete_ppo_final_stats.json` once Phase 6 (10-seed confirmation)
> completes. Generate that stats JSON with:
>
> ```powershell
> python src/scripts/analyze_final.py `
>     --rl-sweep models/final_10seed_<DATE>/sweep_summary.json `
>     --rule-baseline reports/rl_vs_rule_baseline_2026-05-20.json `
>     --output docs/discrete_ppo_final_stats.json `
>     --config-label "<config_label>"
> ```

## Headline

**Final config:** `(total_steps=<X>, discrete_action_bins=<Y>, ppo_clip_param=<Z>, ppo_entropy_coeff=<W>, ppo_entropy_anneal_end=<V or None>, lr_actor=<U>)`

**10-seed OOS test_pnl:** mean = $<A>, median = $<A_med>, 95% bootstrap CI = ($<B>, $<C>)

**Rule baseline OOS test_pnl:** -$2,160.19 (verified single-pass, 14 setups,
2024-11-30 → 2024-12-30 — see `docs/rule_baseline_verification_2026-05-20.md`)

**Verdict:** `<SHIP_RL | PIVOT_TO_RULE | CONTINUE_RESEARCH>`

**Rationale:** <one paragraph: state the verdict and the load-bearing fact
that drove it (RL CI lower above zero / rule above RL CI upper / both below
zero and not statistically separable).>

---

## Per-seed table

| seed | test_pnl | win_rate | mean_winner | mean_bars_in_position |
|------|---------:|---------:|------------:|----------------------:|
| 42   | $<...>   | <...>%   | $<...>      | <...>                 |
| 43   | $<...>   | <...>%   | $<...>      | <...>                 |
| 44   | $<...>   | <...>%   | $<...>      | <...>                 |
| 45   | $<...>   | <...>%   | $<...>      | <...>                 |
| 46   | $<...>   | <...>%   | $<...>      | <...>                 |
| 47   | $<...>   | <...>%   | $<...>      | <...>                 |
| 48   | $<...>   | <...>%   | $<...>      | <...>                 |
| 49   | $<...>   | <...>%   | $<...>      | <...>                 |
| 50   | $<...>   | <...>%   | $<...>      | <...>                 |
| 51   | $<...>   | <...>%   | $<...>      | <...>                 |

**Aggregate:** mean = $<A>, median = $<A_med>, std (ddof=1) = $<S>

---

## Action distribution

Mean action-bin usage across all 10 seeds (each seed's
`action_distribution` from `quick_test_results.json`, averaged element-wise):

| bin | action label    | probability |
|----:|-----------------|------------:|
| 0   | HOLD            | <...>       |
| 1   | ENTRY-100%      | <...>       |
| 2   | ENTRY-50%       | <...>       |
| 3   | COVER-100%      | <...>       |
| 4   | COVER-50%       | <...>       |
| 5   | ADD-50%         | <...>       |
| 6   | ADD-25%         | <...>       |

(Bin labels above assume `discrete_action_bins=7`. If `Y != 7`, regenerate
the table from `src/rl/env.py::_discrete_action_to_target` for the locked
`Y`.)

**Entropy:** `<H>` nats (max for N=`<Y>`: `<log(Y)>`)

**Collapse check:** dominant bin gets `<p_max>`. Threshold for "collapsed":
≥80%. Status: `<Y / N>`.

---

## Trade quality (aggregated across 10 seeds × 14 setups)

- **Win rate:** `<...>` %
- **Mean winner PnL:** $`<...>`
- **Mean loser PnL:** $`<...>`
- **MFE capture (winners only):** `<...>` %
- **Median bars in position:** `<...>`
- **Per-setup PnL std across seeds (proxy for policy noise):** $`<...>`

Rule baseline comparison on same 14-setup OOS window:

| Metric            | Rule (V5 Relaxed) | RL (10-seed mean) | Delta (RL − Rule) |
|-------------------|------------------:|------------------:|------------------:|
| total_test_pnl    | -$2,160.19        | $`<A>`            | $`<A − (−2160)>`  |
| win_rate          | 35.7%             | `<...>` %         | `<...>` pp        |
| mean_winner       | +$51.18           | $`<...>`          | $`<...>`          |
| mean_loser        | -$302.01          | $`<...>`          | $`<...>`          |
| n_trades total    | 13                | `<...>`           | `<...>`           |

---

## Decision gate verdict

**`<SHIP_RL | PIVOT_TO_RULE | CONTINUE_RESEARCH>`**

Reasoning:
- <bullet 1: the load-bearing statistical fact (e.g. "RL CI lower bound
  $<B> is non-positive — RL is not confidently profitable")>
- <bullet 2: the comparison to rule baseline ("Rule total -$2,160 sits
  below RL CI lower $<B>; rule does not statistically beat RL")>
- <bullet 3: the implication for next step (continue tuning / pivot to
  paper trading / promote to live)>

---

## Open questions / next moves

### If CONTINUE_RESEARCH

- **Recurrent PPO** — add LSTM head over current MLP; the parabolic-reversal
  setup has obvious sequential structure that a memory-less policy may be
  failing to exploit.
- **IQL on the trade-journal data** — offline RL on the historical winners
  CSV; sidesteps the high-variance online rollouts that have dominated this
  iteration.
- **Behavioral cloning refinement** — re-pretrain on the verified rule
  baseline's per-bar actions, then resume PPO from that initialization with
  a smaller LR (current BC pretrain uses an older rule definition).
- **Symbol-conditioned regime features** — add per-symbol historical
  realized vol and short-interest features to the 74-dim observation;
  current feature set is symbol-agnostic.

### If PIVOT_TO_RULE

- Wire `src/baselines/rule_baseline.py::RuleBasedAgent` into the live
  `TradingEngine` (`src/main_engine.py`); the verified rule does not yet
  have a live-trading code path.
- Add `--dry-run` to `src/scripts/preflight_paper_trade.py` to exercise
  the integration without hitting Alpaca.
- Add kill switch + intraday alerts on per-trade drawdown ≥ $300 (the
  observed mean_loser is -$302; one losing trade is the budget).
- Estimated integration effort: ~6 hours, per the integration checklist
  in `docs/rule_baseline_verification_2026-05-20.md`.

### If SHIP_RL

- Lock the config + best checkpoint and wire into the live engine
  (mirror the rule-baseline integration plumbing above).
- Paper-trade for ≥2 weeks before promoting to live capital.
- Set up drift detection: weekly recomputation of `total_test_pnl` on a
  rolling 1-month window, alert if it drops > 2σ below the in-sample
  expectation.
- Document the hyperparameter lock in `config/settings.yaml` (a new
  `rl.locked_config` section) so the live engine cannot accidentally
  load a different model.

---

## Reproducibility

- Sweep JSON: `models/final_10seed_<DATE>/sweep_summary.json`
- Stats JSON: `docs/discrete_ppo_final_stats.json`
- Rule baseline JSON: `reports/rl_vs_rule_baseline_2026-05-20.json`
- Analyzer: `src/scripts/analyze_final.py` (commit `<SHA>`)
- Training command (one seed shown; sweep runs 10 in parallel via
  `src/scripts/run_sweep.py`):
  ```bash
  python src/scripts/train_wfo_quick_test.py \
      --algo ppo --action-space discrete \
      --total-steps <X> \
      --discrete-action-bins <Y> \
      --ppo-clip-param <Z> \
      --ppo-entropy-coeff <W> \
      --ppo-entropy-anneal-end <V> \
      --lr-actor <U> \
      --seed <SEED> \
      --output-dir models/final_10seed_<DATE>/<config>=<X>_seed=<SEED>
  ```
