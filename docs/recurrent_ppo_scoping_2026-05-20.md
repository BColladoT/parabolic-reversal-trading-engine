# Recurrent PPO (LSTM) Scoping — Contingency for CONTINUE_RESEARCH

**Date:** 2026-05-20 · **Branch:** `feat/rl-discrete-ppo-tuning` · **Status:** Read-only scoping doc, untracked.

Trigger: Phase 6 decision gate selects CONTINUE_RESEARCH (Discrete PPO sweeps fail to reach break-even; current baseline -$1,610). This doc scopes the recurrent PPO (LSTM head) contingency.

---

## Section 1: Current model architecture

- **MLP**: PPO uses RLlib's default `FullyConnectedNetwork` — `[256, 256]` ReLU MLP, wired in `train_wfo_quick_test.py:286` via `model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"}`. Memoryless: each forward pass depends only on the current observation.
- **Observation**: `gym.spaces.Dict` defined in `src/rl/env.py:574-578`:
  - `state`: Box(-inf, inf, (74,)) — 64-dim TCN-AE latent + 10 explicit features
  - `action_mask`: Box(0, 1, (3,) for continuous or (N,) for Discrete(N))
  - `kelly_leverage`: Box(0, 5, (1,))
- **DictFlatteningPreprocessor**: Used implicitly by RLlib for the rollout pipeline. For the manual eval loop, `train_wfo_quick_test.py:512-523` builds it explicitly via `get_preprocessor(env.observation_space)` because `compute_single_action()` bypasses the rollout preprocessor. PPO sees a flat ~78-dim tensor.
- **Custom models already in repo**: `MaskedSACModel` (legacy `TorchModelV2`) and `MaskedSACRLModule` (new-API `SACTorchRLModule`) live in `src/rl/agent.py:477-662`, registered via `ModelCatalog.register_custom_model("masked_sac_model", ...)` (`train_wfo.py:65`). **PPO does not use these** — it relies on the action mask in the obs Dict but does not enforce it neurally; RLlib's `entropy_coeff` provides exploration noise.

## Section 2: What changes for LSTM PPO

- **Wrapper vs custom subclass**: Use RLlib's `LSTMWrapper` (set `model={"use_lstm": True, "lstm_cell_size": 128, "max_seq_len": 60}`). It wraps the existing `FullyConnectedNetwork` trunk and prepends a per-step LSTM. A fully custom `TorchModelV2` is only needed if we want to also enforce action masking inside the model, which Discrete PPO currently doesn't do. **Recommendation: wrapper.**
- **New API vs old API**: `src/rl/agent.py:54` flags `USE_NEW_API = True` when the import succeeds, **but** the PPO config in `train_wfo_quick_test.py:286` passes `model={"fcnet_hiddens": ...}` — the old `ModelV2` config dict. Confirmed: PPO sits on the old API path (`use_lstm` is a `ModelV2` flag). The new RLModule LSTM path requires building an `RLModuleSpec` and is not currently wired.
- **Sequence length**: Episodes are intraday (9:30-15:25 ET → ~355 bars max; the rollout cap is `step_count < 500` per `train_wfo_quick_test.py:546`). Typical episodes are ~60-200 bars because circuit breakers terminate early. `max_seq_len=60` is the right starting point: it covers ~1 hour of action and matches the TCN-AE's 60-bar receptive field. Setting it to the full episode (~200+) explodes BPTT memory with diminishing return.
- **Hidden state init**: RLlib's wrapper zero-inits at episode start (default). This is fine — overnight positions are flat (per CLAUDE.md), so there's no episode-to-episode state to carry. Learned init is unnecessary complexity.
- **Action mask in obs Dict**: `LSTMWrapper` passes through Dict obs via the preprocessor; the `action_mask` and `kelly_leverage` keys are concatenated into the flat input that feeds the LSTM. The mask is not enforced unless we subclass — it's just additional features. For Discrete PPO this matches the current (non-masked) behavior; the env still hard-overrides invalid actions at `env.py:975`.

## Section 3: Implementation effort estimate

**New files:**
- *None required for the wrapper path.* If we later want a masked LSTM (custom subclass), then `src/rl/models/lstm_masked_actor_critic.py` (~150 lines). Out of scope for first experiment.

**Files to modify:**
- `src/scripts/train_wfo_quick_test.py:286` — add `use_lstm`, `lstm_cell_size`, `lstm_num_layers`, `max_seq_len` keys to the `model` dict, gated by `getattr(self.config, '_use_lstm', False)`. **0.5 hr.**
- `src/scripts/train_wfo_quick_test.py:1194+` (argparse) — add `--use-lstm`, `--lstm-hidden`, `--lstm-num-layers`, `--lstm-max-seq-len` flags and propagate to `config._use_lstm` etc. **0.5 hr.**
- `src/scripts/train_wfo_quick_test.py:512-523` (eval preprocessor) — verify `LSTMWrapper` is compatible with `compute_single_action(state=..., prev_action=..., prev_reward=...)`. The manual eval loop must pass `state_in_0`, `state_in_1` between calls. **1.5 hr** (this is the trickiest part — the current eval loop doesn't thread RNN state through `policy.compute_single_action`).
- `src/rl/env.py` — **no change.** The observation shape, action space, and reward are unchanged.

**Tests:**
- `tests/test_lstm_ppo_smoke.py` — assert config builds, one `algo.train()` iteration completes, OOS eval runs without state-threading errors. **1.5 hr.**
- `tests/test_eval_loop_rnn_state.py` — assert hidden state resets at episode start and carries within episode. **1.0 hr.**

**Sweep infrastructure:** `run_sweep.py` accepts a single `--sweep param=v1,v2,v3` (see `run_sweep.py:165-195`), and forwards extras via `--extra`. Multi-axis LSTM sweeps (hidden × layers × seq_len) work today only via outer-loop scripting. **No change needed** for a 1-axis hidden-size sweep; add a `--sweep-grid` mode later if 2+ axes are needed (~3 hr).

**Total**: scaffold + wiring **1.0 hr**; eval RNN-state threading **1.5 hr**; tests **2.5 hr**; smoke training run **1.0 hr (wall) / 0.25 hr coding**. **≈5.5 engineer-hours** for the first runnable experiment.

## Section 4: GPU budget for LSTM PPO sweeps

Empirical assumptions (RLlib + RTX-class single GPU):
- Non-recurrent Discrete PPO @ 25K steps: ~10 min/seed (current observation from Phase 5 sweeps).
- LSTM forward+backward adds ~1.8× step compute (hidden=128). Sequence batching with `max_seq_len=60` over `train_batch_size=4000` yields ~67 sequences/iter, well-utilized.
- Estimated wall time per seed: **25K=18 min, 50K=36 min, 75K=54 min, 150K=108 min**.

**Full LSTM sweep cost** (6 configs × 3 seeds = 18 runs):
- @ 25K: 18 × 18 min = **5.4 GPU-hr** (smoke)
- @ 75K: 18 × 54 min = **16.2 GPU-hr** (match current Discrete budget)
- @ 150K: 18 × 108 min = **32.4 GPU-hr**

Current non-recurrent Discrete PPO budget sweep ≈ 21 GPU-hr → LSTM @ 75K is roughly the same envelope.

## Section 5: Risks and unknowns

- **`episode_step_count` (env.py:534, 881) helps debugging**: yes — log it alongside RNN hidden norm per step to detect state collapse.
- **RLlib LSTM + Discrete + action masking**: known footgun. `LSTMWrapper` does not respect action masks; the env's hard override at `env.py:975` will still fire. If the policy distribution heavily over-weights a masked action, KL divergence between sampled and clipped actions silently corrupts the surrogate loss. **Mitigation: log mask-override rate; if >5%, build a masked LSTM subclass.**
- **TCN-AE temporal redundancy**: the TCN-AE encoder already aggregates a 60-bar receptive field into the 64-dim latent (`perception.py`, causal convs). An LSTM on top is **stacking temporal models**. The hypothesis must be: TCN captures price *patterns* (local), LSTM captures *trade-state memory* (policy history, recent rewards, time-since-entry). If true, LSTM should help when prev_action and prev_reward are wired in (which `LSTMWrapper` does by default).
- **Discrete bin compatibility**: independent — bin count is the Discrete output dim of the final linear head, orthogonal to LSTM. Sweep over bins after LSTM lands.
- **Highest-risk unknown**: whether the **manual OOS eval loop** at `train_wfo_quick_test.py:536-570` can be cleanly retrofitted to thread RNN hidden state through `policy.compute_single_action` without re-architecting eval. If this turns out to require a from-scratch rollout, +4 hr.

## Section 6: Alternative paths

- **GRU** — Same wrapper (`use_lstm` is the only switch RLlib exposes; GRU would need a custom subclass). Lower memory, ~10% faster. Defer unless LSTM oversmooths.
- **Transformer/attention** — RLlib's `use_attention` flag exists but is poorly maintained in 2.x; high risk for a contingency path. Reject for this branch.
- **Frame-stacking** — Concat last K=4 obs into a (4×74)-dim input. Zero new code beyond an env wrapper. Cheaper smoke test than LSTM; consider as a precursor.
- **V-trace / PPG** — Algorithm swap. IMPALA/V-trace is async-friendly; PPG separates value and policy heads. Both are 2+ week ports. Reject for contingency.
- **Offline IQL/AWAC on trade journal** — `trades.jsonl` already accumulates per-step transitions. Promising if on-policy methods saturate, but a separate research thread.

## Section 7: Recommended first experiment IF CONTINUE_RESEARCH

**Spec**: Add `--use-lstm` opt-in to `train_wfo_quick_test.py`. Defaults: `lstm_cell_size=128`, `lstm_num_layers=1`, `max_seq_len=60`. Hold all other PPO hparams at the current Discrete(7) baseline values. Run 3 seeds × 25K steps.

**Command** (illustrative, not run here):
```
python src/scripts/run_sweep.py --algo ppo --action-space discrete \
  --total-steps 25000 --sweep use_lstm=true --n-seeds 3 \
  --extra --discrete-action-bins 7 --lstm-hidden 128 --lstm-max-seq-len 60 \
  --output-dir models/sweep_lstm_smoke_2026-05-20
```

**Why this config**:
- `hidden=128` (not 256): the input is 78-dim; 256 over-parameterizes for 3K-step trajectories per episode and risks variance blowup.
- `num_layers=1`: stacked LSTMs need >>50K steps to learn; one layer isolates whether recurrence helps at all.
- `max_seq_len=60`: matches TCN-AE receptive field and ~10% of a trading session — captures intra-position dynamics without exploding BPTT.

**Success threshold for continuing on LSTM**: mean test PnL across 3 seeds **≥ -$1,000** (vs current -$1,610) **with σ ≤ $2,500**. If mean is better but σ explodes, run 3 more seeds before declaring. If mean is worse than baseline at 25K, pivot to frame-stacking before trying longer LSTM training — the latter is a 30+ GPU-hr commitment that should not be made on a hunch.
