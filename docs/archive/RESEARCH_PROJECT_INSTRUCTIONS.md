# Quant Trading RL System — Reward Structure Redesign

## Project Context

This Claude project contains the full source code for a reinforcement learning (RL) system that learns intraday trading strategies on US equities. The system uses a Soft Actor-Critic (SAC) agent operating on 1-minute OHLCV bars, with a TCN autoencoder providing compressed market representations. The core problem: **the agent consistently collapses to a do-nothing policy** — earning zero trades and zero returns — because the current reward structure makes inaction the rational choice. Your job is to redesign the reward function and related training mechanisms to produce a policy that actively trades.

---

# SECTION 1: Data Structures & Architecture Overview

This section documents every data structure the RL agent interacts with so you can reason precisely about where reward signal flows, what the agent observes, and how training unfolds.

---

## 1.1 Observation Space

The agent receives a **74-dimensional continuous vector** at every step:

| Dimensions | Source | Description |
|---|---|---|
| 0–63 | TCN latent | 64-dim encoding from the frozen TCN autoencoder (compressed representation of the last 60 bars of OHLCV data) |
| 64 | Hand-crafted | Normalized position indicator (0 = flat, 1 = long) |
| 65 | Hand-crafted | Unrealized PnL (normalized by entry price) |
| 66 | Hand-crafted | Time-in-trade (fraction of episode elapsed since entry) |
| 67 | Hand-crafted | Bars remaining in episode (normalized 1→0) |
| 68 | Hand-crafted | Current drawdown from equity peak (negative value) |
| 69 | Hand-crafted | RSI (14-period, scaled 0–1) |
| 70 | Hand-crafted | VWAP deviation (current price vs. session VWAP, normalized) |
| 71 | Hand-crafted | Spread / volatility ratio |
| 72 | Hand-crafted | Volume z-score (current bar volume vs. rolling mean) |
| 73 | Hand-crafted | Momentum signal (rate of change, normalized) |

The TCN latent (dims 0–63) captures temporal patterns across the 60-bar lookback window. The hand-crafted features (dims 64–73) provide the agent with explicit awareness of its own position state, risk exposure, and key technical indicators that the TCN might not surface directly.

**Key design note:** The agent has no direct access to raw price — only the TCN encoding and derived indicators. This is intentional to prevent overfitting to absolute price levels.

---

## 1.2 Action Space

The raw action output is a **single continuous scalar in [-1, 1]**, produced by SAC's squashed Gaussian policy (tanh output). This is mapped to three discrete trading behaviors via thresholds:

```
action ∈ [-1.0, -0.5)  →  SELL / EXIT    (close any open long position)
action ∈ [-0.5, +0.5]  →  HOLD / NO-OP   (do nothing)
action ∈ (+0.5, +1.0]  →  BUY / ENTER    (open a long position if flat)
```

**Why this matters for policy collapse:** The HOLD region occupies 50% of the action range [-0.5, +0.5]. When the agent is uncertain (high entropy), the tanh-squashed Gaussian naturally concentrates mass around 0, which maps to HOLD. Combined with a reward function that doesn't penalize inaction, the agent learns that HOLD is always safe — and collapses there permanently.

---

## 1.3 Action Masking

A **3-element boolean mask** `[can_buy, can_hold, can_sell]` restricts which actions are valid at each step:

| Mask Element | True When | False When |
|---|---|---|
| `can_buy` | Agent is flat (no open position) AND sufficient buying power | Already in a position OR insufficient capital |
| `can_hold` | Always True | Never masked — hold is always valid |
| `can_sell` | Agent has an open long position | Agent is flat (nothing to sell) |

**Current masking implementation:** The mask is applied at the environment level — invalid actions are intercepted and converted to HOLD, with a small penalty applied to the reward. The mask is **not** fed into the neural network or used to modify the policy distribution.

**Problem with current approach:** Env-level masking means the policy network still outputs invalid actions, receives a penalty, but has no gradient signal explaining *why* the action was invalid. The agent can't learn the structure of the constraint — it just learns that certain regions of state space are "bad," which contributes to overly conservative behavior.

---

## 1.4 Reward Components

The current reward function is computed per-step and has three additive components:

### 1.4.1 Base Reward — Equity Delta

```
r_base(t) = (equity(t) - equity(t-1)) / initial_equity
```

This is the fractional change in portfolio equity from one step to the next. When flat, `equity(t) = equity(t-1)` so `r_base = 0`. When holding a position, equity changes with the mark-to-market value of the open trade. On trade close, realized PnL is captured.

**Scale:** Typical values are O(1e-5) to O(1e-4) per bar for intraday equity movements. This is extremely small and easily overwhelmed by entropy bonuses in SAC.

### 1.4.2 Drawdown Penalty

```
r_drawdown(t) = -λ_dd * max(0, peak_equity - equity(t)) / initial_equity
```

Where `λ_dd` is a scaling coefficient (currently `λ_dd = 2.0`). This penalizes the agent whenever equity drops below its running maximum. The penalty is proportional to the drawdown magnitude.

**Problem:** This penalty is asymmetric — it punishes losses but doesn't reward gains of equal magnitude. Combined with the tiny base reward, the optimal strategy becomes: never trade, never draw down, receive `r = 0` forever. Zero is better than the expected value of trading (small positive expectation minus drawdown penalty risk).

### 1.4.3 Mask Violation Penalty

```
r_mask(t) = -0.01  if action was overridden by mask
           =  0.0   otherwise
```

Applied when the agent outputs a BUY while already in a position, or a SELL while flat. The overridden action becomes HOLD and this small penalty is added.

### Combined Reward

```
r(t) = r_base(t) + r_drawdown(t) + r_mask(t)
```

**Summary of the reward problem:**

- `r_base` is too small relative to SAC's entropy bonus (α ≈ 0.2)
- `r_drawdown` is asymmetrically punitive
- There is **no opportunity cost** for holding — the agent is never punished for missing profitable setups
- There is **no reward shaping** for trade lifecycle events (entry quality, hold discipline, exit timing)
- The mask penalty provides no structural learning signal

---

## 1.5 Episode Structure

A single training episode represents **one stock-day**: all 1-minute bars for a single ticker on a single trading day.

### Episode Flow

```
1. Sample a (ticker, date) pair from the data provider
2. Load all 1-min OHLCV bars for that day (typically 390 bars, 9:30 AM – 4:00 PM ET)
3. Compute session VWAP incrementally
4. Initialize: equity = starting_capital, position = flat, peak_equity = starting_capital
5. For each bar t = 1..T:
   a. Run TCN encoder on lookback window → 64-dim latent
   b. Compute hand-crafted features → 10-dim vector
   c. Concatenate → 74-dim observation
   d. Agent produces action ∈ [-1, 1]
   e. Apply action masking (override if invalid)
   f. Execute trade logic:
      - BUY: open long at bar's VWAP-approximated fill price
      - SELL: close position at bar's VWAP-approximated fill price
      - HOLD: no action
   g. Update equity (mark-to-market if in position)
   h. Compute reward r(t)
   i. Check termination: forced liquidation on last bar, or if drawdown > max_drawdown
6. Episode ends → return trajectory [(s_t, a_t, r_t, s_{t+1}, done_t)]
```

### Fill Price Model

Trades are filled at an approximated VWAP for the bar, not at the close. This is more realistic than close-price fills and prevents the agent from exploiting bar-boundary artifacts.

### Forced Liquidation

If the agent is still holding a position when bars run out (end of day), the position is forcibly closed. This is intentional — the system is designed for intraday-only strategies with no overnight holds.

---

## 1.6 State Flow Pipeline

The full data flow from raw market data to training update:

```
Raw tick data
  ↓ (aggregation)
1-min OHLCV bars
  ↓ (rolling window)
60-bar lookback tensor [1, 5, 60]  (batch=1, channels=OHLCV, length=60)
  ↓ (frozen TCN encoder)
64-dim latent vector
  ↓ (concatenation with hand-crafted features)
74-dim observation vector
  ↓ (SAC policy network)
Continuous action ∈ [-1, 1]
  ↓ (threshold discretization + mask)
Discrete trade decision {BUY, HOLD, SELL}
  ↓ (environment execution)
New equity, new position state
  ↓ (reward computation)
Scalar reward r(t)
  ↓ (replay buffer)
(s, a, r, s', done) tuple stored
  ↓ (SAC update)
Critic loss, Actor loss, Entropy α adjustment
```

---

## 1.7 Training Data: HybridDataProvider

The data provider mixes two sources to balance exploitability with diversity:

| Source | Share | Format | Selection Criteria |
|---|---|---|---|
| CSV Winners | 70% | `.csv` files | Pre-screened stock-days with known profitable intraday patterns (VWAP reclaim, momentum breakouts). These are "easy" episodes where a good policy should find alpha. |
| High-Vol Parquet | 30% | `.parquet` files | High-volume stock-days sampled broadly. Not pre-screened for profitability. Provides diversity and prevents overfitting to the curated set. |

**Episode sampling:** At the start of each episode, the data provider flips a weighted coin (70/30) to choose the source, then uniformly samples a (ticker, date) pair from that source. This means the agent sees mostly "winnable" episodes but must also handle noise.

---

## 1.8 TCN Autoencoder

The Temporal Convolutional Network autoencoder compresses raw bar data into a dense representation:

### Architecture

```
Encoder:
  Input: [batch, 5, 60]  — 5 channels (OHLCV), 60 timesteps
  → Conv1D(5, 32, kernel=3, dilation=1) + ReLU + BatchNorm
  → Conv1D(32, 64, kernel=3, dilation=2) + ReLU + BatchNorm
  → Conv1D(64, 64, kernel=3, dilation=4) + ReLU + BatchNorm
  → AdaptiveAvgPool1D(1)  — collapse temporal dim
  → Linear(64, 64)        — bottleneck latent

Decoder (mirror, used only during autoencoder pre-training):
  Latent 64 → reconstruct [batch, 5, 60]
```

### Training & Usage

- **Pre-trained** on historical bar data using reconstruction loss (MSE on normalized OHLCV)
- **Frozen during RL training** — weights are not updated by SAC gradients
- The 64-dim latent output feeds directly into the SAC observation vector
- Rationale: decouples representation learning from policy learning, stabilizes training

---

## 1.9 SAC Agent Configuration

The Soft Actor-Critic agent uses the following architecture and hyperparameters:

### Network Architecture

```
Actor (Policy):
  Input: 74-dim observation
  → Linear(74, 256) + ReLU
  → Linear(256, 256) + ReLU
  → Linear(256, 2)        — outputs (μ, log_σ) for Gaussian
  → Tanh squashing         — produces action ∈ [-1, 1]

Critic (x2 for twin Q):
  Input: 74-dim obs ⊕ 1-dim action = 75-dim
  → Linear(75, 256) + ReLU
  → Linear(256, 256) + ReLU
  → Linear(256, 1)         — Q-value estimate
```

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Learning rate (actor) | 3e-4 | Adam optimizer |
| Learning rate (critic) | 3e-4 | Adam optimizer |
| Learning rate (α) | 3e-4 | Entropy temperature auto-tuning |
| Discount (γ) | 0.99 | Standard for episodic tasks |
| Soft update (τ) | 0.005 | Target network polyak averaging |
| Replay buffer size | 1,000,000 | Stores transitions across episodes |
| Batch size | 256 | Per gradient update |
| Target entropy | -1.0 | For 1-dim action space (= -dim(A)) |
| Initial α | 0.2 | Entropy coefficient, auto-tuned |
| Warmup steps | 10,000 | Random actions before policy is used |

**Critical insight on entropy tuning:** With target entropy = -1.0 and initial α = 0.2, the entropy bonus `α * H(π)` can dominate the reward signal when `r_base` is O(1e-5). The agent is incentivized to maintain high entropy (output actions near 0 = HOLD) rather than commit to directional trades.

---

## 1.10 Walk-Forward Optimization (WFO) Structure

Training uses a walk-forward validation scheme with 4 folds:

```
Fold 1: Train [2018-01 → 2019-12]  Test [2020-01 → 2020-06]
Fold 2: Train [2018-07 → 2020-06]  Test [2020-07 → 2020-12]
Fold 3: Train [2019-01 → 2020-12]  Test [2021-01 → 2021-06]
Fold 4: Train [2019-07 → 2021-06]  Test [2021-07 → 2021-12]
```

Each fold has **two training phases**:

### Phase 1 — Critic-Only (Actor Frozen)

- Duration: First N episodes (configurable, typically 50% of training budget)
- Actor weights are frozen — policy doesn't change
- Only critic networks are updated
- Purpose: Let critics learn accurate Q-value estimates under the current (or behavioral cloning) policy before the actor starts optimizing against them
- Prevents early actor updates based on inaccurate Q-values

### Phase 2 — Full Training (Actor Active)

- Duration: Remaining episodes
- Both actor and critic networks are updated
- Actor optimizes against the now-calibrated critics
- Entropy coefficient α continues auto-tuning

**Rationale for two-phase:** In standard SAC, the actor and critics bootstrap off each other from step 1. When rewards are tiny and noisy (as in this system), this creates a death spiral: bad Q-estimates → bad policy updates → worse data → worse Q-estimates. Phase 1 breaks this by giving the critic a head start.

---

## 1.11 Behavioral Cloning Pre-Training

Before RL training begins, the actor network is pre-trained via behavioral cloning (BC) on expert demonstrations:

### Expert Data Format

```python
# Each expert trajectory is a list of (observation, expert_action) pairs
expert_data = [
    {
        "observations": np.array of shape [T, 74],   # same obs space as RL
        "actions": np.array of shape [T, 1],          # continuous [-1, 1]
        "metadata": {
            "ticker": str,
            "date": str,
            "total_pnl": float,
            "num_trades": int
        }
    },
    ...
]
```

### BC Training Pipeline

1. Load expert trajectories (generated from rule-based strategies that are known profitable)
2. Train actor network with supervised loss: `L_BC = MSE(π(s), a_expert)`
3. Typically 50–100 epochs until convergence
4. The BC-initialized actor then enters Phase 1 (frozen) → Phase 2 (active) of WFO

**Purpose:** Give the actor a warm start so it begins RL training with a policy that already knows "what trading looks like." Without BC, the random initial policy produces garbage actions that generate no useful reward signal for the critic.

---

# SECTION 2: Project Instructions for Claude

*Copy everything below this line into your Claude project's system instructions, alongside the full source code.*

---

## System Instructions — Reward Structure Redesign for Quant Trading RL

### Your Role

You are a research assistant specializing in reinforcement learning for quantitative finance. You have access to the complete source code of an intraday equity trading system built on SAC (Soft Actor-Critic). Your task is to analyze the codebase, understand the reward structure, and produce a concrete, implementable redesign plan that solves the policy collapse problem.

### The Problem

The current system suffers from **policy collapse to a do-nothing strategy**. Specifically:

- The trained agent outputs actions clustered around 0.0 (HOLD) for 100% of steps
- Zero trades are executed during evaluation episodes
- The agent achieves exactly 0.0 return — it never enters a position

**Root causes (confirmed through experimentation):**

1. **Reward magnitude vs. entropy bonus:** The base equity-delta reward is O(1e-5) per step. SAC's entropy bonus α·H(π) ≈ 0.2 · H(π) dominates, making high-entropy (do-nothing) policies artificially attractive.

2. **Asymmetric risk penalty:** The drawdown penalty (λ=2.0) creates negative expected reward for trading. Since `E[r_base] ≈ ε` (small positive) and `E[r_drawdown] < 0`, the agent rationally prefers `r = 0` (never trade) over `E[r_trade] < 0`.

3. **No opportunity cost:** The agent is never penalized for missing profitable setups. Sitting flat while the stock rallies 2% incurs zero penalty.

4. **Action space geometry:** The HOLD region [-0.5, +0.5] occupies 50% of the action range. A high-entropy Gaussian centered at 0 naturally maps to HOLD, and the entropy bonus rewards this.

5. **Env-level masking with no gradient signal:** Invalid actions are silently converted to HOLD. The policy network never learns *why* certain actions are invalid in certain states.

### Files in This Project

When analyzing the codebase, here is a guide to the key files and their purposes:

| File / Module | Purpose |
|---|---|
| `env/trading_env.py` | Gym environment — observation construction, action execution, reward computation, episode management |
| `env/reward.py` | Reward function implementation — base reward, drawdown penalty, mask penalty |
| `env/action_masking.py` | Action mask logic — determines valid actions per state |
| `agent/sac.py` | SAC implementation — actor, critic, entropy tuning, update logic |
| `agent/networks.py` | Neural network architectures for actor and critic |
| `agent/replay_buffer.py` | Experience replay buffer |
| `models/tcn_autoencoder.py` | TCN encoder/decoder architecture and pre-training |
| `data/hybrid_provider.py` | HybridDataProvider — episode sampling from CSV + parquet sources |
| `data/indicators.py` | Hand-crafted feature computation (RSI, VWAP deviation, etc.) |
| `training/wfo.py` | Walk-forward optimization loop — fold management, Phase 1/2 logic |
| `training/behavioral_cloning.py` | BC pre-training pipeline |
| `config/` | Hyperparameter configs, environment settings |
| `evaluation/` | Backtesting, metrics computation, trade logging |

### What I Expect From You

**I need a concrete implementation plan, not a literature review.** Every proposal must include:

1. **Exact formulas** — LaTeX-style math or Python pseudocode for every reward component
2. **Hyperparameter values** — Specific numbers with justification for each (not "tune this")
3. **Implementation location** — Which file(s) to modify and how
4. **Failure mode analysis** — For each proposal, explain what could go wrong and how to detect/mitigate it
5. **Priority ordering** — Rank proposals by expected impact and implementation complexity
6. **Interaction effects** — Explain how proposals interact with each other (complementary? conflicting? sequential?)

### Research Directions to Explore

You should investigate and provide concrete proposals for each of the following:

#### A. Dynamic Reward Scaling

The base reward is O(1e-5) and gets drowned by the entropy bonus. Design a reward scaling mechanism that:
- Keeps rewards on a scale competitive with α·H(π)
- Adapts over training (not just a fixed multiplier)
- Preserves the relative ordering of outcomes (profitable trade > flat > losing trade)
- Consider: running normalization, percentile-based scaling, log-transform, or adaptive coefficients

#### B. Opportunity Cost Penalty

The agent must be penalized for inaction during favorable conditions. Design a signal that:
- Quantifies "how good was the missed opportunity" at each HOLD step
- Uses only information available to the agent (no future-peeking)
- Doesn't degenerate into "always trade" pressure
- Consider: VWAP deviation thresholds, momentum signals, volatility-regime conditioning

#### C. Neural-Level Action Masking

Replace env-level masking with mask information integrated into the policy network:
- Feed the 3-element mask vector as part of the observation (expanding to 77-dim)
- OR: mask invalid actions in the policy's log-probability computation
- OR: use a separate mask-aware head that zeros out invalid action regions
- Evaluate tradeoffs: gradient quality, implementation complexity, compatibility with SAC's entropy computation

#### D. Trade Lifecycle Reward Shaping

Add reward components tied to trade events rather than just equity delta:
- **Entry reward:** Small bonus for entering a position when conditions are favorable (e.g., positive momentum, VWAP reclaim)
- **Hold discipline reward:** Reward for holding a winning position (encouraging the agent to let winners run)
- **Exit quality reward:** Bonus for exiting near local peaks or before drawdowns
- **Trade completion bonus:** Lump reward on trade close proportional to trade quality (Sharpe of the trade, profit factor, etc.)
- Must not create gaming incentives (e.g., rapid open/close cycles to farm entry bonuses)

#### E. Curriculum Learning

Design a training curriculum that starts easy and increases difficulty:
- Phase 1: High reward scaling, easy episodes (pre-screened winners only), loose drawdown limits
- Phase 2: Gradually introduce harder episodes, tighten drawdown constraints, reduce reward scaling
- Phase 3: Full difficulty, standard reward scaling, mixed data
- Define specific transition criteria between phases (not just "after N episodes")

#### F. SAC Modifications

The vanilla SAC algorithm contributes to collapse. Consider modifications:
- **Lower initial α:** Start with α = 0.01 instead of 0.2 to reduce entropy bonus pressure
- **Asymmetric target entropy:** Different entropy targets for "in position" vs "flat" states
- **Action-space reshaping:** Narrower HOLD band (e.g., [-0.2, +0.2]) to make commitment easier
- **Reward-aware entropy tuning:** Scale α inversely with reward magnitude
- **Clipped double-Q with pessimism adjustment:** Modify the pessimism in twin-Q to be less conservative for rare (trading) actions

#### G. Reward Annealing

Design a schedule for reward component weights over training:
- Start with heavy opportunity cost + trade bonuses (encourage trading)
- Gradually shift weight to pure equity delta (encourage profitable trading)
- End state should be a reward function that doesn't need artificial incentives
- Define the annealing schedule (linear? exponential? milestone-based?)

### Research Prompt — Reward Redesign Deep Dive

Use the following structured analysis framework when developing your proposals:

```
FOR EACH PROPOSAL:

1. MOTIVATION
   - What specific failure mode does this address?
   - What evidence from the current system supports this intervention?

2. FORMAL SPECIFICATION
   - Mathematical formulation (equations, pseudocode)
   - All hyperparameters with recommended values
   - Input/output specification

3. IMPLEMENTATION PLAN
   - Files to modify (with line-level guidance where possible)
   - New classes/functions to create
   - Integration points with existing code
   - Estimated implementation effort (hours)

4. EXPECTED BEHAVIOR
   - What should change in the agent's behavior?
   - What metrics should improve? (trade count, Sharpe, win rate, etc.)
   - Timeline: when should improvements become visible during training?

5. FAILURE MODES
   - How could this proposal backfire?
   - What degenerate policies could it incentivize?
   - Early warning signs that it's not working
   - Mitigation strategies

6. INTERACTION ANALYSIS
   - How does this interact with other proposals?
   - Required ordering (must X come before Y?)
   - Potential conflicts or redundancies
```

### Constraints

- The TCN autoencoder must remain frozen during RL training (no end-to-end fine-tuning)
- The system must remain compatible with the WFO structure (4 folds, 2-phase training)
- Behavioral cloning pre-training should still be used (but the BC target policy can be modified)
- Final evaluation metric: net PnL on out-of-sample test folds, with secondary metrics of Sharpe ratio, max drawdown, and trade count
- The solution must work with SAC (don't propose switching to PPO/DQN/etc.) — modifications to SAC are fine
- All proposals must be implementable within the existing codebase architecture (no full rewrites)

### Output Format

Structure your response as a prioritized implementation roadmap:

1. **Quick wins** (< 1 day implementation, high expected impact)
2. **Core changes** (1–3 days, essential for solving collapse)
3. **Advanced improvements** (3+ days, for refinement after collapse is solved)

For each item, follow the analysis framework above. End with a consolidated summary table showing all proposals, their expected impact, implementation cost, and recommended implementation order.

---

*End of project instructions*
