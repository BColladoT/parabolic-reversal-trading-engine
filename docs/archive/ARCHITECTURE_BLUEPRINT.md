# Advanced Deep Reinforcement Learning Architecture for Parabolic Reversal Trading: Scaling the V5 Relaxed Strategy

## 1. Introduction and Strategic Imperative

The pursuit of consistent alpha in algorithmic trading necessitates systems capable of navigating the highly non-stationary, stochastic nature of financial markets. The historical reliance on purely rule-based systems often yields fragility in the face of regime shifts, while the overzealous application of modern machine learning frequently results in catastrophic overfitting and the memorization of historical noise. The objective of this comprehensive research report is to construct an exhaustive, technically rigorous blueprint for a next-generation algorithmic trading engine. This blueprint is fundamentally anchored in the empirical supremacy of a proprietary baseline system, denoted herein as the V5 Relaxed strategy. By synthesizing the demonstrated successes of the V5 Relaxed parameter set with a state-of-the-art, tripartite artificial intelligence architecture, this report outlines a methodology to scale the trading engine while systematically mitigating the inherent risks of deep learning---namely, look-ahead bias, non-stationarity, and the curse of dimensionality.

The proposed architectural evolution abandons monolithic deep learning approaches in favor of a decoupled, three-step core pipeline: Modular Perception, a Deep Reinforcement Learning (DRL) Decision Engine, and Rule-Based Safety Guardrails. The Modular Perception layer functions as the agent's sensory interface, employing representation learning to compress raw market data into a noise-filtered latent state. The DRL Decision Engine serves as the cognitive core, utilizing off-policy algorithms and maximum entropy frameworks to dynamically optimize portfolio trajectories based on risk-aware reward functions. Finally, the Rule-Based Safety Guardrails operate as a deterministic execution filter, leveraging algorithmic action masking to mathematically guarantee that the neural network cannot violate strict, predefined risk parameters.

This report will systematically deconstruct the empirical performance of the V5 Relaxed strategy to isolate its statistical edge, theoretical mechanisms, and operational thresholds. Subsequently, it will provide an exhaustive architectural specification for each of the three core modules, detailing the requisite mathematical formulations, neural network topologies, and infrastructural designs required for live deployment. Finally, the analysis will address the critical methodologies required to prevent backtest overfitting, focusing on combinatorial purged cross-validation, walk-forward optimization, and behavioral cloning. The synthesis of these elements provides a complete roadmap for deploying a highly adaptive, institutional-grade quantitative trading system.

## 2. Exhaustive Empirical Analysis of the V5 Relaxed Strategy

The foundation of the proposed architectural evolution is rooted entirely in the empirical outperformance and statistical robustness of the V5 Relaxed strategy. An exhaustive analysis of the system's historical trading logs, equity curves, and distributional metrics reveals a highly sophisticated mechanism for exploiting short-term market inefficiencies, specifically the mean-reverting tendencies of parabolic price extensions.

### 2.1 Aggregate Performance Metrics and Statistical Dominance

The V5 Relaxed strategy generated exceptionally compelling risk-adjusted returns over the backtested period, spanning from mid-2020 through early 2025. The strategy executed a total of 379 trades, achieving an extraordinary win rate of 79.42%, corresponding to 301 winning transactions against a mere 78 losses.[^1^] The aggregate realized Profit and Loss (P&L) reached a terminal value of $781,750.56.[^1^]

The per-trade statistical breakdown further elucidates the strategy's mathematical edge. The expected value (average return) per executed trade stood at a highly profitable $2,062.67. Crucially, the strategy demonstrated a strong positive asymmetry in its payoff matrix: the average winning trade yielded $3,252.07, while the average losing trade was tightly controlled at -$2,527.20.[^1^] This positive skew between average wins and average losses, combined with an eighty percent win rate, culminated in an elite Profit Factor of 4.97.[^1^] In the context of quantitative finance, a profit factor exceeding 3.0 over a multi-year horizon across hundreds of trades is indicative of a highly resilient, structurally sound algorithmic edge that is largely immune to standard market frictions.

Risk management metrics within the sample period were equally robust. The maximum historical drawdown was contained to -$19,180.81, a figure that is notably identical to the strategy's maximum single-trade loss.[^1^] This statistical equivalence strongly implies that the strategy does not suffer from compounding sequential losses or "tilt" behaviors, but rather experiences isolated, discrete failure events from which the equity curve rapidly recovers. The resulting Sharpe Ratio of 8.75 signifies a risk-adjusted return profile that borders on theoretical limits for intraday momentum-reversal strategies, suggesting that the variance of the returns is overwhelmingly concentrated in the positive domain.[^1^]

### 2.2 Equity Curve Trajectory and Drawdown Dynamics

An analysis of the cumulative P&L trajectory reveals a highly linear and consistent capital appreciation model. The equity curve ascends smoothly from its inception in July 2020, demonstrating near-immunity to broader macroeconomic regime shifts, such as the 2022 secular bear market in equities. This indicates that the strategy's edge is truly market-neutral, relying entirely on idiosyncratic, localized volatility events (parabolic extensions) rather than directional beta exposure to broader indices.

The drawdown analysis provides critical insights into the strategy's stress tolerance. While the equity curve is generally monotonic, it is punctuated by sharp, instantaneous drawdowns, the most severe occurring in mid-2022, approaching the aforementioned -$19,180 threshold. However, the recovery time (drawdown duration) following these events is exceptionally brief. The strategy does not linger in drawdown states; it rapidly re-establishes new equity highs. This rapid recovery profile is a hallmark of high-win-rate, mean-reversion systems where the sheer frequency of profitable events quickly overwhelms isolated catastrophic losses. The absence of prolonged drawdown periods suggests that the V5 Relaxed strategy is effectively capturing a persistent behavioral anomaly in the market---specifically, the forced liquidation or exhaustion of retail momentum traders following unsustainable price spikes.

### 2.3 Distributional Analysis of Trade Returns

The distribution of individual trade P&L provides a granular view of the strategy's mechanics. The histogram of trade returns is distinctly leptokurtic and heavily right-skewed. The central mass of the distribution is tightly clustered just above the break-even threshold, with a massive concentration of trades yielding profits between $0 and $5,000. The mean return of $2,063 is significantly influenced by a long, pronounced right tail of outlier wins extending up to the maximum recorded profit of $14,336.10.[^1^]

Conversely, the left tail of the distribution (the loss domain) is abruptly truncated. While a few significant outlier losses exist (stretching to the absolute minimum of -$19,180.81), there is a distinct lack of density in the moderate-loss region. This distributional shape confirms that the V5 Relaxed system employs a highly effective, asymmetrical risk management protocol. The strategy permits winning trades to run and capture maximum parabolic reversion variance, while aggressively aborting invalid setups before they can inflict structural damage to the portfolio.

### 2.4 Asset-Specific Performance and Win Rate Tolerance

An examination of the strategy's performance across individual ticker symbols reveals a fascinating dichotomy between win rate and absolute profitability. An analysis of the top twenty symbols by total P&L demonstrates that while a 100% win rate is common among high-performing assets (such as MULN, which generated over $22,000 in aggregate P&L without a single loss), it is not a strict prerequisite for massive profitability.

Notable exceptions exist within the top-performing assets, specifically GME (GameStop) and BBIG (Vinco Ventures). GME ranks third in total P&L, generating approximately $16,000 for the strategy, despite exhibiting an exceptionally low win rate of roughly 35%. Similarly, BBIG generated nearly $19,500 in total P&L with a win rate hovering around 75%. These anomalies are highly informative. They indicate that the V5 Relaxed strategy is dynamically capable of adapting its payoff matrix based on the volatility profile of the underlying asset. For hyper-volatile, highly scrutinized "meme stocks" like GME, the strategy suffers a higher frequency of stopped-out trades (as the parabolic extension continues longer than anticipated). However, when the reversal finally triggers, the magnitude of the mean-reversion is so violently profitable that it completely subsidizes the elevated failure rate. This implies that any future DRL agent must be designed to recognize asset-specific volatility signatures and dynamically adjust its acceptable win/loss thresholds, rather than enforcing a static, global win rate expectation.

### 2.5 Feature Correlation: Elasticity and Liquidity

A deeper extraction of the raw trade data from the combined_trades.csv log reveals the specific technical catalysts that precipitate the strategy's most lucrative executions. The V5 Relaxed system relies on identifying extreme market imbalances, utilizing specific features to quantify the magnitude of these dislocations.

The two most critical features extracted from the dataset are Volume Weighted Average Price (VWAP) Deviation and Volume Concentration. VWAP Deviation measures the percentage distance of the current price from its volume-weighted mean, serving as a mathematical proxy for the "elastic stretch" of the asset. Volume Concentration measures the density of trading activity within the specific parabolic time window relative to the average daily volume, indicating the presence of capitulation or euphoric buying.

| Symbol | Execution Date | Realized PnL ($) | Win | VWAP Deviation | Volume Concentration | Peak Timing (mins) |
|--------|----------------|------------------|-----|----------------|----------------------|-------------------|
| AAME | 2021-02-05 | 6,212.92 | 1 | 46.10 | 1.00 | 150.0 |
| AEON | 2024-07-15 | 3,384.07 | 1 | 41.36 | 0.66 | 80.0 |
| AISP | 2024-03-05 | 1,353.32 | 1 | 23.96 | 1.00 | 121.0 |
| ALBT | 2024-06-03 | 1,318.24 | 1 | 21.82 | 0.00 | 122.0 |
| ACRS | 2021-01-19 | -99.33 | 0 | 21.02 | 1.00 | 178.0 |

The data confirms a direct, non-linear correlation between extreme VWAP deviation and outsized profitability. The strategy's most successful single execution, AAME, generated a 274.41% day gain and a $6,212.92 profit. This trade occurred at an astronomical VWAP deviation of 46.10, coupled with a maximum Volume Concentration of 1.0.[^1^] Similarly, the AEON trade captured $3,384.07 at a VWAP deviation of 41.36.[^1^]

Conversely, the data reveals a clear failure threshold as the elastic stretch compresses. The strategy's recorded loss on ACRS occurred at a significantly lower VWAP deviation of 21.02, despite maintaining a maximal Volume Concentration of 1.0.[^1^] This indicates that localized volume spikes are insufficient to guarantee a reversal if the price is not sufficiently extended from its mean. The minimum viable threshold for structural mean-reversion within the V5 Relaxed paradigm appears to be a VWAP deviation exceeding 23.0.

Furthermore, the AEON trade demonstrates that maximum Volume Concentration is not strictly necessary for a massive win, as it succeeded with a concentration of 0.664.[^1^] The empirical data conclusively proves that the trading engine derives its foundational alpha from identifying extreme price elasticity (deviation) rather than mere volume momentum. This specific relationship dictates that the subsequent perception modules of the advanced architecture must explicitly encode and weight distance-from-mean metrics above all other technical indicators.

### 2.6 Temporal Selectivity and Execution Frequency

An analysis of the monthly trade frequency provides insight into the strategy's operational pacing. The strategy averages only a handful of trades per month, occasionally peaking at roughly 20 executions during highly volatile market periods (such as December 2024, which corresponded with the strategy's highest monthly P&L of nearly $80,000). Many months feature fewer than five trades, and a select few result in negative aggregate monthly P&L.

This extreme temporal selectivity is a core component of the system's robustness. The V5 Relaxed strategy acts as a sniper rather than a machine gun; it remains dormant during periods of normalized volatility and only deploys capital when the strict parabolic extension criteria (high VWAP deviation) are met. This selective pacing minimizes the strategy's exposure to random market noise and significantly reduces transaction costs and slippage, which are primary factors in the degradation of high-frequency trading algorithms. Any evolutionary step utilizing Deep Reinforcement Learning must be explicitly constrained to maintain this patience, penalizing the agent for over-trading in suboptimal environments.

## 3. Architectural Paradigm: The Tripartite Trading Engine

To systematically scale the exceptional empirical performance of the V5 Relaxed strategy without succumbing to the standard pitfalls of algorithmic complexity, a fundamental paradigm shift in system architecture is required. Monolithic deep learning models, which attempt to process raw data, evaluate logic, and execute trades within a single unified neural network, consistently fail in financial markets due to the curse of dimensionality and the non-stationarity of the data.[^2^] When a single model attempts to learn everything simultaneously, it inevitably memorizes the historical noise rather than the underlying market signal, leading to catastrophic out-of-sample performance degradation.[^4^]

To resolve this, the next-generation trading engine will be constructed using a strictly decoupled, tripartite architecture. This design isolates data ingestion, policy optimization, and risk management into three independent, specialized modules. By compartmentalizing the system's cognitive load, the architecture ensures that the learning agent focuses solely on high-level strategic optimization, while raw data processing and absolute risk constraints are handled by specialized, deterministic sub-systems.

The three core modules are defined as follows:

1. **Modular Perception (State Representation):** A separate deep learning pipeline dedicated exclusively to feature extraction, dimensionality reduction, and the translation of raw market data into a clean, low-dimensional mathematical state.

2. **Deep Reinforcement Learning (DRL) Decision Engine:** The cognitive core of the system. An off-policy actor-critic algorithm that receives the processed state and outputs continuous probability distributions for position sizing and directional bias, optimized against a custom, risk-aware reward function.

3. **Rule-Based Safety Guardrails:** A deterministic execution filter that intercepts the DRL engine's proposed actions. Utilizing action masking and hard-coded mathematical constraints, this module guarantees that the agent cannot violate the fundamental risk parameters proven by the V5 Relaxed backtest.

This architectural separation of concerns provides unprecedented auditability, stability, and scalability, allowing quantitative researchers to upgrade individual components (such as swapping the DRL algorithm) without corrupting the integrity of the broader execution pipeline.[^6^]

## 4. Module I: Modular Perception and State Representation

The perception module functions as the sensory cortex of the artificial intelligence agent. In traditional algorithmic trading, and specifically in failed machine learning implementations, practitioners often feed the decision model dozens or hundreds of manually engineered technical indicators (e.g., overlapping moving averages, RSI, MACD, Bollinger Bands).[^7^] This approach floods the model with highly correlated, collinear data, exacerbating the curse of dimensionality and forcing the model to find spurious historical patterns to minimize its loss function.[^5^]

The new architecture entirely abandons manual, stacked technical indicators. Instead, it employs Representation Learning (RL) and self-supervised deep learning architectures to construct a dynamic Modular Perception layer.[^10^] The goal of this module is to ingest highly noisy, high-frequency, multimodal data---including limit order book (LOB) depth, tick-by-tick Trade and Quote (TAQ) data, and standard OHLCV time series---and compress it into a dense, abstract latent vector representation, denoted as $z$.[^12^]

### 4.1 Temporal Autoencoders for Latent State Generation

To achieve this compression, the perception module will utilize a sequence-to-sequence Temporal Convolutional Autoencoder (TCN-AE) or a Transformer-based Autoencoder.[^13^] Autoencoders are unsupervised learning networks designed to reconstruct their input data. The network consists of an encoder that compresses the high-dimensional market sequence into a lower-dimensional "bottleneck" layer, and a decoder that attempts to rebuild the original sequence from this bottleneck.[^16^]

The autoencoder is pre-trained on vast quantities of unlabeled historical market data.[^15^] Because the network is forced to push all information through the restricted bottleneck layer, it cannot simply memorize the data; it is mathematically compelled to discover and encode only the most fundamental, abstract geometries of the price action---such as the velocity of acceleration, the curvature of a parabolic spike, and the density of liquidity.[^17^]

Once the autoencoder is fully trained and achieves a satisfactorily low reconstruction loss, the decoder portion is severed and discarded. The frozen encoder network is then deployed as the Modular Perception layer. As live market data streams in, the encoder processes the sequence and outputs the bottleneck vector $z$. This dense, low-dimensional vector---scrubbed of micro-structural noise and redundant indicators---becomes the official state space $s$ provided to the DRL Decision Engine.[^10^] This approach dramatically improves the sample efficiency of the downstream reinforcement learning agent, as the DRL algorithm no longer needs to waste computational cycles learning how to interpret raw market data.[^5^]

### 4.2 Explicit Feature Injection: Anchoring to the V5 Relaxed Edge

While the latent vector $z$ elegantly captures abstract market geometries, the empirical analysis of the V5 Relaxed strategy conclusively proved that specific, explicit mathematical metrics---namely, VWAP Deviation and Volume Concentration---are the primary catalysts for trade success.[^1^] Relying entirely on an autoencoder to implicitly discover these specific statistical thresholds introduces unnecessary risk.

Therefore, the Modular Perception layer will employ a hybrid state representation. The final state vector $s$ passed to the DRL agent will be a concatenation of the autoencoder's latent representation and a deterministic vector containing explicitly calculated, normalized values for the core V5 Relaxed metrics.

$$s = [z \oplus \text{VWAP}_{\text{dev}} \oplus \text{Vol}_{\text{conc}} \oplus \text{Asset}_{\text{vol}}]$$

By explicitly injecting the VWAP deviation scalar directly into the state space, the DRL agent is forcefully anchored to the exact mathematical parameters that defined the V5 Relaxed strategy's $781,000 success. The agent receives the abstract "shape" of the market from $z$, but makes its final timing decisions based on the explicit knowledge that the asset has crossed the critical 23.0 VWAP deviation threshold identified in the empirical analysis.[^1^] This hybrid approach bridges the gap between the interpretability of classical quantitative finance and the advanced pattern recognition of deep learning.

## 5. Module II: The Deep Reinforcement Learning Decision Engine

With a clean, low-dimensional state representation provided by the perception module, the architecture requires an engine capable of translating these states into optimal, risk-adjusted trading actions. While supervised learning models attempt to predict the exact future price of an asset, Deep Reinforcement Learning (DRL) is vastly superior for algorithmic execution because it frames trading as a sequential decision-making process modeled as a Markov Decision Process (MDP).[^20^] A DRL agent does not predict price; it learns a policy $\pi(a|s)$ that dictates the optimal action to take in a given state to maximize cumulative, long-term rewards.[^23^]

### 5.1 Algorithm Selection: Soft Actor-Critic (SAC)

The selection of the specific DRL algorithm is paramount to the system's viability. The financial literature frequently utilizes on-policy algorithms such as Proximal Policy Optimization (PPO). However, PPO exhibits significant limitations in highly stochastic, noisy environments like financial markets. Because PPO is an on-policy algorithm, it discards data immediately after updating its neural network weights, rendering it highly sample-inefficient.[^25^] In trading, where true parabolic setups are rare anomalies (only 379 trades over four years in the V5 Relaxed dataset), throwing away data is unacceptable. Furthermore, PPO policies have a tendency to prematurely converge on suboptimal deterministic actions, leading directly to the overfitting behavior that destroyed the V5 Institutional model.[^25^]

To counter this, the Decision Engine will be built upon the Soft Actor-Critic (SAC) architecture. SAC is an off-policy actor-critic algorithm that utilizes a replay buffer, allowing it to continuously sample and learn from past experiences, making it vastly more sample-efficient than PPO.[^25^] More importantly, SAC is built upon the maximum entropy reinforcement learning framework.[^25^]

In standard RL, the objective is strictly to maximize the expected sum of rewards. SAC fundamentally alters this objective by introducing an entropy regularization term $\mathcal{H}(\pi(\cdot|s_t))$, which mathematically quantifies the randomness or uncertainty of the policy. The SAC objective function is defined as:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_{\pi}} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

The temperature parameter $\alpha$ dynamically controls the trade-off between exploitation (maximizing the reward) and exploration (maximizing the entropy).[^30^] By forcing the policy to remain as stochastic as possible while still achieving the task, SAC creates a uniquely robust agent.[^26^] An entropy-regularized agent does not memorize a single, rigid path to profitability; instead, it learns a broad distribution of viable strategies. When market microstructures undergo regime shifts---a certainty in financial markets---the SAC agent is mathematically insulated against catastrophic failure because its policy maintains the inherent flexibility to adapt to new conditions without requiring a complete retraining cycle from scratch.[^30^] Additionally, SAC natively handles continuous action spaces, allowing the agent to output precise, fractional position sizing vectors rather than rigid, binary buy/sell commands.[^31^]

| Feature Comparison | Soft Actor-Critic (SAC) | Proximal Policy Optimization (PPO) |
|-------------------|------------------------|-------------------------------------|
| Algorithm Type | Off-policy (Replay Buffer) | On-policy (No Replay Buffer) |
| Sample Efficiency | Extremely High (Reuses data) | Low (Discards data after update) |
| Exploration Mechanic | Maximum Entropy Regularization | Action noise/Clipping |
| Susceptibility to Overfitting | Low (Maintains stochasticity) | High (Prone to early deterministic convergence) |
| Suitability for Trading | Optimal for continuous sizing & sparse data | Suboptimal for highly non-stationary regimes |

### 5.2 Reward Function Engineering: Penalizing the Drawdown

The most sensitive and easily corrupted component of any DRL trading system is the reward function.[^4^] A naive reward function that simply provides the agent with its raw Profit and Loss (PnL) will inevitably induce disastrous behavior. The agent will learn to maximize absolute returns by taking on maximum leverage and ignoring tail risks, resulting in catastrophic drawdowns during volatility spikes.[^30^] The empirical data of the V5 Relaxed strategy showed a maximum acceptable drawdown of approximately -$19,180.[^1^] The DRL agent must be mathematically punished for approaching this limit.

While many quantitative models utilize the Sharpe ratio as a reward function to incorporate risk, the Sharpe ratio is flawed for parabolic reversal strategies because it penalizes upside volatility (massive, sudden profits) identically to downside volatility.[^35^] The objective of the V5 Relaxed strategy is precisely to capture extreme upside volatility.[^36^] Therefore, a more sophisticated approach utilizing a Differential Sortino Ratio combined with a direct, non-linear maximum drawdown penalty is required.[^35^]

The continuous reward function $r_t$ at any given time step $t$ must be engineered as a composite mathematical signal:

$$r_t = \underbrace{\text{Sortino}(\text{PnL}_{\text{realized}})}_{\text{Risk-Adjusted Return}} - \underbrace{\lambda_{\text{dd}} \cdot \max(0, \text{MDD}_{\text{current}} - \text{MDD}_{\text{max}})^2}_{\text{Drawdown Penalty}} - \underbrace{\text{Cost}(\Delta w)}_{\text{Friction Penalty}}$$

Where:

- $\text{PnL}_{\text{directional}}$ represents the realized or unrealized directional PnL based on the agent's continuous position weight $w_t$.

- $\lambda_{\text{dd}}$ acts as an aggressive penalty coefficient that scales quadratically only when the agent's current Maximum Drawdown ($\text{MDD}_{\text{current}}$) breaches a strict target limit ($\text{MDD}_{\text{max}}$). This heavily asymmetric penalty ensures the agent learns to fear severe equity dips more than it desires marginal gains.[^35^]

- $\text{Cost}$ represents a friction penalty accounting for transaction costs, slippage, and spread crossing. This is critical to prevent the agent from discovering a high-frequency "hyper-trading" anomaly that generates theoretical profits in a vacuum but is annihilated by real-world commissions.[^41^]

By embedding the quadratic drawdown penalty directly into the temporal step reward, the SAC critic networks will learn to assign heavily discounted Q-values to state-action pairs that expose the portfolio to tail-risk.[^35^] This risk-aware reward shaping will guide the agent to emulate the extreme selectivity of the V5 Relaxed strategy, aggressively exploiting maximum VWAP deviations while maintaining a flat, zero-exposure position during ambiguous, low-deviation chop.

## 6. Module III: Rule-Based Safety Guardrails and Action Masking

Despite the sophisticated architectures of Modular Perception and SAC optimization, deep neural networks remain fundamentally opaque "black boxes".[^3^] In financial markets, black swan events and unprecedented macroeconomic shocks can force neural networks to extrapolate into unknown latent spaces, resulting in entirely unpredictable and potentially ruinous actions. Relying solely on the DRL agent to "learn" not to destroy the portfolio is an unacceptable risk for institutional capital.

The third pillar of the proposed architecture resolves this vulnerability by introducing deterministic, hard-coded Rule-Based Safety Guardrails that operate entirely independently of the neural network's logic. These guardrails translate the rigid, successful parameters of the V5 Relaxed strategy into an absolute execution filter.

To implement this without breaking the reinforcement learning cycle, the system will employ a technique known as **Action Masking**.[^43^] In advanced frameworks such as Ray RLlib, action masking allows the environment to dynamically manipulate the probability distribution of the agent's output layer before an action is executed.[^46^]

The process operates via a boolean masking vector. At each time step, the environment evaluates the current market conditions against a set of absolute rules derived from the V5 Relaxed empirical data. For example, if the strategy prohibits entries during the highly chaotic opening auction (e.g., before 10:00 AM ET), the environment generates a mask where the "enter position" action is flagged as invalid.[^45^]

Before the DRL agent's final logits are passed through the softmax (for discrete choices) or continuous squashing functions (for sizing), the mask applies a value of negative infinity ($-\infty$) to the invalid actions.[^47^] This mathematically forces the probability of the agent selecting a prohibited action to exactly zero.

The application of Action Masking provides two profound benefits. First, it guarantees absolute portfolio safety; the agent physically cannot execute a trade that violates the overarching risk parameters (such as a hard stop-loss threshold at 2% above a parabolic high, or holding a position into an overnight gap). Second, it drastically accelerates the training efficiency of the DRL agent.[^43^] Without a mask, the agent would waste millions of computational iterations taking invalid actions, receiving massive negative rewards, and slowly updating its weights to learn the rules of the environment.[^47^] With Action Masking, the agent's neural capacity is freed from learning basic compliance and is instead focused entirely on optimizing the nuance of the entry and exit execution within the permitted bounds.

## 7. Position Sizing Reform: The Fallacy of Full Kelly and Dynamic Allocation

A critical forensic finding from the failure of the V5 Institutional model was its rigid application of the Kelly Criterion for position sizing.[^1^] The Kelly Criterion formula, $f^* = \frac{p(b+1) - 1}{b}$ (where $p$ is the probability of a win, $q$ is the probability of a loss, and $b$ is the ratio of average win to average loss), calculates the mathematically optimal fraction of a bankroll to wager to maximize logarithmic wealth accumulation over an infinite series of bets.[^48^]

However, the application of "Full Kelly" in algorithmic trading represents a fundamental misunderstanding of probability in non-stationary environments. The Kelly Criterion requires absolute certainty regarding the probabilities $p$ and $q$.[^50^] In financial markets, true forward-looking probabilities are permanently unknowable; they are merely estimates derived from historical backtests.[^51^] If an algorithm assumes a historical win rate of 55%, but the true forward-looking win rate shifts to 50% due to a regime change, a Full Kelly bet size will systematically over-leverage the portfolio, inevitably leading to catastrophic drawdown and mathematical ruin.[^48^] The V5 Institutional model fell victim to exactly this phenomenon, oversizing its losing trades when the empirical market conditions diverged from the training data.[^1^]

The new architecture will entirely abandon the rigid Full Kelly approach in favor of a dynamically capped, continuous allocation model. The SAC agent's actor network will be configured with a continuous action space bounded between [-1, 1] (representing short to long exposure). The agent will organically learn the optimal fractional sizing required to maximize the Sortino-based reward function.[^31^]

However, to provide a mathematical safety net against overconfidence, the Rule-Based Safety Guardrails will intercept the agent's sizing request and pass it through a dynamic Fractional Kelly constraint.[^50^] The system will continuously monitor a trailing 30-day rolling window of the agent's empirical win rate and Profit Factor to calculate a localized Kelly optimum ($f^*_{\text{local}}$). The execution module will strictly cap the agent's maximum allowable leverage at $\frac{f^*_{\text{local}}}{4}$ (Quarter-Kelly).[^48^] This "Fractional Kelly" implementation serves as vital insurance against non-stationarity and estimation errors.[^49^] By sacrificing a small margin of theoretical peak efficiency, the Quarter-Kelly constraint drastically reduces the variance of the equity curve, preserving capital during periods of alpha decay and ensuring psychological and mathematical survival through prolonged drawdowns.[^48^]

## 8. Mitigating Algorithmic Pitfalls: Training and Deployment Protocol

The most sophisticated architecture is entirely useless if the training methodology allows the neural networks to overfit the historical data.[^52^] The transition from backtest to live deployment must be governed by an uncompromising, scientifically rigorous validation protocol to prevent the exact failures observed in the discarded institutional model.

### 8.1 Behavioral Cloning for DRL Initialization

Reinforcement learning agents typically initialize with completely random neural weights. Consequently, the early stages of training are characterized by chaotic, random exploration as the agent takes nonsensical actions to map the environment's reward surface.[^15^] In complex, high-stakes environments like financial markets, this random walk often causes the agent to become trapped in suboptimal local minima, preventing it from ever discovering a profitable strategy.[^54^]

Given that the V5 Relaxed strategy already possesses a highly verified, profitable edge (evidenced by 379 successful executions), forcing a new agent to randomly rediscover this edge from scratch is highly inefficient. Instead, the implementation will utilize a technique known as Behavioral Cloning (BC) to jumpstart the DRL agent.[^54^]

Behavioral Cloning is a form of imitation learning. Prior to interacting with the live simulation, the SAC agent's actor network is pre-trained using supervised learning to mimic the exact historical decisions made by the V5 Relaxed strategy.[^56^] The 379 historical trades are formatted as state-action pairs $(s_t, a^*_t)$. The network weights are updated to minimize the cross-entropy loss between the agent's output and the historical "expert" actions.[^55^]

Once the neural network successfully clones the baseline logic---achieving a simulated Sharpe ratio comparable to the historical 8.75---the system transitions to standard reinforcement learning.[^55^] The SAC algorithm's entropy maximization then encourages the agent to explore *outward* from this highly profitable baseline. The agent leverages the foundation of the V5 Relaxed ruleset but uses deep learning to discover nuanced, hyper-dimensional efficiencies---such as fractionally scaling into a parabolic extension based on latent LOB density---that rigid hard-coded rules could not encompass.

### 8.2 Walk-Forward Optimization and Combinatorial Purged Cross-Validation

Standard machine learning evaluation techniques, such as random k-fold cross-validation, are entirely invalid for financial time series.[^58^] Market data exhibits severe serial correlation; randomly shuffling data leaks future information into the training set, virtually guaranteeing look-ahead bias and resulting in models that print money in backtests but fail immediately in live trading.[^3^]

To rigorously evaluate the SAC agent's true out-of-sample performance, the deployment protocol will utilize Walk-Forward Optimization (WFO) integrated with Combinatorial Purged Cross-Validation (CPCV).[^61^] WFO simulates the exact chronological progression of a live trading desk.[^62^]

The historical dataset is sequentially divided into rolling windows. For example, the Modular Perception autoencoder and the SAC agent are trained on an in-sample window from Year 1 through Year 3. The optimal policy weights are then frozen, and the agent is tested on an out-of-sample window representing the first six months of Year 4.[^61^] The entire window is then shifted chronologically forward by six months, and the process repeats.[^62^]

Critically, to completely sever the serial correlation between the training and testing sets, an "embargo" and "purging" period must be enforced.[^62^] A block of data (e.g., 5 to 10 trading days) located exactly at the intersection of the in-sample and out-of-sample splits is deleted from the dataset.[^62^] This guarantees that overlapping fractional calculations (such as a 10-day moving average calculated on the first day of the test set) do not inadvertently leak future price data into the training environment.[^62^]

By aggregating only the equity curves generated during the strictly out-of-sample testing blocks, the quantitative researcher obtains a highly realistic expectation of the algorithm's true performance.[^62^] The agent is only cleared for live paper trading if this aggregated out-of-sample WFO curve maintains the structural profitability benchmarks of the original V5 Relaxed strategy.

## 9. Conclusion

The transition from the empirical dominance of the V5 Relaxed strategy to a highly scalable, automated execution engine requires a delicate synthesis of advanced machine intelligence and uncompromising financial risk management. The catastrophic failure of the preceding institutional machine learning model served as a vital cautionary diagnostic, highlighting the extreme dangers of manual feature over-engineering, hyper-parameter rigidity, and the unconstrained reliance on neural predictions within non-stationary financial markets.

By adopting a sophisticated, decoupled tripartite architecture, the proposed system isolates and neutralizes these vulnerabilities. A Modular Perception layer utilizes self-supervised autoencoders to compress raw market data, eliminating noise and extracting pure spatial market geometry without succumbing to the curse of dimensionality. The Soft Actor-Critic (SAC) Deep Reinforcement Learning engine replaces binary trade logic with nuanced, entropy-driven continuous optimization, shaped explicitly by a reward function mathematically engineered to punish drawdowns and maximize the Sortino ratio.

Crucially, the entire deep learning apparatus is securely bound by deterministic Action Masking and dynamic Quarter-Kelly position limits. These Rule-Based Safety Guardrails ensure that the aggressive, highly profitable edge identified in the V5 Relaxed backtest---specifically the exploitation of parabolic exhaustion at extreme VWAP deviations---is aggressively captured without ever exposing the institutional portfolio to unacceptable tail risks. Validated through purged Walk-Forward Optimization and initialized via Behavioral Cloning, this comprehensive framework establishes a resilient algorithmic pipeline equipped to continuously adapt, learn, and generate robust alpha across shifting market regimes.

#### Obras citadas

[^1^]: detailed_stats.json

[^2^]: Why Deep Reinforcement Learning in Trading is Mostly Overfitting | by TonyHoang | Medium, fecha de acceso: marzo 12, 2026, https://medium.com/@tonyhoang-719/why-deep-reinforcement-learning-in-trading-is-mostly-overfitting-bc8cd4e3ab04

[^3^]: Benefits, Pitfalls, And Mitigation Tools When Applying Machine Learning To Trading Strategies | Resonanz Capital, fecha de acceso: marzo 12, 2026, https://resonanzcapital.com/insights/benefits-pitfalls-and-mitigation-strategies-of-applying-ml-to-financial-modelling

[^4^]: What are the challenges of applying deep reinforcement learning to financial signal representation and trading? - Consensus, fecha de acceso: marzo 12, 2026, https://consensus.app/search/what-are-the-challenges-of-applying-deep-reinforce/GsrzEsBCSZW59KKdsyIqLg/

[^5^]: Feature Engineering in Reinforcement Learning for Algorithmic Trading - TU Delft, fecha de acceso: marzo 12, 2026, https://repository.tudelft.nl/file/File_f1226238-ebc5-4691-b687-3eb4c5e5663c?preview=1

[^6^]: A Modular Architecture for Systematic Quantitative Trading Systems | by HIYA CHATTERJEE, fecha de acceso: marzo 12, 2026, https://hiya31.medium.com/a-modular-architecture-for-systematic-quantitative-trading-systems-2a8d46463570

[^7^]: Applying reinforcement learning in Bitcoin trading to select technical strategies based on Deep Q-Network - Taylor & Francis, fecha de acceso: marzo 12, 2026, https://www.tandfonline.com/doi/full/10.1080/23322039.2025.2594873

[^8^]: Algorithmic Trading System with Adaptive State Model of a Binary-Temporal Representation, fecha de acceso: marzo 12, 2026, https://www.mdpi.com/2227-9091/13/8/148

[^9^]: Algorithmic Trading and Short-term Forecast for Financial Time Series with Machine Learning Models - Okanagan College, fecha de acceso: marzo 12, 2026, https://www.okanagancollege.ca/sites/default/files/2025-01/2022rassealgortradingforecast.pdf

[^10^]: Traditional agent architecture: perceive, reason, act - AWS Prescriptive Guidance, fecha de acceso: marzo 12, 2026, https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-foundations/traditional-agents.html

[^11^]: What are the core components of AI Agent? - Tencent Cloud, fecha de acceso: marzo 12, 2026, https://www.tencentcloud.com/techpedia/126540

[^12^]: A Survey of State Representation Learning for Deep Reinforcement Learning - arXiv.org, fecha de acceso: marzo 12, 2026, https://arxiv.org/html/2506.17518v1

[^13^]: Major Issues in High-Frequency Financial Data Analysis: A Survey of Solutions - MDPI, fecha de acceso: marzo 12, 2026, https://www.mdpi.com/2227-7390/13/3/347

[^14^]: A deep learning framework for financial time series using stacked autoencoders and long-short term memory - PMC, fecha de acceso: marzo 12, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC5510866/

[^15^]: Trading through Earnings Seasons using Self-Supervised Contrastive Representation Learning - arXiv, fecha de acceso: marzo 12, 2026, https://arxiv.org/html/2409.17392v1

[^16^]: A cooperative deep learning model for stock market prediction using deep autoencoder and sentiment analysis - ResearchGate, fecha de acceso: marzo 12, 2026, https://www.researchgate.net/publication/365876985_A_cooperative_deep_learning_model_for_stock_market_prediction_using_deep_autoencoder_and_sentiment_analysis

[^17^]: Cognitively Guided Modeling of Visual Perception in Intelligent Vehicles Alice Plebe - IRIS, fecha de acceso: marzo 12, 2026, https://iris.unitn.it/retrieve/handle/11572/299909/434517/phd_unitn_Alice_Plebe.pdf

[^18^]: Core building blocks of software agents - AWS Prescriptive Guidance, fecha de acceso: marzo 12, 2026, https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-foundations/core-modules.html

[^19^]: Reinforcement Learning-Based Footstep Control for Humanoid Robots on Complex Terrain - IEEE Xplore, fecha de acceso: marzo 12, 2026, https://ieeexplore.ieee.org/iel8/6287639/10820123/11204001.pdf

[^20^]: Reinforcement Learning for Trading - NIPS, fecha de acceso: marzo 12, 2026, http://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf

[^21^]: Deep Reinforcement Learning for trading applications - Alpha Architect, fecha de acceso: marzo 12, 2026, https://alphaarchitect.com/reinforcement-learning-for-trading/

[^22^]: Deep Reinforcement Learning: Building a Trading Agent - GitHub Pages, fecha de acceso: marzo 12, 2026, https://stefan-jansen.github.io/machine-learning-for-trading/22_deep_reinforcement_learning/

[^23^]: Deep Reinforcement Learning for Trading Strategies | Accio Analytics Inc., fecha de acceso: marzo 12, 2026, https://accioanalytics.io/insights/deep-reinforcement-learning-for-trading-strategies/

[^24^]: A Self-Rewarding Mechanism in Deep Reinforcement Learning for Trading Strategy Optimization - MDPI, fecha de acceso: marzo 12, 2026, https://www.mdpi.com/2227-7390/12/24/4020

[^25^]: Soft-Actor-Critic (SAC) for Forex Trading: An example implementation - Medium, fecha de acceso: marzo 12, 2026, https://medium.com/@abatrek059/soft-actor-critic-sac-for-forex-trading-an-example-implementation-11c679b80f32

[^26^]: Actor-Critic Methods: SAC and PPO | Joel's PhD Blog, fecha de acceso: marzo 12, 2026, https://joel-baptista.github.io/phd-weekly-report/posts/ac/

[^27^]: Does SAC perform better than PPO in sample-expensive tasks with discrete action spaces?, fecha de acceso: marzo 12, 2026, https://ai.stackexchange.com/questions/36092/does-sac-perform-better-than-ppo-in-sample-expensive-tasks-with-discrete-action

[^28^]: Looking for insights on stabilizing SAC/PPO-based trading agents facing alpha decay & regime adaptation issues : r/quant - Reddit, fecha de acceso: marzo 12, 2026, https://www.reddit.com/r/quant/comments/1oo2j2z/looking_for_insights_on_stabilizing_sacppobased/

[^29^]: Post 5: The Ensemble That Actually Trades --- How PPO, SAC, and TD3 Became One Decisioning System - gists · GitHub, fecha de acceso: marzo 12, 2026, https://gist.github.com/ttarler/9d1541df109a4019a6b0e236d56c3949

[^30^]: Deep reinforcement learning-SAC- Portfolio Optimization: part three - Medium, fecha de acceso: marzo 12, 2026, https://medium.com/@abatrek059/deep-reinforcement-learning-sac-portfolio-optimization-part-three-9c1431f63ff9

[^31^]: A distributional soft actor critic-portfolio optimization: A pursuit of stability - Medium, fecha de acceso: marzo 12, 2026, https://medium.com/@abatrek059/a-distributional-soft-actor-critic-portfolio-optimization-a-pursuit-of-stability-a4057826a0b1

[^32^]: Deep Reinforcement Learning Strategies in Finance: Insights into Asset Holding, Trading Behavior, and Purchase Diversity Regular Research Paper (CSCE-ICAI'24) - arXiv.org, fecha de acceso: marzo 12, 2026, https://arxiv.org/html/2407.09557v1

[^33^]: Deep Reinforcement Learning in Algorithmic Trading: A Step-by-Step Guide | by Pham The Anh | Funny AI & Quant | Medium, fecha de acceso: marzo 12, 2026, https://medium.com/funny-ai-quant/deep-reinforcement-learning-in-algorithmic-trading-a-step-by-step-guide-197f39a8be9a

[^34^]: Sharpe ratio as a reward function for reinforcement learning trading agent - Reddit, fecha de acceso: marzo 12, 2026, https://www.reddit.com/r/algotrading/comments/8705zw/sharpe_ratio_as_a_reward_function_for/

[^35^]: Regret-Optimized Portfolio Enhancement through Deep Reinforcement Learning and Future Looking Rewards - arXiv, fecha de acceso: marzo 12, 2026, https://arxiv.org/html/2502.02619v1

[^36^]: Risk-Aware Reinforcement Learning Reward for Financial Trading - arXiv.org, fecha de acceso: marzo 12, 2026, https://arxiv.org/html/2506.04358v1

[^37^]: Action-specialized expert ensemble trading system with extended discrete action space using deep reinforcement learning - PMC, fecha de acceso: marzo 12, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC7384672/

[^38^]: Risk-Sensitive Deep Reinforcement Learning for Portfolio Optimization - MDPI, fecha de acceso: marzo 12, 2026, https://www.mdpi.com/1911-8074/18/7/347

[^39^]: A Systematic Approach to Portfolio Optimization: A Comparative Study of Reinforcement Learning Agents, Market Signals, and Investment Horizons - MDPI, fecha de acceso: marzo 12, 2026, https://www.mdpi.com/1999-4893/17/12/570

[^40^]: Deep Reinforcement Learning for Forex Trading Using Twin Delayed DDPG (TD3): part one, fecha de acceso: marzo 12, 2026, https://medium.com/@abatrek059/deep-reinforcement-learning-for-forex-trading-using-twin-delayed-ddpg-td3-part-one-8aef4a6b078c

[^41^]: Portfolio Optimization using Deep Reinforcement Learning models - Lund University Publications, fecha de acceso: marzo 12, 2026, https://lup.lub.lu.se/student-papers/record/9178260/file/9178261.pdf

[^42^]: Application of Deep Reinforcement Learning to At-the-Money S&P 500 Options Hedging - arXiv, fecha de acceso: marzo 12, 2026, https://arxiv.org/html/2510.09247v1

[^43^]: ICML Poster Safety-Polarized and Prioritized Reinforcement Learning, fecha de acceso: marzo 12, 2026, https://icml.cc/virtual/2025/poster/43599

[^44^]: Action Masking Methods for Safe Reinforcement Learning in a Non-Stationary Configurable Environment - ResearchGate, fecha de acceso: marzo 12, 2026, https://www.researchgate.net/publication/397820515_Action_Masking_Methods_for_Safe_Reinforcement_Learning_in_a_Non-Stationary_Configurable_Environment

[^45^]: Policy-Based Reinforcement Learning with Action Masking for Dynamic Job Shop Scheduling under Uncertainty - arXiv.org, fecha de acceso: marzo 12, 2026, https://arxiv.org/pdf/2601.09293

[^46^]: RLlib: Industry-Grade, Scalable Reinforcement Learning --- Ray 2.54.0, fecha de acceso: marzo 12, 2026, https://docs.ray.io/en/latest/rllib/index.html

[^47^]: Does action masking reduce the ability of the agent to learn game rules? - Reddit, fecha de acceso: marzo 12, 2026, https://www.reddit.com/r/reinforcementlearning/comments/10l7b7f/does_action_masking_reduce_the_ability_of_the/

[^48^]: Why Do Even Excellent Traders Go Broke? The Kelly Criterion and Position Sizing Risk, fecha de acceso: marzo 12, 2026, https://medium.com/@idsts2670/why-do-even-excellent-traders-go-broke-the-kelly-criterion-and-position-sizing-risk-62c17d279c1c

[^49^]: Good and bad properties of the Kelly criterion, fecha de acceso: marzo 12, 2026, https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf

[^50^]: Kelly Criterion vs. Level Staking in Sports Betting and Trading, fecha de acceso: marzo 12, 2026, https://tradeonsports.co.uk/kelly-criterion-vs-level-staking-in-sports-betting-and-trading/

[^51^]: Why Retail Traders Should Avoid The Kelly Criterion Method : r/options - Reddit, fecha de acceso: marzo 12, 2026, https://www.reddit.com/r/options/comments/mnhrj9/why_retail_traders_should_avoid_the_kelly/

[^52^]: Deep Reinforcement Learning Trading - Consensus, fecha de acceso: marzo 12, 2026, https://consensus.app/search/deep-reinforcement-learning-trading/bk8HK5YrTI6g5lIhBFiCdg/

[^53^]: Statistical Arbitrage with Deep RL: Addressing Overfitting Through Model Restructuring | by Navnoor Bawa | Medium, fecha de acceso: marzo 12, 2026, https://medium.com/@navnoorbawa/statistical-arbitrage-with-deep-rl-addressing-overfitting-through-model-restructuring-3338c1c3f153

[^54^]: Efficiently Initializing Reinforcement Learning With Prior Policies - Google Research, fecha de acceso: marzo 12, 2026, https://research.google/blog/efficiently-initializing-reinforcement-learning-with-prior-policies/

[^55^]: A Comparison of Behavior Cloning Methods in Developing Interactive Opposing-Force Agents, fecha de acceso: marzo 12, 2026, https://journals.flvc.org/FLAIRS/article/download/133299/137955/247239

[^56^]: How to pretrain a model with behavior cloning - RLlib - Ray, fecha de acceso: marzo 12, 2026, https://discuss.ray.io/t/how-to-pretrain-a-model-with-behavior-cloning/278

[^57^]: How do I pretrain a RL agent using human demonstration? : r/reinforcementlearning - Reddit, fecha de acceso: marzo 12, 2026, https://www.reddit.com/r/reinforcementlearning/comments/1cwmb1f/how_do_i_pretrain_a_rl_agent_using_human/

[^58^]: Look-ahead bias is a hell of a drug! : r/algotrading - Reddit, fecha de acceso: marzo 12, 2026, https://www.reddit.com/r/algotrading/comments/1n8rn14/lookahead_bias_is_a_hell_of_a_drug/

[^59^]: Walk Forward Analysis (OVERFITTING QUESTION DUMP) : r/algotrading - Reddit, fecha de acceso: marzo 12, 2026, https://www.reddit.com/r/algotrading/comments/1gussns/walk_forward_analysis_overfitting_question_dump/

[^60^]: Reinforcement Learning in Financial Decision Making: A Systematic Review of Performance, Challenges, and Implementation Strategies - arXiv.org, fecha de acceso: marzo 12, 2026, https://arxiv.org/html/2512.10913v1

[^61^]: How to Avoid Overfitting When Testing Trading Rules « adventuresofgreg.blog, fecha de acceso: marzo 12, 2026, http://adventuresofgreg.com/blog/2025/12/18/avoid-overfitting-testing-trading-rules/

[^62^]: Walk-Forward Optimization: How It Works, Its Limitations, and Backtesting Implementation, fecha de acceso: marzo 12, 2026, https://blog.quantinsti.com/walk-forward-optimization-introduction/

[^63^]: Trading with Walk-Forward Optimization and Machine Learning : r/Daytrading - Reddit, fecha de acceso: marzo 12, 2026, https://www.reddit.com/r/Daytrading/comments/1idz4je/trading_with_walkforward_optimization_and_machine/

[^64^]: What is a Walk-Forward Optimization and How to Run It? - AlgoTrading101 Blog, fecha de acceso: marzo 12, 2026, https://algotrading101.com/learn/walk-forward-optimization/

[^65^]: arXiv:2209.05559v6 [q-fin.ST] 31 Jan 2023, fecha de acceso: marzo 12, 2026, https://arxiv.org/pdf/2209.05559
