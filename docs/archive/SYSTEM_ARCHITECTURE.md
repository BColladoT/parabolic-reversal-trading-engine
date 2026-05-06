# Parabolic Reversal Trading Engine - Complete System Documentation

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [Market Thesis & Edge](#2-market-thesis--edge)
3. [Data Architecture](#3-data-architecture)
4. [Asset Screening & Universe Selection](#4-asset-screening--universe-selection)
5. [Signal Generation Engine](#5-signal-generation-engine)
6. [Trade Entry Logic](#6-trade-entry-logic)
7. [Position Building (Scale-In)](#7-position-building-scale-in)
8. [Exit Logic (Layered Profit Taking)](#8-exit-logic-layered-profit-taking)
9. [Risk Management](#9-risk-management)
10. [Technical Indicators (Numba JIT)](#10-technical-indicators-numba-jit)
11. [Live Trading Engine](#11-live-trading-engine)
12. [Historical Backtesting](#12-historical-backtesting)
13. [Reinforcement Learning System](#13-reinforcement-learning-system)
14. [TCN-AE Perception Module](#14-tcn-ae-perception-module)
15. [SAC Agent Architecture](#15-sac-agent-architecture)
16. [Walk-Forward Optimization (WFO)](#16-walk-forward-optimization-wfo)
17. [Behavioral Cloning (BC) Pre-Training](#17-behavioral-cloning-bc-pre-training)
18. [Data Provider (Hybrid)](#18-data-provider-hybrid)
19. [Configuration Reference](#19-configuration-reference)
20. [Performance Metrics](#20-performance-metrics)
21. [Complete Data Flow Diagrams](#21-complete-data-flow-diagrams)
22. [Formula Reference](#22-formula-reference)

---

## 1. Strategy Overview

### What We Trade

We trade **parabolic reversals** in micro-cap equities. These are stocks that have experienced explosive intraday price moves -- typically 60% to 500% gains from the open -- and are showing signs of exhaustion. We **short-sell** these stocks at the point of exhaustion, betting that the parabolic move will reverse and mean-revert back toward VWAP (Volume-Weighted Average Price).

### Why It Works (The Edge)

Parabolic moves in micro-cap stocks are driven by retail FOMO (Fear Of Missing Out), short squeezes, and momentum algorithms. These moves are inherently unsustainable because:

1. **Volume exhaustion**: The pool of new buyers eventually dries up. When volume drops significantly from its peak while price remains elevated, it signals that buying pressure is fading.
2. **VWAP gravity**: Stocks that extend far above their VWAP (20%+ deviation) have statistical tendency to revert. VWAP represents the "fair value" where most volume transacted.
3. **Overextension**: When price is 20-50%+ above VWAP and within 5% of the day's high on declining volume, the risk/reward for a short entry becomes favorable.

### Strategy Identity

- **Style**: Short-only, intraday mean-reversion
- **Timeframe**: Strictly intraday (all positions flat by 3:25 PM ET)
- **Universe**: Micro-cap equities ($2-$50 price range, <100M float)
- **Entry criteria**: 60-500% intraday gain, >20% VWAP extension, volume exhaustion
- **Edge**: Fading retail euphoria at the point of volume exhaustion
- **Risk per trade**: 1% of account equity
- **Max concurrent positions**: 3
- **Broker**: Alpaca Markets (paper or live)

---

## 2. Market Thesis & Edge

### The Parabolic Move Lifecycle

A typical parabolic micro-cap move follows this lifecycle:

```
Phase 1: CATALYST (09:30-09:45 ET)
   News/PR/earnings drops pre-market or at open
   Stock gaps up 20-40% from previous close
   Volume surges 5-10x above average

Phase 2: MOMENTUM (09:45-11:00 ET)
   Retail traders pile in via social media alerts
   Short sellers begin covering (adding fuel)
   Volume peaks in first 30-60 minutes
   Price extends 60-200%+ from open

Phase 3: EXHAUSTION (11:00-14:30 ET)     <-- WE ENTER HERE
   New buyer flow diminishes
   Volume drops 40-60% from peak
   Price stalls near highs but stops making new highs
   Momentum divergence: price flat/up, volume dropping
   Absorption patterns: large volume, no price movement

Phase 4: REVERSAL (variable timing)       <-- WE PROFIT HERE
   First wave of profit-taking begins
   Stop-loss cascades from late buyers
   Price reverts toward VWAP (mean reversion)
   Our TP1 (35%) hits at VWAP
   Further sell-off to TP2 (-8%) and TP3 (-15%)
```

### Why Volume Exhaustion Matters

Volume exhaustion is the single most important signal. When a stock has run 100%+ on 5 million shares in the first hour, but is now only trading 2 million shares per hour while holding near highs, it means:

- **Supply of new buyers is drying up** (everyone who wanted in is already in)
- **Remaining holders are "trapped"** (bought near highs, won't sell for a loss yet)
- **Any selling pressure cascades** (once one large seller exits, stop-losses trigger)

We measure this as a **volume ratio**: `current_5min_volume / peak_5min_volume`. When this ratio drops below 0.60 (60% of peak), we consider volume "exhausted."

### Entry Confirmation Factors

We require at least **2 of 5 confirming factors** to enter:

| Factor | What It Measures | Threshold |
|--------|------------------|-----------|
| VWAP Extension | How far price has stretched from fair value | >20% above VWAP |
| Volume Exhaustion | Buying pressure fading | <60% of session peak volume |
| Price Near High | Stock hasn't started falling yet (ideal short entry) | Within 5% of day's high |
| Momentum Divergence | Price up but volume down (bearish divergence) | Price +2%, Volume -20% |
| Volume Absorption | Large orders being absorbed without price impact | 2x avg volume, <0.5% move |

---

## 3. Data Architecture

### Data Sources

| Source | Format | Location | Purpose |
|--------|--------|----------|---------|
| Alpaca Markets API | REST + WebSocket | Live connection | Real-time ticks, order execution |
| Historical 1-min bars | Parquet | `data/cache/1min_extended/` | Backtesting, RL training |
| Setup cache | Pickle | `data/cache/setups/` | Pre-computed qualified setups |
| Backtest results | CSV | `reports/` | Proven winners for hybrid training |

### Historical Data Coverage

- **Date range**: 2019-01-01 through 2024-12-31 (6 years)
- **Symbols**: 3,090+ symbol-years of 1-minute OHLCV bars
- **Total size**: ~2.2 GB in Parquet format
- **Naming**: `{SYMBOL}_1min_20190101_20241231.parquet`
- **Columns**: timestamp, open, high, low, close, volume

### Data Processing Stack

- **Polars**: All DataFrame operations use Polars (not Pandas) for 30-50x speedup
- **LazyFrames**: Complex queries use Polars LazyFrame for query optimization
- **Numba**: All numerical indicator calculations are JIT-compiled to machine code
- **Parquet**: Columnar storage format for fast I/O and compression

### Real-Time Data Flow

```
Alpaca WebSocket (SIP feed)
  -> TickData dataclass (symbol, price, size, timestamp)
  -> StreamingBuffer (ring buffer, max 10,000 ticks per symbol)
  -> 1-minute bar aggregation (BarData: OHLCV + VWAP)
  -> Bar history (rolling 200-bar window)
  -> Signal evaluation
```

### VWAP Calculation (Session-Anchored)

VWAP resets every day at 9:30 AM ET. All calculations use the session VWAP anchored from market open:

```
Typical Price = (High + Low + Close) / 3
Cumulative PV = sum(Typical Price * Volume) from 9:30 AM
Cumulative Vol = sum(Volume) from 9:30 AM
VWAP = Cumulative PV / Cumulative Vol
```

For real-time streaming, VWAP is updated incrementally on each tick:

```python
new_cum_pv = prev_cum_pv + (typical_price * volume)
new_cum_vol = prev_cum_vol + volume
new_vwap = new_cum_pv / new_cum_vol
```

VWAP deviation (the key entry metric) is calculated as:

```
VWAP Deviation (%) = ((Close - VWAP) / VWAP) * 100
```

A deviation of 20% means the stock is trading 20% above its volume-weighted fair value.

---

## 4. Asset Screening & Universe Selection

### Screening Criteria

To qualify as a potential short target, a stock must meet ALL of the following:

| Criterion | Value | Rationale |
|-----------|-------|-----------|
| Intraday gain from open | 60-500% | Parabolic move must be significant |
| Current price | $2.00 - $50.00 | Micro/small cap focus, avoid penny stocks |
| Daily volume | >= 500,000 shares | Must be liquid enough to short |
| Float | < 100 million shares | Low float = more volatile reversals |
| Shortable | Yes (Alpaca flag) | Must be available for short selling |
| Easy to borrow | Yes (Alpaca flag) | Avoid hard-to-borrow fees |

### Quality Score (0-100)

Each qualifying stock receives a quality score used for prioritization:

**Gain Magnitude (0-35 points)**:
- 60-100% gain: 35 pts (optimal range for reversals)
- 100-200% gain: 30 pts (still good, more volatile)
- >200% gain: 20 pts (extreme, may continue running)

**Distance from High of Day (0-25 points)**:
- 2-8% below HOD: 25 pts (exhaustion just beginning)
- <2% below HOD: 20 pts (still pushing, entry risk higher)
- 8-15% below HOD: 15 pts (pullback deeper, may have missed)
- >15% below HOD: 10 pts (too late for optimal entry)

**Volume (0-25 points)**:
- >5M shares: 25 pts | >2M: 20 pts | >1M: 15 pts | <=1M: 10 pts

**Intraday Range (0-15 points)**:
- >80% range: 15 pts | >50%: 10 pts | <=50%: 5 pts

### Extended Universe

The full scan universe consists of 3,527 historically identified micro-cap setups spanning 2019-2024. The scanner (`scan_extended_universe.py`) evaluates all symbols in the extended universe (`src/backtest/extended_universe.py`, 1,100+ symbols) against the screening criteria.

---

## 5. Signal Generation Engine

**File**: `src/execution/signal_engine.py` (ParabolicSignalEngine)

### Signal Types

```python
class SignalType(Enum):
    ENTRY_SHORT = "entry_short"       # Initial short entry
    ADD_POSITION = "add_position"     # Scale-in to existing short
    EXIT_COVER = "exit_cover"         # Take-profit cover
    STOP_LOSS = "stop_loss"           # Stop-loss cover
    TP1_VWAP = "tp1_vwap"            # Cover at VWAP (35%)
    TP2_MOMENTUM = "tp2_momentum"     # Cover at -8% (35%)
    TP3_FINAL = "tp3_final"          # Cover at -15% (30%)
    FLATTEN = "flatten"               # End-of-day forced close
```

### Signal Data Structure

```python
@dataclass
class TradeSignal:
    symbol: str                    # Stock ticker
    signal_type: SignalType        # Type of signal
    timestamp: datetime            # When signal was generated
    price: float                   # Current price at signal time
    confidence: float              # 0.0 to 1.0 confidence score
    vwap: float                    # Current session VWAP
    atr: float                     # Current ATR (14-period)
    volume_ratio: float            # Current volume / peak volume
    volume_exhaustion: bool        # Whether volume is exhausted
    is_add_signal: bool = False    # True if scale-in signal
    add_level: int = 0             # 1 (initial), 2 (first add), 3 (second add)
    notes: str = ""                # Human-readable notes
```

### Volume Profile Tracking

The signal engine maintains a real-time volume profile for each tracked symbol:

```python
@dataclass
class VolumeProfile:
    peak_volume: float = 0.0              # Highest 5-min volume in session
    peak_volume_time: Optional[datetime]  # When peak occurred
    current_volume_5min: float = 0.0      # Current 5-min volume
    volume_ratio: float = 1.0             # Current / Peak
    volume_trend: str = "neutral"         # "increasing", "decreasing", "neutral"
```

The volume peak is established from the first 30 minutes of trading (the period of maximum buying pressure in parabolic moves).

### Confidence Scoring

Each confirming factor adds to the overall confidence score:

| Factor | Condition | Confidence Added |
|--------|-----------|-----------------|
| VWAP Extension 1.20-1.50 | 20-50% above VWAP | +0.30 |
| VWAP Extension 1.50-2.00 | 50-100% above VWAP | +0.25 |
| VWAP Extension >2.00 | >100% above VWAP | +0.15 |
| Volume Exhaustion <0.40 | Severe exhaustion | +0.30 |
| Volume Exhaustion <0.50 | Strong exhaustion | +0.25 |
| Volume Exhaustion <0.60 | Moderate exhaustion | +0.20 |
| Price Near High (>0.95) | Within 5% of HOD | +0.15 |
| Momentum Divergence | Price up, volume down | +0.15 |
| Absorption Pattern | High vol, no price move | +0.10 |

Maximum confidence: 1.0 (capped). Entry requires 2+ confirming factors.

---

## 6. Trade Entry Logic

### Entry Decision Flow

```
Tick arrives for qualified symbol
    |
    v
1. Fetch current metrics from data engine
   (price, VWAP, ATR, volume profile)
    |
    v
2. Calculate VWAP extension ratio
   extension = current_price / vwap
   Requirement: extension >= 1.20 (20% above VWAP)
   FAIL -> No signal
    |
    v
3. Evaluate 5 confirming factors:
   [x] VWAP extension    (always true if we got here)
   [ ] Volume exhaustion  (volume_ratio < 0.60)
   [ ] Price near high    (price >= day_high * 0.95)
   [ ] Momentum divergence (price +2%, volume -20% over 3 bars)
   [ ] Absorption pattern  (2x avg volume, <0.5% price change)
    |
    v
4. Count confirming factors
   Requirement: >= 2 factors
   FAIL -> No signal
    |
    v
5. Sum confidence scores from each factor
    |
    v
6. Create TradeSignal(ENTRY_SHORT)
    |
    v
7. Emit to TradingEngine callback
    |
    v
8. RiskManager calculates position size
   (see Risk Management section)
    |
    v
9. Submit short sell order to Alpaca
   Order type: Limit (0.1% above market)
   Time in force: IOC (Immediate or Cancel)
    |
    v
10. Position opened, tracking begins
```

### Entry Timing

- **Earliest entry**: 9:45 AM ET (15 minutes after open to allow price discovery)
- **Latest entry**: 2:30 PM ET (need enough time for reversal to play out)
- **No entries**: Before 9:45 AM or after 2:30 PM ET

---

## 7. Position Building (Scale-In)

### Scale-In Philosophy

We don't enter the full position at once. Instead, we **scale in progressively** as the exhaustion thesis is confirmed. Each add requires the stock to make a new high on even lower volume -- strengthening our conviction that the move is unsustainable.

### Three-Tier Scale-In

| Add Level | Size % | Dollar Amount | Volume Threshold | Requirement |
|-----------|--------|---------------|-----------------|-------------|
| Add 1 (Initial) | 25% | $7,500 | <60% of peak | 2+ exhaustion factors |
| Add 2 (First scale-in) | 25% | $7,500 | <50% of peak | New HOD + lower volume |
| Add 3 (Final scale-in) | 50% | $15,000 | <40% of peak | New HOD + even lower volume |
| **Total** | **100%** | **$30,000** | | |

### Add Signal Criteria

For Add 2 and Add 3, ALL of the following must be true:

1. **New High of Day**: Current price > previous HOD since entry. The stock must make a new high -- this confirms that the parabolic move is continuing but we see increasingly exhausted volume.

2. **Lower Volume**: Volume ratio must be below the threshold for the add level:
   - Add 2: volume_ratio < 0.50 (volume dropped to 50% of peak)
   - Add 3: volume_ratio < 0.40 (volume dropped to 40% of peak)

3. **Cooldown Period**: Minimum 10 minutes since the last add. This prevents rapid-fire entries in volatile conditions.

4. **Position Not Full**: Must not have already reached 3 adds.

### Weighted Average Entry Price

When adding to a position, the average entry price is recalculated:

```
new_avg_entry = (old_shares * old_avg_entry + new_shares * new_entry_price) / total_shares

Example:
  Add 1: 200 shares @ $10.00 -> avg = $10.00
  Add 2: 200 shares @ $11.00 -> avg = (200*10 + 200*11) / 400 = $10.50
  Add 3: 400 shares @ $12.00 -> avg = (400*10.50 + 400*12) / 800 = $11.25
```

---

## 8. Exit Logic (Layered Profit Taking)

### Exit Philosophy

We exit in layers rather than all at once. This captures profit if the stock mean-reverts partially (TP1 at VWAP) while keeping exposure for deeper reversals (TP2 at -8%, TP3 at -15%).

### Exit Hierarchy (Priority Order)

**Stop Loss always takes absolute priority.** If a stop is triggered, the entire position is closed immediately regardless of any take-profit levels.

```
CHECK ORDER (every tick):
  1. Stop Loss -> Close 100% immediately
  2. TP1 (VWAP) -> Close 35% of position
  3. TP2 (-8%) -> Close 35% of position
  4. TP3 (-15%) -> Close remaining 30%
  5. Flatten (3:25 PM ET) -> Close whatever remains
```

### TP1: VWAP Mean Reversion (35% of position)

- **Trigger**: Price touches or crosses below VWAP
- **Closes**: 35% of total shares
- **Rationale**: VWAP is the "fair value" for the day. Many parabolic stocks revert to VWAP as the first move. Capturing 35% here locks in a guaranteed profit on the most common outcome.
- **Confidence**: 0.90

### TP2: Momentum Continuation (-8% from entry, 35% of position)

- **Trigger**: Price drops 8% from weighted average entry price
- **Formula**: `(avg_entry - current_price) / avg_entry * 100 >= 8.0`
- **Closes**: 35% of total shares
- **Rationale**: If the stock has already reverted past VWAP and continues selling off, the next natural target is 8% below our entry. This captures the momentum continuation.
- **Confidence**: 0.85

### TP3: Final Exit (-15% from entry, remaining 30%)

- **Trigger**: Price drops 15% from weighted average entry price
- **Formula**: `(avg_entry - current_price) / avg_entry * 100 >= 15.0`
- **Closes**: Remaining 30% of position
- **Rationale**: A 15% reversal from our entry is an excellent outcome. This is the final target where we want full exposure removed.
- **Confidence**: 0.80

### Trailing Stop (After TP1)

After TP1 is hit, a trailing stop can be activated:
- **Activation**: After 3% profit from entry
- **Behavior**: Follows price down, locks in gains
- **Purpose**: Prevents giving back profits on the remaining 65% position

### End-of-Day Flatten

- **Time**: 3:25 PM ET (35 minutes before market close)
- **Action**: ALL remaining open positions are force-closed at market price
- **No exceptions**: The system is strictly intraday -- no overnight positions ever

### Stop Loss Placement

| Scenario | Stop Distance | Formula |
|----------|---------------|---------|
| Initial entry (Add 1) | 4.0% above entry | `entry_price * 1.04` |
| Full position (Add 2+) | 3.5% above average entry | `avg_entry * 1.035` |
| Hard stop | 1% above day's high | `day_high * 1.01` |
| **Actual stop** | **Maximum of calculated and hard stop** | `max(calculated, hard_stop)` |

The stop loss is placed ABOVE the entry price because we're short. If the stock continues higher through our stop, we accept the loss and exit.

---

## 9. Risk Management

**File**: `src/risk/position_manager.py` (RiskManager)

### Core Risk Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Max risk per position | 1% of account equity | Limits single-trade loss |
| Daily loss limit | 2% of account equity | Stops trading after bad day |
| Max concurrent positions | 3 | Limits total exposure |
| Max daily trades | 9 | 3 positions x 3 adds max |
| Max position value | $30,000 | Hard dollar cap per position |
| Max shares per position | 5,000 | Hard share cap |
| Min account equity | $2,000 | Safety floor |

### Position Sizing Algorithm

The position size is determined by the intersection of risk-based sizing and value-based sizing:

```
Step 1: Calculate risk budget
  max_risk = account_equity * 0.01  (1% of account)

  Risk allocation by add level:
    Add 1: 50% of max_risk = 0.5% of account
    Add 2: 30% of max_risk = 0.3% of account
    Add 3: 20% of max_risk = 0.2% of account

Step 2: Calculate stop distance
  If Add 1: stop_distance = entry_price * 0.04
  If Add 2+: stop_distance = entry_price * 0.035
  hard_stop = day_high * 1.01
  stop_loss = max(entry_price + stop_distance, hard_stop)
  risk_per_share = stop_loss - entry_price

Step 3: Risk-based shares
  max_shares_by_risk = risk_allocation / risk_per_share

Step 4: Value-based shares
  position_value_limit = $30,000 * size_percent_for_add_level
    Add 1: $30,000 * 25% = $7,500
    Add 2: $30,000 * 25% = $7,500
    Add 3: $30,000 * 50% = $15,000
  max_shares_by_value = position_value_limit / entry_price

Step 5: Final share count
  shares = min(max_shares_by_risk, max_shares_by_value, 5000)
```

### Pre-Trade Validation Checks

Before any trade is submitted, the following checks must ALL pass:

```
1. Account equity >= $2,000                    (minimum equity)
2. daily_pnl > -(account_equity * 0.02)        (daily loss limit not hit)
3. trades_today < 9                             (daily trade limit)
4. For new positions: open_positions < 3        (max concurrent)
5. risk_per_share > 0                           (valid stop placement)
6. shares > 0                                   (sizing produced valid result)
7. position_value * 1.04 <= buying_power        (margin check)
```

### Daily Loss Limit Enforcement

```
On each trade close:
  daily_pnl += realized_pnl

  if daily_pnl <= -(account_equity * 0.02):
    daily_loss_limit_hit = True
    ALL new entries blocked for rest of day
    Log CRITICAL alert
    Emergency flatten may trigger
```

### Position State Machine

```
  PENDING --> BUILDING --> FULL_SIZE --> SCALING_OUT --> CLOSED
     |           |            |              |             |
  (order     (accepting   (all adds      (taking       (fully
   placed)    scale-ins)   complete)     profits)     exited)
```

---

## 10. Technical Indicators (Numba JIT)

**File**: `src/indicators/numba_kernels.py`

All indicator calculations are compiled to machine code using Numba's `@njit(cache=True, fastmath=True)` decorator for maximum performance. Data is converted from Polars/NumPy arrays to raw NumPy arrays before calling Numba functions.

### VWAP (Volume-Weighted Average Price)

**Batch calculation** (for historical data):

```python
@njit(cache=True, fastmath=True)
def calculate_vwap_numba(highs, lows, closes, volumes):
    n = len(highs)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_vol = 0.0

    for i in range(n):
        typical_price = (highs[i] + lows[i] + closes[i]) / 3.0
        pv = typical_price * volumes[i]
        cum_pv += pv
        cum_vol += volumes[i]
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else typical_price

    return vwap
```

**Incremental calculation** (for real-time streaming):

```python
def calculate_vwap_incremental_numba(prev_vwap, prev_cum_pv, prev_cum_vol,
                                      high, low, close, volume):
    typical_price = (high + low + close) / 3.0
    new_cum_pv = prev_cum_pv + (typical_price * volume)
    new_cum_vol = prev_cum_vol + volume
    new_vwap = new_cum_pv / new_cum_vol if new_cum_vol > 0 else typical_price
    return new_vwap, new_cum_pv, new_cum_vol
```

### ATR (Average True Range)

**True Range**:
```
TR = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
```

**ATR (Wilder's Smoothing)**:
```
Initial ATR = Simple Average of first 14 True Range values
ATR[i] = ((ATR[i-1] * 13) + TR[i]) / 14
```

### Momentum Divergence Detection

Detects bearish divergence: price making new highs while volume is declining.

```python
def detect_momentum_divergence(prices, volumes, lookback=3):
    recent_prices = prices[-lookback:]
    recent_volumes = volumes[-lookback:]
    previous_prices = prices[-lookback*2:-lookback]
    previous_volumes = volumes[-lookback*2:-lookback]

    price_increasing = mean(recent_prices) > mean(previous_prices) * 1.02  # +2%
    volume_decreasing = mean(recent_volumes) < mean(previous_volumes) * 0.80  # -20%

    return price_increasing and volume_decreasing
```

### Absorption Pattern Detection

Detects institutional iceberg orders: unusually high volume with minimal price movement.

```python
def detect_absorption(prices, volumes, lookback=50,
                      volume_threshold=2.0, price_change_threshold=0.005):
    recent_volumes = volumes[-lookback:]
    avg_volume = mean(volumes[:-lookback])

    high_volume = mean(recent_volumes) > avg_volume * volume_threshold  # 2x avg

    price_range = (max(prices[-lookback:]) - min(prices[-lookback:])) / mean(prices[-lookback:])
    price_stalled = price_range < price_change_threshold  # <0.5% range

    return high_volume and price_stalled
```

---

## 11. Live Trading Engine

**File**: `src/main_engine.py` (TradingEngine)

### Architecture

The live engine is fully event-driven, using callbacks from the broker WebSocket:

```
TradingEngine
  |-- AlpacaClient (broker connection)
  |     |-- WebSocket: Real-time tick stream
  |     |-- REST API: Order submission, position queries
  |     '-- Callbacks: tick_callback, trade_callback
  |
  |-- PolarsSignalEngine (data processing)
  |     |-- StreamingBuffer (per-symbol tick ring buffer)
  |     |-- Bar aggregation (tick -> 1-min bars)
  |     '-- Incremental VWAP calculation
  |
  |-- ParabolicScreener (asset screening)
  |     |-- Real-time screening of top movers
  |     '-- Quality scoring
  |
  |-- ParabolicSignalEngine (signal generation)
  |     |-- Volume profile tracking
  |     |-- Entry/exit signal evaluation
  |     '-- Signal callbacks
  |
  '-- RiskManager (position management)
        |-- Position sizing
        |-- Stop/TP tracking
        '-- Daily P&L enforcement
```

### Event Flow

**On each tick**:

```python
def _on_tick(self, tick: TickData):
    # 1. Process tick through data engine
    self.data_engine.process_tick(tick)
    #    -> Updates StreamingBuffer
    #    -> Calculates incremental VWAP
    #    -> Aggregates into 1-min bars

    # 2. Update signal engine
    self.signal_engine.update_tick(tick.symbol, tick.price, tick.size, tick.timestamp)
    #    -> Updates volume profile
    #    -> Checks entry/add conditions

    # 3. Update risk manager
    self.risk_manager.update_positions(tick.symbol, tick.price)
    #    -> Updates unrealized P&L
    #    -> Tracks highest price since entry

    # 4. Check exit conditions
    self._check_exits(tick.symbol, tick.price)
    #    -> Generates exit signals if stop/TP hit
```

### Order Execution

**Entry orders**:
- Order type: Limit
- Price: 0.1% above current market (for short fills)
- Time in force: IOC (Immediate or Cancel)
- Fallback: Market order if limit fails

**Exit orders**:
- Order type: Limit
- Price: 0.1% below current market (for cover fills)
- Time in force: IOC
- Fallback: Market order for stops and forced flattens

### Market Hours Management

```
09:30 ET  Market opens, VWAP resets
09:45 ET  Monitoring begins, entries allowed
14:30 ET  Entry window closes, no new positions
15:25 ET  Flatten time, all positions force-closed
16:00 ET  Market closes, system idles
```

---

## 12. Historical Backtesting

**File**: `src/backtest/tick_backtest_engine.py`

### Backtest Architecture

The backtester simulates the Progressive Exhaustion Scale-In strategy using historical 1-minute bar data. It replays market data tick-by-tick and applies the same signal, risk, and execution logic as the live engine.

### Fill Simulation

```python
# Entry fill (short sell)
fill_price = price * (1 + entry_slippage_bps / 10000)  # 5 bps above market

# Exit fill (cover buy)
fill_price = price * (1 + exit_slippage_bps / 10000)   # 5 bps above market
```

**Default slippage**: 5 bps (0.05%) per side = 10 bps round-trip.

**Note**: The RL environment uses 30 bps (0.30%) per side to be more conservative about micro-cap execution quality. The backtest uses 5 bps because it assumes limit order fills.

### Backtest Execution

```
For each setup in universe:
  1. Load 1-min bars for (symbol, date)
  2. Calculate session VWAP from 9:30 AM
  3. Establish volume peak (first 30 minutes)
  4. Walk through each bar:
     a. Update volume profile
     b. Check entry criteria (2+ exhaustion factors)
     c. If in position: check exit criteria
     d. Record fills with slippage
  5. Force flatten at 3:25 PM
  6. Record P&L and metrics
```

### Volume Peak Establishment

```python
for bar in first_30_minutes:
    volume_peak = max(volume_peak, bar.volume)

# After 30 minutes, peak is locked
# All subsequent volume is measured against this peak
```

### Running Backtests

```bash
# Quick test (10 setups)
python run_historical_backtest.py --quick-test

# Single setup
python run_historical_backtest.py --symbol AMC --date 2021-06-02

# Full universe scan
python scan_extended_universe.py
```

---

## 13. Reinforcement Learning System

### Why RL?

The rule-based strategy uses fixed thresholds (20% VWAP extension, 60% volume exhaustion, etc.). RL can potentially learn:

1. **Adaptive entry timing**: Not just "is volume exhausted?" but "is THIS the optimal moment to enter given the full price/volume pattern?"
2. **Dynamic position sizing**: How much to short based on the specific setup quality, not just fixed 25/25/50 splits.
3. **Flexible exits**: When to cover based on the evolving market microstructure, not just fixed VWAP/8%/15% targets.

### RL System Architecture

```
Historical Parquet Data (1-min OHLCV bars)
  |
  v
HybridDataProvider (70% CSV winners + 30% all high-vol days)
  |-- Loads episode: (symbol, date)
  |-- Calculates VWAP from market open
  |-- Filters to entry window
  '-- Returns bar-by-bar data
  |
  v
ParabolicReversalEnv (Gymnasium-compatible)
  |-- Observation: 74-dim state vector
  |     |-- TCN-AE latent encoding (64-dim) of 60-bar OHLCV
  |     '-- Explicit features (10-dim): VWAP dev, volume, position, PnL, etc.
  |-- Action: Continuous [-1, 1] = target short exposure
  |-- Reward: Normalized equity delta + drawdown penalty
  |-- Constraints: Action masking, circuit breaker, time windows
  |
  v
SAC Agent (Soft Actor-Critic)
  |-- Actor: Gaussian policy -> continuous action
  |-- Critic: Twin Q-networks for value estimation
  |-- Temperature: Auto-tuned entropy coefficient
  '-- Target networks: Soft-updated copies
```

### Environment (ParabolicReversalEnv)

**File**: `src/rl/env.py`

#### Observation Space (74 dimensions)

```
Dimensions [0:64]  - TCN-AE latent encoding
  Input: 60 consecutive OHLCV bars (pre-decision window)
  Encoding: Causal temporal convolutions -> 64-dim bottleneck
  Property: Strictly causal (no future information leakage)

Dimension [64]     - VWAP deviation (%)
  Normalized: (vwap_dev - 30.0) / 15.0
  Range: Typically 0-80% for parabolic stocks

Dimension [65]     - Volume concentration ratio
  Range: [0, 1] where 1 = all volume at one price level

Dimension [66]     - Current position size
  Normalized: position_value / 50,000
  Negative = short position

Dimension [67]     - Unrealized P&L (%)
  Normalized: unrealized_pnl_pct (already in %)

Dimension [68]     - Current drawdown
  Normalized: drawdown / -19,180 (reference max)

Dimension [69]     - Kelly fraction
  Normalized: kelly / 3.0 (max leverage)

Dimension [70]     - Hour of day (ET)
  Normalized: hour / 24

Dimension [71]     - Minute
  Normalized: minute / 60

Dimension [72]     - Day of week
  Normalized: day / 7

Dimension [73]     - Is entry window
  Binary: 1.0 if 09:45-14:30 ET, 0.0 otherwise
```

#### Action Space

Single continuous value in [-1.0, +1.0]:

```
Action < -0.1:  INCREASE short exposure (enter or add to short)
Action > +0.1:  DECREASE short exposure (cover/close short)
|Action| <= 0.1: HOLD current position (no-op, no transaction costs)
```

The action is converted to a target dollar position:

```
kelly_leverage = 1.0 (training) or Kelly-calculated (eval)
target_exposure = action * kelly_leverage
target_position_value = target_exposure * current_capital

# Clipped to [−$30,000, $0] (short-only, max $30K position)
```

#### Action Masking

The environment enforces hard constraints via an action mask:

| Mask Index | Action | Blocked When |
|------------|--------|-------------|
| mask[0] | Increase short | Outside entry window (before 09:45 or after 14:30 ET) |
| | | VWAP deviation < 20% |
| | | Position value >= $30,000 (max position) |
| | | Circuit breaker triggered |
| | | Must flatten (after 15:25 ET) |
| mask[1] | Decrease short (cover) | Not in a position (flat) |
| mask[2] | Hold | Circuit breaker triggered AND in position (must cover) |

When the agent selects an action that violates the mask, the action is overridden to HOLD (0.0) and a masking penalty of -0.5 is added to the reward.

#### Reward Function

**Base reward**: Normalized incremental equity change.

```
equity_delta = current_capital - previous_capital
base_reward = (equity_delta / initial_capital) * 1000.0

Scaling: $100 profit -> +1.0 reward
         $500 profit -> +5.0 reward
         $100 loss   -> -1.0 reward
```

**Drawdown penalty** (quadratic, activates at $5K drawdown):

```
if |drawdown| > $5,000:
    excess_dd = |drawdown| - 5,000
    max_excess = $10,000 - $5,000 = $5,000
    penalty = -((excess_dd / max_excess)^2) * 50.0

    Range: [-50.0, 0.0]

    Examples:
      $6,000 drawdown: penalty = -((1000/5000)^2) * 50 = -2.0
      $7,500 drawdown: penalty = -((2500/5000)^2) * 50 = -12.5
      $10,000 drawdown: penalty = -(1.0)^2 * 50 = -50.0 (circuit breaker triggers)
```

**Masking penalty** (additive, when agent violates action mask):

```
if action_violates_mask:
    reward = base_reward + (-0.5)
```

**Accounting invariant**: Sum of base rewards over an episode equals total return (excluding shaping penalties):

```
sum(base_rewards) = (final_equity - initial_equity) / initial_equity * 1000
```

#### Circuit Breaker

```
if current_drawdown <= -$10,000:
    circuit_breaker_triggered = True
    -> Force close any open position
    -> Episode terminates immediately
    -> Prevents catastrophic losses
```

#### Episode Lifecycle

```
reset():
  1. Initialize capital to $100,000
  2. Clear position state (flat)
  3. Load trading day data (random or fixed)
  4. Load 60-bar pre-decision price history
  5. Return initial observation + info

step(action):
  1. Clip action to [-1, 1]
  2. Check action mask
  3. Compute target position value
  4. Execute position change (open/add/cover/close)
  5. Save current bar to price history
  6. Advance to next bar (time moves forward)
  7. Update portfolio metrics with new price
  8. Check circuit breaker
  9. Calculate reward
  10. Check termination (circuit breaker or end of day)
  11. Return observation, reward, terminated, truncated, info
```

#### Position Execution (`_execute_position_change`)

**Open Short** (from flat):
```
entry_price = current_price
shares = target_value / current_price  (negative for short)
cash += |shares| * current_price - slippage_cost
slippage_cost = |delta_value| * 0.003  (30 bps)
```

**Add to Short** (increase existing):
```
delta_shares = (target_value - prev_value) / current_price
new_entry_price = weighted average of old and new entries:
  = (old_shares * old_entry + delta_shares * current_price) / new_total_shares
cash += |delta_shares| * current_price - slippage_cost
```

**Cover Short** (decrease existing):
```
shares_to_cover = |delta_shares|
realized_pnl = shares_covered * (entry_price - cover_price)
cash -= shares_covered * cover_price + slippage_cost
```

---

## 14. TCN-AE Perception Module

**File**: `src/rl/perception.py`

### Purpose

The TCN-AE (Temporal Convolutional Network Autoencoder) compresses a 60-bar OHLCV price sequence into a 64-dimensional latent vector. This captures the "shape" of the price/volume pattern (parabolic curve, volume exhaustion signature) in a compact form that SAC can learn from.

### Why TCN (Not LSTM/Transformer)?

1. **Causal**: TCN uses causal convolutions -- each output only depends on current and past inputs, never future. This is critical for preventing look-ahead bias.
2. **Parallelizable**: Unlike LSTMs, TCN processes the entire sequence in parallel.
3. **Dilated receptive field**: Exponentially increasing dilation (1, 2, 4, 8) gives the network a large receptive field without deep stacking.

### Encoder Architecture

```
Input: [batch, 5, 60]  (5 OHLCV features x 60 bars)
  |
  v
Layer 0: CausalConv1d(5 -> 32, kernel=3, dilation=1)
  -> ReLU -> Dropout(0.2) -> ResidualBlock(32, dilation=1)
  |
Layer 1: CausalConv1d(32 -> 64, kernel=3, dilation=2)
  -> ReLU -> Dropout(0.2) -> ResidualBlock(64, dilation=2)
  |
Layer 2: CausalConv1d(64 -> 128, kernel=3, dilation=4)
  -> ReLU -> Dropout(0.2) -> ResidualBlock(128, dilation=4)
  |
Layer 3: CausalConv1d(128 -> 64, kernel=3, dilation=8)
  -> ReLU -> Dropout(0.2) -> ResidualBlock(64, dilation=8)
  |
  v
Global Average Pooling: [batch, 64, 60] -> [batch, 64]
  |
  v
Linear + LayerNorm: [batch, 64] -> [batch, 64]
  |
  v
Output: z_t (64-dim latent vector)
```

### Causal Convolution

```
Standard convolution: output[t] depends on input[t-k..t+k]  (future leakage!)
Causal convolution:   output[t] depends on input[t-k..t]     (past only)

Implementation: Pad LEFT side with (kernel_size - 1) * dilation zeros
                Then trim RIGHT side to maintain input length
```

### Residual Block

```
Input ---+---> CausalConv1d -> ReLU -> Dropout(0.2) -> CausalConv1d -> ReLU -> Dropout(0.2) ---> Add -> Output
         |                                                                                        ^
         +----------------------------------------------------------------------------------------+
         (skip connection: preserves gradient flow)
```

### Decoder Architecture (for self-supervised pre-training)

The decoder reconstructs the original OHLCV sequence from the latent vector. It's only used during pre-training and is discarded during RL training.

```
Input: z_t [batch, 64]
  |
  v
Linear: 64 -> 64 * 15 = 960
Reshape: [batch, 960] -> [batch, 64, 15]
  |
ConvTranspose layers (reversed encoder):
  64 -> 128 (stride=2) -> 64 (stride=2) -> 32 (stride=1) -> 5 (stride=1)
  |
  v
Output: [batch, 5, 60] (reconstructed OHLCV)
```

### State Representation (74-dim)

The `StateRepresentation` module combines the frozen encoder output with explicit features:

```
TCN-AE Encoder (frozen) -> z_t [64-dim]
  +
Explicit Features [10-dim]:
  [0] VWAP deviation: (vwap_dev - 30.0) / 15.0
  [1] Volume concentration: raw [0, 1]
  [2] Position size: position_value / 50000.0
  [3] Unrealized PnL: raw (already %)
  [4] Drawdown: drawdown / -19180.0
  [5] Kelly fraction: kelly / 3.0
  [6] Hour: hour / 24
  [7] Minute: minute / 60
  [8] Day of week: day / 7
  [9] Is entry window: 0.0 or 1.0
  =
Total: 74-dim state vector
```

---

## 15. SAC Agent Architecture

**File**: `src/rl/agent.py`

### Soft Actor-Critic (SAC) Overview

SAC is an off-policy, maximum entropy reinforcement learning algorithm. It simultaneously learns:
- **A policy (Actor)**: Maps states to actions, maximizing expected reward + entropy
- **Q-functions (Critics)**: Estimate the value of state-action pairs
- **Temperature (alpha)**: Balances reward maximization vs. exploration

### Why SAC?

1. **Continuous actions**: Our action space is a continuous value [-1, 1] (target exposure). SAC handles continuous actions natively.
2. **Sample efficient**: Off-policy learning with replay buffer means we can reuse past experience.
3. **Stable**: Maximum entropy formulation prevents premature convergence.
4. **Exploration**: Entropy bonus encourages diverse action sampling during training.

### Network Architectures

**Actor (MaskedGaussianPolicy)**:
```
State [74] -> Linear(256) -> ReLU -> Linear(256) -> ReLU
  |-> Mean head -> Linear(1)      -> Mean action
  |-> Log-std head -> Linear(1)   -> Log standard deviation (clamped to [-20, 2])

Sampling: action = tanh(mean + std * noise)
  where noise ~ Normal(0, 1)
```

**Critic (Twin Q-Networks)**:
```
[State, Action] [75] -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> Linear(1) -> Q-value

Q1 and Q2 are independent networks (twin Q for overestimation mitigation)
```

### Training Algorithm

**Critic update** (minimize TD error):
```
target_Q = reward + gamma * (1 - done) * (min(Q1_target, Q2_target) - alpha * log_prob)
loss_Q1 = MSE(Q1(s, a), target_Q)
loss_Q2 = MSE(Q2(s, a), target_Q)
```

**Actor update** (maximize expected Q + entropy):
```
a_new ~ policy(s)
loss_actor = mean(alpha * log_prob(a_new) - min(Q1(s, a_new), Q2(s, a_new)))
```

**Temperature update** (auto-tune entropy coefficient):
```
loss_alpha = -alpha * (log_prob(a_new) + target_entropy)
```

**Target network update** (soft Polyak averaging):
```
theta_target = tau * theta + (1 - tau) * theta_target
  where tau = 0.005
```

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Actor hidden layers | [256, 256] | Network capacity |
| Critic hidden layers | [256, 256] | Network capacity |
| Learning rate (actor) | 3e-4 | Adam optimizer |
| Learning rate (critic) | 3e-4 | Adam optimizer |
| Learning rate (alpha) | 3e-4 | Adam optimizer |
| Discount factor (gamma) | 0.99 | Future reward weighting |
| Soft update rate (tau) | 0.005 | Target network update speed |
| Initial temperature (alpha) | 0.2 | Entropy coefficient |
| Target entropy | -1.0 | Entropy target (-action_dim) |
| Replay buffer size | 1,000,000 | Experience storage |
| Batch size | 256 | Training batch size |

---

## 16. Walk-Forward Optimization (WFO)

**File**: `src/scripts/train_wfo.py`

### What is WFO?

Walk-Forward Optimization trains the RL agent on rolling time windows to prevent overfitting to any single period. Each "fold" trains on a historical window and tests on the immediately following out-of-sample period.

### Fold Structure

```
|<--- Train Window (1 year) --->|<- Purge ->|<--- Test Window (3 months) --->|
|          TRAINING DATA         | 10 days   |        EVALUATION DATA         |

Fold 1: Train 2020-07 to 2021-07 | Purge 10d | Test 2021-08 to 2021-11
Fold 2: Train 2021-01 to 2022-01 | Purge 10d | Test 2022-02 to 2022-05
Fold 3: Train 2021-07 to 2022-07 | Purge 10d | Test 2022-08 to 2022-11
...
```

**Purge buffer**: 10 days of data between train and test are excluded to prevent any information leakage from training into evaluation (e.g., multi-day patterns or serial correlation).

### Two-Phase Training (Per Fold)

**Phase 1: Critic Warm-Up (30,000 timesteps)**
- Actor learning rate: 0.0 (FROZEN -- actor weights don't change)
- Critic learning rate: 3e-4 (active)
- Purpose: Let the critic learn the Q-function landscape before the actor starts adapting. This prevents the actor from chasing noisy Q-estimates early in training.

**Phase 2: SAC Fine-Tuning (70,000 timesteps)**
- Actor learning rate: 3e-4 (UNFROZEN)
- Critic learning rate: 3e-4 (still active)
- Purpose: Actor and critic co-train. The critic has already developed a reasonable value landscape, so the actor's gradient updates are more meaningful.

**Total**: 100,000 timesteps per fold (~416 episodes at ~240 steps/episode).

### Actor Freezing/Unfreezing

The `WarmupCallback` (custom RLlib callback) controls the phase transition:

```python
class WarmupCallback(DefaultCallbacks):
    def on_train_result(self, algorithm, result, **kwargs):
        timesteps_total = result['timesteps_total']

        if self.phase == 1 and timesteps_total >= warmup_timesteps:
            # Phase 1 -> Phase 2: Unfreeze actor
            policy = algorithm.get_policy()
            for param_group in actor_optimizer.param_groups:
                param_group['lr'] = finetune_lr_actor  # 3e-4
            self.phase = 2
```

### Exhaustive Evaluation

After training each fold, the agent is evaluated on ALL available test setups:

```python
for setup in test_setups:  # Every (symbol, date) in test window
    obs, info = eval_env.reset(options={"fixed_setup": setup})

    if info.get("episode_load_failed"):
        skip  # Data not available for this setup

    while not done:
        action = policy.compute_single_action(obs)
        obs, reward, done, _, info = eval_env.step(action)

    record episode P&L, trades, etc.
```

### WFO Data Leakage Prevention

Three layers of protection:

1. **Date-range filtering**: Each fold's data provider is created with strict `date_range=(start, end)`. Only episodes within this range are available.

2. **Runtime assertions**: If the data provider samples an episode outside its configured date range, a `RuntimeError` is raised immediately (hard crash, not a warning).

3. **Isolated instances**: WFO-constrained data providers are never cached in the global singleton. Each fold creates a fresh provider instance.

```python
# Training provider (Jan-Jun 2023)
train_provider = get_data_provider(
    date_range=("2023-01-01", "2023-06-30"),
    mode="train", seed=42
)
# -> Creates new instance, NOT cached

# Test provider (Jul-Sep 2023)
eval_provider = get_data_provider(
    date_range=("2023-07-10", "2023-09-30"),  # After purge
    mode="eval", seed=42
)
# -> Creates DIFFERENT new instance, completely isolated
```

---

## 17. Behavioral Cloning (BC) Pre-Training

**File**: `src/scripts/behavioral_cloning.py`

### Purpose

BC pre-trains the SAC actor network using "expert demonstrations" derived from the V5 Relaxed rule-based strategy. This gives the actor a reasonable starting policy (when to enter, when to hold) so that SAC fine-tuning starts from a good initialization rather than random behavior.

### How BC Works

1. **Generate anchored samples** from historical data:
   - **Positive samples** (entry): Bars where the V5 Relaxed strategy would have entered a short (VWAP deviation > 20%, in entry window). The anchor point is the bar with maximum VWAP extension in the entry window.
   - **Negative samples** (hold/flat): Bars where the strategy would NOT have entered (VWAP too low, outside window, etc.).

2. **Train a supervised model** to predict the action given the state:
   - Input: 74-dim state vector (same as RL environment)
   - Output: Continuous action (target: -1.0 for entry, 0.0 for hold)
   - Loss: MSE between predicted and target action

3. **Export trained weights** for SAC actor initialization:
   - Save as `.pt` file compatible with RLlib's policy loading

### Anchored Sample Construction

```python
@dataclass
class AnchoredSample:
    """Weakly supervised anchoring, not exact expert cloning."""
    # Anchor to bar with MAXIMUM VWAP extension (heuristic)
    # This is where the V5 Relaxed strategy would most likely enter

For each (symbol, date) in backtest CSV:
    1. Load 1-min bars
    2. Calculate VWAP and VWAP deviation
    3. Find bar with max VWAP deviation in entry window (09:45-14:30)
    4. If max_vwap_dev > 20%:
       -> Create positive sample at this bar
       -> Target action: -1.0 (full short)
    5. Sample random bars where vwap_dev < 20%:
       -> Create negative samples
       -> Target action: 0.0 (hold/flat)
```

### Sampling Ratio

- **Ratio**: 1:1 positive to negative (balanced)
- **Purpose**: Prevents the actor from learning a hold-biased policy. A 2:1 ratio (previous setting) taught the actor "67% of the time, don't trade" which made SAC fine-tuning struggle to overcome the inertia.

### BC Training Loop

```python
for epoch in range(100):
    for batch in dataloader:
        states, target_actions = batch
        predicted_actions = actor(states)
        loss = MSE(predicted_actions, target_actions)
        loss.backward()
        optimizer.step()

    # Early stopping with patience=15
    if val_loss hasn't improved in 15 epochs:
        break
```

---

## 18. Data Provider (Hybrid)

**File**: `src/rl/data_provider_hybrid.py`

### Design Philosophy

The data provider mixes two sources:

1. **CSV setups (70% weight)**: Proven winners from the V5 Relaxed backtest. These are (symbol, date) pairs where the rule-based strategy made money. Training on these teaches the RL agent what "good" setups look like.

2. **Parquet setups (30% weight)**: ALL high-volatility trading days from the historical data, including losing days and non-event days. Training on these teaches the agent when NOT to trade and how to handle adverse conditions.

### CSV Setup Loading

```python
Source: reports/relaxed_909_backtest.csv
Columns: symbol, date, pnl, ...

Filter: pnl > $100  (only meaningful trades)
Validation: Parquet data file must exist for the symbol
VWAP check: Max VWAP deviation must exceed (20% - 3% buffer = 17%)
```

### Parquet Setup Scanning

```python
Source: data/cache/1min_extended/*.parquet

For each symbol file:
    For each trading date in the data:
        1. Filter to market hours (09:00-16:00 ET)
        2. Calculate VWAP from 9:30 AM
        3. Find max VWAP deviation for the day
        4. Add to setup list (regardless of threshold)
        5. Log: "volatile" if max_dev >= 20%, "boring" otherwise
```

The key insight is that ALL days are included, not just volatile ones. This means during training, the agent will encounter many days where VWAP deviation never reaches 20%, and it must learn to hold flat (action mask blocks entries anyway).

### Episode Loading

When the environment requests a new episode:

```python
1. Random source selection:
   - 70% chance: sample from CSV setups
   - 30% chance: sample from Parquet setups

2. Load trading day data:
   a. Read Parquet file for symbol
   b. Filter to specific date + market hours
   c. Recalculate VWAP from market open
   d. Calculate VWAP deviation for each bar
   e. Find first bar with |vwap_dev| > 17% (entry threshold - 3% buffer)
   f. Set start_bar_idx to that bar

3. Episode runs from start_bar_idx to end of day
   - Agent makes decisions at each bar
   - Environment advances one bar per step
```

### Pre-Decision Sequence (Causal Data Access)

The observation includes a 60-bar OHLCV history. This sequence is STRICTLY PRE-DECISION:

```
Window: [current_bar_idx - 60, current_bar_idx)
                                 ^
                                 |
                          NOT INCLUDED (this is the bar being decided on)

The last bar in the sequence is current_bar_idx - 1.
The current bar is NEVER in the observation.
```

If fewer than 60 bars are available (near the start of the episode), the prefix is zero-padded. Real bars are never repeated.

---

## 19. Configuration Reference

### settings.yaml (Complete)

```yaml
# Broker
broker:
  name: "alpaca"
  paper_trading: true

# Trading Schedule (ET)
timezone:
  scan_start: "09:45"
  entry_window_start: "09:45"
  entry_window_end: "14:30"
  flatten_time: "15:25"
  market_open: "09:30"
  market_close: "16:00"

# Asset Screening
screening:
  min_percent_gain: 60.0          # Minimum intraday gain (%)
  max_percent_gain: 500.0         # Maximum (avoid outliers)
  min_price: 2.0                  # Stock price floor ($)
  max_price: 50.0                 # Stock price ceiling ($)
  min_volume: 500000              # Minimum volume (shares)
  max_float_millions: 100         # Maximum float (millions)

# Volume Exhaustion Detection
volume_exhaustion:
  peak_lookback_minutes: 390      # Full session for peak detection
  entry_threshold: 0.60           # Volume < 60% of peak = exhausted
  add2_threshold: 0.50            # Add 2 requires < 50%
  add3_threshold: 0.40            # Add 3 requires < 40%
  price_proximity_to_high: 0.95   # Must be within 5% of HOD
  new_high_required_for_add: true # Adds require new HOD

# Signal Detection
signals:
  vwap_extension_threshold: 1.20  # 20% above VWAP
  min_exhaustion_factors: 2       # Minimum confirming factors
  absorption_lookback_ticks: 50
  momentum_divergence_periods: 3
  min_minutes_between_adds: 10    # Add cooldown

# Position Building
scaling:
  initial_size_percent: 25        # Add 1: 25% of max
  add2_size_percent: 25           # Add 2: 25% of max
  add3_size_percent: 50           # Add 3: 50% of max
  max_shares_per_position: 5000   # Hard share limit
  max_position_value: 30000       # Hard dollar limit

# Risk Management
risk:
  max_portfolio_risk_percent: 1.0 # 1% risk per position
  initial_stop_percent: 4.0       # 4% stop on Add 1
  average_stop_percent: 3.5       # 3.5% stop on Add 2+
  daily_loss_limit_percent: 2.0   # Stop after -2% daily
  max_positions: 3                # Max concurrent positions
  max_daily_trades: 9             # Max trades per day

# Exit Targets
exits:
  tp1_percent: 35                 # Close 35% at VWAP
  tp2_percent: 35                 # Close 35% at -8%
  tp2_percent_drop: 8.0
  tp3_percent: 30                 # Close 30% at -15%
  tp3_percent_drop: 15.0
  use_trailing_after_tp1: true    # Trailing stop after TP1
  trailing_activation_percent: 3.0

# Execution
execution:
  order_type: "limit"
  limit_offset_ticks: 2
  time_in_force: "ioc"
```

### RL Configuration (config.py)

```python
RL_CONFIG = {
    'min_vwap_deviation_entry': 20.0,   # VWAP entry threshold (%)
    'max_single_trade_loss': -10000.0,  # Single trade loss limit ($)
    'max_drawdown': -10000.0,           # Max drawdown ($)
    'circuit_breaker_threshold': -10000.0, # Circuit breaker ($)
    'kelly_fraction': 0.25,             # Quarter-Kelly
    'max_leverage_cap': 3.0,            # Max leverage
    'min_leverage_floor': 0.5,          # Min leverage
    'max_shares_per_position': 5000,    # Max shares
    'max_position_value': 30000.0,      # Max position ($)
}
```

### RL Environment Parameters

```python
EnvironmentConfig:
    transaction_cost_per_dollar: 0.003  # 30 bps per trade
    masking_penalty: -0.5               # Penalty for invalid actions
    max_acceptable_drawdown: -5000.0    # Drawdown penalty onset ($)
    circuit_breaker_drawdown: -10000.0  # Circuit breaker ($)
    entry_window: "09:45" - "14:30"     # Entry hours (ET)
    flatten_time: "15:25"               # Force close time (ET)
    initial_capital: 100000.0           # Starting capital ($)
```

### WFO Training Parameters

```python
WFOConfig:
    train_years: 2                      # Training window length
    test_months: 6                      # Test window length
    purge_days: 10                      # Data purge between windows
    warmup_timesteps: 30000             # Phase 1 (critic only)
    finetune_timesteps: 70000           # Phase 2 (actor + critic)
    buffer_size: 1000000                # Replay buffer
    batch_size: 256                     # Training batch
    tau: 0.005                          # Target net update rate
    gamma: 0.99                         # Discount factor
    alpha: 0.2                          # Initial entropy temp
```

---

## 20. Performance Metrics

**File**: `src/utils/metrics.py`

### Metrics Computed

| Metric | Formula | Purpose |
|--------|---------|---------|
| Total P&L | Sum of all trade P&Ls | Absolute profitability |
| Win Rate | Winning trades / Total trades | Consistency |
| Profit Factor | Gross profit / Gross loss | Risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline | Worst-case risk |
| Sortino Ratio | (Return - Rf) / Downside Deviation | Risk-adjusted return (downside only) |
| Sharpe Ratio | (Return - Rf) / Std Dev | Risk-adjusted return (all volatility) |
| Avg Win | Mean P&L of winning trades | Expected gain |
| Avg Loss | Mean P&L of losing trades | Expected loss |
| Largest Win | Max single trade P&L | Best case |
| Largest Loss | Min single trade P&L | Worst case |

### Sortino Ratio Calculation

```
daily_returns = [list of daily returns]
downside_returns = [r for r in daily_returns if r < 0]
downside_deviation = std(downside_returns)

sortino = (mean(daily_returns) - risk_free_rate) / downside_deviation
```

The Sortino ratio is preferred over Sharpe for this strategy because we expect asymmetric returns (small frequent losses from stops, larger gains from successful reversals). Sortino only penalizes downside volatility.

---

## 21. Complete Data Flow Diagrams

### Live Trading Flow

```
                    Alpaca WebSocket (SIP)
                           |
                    TickData (symbol, price, size, timestamp)
                           |
              +------------+------------+
              |                         |
    PolarsSignalEngine          ParabolicSignalEngine
              |                         |
    StreamingBuffer              VolumeProfile tracking
    (ring buffer, 10K ticks)     Exhaustion factor evaluation
              |                         |
    Bar Aggregation (1-min)      Entry/Exit signal generation
    Incremental VWAP                    |
              |                  TradeSignal emission
    Signal data metrics                 |
    (VWAP ext, ATR, vol trend)   +------+------+
              |                  |             |
              +---> Combined --->|  RiskManager |
                                 |             |
                          Position sizing      |
                          Stop/TP tracking     |
                          Daily P&L check      |
                                 |             |
                          Order submission     |
                          (Alpaca REST API)    |
                                 |             |
                          Fill confirmation    |
                          Position update      |
```

### RL Training Flow

```
HybridDataProvider
  |-- CSV setups (70%): Proven winners
  |-- Parquet setups (30%): All high-vol days
  |-- Episode: (symbol, date) -> 1-min OHLCV bars
  |
  v
ParabolicReversalEnv
  |-- Load episode bars
  |-- Initialize portfolio ($100K)
  |-- Build 60-bar pre-decision window
  |
  v  (observation: 74-dim)
  |
  v
TCN-AE Encoder (frozen)
  |-- Input: 60-bar OHLCV [5, 60]
  |-- Output: Latent z [64]
  |-- + Explicit features [10]
  |-- = State [74]
  |
  v
SAC Actor (MaskedGaussianPolicy)
  |-- Input: State [74]
  |-- Output: Action [-1, 1]
  |-- Action mask applied
  |
  v
Environment Step
  |-- Execute position change
  |-- Advance to next bar
  |-- Calculate reward
  |-- Check termination
  |
  v
SAC Training
  |-- Store (s, a, r, s', done) in replay buffer
  |-- Sample batch from buffer
  |-- Update critic (TD error)
  |-- Update actor (policy gradient)
  |-- Update temperature (entropy tuning)
  |-- Soft-update target networks
```

### WFO Training Flow

```
Historical Data (2019-2024)
  |
  v
Generate WFO Folds
  |
  +-- Fold 1: Train 2020-07 to 2021-07 | Purge | Test 2021-08 to 2021-11
  |     |
  |     +-- Phase 1: Critic warm-up (30K steps, actor frozen)
  |     +-- Phase 2: SAC fine-tuning (70K steps, actor active)
  |     +-- Exhaustive evaluation on ALL test setups
  |     +-- Save checkpoint + results
  |
  +-- Fold 2: Train 2021-01 to 2022-01 | Purge | Test 2022-02 to 2022-05
  |     |
  |     ... (same process)
  |
  +-- Fold N: ...
  |
  v
Aggregate Results
  |-- Per-fold metrics (P&L, win rate, Sortino, etc.)
  |-- Cross-fold comparison
  |-- Save to wfo_results.json
```

---

## 22. Formula Reference

### Core Trading Formulas

| Formula | Expression | Example |
|---------|-----------|---------|
| VWAP | `sum(TypicalPrice * Volume) / sum(Volume)` | TP=$10.50, Vol=100K -> VWAP contribution |
| Typical Price | `(High + Low + Close) / 3` | (10.80 + 10.20 + 10.50) / 3 = 10.50 |
| VWAP Extension | `Price / VWAP` | $12.00 / $10.00 = 1.20 (20% extension) |
| VWAP Deviation (%) | `(Price - VWAP) / VWAP * 100` | (12 - 10) / 10 * 100 = 20% |
| Volume Ratio | `Current_5min_Vol / Peak_5min_Vol` | 2M / 5M = 0.40 (60% exhaustion) |
| True Range | `max(H-L, |H-PC|, |L-PC|)` | max(0.60, 0.30, 0.30) = 0.60 |
| ATR (Wilder) | `(ATR_prev * 13 + TR) / 14` | (0.50 * 13 + 0.60) / 14 = 0.507 |
| Depreciation | `(AvgEntry - Current) / AvgEntry * 100` | (12 - 11.04) / 12 * 100 = 8.0% |

### Position Sizing Formulas

| Formula | Expression | Example |
|---------|-----------|---------|
| Risk Budget | `AccountEquity * 0.01` | $100K * 1% = $1,000 |
| Risk Per Share | `StopLoss - EntryPrice` | $10.40 - $10.00 = $0.40 |
| Max Shares (risk) | `RiskAllocation / RiskPerShare` | $500 / $0.40 = 1,250 shares |
| Max Shares (value) | `MaxPositionValue / EntryPrice` | $7,500 / $10 = 750 shares |
| Final Shares | `min(risk_shares, value_shares, 5000)` | min(1250, 750, 5000) = 750 |
| Weighted Avg Entry | `(Old * OldShares + New * NewShares) / TotalShares` | (10 * 200 + 11 * 200) / 400 = 10.50 |

### RL Formulas

| Formula | Expression | Example |
|---------|-----------|---------|
| Base Reward | `(EquityDelta / InitialCapital) * 1000` | $100 / $100K * 1000 = +1.0 |
| Drawdown Penalty | `-((ExcessDD / MaxExcess)^2) * 50` | -((2500/5000)^2) * 50 = -12.5 |
| Kelly Full | `(WinRate * (b+1) - 1) / b` | (0.6 * 2.5 - 1) / 1.5 = 0.333 |
| Quarter Kelly | `KellyFull * 0.25` | 0.333 * 0.25 = 0.083 |
| Target Position | `Action * KellyLeverage * Capital` | -0.5 * 1.0 * $100K = -$50K -> capped at -$30K |
| Slippage Cost | `|DeltaValue| * 0.003` | $30K * 0.003 = $90 |
| SAC Target Q | `r + gamma * (min(Q1t, Q2t) - alpha * logprob)` | Standard Bellman backup |
| Soft Update | `theta_t = tau * theta + (1-tau) * theta_t` | tau=0.005 |

---

## Appendix: File Map

| Concern | File | Lines |
|---------|------|-------|
| **Live Trading** | | |
| Orchestrator | `src/main_engine.py` | ~500 |
| Broker API | `src/data/alpaca_client.py` | ~400 |
| Data processing | `src/data/polars_engine.py` | ~300 |
| JIT indicators | `src/indicators/numba_kernels.py` | ~200 |
| Signal generation | `src/execution/signal_engine.py` | ~450 |
| Risk management | `src/risk/position_manager.py` | ~350 |
| Asset screening | `src/screening/screener.py` | ~200 |
| Strategy params | `src/strategies/strategy_registry.py` | ~100 |
| Configuration | `config/settings.yaml` | 164 |
| Config loader | `src/utils/config.py` | ~200 |
| **Backtesting** | | |
| Tick backtest | `src/backtest/tick_backtest_engine.py` | ~500 |
| Backtest runner | `run_historical_backtest.py` | ~200 |
| Universe scanner | `scan_extended_universe.py` | ~150 |
| **RL System** | | |
| Environment | `src/rl/env.py` | ~1,150 |
| TCN-AE perception | `src/rl/perception.py` | ~1,050 |
| SAC agent | `src/rl/agent.py` | ~1,300 |
| Data provider | `src/rl/data_provider_hybrid.py` | ~860 |
| RL config | `src/rl/config.py` | 15 |
| **Training Scripts** | | |
| WFO training | `src/scripts/train_wfo.py` | ~870 |
| WFO quick test | `src/scripts/train_wfo_quick_test.py` | ~370 |
| Behavioral cloning | `src/scripts/behavioral_cloning.py` | ~400 |
| RL vs rule compare | `src/scripts/compare_rl_vs_rule.py` | ~300 |
| **Baselines** | | |
| Rule baseline | `src/baselines/rule_baseline.py` | 220 |
| **Metrics** | | |
| Shared metrics | `src/utils/metrics.py` | ~150 |
| **WFO Context** | `wfo_context/` | Mirror of above |
