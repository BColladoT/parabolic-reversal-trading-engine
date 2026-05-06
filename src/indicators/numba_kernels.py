"""
Numba-Optimized Mathematical Kernels
High-performance calculations for VWAP, ATR, and other indicators.
Compiled to LLVM machine code for C-level execution speed.
"""
import numpy as np
from numba import njit, prange, float64, int64
from src.utils.config import CONFIG

# Compiler flags for maximum performance
NUMBA_CACHE = CONFIG.performance.numba_cache
NUMBA_FASTMATH = CONFIG.performance.numba_fastmath
NUMBA_PARALLEL = CONFIG.performance.numba_parallel


@njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, parallel=NUMBA_PARALLEL)
def calculate_vwap_numba(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray
) -> np.ndarray:
    """
    Calculate Volume-Weighted Average Price (VWAP) using Numba.
    
    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3
    
    Parameters:
    -----------
    highs, lows, closes, volumes : np.ndarray
        Price and volume arrays of equal length
        
    Returns:
    --------
    np.ndarray : VWAP values
    """
    n = len(highs)
    vwap = np.empty(n, dtype=float64)
    
    cum_pv = 0.0  # Cumulative price * volume
    cum_vol = 0.0  # Cumulative volume
    
    for i in prange(n):
        typical_price = (highs[i] + lows[i] + closes[i]) / 3.0
        pv = typical_price * volumes[i]
        
        cum_pv += pv
        cum_vol += volumes[i]
        
        if cum_vol > 0:
            vwap[i] = cum_pv / cum_vol
        else:
            vwap[i] = typical_price
            
    return vwap


@njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, parallel=NUMBA_PARALLEL)
def calculate_vwap_incremental_numba(
    previous_vwap: float,
    previous_cum_pv: float,
    previous_cum_vol: float,
    high: float,
    low: float,
    close: float,
    volume: float
) -> tuple:
    """
    Incremental VWAP calculation for real-time streaming.
    Returns (new_vwap, new_cum_pv, new_cum_vol)
    """
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    
    new_cum_pv = previous_cum_pv + pv
    new_cum_vol = previous_cum_vol + volume
    
    if new_cum_vol > 0:
        new_vwap = new_cum_pv / new_cum_vol
    else:
        new_vwap = typical_price
        
    return new_vwap, new_cum_pv, new_cum_vol


@njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def calculate_true_range_numba(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray
) -> np.ndarray:
    """
    Calculate True Range for volatility measurement.
    
    TR = max(
        High - Low,
        |High - Previous Close|,
        |Low - Previous Close|
    )
    """
    n = len(highs)
    tr = np.empty(n, dtype=float64)
    
    # First element has no previous close
    tr[0] = highs[0] - lows[0]
    
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i-1])
        lpc = abs(lows[i] - closes[i-1])
        
        tr[i] = max(hl, max(hpc, lpc))
        
    return tr


@njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def calculate_atr_numba(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range (ATR) using Wilder's smoothing.
    
    Parameters:
    -----------
    highs, lows, closes : np.ndarray
        Price arrays
    period : int
        Lookback period (default 14)
        
    Returns:
    --------
    np.ndarray : ATR values
    """
    n = len(highs)
    tr = calculate_true_range_numba(highs, lows, closes)
    atr = np.empty(n, dtype=float64)
    
    # First ATR is simple average of first 'period' TR values
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        
        # Wilder's smoothing: ATR = ((Prior ATR * 13) + Current TR) / 14
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    else:
        atr[:] = np.mean(tr) if n > 0 else 0.0
        
    # Fill initial values
    for i in range(min(period-1, n)):
        atr[i] = atr[period-1] if n >= period else 0.0
        
    return atr


@njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def calculate_atr_latest_numba(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> float:
    """
    Calculate only the latest ATR value for real-time use.
    More efficient when only current value is needed.
    """
    n = len(highs)
    if n < 2:
        return highs[0] - lows[0] if n > 0 else 0.0
    
    # Calculate TR for all periods
    tr_values = np.empty(min(n, period + 1), dtype=float64)
    
    start_idx = max(0, n - period - 1)
    
    # First TR in window
    if start_idx == 0:
        tr_values[0] = highs[0] - lows[0]
    else:
        hl = highs[start_idx] - lows[start_idx]
        hpc = abs(highs[start_idx] - closes[start_idx - 1])
        lpc = abs(lows[start_idx] - closes[start_idx - 1])
        tr_values[0] = max(hl, max(hpc, lpc))
    
    # Remaining TR values
    for i in range(1, len(tr_values)):
        idx = start_idx + i
        hl = highs[idx] - lows[idx]
        hpc = abs(highs[idx] - closes[idx - 1])
        lpc = abs(lows[idx] - closes[idx - 1])
        tr_values[i] = max(hl, max(hpc, lpc))
    
    return np.mean(tr_values)


@njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def detect_momentum_divergence_numba(
    prices: np.ndarray,
    volumes: np.ndarray,
    lookback: int = 3
) -> bool:
    """
    Detect price-volume momentum divergence.
    Returns True if price makes new high but volume decreases.
    """
    n = len(prices)
    if n < lookback * 2:
        return False
    
    # Compare recent window vs previous window
    recent_prices = prices[-lookback:]
    recent_volumes = volumes[-lookback:]
    
    previous_prices = prices[-lookback*2:-lookback]
    previous_volumes = volumes[-lookback*2:-lookback]
    
    price_increasing = np.mean(recent_prices) > np.mean(previous_prices) * 1.02
    volume_decreasing = np.mean(recent_volumes) < np.mean(previous_volumes) * 0.8
    
    return price_increasing and volume_decreasing


@njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def calculate_position_size_numba(
    account_equity: float,
    risk_percent: float,
    entry_price: float,
    stop_price: float,
    max_position_value: float = 50000.0
) -> int:
    """
    Calculate position size based on volatility (ATR-based stop).
    Works for both long and short positions.
    
    For LONGS: stop_price < entry_price, risk = entry - stop
    For SHORTS: stop_price > entry_price, risk = stop - entry
    
    Returns:
    --------
    int : Number of shares (always positive)
    """
    risk_amount = account_equity * (risk_percent / 100.0)
    risk_per_share = abs(entry_price - stop_price)
    
    if risk_per_share <= 0:
        return 0
    
    shares = int(risk_amount / risk_per_share)
    
    # Cap position value
    max_shares = int(max_position_value / entry_price)
    shares = min(shares, max_shares)
    
    return max(0, shares)


@njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def calculate_ema_numba(
    values: np.ndarray,
    period: int
) -> np.ndarray:
    """
    Calculate Exponential Moving Average.
    """
    n = len(values)
    ema = np.empty(n, dtype=float64)
    
    multiplier = 2.0 / (period + 1)
    
    # Initialize with SMA
    ema[0] = values[0]
    sma_sum = 0.0
    count = 0
    
    for i in range(n):
        if i < period:
            sma_sum += values[i]
            count += 1
            ema[i] = sma_sum / count
        else:
            ema[i] = (values[i] - ema[i-1]) * multiplier + ema[i-1]
            
    return ema


@njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def calculate_volume_profile_numba(
    prices: np.ndarray,
    volumes: np.ndarray,
    num_bins: int = 10
) -> tuple:
    """
    Calculate volume profile (POC - Point of Control).
    Returns (bin_edges, volume_per_bin).
    """
    min_p = np.min(prices)
    max_p = np.max(prices)
    bin_size = (max_p - min_p) / num_bins
    
    volume_bins = np.zeros(num_bins, dtype=float64)
    
    for i in range(len(prices)):
        bin_idx = int((prices[i] - min_p) / bin_size)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        volume_bins[bin_idx] += volumes[i]
    
    bin_edges = np.linspace(min_p, max_p, num_bins + 1)
    
    return bin_edges, volume_bins


@njit(cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def detect_absorption_numba(
    prices: np.ndarray,
    volumes: np.ndarray,
    lookback: int = 20,
    volume_threshold: float = 2.0,
    price_change_threshold: float = 0.001
) -> bool:
    """
    Detect absorption pattern:
    High volume but minimal price movement (institutional iceberg orders).
    
    Parameters:
    -----------
    prices : np.ndarray
        Price levels
    volumes : np.ndarray
        Volume at each level
    lookback : int
        Periods to analyze
    volume_threshold : float
        Multiple of average volume to trigger
    price_change_threshold : float
        Max price change % to consider "stalled"
        
    Returns:
    --------
    bool : True if absorption detected
    """
    n = len(prices)
    if n < lookback:
        return False
    
    recent_volumes = volumes[-lookback:]
    avg_volume = np.mean(volumes[:-lookback]) if n > lookback else np.mean(volumes)
    
    high_volume = np.mean(recent_volumes) > avg_volume * volume_threshold
    
    price_range = (np.max(prices[-lookback:]) - np.min(prices[-lookback:])) / np.mean(prices[-lookback:])
    price_stalled = price_range < price_change_threshold
    
    return high_volume and price_stalled
