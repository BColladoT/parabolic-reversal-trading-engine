"""
Advanced Feature Engineering for Market Microstructure Analysis

Extracts sophisticated features from tick-level data including:
- Order flow toxicity (VPIN)
- Volatility regimes
- Liquidity metrics
- Price impact measures
- Market efficiency ratios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.fft import fft
import warnings


@dataclass
class TickFeatures:
    """Container for tick-level features."""
    # Price action features
    returns: np.ndarray
    log_returns: np.ndarray
    realized_volatility: float
    intraday_volatility: float
    
    # Volume features
    volume_profile: np.ndarray
    volume_imbalance: float
    dollar_volume: float
    
    # Liquidity features
    bid_ask_spread: float
    effective_spread: float
    price_impact: float
    
    # Microstructure features
    autocorrelation: float
    variance_ratio: float
    efficiency_ratio: float


class MarketMicrostructureFeatures:
    """
    Extract features from market microstructure data.
    Based on academic research in high-frequency trading.
    """
    
    def __init__(self, window_sizes: List[int] = [5, 10, 20, 50]):
        self.window_sizes = window_sizes
    
    def extract_all_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract comprehensive feature set from bar data."""
        features = {}
        
        # Basic price features
        features.update(self._price_features(bars))
        
        # Volatility features
        features.update(self._volatility_features(bars))
        
        # Volume features
        features.update(self._volume_features(bars))
        
        # Liquidity features
        features.update(self._liquidity_features(bars))
        
        # Market microstructure
        features.update(self._microstructure_features(bars))
        
        # Trend and momentum
        features.update(self._trend_features(bars))
        
        # Statistical features
        features.update(self._statistical_features(bars))
        
        # Frequency domain features (FFT)
        features.update(self._frequency_features(bars))
        
        return features
    
    def _price_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract price-based features."""
        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values
        open_ = bars['open'].values
        
        features = {}
        
        # Returns
        returns = np.diff(close) / close[:-1]
        log_returns = np.diff(np.log(close))
        
        features['price_range'] = (high.max() - low.min()) / close[0]
        features['body_mean'] = np.mean(np.abs(close - open_)) / np.mean(close)
        features['upper_shadow_mean'] = np.mean(high - np.maximum(close, open_)) / np.mean(close)
        features['lower_shadow_mean'] = np.mean(np.minimum(close, open_) - low) / np.mean(close)
        
        # OHLC relationships
        features['close_to_high'] = close[-1] / high.max()
        features['close_to_low'] = close[-1] / low.min()
        features['high_low_ratio'] = high.max() / low.min()
        
        # Price momentum at different horizons
        for window in self.window_sizes:
            if len(close) >= window:
                features[f'return_{window}bar'] = (close[-1] - close[-window]) / close[-window]
                features[f'momentum_{window}bar'] = self._calculate_momentum(close, window)
        
        return features
    
    def _volatility_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract volatility features."""
        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values
        
        features = {}
        
        # Realized volatility (annualized)
        returns = np.diff(np.log(close))
        if len(returns) > 1:
            features['realized_vol'] = np.std(returns) * np.sqrt(252 * 390)  # 1-min bars
            features['realized_var'] = np.var(returns)
        else:
            features['realized_vol'] = 0
            features['realized_var'] = 0
        
        # Parkinson volatility (uses high-low range)
        if len(high) > 0 and len(low) > 0:
            hl_log = np.log(high / low)
            features['parkinson_vol'] = np.sqrt(np.mean(hl_log**2) / (4 * np.log(2)))
        
        # Garman-Klass volatility (uses OHLC)
        log_hl = np.log(high / low)
        log_co = np.log(close / bars['open'].values)
        features['garman_klass_vol'] = np.sqrt(0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2).mean()
        
        # Volatility of volatility
        if len(returns) >= 20:
            rolling_vol = pd.Series(returns).rolling(10).std().dropna()
            features['vol_of_vol'] = rolling_vol.std()
        
        # Volatility regime
        features['vol_trend'] = self._calculate_volatility_trend(returns)
        
        return features
    
    def _volume_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract volume-based features."""
        volume = bars['volume'].values
        close = bars['close'].values
        
        features = {}
        
        # Basic volume stats
        features['total_volume'] = volume.sum()
        features['avg_volume'] = volume.mean()
        features['volume_std'] = volume.std()
        features['volume_cv'] = volume.std() / volume.mean() if volume.mean() > 0 else 0
        
        # Dollar volume
        features['dollar_volume'] = (volume * close).sum()
        
        # Volume profile (concentration in first hour)
        if 'timestamp' in bars.columns:
            bars_copy = bars.copy()
            bars_copy['timestamp'] = pd.to_datetime(bars_copy['timestamp'])
            first_hour_mask = bars_copy['timestamp'].dt.hour < 11
            first_hour_vol = volume[first_hour_mask].sum() if first_hour_mask.any() else 0
            features['volume_first_hour_pct'] = first_hour_vol / volume.sum() if volume.sum() > 0 else 0
        
        # Volume trend
        if len(volume) >= 10:
            features['volume_trend'] = np.polyfit(range(len(volume)), volume, 1)[0] / volume.mean()
        
        # Volume-weighted average price deviation
        if 'vwap' in bars.columns:
            vwap = bars['vwap'].values
            features['vwap_deviation'] = (close[-1] - vwap[-1]) / vwap[-1]
            features['max_vwap_deviation'] = np.max(np.abs(close - vwap) / vwap)
        
        # On-Balance Volume (OBV)
        obv = np.cumsum(np.where(close[1:] > close[:-1], volume[1:], 
                                np.where(close[1:] < close[:-1], -volume[1:], 0)))
        features['obv_trend'] = np.polyfit(range(len(obv)), obv, 1)[0] / np.abs(obv).mean() if len(obv) > 0 else 0
        
        return features
    
    def _liquidity_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract liquidity features."""
        features = {}
        
        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values
        volume = bars['volume'].values
        
        # Amihud illiquidity ratio
        returns = np.diff(close) / close[:-1]
        dollar_volume = volume[1:] * close[1:]
        if len(returns) > 0 and dollar_volume.sum() > 0:
            features['amihud_ratio'] = np.mean(np.abs(returns) / dollar_volume) * 1e6
        
        # Price impact (Kyle's lambda proxy)
        price_changes = np.diff(close)
        signed_volume = np.sign(price_changes) * volume[1:]
        if len(price_changes) > 1 and signed_volume.std() > 0:
            features['price_impact'] = np.corrcoef(np.abs(price_changes), signed_volume)[0,1]
        
        # Roll's spread estimator (proxy when bid-ask not available)
        if len(close) >= 2:
            price_diffs = np.diff(close)
            cov_diffs = np.cov(price_diffs[:-1], price_diffs[1:])[0,1] if len(price_diffs) > 1 else 0
            features['roll_spread'] = 2 * np.sqrt(-cov_diffs) if cov_diffs < 0 else 0
        
        # Effective spread (high-low based)
        features['effective_spread'] = np.mean((high - low) / close)
        
        return features
    
    def _microstructure_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract market microstructure features."""
        features = {}
        
        close = bars['close'].values
        returns = np.diff(close) / close[:-1]
        
        if len(returns) < 2:
            return features
        
        # Autocorrelation of returns (mean reversion vs momentum)
        for lag in [1, 2, 5, 10]:
            if len(returns) > lag:
                features[f'autocorr_lag{lag}'] = np.corrcoef(returns[lag:], returns[:-lag])[0,1]
        
        # Variance ratio test (market efficiency)
        features['variance_ratio_5'] = self._variance_ratio(returns, 5)
        features['variance_ratio_10'] = self._variance_ratio(returns, 10)
        
        # Efficiency ratio (Kaufman)
        if len(close) >= 10:
            change = abs(close[-1] - close[0])
            volatility = np.sum(np.abs(np.diff(close)))
            features['efficiency_ratio'] = change / volatility if volatility > 0 else 0
        
        # Hurst exponent (persistence)
        if len(returns) >= 100:
            features['hurst_exponent'] = self._hurst_exponent(returns)
        
        return features
    
    def _trend_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract trend and momentum features."""
        features = {}
        
        close = bars['close'].values
        
        # Moving averages and slopes
        for window in [10, 20, 50]:
            if len(close) >= window:
                ma = np.convolve(close, np.ones(window)/window, mode='valid')
                features[f'ma{window}_slope'] = np.polyfit(range(len(ma)), ma, 1)[0] / ma[-1] if ma[-1] != 0 else 0
                features[f'price_above_ma{window}'] = 1 if close[-1] > ma[-1] else 0
        
        # RSI
        features['rsi'] = self._calculate_rsi(close, 14)
        
        # MACD
        features['macd'], features['macd_signal'] = self._calculate_macd(close)
        
        # Time to peak analysis
        peak_idx = np.argmax(close)
        features['time_to_peak_pct'] = peak_idx / len(close)
        features['price_at_peak_ratio'] = close[peak_idx] / close[0]
        
        # Drawdown from peak
        running_max = np.maximum.accumulate(close)
        drawdown = (close - running_max) / running_max
        features['max_drawdown'] = drawdown.min()
        features['current_drawdown'] = drawdown[-1]
        
        return features
    
    def _statistical_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract statistical distribution features."""
        features = {}
        
        close = bars['close'].values
        returns = np.diff(close) / close[:-1]
        
        if len(returns) < 5:
            return features
        
        # Higher moments
        features['skewness'] = stats.skew(returns)
        features['kurtosis'] = stats.kurtosis(returns)
        
        # Jarque-Bera test for normality
        if len(returns) >= 10:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            features['jarque_bera_stat'] = jb_stat
            features['jarque_bera_pvalue'] = jb_pvalue
        
        # Percentiles
        features['return_95th'] = np.percentile(returns, 95)
        features['return_5th'] = np.percentile(returns, 5)
        features['return_percentile_range'] = features['return_95th'] - features['return_5th']
        
        # Outlier detection
        z_scores = np.abs(stats.zscore(returns))
        features['outlier_count'] = np.sum(z_scores > 3)
        features['max_zscore'] = z_scores.max()
        
        return features
    
    def _frequency_features(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Extract frequency domain features using FFT."""
        features = {}
        
        close = bars['close'].values
        
        if len(close) < 32:
            return features
        
        # Detrend
        detrended = close - np.polyval(np.polyfit(range(len(close)), close, 1), range(len(close)))
        
        # FFT
        fft_vals = fft(detrended)
        power = np.abs(fft_vals)**2
        
        # Dominant frequency
        freqs = np.fft.fftfreq(len(detrended))
        positive_freqs = freqs[freqs > 0]
        positive_power = power[freqs > 0]
        
        if len(positive_power) > 0:
            dominant_idx = np.argmax(positive_power)
            features['dominant_frequency'] = positive_freqs[dominant_idx]
            features['dominant_power'] = positive_power[dominant_idx]
            
            # Spectral entropy (randomness measure)
            power_norm = positive_power / positive_power.sum()
            features['spectral_entropy'] = -np.sum(power_norm * np.log(power_norm + 1e-10))
        
        return features
    
    # Helper methods
    def _calculate_momentum(self, prices: np.ndarray, window: int) -> float:
        """Calculate momentum indicator."""
        if len(prices) < window:
            return 0
        return (prices[-1] - prices[-window]) / prices[-window]
    
    def _calculate_volatility_trend(self, returns: np.ndarray) -> float:
        """Calculate volatility trend."""
        if len(returns) < 20:
            return 0
        first_half_vol = np.std(returns[:len(returns)//2])
        second_half_vol = np.std(returns[len(returns)//2:])
        return (second_half_vol - first_half_vol) / first_half_vol if first_half_vol > 0 else 0
    
    def _variance_ratio(self, returns: np.ndarray, q: int) -> float:
        """Calculate variance ratio for market efficiency test."""
        if len(returns) < q * 2:
            return 1.0
        
        var_1 = np.var(returns)
        
        # Q-period returns
        q_returns = np.array([np.sum(returns[i:i+q]) for i in range(0, len(returns)-q+1, q)])
        var_q = np.var(q_returns) / q
        
        return var_q / var_1 if var_1 > 0 else 1.0
    
    def _hurst_exponent(self, returns: np.ndarray, max_lag: int = 100) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        lags = range(2, min(max_lag, len(returns)//4))
        tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
        
        if len(tau) < 2:
            return 0.5
        
        # Linear fit on log-log scale
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)
        
        slope, _, _, _, _ = stats.linregress(log_lags, log_tau)
        return slope
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD and signal line."""
        if len(prices) < slow:
            return 0, 0
        
        ema_fast = pd.Series(prices).ewm(span=fast).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow).mean().values
        
        macd = ema_fast[-1] - ema_slow[-1]
        
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal).mean().values[-1]
        
        return macd / prices[-1], signal_line / prices[-1]


class FeaturePipeline:
    """Pipeline for feature extraction and preprocessing."""
    
    def __init__(self):
        self.extractor = MarketMicrostructureFeatures()
        self.feature_names: List[str] = []
        self.scaler_params: Dict = {}
    
    def fit_transform(self, raw_data: List[Dict]) -> np.ndarray:
        """Fit pipeline and transform data."""
        features_list = []
        
        for data in raw_data:
            bars = pd.DataFrame(data['bars'])
            features = self.extractor.extract_all_features(bars)
            
            # Add metadata features
            features['symbol_encoded'] = hash(data.get('symbol', '')) % 1000
            features['day_of_week'] = pd.to_datetime(data.get('date', '')).dayofweek if 'date' in data else 0
            
            features_list.append(features)
        
        # Create feature matrix
        self.feature_names = list(features_list[0].keys())
        X = np.array([[f.get(name, 0) for name in self.feature_names] for f in features_list])
        
        # Store scaling parameters
        self.scaler_params['mean'] = np.mean(X, axis=0)
        self.scaler_params['std'] = np.std(X, axis=0)
        self.scaler_params['std'][self.scaler_params['std'] == 0] = 1  # Avoid div by zero
        
        # Standardize
        X_scaled = (X - self.scaler_params['mean']) / self.scaler_params['std']
        
        return X_scaled
    
    def transform(self, raw_data: Dict) -> np.ndarray:
        """Transform single data point using fitted parameters."""
        bars = pd.DataFrame(raw_data['bars'])
        features = self.extractor.extract_all_features(bars)
        
        features['symbol_encoded'] = hash(raw_data.get('symbol', '')) % 1000
        features['day_of_week'] = pd.to_datetime(raw_data.get('date', '')).dayofweek if 'date' in raw_data else 0
        
        X = np.array([[features.get(name, 0) for name in self.feature_names]])
        
        if self.scaler_params:
            X = (X - self.scaler_params['mean']) / self.scaler_params['std']
        
        return X
    
    def get_feature_importance(self, model) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
