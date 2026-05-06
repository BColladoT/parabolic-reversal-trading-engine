"""
Online Learning and Adaptive Risk Models

Implements continuous learning capabilities:
- Concept drift detection
- Incremental model updates
- Adaptive threshold adjustment
- Regime change detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
from scipy import stats
import warnings


class AdaptiveRiskModel:
    """
    Adaptive risk model that learns from recent trade outcomes.
    
    Uses sliding window approach with concept drift detection
    to identify when market conditions have changed.
    """
    
    def __init__(self, window_size: int = 100, drift_threshold: float = 2.0):
        """
        Initialize adaptive model.
        
        Args:
            window_size: Number of trades to keep in rolling window
            drift_threshold: Z-score threshold for drift detection
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Rolling windows for metrics
        self.win_history: deque = deque(maxlen=window_size)
        self.pnl_history: deque = deque(maxlen=window_size)
        self.prediction_history: deque = deque(maxlen=window_size)
        self.feature_history: deque = deque(maxlen=window_size)
        
        # Baseline statistics (from training period)
        self.baseline_win_rate: Optional[float] = None
        self.baseline_avg_pnl: Optional[float] = None
        self.baseline_volatility: Optional[float] = None
        
        # Adaptive parameters
        self.current_threshold_adjustment = 0.0
        self.regime = 'normal'  # 'normal', 'volatile', 'trending'
        
        # Performance tracking
        self.correct_predictions = 0
        self.total_predictions = 0
        
    def update(self, trade_result: Dict):
        """
        Update model with new trade outcome.
        
        Args:
            trade_result: Dict with:
                - predicted_win_prob: Model's prediction
                - actual_outcome: 1 for win, 0 for loss
                - actual_pnl: Actual P&L
                - features: Feature vector (optional)
        """
        predicted_prob = trade_result.get('predicted_win_prob', 0.5)
        actual_outcome = trade_result.get('actual_outcome', 0)
        actual_pnl = trade_result.get('actual_pnl', 0)
        features = trade_result.get('features', None)
        
        # Add to history
        self.win_history.append(actual_outcome)
        self.pnl_history.append(actual_pnl)
        self.prediction_history.append(predicted_prob)
        if features is not None:
            self.feature_history.append(features)
        
        # Track prediction accuracy
        predicted_outcome = 1 if predicted_prob > 0.5 else 0
        if predicted_outcome == actual_outcome:
            self.correct_predictions += 1
        self.total_predictions += 1
        
        # Update baseline if first time
        if self.baseline_win_rate is None and len(self.win_history) >= 30:
            self._initialize_baseline()
        
        # Check for regime change
        self._detect_regime_change()
        
        # Adjust thresholds based on recent performance
        self._adaptive_threshold_adjustment()
    
    def _initialize_baseline(self):
        """Initialize baseline statistics from first window."""
        self.baseline_win_rate = np.mean(self.win_history)
        self.baseline_avg_pnl = np.mean(self.pnl_history)
        self.baseline_volatility = np.std(self.pnl_history)
        
        print(f"[ADAPTIVE] Baseline initialized: WR={self.baseline_win_rate:.2%}, "
              f"Avg PnL=${self.baseline_avg_pnl:,.0f}")
    
    def detect_drift(self) -> Tuple[bool, str]:
        """
        Detect concept drift in recent performance.
        
        Returns:
            (drift_detected, drift_type)
        """
        if len(self.win_history) < 50 or self.baseline_win_rate is None:
            return False, 'insufficient_data'
        
        recent_window = list(self.win_history)[-50:]
        recent_pnl = list(self.pnl_history)[-50:]
        
        # Test 1: Win rate drift
        recent_win_rate = np.mean(recent_window)
        win_rate_std = np.sqrt(self.baseline_win_rate * (1 - self.baseline_win_rate) / 50)
        win_rate_zscore = (recent_win_rate - self.baseline_win_rate) / win_rate_std if win_rate_std > 0 else 0
        
        # Test 2: PnL drift
        recent_avg_pnl = np.mean(recent_pnl)
        pnl_std = self.baseline_volatility / np.sqrt(50) if self.baseline_volatility else 1
        pnl_zscore = (recent_avg_pnl - self.baseline_avg_pnl) / pnl_std if pnl_std > 0 else 0
        
        # Test 3: Volatility drift
        recent_vol = np.std(recent_pnl)
        vol_ratio = recent_vol / self.baseline_volatility if self.baseline_volatility else 1
        
        drift_detected = False
        drift_type = 'none'
        
        if abs(win_rate_zscore) > self.drift_threshold:
            drift_detected = True
            drift_type = 'win_rate_drift'
            if win_rate_zscore < 0:
                print(f"[DRIFT WARNING] Win rate degraded: {recent_win_rate:.1%} "
                      f"vs baseline {self.baseline_win_rate:.1%}")
        
        elif abs(pnl_zscore) > self.drift_threshold:
            drift_detected = True
            drift_type = 'pnl_drift'
        
        elif vol_ratio > 1.5:
            drift_detected = True
            drift_type = 'volatility_increase'
            print(f"[DRIFT WARNING] Volatility increased: {recent_vol:,.0f} "
                  f"vs baseline {self.baseline_volatility:,.0f}")
        
        return drift_detected, drift_type
    
    def _detect_regime_change(self):
        """Detect market regime changes."""
        if len(self.pnl_history) < 30:
            return
        
        recent_pnl = list(self.pnl_history)[-30:]
        recent_vol = np.std(recent_pnl)
        recent_sharpe = np.mean(recent_pnl) / recent_vol if recent_vol > 0 else 0
        
        # Regime classification
        if recent_vol > self.baseline_volatility * 1.5:
            new_regime = 'volatile'
        elif recent_sharpe > 2.0:
            new_regime = 'trending'
        else:
            new_regime = 'normal'
        
        if new_regime != self.regime:
            print(f"[REGIME CHANGE] {self.regime} -> {new_regime}")
            self.regime = new_regime
            self._adjust_for_regime()
    
    def _adjust_for_regime(self):
        """Adjust model parameters based on current regime."""
        if self.regime == 'volatile':
            # Increase risk threshold in volatile periods
            self.current_threshold_adjustment = 0.1
        elif self.regime == 'trending':
            # More aggressive in trending markets
            self.current_threshold_adjustment = -0.05
        else:  # normal
            self.current_threshold_adjustment = 0.0
    
    def _adaptive_threshold_adjustment(self):
        """Dynamically adjust entry threshold based on recent performance."""
        if len(self.prediction_history) < 20 or self.total_predictions < 20:
            return
        
        # Calculate calibration (predicted vs actual win rate by bin)
        recent_preds = list(self.prediction_history)[-20:]
        recent_wins = list(self.win_history)[-20:]
        
        # Simple calibration: if we're over-predicting wins, raise threshold
        avg_pred_prob = np.mean(recent_preds)
        actual_win_rate = np.mean(recent_wins)
        
        calibration_error = avg_pred_prob - actual_win_rate
        
        # Adjust threshold slowly
        adjustment = -calibration_error * 0.1  # Small adjustment
        self.current_threshold_adjustment += adjustment
        self.current_threshold_adjustment = np.clip(self.current_threshold_adjustment, -0.2, 0.2)
    
    def get_adjusted_threshold(self, base_threshold: float = 0.5) -> float:
        """Get adjusted entry threshold."""
        return base_threshold + self.current_threshold_adjustment
    
    def get_recent_performance(self) -> Dict:
        """Get performance statistics for recent trades."""
        if len(self.win_history) < 10:
            return {'message': 'Insufficient data'}
        
        recent_wins = list(self.win_history)[-20:]
        recent_pnls = list(self.pnl_history)[-20:]
        
        return {
            'recent_win_rate': np.mean(recent_wins),
            'recent_avg_pnl': np.mean(recent_pnls),
            'recent_volatility': np.std(recent_pnls),
            'recent_sharpe': np.mean(recent_pnls) / np.std(recent_pnls) if np.std(recent_pnls) > 0 else 0,
            'cumulative_recent_pnl': np.sum(recent_pnls),
            'prediction_accuracy': self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0,
            'current_regime': self.regime,
            'threshold_adjustment': self.current_threshold_adjustment
        }
    
    def get_feature_drift(self) -> Dict:
        """Detect drift in feature distributions."""
        if len(self.feature_history) < 100:
            return {'message': 'Insufficient feature data'}
        
        # Split into two windows
        n = len(self.feature_history)
        early_window = list(self.feature_history)[:n//2]
        recent_window = list(self.feature_history)[n//2:]
        
        # Kolmogorov-Smirnov test for each feature
        drift_scores = []
        for feature_idx in range(len(early_window[0])):
            early_vals = [f[feature_idx] for f in early_window]
            recent_vals = [f[feature_idx] for f in recent_window]
            
            ks_stat, p_value = stats.ks_2samp(early_vals, recent_vals)
            drift_scores.append({
                'feature_idx': feature_idx,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < 0.05
            })
        
        # Count significant drifts
        n_drifts = sum(1 for d in drift_scores if d['drift_detected'])
        
        return {
            'n_features_drifted': n_drifts,
            'drift_percentage': n_drifts / len(drift_scores),
            'drift_details': drift_scores,
            'significant_drift': n_drifts > len(drift_scores) * 0.1
        }
    
    def suggest_retraining(self) -> bool:
        """Suggest whether model should be retrained."""
        drift_detected, drift_type = self.detect_drift()
        feature_drift = self.get_feature_drift()
        
        # Retrain if significant drift detected
        if drift_detected and drift_type != 'insufficient_data':
            return True
        
        if isinstance(feature_drift, dict) and feature_drift.get('significant_drift', False):
            return True
        
        # Retrain if accuracy drops significantly
        if self.total_predictions > 50:
            recent_accuracy = self.correct_predictions / self.total_predictions
            if recent_accuracy < 0.55:  # Below random guessing
                return True
        
        return False


class ExponentialWeighting:
    """
    Exponential weighting scheme for recent observations.
    Gives more weight to recent trades.
    """
    
    def __init__(self, decay_factor: float = 0.95):
        """
        Initialize exponential weighting.
        
        Args:
            decay_factor: Decay factor (0.95 means 5% decay per observation)
        """
        self.decay_factor = decay_factor
        self.observations: List[Dict] = []
        self.weights: List[float] = []
    
    def add_observation(self, observation: Dict):
        """Add new observation with decay."""
        # Decay existing weights
        self.weights = [w * self.decay_factor for w in self.weights]
        
        # Add new observation with weight 1
        self.observations.append(observation)
        self.weights.append(1.0)
        
        # Trim old observations with negligible weight
        while len(self.weights) > 1000 or (self.weights and self.weights[0] < 0.001):
            self.observations.pop(0)
            self.weights.pop(0)
    
    def get_weighted_stats(self) -> Dict:
        """Calculate weighted statistics."""
        if not self.observations:
            return {}
        
        total_weight = sum(self.weights)
        
        if total_weight == 0:
            return {}
        
        # Weighted win rate
        wins = [obs.get('actual_outcome', 0) for obs in self.observations]
        weighted_win_rate = np.average(wins, weights=self.weights)
        
        # Weighted PnL
        pnls = [obs.get('actual_pnl', 0) for obs in self.observations]
        weighted_avg_pnl = np.average(pnls, weights=self.weights)
        
        # Weighted variance
        weighted_var = np.average((np.array(pnls) - weighted_avg_pnl) ** 2, 
                                   weights=self.weights)
        
        return {
            'weighted_win_rate': weighted_win_rate,
            'weighted_avg_pnl': weighted_avg_pnl,
            'weighted_volatility': np.sqrt(weighted_var),
            'effective_sample_size': total_weight,
            'n_observations': len(self.observations)
        }


class MetaLearner:
    """
    Meta-learning layer that learns when to trust the model.
    """
    
    def __init__(self):
        self.model_confidence_history: deque = deque(maxlen=100)
        self.model_accuracy_by_confidence: Dict[str, List[bool]] = {
            'high': [],  # confidence > 0.8
            'medium': [],  # 0.6-0.8
            'low': []  # < 0.6
        }
    
    def record_prediction(self, confidence: float, was_correct: bool):
        """Record prediction outcome."""
        self.model_confidence_history.append(confidence)
        
        # Categorize by confidence
        if confidence > 0.8:
            self.model_accuracy_by_confidence['high'].append(was_correct)
        elif confidence > 0.6:
            self.model_accuracy_by_confidence['medium'].append(was_correct)
        else:
            self.model_accuracy_by_confidence['low'].append(was_correct)
        
        # Trim histories
        for key in self.model_accuracy_by_confidence:
            if len(self.model_accuracy_by_confidence[key]) > 100:
                self.model_accuracy_by_confidence[key] = \
                    self.model_accuracy_by_confidence[key][-100:]
    
    def get_trust_score(self, model_confidence: float) -> float:
        """
        Get adjusted trust score based on historical calibration.
        
        Returns multiplier (0-1) for model confidence.
        """
        # Calculate accuracy by confidence bucket
        accuracies = {}
        for bucket, results in self.model_accuracy_by_confidence.items():
            if results:
                accuracies[bucket] = np.mean(results)
        
        # Adjust trust based on historical accuracy
        if model_confidence > 0.8:
            return accuracies.get('high', model_confidence)
        elif model_confidence > 0.6:
            return accuracies.get('medium', model_confidence)
        else:
            return accuracies.get('low', model_confidence)
    
    def should_override_model(self, model_prediction: float, 
                              features: Dict) -> Tuple[bool, float]:
        """
        Decide whether to override model prediction.
        
        Returns:
            (should_override, adjusted_prediction)
        """
        trust_score = self.get_trust_score(abs(model_prediction - 0.5) + 0.5)
        
        # If trust is very low, move prediction toward 0.5 (uncertain)
        if trust_score < 0.5:
            adjusted = 0.5 + (model_prediction - 0.5) * trust_score
            return True, adjusted
        
        return False, model_prediction
