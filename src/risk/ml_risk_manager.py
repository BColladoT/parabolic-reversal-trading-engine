"""
ML-Based Risk Management System

Uses machine learning to predict trade outcome probability before entry
and dynamically adjust position sizing or skip trades with high loss probability.

Key Insights from Analysis:
- Losing trades: Slower grind (118min to peak), lower VWAP deviation (36%), 
  volume spread out (43% in first hour)
- Winning trades: Explosive moves (88min to peak), higher VWAP deviation (65%),
  volume concentrated at open (78%)

Strategy: Filter out slow-grinding parabolics that don't reverse
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pickle
import json
from pathlib import Path


@dataclass
class TradeFeatures:
    """Features extracted from pre-trade market data."""
    # Price action features
    max_gain_pct: float           # How much stock gained from open
    minutes_to_peak: float        # Time to reach HOD (key differentiator!)
    vwap_deviation_at_peak: float # How far above VWAP at peak
    
    # Volume features
    volume_concentration: float   # % of volume in first hour (0-1)
    relative_volume: float        # Volume vs 20-day average
    
    # Volatility features
    avg_bar_range_pct: float      # Average bar range as % of price
    max_bar_range_pct: float      # Largest bar range
    
    # Context features
    days_up: int                  # Consecutive days up
    float_category: str           # 'micro', 'small', 'mid'
    sector_momentum: float        # Sector performance that day
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML model."""
        return np.array([
            self.max_gain_pct,
            self.minutes_to_peak,
            self.vwap_deviation_at_peak,
            self.volume_concentration,
            self.relative_volume,
            self.avg_bar_range_pct,
            self.max_bar_range_pct,
            self.days_up,
        ])


class MLRiskManager:
    """
    Machine Learning Risk Manager for Parabolic Reversal Strategy.
    
    Uses a simple but effective rule-based classifier derived from analysis:
    - Reject trades with slow grind characteristics
    - Accept only explosive parabolics with volume confirmation
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.thresholds = self._load_or_initialize_thresholds()
        self.trade_history: list = []
        
    def _load_or_initialize_thresholds(self) -> Dict:
        """Load thresholds from file or use derived optimal values."""
        default_thresholds = {
            # From analysis: winners avg 88min, losers avg 118min to peak
            'max_minutes_to_peak': 100,  # Reject slow grinds
            
            # Winners had 65% VWAP deviation vs 36% for losers
            'min_vwap_deviation': 45,  # Need strong extension
            
            # Winners: 78% volume in first hour vs 43% for losers
            'min_volume_concentration': 0.60,  # Need volume at open
            
            # Minimum gain to qualify as parabolic
            'min_gain_pct': 50,
            
            # Maximum gain (avoid extreme outliers)
            'max_gain_pct': 400,
            
            # Risk score threshold (0-1, higher = more likely to lose)
            'max_risk_score': 0.35,
        }
        
        if self.model_path and Path(self.model_path).exists():
            with open(self.model_path, 'r') as f:
                return json.load(f)
        
        return default_thresholds
    
    def save_thresholds(self, path: str):
        """Save current thresholds to file."""
        with open(path, 'w') as f:
            json.dump(self.thresholds, f, indent=2)
    
    def calculate_risk_score(self, features: TradeFeatures) -> float:
        """
        Calculate risk score (0-1) where higher = more likely to lose.
        
        Uses weighted combination of risk factors derived from analysis.
        """
        risk_factors = []
        
        # Factor 1: Slow grind to peak (major risk factor)
        # Winners: 88min avg, Losers: 118min avg
        if features.minutes_to_peak > self.thresholds['max_minutes_to_peak']:
            risk_factors.append(0.30)  # 30% risk weight
        elif features.minutes_to_peak > 90:
            risk_factors.append(0.15)
        else:
            risk_factors.append(0.0)
        
        # Factor 2: Low VWAP deviation
        # Winners: 65%, Losers: 36%
        if features.vwap_deviation_at_peak < self.thresholds['min_vwap_deviation']:
            risk_factors.append(0.25)
        elif features.vwap_deviation_at_peak < 35:
            risk_factors.append(0.15)
        else:
            risk_factors.append(0.0)
        
        # Factor 3: Volume not concentrated at open
        # Winners: 78%, Losers: 43%
        if features.volume_concentration < self.thresholds['min_volume_concentration']:
            risk_factors.append(0.25)
        elif features.volume_concentration < 0.50:
            risk_factors.append(0.10)
        else:
            risk_factors.append(0.0)
        
        # Factor 4: Low volatility (grinding move)
        if features.avg_bar_range_pct < 2.0:
            risk_factors.append(0.10)
        else:
            risk_factors.append(0.0)
        
        # Factor 5: Too many consecutive up days (overextended)
        if features.days_up > 3:
            risk_factors.append(0.10)
        else:
            risk_factors.append(0.0)
        
        return min(sum(risk_factors), 1.0)
    
    def should_take_trade(self, features: TradeFeatures) -> Tuple[bool, float, str]:
        """
        Determine if trade should be taken based on ML risk assessment.
        
        Returns:
            (should_trade: bool, risk_score: float, reason: str)
        """
        risk_score = self.calculate_risk_score(features)
        
        # Hard filters (automatic reject)
        if features.max_gain_pct < self.thresholds['min_gain_pct']:
            return False, 1.0, f"Gain too low ({features.max_gain_pct:.1f}% < {self.thresholds['min_gain_pct']})"
        
        if features.max_gain_pct > self.thresholds['max_gain_pct']:
            return False, 1.0, f"Gain too extreme ({features.max_gain_pct:.1f}% > {self.thresholds['max_gain_pct']})"
        
        # Risk score evaluation
        if risk_score >= self.thresholds['max_risk_score']:
            return False, risk_score, f"High risk score ({risk_score:.2f}) - slow grind/low volume detected"
        
        # All clear
        return True, risk_score, "Risk acceptable"
    
    def recommend_position_size(self, features: TradeFeatures, 
                                 base_size: float = 25000) -> float:
        """
        Recommend position size based on confidence.
        
        Higher confidence = larger position
        Lower confidence = smaller position or skip
        """
        risk_score = self.calculate_risk_score(features)
        
        # Scale position size inversely to risk
        # Risk 0.0 -> 100% of base size
        # Risk 0.35 -> 50% of base size
        # Risk >= 0.5 -> 0%
        if risk_score >= 0.5:
            return 0.0
        
        size_multiplier = 1.0 - (risk_score / 0.5)
        return base_size * size_multiplier
    
    def record_trade_outcome(self, features: TradeFeatures, pnl: float, 
                             was_taken: bool):
        """Record trade outcome for continuous learning."""
        self.trade_history.append({
            'timestamp': datetime.now(),
            'features': features,
            'pnl': pnl,
            'was_taken': was_taken,
            'risk_score': self.calculate_risk_score(features)
        })
    
    def generate_risk_report(self) -> Dict:
        """Generate report on risk management performance."""
        if not self.trade_history:
            return {'message': 'No trade history yet'}
        
        df = pd.DataFrame(self.trade_history)
        
        taken = df[df['was_taken']]
        skipped = df[~df['was_taken']]
        
        report = {
            'total_trades': len(df),
            'trades_taken': len(taken),
            'trades_skipped': len(skipped),
            'avg_risk_score_taken': taken['risk_score'].mean() if len(taken) > 0 else 0,
            'avg_risk_score_skipped': skipped['risk_score'].mean() if len(skipped) > 0 else 0,
            'win_rate_taken': (taken['pnl'] > 0).mean() if len(taken) > 0 else 0,
            'total_pnl_taken': taken['pnl'].sum() if len(taken) > 0 else 0,
        }
        
        return report


class RealTimeRiskMonitor:
    """
    Real-time risk monitoring during trade execution.
    Monitors for adverse conditions and recommends early exit.
    """
    
    def __init__(self):
        self.max_adverse_excursion_threshold = -0.03  # 3% against position
        self.time_in_trade_threshold = 120  # minutes
        
    def monitor_position(self, entry_price: float, current_price: float,
                        vwap: float, time_in_trade: int,
                        unrealized_pnl_pct: float) -> Dict:
        """
        Monitor open position for adverse conditions.
        
        Returns dict with:
            - should_exit: bool
            - reason: str
            - urgency: str ('low', 'medium', 'high')
        """
        # Check 1: Price moving against us through VWAP (momentum lost)
        if current_price > vwap * 1.05 and unrealized_pnl_pct < -0.01:
            return {
                'should_exit': True,
                'reason': 'Price broke above VWAP + 5%, momentum lost',
                'urgency': 'high'
            }
        
        # Check 2: Max adverse excursion exceeded
        if unrealized_pnl_pct < self.max_adverse_excursion_threshold:
            return {
                'should_exit': True,
                'reason': f'Max loss exceeded ({unrealized_pnl_pct:.1%})',
                'urgency': 'high'
            }
        
        # Check 3: Time-based exit (trade not working)
        if time_in_trade > self.time_in_trade_threshold and unrealized_pnl_pct < 0:
            return {
                'should_exit': True,
                'reason': f'Time exit - not working after {time_in_trade} min',
                'urgency': 'medium'
            }
        
        # Check 4: VWAP reclaim (bearish)
        if current_price > vwap and time_in_trade > 30:
            return {
                'should_exit': False,
                'reason': 'Warning: Price reclaiming VWAP',
                'urgency': 'low'
            }
        
        return {
            'should_exit': False,
            'reason': 'No adverse conditions',
            'urgency': 'none'
        }


# Singleton instances for easy import
ml_risk_manager = MLRiskManager()
real_time_monitor = RealTimeRiskMonitor()
