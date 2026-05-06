"""
Institutional-Grade Machine Learning Risk Management System

A comprehensive ML framework for trade risk assessment inspired by
quantitative hedge fund practices (Renaissance Technologies, Two Sigma, etc.)

Components:
    - feature_engineering: Advanced feature extraction from market microstructure
    - ensemble_models: Multi-model ensemble with XGBoost, Random Forest, and Neural Networks
    - bayesian_inference: Probabilistic risk assessment with confidence intervals
    - risk_metrics: VaR, CVaR, Kelly Criterion, and drawdown analysis
    - online_learning: Adaptive model updates with concept drift detection
    - model_validator: Rigorous backtesting and statistical validation

Usage:
    from src.risk.ml import InstitutionalRiskManager
    
    risk_manager = InstitutionalRiskManager()
    
    # Get comprehensive risk assessment
    assessment = risk_manager.assess_trade(features)
    
    # Returns dict with:
    # - win_probability: Bayesian probability of winning
    # - expected_return: Expected P&L
    # - kelly_fraction: Optimal position sizing (Kelly Criterion)
    # - var_95: Value at Risk (95% confidence)
    # - risk_score: Composite risk metric (0-1)
    # - model_confidence: Model certainty in prediction
"""

from .feature_engineering import MarketMicrostructureFeatures, FeaturePipeline
from .ensemble_models import EnsembleRiskModel
from .bayesian_inference import BayesianRiskAssessor
from .risk_metrics import RiskMetricsCalculator
from .online_learning import AdaptiveRiskModel
from .model_validator import ModelValidator

# Main interface class
class InstitutionalRiskManager:
    """
    High-level interface for institutional-grade risk management.
    
    Combines multiple ML models, Bayesian inference, and risk metrics
    into a unified risk assessment framework.
    """
    
    def __init__(self, model_path: str = None):
        self.feature_pipeline = FeaturePipeline()
        self.ensemble = EnsembleRiskModel()
        self.bayesian = BayesianRiskAssessor()
        self.risk_calc = RiskMetricsCalculator()
        self.adaptive = AdaptiveRiskModel()
        self.validator = ModelValidator()
        
        # Load pre-trained models if available
        if model_path:
            self.load_models(model_path)
    
    def assess_trade(self, raw_data: dict) -> dict:
        """
        Comprehensive trade risk assessment.
        
        Returns:
            dict with complete risk profile including:
            - win_probability: float (0-1)
            - expected_return: float ($)
            - kelly_fraction: float (0-1, position size)
            - var_95: float ($)
            - cvar_95: float ($)
            - risk_score: float (0-1)
            - model_confidence: float (0-1)
            - recommendation: str ('STRONG_BUY', 'BUY', 'NEUTRAL', 'AVOID')
        """
        # Extract features
        features = self.feature_pipeline.transform(raw_data)
        
        # Ensemble prediction
        ensemble_pred = self.ensemble.predict(features)
        
        # Bayesian update
        bayesian_pred = self.bayesian.update(ensemble_pred, features)
        
        # Risk metrics
        risk_metrics = self.risk_calc.calculate(features, bayesian_pred)
        
        # Combine into recommendation
        return self._compile_assessment(bayesian_pred, risk_metrics)
    
    def _compile_assessment(self, bayesian_pred, risk_metrics) -> dict:
        """Compile final risk assessment."""
        win_prob = bayesian_pred['win_probability']
        expected_return = bayesian_pred['expected_return']
        confidence = bayesian_pred['confidence']
        
        # Kelly Criterion calculation
        kelly = self._kelly_criterion(win_prob, expected_return, risk_metrics['avg_loss'])
        
        # Risk score (composite)
        risk_score = self._calculate_risk_score(win_prob, confidence, risk_metrics)
        
        # Recommendation
        recommendation = self._get_recommendation(win_prob, risk_score, kelly)
        
        return {
            'win_probability': win_prob,
            'expected_return': expected_return,
            'kelly_fraction': kelly,
            'var_95': risk_metrics['var_95'],
            'cvar_95': risk_metrics['cvar_95'],
            'risk_score': risk_score,
            'model_confidence': confidence,
            'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
            'max_drawdown': risk_metrics.get('max_drawdown', 0),
            'recommendation': recommendation,
            'raw_predictions': bayesian_pred['ensemble_votes']
        }
    
    def _kelly_criterion(self, win_prob, avg_win, avg_loss) -> float:
        """Calculate Kelly Criterion for optimal position sizing."""
        if avg_loss == 0:
            return 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        return max(0, min(kelly, 0.5))  # Cap at 50% for safety
    
    def _calculate_risk_score(self, win_prob, confidence, risk_metrics) -> float:
        """Calculate composite risk score (0 = safe, 1 = dangerous)."""
        # Inverse of win probability (higher = more risk)
        prob_component = (1 - win_prob) * 0.4
        
        # Model uncertainty
        confidence_component = (1 - confidence) * 0.2
        
        # VaR component
        var_component = min(abs(risk_metrics['var_95']) / 10000, 1) * 0.2
        
        # Drawdown component
        dd_component = abs(risk_metrics.get('max_drawdown', 0)) * 0.2
        
        return prob_component + confidence_component + var_component + dd_component
    
    def _get_recommendation(self, win_prob, risk_score, kelly) -> str:
        """Generate trade recommendation."""
        if win_prob > 0.75 and risk_score < 0.3 and kelly > 0.3:
            return 'STRONG_BUY'
        elif win_prob > 0.65 and risk_score < 0.5 and kelly > 0.15:
            return 'BUY'
        elif win_prob > 0.55 and risk_score < 0.6 and kelly > 0.05:
            return 'NEUTRAL'
        else:
            return 'AVOID'
    
    def train(self, X_train, y_train, validation_split=0.2):
        """Train all models with cross-validation."""
        # Feature engineering
        X_features = self.feature_pipeline.fit_transform(X_train)
        
        # Train ensemble
        self.ensemble.train(X_features, y_train, validation_split)
        
        # Calibrate Bayesian
        self.bayesian.calibrate(X_features, y_train)
        
        # Validate
        return self.validator.validate(self, X_features, y_train)
    
    def update_online(self, trade_result: dict):
        """Update models with new trade result (online learning)."""
        self.adaptive.update(trade_result)
        
        # Check for concept drift
        if self.adaptive.detect_drift():
            print("[WARNING] Concept drift detected - retraining recommended")
    
    def load_models(self, path: str):
        """Load pre-trained models."""
        import pickle
        with open(path, 'rb') as f:
            models = pickle.load(f)
        self.ensemble = models['ensemble']
        self.bayesian = models['bayesian']
    
    def save_models(self, path: str):
        """Save trained models."""
        import pickle
        models = {
            'ensemble': self.ensemble,
            'bayesian': self.bayesian
        }
        with open(path, 'wb') as f:
            pickle.dump(models, f)


__all__ = [
    'InstitutionalRiskManager',
    'MarketMicrostructureFeatures',
    'FeaturePipeline',
    'EnsembleRiskModel',
    'BayesianRiskAssessor',
    'RiskMetricsCalculator',
    'AdaptiveRiskModel',
    'ModelValidator'
]
