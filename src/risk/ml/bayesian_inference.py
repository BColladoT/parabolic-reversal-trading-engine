"""
Bayesian Inference for Risk Assessment

Implements probabilistic risk assessment using:
- Bayesian updating of win probability
- Credible intervals for predictions
- Prior calibration from historical data
- Posterior predictive distributions
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from scipy.special import expit, logit
import warnings


class BayesianRiskAssessor:
    """
    Bayesian risk assessment with dynamic prior updating.
    
    Uses Beta-Binomial model for win probability with
    Normal-Normal model for expected returns.
    """
    
    def __init__(self, prior_wins: int = 50, prior_losses: int = 20):
        """
        Initialize with prior beliefs.
        
        Args:
            prior_wins: Prior wins (from historical win rate ~70%)
            prior_losses: Prior losses
        """
        # Beta distribution parameters for win probability
        self.alpha = prior_wins
        self.beta = prior_losses
        
        # Normal distribution parameters for returns
        self.return_mean = 1500  # $1,500 avg win
        self.return_precision = 1 / 2000**2  # Precision = 1/variance
        
        # Historical tracking for adaptive updating
        self.observed_wins = 0
        self.observed_losses = 0
        self.observed_returns = []
        
        # Model calibration parameters
        self.model_reliability = 0.8  # How much to trust model vs prior
        self.confidence_threshold = 0.6
    
    def update(self, ensemble_prediction: Dict, features: np.ndarray) -> Dict:
        """
        Update beliefs with new ensemble prediction.
        
        Args:
            ensemble_prediction: Output from ensemble model
            features: Feature vector
            
        Returns:
            Dict with posterior estimates
        """
        # Extract ensemble prediction
        model_prob = ensemble_prediction['probability'][0] if isinstance(
            ensemble_prediction['probability'], np.ndarray) else ensemble_prediction['probability']
        model_confidence = ensemble_prediction['confidence'][0] if isinstance(
            ensemble_prediction['confidence'], np.ndarray) else ensemble_prediction['confidence']
        
        # Bayesian update for win probability
        posterior_prob, credible_interval = self._update_win_probability(
            model_prob, model_confidence
        )
        
        # Bayesian update for expected return
        expected_return, return_ci = self._update_expected_return(
            model_prob, features
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            posterior_prob, expected_return, credible_interval
        )
        
        return {
            'win_probability': posterior_prob,
            'win_prob_ci': credible_interval,
            'expected_return': expected_return,
            'return_ci': return_ci,
            'confidence': model_confidence * self.model_reliability,
            'ensemble_votes': ensemble_prediction.get('model_votes', {}),
            'posterior_alpha': self.alpha + self.observed_wins,
            'posterior_beta': self.beta + self.observed_losses,
            **risk_metrics
        }
    
    def _update_win_probability(self, model_prob: float, 
                                 confidence: float) -> Tuple[float, Tuple[float, float]]:
        """
        Bayesian update of win probability using Beta-Binomial model.
        
        Prior: Beta(alpha, beta)
        Likelihood: Binomial from model prediction weighted by confidence
        Posterior: Beta(alpha + pseudo_wins, beta + pseudo_losses)
        """
        # Convert model probability to pseudo-observations
        # Weight by model confidence
        pseudo_observations = 10 * confidence
        pseudo_wins = model_prob * pseudo_observations
        pseudo_losses = (1 - model_prob) * pseudo_observations
        
        # Update Beta parameters
        posterior_alpha = self.alpha + pseudo_wins
        posterior_beta = self.beta + pseudo_losses
        
        # Posterior mean (expected win probability)
        posterior_prob = posterior_alpha / (posterior_alpha + posterior_beta)
        
        # Credible interval (95%)
        ci_lower = stats.beta.ppf(0.025, posterior_alpha, posterior_beta)
        ci_upper = stats.beta.ppf(0.975, posterior_alpha, posterior_beta)
        
        return posterior_prob, (ci_lower, ci_upper)
    
    def _update_expected_return(self, win_prob: float, 
                                 features: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """
        Bayesian update of expected return using Normal-Normal model.
        
        Models the distribution of returns given win/loss outcome.
        """
        # Prior for win return: N(4000, 2000^2) - $4K avg win, $2K std
        # Prior for loss return: N(-2500, 1500^2) - $2.5K avg loss, $1.5K std
        
        win_return_mean = 4000
        win_return_var = 2000**2
        loss_return_mean = -2500
        loss_return_var = 1500**2
        
        # Expected return as weighted average
        expected_win = win_prob * win_return_mean
        expected_loss = (1 - win_prob) * loss_return_mean
        expected_return = expected_win + expected_loss
        
        # Variance of return
        return_var = (win_prob * (win_return_var + win_return_mean**2) + 
                     (1 - win_prob) * (loss_return_var + loss_return_mean**2) - 
                     expected_return**2)
        
        # Credible interval
        ci_lower = expected_return - 1.96 * np.sqrt(return_var)
        ci_upper = expected_return + 1.96 * np.sqrt(return_var)
        
        return expected_return, (ci_lower, ci_upper)
    
    def _calculate_risk_metrics(self, win_prob: float, expected_return: float,
                                 win_prob_ci: Tuple[float, float]) -> Dict:
        """Calculate additional risk metrics."""
        # Probability of ruin (simplified - assuming $25K position, $5K max loss)
        max_loss = 5000
        position_size = 25000
        
        # Approximate probability of hitting max loss
        # Using normal approximation
        loss_threshold = -max_loss / position_size
        prob_ruin_approx = stats.norm.cdf(loss_threshold, 
                                         loc=expected_return/position_size, 
                                         scale=0.15)  # 15% daily vol assumption
        
        # Edge ratio (expected return / tail risk)
        tail_risk = abs(win_prob_ci[0] - win_prob) * 10000  # Approximate $ loss at 95% CI
        edge_ratio = expected_return / tail_risk if tail_risk > 0 else 0
        
        # Kelly fraction (simplified)
        avg_win = 4000
        avg_loss = 2500
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly = max(0, min(kelly, 0.5))  # Bounded
        
        return {
            'prob_ruin_approx': prob_ruin_approx,
            'edge_ratio': edge_ratio,
            'kelly_fraction': kelly,
            'sharpe_estimate': expected_return / 2000 if 2000 > 0 else 0,  # Rough Sharpe
        }
    
    def calibrate(self, X: np.ndarray, y: np.ndarray):
        """
        Calibrate prior parameters from historical data.
        
        Updates prior beliefs based on actual outcomes.
        """
        # Update win/loss counts
        self.observed_wins = np.sum(y == 1)
        self.observed_losses = np.sum(y == 0)
        
        # Update Beta prior
        self.alpha += self.observed_wins
        self.beta += self.observed_losses
        
        # Update return prior if we have P&L data
        # This would require actual returns, not just win/loss
        
        print(f"[BAYESIAN] Calibrated with {self.observed_wins} wins, {self.observed_losses} losses")
        print(f"[BAYESIAN] Posterior win rate: {self.alpha/(self.alpha+self.beta):.2%}")
    
    def update_with_outcome(self, prediction: Dict, actual_outcome: int, 
                           actual_return: float):
        """
        Update beliefs with actual trade outcome (online learning).
        
        Args:
            prediction: The prediction dict from update()
            actual_outcome: 1 for win, 0 for loss
            actual_return: Actual P&L
        """
        # Update counts
        if actual_outcome == 1:
            self.observed_wins += 1
        else:
            self.observed_losses += 1
        
        self.observed_returns.append(actual_return)
        
        # Adapt model reliability based on accuracy
        predicted_win = prediction['win_probability'] > 0.5
        if predicted_win == (actual_outcome == 1):
            # Prediction was correct, increase reliability slightly
            self.model_reliability = min(0.95, self.model_reliability + 0.01)
        else:
            # Prediction was wrong, decrease reliability
            self.model_reliability = max(0.5, self.model_reliability - 0.02)
        
        # Keep recent history only
        if len(self.observed_returns) > 1000:
            self.observed_returns = self.observed_returns[-1000:]
    
    def get_prior_strength(self) -> float:
        """Get effective sample size of prior."""
        return self.alpha + self.beta
    
    def reset_priors(self, prior_wins: int = 50, prior_losses: int = 20):
        """Reset to initial priors."""
        self.alpha = prior_wins
        self.beta = prior_losses
        self.observed_wins = 0
        self.observed_losses = 0
        self.observed_returns = []


class BayesianPortfolioOptimizer:
    """
    Bayesian optimization of trade allocation across multiple opportunities.
    """
    
    def __init__(self, max_positions: int = 3, risk_aversion: float = 2.0):
        self.max_positions = max_positions
        self.risk_aversion = risk_aversion
    
    def optimize_allocation(self, trade_assessments: list) -> Dict:
        """
        Optimize position sizing across multiple trade opportunities.
        
        Args:
            trade_assessments: List of assessment dicts from InstitutionalRiskManager
            
        Returns:
            Dict with allocation decisions
        """
        if not trade_assessments:
            return {'allocations': [], 'expected_portfolio_return': 0}
        
        # Extract expected returns and variances
        expected_returns = np.array([t['expected_return'] for t in trade_assessments])
        uncertainties = np.array([1 - t['confidence'] for t in trade_assessments])
        win_probs = np.array([t['win_probability'] for t in trade_assessments])
        
        # Risk-adjusted expected returns
        risk_adjusted_returns = expected_returns / (1 + self.risk_aversion * uncertainties)
        
        # Sort by risk-adjusted return
        sorted_indices = np.argsort(risk_adjusted_returns)[::-1]
        
        # Select top N positions
        selected = sorted_indices[:self.max_positions]
        
        # Allocate capital proportionally to Kelly fraction
        kelly_fractions = np.array([trade_assessments[i]['kelly_fraction'] for i in selected])
        
        # Normalize to sum to 1
        if kelly_fractions.sum() > 0:
            allocations = kelly_fractions / kelly_fractions.sum()
        else:
            allocations = np.ones(len(selected)) / len(selected)
        
        # Calculate portfolio metrics
        portfolio_return = sum(
            allocations[i] * expected_returns[selected[i]] 
            for i in range(len(selected))
        )
        
        portfolio_variance = sum(
            allocations[i]**2 * uncertainties[selected[i]] * 10000**2
            for i in range(len(selected))
        )
        
        return {
            'selected_trades': [trade_assessments[i] for i in selected],
            'allocations': allocations.tolist(),
            'expected_portfolio_return': portfolio_return,
            'portfolio_variance': portfolio_variance,
            'portfolio_sharpe': portfolio_return / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0,
            'diversification_score': 1 - np.sum(allocations**2)  # Herfindahl index inverse
        }


class PredictiveDistribution:
    """
    Model the full predictive distribution of trade outcomes.
    """
    
    def __init__(self):
        self.win_mixture = None
        self.loss_mixture = None
    
    def fit(self, historical_returns: np.ndarray, outcomes: np.ndarray):
        """
        Fit mixture models to historical returns.
        
        Models wins and losses separately as Gaussian mixtures.
        """
        wins = historical_returns[outcomes == 1]
        losses = historical_returns[outcomes == 0]
        
        # Simple Gaussian fit (could extend to mixture)
        if len(wins) > 5:
            self.win_mean = np.mean(wins)
            self.win_std = np.std(wins)
        else:
            self.win_mean = 4000
            self.win_std = 2000
        
        if len(losses) > 5:
            self.loss_mean = np.mean(losses)
            self.loss_std = np.std(losses)
        else:
            self.loss_mean = -2500
            self.loss_std = 1500
    
    def sample(self, win_probability: float, n_samples: int = 1000) -> np.ndarray:
        """Sample from predictive distribution."""
        outcomes = np.random.random(n_samples) < win_probability
        
        samples = np.zeros(n_samples)
        n_wins = np.sum(outcomes)
        n_losses = n_samples - n_wins
        
        samples[outcomes] = np.random.normal(self.win_mean, self.win_std, n_wins)
        samples[~outcomes] = np.random.normal(self.loss_mean, self.loss_std, n_losses)
        
        return samples
    
    def calculate_var(self, win_probability: float, confidence: float = 0.95) -> float:
        """Calculate Value at Risk from predictive distribution."""
        samples = self.sample(win_probability, n_samples=10000)
        return np.percentile(samples, (1 - confidence) * 100)
    
    def calculate_cvar(self, win_probability: float, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        samples = self.sample(win_probability, n_samples=10000)
        var = np.percentile(samples, (1 - confidence) * 100)
        return np.mean(samples[samples <= var])
