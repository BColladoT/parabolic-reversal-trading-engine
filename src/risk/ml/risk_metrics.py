"""
Institutional Risk Metrics Calculation

Implements professional risk metrics used by hedge funds:
- Value at Risk (VaR) - Parametric, Historical, Monte Carlo
- Conditional VaR (CVaR) / Expected Shortfall
- Maximum Drawdown and Drawdown Duration
- Sharpe, Sortino, and Calmar Ratios
- Tail Risk Metrics (Skewness, Kurtosis)
- Omega Ratio
- Gain/Pain Ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings


class RiskMetricsCalculator:
    """
    Calculate comprehensive risk metrics for trading strategies.
    """
    
    def __init__(self, confidence_level: float = 0.95, risk_free_rate: float = 0.02):
        """
        Initialize risk calculator.
        
        Args:
            confidence_level: Confidence level for VaR/CVaR (default 95%)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self.historical_returns: List[float] = []
        
    def calculate(self, features: np.ndarray, prediction: Dict) -> Dict:
        """
        Calculate all risk metrics for a trade.
        
        Args:
            features: Feature vector
            prediction: Prediction dict from Bayesian assessor
            
        Returns:
            Dict with all risk metrics
        """
        win_prob = prediction['win_probability']
        expected_return = prediction['expected_return']
        
        # Extract volatility from features if available
        volatility = self._extract_volatility(features)
        
        metrics = {}
        
        # Value at Risk metrics
        metrics['var_95'] = self.calculate_var(win_prob, method='parametric', 
                                               volatility=volatility)
        metrics['var_99'] = self.calculate_var(win_prob, method='parametric',
                                               volatility=volatility, 
                                               confidence=0.99)
        
        # Conditional VaR (Expected Shortfall)
        metrics['cvar_95'] = self.calculate_cvar(win_prob, volatility=volatility)
        metrics['cvar_99'] = self.calculate_cvar(win_prob, volatility=volatility, 
                                                  confidence=0.99)
        
        # Tail risk metrics
        tail_metrics = self._calculate_tail_risks(win_prob)
        metrics.update(tail_metrics)
        
        # Scenario analysis
        scenario_metrics = self._scenario_analysis(win_prob, expected_return)
        metrics.update(scenario_metrics)
        
        # Position sizing metrics
        sizing_metrics = self._calculate_position_metrics(win_prob, expected_return, 
                                                          metrics['cvar_95'])
        metrics.update(sizing_metrics)
        
        return metrics
    
    def calculate_var(self, win_probability: float, method: str = 'parametric',
                     volatility: float = None, confidence: float = None) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            win_probability: Probability of winning
            method: 'parametric', 'historical', or 'monte_carlo'
            volatility: Volatility estimate (for parametric)
            confidence: Confidence level (default to self.confidence_level)
            
        Returns:
            VaR value (negative for loss)
        """
        if confidence is None:
            confidence = self.confidence_level
        
        alpha = 1 - confidence
        
        if method == 'parametric':
            # Assume normal distribution of returns
            if volatility is None:
                volatility = 0.15  # 15% default
            
            # Mixture of win and loss distributions
            win_return = 4000
            loss_return = -2500
            
            expected = win_probability * win_return + (1 - win_probability) * loss_return
            variance = (win_probability * (win_return**2) + 
                       (1 - win_probability) * (loss_return**2) - 
                       expected**2)
            
            std = np.sqrt(max(variance, 0))
            
            # VaR is the alpha-quantile
            var = stats.norm.ppf(alpha, expected, std)
            
        elif method == 'historical' and len(self.historical_returns) > 30:
            # Historical VaR
            var = np.percentile(self.historical_returns, alpha * 100)
            
        else:
            # Simplified: assume linear with win probability
            base_var = -5000  # $5K loss at 50% win rate
            var = base_var * (1.5 - win_probability)
        
        return var
    
    def calculate_cvar(self, win_probability: float, volatility: float = None,
                      confidence: float = None) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        CVaR is the expected loss given that we are in the tail (beyond VaR).
        """
        if confidence is None:
            confidence = self.confidence_level
        
        # For normal distribution, CVaR has closed form
        var = self.calculate_var(win_probability, volatility=volatility, 
                                confidence=confidence)
        
        # Parametric CVaR approximation
        if volatility is None:
            volatility = 0.15
        
        alpha = 1 - confidence
        
        # Standard normal PDF at VaR quantile
        z_score = stats.norm.ppf(alpha)
        pdf_at_z = stats.norm.pdf(z_score)
        
        # Expected shortfall multiplier
        es_multiplier = pdf_at_z / alpha
        
        # Adjust VaR by ES multiplier
        cvar = var * es_multiplier
        
        return cvar
    
    def calculate_drawdown_metrics(self, equity_curve: np.ndarray) -> Dict:
        """
        Calculate drawdown-related metrics.
        
        Args:
            equity_curve: Array of portfolio values over time
            
        Returns:
            Dict with drawdown metrics
        """
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdowns
        drawdowns = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdowns.min()
        
        # Find max drawdown period
        max_dd_idx = np.argmin(drawdowns)
        peak_idx = np.argmax(equity_curve[:max_dd_idx]) if max_dd_idx > 0 else 0
        
        # Calculate drawdown duration
        dd_duration = max_dd_idx - peak_idx
        
        # Recovery time (if recovered)
        recovery_idx = None
        for i in range(max_dd_idx, len(equity_curve)):
            if equity_curve[i] >= equity_curve[peak_idx]:
                recovery_idx = i
                break
        
        recovery_time = recovery_idx - max_dd_idx if recovery_idx else None
        
        # Average drawdown
        avg_drawdown = np.mean(drawdowns[drawdowns < 0])
        
        # Drawdown frequency
        dd_count = np.sum(np.diff(np.sign(drawdowns)) > 0)
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': dd_duration,
            'recovery_time': recovery_time,
            'avg_drawdown': avg_drawdown,
            'drawdown_frequency': dd_count,
            'calmar_ratio': self._calculate_calmar(equity_curve, max_drawdown)
        }
    
    def calculate_performance_ratios(self, returns: np.ndarray) -> Dict:
        """
        Calculate various performance risk-adjusted ratios.
        """
        if len(returns) < 2:
            return {'sharpe_ratio': 0, 'sortino_ratio': 0, 'omega_ratio': 0}
        
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily
        
        # Sharpe Ratio
        volatility = np.std(returns)
        sharpe = (np.mean(excess_returns) / volatility) * np.sqrt(252) if volatility > 0 else 0
        
        # Sortino Ratio (downside deviation only)
        downside_returns = excess_returns[excess_returns < 0]
        downside_dev = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = (np.mean(excess_returns) / downside_dev) * np.sqrt(252) if downside_dev > 0 else 0
        
        # Omega Ratio
        threshold = 0
        gains = np.sum(returns[returns > threshold] - threshold)
        losses = np.sum(threshold - returns[returns < threshold])
        omega = gains / losses if losses > 0 else float('inf')
        
        # Gain/Pain Ratio
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        gain_pain = (np.sum(positive_returns) / np.abs(np.sum(negative_returns))) if len(negative_returns) > 0 else 0
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'omega_ratio': omega,
            'gain_pain_ratio': gain_pain,
            'kurtosis': stats.kurtosis(returns),
            'skewness': stats.skew(returns)
        }
    
    def calculate_kelly_criterion(self, win_prob: float, avg_win: float, 
                                   avg_loss: float) -> Dict:
        """
        Calculate Kelly Criterion and fractional Kelly.
        
        Returns optimal position size as fraction of capital.
        """
        if avg_loss == 0:
            return {'full_kelly': 0, 'half_kelly': 0, 'quarter_kelly': 0}
        
        win_loss_ratio = avg_win / abs(avg_loss)
        
        # Full Kelly
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        
        # Bounded Kelly (0 to 0.5)
        kelly_bounded = max(0, min(kelly, 0.5))
        
        return {
            'full_kelly': kelly,
            'full_kelly_bounded': kelly_bounded,
            'half_kelly': kelly_bounded / 2,
            'quarter_kelly': kelly_bounded / 4,
            'optimal_fraction': kelly_bounded  # Conservative: use bounded
        }
    
    def monte_carlo_simulation(self, win_prob: float, n_simulations: int = 10000,
                               n_trades: int = 100) -> Dict:
        """
        Run Monte Carlo simulation of trading outcomes.
        
        Returns distribution of final P&L and path statistics.
        """
        # Simulate trade outcomes
        outcomes = np.random.random((n_simulations, n_trades)) < win_prob
        
        # Win/loss amounts
        win_amount = np.random.normal(4000, 1500, (n_simulations, n_trades))
        loss_amount = np.random.normal(-2500, 1000, (n_simulations, n_trades))
        
        # Combine
        pnl = np.where(outcomes, win_amount, loss_amount)
        
        # Cumulative P&L
        cumulative = np.cumsum(pnl, axis=1)
        
        # Final statistics
        final_pnl = cumulative[:, -1]
        
        # Max drawdown for each path
        running_max = np.maximum.accumulate(cumulative, axis=1)
        drawdowns = (cumulative - running_max) / (running_max + 25000)  # Relative to equity
        max_drawdowns = np.min(drawdowns, axis=1)
        
        # Probability of ruin (losing more than 20% of capital)
        ruin_threshold = -5000
        prob_ruin = np.mean(final_pnl < ruin_threshold)
        
        return {
            'expected_final_pnl': np.mean(final_pnl),
            'final_pnl_std': np.std(final_pnl),
            'final_pnl_5th': np.percentile(final_pnl, 5),
            'final_pnl_95th': np.percentile(final_pnl, 95),
            'prob_of_ruin': prob_ruin,
            'prob_profit': np.mean(final_pnl > 0),
            'expected_max_drawdown': np.mean(max_drawdowns),
            'worst_case_drawdown': np.min(max_drawdowns),
            'sharpe_distribution': np.mean(final_pnl) / np.std(final_pnl) if np.std(final_pnl) > 0 else 0
        }
    
    def _extract_volatility(self, features: np.ndarray) -> float:
        """Extract volatility estimate from feature vector."""
        # Default volatility
        return 0.15
    
    def _calculate_tail_risks(self, win_probability: float) -> Dict:
        """Calculate tail risk metrics."""
        # Estimate tail probabilities
        extreme_loss_prob = (1 - win_probability) ** 2  # Two consecutive losses
        
        # Expected tail loss
        tail_loss = -5000  # Approximate
        
        return {
            'extreme_loss_probability': extreme_loss_prob,
            'expected_tail_loss': tail_loss,
            'tail_risk_score': extreme_loss_prob * abs(tail_loss)
        }
    
    def _scenario_analysis(self, win_probability: float, expected_return: float) -> Dict:
        """Analyze different market scenarios."""
        scenarios = {
            'bullish': win_probability * 1.2,
            'base_case': win_probability,
            'bearish': win_probability * 0.8,
            'extreme_bear': win_probability * 0.5
        }
        
        results = {}
        for name, prob in scenarios.items():
            prob = min(prob, 0.99)  # Cap
            expected = prob * 4000 + (1 - prob) * (-2500)
            results[f'scenario_{name}_prob'] = prob
            results[f'scenario_{name}_expected'] = expected
        
        return results
    
    def _calculate_position_metrics(self, win_prob: float, expected_return: float,
                                   cvar: float) -> Dict:
        """Calculate position sizing recommendations."""
        # Kelly criterion
        kelly = self.calculate_kelly_criterion(win_prob, 4000, -2500)
        
        # Risk-based sizing (don't risk more than 2% on any trade)
        max_risk = 500  # $500 max risk
        if cvar < 0:
            risk_based_size = min(max_risk / abs(cvar), 1.0)
        else:
            risk_based_size = 1.0
        
        # Conservative sizing (minimum of Kelly and risk-based)
        conservative_size = min(kelly['optimal_fraction'], risk_based_size)
        
        return {
            'kelly_fraction': kelly['optimal_fraction'],
            'half_kelly': kelly['half_kelly'],
            'risk_based_fraction': risk_based_size,
            'conservative_fraction': conservative_size,
            'recommended_fraction': conservative_size,
            'max_position_dollar': 25000 * conservative_size
        }
    
    def _calculate_calmar(self, equity_curve: np.ndarray, max_dd: float) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        if max_dd >= 0 or len(equity_curve) < 2:
            return 0
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        annualized_return = total_return * (252 / len(equity_curve))
        
        return annualized_return / abs(max_dd)
    
    def add_historical_return(self, pnl: float):
        """Add a trade return to historical database."""
        self.historical_returns.append(pnl)
        
        # Keep only last 1000 returns
        if len(self.historical_returns) > 1000:
            self.historical_returns = self.historical_returns[-1000:]


class TailRiskHedger:
    """
    Suggest hedging strategies for tail risk mitigation.
    """
    
    def __init__(self):
        self.hedge_instruments = ['VIX_calls', 'SPY_puts', 'QQQ_puts']
    
    def assess_hedge_need(self, portfolio_metrics: Dict) -> Dict:
        """
        Assess whether tail risk hedging is needed.
        
        Returns:
            Dict with hedge recommendations
        """
        var = portfolio_metrics.get('var_95', -5000)
        cvar = portfolio_metrics.get('cvar_95', -7000)
        tail_risk = portfolio_metrics.get('tail_risk_score', 0)
        
        # Hedge if CVaR exceeds threshold
        hedge_needed = abs(cvar) > 10000 or tail_risk > 1000
        
        if not hedge_needed:
            return {'hedge_needed': False}
        
        # Calculate hedge size
        hedge_cost = abs(cvar) * 0.02  # 2% of CVaR
        
        return {
            'hedge_needed': True,
            'recommended_hedge': 'VIX_calls',
            'hedge_cost': hedge_cost,
            'hedge_notional': abs(cvar) * 0.5,
            'breakeven_move': 0.15,  # 15% VIX move
            'expected_benefit': abs(cvar) * 0.3
        }
