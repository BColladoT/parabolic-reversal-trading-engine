"""
Model Validation and Backtesting Framework

Rigorous statistical validation of ML risk models:
- Walk-forward cross-validation
- Out-of-sample testing
- Statistical significance tests
- Overfitting detection
- Monte Carlo validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, precision_recall_curve, 
                            classification_report, confusion_matrix,
                            log_loss, brier_score_loss)
from scipy import stats
import warnings


class ModelValidator:
    """
    Comprehensive model validation framework.
    
    Implements best practices from quantitative finance for
    ensuring model robustness and avoiding overfitting.
    """
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.validation_results: List[Dict] = []
        
    def validate(self, risk_manager, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Run comprehensive validation suite.
        
        Args:
            risk_manager: Risk manager instance to validate
            X: Feature matrix
            y: Target labels (1=win, 0=loss)
            
        Returns:
            Dict with validation results
        """
        print("[VALIDATOR] Starting comprehensive validation...")
        
        results = {}
        
        # 1. Walk-forward cross-validation
        print("[VALIDATOR] Running walk-forward cross-validation...")
        results['walk_forward'] = self._walk_forward_cv(risk_manager, X, y)
        
        # 2. Statistical significance tests
        print("[VALIDATOR] Running statistical tests...")
        results['statistical'] = self._statistical_tests(X, y, results['walk_forward'])
        
        # 3. Overfitting detection
        print("[VALIDATOR] Checking for overfitting...")
        results['overfitting'] = self._detect_overfitting(results['walk_forward'])
        
        # 4. Calibration assessment
        print("[VALIDATOR] Assessing probability calibration...")
        results['calibration'] = self._assess_calibration(risk_manager, X, y)
        
        # 5. Feature importance stability
        print("[VALIDATOR] Checking feature stability...")
        results['feature_stability'] = self._check_feature_stability(risk_manager, X, y)
        
        # Summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _walk_forward_cv(self, risk_manager, X: np.ndarray, 
                        y: np.ndarray) -> Dict:
        """
        Walk-forward cross-validation (time series aware).
        
        Simulates real-world deployment where model is trained on past
        data and tested on future data.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"  [VALIDATOR] Fold {fold+1}/{self.n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train on this fold
            risk_manager.train(X_train, y_train, validation_split=0.1)
            
            # Predict on test set
            y_pred_proba = []
            y_pred_binary = []
            
            for x in X_test:
                pred = risk_manager.assess_trade({'bars': x.reshape(1, -1)})
                prob = pred['win_probability']
                y_pred_proba.append(prob)
                y_pred_binary.append(1 if prob > 0.5 else 0)
            
            y_pred_proba = np.array(y_pred_proba)
            y_pred_binary = np.array(y_pred_binary)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Classification metrics
            cm = confusion_matrix(y_test, y_pred_binary)
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            logloss = log_loss(y_test, np.clip(y_pred_proba, 0.001, 0.999))
            brier = brier_score_loss(y_test, y_pred_proba)
            
            fold_results.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'log_loss': logloss,
                'brier_score': brier,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            })
            
            all_predictions.extend(y_pred_proba)
            all_actuals.extend(y_test)
        
        # Aggregate results
        aucs = [f['auc'] for f in fold_results]
        f1s = [f['f1'] for f in fold_results]
        
        return {
            'fold_results': fold_results,
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'mean_f1': np.mean(f1s),
            'std_f1': np.std(f1s),
            'auc_ci': (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)),
            'all_predictions': all_predictions,
            'all_actuals': all_actuals
        }
    
    def _statistical_tests(self, X: np.ndarray, y: np.ndarray, 
                          cv_results: Dict) -> Dict:
        """Run statistical significance tests."""
        
        # Test 1: Is AUC significantly better than random (0.5)?
        aucs = [f['auc'] for f in cv_results['fold_results']]
        t_stat, p_value = stats.ttest_1samp(aucs, 0.5)
        
        auc_significant = p_value < 0.05 and np.mean(aucs) > 0.5
        
        # Test 2: Information coefficient (correlation with returns)
        # Would need actual returns for this
        
        # Test 3: Prediction consistency
        predictions = cv_results['all_predictions']
        actuals = cv_results['all_actuals']
        
        # Mann-Whitney U test (do winners have higher predictions?)
        winner_preds = [p for p, a in zip(predictions, actuals) if a == 1]
        loser_preds = [p for p, a in zip(predictions, actuals) if a == 0]
        
        if winner_preds and loser_preds:
            u_stat, mann_p = stats.mannwhitneyu(winner_preds, loser_preds, 
                                                alternative='greater')
        else:
            mann_p = 1.0
        
        return {
            'auc_vs_random_tstat': t_stat,
            'auc_vs_random_pvalue': p_value,
            'auc_significant': auc_significant,
            'mann_whitney_pvalue': mann_p,
            'predictions_discriminate': mann_p < 0.05
        }
    
    def _detect_overfitting(self, cv_results: Dict) -> Dict:
        """Detect signs of overfitting."""
        
        # Check for high variance in fold performance
        aucs = [f['auc'] for f in cv_results['fold_results']]
        f1s = [f['f1'] for f in cv_results['fold_results']]
        
        auc_cv = np.std(aucs) / np.mean(aucs) if np.mean(aucs) > 0 else float('inf')
        f1_cv = np.std(f1s) / np.mean(f1s) if np.mean(f1s) > 0 else float('inf')
        
        # High coefficient of variation suggests instability
        unstable = auc_cv > 0.2 or f1_cv > 0.3
        
        # Check for declining performance in later folds
        # (might indicate look-ahead bias or non-stationarity)
        early_auc = np.mean(aucs[:len(aucs)//2])
        late_auc = np.mean(aucs[len(aucs)//2:])
        degradation = (early_auc - late_auc) / early_auc if early_auc > 0 else 0
        
        overfitting_risk = unstable or degradation > 0.1
        
        return {
            'auc_coefficient_of_variation': auc_cv,
            'f1_coefficient_of_variation': f1_cv,
            'performance_unstable': unstable,
            'performance_degradation': degradation,
            'degradation_significant': degradation > 0.1,
            'overfitting_risk': overfitting_risk,
            'risk_level': 'HIGH' if overfitting_risk else 'LOW'
        }
    
    def _assess_calibration(self, risk_manager, X: np.ndarray, 
                           y: np.ndarray) -> Dict:
        """
        Assess probability calibration.
        
        A well-calibrated model should have:
        - Of predictions with probability 0.8, 80% should be wins
        """
        
        # Get predictions
        predictions = []
        for x in X:
            pred = risk_manager.assess_trade({'bars': x.reshape(1, -1)})
            predictions.append(pred['win_probability'])
        
        predictions = np.array(predictions)
        
        # Bin predictions and calculate observed win rates
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        calibration_data = []
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
            if i == n_bins - 1:  # Include right edge for last bin
                mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i+1])
            
            if np.sum(mask) > 0:
                mean_pred = np.mean(predictions[mask])
                mean_actual = np.mean(y[mask])
                n_samples = np.sum(mask)
                
                calibration_data.append({
                    'bin': i,
                    'mean_predicted': mean_pred,
                    'mean_actual': mean_actual,
                    'n_samples': n_samples,
                    'calibration_error': abs(mean_pred - mean_actual)
                })
        
        # Calculate expected calibration error
        if calibration_data:
            ece = np.mean([d['calibration_error'] for d in calibration_data])
            mce = np.max([d['calibration_error'] for d in calibration_data])
        else:
            ece = mce = 1.0
        
        # Brier score (already calculated in CV)
        brier = brier_score_loss(y, predictions)
        
        return {
            'expected_calibration_error': ece,
            'max_calibration_error': mce,
            'brier_score': brier,
            'well_calibrated': ece < 0.1,
            'calibration_data': calibration_data
        }
    
    def _check_feature_stability(self, risk_manager, X: np.ndarray, 
                                 y: np.ndarray) -> Dict:
        """Check if feature importance is stable across folds."""
        
        # Would need access to feature names from risk_manager
        # For now, return placeholder
        
        return {
            'message': 'Feature stability check requires feature names',
            'recommendation': 'Run permutation importance across folds'
        }
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate validation summary."""
        
        walk_forward = results['walk_forward']
        statistical = results['statistical']
        overfitting = results['overfitting']
        calibration = results['calibration']
        
        # Overall score (0-100)
        score = 0
        score += min(walk_forward['mean_auc'] * 50, 50)  # Up to 50 points for AUC
        score += 20 if statistical['auc_significant'] else 0
        score += 15 if calibration['well_calibrated'] else 0
        score += 15 if not overfitting['overfitting_risk'] else 0
        
        # Determine if model is production-ready
        production_ready = (
            walk_forward['mean_auc'] > 0.65 and
            statistical['auc_significant'] and
            not overfitting['overfitting_risk'] and
            calibration['well_calibrated']
        )
        
        return {
            'overall_score': score,
            'production_ready': production_ready,
            'recommendation': 'PROCEED' if production_ready else 'IMPROVE_MODEL',
            'key_strengths': self._get_strengths(results),
            'key_concerns': self._get_concerns(results),
            'mean_auc': walk_forward['mean_auc'],
            'auc_std': walk_forward['std_auc'],
            'auc_ci_95': walk_forward['auc_ci']
        }
    
    def _get_strengths(self, results: Dict) -> List[str]:
        """Identify model strengths."""
        strengths = []
        
        if results['statistical'].get('auc_significant', False):
            strengths.append('AUC significantly better than random')
        
        if results['calibration'].get('well_calibrated', False):
            strengths.append('Well-calibrated probabilities')
        
        if results['walk_forward']['mean_auc'] > 0.7:
            strengths.append('Strong discriminative power (AUC > 0.7)')
        
        return strengths
    
    def _get_concerns(self, results: Dict) -> List[str]:
        """Identify model concerns."""
        concerns = []
        
        if results['overfitting'].get('overfitting_risk', False):
            concerns.append('Overfitting risk detected')
        
        if not results['calibration'].get('well_calibrated', False):
            concerns.append('Poor probability calibration')
        
        if results['walk_forward']['std_auc'] > 0.1:
            concerns.append('Unstable performance across folds')
        
        return concerns


class MonteCarloValidator:
    """
    Monte Carlo validation for assessing strategy robustness.
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
    
    def validate_strategy(self, trades: List[Dict]) -> Dict:
        """
        Run Monte Carlo simulation on trade history.
        
        Shuffles trade order to assess path dependency.
        """
        pnls = [t['pnl'] for t in trades]
        
        final_equities = []
        max_drawdowns = []
        
        for _ in range(self.n_simulations):
            # Shuffle trades
            shuffled = np.random.permutation(pnls)
            
            # Calculate equity curve
            equity = np.cumsum(shuffled) + 100000
            
            # Record final equity
            final_equities.append(equity[-1])
            
            # Calculate max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max
            max_drawdowns.append(np.min(drawdown))
        
        return {
            'mean_final_equity': np.mean(final_equities),
            'std_final_equity': np.std(final_equities),
            'worst_final_equity': np.min(final_equities),
            'best_final_equity': np.max(final_equities),
            'prob_profit': np.mean(np.array(final_equities) > 100000),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),
            'var_95_equity': np.percentile(final_equities, 5),
            'cvar_95_equity': np.mean(np.array(final_equities)[
                np.array(final_equities) <= np.percentile(final_equities, 5)
            ])
        }


class BenchmarkComparator:
    """
    Compare model against benchmarks.
    """
    
    def __init__(self):
        self.benchmarks = {
            'random': lambda y: np.random.random(len(y)),
            'always_win': lambda y: np.ones(len(y)) * 0.7,
            'base_rate': lambda y: np.ones(len(y)) * np.mean(y)
        }
    
    def compare(self, model_predictions: np.ndarray, y_true: np.ndarray) -> Dict:
        """Compare model against benchmarks."""
        
        model_auc = roc_auc_score(y_true, model_predictions)
        
        comparisons = {}
        for name, predictor in self.benchmarks.items():
            bench_pred = predictor(y_true)
            bench_auc = roc_auc_score(y_true, bench_pred)
            
            comparisons[name] = {
                'auc': bench_auc,
                'model_beat_benchmark': model_auc > bench_auc,
                'difference': model_auc - bench_auc
            }
        
        return {
            'model_auc': model_auc,
            'benchmarks': comparisons,
            'beats_all_benchmarks': all(c['model_beat_benchmark'] 
                                       for c in comparisons.values())
        }
