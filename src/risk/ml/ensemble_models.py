"""
Ensemble Machine Learning Models for Risk Prediction

Implements a multi-model ensemble approach combining:
- XGBoost (gradient boosting)
- Random Forest (bagging)
- Neural Network (deep learning)
- Logistic Regression (baseline)

Ensemble voting with weighted averaging based on model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fallback to sklearn if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import PyTorch for neural network
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class NeuralNetworkRiskModel(nn.Module):
    """Deep neural network for risk prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class EnsembleRiskModel:
    """
    Ensemble of multiple ML models for robust risk prediction.
    
    Uses weighted voting based on out-of-sample performance.
    """
    
    def __init__(self, use_xgboost: bool = True, use_neural_net: bool = True):
        self.models: Dict[str, any] = {}
        self.model_weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        self.use_neural_net = use_neural_net and PYTORCH_AVAILABLE
        
        # Model configurations
        self.rf_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced_subsample',
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.gb_params = {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
        
        if self.use_xgboost:
            self.xgb_params = {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 2,  # Handle class imbalance
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
        
        self.nn_params = {
            'hidden_dims': [128, 64, 32],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict:
        """Train all models in the ensemble."""
        print(f"[ENSEMBLE] Training with {len(X)} samples...")
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        results = {}
        
        # 1. Random Forest
        print("[ENSEMBLE] Training Random Forest...")
        self.models['rf'] = RandomForestClassifier(**self.rf_params)
        self.models['rf'].fit(X_train, y_train)
        rf_pred = self.models['rf'].predict_proba(X_val)[:, 1]
        rf_auc = roc_auc_score(y_val, rf_pred)
        self.model_weights['rf'] = rf_auc
        results['rf_auc'] = rf_auc
        print(f"  RF AUC: {rf_auc:.4f}")
        
        # 2. Gradient Boosting (sklearn)
        print("[ENSEMBLE] Training Gradient Boosting...")
        self.models['gb'] = GradientBoostingClassifier(**self.gb_params)
        self.models['gb'].fit(X_train, y_train)
        gb_pred = self.models['gb'].predict_proba(X_val)[:, 1]
        gb_auc = roc_auc_score(y_val, gb_pred)
        self.model_weights['gb'] = gb_auc
        results['gb_auc'] = gb_auc
        print(f"  GB AUC: {gb_auc:.4f}")
        
        # 3. XGBoost
        if self.use_xgboost:
            print("[ENSEMBLE] Training XGBoost...")
            self.models['xgb'] = xgb.XGBClassifier(**self.xgb_params)
            self.models['xgb'].fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            xgb_pred = self.models['xgb'].predict_proba(X_val)[:, 1]
            xgb_auc = roc_auc_score(y_val, xgb_pred)
            self.model_weights['xgb'] = xgb_auc
            results['xgb_auc'] = xgb_auc
            print(f"  XGB AUC: {xgb_auc:.4f}")
        
        # 4. Neural Network
        if self.use_neural_net and len(X_train) > 100:
            print("[ENSEMBLE] Training Neural Network...")
            nn_auc = self._train_neural_network(X_train, y_train, X_val, y_val)
            self.model_weights['nn'] = nn_auc
            results['nn_auc'] = nn_auc
            print(f"  NN AUC: {nn_auc:.4f}")
        
        # 5. Logistic Regression (baseline)
        print("[ENSEMBLE] Training Logistic Regression...")
        self.models['lr'] = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.models['lr'].fit(X_train, y_train)
        lr_pred = self.models['lr'].predict_proba(X_val)[:, 1]
        lr_auc = roc_auc_score(y_val, lr_pred)
        self.model_weights['lr'] = lr_auc
        results['lr_auc'] = lr_auc
        print(f"  LR AUC: {lr_auc:.4f}")
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
        
        print(f"[ENSEMBLE] Model weights: {self.model_weights}")
        
        # Calculate ensemble performance
        ensemble_pred = self._ensemble_predict(X_val)
        ensemble_auc = roc_auc_score(y_val, ensemble_pred)
        results['ensemble_auc'] = ensemble_auc
        print(f"[ENSEMBLE] Ensemble AUC: {ensemble_auc:.4f}")
        
        return results
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val) -> float:
        """Train neural network model."""
        if not PYTORCH_AVAILABLE:
            return 0.5
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.nn_params['batch_size'], shuffle=True)
        
        # Initialize model
        self.models['nn'] = NeuralNetworkRiskModel(
            input_dim=X_train.shape[1],
            hidden_dims=self.nn_params['hidden_dims']
        )
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.models['nn'].parameters(),
            lr=self.nn_params['learning_rate']
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.nn_params['epochs']):
            self.models['nn'].train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.models['nn'](batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.models['nn'].eval()
            with torch.no_grad():
                val_outputs = self.models['nn'](X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.nn_best_state = self.models['nn'].state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.nn_params['early_stopping_patience']:
                    break
        
        # Load best model
        if hasattr(self, 'nn_best_state'):
            self.models['nn'].load_state_dict(self.nn_best_state)
        
        # Calculate AUC
        self.models['nn'].eval()
        with torch.no_grad():
            val_pred = self.models['nn'](X_val_t).numpy().flatten()
        
        return roc_auc_score(y_val, val_pred)
    
    def predict(self, X: np.ndarray) -> Dict:
        """
        Generate ensemble prediction.
        
        Returns:
            Dict with:
            - probability: Weighted ensemble probability
            - individual_predictions: Dict of individual model predictions
            - uncertainty: Prediction uncertainty (std of individual predictions)
            - confidence: Confidence in prediction (inverse of uncertainty)
        """
        individual_preds = {}
        
        # Random Forest
        if 'rf' in self.models:
            individual_preds['rf'] = self.models['rf'].predict_proba(X)[:, 1]
        
        # Gradient Boosting
        if 'gb' in self.models:
            individual_preds['gb'] = self.models['gb'].predict_proba(X)[:, 1]
        
        # XGBoost
        if 'xgb' in self.models:
            individual_preds['xgb'] = self.models['xgb'].predict_proba(X)[:, 1]
        
        # Neural Network
        if 'nn' in self.models and PYTORCH_AVAILABLE:
            self.models['nn'].eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X)
                individual_preds['nn'] = self.models['nn'](X_t).numpy().flatten()
        
        # Logistic Regression
        if 'lr' in self.models:
            individual_preds['lr'] = self.models['lr'].predict_proba(X)[:, 1]
        
        # Weighted ensemble
        ensemble_prob = 0
        for model_name, pred in individual_preds.items():
            weight = self.model_weights.get(model_name, 0.2)
            ensemble_prob += weight * pred
        
        # Uncertainty (disagreement between models)
        pred_matrix = np.array(list(individual_preds.values()))
        uncertainty = np.std(pred_matrix, axis=0)
        confidence = 1 - uncertainty
        
        return {
            'probability': ensemble_prob,
            'individual_predictions': individual_preds,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'model_votes': {k: (v > 0.5).astype(int) for k, v in individual_preds.items()}
        }
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Simple ensemble prediction for internal use."""
        result = self.predict(X)
        return result['probability']
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get aggregated feature importance across models."""
        importance_dict = {}
        
        # Random Forest importance
        if 'rf' in self.models:
            for name, imp in zip(feature_names, self.models['rf'].feature_importances_):
                importance_dict[name] = importance_dict.get(name, 0) + imp * self.model_weights.get('rf', 0)
        
        # Gradient Boosting importance
        if 'gb' in self.models:
            for name, imp in zip(feature_names, self.models['gb'].feature_importances_):
                importance_dict[name] = importance_dict.get(name, 0) + imp * self.model_weights.get('gb', 0)
        
        # XGBoost importance
        if 'xgb' in self.models:
            for name, imp in zip(feature_names, self.models['xgb'].feature_importances_):
                importance_dict[name] = importance_dict.get(name, 0) + imp * self.model_weights.get('xgb', 0)
        
        # Logistic Regression coefficients (absolute)
        if 'lr' in self.models:
            for name, coef in zip(feature_names, np.abs(self.models['lr'].coef_[0])):
                importance_dict[name] = importance_dict.get(name, 0) + coef * self.model_weights.get('lr', 0)
        
        return pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict:
        """Perform cross-validation on ensemble."""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train on this fold
            self.train(X_train, y_train, validation_split=0)
            
            # Predict
            y_pred = self._ensemble_predict(X_val)
            auc = roc_auc_score(y_val, y_pred)
            
            cv_scores.append(auc)
            fold_results.append({
                'fold': fold + 1,
                'auc': auc,
                'n_train': len(train_idx),
                'n_val': len(val_idx)
            })
        
        return {
            'mean_auc': np.mean(cv_scores),
            'std_auc': np.std(cv_scores),
            'min_auc': np.min(cv_scores),
            'max_auc': np.max(cv_scores),
            'fold_results': fold_results
        }
