"""
DelayPredictor: ML model for predicting project delays.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import joblib
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

logger = logging.getLogger(__name__)


class DelayPredictor:
    """
    Predicts project delay in days based on project features.
    
    Supports multiple model types:
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - XGBoost Regressor
    """
    
    def __init__(self, model_type: str = 'xgboost', model_params: Optional[Dict] = None):
        """
        Initialize DelayPredictor.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'xgboost')
            model_params: Optional parameters for the model
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model: Optional[Any] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.is_fitted = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on model_type."""
        if self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1
            }
            params = {**default_params, **self.model_params}
            self.model = RandomForestRegressor(**params)
            
        elif self.model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            self.model = GradientBoostingRegressor(**params)
            
        elif self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1
            }
            params = {**default_params, **self.model_params}
            self.model = xgb.XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model for delay prediction")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train the delay prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels (delay in days)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples...")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        train_mae = mean_absolute_error(y_train, y_pred)
        train_r2 = r2_score(y_train, y_pred)
        
        logger.info(f"Training completed. RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R2: {train_r2:.4f}")
        
        return {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels (delay in days)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100  # Avoid division by zero
        
        logger.info(f"Evaluation completed. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'n_test_samples': len(X_test),
            'predictions': y_pred.tolist()[:10]  # Sample predictions
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict delay days for new projects.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted delay in days
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(0, predictions)
        
        return predictions
    
    def predict_delay_probability(self, X: pd.DataFrame, threshold_days: float = 0) -> np.ndarray:
        """
        Predict probability of delay beyond a threshold.
        
        Args:
            X: Features for prediction
            threshold_days: Threshold for delay (default: 0, any delay)
            
        Returns:
            Array of probabilities (1 if delay predicted > threshold, 0 otherwise)
        """
        predictions = self.predict(X)
        return (predictions > threshold_days).astype(int)
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with features and their importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available for this model")
        
        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else range(len(self.feature_importances_)),
            'importance': self.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: Path):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_importances': self.feature_importances_
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'DelayPredictor':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded DelayPredictor instance
        """
        model_data = joblib.load(filepath)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.feature_importances_ = model_data.get('feature_importances')
        predictor.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return predictor
