"""
RiskPredictor: ML model for predicting project risk levels.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import joblib
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb

logger = logging.getLogger(__name__)


class RiskPredictor:
    """
    Predicts project risk level (Low, Medium, High) based on project features.
    
    Supports multiple model types:
    - Random Forest
    - Gradient Boosting
    - XGBoost
    """
    
    def __init__(self, model_type: str = 'xgboost', model_params: Optional[Dict] = None):
        """
        Initialize RiskPredictor.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'xgboost')
            model_params: Optional parameters for the model
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model: Optional[Any] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
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
            self.model = RandomForestClassifier(**params)
            
        elif self.model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            self.model = GradientBoostingClassifier(**params)
            
        elif self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'objective': 'multi:softmax',
                'random_state': 42,
                'n_jobs': -1
            }
            params = {**default_params, **self.model_params}
            self.model = xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train the risk prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels (risk levels)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples...")
        
        # Encode string labels if necessary
        if y_train.dtype == 'object':
            label_map = {'Low': 0, 'Medium': 1, 'High': 2}
            y_train = y_train.map(label_map)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Store classes
        self.classes_ = self.model.classes_
        
        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        train_f1 = f1_score(y_train, y_pred, average='weighted')
        
        logger.info(f"Training completed. Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        
        return {
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        # Encode string labels if necessary
        if y_test.dtype == 'object':
            label_map = {'Low': 0, 'Medium': 1, 'High': 2}
            y_test = y_test.map(label_map)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Generate classification report
        target_names = ['Low', 'Medium', 'High']
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'n_test_samples': len(X_test)
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk levels for new projects.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted risk levels
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        
        # Convert numeric predictions back to labels if needed
        if hasattr(self, 'classes_'):
            label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
            predictions = np.array([label_map.get(p, p) for p in predictions])
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of each risk level.
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
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
            'feature_importances': self.feature_importances_,
            'classes': self.classes_
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'RiskPredictor':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded RiskPredictor instance
        """
        model_data = joblib.load(filepath)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.feature_importances_ = model_data.get('feature_importances')
        predictor.classes_ = model_data.get('classes')
        predictor.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return predictor
