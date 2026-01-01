"""
ModelExplainer: Provides explainability for model predictions using SHAP and LIME.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Provides explainable insights for model predictions.
    
    Uses SHAP (SHapley Additive exPlanations) for global and local explanations.
    """
    
    def __init__(self, model, X_background: Optional[pd.DataFrame] = None):
        """
        Initialize ModelExplainer.
        
        Args:
            model: Trained model to explain
            X_background: Background data for SHAP explainer (optional)
        """
        self.model = model
        self.X_background = X_background
        self.explainer = None
        self.shap_values = None
        
        # Try to import SHAP
        try:
            import shap
            self.shap_available = True
            self._initialize_explainer()
        except ImportError:
            logger.warning("SHAP not available. Install with: pip install shap")
            self.shap_available = False
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type."""
        if not self.shap_available:
            return
        
        import shap
        
        try:
            # Try TreeExplainer first (for tree-based models)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Initialized SHAP TreeExplainer")
        except Exception:
            # Fall back to KernelExplainer
            if self.X_background is not None:
                # Sample background data if too large
                if len(self.X_background) > 100:
                    background = shap.sample(self.X_background, 100)
                else:
                    background = self.X_background
                
                self.explainer = shap.KernelExplainer(
                    self.model.predict, 
                    background
                )
                logger.info("Initialized SHAP KernelExplainer")
            else:
                logger.warning("No background data provided for KernelExplainer")
    
    def explain_prediction(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Explain a single prediction or batch of predictions.
        
        Args:
            X: Input features for explanation
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with explanation data
        """
        if not self.shap_available or self.explainer is None:
            return self._fallback_explanation(X, feature_names)
        
        import shap
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # For multi-class, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first class for now
            
            # Get feature names
            if feature_names is None:
                feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
            
            # Calculate feature contributions for first sample
            contributions = {}
            if len(X) > 0:
                first_sample_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                for fname, shap_val in zip(feature_names, first_sample_shap):
                    contributions[fname] = float(shap_val)
            
            # Sort by absolute contribution
            sorted_contributions = dict(sorted(
                contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            explanation = {
                'method': 'SHAP',
                'feature_contributions': sorted_contributions,
                'top_positive_features': self._get_top_features(sorted_contributions, positive=True, n=5),
                'top_negative_features': self._get_top_features(sorted_contributions, positive=False, n=5),
                'shap_values_shape': shap_values.shape
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            return self._fallback_explanation(X, feature_names)
    
    def _fallback_explanation(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Provide fallback explanation when SHAP is not available.
        
        Args:
            X: Input features
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with basic explanation
        """
        if feature_names is None:
            feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        
        # Use feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            sorted_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            return {
                'method': 'Feature Importance',
                'feature_importances': sorted_importance,
                'top_features': list(sorted_importance.keys())[:10],
                'note': 'SHAP not available, showing model feature importances'
            }
        
        return {
            'method': 'None',
            'note': 'No explanation method available'
        }
    
    def _get_top_features(self, contributions: Dict[str, float], positive: bool = True, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top contributing features.
        
        Args:
            contributions: Dictionary of feature contributions
            positive: If True, get positive contributions; else negative
            n: Number of features to return
            
        Returns:
            List of top features with their contributions
        """
        filtered = {k: v for k, v in contributions.items() if (v > 0) == positive}
        sorted_features = sorted(filtered.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
        
        return [{'feature': feat, 'contribution': contrib} for feat, contrib in sorted_features]
    
    def get_global_feature_importance(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get global feature importance across all samples.
        
        Args:
            X: Input features
            feature_names: Optional list of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not self.shap_available or self.explainer is None:
            # Fall back to model's feature importances
            if hasattr(self.model, 'feature_importances_'):
                if feature_names is None:
                    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.model.feature_importances_
                })
                return importance_df.sort_values('importance', ascending=False)
            
            return pd.DataFrame()
        
        import shap
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # For multi-class, use first class
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            if feature_names is None:
                feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap
            })
            
            return importance_df.sort_values('importance', ascending=False)
            
        except Exception as e:
            logger.error(f"Error calculating global importance: {str(e)}")
            return pd.DataFrame()
    
    def generate_explanation_text(self, X: pd.DataFrame, prediction: Any, feature_names: Optional[List[str]] = None) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            X: Input features
            prediction: Model prediction
            feature_names: Optional list of feature names
            
        Returns:
            Explanation text
        """
        explanation = self.explain_prediction(X, feature_names)
        
        text_parts = [f"Prediction: {prediction}\n"]
        
        if explanation['method'] == 'SHAP':
            text_parts.append("\nTop factors increasing risk/delay:")
            for item in explanation.get('top_positive_features', [])[:3]:
                text_parts.append(f"  - {item['feature']}: +{item['contribution']:.4f}")
            
            text_parts.append("\nTop factors decreasing risk/delay:")
            for item in explanation.get('top_negative_features', [])[:3]:
                text_parts.append(f"  - {item['feature']}: {item['contribution']:.4f}")
        
        elif explanation['method'] == 'Feature Importance':
            text_parts.append("\nMost important features:")
            for feat in explanation.get('top_features', [])[:5]:
                imp = explanation['feature_importances'].get(feat, 0)
                text_parts.append(f"  - {feat}: {imp:.4f}")
        
        return "\n".join(text_parts)
