"""
Main Framework: AI-Driven Project Risk Prediction and Optimization

This module orchestrates the entire pipeline from data loading to prediction and explanation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import json

from .data.data_loader import DataLoader
from .data.preprocessor import DataPreprocessor
from .data.lessons_learned_parser import LessonsLearnedParser
from .models.risk_predictor import RiskPredictor
from .models.delay_predictor import DelayPredictor
from .explainability.explainer import ModelExplainer
from .utils.visualization import Visualizer
from .utils.logger import setup_logging

logger = logging.getLogger(__name__)


class RiskPredictionFramework:
    """
    Main framework for AI-driven project risk prediction and optimization.
    
    Integrates:
    - Historical project data loading
    - Lessons Learned parsing
    - Feature engineering
    - Risk prediction (classification)
    - Delay prediction (regression)
    - Model explainability (SHAP)
    - Visualization
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data/raw"),
        models_dir: Path = Path("models/saved"),
        output_dir: Path = Path("outputs"),
        log_level: str = "INFO"
    ):
        """
        Initialize the framework.
        
        Args:
            data_dir: Directory containing raw data
            models_dir: Directory to save/load models
            output_dir: Directory for outputs and visualizations
            log_level: Logging level
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(log_level=log_level, log_file=self.output_dir / "framework.log")
        
        # Initialize components
        self.data_loader = DataLoader(data_dir=self.data_dir)
        self.preprocessor = DataPreprocessor()
        self.ll_parser = LessonsLearnedParser()
        self.visualizer = Visualizer(output_dir=self.output_dir / "plots")
        
        # Models
        self.risk_predictor: Optional[RiskPredictor] = None
        self.delay_predictor: Optional[DelayPredictor] = None
        
        # Explainers
        self.risk_explainer: Optional[ModelExplainer] = None
        self.delay_explainer: Optional[ModelExplainer] = None
        
        # Data
        self.project_data: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_risk_train: Optional[pd.Series] = None
        self.y_risk_test: Optional[pd.Series] = None
        self.y_delay_train: Optional[pd.Series] = None
        self.y_delay_test: Optional[pd.Series] = None
        
        logger.info("Framework initialized")
    
    def load_data(self, filepath: Optional[Path] = None, create_sample: bool = False) -> pd.DataFrame:
        """
        Load project data.
        
        Args:
            filepath: Path to data file (CSV or Excel)
            create_sample: If True, create sample data
            
        Returns:
            Loaded project data
        """
        if create_sample:
            logger.info("Creating sample project data...")
            sample_path = self.data_dir / "sample_projects.csv"
            self.project_data = self.data_loader.create_sample_data(
                n_projects=200,
                save_path=sample_path
            )
        elif filepath:
            logger.info(f"Loading project data from {filepath}...")
            self.project_data = self.data_loader.load_project_data(filepath)
        else:
            raise ValueError("Either provide filepath or set create_sample=True")
        
        logger.info(f"Loaded {len(self.project_data)} projects")
        return self.project_data
    
    def prepare_data(self, test_size: float = 0.2):
        """
        Prepare data for training both risk and delay models.
        
        Args:
            test_size: Proportion of data for testing
        """
        if self.project_data is None:
            raise ValueError("Data must be loaded first")
        
        logger.info("Preparing data for risk prediction...")
        self.X_train, self.X_test, self.y_risk_train, self.y_risk_test = \
            self.preprocessor.prepare_for_training(
                self.project_data,
                target_col='risk_level',
                test_size=test_size
            )
        
        # For delay prediction, we need to prepare separately
        # Use the same preprocessor but different target
        delay_preprocessor = DataPreprocessor()
        logger.info("Preparing data for delay prediction...")
        X_train_delay, X_test_delay, self.y_delay_train, self.y_delay_test = \
            delay_preprocessor.prepare_for_training(
                self.project_data,
                target_col='delay_days',
                test_size=test_size
            )
        
        # Use same train/test split for consistency
        assert len(X_train_delay) == len(self.X_train), "Train set sizes must match"
    
    def train_risk_model(self, model_type: str = 'xgboost') -> Dict[str, Any]:
        """
        Train risk prediction model.
        
        Args:
            model_type: Type of model to train
            
        Returns:
            Training metrics
        """
        if self.X_train is None or self.y_risk_train is None:
            raise ValueError("Data must be prepared first")
        
        logger.info(f"Training risk prediction model ({model_type})...")
        self.risk_predictor = RiskPredictor(model_type=model_type)
        train_metrics = self.risk_predictor.train(self.X_train, self.y_risk_train)
        
        # Initialize explainer
        self.risk_explainer = ModelExplainer(
            self.risk_predictor.model,
            X_background=self.X_train.sample(min(100, len(self.X_train)))
        )
        
        logger.info("Risk prediction model trained successfully")
        return train_metrics
    
    def train_delay_model(self, model_type: str = 'xgboost') -> Dict[str, Any]:
        """
        Train delay prediction model.
        
        Args:
            model_type: Type of model to train
            
        Returns:
            Training metrics
        """
        if self.X_train is None or self.y_delay_train is None:
            raise ValueError("Data must be prepared first")
        
        logger.info(f"Training delay prediction model ({model_type})...")
        self.delay_predictor = DelayPredictor(model_type=model_type)
        train_metrics = self.delay_predictor.train(self.X_train, self.y_delay_train)
        
        # Initialize explainer
        self.delay_explainer = ModelExplainer(
            self.delay_predictor.model,
            X_background=self.X_train.sample(min(100, len(self.X_train)))
        )
        
        logger.info("Delay prediction model trained successfully")
        return train_metrics
    
    def evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate both models on test data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        if self.risk_predictor and self.X_test is not None:
            logger.info("Evaluating risk prediction model...")
            results['risk_model'] = self.risk_predictor.evaluate(self.X_test, self.y_risk_test)
            
            # Visualize confusion matrix
            cm = np.array(results['risk_model']['confusion_matrix'])
            self.visualizer.plot_confusion_matrix(
                cm,
                class_names=['Low', 'Medium', 'High'],
                title='Risk Prediction Confusion Matrix',
                save_name='risk_confusion_matrix.png'
            )
            
            # Feature importance
            importance_df = self.risk_predictor.get_feature_importance(
                feature_names=self.preprocessor.feature_names
            )
            self.visualizer.plot_feature_importance(
                importance_df,
                title='Risk Model Feature Importance',
                save_name='risk_feature_importance.png'
            )
        
        if self.delay_predictor and self.X_test is not None:
            logger.info("Evaluating delay prediction model...")
            results['delay_model'] = self.delay_predictor.evaluate(self.X_test, self.y_delay_test)
            
            # Prediction vs actual plot
            y_pred = self.delay_predictor.predict(self.X_test)
            self.visualizer.plot_prediction_vs_actual(
                self.y_delay_test.values,
                y_pred,
                title='Delay Prediction: Actual vs Predicted',
                save_name='delay_prediction_vs_actual.png'
            )
            
            # Feature importance
            importance_df = self.delay_predictor.get_feature_importance(
                feature_names=self.preprocessor.feature_names
            )
            self.visualizer.plot_feature_importance(
                importance_df,
                title='Delay Model Feature Importance',
                save_name='delay_feature_importance.png'
            )
        
        # Save results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._make_json_serializable(results)
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        return results
    
    def predict(self, project_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions for new projects.
        
        Args:
            project_data: DataFrame with project features
            
        Returns:
            Dictionary with predictions and explanations
        """
        if self.risk_predictor is None or self.delay_predictor is None:
            raise ValueError("Models must be trained first")
        
        # Preprocess data
        X = self.preprocessor.transform(project_data)
        
        # Make predictions
        risk_predictions = self.risk_predictor.predict(X)
        risk_probabilities = self.risk_predictor.predict_proba(X)
        delay_predictions = self.delay_predictor.predict(X)
        
        # Generate explanations
        risk_explanation = None
        delay_explanation = None
        
        if self.risk_explainer and len(X) > 0:
            risk_explanation = self.risk_explainer.explain_prediction(
                X.iloc[:1],
                feature_names=self.preprocessor.feature_names
            )
        
        if self.delay_explainer and len(X) > 0:
            delay_explanation = self.delay_explainer.explain_prediction(
                X.iloc[:1],
                feature_names=self.preprocessor.feature_names
            )
        
        results = {
            'n_projects': len(project_data),
            'risk_predictions': risk_predictions.tolist(),
            'risk_probabilities': risk_probabilities.tolist(),
            'delay_predictions': delay_predictions.tolist(),
            'risk_explanation': risk_explanation,
            'delay_explanation': delay_explanation
        }
        
        return results
    
    def save_models(self):
        """Save trained models to disk."""
        if self.risk_predictor:
            risk_path = self.models_dir / "risk_predictor.pkl"
            self.risk_predictor.save_model(risk_path)
            logger.info(f"Risk model saved to {risk_path}")
        
        if self.delay_predictor:
            delay_path = self.models_dir / "delay_predictor.pkl"
            self.delay_predictor.save_model(delay_path)
            logger.info(f"Delay model saved to {delay_path}")
    
    def load_models(self):
        """Load trained models from disk."""
        risk_path = self.models_dir / "risk_predictor.pkl"
        if risk_path.exists():
            self.risk_predictor = RiskPredictor.load_model(risk_path)
            logger.info(f"Risk model loaded from {risk_path}")
        
        delay_path = self.models_dir / "delay_predictor.pkl"
        if delay_path.exists():
            self.delay_predictor = DelayPredictor.load_model(delay_path)
            logger.info(f"Delay model loaded from {delay_path}")
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report of the framework performance.
        
        Returns:
            Report text
        """
        report_lines = [
            "=" * 80,
            "AI-DRIVEN PROJECT RISK PREDICTION FRAMEWORK - REPORT",
            "=" * 80,
            ""
        ]
        
        if self.project_data is not None:
            report_lines.extend([
                f"Dataset: {len(self.project_data)} projects",
                f"Features: {len(self.preprocessor.feature_names) if self.preprocessor.feature_names else 'N/A'}",
                ""
            ])
        
        if self.risk_predictor:
            report_lines.extend([
                "RISK PREDICTION MODEL",
                "-" * 40,
                f"Model Type: {self.risk_predictor.model_type}",
                ""
            ])
        
        if self.delay_predictor:
            report_lines.extend([
                "DELAY PREDICTION MODEL",
                "-" * 40,
                f"Model Type: {self.delay_predictor.model_type}",
                ""
            ])
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / "framework_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_path}")
        
        return report_text
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
