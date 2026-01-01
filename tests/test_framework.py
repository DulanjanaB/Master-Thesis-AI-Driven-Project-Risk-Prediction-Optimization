"""
Unit tests for the AI-Driven Project Risk Prediction Framework.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.lessons_learned_parser import LessonsLearnedParser
from src.models.risk_predictor import RiskPredictor
from src.models.delay_predictor import DelayPredictor


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        loader = DataLoader()
        df = loader.create_sample_data(n_projects=50)
        
        assert len(df) == 50
        assert 'project_id' in df.columns
        assert 'risk_level' in df.columns
        assert 'delayed' in df.columns
        assert 'delay_days' in df.columns
    
    def test_data_summary(self):
        """Test data summary generation."""
        loader = DataLoader()
        loader.create_sample_data(n_projects=30)
        
        summary = loader.get_data_summary()
        
        assert 'project_data' in summary
        assert summary['project_data']['n_projects'] == 30


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    def test_prepare_features(self):
        """Test feature preparation."""
        loader = DataLoader()
        df = loader.create_sample_data(n_projects=50)
        
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.prepare_features(df, target_col='risk_level')
        
        # Check that derived features are created
        assert 'cost_overrun' in df_processed.columns
        assert 'duration_ratio' in df_processed.columns
        assert 'budget_per_person' in df_processed.columns
    
    def test_prepare_for_training(self):
        """Test train-test split preparation."""
        loader = DataLoader()
        df = loader.create_sample_data(n_projects=100)
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_training(
            df, target_col='risk_level', test_size=0.2
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20


class TestLessonsLearnedParser:
    """Tests for LessonsLearnedParser class."""
    
    def test_parse_document(self):
        """Test document parsing."""
        parser = LessonsLearnedParser()
        
        sample_content = """
        The project faced significant delays due to technical challenges.
        Budget overruns were a major issue. However, the team successfully
        delivered the core functionality. Communication with stakeholders
        was effective throughout the project.
        """
        
        parsed = parser.parse_document(sample_content)
        
        assert 'risk_themes' in parsed
        assert 'sentiment_score' in parsed
        assert 'issues_mentioned' in parsed
        assert 'successes_mentioned' in parsed
    
    def test_generate_risk_features(self):
        """Test risk feature generation."""
        parser = LessonsLearnedParser()
        
        sample_content = """
        Technical debt accumulated during development. Schedule delays
        impacted the timeline. Resource constraints were challenging.
        """
        
        features = parser.generate_risk_features(sample_content)
        
        assert 'll_technical_risk' in features
        assert 'll_schedule_risk' in features
        assert 'll_resource_risk' in features
        assert 'll_sentiment' in features


class TestRiskPredictor:
    """Tests for RiskPredictor class."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        predictor = RiskPredictor(model_type='xgboost')
        assert predictor.model is not None
        assert predictor.model_type == 'xgboost'
    
    def test_train_predict(self):
        """Test model training and prediction."""
        # Create sample data
        loader = DataLoader()
        df = loader.create_sample_data(n_projects=100)
        
        # Prepare data
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_training(
            df, target_col='risk_level', test_size=0.2
        )
        
        # Train model
        predictor = RiskPredictor(model_type='xgboost')
        metrics = predictor.train(X_train, y_train)
        
        assert 'train_accuracy' in metrics
        assert metrics['train_accuracy'] > 0
        
        # Make predictions
        predictions = predictor.predict(X_test)
        assert len(predictions) == len(X_test)


class TestDelayPredictor:
    """Tests for DelayPredictor class."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        predictor = DelayPredictor(model_type='xgboost')
        assert predictor.model is not None
        assert predictor.model_type == 'xgboost'
    
    def test_train_predict(self):
        """Test model training and prediction."""
        # Create sample data
        loader = DataLoader()
        df = loader.create_sample_data(n_projects=100)
        
        # Prepare data
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_training(
            df, target_col='delay_days', test_size=0.2
        )
        
        # Train model
        predictor = DelayPredictor(model_type='xgboost')
        metrics = predictor.train(X_train, y_train)
        
        assert 'train_rmse' in metrics
        assert metrics['train_rmse'] >= 0
        
        # Make predictions
        predictions = predictor.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(predictions >= 0)  # Predictions should be non-negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
