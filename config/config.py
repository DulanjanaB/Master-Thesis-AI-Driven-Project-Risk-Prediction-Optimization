"""
Configuration file for the AI-Driven Project Risk Prediction Framework.
"""

# Data configuration
DATA_CONFIG = {
    'raw_data_dir': 'data/raw',
    'processed_data_dir': 'data/processed',
    'test_size': 0.2,
    'random_state': 42
}

# Model configuration
RISK_MODEL_CONFIG = {
    'model_type': 'xgboost',  # Options: 'random_forest', 'gradient_boosting', 'xgboost'
    'params': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42
    }
}

DELAY_MODEL_CONFIG = {
    'model_type': 'xgboost',  # Options: 'random_forest', 'gradient_boosting', 'xgboost'
    'params': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'random_state': 42
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'categorical_features': ['complexity', 'risk_level'],
    'numerical_features': [
        'budget', 'duration_days', 'team_size',
        'budget_per_person', 'person_days'
    ],
    'derived_features': [
        'cost_overrun', 'duration_ratio', 'budget_per_person', 'person_days'
    ]
}

# Lessons Learned configuration
LESSONS_LEARNED_CONFIG = {
    'll_dir': 'data/lessons_learned',
    'enable_parsing': True,
    'extract_risk_features': True
}

# Explainability configuration
EXPLAINABILITY_CONFIG = {
    'use_shap': True,
    'n_background_samples': 100,
    'generate_plots': True
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'output_dir': 'outputs/plots',
    'figure_format': 'png',
    'dpi': 300,
    'style': 'whitegrid'
}

# Logging configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'outputs/framework.log',
    'log_to_console': True
}

# Output configuration
OUTPUT_CONFIG = {
    'output_dir': 'outputs',
    'models_dir': 'models/saved',
    'save_predictions': True,
    'save_explanations': True,
    'generate_report': True
}
