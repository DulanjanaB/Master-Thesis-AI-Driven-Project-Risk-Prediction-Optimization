# API Documentation

## Overview

This document provides detailed API documentation for the AI-Driven Project Risk Prediction Framework.

## Core Classes

### RiskPredictionFramework

Main orchestrator class that integrates all components.

**Initialization:**
```python
from src.framework import RiskPredictionFramework

framework = RiskPredictionFramework(
    data_dir=Path("data/raw"),
    models_dir=Path("models/saved"),
    output_dir=Path("outputs"),
    log_level="INFO"
)
```

**Key Methods:**
- `load_data()` - Load project data
- `prepare_data()` - Prepare data for training
- `train_risk_model()` - Train risk classifier
- `train_delay_model()` - Train delay regressor
- `evaluate_models()` - Evaluate performance
- `predict()` - Make predictions with explanations
- `save_models()` / `load_models()` - Persist models

### DataLoader

Handles data loading and sample generation.

### DataPreprocessor

Feature engineering and preprocessing pipeline.

### LessonsLearnedParser

Extract insights from unstructured documents.

### RiskPredictor / DelayPredictor

ML models for risk and delay prediction.

### ModelExplainer

SHAP-based explainability for predictions.

### Visualizer

Generate plots and visualizations.

See full API documentation in the source code docstrings.
