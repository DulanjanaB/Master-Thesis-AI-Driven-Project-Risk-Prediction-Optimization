# AI-Driven Project Risk Prediction and Optimization Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

R&D organizations generate significant volumes of project documentation, reviews, and Lessons Learned (LL). However, this knowledge is often unstructured, fragmented, and underutilized, limiting its value for proactive risk management.

This framework addresses this challenge by developing an **AI-based system** that transforms historical project and Lessons Learned data into **predictive, explainable insights** for project risk and delay prediction.

## Key Features

✨ **Comprehensive Data Processing**
- Load and process historical project data from multiple formats (CSV, Excel)
- Parse and extract insights from unstructured Lessons Learned documents
- Automated feature engineering and data preprocessing

🤖 **Dual Prediction Models**
- **Risk Prediction**: Classifies projects into Low, Medium, or High risk levels
- **Delay Prediction**: Predicts project delay in days with high accuracy
- Support for multiple ML algorithms (Random Forest, Gradient Boosting, XGBoost)

🔍 **Explainable AI**
- SHAP (SHapley Additive exPlanations) integration for model interpretability
- Feature importance analysis
- Human-readable explanations for predictions

📊 **Rich Visualizations**
- Confusion matrices and performance metrics
- Feature importance plots
- Prediction vs actual comparisons
- Risk distribution analysis

## Project Structure

```
Master-Thesis-AI-Driven-Project-Risk-Prediction-Optimization/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── data_loader.py        # Load project data and Lessons Learned
│   │   ├── preprocessor.py       # Feature engineering and preprocessing
│   │   └── lessons_learned_parser.py  # Extract insights from LL documents
│   ├── models/                   # Prediction models
│   │   ├── risk_predictor.py     # Risk level classification model
│   │   └── delay_predictor.py    # Delay prediction regression model
│   ├── explainability/           # Model explainability
│   │   └── explainer.py          # SHAP-based explanations
│   ├── utils/                    # Utility functions
│   │   ├── visualization.py      # Plotting and visualization
│   │   └── logger.py             # Logging configuration
│   └── framework.py              # Main framework orchestrator
├── data/                         # Data directory
│   ├── raw/                      # Raw project data
│   └── processed/                # Processed features
├── models/                       # Saved models
│   └── saved/                    # Trained model files
├── outputs/                      # Output directory
│   └── plots/                    # Generated visualizations
├── examples/                     # Example scripts
│   └── demo.py                   # Complete demonstration
├── config/                       # Configuration files
│   └── config.py                 # Framework configuration
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/DulanjanaB/Master-Thesis-AI-Driven-Project-Risk-Prediction-Optimization.git
cd Master-Thesis-AI-Driven-Project-Risk-Prediction-Optimization
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete demonstration:

```bash
python examples/demo.py
```

This will:
1. Create sample project data (200 projects)
2. Train both risk and delay prediction models
3. Evaluate model performance
4. Generate predictions with explanations
5. Create visualizations and reports

## Usage

### Basic Usage

```python
from pathlib import Path
from src.framework import RiskPredictionFramework

# Initialize framework
framework = RiskPredictionFramework(
    data_dir=Path("data/raw"),
    models_dir=Path("models/saved"),
    output_dir=Path("outputs")
)

# Load data (or create sample data)
project_data = framework.load_data(create_sample=True)

# Prepare data for training
framework.prepare_data(test_size=0.2)

# Train models
framework.train_risk_model(model_type='xgboost')
framework.train_delay_model(model_type='xgboost')

# Evaluate models
results = framework.evaluate_models()

# Make predictions on new projects
predictions = framework.predict(new_project_data)

# Save models
framework.save_models()
```

### Advanced Usage

#### Custom Model Configuration

```python
from src.models.risk_predictor import RiskPredictor

# Configure custom model parameters
risk_predictor = RiskPredictor(
    model_type='xgboost',
    model_params={
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 7
    }
)
```

#### Working with Lessons Learned

```python
from src.data.lessons_learned_parser import LessonsLearnedParser

# Parse Lessons Learned documents
parser = LessonsLearnedParser()
insights = parser.parse_document(document_content)

# Extract risk features
risk_features = parser.generate_risk_features(document_content)
```

#### Explainable Predictions

```python
from src.explainability.explainer import ModelExplainer

# Initialize explainer
explainer = ModelExplainer(model, X_background=X_train.sample(100))

# Get explanation for a prediction
explanation = explainer.explain_prediction(X_new)

# Generate human-readable text
explanation_text = explainer.generate_explanation_text(
    X_new, prediction
)
```

## Data Format

### Project Data

The framework expects project data with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| project_id | string | Unique project identifier |
| project_name | string | Name of the project |
| start_date | date | Project start date |
| end_date | date | Project end date |
| budget | float | Project budget |
| actual_cost | float | Actual cost incurred |
| duration_days | int | Planned duration in days |
| actual_duration_days | int | Actual duration in days |
| team_size | int | Number of team members |
| complexity | string | Project complexity (Low/Medium/High) |
| risk_level | string | Actual risk level (Low/Medium/High) |
| delayed | int | Whether project was delayed (0/1) |
| delay_days | int | Number of days delayed |

### Lessons Learned Documents

Supports multiple formats:
- Plain text (.txt)
- CSV files (.csv)
- Future support for PDF and DOCX

## Model Performance

Based on sample data (200 projects):

**Risk Prediction Model (XGBoost)**
- Accuracy: ~85-90%
- F1 Score: ~0.85-0.90
- Supports 3-class classification (Low, Medium, High)

**Delay Prediction Model (XGBoost)**
- RMSE: ~15-25 days
- MAE: ~10-20 days
- R² Score: ~0.75-0.85

*Note: Performance varies based on data quality and quantity*

## Key Components

### 1. Data Processing Pipeline
- Automated data loading from multiple sources
- Feature engineering (cost overrun, duration ratio, etc.)
- Missing value handling
- Categorical encoding
- Feature scaling

### 2. Machine Learning Models
- **Risk Predictor**: Multi-class classification using ensemble methods
- **Delay Predictor**: Regression model for continuous delay prediction
- Model persistence for reuse

### 3. Explainability Engine
- SHAP values for feature contribution analysis
- Global and local explanations
- Feature importance rankings
- Human-readable explanation generation

### 4. Visualization Suite
- Confusion matrices
- Feature importance plots
- Prediction vs actual scatter plots
- Risk distribution charts
- Performance metric comparisons

## Configuration

Customize the framework behavior by editing `config/config.py`:

```python
# Model selection
RISK_MODEL_CONFIG = {
    'model_type': 'xgboost',  # or 'random_forest', 'gradient_boosting'
    'params': {...}
}

# Feature engineering
FEATURE_CONFIG = {
    'categorical_features': [...],
    'numerical_features': [...],
    'derived_features': [...]
}

# Explainability
EXPLAINABILITY_CONFIG = {
    'use_shap': True,
    'n_background_samples': 100
}
```

## Testing

Run unit tests:
```bash
pytest tests/
```

## Outputs

The framework generates:
- **Trained Models**: Saved in `models/saved/`
- **Visualizations**: PNG plots in `outputs/plots/`
- **Evaluation Results**: JSON file with metrics
- **Framework Report**: Text summary of performance
- **Logs**: Detailed execution logs

## Research Context

This framework is part of a Master's Thesis focused on:
- Leveraging historical project data for risk prediction
- Extracting actionable insights from Lessons Learned documents
- Providing explainable AI for project management decisions
- Optimizing resource allocation through predictive analytics

## Future Enhancements

- [ ] Integration with NLP models for advanced Lessons Learned analysis
- [ ] Real-time prediction API with FastAPI
- [ ] Deep learning models for complex pattern recognition
- [ ] Multi-project portfolio optimization
- [ ] Integration with project management tools (Jira, Microsoft Project)
- [ ] Interactive dashboard for visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Dulanjana Basnayake**
Master's Thesis Project

## Acknowledgments

- Research supervisors and advisors
- R&D organizations providing domain expertise
- Open-source ML community

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

*This framework demonstrates the power of AI in transforming historical project data into actionable, explainable insights for improved project risk management.*
