# Architecture Overview

## System Architecture

The AI-Driven Project Risk Prediction Framework follows a modular architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    RiskPredictionFramework                  │
│                   (Main Orchestrator)                       │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│ Data Layer   │        │ Model Layer  │
├──────────────┤        ├──────────────┤
│ DataLoader   │        │ RiskPredictor│
│ Preprocessor │        │DelayPredictor│
│ LL Parser    │        │              │
└──────┬───────┘        └──────┬───────┘
       │                       │
       └───────────┬───────────┘
                   ▼
        ┌──────────────────┐
        │ Explainability   │
        │ & Visualization  │
        ├──────────────────┤
        │ ModelExplainer   │
        │ Visualizer       │
        └──────────────────┘
```

## Data Flow

1. **Data Ingestion**
   - Load historical project data
   - Parse Lessons Learned documents
   - Generate sample data if needed

2. **Preprocessing**
   - Feature engineering
   - Encoding categorical variables
   - Scaling numerical features
   - Train-test split

3. **Model Training**
   - Risk prediction (classification)
   - Delay prediction (regression)
   - Model persistence

4. **Evaluation**
   - Performance metrics
   - Confusion matrices
   - Feature importance

5. **Prediction**
   - Risk level classification
   - Delay estimation
   - Explainable insights

6. **Visualization**
   - Performance plots
   - Feature importance charts
   - Prediction analysis

## Component Details

### Data Layer

**DataLoader**: Handles various data sources (CSV, Excel, text files)
- Multi-format support
- Sample data generation
- Data validation

**DataPreprocessor**: Feature engineering pipeline
- Derived features (cost_overrun, duration_ratio, etc.)
- Missing value handling
- Encoding and scaling
- Train-test splitting

**LessonsLearnedParser**: Text analysis
- Risk theme extraction
- Sentiment analysis
- Issue and success identification
- Feature generation

### Model Layer

**RiskPredictor**: Multi-class classification
- Support for multiple algorithms (RF, GB, XGBoost)
- 3-class output (Low, Medium, High risk)
- Probability predictions
- Feature importance

**DelayPredictor**: Regression model
- Continuous delay prediction in days
- Multiple algorithm support
- Non-negative predictions
- Feature importance

### Explainability Layer

**ModelExplainer**: SHAP integration
- Global feature importance
- Local explanations per prediction
- Human-readable text generation
- Fallback to feature importance

### Visualization Layer

**Visualizer**: Comprehensive plotting
- Confusion matrices
- Feature importance charts
- Prediction vs actual plots
- Distribution analysis

## Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Extensibility**: Easy to add new models or features
3. **Explainability**: All predictions include interpretable insights
4. **Configurability**: Centralized configuration management
5. **Testability**: Unit tests for all core components
6. **Logging**: Comprehensive logging for debugging

## Technology Stack

- **Core**: Python 3.8+
- **ML**: scikit-learn, XGBoost
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Explainability**: SHAP
- **Testing**: pytest

## Future Extensions

- Deep learning models (LSTM, Transformers)
- Real-time API with FastAPI
- Advanced NLP for Lessons Learned
- Portfolio-level optimization
- Integration with PM tools
