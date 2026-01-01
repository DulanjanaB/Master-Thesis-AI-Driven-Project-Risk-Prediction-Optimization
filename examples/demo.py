"""
Example script demonstrating the AI-Driven Project Risk Prediction Framework.

This script shows how to:
1. Load or create sample project data
2. Train risk and delay prediction models
3. Evaluate model performance
4. Make predictions on new projects
5. Generate explainable insights
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.framework import RiskPredictionFramework


def main():
    """Run the complete framework pipeline."""
    
    print("=" * 80)
    print("AI-DRIVEN PROJECT RISK PREDICTION FRAMEWORK")
    print("=" * 80)
    print()
    
    # Initialize framework
    print("Initializing framework...")
    framework = RiskPredictionFramework(
        data_dir=Path("data/raw"),
        models_dir=Path("models/saved"),
        output_dir=Path("outputs"),
        log_level="INFO"
    )
    print("✓ Framework initialized\n")
    
    # Load data (create sample data for demonstration)
    print("Loading project data...")
    project_data = framework.load_data(create_sample=True)
    print(f"✓ Loaded {len(project_data)} projects")
    print(f"  Columns: {list(project_data.columns)}\n")
    
    # Prepare data
    print("Preparing data for training...")
    framework.prepare_data(test_size=0.2)
    print(f"✓ Data prepared")
    print(f"  Training samples: {len(framework.X_train)}")
    print(f"  Test samples: {len(framework.X_test)}")
    print(f"  Features: {len(framework.preprocessor.feature_names)}\n")
    
    # Train risk prediction model
    print("Training risk prediction model...")
    risk_metrics = framework.train_risk_model(model_type='xgboost')
    print(f"✓ Risk model trained")
    print(f"  Training Accuracy: {risk_metrics['train_accuracy']:.4f}")
    print(f"  Training F1 Score: {risk_metrics['train_f1']:.4f}\n")
    
    # Train delay prediction model
    print("Training delay prediction model...")
    delay_metrics = framework.train_delay_model(model_type='xgboost')
    print(f"✓ Delay model trained")
    print(f"  Training RMSE: {delay_metrics['train_rmse']:.2f} days")
    print(f"  Training MAE: {delay_metrics['train_mae']:.2f} days")
    print(f"  Training R²: {delay_metrics['train_r2']:.4f}\n")
    
    # Evaluate models
    print("Evaluating models on test data...")
    eval_results = framework.evaluate_models()
    
    if 'risk_model' in eval_results:
        print(f"✓ Risk Model Evaluation:")
        print(f"  Test Accuracy: {eval_results['risk_model']['accuracy']:.4f}")
        print(f"  Test F1 Score: {eval_results['risk_model']['f1_score']:.4f}")
    
    if 'delay_model' in eval_results:
        print(f"✓ Delay Model Evaluation:")
        print(f"  Test RMSE: {eval_results['delay_model']['rmse']:.2f} days")
        print(f"  Test MAE: {eval_results['delay_model']['mae']:.2f} days")
        print(f"  Test R²: {eval_results['delay_model']['r2_score']:.4f}\n")
    
    # Make predictions on new data
    print("Making predictions on sample projects...")
    sample_projects = project_data.head(5)
    predictions = framework.predict(sample_projects)
    
    print(f"✓ Predictions generated for {predictions['n_projects']} projects")
    print("\nSample Predictions:")
    for i in range(min(3, predictions['n_projects'])):
        print(f"  Project {i+1}:")
        print(f"    Risk Level: {predictions['risk_predictions'][i]}")
        print(f"    Predicted Delay: {predictions['delay_predictions'][i]:.1f} days")
    
    print()
    
    # Display feature importance
    if framework.risk_predictor:
        print("Top 5 Risk Prediction Features:")
        importance_df = framework.risk_predictor.get_feature_importance(
            feature_names=framework.preprocessor.feature_names
        )
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print()
    
    # Save models
    print("Saving trained models...")
    framework.save_models()
    print("✓ Models saved\n")
    
    # Generate report
    print("Generating framework report...")
    report = framework.generate_report()
    print("✓ Report generated\n")
    
    print("=" * 80)
    print("FRAMEWORK EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nOutputs saved to: {framework.output_dir}")
    print(f"  - Plots: {framework.output_dir / 'plots'}")
    print(f"  - Models: {framework.models_dir}")
    print(f"  - Results: {framework.output_dir / 'evaluation_results.json'}")
    print(f"  - Report: {framework.output_dir / 'framework_report.txt'}")
    print()


if __name__ == "__main__":
    main()
