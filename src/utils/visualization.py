"""
Visualization utilities for model results and explanations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class Visualizer:
    """
    Creates visualizations for model performance and explainability.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize Visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_feature_importance(
        self, 
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_name: Optional[str] = None
    ):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display
            title: Plot title
            save_name: Optional filename to save plot
        """
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        save_name: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Names of classes
            title: Plot title
            save_name: Optional filename to save plot
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        plt.close()
    
    def plot_prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Prediction vs Actual",
        save_name: Optional[str] = None
    ):
        """
        Plot predicted vs actual values (for regression).
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            save_name: Optional filename to save plot
        """
        plt.figure(figsize=(8, 8))
        
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        plt.close()
    
    def plot_risk_distribution(
        self,
        risk_levels: pd.Series,
        title: str = "Risk Level Distribution",
        save_name: Optional[str] = None
    ):
        """
        Plot distribution of risk levels.
        
        Args:
            risk_levels: Series of risk levels
            title: Plot title
            save_name: Optional filename to save plot
        """
        plt.figure(figsize=(8, 6))
        
        risk_counts = risk_levels.value_counts().sort_index()
        colors = ['green', 'orange', 'red'][:len(risk_counts)]
        
        plt.bar(risk_counts.index, risk_counts.values, color=colors)
        plt.xlabel('Risk Level')
        plt.ylabel('Number of Projects')
        plt.title(title)
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        plt.close()
    
    def plot_delay_histogram(
        self,
        delays: np.ndarray,
        title: str = "Project Delay Distribution",
        save_name: Optional[str] = None
    ):
        """
        Plot histogram of project delays.
        
        Args:
            delays: Array of delay values in days
            title: Plot title
            save_name: Optional filename to save plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.hist(delays, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', label='No Delay')
        plt.xlabel('Delay (days)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        plt.close()
    
    def plot_metrics_comparison(
        self,
        metrics: Dict[str, float],
        title: str = "Model Performance Metrics",
        save_name: Optional[str] = None
    ):
        """
        Plot comparison of model metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            title: Plot title
            save_name: Optional filename to save plot
        """
        plt.figure(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        plt.bar(metric_names, metric_values)
        plt.ylabel('Score')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(metric_values) * 1.1)
        
        # Add value labels on bars
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_name:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        plt.close()
