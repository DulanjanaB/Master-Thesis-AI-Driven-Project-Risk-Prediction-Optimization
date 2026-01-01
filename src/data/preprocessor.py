"""
DataPreprocessor: Prepares and transforms project data for model training.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses project data for risk and delay prediction models.
    
    Handles:
    - Feature engineering
    - Missing value imputation
    - Encoding categorical variables
    - Feature scaling
    - Train-test splitting
    """
    
    def __init__(self):
        """Initialize DataPreprocessor with encoders and scalers."""
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: Optional[List[str]] = None
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []
        
    def prepare_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare and engineer features from raw project data.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column (if present)
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Convert date columns to datetime if present
        date_columns = ['start_date', 'end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Feature engineering
        if 'budget' in df.columns and 'actual_cost' in df.columns:
            df['cost_overrun'] = (df['actual_cost'] - df['budget']) / df['budget']
            df['over_budget'] = (df['actual_cost'] > df['budget']).astype(int)
        
        if 'duration_days' in df.columns and 'actual_duration_days' in df.columns:
            df['duration_ratio'] = df['actual_duration_days'] / df['duration_days']
        
        if 'budget' in df.columns and 'team_size' in df.columns:
            df['budget_per_person'] = df['budget'] / df['team_size']
        
        if 'duration_days' in df.columns and 'team_size' in df.columns:
            df['person_days'] = df['duration_days'] * df['team_size']
        
        # Extract temporal features if dates are available
        if 'start_date' in df.columns:
            df['start_month'] = df['start_date'].dt.month
            df['start_quarter'] = df['start_date'].dt.quarter
            df['start_year'] = df['start_date'].dt.year
        
        # Complexity score (if complexity is categorical)
        if 'complexity' in df.columns:
            complexity_map = {'Low': 1, 'Medium': 2, 'High': 3}
            df['complexity_score'] = df['complexity'].map(complexity_map)
        
        # Identify feature types
        self._identify_feature_types(df, target_col)
        
        return df
    
    def _identify_feature_types(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """
        Identify categorical and numerical features.
        
        Args:
            df: Input DataFrame
            target_col: Target column to exclude
        """
        # Exclude certain columns
        exclude_cols = ['project_id', 'project_name', 'start_date', 'end_date', 
                       'actual_duration_days', 'actual_cost', 'delayed', 'delay_days']
        if target_col:
            exclude_cols.append(target_col)
        
        self.categorical_features = []
        self.numerical_features = []
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                self.categorical_features.append(col)
            elif np.issubdtype(df[col].dtype, np.number):
                self.numerical_features.append(col)
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        for col in self.categorical_features:
            if col not in df.columns:
                continue
            
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                # Handle missing values
                df[col] = df[col].fillna('Unknown')
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    df[col] = df[col].fillna('Unknown')
                    # Map unseen categories to a default value
                    known_classes = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_classes else 'Unknown'
                    )
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scaler (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        # Get numerical columns that exist in df
        numeric_cols = [col for col in self.numerical_features if col in df.columns]
        
        if not numeric_cols:
            return df
        
        # Handle missing values
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def prepare_for_training(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Prepare features
        df_processed = self.prepare_features(df, target_col)
        
        # Separate features and target
        y = df_processed[target_col].copy()
        X = df_processed.drop(columns=[target_col])
        
        # Remove non-feature columns
        exclude_cols = ['project_id', 'project_name', 'start_date', 'end_date']
        X = X.drop(columns=[col for col in exclude_cols if col in X.columns])
        
        # Encode categorical features
        X = self.encode_features(X, fit=True)
        
        # Scale numerical features
        X = self.scale_features(X, fit=True)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) < 20 else None
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Number of features: {len(self.feature_names)}")
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        # Prepare features
        df_processed = self.prepare_features(df)
        
        # Remove non-feature columns
        exclude_cols = ['project_id', 'project_name', 'start_date', 'end_date']
        df_processed = df_processed.drop(columns=[col for col in exclude_cols if col in df_processed.columns])
        
        # Encode and scale
        df_processed = self.encode_features(df_processed, fit=False)
        df_processed = self.scale_features(df_processed, fit=False)
        
        # Ensure all required features are present
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0
            
            # Reorder columns to match training data
            df_processed = df_processed[self.feature_names]
        
        return df_processed
