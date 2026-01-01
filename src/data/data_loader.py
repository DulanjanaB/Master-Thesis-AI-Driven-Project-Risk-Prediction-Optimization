"""
DataLoader: Handles loading project data and Lessons Learned from various sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads historical project data and Lessons Learned from various file formats.
    
    Supports:
    - CSV, Excel files for structured project data
    - Text, PDF, DOCX files for Lessons Learned documents
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data/raw"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.project_data: Optional[pd.DataFrame] = None
        self.lessons_learned: Optional[List[Dict]] = None
        
    def load_project_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load structured project data from CSV or Excel files.
        
        Expected columns:
        - project_id: Unique project identifier
        - project_name: Name of the project
        - start_date: Project start date
        - end_date: Project end date (or planned end date)
        - budget: Project budget
        - actual_cost: Actual cost incurred
        - duration_days: Planned project duration
        - actual_duration_days: Actual project duration
        - team_size: Number of team members
        - complexity: Project complexity level (Low, Medium, High)
        - risk_level: Actual risk level observed (Low, Medium, High)
        - delayed: Whether project was delayed (0/1)
        - delay_days: Number of days delayed (if applicable)
        
        Args:
            filepath: Path to the data file
            
        Returns:
            DataFrame containing project data
        """
        filepath = Path(filepath)
        
        try:
            if filepath.suffix.lower() == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            logger.info(f"Loaded {len(df)} projects from {filepath}")
            self.project_data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading project data: {str(e)}")
            raise
    
    def load_lessons_learned(self, filepath: Union[str, Path]) -> List[Dict]:
        """
        Load Lessons Learned documents.
        
        Args:
            filepath: Path to Lessons Learned file or directory
            
        Returns:
            List of dictionaries containing lessons learned
        """
        filepath = Path(filepath)
        lessons = []
        
        try:
            if filepath.is_file():
                lessons.append(self._load_single_lesson(filepath))
            elif filepath.is_dir():
                for file in filepath.glob('*'):
                    if file.is_file():
                        try:
                            lessons.append(self._load_single_lesson(file))
                        except Exception as e:
                            logger.warning(f"Failed to load {file}: {str(e)}")
            
            logger.info(f"Loaded {len(lessons)} Lessons Learned documents")
            self.lessons_learned = lessons
            return lessons
            
        except Exception as e:
            logger.error(f"Error loading Lessons Learned: {str(e)}")
            raise
    
    def _load_single_lesson(self, filepath: Path) -> Dict:
        """
        Load a single Lessons Learned document.
        
        Args:
            filepath: Path to the document
            
        Returns:
            Dictionary with document metadata and content
        """
        content = ""
        
        if filepath.suffix.lower() == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        elif filepath.suffix.lower() == '.csv':
            df = pd.read_csv(filepath)
            content = df.to_string()
        else:
            # For other formats, store filepath for later processing
            logger.warning(f"Format {filepath.suffix} requires specialized parser")
        
        return {
            'filename': filepath.name,
            'filepath': str(filepath),
            'content': content,
            'loaded_at': datetime.now().isoformat()
        }
    
    def create_sample_data(self, n_projects: int = 100, save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Create sample project data for demonstration and testing.
        
        Args:
            n_projects: Number of sample projects to generate
            save_path: Optional path to save the generated data
            
        Returns:
            DataFrame with sample project data
        """
        np.random.seed(42)
        
        # Generate synthetic project data
        data = {
            'project_id': [f'PRJ{i:04d}' for i in range(1, n_projects + 1)],
            'project_name': [f'Project {i}' for i in range(1, n_projects + 1)],
            'start_date': pd.date_range('2020-01-01', periods=n_projects, freq='3D'),
            'budget': np.random.uniform(50000, 500000, n_projects).round(2),
            'duration_days': np.random.randint(30, 365, n_projects),
            'team_size': np.random.randint(3, 20, n_projects),
            'complexity': np.random.choice(['Low', 'Medium', 'High'], n_projects, p=[0.3, 0.5, 0.2]),
            'risk_level': np.random.choice(['Low', 'Medium', 'High'], n_projects, p=[0.4, 0.4, 0.2]),
        }
        
        df = pd.DataFrame(data)
        
        # Generate actual values with some correlation to features
        df['actual_duration_days'] = (df['duration_days'] * 
                                       np.random.uniform(0.8, 1.5, n_projects)).astype(int)
        df['delayed'] = (df['actual_duration_days'] > df['duration_days']).astype(int)
        df['delay_days'] = np.maximum(0, df['actual_duration_days'] - df['duration_days'])
        df['actual_cost'] = (df['budget'] * 
                            np.random.uniform(0.7, 1.4, n_projects)).round(2)
        
        df['end_date'] = df['start_date'] + pd.to_timedelta(df['actual_duration_days'], unit='D')
        
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Sample data saved to {save_path}")
        
        self.project_data = df
        return df
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of loaded data.
        
        Returns:
            Dictionary with data summary
        """
        summary = {}
        
        if self.project_data is not None:
            summary['project_data'] = {
                'n_projects': len(self.project_data),
                'columns': list(self.project_data.columns),
                'date_range': (
                    self.project_data['start_date'].min() if 'start_date' in self.project_data else None,
                    self.project_data['end_date'].max() if 'end_date' in self.project_data else None
                ),
                'delayed_projects': self.project_data['delayed'].sum() if 'delayed' in self.project_data else None
            }
        
        if self.lessons_learned is not None:
            summary['lessons_learned'] = {
                'n_documents': len(self.lessons_learned),
                'total_content_length': sum(len(ll.get('content', '')) for ll in self.lessons_learned)
            }
        
        return summary
