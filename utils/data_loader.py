import pandas as pd
import streamlit as st
import os
from typing import Dict, List, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define cache functions outside the class to avoid hashing issues
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def cached_load_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load data from a CSV file with caching.
    
    Args:
        filepath: Full path to the CSV file
        
    Returns:
        DataFrame containing the data, or None if loading fails
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"File {filepath} does not exist")
            return None
        
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded {os.path.basename(filepath)}: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Original columns: {list(df.columns)}")
        
        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        logger.info(f"Lowercased columns: {list(df.columns)}")
        
        # Map specific column names to expected format
        column_mapping = {
            'dau': 'daily_active_users',
            'wau': 'weekly_active_users',
            'mau': 'monthly_active_users'
        }
        
        # Rename columns based on mapping
        df = df.rename(columns=column_mapping)
        logger.info(f"After mapping columns: {list(df.columns)}")
        
        # Convert date columns to datetime
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                logger.warning(f"Could not convert column {col} to datetime: {str(e)}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading {os.path.basename(filepath)}: {str(e)}")
        return None

class DataLoader:
    """
    A class to handle data loading with caching for the Matiks dashboard.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataLoader with the data directory path.
        
        Args:
            data_dir: Path to the directory containing data files
        """
        self.data_dir = data_dir
        self._available_files = self._get_available_files()
        logger.info(f"DataLoader initialized with {len(self._available_files)} available files")
    
    def _get_available_files(self) -> List[str]:
        """
        Get a list of available CSV files in the data directory.
        
        Returns:
            List of CSV filenames
        """
        try:
            if not os.path.exists(self.data_dir):
                logger.warning(f"Data directory {self.data_dir} does not exist")
                return []
            
            files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            logger.info(f"Found {len(files)} CSV files in {self.data_dir}")
            return files
        except Exception as e:
            logger.error(f"Error getting available files: {str(e)}")
            return []
    
    def load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load data from a CSV file with caching.
        
        Args:
            filename: Name of the CSV file to load
            
        Returns:
            DataFrame containing the data, or None if loading fails
        """
        filepath = os.path.join(self.data_dir, filename)
        return cached_load_csv(filepath)
    
    def get_available_datasets(self) -> Dict[str, str]:
        """
        Get a dictionary of available datasets with descriptive names.
        
        Returns:
            Dictionary mapping file names to descriptive names
        """
        dataset_descriptions = {
            'active_users.csv': 'Active Users (DAU/WAU/MAU)',
            'revenue_by_date.csv': 'Revenue Over Time',
            'revenue_by_device.csv': 'Revenue by Device Type',
            'revenue_by_mode.csv': 'Revenue by Game Mode',
            'revenue_by_segment.csv': 'Revenue by User Segment',
            'device_distribution.csv': 'Device Distribution',
            'game_mode_distribution.csv': 'Game Mode Distribution',
            'funnel_analysis.csv': 'User Funnel Analysis',
            'cohort_retention.csv': 'Cohort Retention',
            'preprocessed_data.csv': 'Complete User Data'
        }
        
        available_datasets = {}
        for file in self._available_files:
            if file in dataset_descriptions:
                available_datasets[file] = dataset_descriptions[file]
            else:
                # Create a readable name from the filename
                name = file.replace('.csv', '').replace('_', ' ').title()
                available_datasets[file] = name
        
        return available_datasets
    
    def create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """
        Create sample data if real data files are not available.
        
        Returns:
            Dictionary of sample DataFrames
        """
        logger.info("Creating sample data")
        
        # Sample date range
        date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Sample active users data
        active_users = pd.DataFrame({
            'date': date_range,
            'daily_active_users': [100 + i % 50 + (i // 30) * 20 for i in range(len(date_range))],
            'weekly_active_users': [500 + i % 100 + (i // 30) * 50 for i in range(len(date_range))],
            'monthly_active_users': [1500 + i % 200 + (i // 30) * 100 for i in range(len(date_range))]
        })
        
        # Sample revenue data
        revenue_by_date = pd.DataFrame({
            'date': date_range,
            'revenue': [1000 + i % 500 + (i // 30) * 200 for i in range(len(date_range))],
            'transactions': [50 + i % 30 for i in range(len(date_range))]
        })
        
        # Sample device distribution
        device_distribution = pd.DataFrame({
            'device_type': ['Mobile', 'Tablet', 'Desktop', 'Console', 'Other'],
            'user_count': [5000, 2500, 1500, 800, 200],
            'percentage': [50, 25, 15, 8, 2]
        })
        
        # Sample game mode distribution
        game_mode_distribution = pd.DataFrame({
            'game_mode': ['Casual', 'Competitive', 'Story', 'Multiplayer', 'Training'],
            'play_count': [10000, 7500, 5000, 8000, 3000],
            'percentage': [30, 22, 15, 24, 9]
        })
        
        # Sample funnel analysis
        funnel_analysis = pd.DataFrame({
            'stage': ['Visit', 'Sign Up', 'Tutorial', 'First Game', 'Return Next Day', 'Purchase'],
            'user_count': [10000, 5000, 3000, 2500, 1200, 500],
            'conversion_rate': [100, 50, 60, 83, 48, 42]
        })
        
        # Sample cohort retention
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        cohort_data = []
        for i, month in enumerate(months):
            for j in range(6):
                if j <= i:
                    retention = 100 if j == 0 else int(100 * (0.7 ** j))
                    cohort_data.append({
                        'cohort': month,
                        'month': j,
                        'retention_rate': retention
                    })
        
        cohort_retention = pd.DataFrame(cohort_data)
        
        # Return all sample datasets
        return {
            'active_users.csv': active_users,
            'revenue_by_date.csv': revenue_by_date,
            'device_distribution.csv': device_distribution,
            'game_mode_distribution.csv': game_mode_distribution,
            'funnel_analysis.csv': funnel_analysis,
            'cohort_retention.csv': cohort_retention
        }
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets or create sample data if files don't exist.
        
        Returns:
            Dictionary mapping file names to DataFrames
        """
        datasets = {}
        
        # Try to load all available files
        for file in self._available_files:
            df = self.load_data(file)
            if df is not None:
                datasets[file] = df
        
        # If no data was loaded, create sample data
        if not datasets:
            logger.warning("No data files could be loaded, creating sample data")
            datasets = self.create_sample_data()
        
        return datasets
