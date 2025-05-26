import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and processing data for the Matiks Gaming Analytics Dashboard."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataLoader with a data directory.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.cache = {}
    
    def load_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load a dataset from a CSV file.
        
        Args:
            filename: Name of the CSV file
        
        Returns:
            DataFrame containing the data
        """
        try:
            # Check if the file exists
            file_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return pd.DataFrame()
            
            # Load the data
            df = pd.read_csv(file_path)
            
            # Process date columns
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Cache the data
            self.cache[filename] = df
            
            return df
        except Exception as e:
            logger.error(f"Error loading dataset {filename}: {str(e)}")
            return pd.DataFrame()
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets from the data directory.
        
        Returns:
            Dictionary of DataFrames
        """
        datasets = {}
        
        try:
            # List of expected datasets
            expected_files = [
                'active_users.csv',
                'revenue_by_date.csv',
                'device_distribution.csv',
                'game_mode_distribution.csv',
                'revenue_by_segment.csv',
                'funnel_analysis.csv',
                'cohort_retention.csv'
            ]
            
            # Load each dataset
            for filename in expected_files:
                datasets[filename] = self.load_dataset(filename)
            
            return datasets
        except Exception as e:
            logger.error(f"Error loading all datasets: {str(e)}")
            return datasets
    
    def filter_data(self, df: pd.DataFrame, date_range: Dict, filters: Dict) -> pd.DataFrame:
        """
        Filter a DataFrame based on date range and filters.
        
        Args:
            df: DataFrame to filter
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        
        Returns:
            Filtered DataFrame
        """
        try:
            # Create a copy of the DataFrame
            filtered_df = df.copy()
            
            # Apply date filter if Date column exists
            if 'Date' in filtered_df.columns and date_range:
                start_date = pd.Timestamp(date_range['start'])
                end_date = pd.Timestamp(date_range['end'])
                filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
            
            # Apply device filter
            if 'Device_Type' in filtered_df.columns and filters.get('device') != 'All Devices':
                filtered_df = filtered_df[filtered_df['Device_Type'] == filters['device']]
            
            # Apply game mode filter
            if 'Game_Mode' in filtered_df.columns and filters.get('mode') != 'All Modes':
                filtered_df = filtered_df[filtered_df['Game_Mode'] == filters['mode']]
            
            # Apply segment filter
            if 'Segment' in filtered_df.columns and filters.get('segment') != 'All Segments':
                filtered_df = filtered_df[filtered_df['Segment'] == filters['segment']]
            
            return filtered_df
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            return df
    
    def calculate_kpis(self, datasets: Dict[str, pd.DataFrame], date_range: Dict) -> Dict[str, Dict]:
        """
        Calculate KPIs based on the datasets.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
        
        Returns:
            Dictionary of KPI data
        """
        kpi_data = {}
        
        try:
            # Calculate DAU, WAU, MAU
            if 'active_users.csv' in datasets:
                df = datasets['active_users.csv']
                
                if not df.empty and 'Date' in df.columns:
                    # Filter by date range
                    start_date = pd.Timestamp(date_range['start'])
                    end_date = pd.Timestamp(date_range['end'])
                    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                    
                    if not df.empty:
                        # Calculate DAU
                        if 'DAU' in df.columns:
                            current_dau = df['DAU'].iloc[-1]
                            prev_dau = df['DAU'].iloc[0] if len(df) > 1 else current_dau
                            dau_change = ((current_dau - prev_dau) / prev_dau * 100) if prev_dau > 0 else 0
                            
                            kpi_data["Daily Active Users"] = {
                                "value": current_dau,
                                "change": dau_change,
                                "trend": "up" if dau_change >= 0 else "down",
                                "format": "{:,.0f}"
                            }
                        
                        # Calculate WAU
                        if 'WAU' in df.columns:
                            current_wau = df['WAU'].iloc[-1]
                            prev_wau = df['WAU'].iloc[0] if len(df) > 1 else current_wau
                            wau_change = ((current_wau - prev_wau) / prev_wau * 100) if prev_wau > 0 else 0
                            
                            kpi_data["Weekly Active Users"] = {
                                "value": current_wau,
                                "change": wau_change,
                                "trend": "up" if wau_change >= 0 else "down",
                                "format": "{:,.0f}"
                            }
                        
                        # Calculate MAU
                        if 'MAU' in df.columns:
                            current_mau = df['MAU'].iloc[-1]
                            prev_mau = df['MAU'].iloc[0] if len(df) > 1 else current_mau
                            mau_change = ((current_mau - prev_mau) / prev_mau * 100) if prev_mau > 0 else 0
                            
                            kpi_data["Monthly Active Users"] = {
                                "value": current_mau,
                                "change": mau_change,
                                "trend": "up" if mau_change >= 0 else "down",
                                "format": "{:,.0f}"
                            }
                        
                        # Calculate Stickiness
                        if 'Stickiness' in df.columns:
                            current_stickiness = df['Stickiness'].iloc[-1]
                            prev_stickiness = df['Stickiness'].iloc[0] if len(df) > 1 else current_stickiness
                            stickiness_change = ((current_stickiness - prev_stickiness) / prev_stickiness * 100) if prev_stickiness > 0 else 0
                            
                            kpi_data["Stickiness"] = {
                                "value": current_stickiness,
                                "change": stickiness_change,
                                "trend": "up" if stickiness_change >= 0 else "down",
                                "format": "{:.2f}"
                            }
            
            # Calculate Revenue KPIs
            if 'revenue_by_date.csv' in datasets:
                df = datasets['revenue_by_date.csv']
                
                if not df.empty and 'Date' in df.columns and 'Revenue' in df.columns:
                    # Filter by date range
                    start_date = pd.Timestamp(date_range['start'])
                    end_date = pd.Timestamp(date_range['end'])
                    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                    
                    if not df.empty:
                        # Calculate Total Revenue
                        total_revenue = df['Revenue'].sum()
                        
                        # Calculate previous period for comparison
                        date_diff = (end_date - start_date).days
                        prev_start_date = start_date - timedelta(days=date_diff)
                        prev_end_date = start_date - timedelta(days=1)
                        
                        prev_df = datasets['revenue_by_date.csv']
                        prev_df = prev_df[(prev_df['Date'] >= prev_start_date) & (prev_df['Date'] <= prev_end_date)]
                        
                        prev_total_revenue = prev_df['Revenue'].sum() if not prev_df.empty else total_revenue
                        revenue_change = ((total_revenue - prev_total_revenue) / prev_total_revenue * 100) if prev_total_revenue > 0 else 0
                        
                        kpi_data["Total Revenue"] = {
                            "value": total_revenue,
                            "change": revenue_change,
                            "trend": "up" if revenue_change >= 0 else "down",
                            "format": "${:,.2f}"
                        }
                        
                        # Calculate ARPU if we have user data
                        if 'active_users.csv' in datasets:
                            users_df = datasets['active_users.csv']
                            
                            if not users_df.empty and 'Date' in users_df.columns and 'DAU' in users_df.columns:
                                # Filter by date range
                                users_df = users_df[(users_df['Date'] >= start_date) & (users_df['Date'] <= end_date)]
                                
                                if not users_df.empty:
                                    # Calculate average DAU
                                    avg_dau = users_df['DAU'].mean()
                                    
                                    # Calculate ARPU
                                    arpu = total_revenue / (avg_dau * date_diff) if avg_dau > 0 and date_diff > 0 else 0
                                    
                                    # Calculate previous ARPU
                                    prev_users_df = datasets['active_users.csv']
                                    prev_users_df = prev_users_df[(prev_users_df['Date'] >= prev_start_date) & (prev_users_df['Date'] <= prev_end_date)]
                                    
                                    prev_avg_dau = prev_users_df['DAU'].mean() if not prev_users_df.empty else avg_dau
                                    prev_arpu = prev_total_revenue / (prev_avg_dau * date_diff) if prev_avg_dau > 0 and date_diff > 0 else arpu
                                    
                                    arpu_change = ((arpu - prev_arpu) / prev_arpu * 100) if prev_arpu > 0 else 0
                                    
                                    kpi_data["ARPU"] = {
                                        "value": arpu,
                                        "change": arpu_change,
                                        "trend": "up" if arpu_change >= 0 else "down",
                                        "format": "${:.2f}"
                                    }
            
            return kpi_data
        except Exception as e:
            logger.error(f"Error calculating KPIs: {str(e)}")
            return kpi_data
    
    def get_cohort_data(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Get cohort data for cohort analysis.
        
        Args:
            datasets: Dictionary of DataFrames
        
        Returns:
            DataFrame with cohort data
        """
        try:
            if 'cohort_retention.csv' in datasets:
                return datasets['cohort_retention.csv']
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting cohort data: {str(e)}")
            return pd.DataFrame()
    
    def get_funnel_data(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Get funnel data for funnel analysis.
        
        Args:
            datasets: Dictionary of DataFrames
        
        Returns:
            DataFrame with funnel data
        """
        try:
            if 'funnel_analysis.csv' in datasets:
                return datasets['funnel_analysis.csv']
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting funnel data: {str(e)}")
            return pd.DataFrame()
    
    def get_segment_data(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Get segment data for segment analysis.
        
        Args:
            datasets: Dictionary of DataFrames
        
        Returns:
            DataFrame with segment data
        """
        try:
            if 'revenue_by_segment.csv' in datasets:
                return datasets['revenue_by_segment.csv']
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting segment data: {str(e)}")
            return pd.DataFrame()
