import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import logging
from utils.data_loader import DataLoader
from components.header import Header

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CohortAnalysisPage:
    """Class for rendering the Cohort Analysis page of the Matiks Gaming Analytics Dashboard."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize the Cohort Analysis page with a data loader.
        
        Args:
            data_loader: DataLoader instance for accessing data
        """
        self.data_loader = data_loader
        self.header = Header()
    
    def render(self, date_range: Dict, filters: Dict):
        """
        Render the Cohort Analysis page.
        
        Args:
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        try:
            # Load all datasets
            datasets = self.data_loader.load_all_datasets()
            
            # Calculate KPIs
            kpi_data = self.data_loader.calculate_kpis(datasets, date_range)
            
            # Render page header
            page_info = {
                "title": "Cohort Analysis",
                "description": "Analysis of user cohorts based on signup/first seen date."
            }
            self.header.render_full_header(page_info, date_range, filters, kpi_data)
            
            # Create tabs for different sections
            tabs = st.tabs(["Retention Heatmap", "Cohort Performance", "Cohort Comparison", "Insights"])
            
            # Retention Heatmap tab
            with tabs[0]:
                self.render_retention_heatmap_section(datasets, date_range, filters)
            
            # Cohort Performance tab
            with tabs[1]:
                self.render_cohort_performance_section(datasets, date_range, filters)
            
            # Cohort Comparison tab
            with tabs[2]:
                self.render_cohort_comparison_section(datasets, date_range, filters)
            
            # Insights tab
            with tabs[3]:
                self.render_insights_section(datasets, date_range, filters)
            
        except Exception as e:
            logger.error(f"Error rendering Cohort Analysis page: {str(e)}")
            st.error(f"An error occurred while rendering the Cohort Analysis page: {str(e)}")
    
    def render_retention_heatmap_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Retention Heatmap section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Cohort Retention Heatmap")
        
        # Cohort retention heatmap
        if 'cohort_retention.csv' in datasets:
            df = datasets['cohort_retention.csv']
            
            if not df.empty and 'Cohort_Month' in df.columns:
                # Find columns with retention data (numeric columns except Cohort_Month)
                retention_cols = [col for col in df.columns if col != 'Cohort_Month' and pd.api.types.is_numeric_dtype(df[col])]
                
                if retention_cols:
                    # Prepare data for heatmap
                    heatmap_data = df[['Cohort_Month'] + retention_cols].copy()
                    
                    # Convert to numeric values, replacing NaN with 0
                    for col in retention_cols:
                        heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce').fillna(0)
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data[retention_cols].values,
                        x=[f"Month {col}" if col.isdigit() else col for col in retention_cols],
                        y=heatmap_data['Cohort_Month'],
                        colorscale='Viridis',
                        colorbar=dict(title='Retention %')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Cohort Retention Heatmap',
                        xaxis_title='Months Since Acquisition',
                        yaxis_title='Cohort',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Heatmap interpretation
                    st.markdown("""
                    **Heatmap Interpretation:**
                    - Each row represents a cohort of users who joined in a specific month
                    - Each column represents the number of months since acquisition
                    - The values show the percentage of users from the original cohort who are still active
                    - Darker colors indicate higher retention rates
                    """)
                    
                    # Interactive cohort selector
                    st.subheader("Cohort Retention Curve")
                    
                    # Create a dropdown to select a specific cohort
                    selected_cohort = st.selectbox(
                        "Select Cohort",
                        options=heatmap_data['Cohort_Month'].tolist(),
                        index=0
                    )
                    
                    # Filter data for the selected cohort
                    selected_cohort_data = heatmap_data[heatmap_data['Cohort_Month'] == selected_cohort]
                    
                    if not selected_cohort_data.empty:
                        # Prepare data for line chart
                        retention_data = []
                        
                        for col in retention_cols:
                            month = int(col) if col.isdigit() else 0
                            retention = selected_cohort_data[col].values[0]
                            retention_data.append({
                                'Month': month,
                                'Retention': retention
                            })
                        
                        retention_df = pd.DataFrame(retention_data)
                        retention_df = retention_df.sort_values('Month')
                        
                        # Create line chart
                        fig = px.line(
                            retention_df,
                            x='Month',
                            y='Retention',
                            title=f'Retention Curve for {selected_cohort} Cohort',
                            markers=True
                        )
                        
                        # Update layout
                        fig.update_layout(
                            xaxis_title='Months Since Acquisition',
                            yaxis_title='Retention %',
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No retention data available in the cohort analysis.")
            else:
                st.info("Cohort data is not properly formatted.")
        else:
            st.info("Cohort data is not available.")
    
    def render_cohort_performance_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Cohort Performance section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Cohort Performance Analysis")
        
        # Cohort performance metrics
        if 'cohort_retention.csv' in datasets:
            df = datasets['cohort_retention.csv']
            
            if not df.empty and 'Cohort_Month' in df.columns:
                # Find columns with retention data (numeric columns except Cohort_Month)
                retention_cols = [col for col in df.columns if col != 'Cohort_Month' and pd.api.types.is_numeric_dtype(df[col])]
                
                if retention_cols:
                    # Prepare data for analysis
                    cohort_data = df[['Cohort_Month'] + retention_cols].copy()
                    
                    # Convert to numeric values, replacing NaN with 0
                    for col in retention_cols:
                        cohort_data[col] = pd.to_numeric(cohort_data[col], errors='coerce').fillna(0)
                    
                    # Calculate key retention metrics for each cohort
                    cohort_metrics = []
                    
                    for _, row in cohort_data.iterrows():
                        cohort = row['Cohort_Month']
                        
                        # Month 1 retention (early retention)
                        month_1_col = '1' if '1' in retention_cols else retention_cols[min(1, len(retention_cols) - 1)]
                        month_1_retention = row[month_1_col]
                        
                        # Month 3 retention (mid-term retention)
                        month_3_col = '3' if '3' in retention_cols else retention_cols[min(3, len(retention_cols) - 1)]
                        month_3_retention = row[month_3_col] if month_3_col in row.index else None
                        
                        # Month 6 retention (long-term retention)
                        month_6_col = '6' if '6' in retention_cols else retention_cols[min(6, len(retention_cols) - 1)]
                        month_6_retention = row[month_6_col] if month_6_col in row.index else None
                        
                        # Calculate retention decay (slope)
                        retention_values = [row[col] for col in retention_cols if col in row.index]
                        retention_decay = (retention_values[0] - retention_values[-1]) / len(retention_values) if len(retention_values) > 1 else 0
                        
                        cohort_metrics.append({
                            'Cohort': cohort,
                            'Month 1 Retention': month_1_retention,
                            'Month 3 Retention': month_3_retention if month_3_retention is not None else 'N/A',
                            'Month 6 Retention': month_6_retention if month_6_retention is not None else 'N/A',
                            'Retention Decay': retention_decay
                        })
                    
                    # Create DataFrame
                    metrics_df = pd.DataFrame(cohort_metrics)
                    
                    # Format retention values
                    for col in ['Month 1 Retention', 'Month 3 Retention', 'Month 6 Retention']:
                        if col in metrics_df.columns:
                            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
                    
                    if 'Retention Decay' in metrics_df.columns:
                        metrics_df['Retention Decay'] = metrics_df['Retention Decay'].apply(lambda x: f"{x:.2f}% per month" if isinstance(x, (int, float)) else x)
                    
                    # Display metrics
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Calculate average retention by month
                    avg_retention = []
                    for col in retention_cols:
                        avg = cohort_data[col].mean()
                        avg_retention.append({'Month': col, 'Average Retention': avg})
                    
                    avg_retention_df = pd.DataFrame(avg_retention)
                    
                    # Create line chart for average retention
                    fig = px.line(
                        avg_retention_df,
                        x='Month',
                        y='Average Retention',
                        title='Average Retention by Month Across All Cohorts',
                        markers=True
                    )
                    
                    fig.update_layout(
                        xaxis_title='Month Since Acquisition',
                        yaxis_title='Average Retention %',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Identify best and worst cohorts
                    if len(retention_cols) > 0:
                        first_retention_col = retention_cols[0]
                        if len(df) > 0:
                            best_cohort_idx = df[first_retention_col].fillna(0).idxmax()
                            worst_cohort_idx = df[first_retention_col].fillna(0).idxmin()
                            
                            best_cohort = df.loc[best_cohort_idx, 'Cohort_Month'] if best_cohort_idx is not None else None
                            worst_cohort = df.loc[worst_cohort_idx, 'Cohort_Month'] if worst_cohort_idx is not None else None
                            
                            if best_cohort and worst_cohort:
                                st.markdown(f"**Best Performing Cohort:** {best_cohort}")
                                st.markdown(f"**Worst Performing Cohort:** {worst_cohort}")
                else:
                    st.info("No retention data available in the cohort analysis.")
            else:
                st.info("Cohort data is not properly formatted.")
        else:
            st.info("Cohort data is not available.")
    
    def render_cohort_comparison_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Cohort Comparison section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Cohort Comparison Analysis")
        
        # Cohort comparison
        if 'cohort_retention.csv' in datasets:
            df = datasets['cohort_retention.csv']
            
            if not df.empty and 'Cohort_Month' in df.columns:
                # Find columns with retention data (numeric columns except Cohort_Month)
                retention_cols = [col for col in df.columns if col != 'Cohort_Month' and pd.api.types.is_numeric_dtype(df[col])]
                
                if retention_cols:
                    # Create multi-select for cohorts
                    selected_cohorts = st.multiselect(
                        "Select Cohorts to Compare",
                        options=df['Cohort_Month'].tolist(),
                        default=df['Cohort_Month'].tolist()[:3]  # Default to first 3 cohorts
                    )
                    
                    if selected_cohorts:
                        # Filter data for selected cohorts
                        selected_data = df[df['Cohort_Month'].isin(selected_cohorts)]
                        
                        # Prepare data for line chart
                        comparison_data = []
                        
                        for _, row in selected_data.iterrows():
                            cohort = row['Cohort_Month']
                            
                            for col in retention_cols:
                                month = int(col) if col.isdigit() else 0
                                retention = row[col]
                                comparison_data.append({
                                    'Cohort': cohort,
                                    'Month': month,
                                    'Retention': retention
                                })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Create line chart
                        fig = px.line(
                            comparison_df,
                            x='Month',
                            y='Retention',
                            color='Cohort',
                            title='Cohort Retention Comparison',
                            markers=True
                        )
                        
                        # Update layout
                        fig.update_layout(
                            xaxis_title='Months Since Acquisition',
                            yaxis_title='Retention %',
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Month 1 retention comparison
                        if '1' in retention_cols:
                            month_1_data = selected_data[['Cohort_Month', '1']].copy()
                            month_1_data = month_1_data.rename(columns={'Cohort_Month': 'Cohort', '1': 'Month 1 Retention'})
                            
                            # Create bar chart
                            fig = px.bar(
                                month_1_data,
                                x='Cohort',
                                y='Month 1 Retention',
                                title='Month 1 Retention by Cohort',
                                color='Cohort'
                            )
                            
                            # Update layout
                            fig.update_layout(
                                xaxis_title='Cohort',
                                yaxis_title='Month 1 Retention %',
                                margin=dict(l=40, r=40, t=40, b=40)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Please select at least one cohort to compare.")
                else:
                    st.info("No retention data available in the cohort analysis.")
            else:
                st.info("Cohort data is not properly formatted.")
        else:
            st.info("Cohort data is not available.")
    
    def render_insights_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Insights section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Cohort Analysis Insights")
        
        # Cohort insights
        if 'cohort_retention.csv' in datasets:
            df = datasets['cohort_retention.csv']
            
            if not df.empty and 'Cohort_Month' in df.columns:
                # Find columns with retention data (numeric columns except Cohort_Month)
                retention_cols = [col for col in df.columns if col != 'Cohort_Month' and pd.api.types.is_numeric_dtype(df[col])]
                
                if retention_cols:
                    # Prepare data for analysis
                    cohort_data = df[['Cohort_Month'] + retention_cols].copy()
                    
                    # Convert to numeric values, replacing NaN with 0
                    for col in retention_cols:
                        cohort_data[col] = pd.to_numeric(cohort_data[col], errors='coerce').fillna(0)
                    
                    # Generate insights
                    insights = []
                    
                    try:
                        # Calculate average retention by month
                        avg_retention = {}
                        for col in retention_cols:
                            avg_retention[col] = cohort_data[col].mean()
                        
                        # Month 1 retention trend
                        if '1' in retention_cols and len(cohort_data) > 1:
                            first_month_1 = cohort_data['1'].iloc[0]
                            last_month_1 = cohort_data['1'].iloc[-1]
                            month_1_change = ((last_month_1 - first_month_1) / first_month_1 * 100) if first_month_1 > 0 else 0
                            
                            if month_1_change > 10:
                                insights.append(f"📈 **Improving early retention:** Month 1 retention has increased by {month_1_change:.1f}% from {first_month_1:.1f}% to {last_month_1:.1f}%.")
                            elif month_1_change < -10:
                                insights.append(f"📉 **Declining early retention:** Month 1 retention has decreased by {abs(month_1_change):.1f}% from {first_month_1:.1f}% to {last_month_1:.1f}%.")
                        
                        # Identify best and worst cohorts
                        if '1' in retention_cols and len(cohort_data) > 0:
                            best_cohort_idx = cohort_data['1'].idxmax()
                            worst_cohort_idx = cohort_data['1'].idxmin()
                            
                            best_cohort = cohort_data.loc[best_cohort_idx, 'Cohort_Month']
                            best_retention = cohort_data.loc[best_cohort_idx, '1']
                            
                            worst_cohort = cohort_data.loc[worst_cohort_idx, 'Cohort_Month']
                            worst_retention = cohort_data.loc[worst_cohort_idx, '1']
                            
                            insights.append(f"🌟 **Best performing cohort:** {best_cohort} with {best_retention:.1f}% Month 1 retention.")
                            insights.append(f"⚠️ **Worst performing cohort:** {worst_cohort} with {worst_retention:.1f}% Month 1 retention.")
                        
                        # Retention decay analysis
                        if len(retention_cols) > 1:
                            first_month = retention_cols[0]
                            last_month = retention_cols[-1]
                            
                            avg_first = avg_retention[first_month]
                            avg_last = avg_retention[last_month]
                            
                            retention_decay = (avg_first - avg_last) / len(retention_cols)
                            
                            if retention_decay > 15:
                                insights.append(f"📉 **High retention decay:** Average retention drops by {retention_decay:.1f}% per month, indicating significant user drop-off over time.")
                            elif retention_decay < 5:
                                insights.append(f"📊 **Low retention decay:** Average retention drops by only {retention_decay:.1f}% per month, indicating strong long-term engagement.")
                        
                        # Recent cohort performance
                        if len(cohort_data) > 1 and '1' in retention_cols:
                            recent_cohort = cohort_data.iloc[-1]
                            recent_cohort_name = recent_cohort['Cohort_Month']
                            recent_cohort_retention = recent_cohort['1']
                            
                            avg_retention_month_1 = avg_retention['1']
                            
                            if recent_cohort_retention > avg_retention_month_1 * 1.1:
                                insights.append(f"🚀 **Strong recent cohort:** The {recent_cohort_name} cohort has {recent_cohort_retention:.1f}% Month 1 retention, which is {(recent_cohort_retention - avg_retention_month_1):.1f}% above average.")
                            elif recent_cohort_retention < avg_retention_month_1 * 0.9:
                                insights.append(f"⚠️ **Weak recent cohort:** The {recent_cohort_name} cohort has {recent_cohort_retention:.1f}% Month 1 retention, which is {(avg_retention_month_1 - recent_cohort_retention):.1f}% below average.")
                    
                    except Exception as e:
                        logger.error(f"Error generating cohort insights: {str(e)}")
                        insights.append("Unable to generate cohort insights due to data processing error.")
                    
                    # Display insights
                    if insights:
                        for insight in insights:
                            st.markdown(insight)
                    else:
                        st.info("No cohort insights available for the selected data.")
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    
                    # Generate recommendations based on insights
                    recommendations = []
                    
                    try:
                        # Month 1 retention recommendations
                        if '1' in retention_cols:
                            avg_month_1 = avg_retention['1']
                            
                            if avg_month_1 < 30:
                                recommendations.append("🔍 **Improve onboarding:** Enhance the onboarding experience to better engage new users and improve early retention.")
                                recommendations.append("🎯 **Early value delivery:** Ensure users experience value within the first few sessions to increase early retention.")
                            
                        # Retention decay recommendations
                        if len(retention_cols) > 1:
                            first_month = retention_cols[0]
                            last_month = retention_cols[-1]
                            
                            avg_first = avg_retention[first_month]
                            avg_last = avg_retention[last_month]
                            
                            retention_decay = (avg_first - avg_last) / len(retention_cols)
                            
                            if retention_decay > 10:
                                recommendations.append("🔄 **Implement re-engagement campaigns:** Target users who haven't been active for specific periods with personalized offers.")
                                recommendations.append("🎮 **Add long-term progression systems:** Implement features that encourage long-term engagement and progression.")
                        
                        # Recent cohort recommendations
                        if len(cohort_data) > 1 and '1' in retention_cols:
                            recent_cohort = cohort_data.iloc[-1]
                            recent_cohort_retention = recent_cohort['1']
                            
                            avg_retention_month_1 = avg_retention['1']
                            
                            if recent_cohort_retention < avg_retention_month_1:
                                recommendations.append("🔍 **Analyze recent changes:** Investigate recent product or marketing changes that might have affected new user retention.")
                                recommendations.append("🧪 **A/B test onboarding:** Test different onboarding experiences to identify optimal approaches for new users.")
                        
                        # Add generic recommendations if we don't have enough specific ones
                        if len(recommendations) < 2:
                            recommendations.append("📊 **Segment cohort analysis:** Break down cohort analysis by user segments to identify specific retention patterns.")
                            recommendations.append("🔄 **Regular cohort review:** Establish a regular review process for cohort performance to identify trends early.")
                    
                    except Exception as e:
                        logger.error(f"Error generating cohort recommendations: {str(e)}")
                        recommendations.append("Unable to generate cohort recommendations due to data processing error.")
                    
                    # Display recommendations
                    if recommendations:
                        for recommendation in recommendations:
                            st.markdown(recommendation)
                    else:
                        st.info("No recommendations available for the selected data.")
                else:
                    st.info("No retention data available in the cohort analysis.")
            else:
                st.info("Cohort data is not properly formatted.")
        else:
            st.info("Cohort data is not available.")
