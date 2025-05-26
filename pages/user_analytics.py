import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.cluster import KMeans
from utils.data_loader import DataLoader
from components.header import Header

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserAnalyticsPage:
    """Class for rendering the User Analytics page of the Matiks Gaming Analytics Dashboard."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize the User Analytics page with a data loader.
        
        Args:
            data_loader: DataLoader instance for accessing data
        """
        self.data_loader = data_loader
        self.header = Header()
    
    def render(self, date_range: Dict, filters: Dict):
        """
        Render the User Analytics page.
        
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
                "title": "User Analytics",
                "description": "Detailed analysis of user behavior and engagement patterns."
            }
            self.header.render_full_header(page_info, date_range, filters, kpi_data)
            
            # Create tabs for different sections
            tabs = st.tabs(["User Growth", "Retention Analysis", "User Segmentation", "Behavioral Patterns"])
            
            # User Growth tab
            with tabs[0]:
                self.render_user_growth_section(datasets, date_range, filters)
            
            # Retention Analysis tab
            with tabs[1]:
                self.render_retention_analysis_section(datasets, date_range, filters)
            
            # User Segmentation tab
            with tabs[2]:
                self.render_user_segmentation_section(datasets, date_range, filters)
            
            # Behavioral Patterns tab
            with tabs[3]:
                self.render_behavioral_patterns_section(datasets, date_range, filters)
            
        except Exception as e:
            logger.error(f"Error rendering User Analytics page: {str(e)}")
            st.error(f"An error occurred while rendering the User Analytics page: {str(e)}")
    
    def render_user_growth_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the User Growth section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Growth Trends")
        
        # DAU/WAU/MAU growth chart
        if 'active_users.csv' in datasets:
            df = datasets['active_users.csv']
            
            if not df.empty and 'Date' in df.columns:
                # Filter by date range
                start_date = pd.Timestamp(date_range['start'])
                end_date = pd.Timestamp(date_range['end'])
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                if not df.empty:
                    # Create columns for metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create a line chart for DAU growth
                        if 'DAU' in df.columns:
                            # Calculate rolling average for smoother trend
                            df['DAU_7day_avg'] = df['DAU'].rolling(window=7).mean()
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=df['Date'],
                                y=df['DAU'],
                                mode='lines',
                                name='Daily Active Users',
                                line=dict(color='#1f77b4', width=1),
                                opacity=0.7
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=df['Date'],
                                y=df['DAU_7day_avg'],
                                mode='lines',
                                name='7-Day Average',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            # Calculate trend line
                            x = np.array(range(len(df)))
                            y = df['DAU'].values
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            
                            fig.add_trace(go.Scatter(
                                x=df['Date'],
                                y=p(x),
                                mode='lines',
                                name='Trend',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                title='Daily Active Users (DAU) Trend',
                                xaxis_title='Date',
                                yaxis_title='User Count',
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                ),
                                margin=dict(l=40, r=40, t=40, b=40),
                                hovermode="x unified"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Create a line chart for WAU/MAU growth
                        if 'WAU' in df.columns and 'MAU' in df.columns:
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=df['Date'],
                                y=df['WAU'],
                                mode='lines',
                                name='Weekly Active Users',
                                line=dict(color='#ff7f0e', width=2)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=df['Date'],
                                y=df['MAU'],
                                mode='lines',
                                name='Monthly Active Users',
                                line=dict(color='#2ca02c', width=2)
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                title='Weekly & Monthly Active Users Trend',
                                xaxis_title='Date',
                                yaxis_title='User Count',
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                ),
                                margin=dict(l=40, r=40, t=40, b=40),
                                hovermode="x unified"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # User growth metrics
                    st.subheader("User Growth Metrics")
                    
                    # Calculate growth metrics
                    if len(df) > 1:
                        # DAU growth
                        first_dau = df['DAU'].iloc[0]
                        last_dau = df['DAU'].iloc[-1]
                        dau_growth = ((last_dau - first_dau) / first_dau * 100) if first_dau > 0 else 0
                        
                        # WAU growth
                        first_wau = df['WAU'].iloc[0]
                        last_wau = df['WAU'].iloc[-1]
                        wau_growth = ((last_wau - first_wau) / first_wau * 100) if first_wau > 0 else 0
                        
                        # MAU growth
                        first_mau = df['MAU'].iloc[0]
                        last_mau = df['MAU'].iloc[-1]
                        mau_growth = ((last_mau - first_mau) / first_mau * 100) if first_mau > 0 else 0
                        
                        # Create metrics table
                        metrics_data = {
                            'Metric': ['DAU Growth', 'WAU Growth', 'MAU Growth'],
                            'Start Value': [f"{first_dau:,.0f}", f"{first_wau:,.0f}", f"{first_mau:,.0f}"],
                            'End Value': [f"{last_dau:,.0f}", f"{last_wau:,.0f}", f"{last_mau:,.0f}"],
                            'Growth %': [f"{dau_growth:+.1f}%", f"{wau_growth:+.1f}%", f"{mau_growth:+.1f}%"]
                        }
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Growth visualization
                        growth_data = {
                            'Metric': ['DAU', 'WAU', 'MAU'],
                            'Growth %': [dau_growth, wau_growth, mau_growth]
                        }
                        
                        growth_df = pd.DataFrame(growth_data)
                        
                        fig = px.bar(
                            growth_df,
                            x='Metric',
                            y='Growth %',
                            title='User Growth Percentage',
                            color='Growth %',
                            color_continuous_scale=px.colors.diverging.RdBu,
                            color_continuous_midpoint=0
                        )
                        
                        fig.update_layout(
                            xaxis_title='Metric',
                            yaxis_title='Growth %',
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No user growth data available for the selected date range.")
            else:
                st.info("User growth data is not properly formatted.")
        else:
            st.info("User growth data is not available.")
    
    def render_retention_analysis_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Retention Analysis section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Retention Analysis")
        
        # Cohort retention analysis
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
                    
                    # Calculate average retention by month
                    avg_retention = []
                    for col in retention_cols:
                        avg = heatmap_data[col].mean()
                        avg_retention.append({'Month': col, 'Average Retention': avg})
                    
                    avg_retention_df = pd.DataFrame(avg_retention)
                    
                    # Create line chart for average retention
                    fig = px.line(
                        avg_retention_df,
                        x='Month',
                        y='Average Retention',
                        title='Average Retention by Month',
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
        
        # Stickiness analysis
        if 'active_users.csv' in datasets:
            df = datasets['active_users.csv']
            
            if not df.empty and 'Date' in df.columns and 'Stickiness' in df.columns:
                # Filter by date range
                start_date = pd.Timestamp(date_range['start'])
                end_date = pd.Timestamp(date_range['end'])
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                if not df.empty:
                    st.subheader("User Stickiness Analysis")
                    
                    # Create stickiness trend chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['Stickiness'],
                        mode='lines',
                        name='Stickiness (DAU/MAU)',
                        line=dict(color='#9467bd', width=2),
                        fill='tozeroy'
                    ))
                    
                    # Calculate 7-day moving average
                    df['Stickiness_MA7'] = df['Stickiness'].rolling(window=7).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['Stickiness_MA7'],
                        mode='lines',
                        name='7-Day Moving Average',
                        line=dict(color='#d62728', width=2, dash='dash')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='User Stickiness Trend (DAU/MAU Ratio)',
                        xaxis_title='Date',
                        yaxis_title='Stickiness Ratio',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=40, r=40, t=40, b=40),
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stickiness distribution
                    fig = px.histogram(
                        df,
                        x='Stickiness',
                        nbins=20,
                        title='Stickiness Distribution',
                        color_discrete_sequence=['#9467bd']
                    )
                    
                    fig.update_layout(
                        xaxis_title='Stickiness Ratio',
                        yaxis_title='Frequency',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No stickiness data available for the selected date range.")
            else:
                st.info("Stickiness data is not properly formatted.")
    
    def render_user_segmentation_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the User Segmentation section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Segmentation Analysis")
        
        # Revenue segmentation
        if 'revenue_by_segment.csv' in datasets:
            df = datasets['revenue_by_segment.csv']
            
            if not df.empty and 'Segment' in df.columns:
                # Create columns for metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'User_Count' in df.columns:
                        # Create a pie chart for user distribution by segment
                        fig = px.pie(
                            df,
                            values='User_Count',
                            names='Segment',
                            title='User Distribution by Segment',
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        
                        # Update layout
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=40, b=20),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.2,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Total_Revenue' in df.columns:
                        # Create a pie chart for revenue distribution by segment
                        fig = px.pie(
                            df,
                            values='Total_Revenue',
                            names='Segment',
                            title='Revenue Distribution by Segment',
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        
                        # Update layout
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=40, b=20),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.2,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Segment comparison
                if 'User_Count' in df.columns and 'Total_Revenue' in df.columns and 'Average_Revenue' in df.columns:
                    # Create a bar chart for segment comparison
                    fig = go.Figure()
                    
                    # Add user count bars
                    fig.add_trace(go.Bar(
                        x=df['Segment'],
                        y=df['User_Count'],
                        name='User Count',
                        marker_color='#1f77b4'
                    ))
                    
                    # Add average revenue line
                    fig.add_trace(go.Scatter(
                        x=df['Segment'],
                        y=df['Average_Revenue'],
                        name='Average Revenue',
                        mode='lines+markers',
                        marker=dict(size=10),
                        line=dict(color='#d62728', width=2),
                        yaxis='y2'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='User Count vs. Average Revenue by Segment',
                        xaxis_title='Segment',
                        yaxis=dict(
                            title='User Count',
                            side='left'
                        ),
                        yaxis2=dict(
                            title='Average Revenue ($)',
                            side='right',
                            overlaying='y',
                            showgrid=False
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(l=40, r=40, t=40, b=40),
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a table with segment metrics
                    segment_metrics = df.copy()
                    
                    # Calculate revenue percentage
                    total_revenue = segment_metrics['Total_Revenue'].sum()
                    segment_metrics['Revenue %'] = segment_metrics['Total_Revenue'] / total_revenue * 100 if total_revenue > 0 else 0
                    
                    # Calculate user percentage
                    total_users = segment_metrics['User_Count'].sum()
                    segment_metrics['User %'] = segment_metrics['User_Count'] / total_users * 100 if total_users > 0 else 0
                    
                    # Format columns
                    segment_metrics['Total_Revenue'] = segment_metrics['Total_Revenue'].apply(lambda x: f"${x:,.2f}")
                    segment_metrics['Average_Revenue'] = segment_metrics['Average_Revenue'].apply(lambda x: f"${x:,.2f}")
                    segment_metrics['Revenue %'] = segment_metrics['Revenue %'].apply(lambda x: f"{x:.1f}%")
                    segment_metrics['User %'] = segment_metrics['User %'].apply(lambda x: f"{x:.1f}%")
                    
                    # Rename columns for display
                    segment_metrics = segment_metrics.rename(columns={
                        'Segment': 'Segment',
                        'User_Count': 'User Count',
                        'Total_Revenue': 'Total Revenue',
                        'Average_Revenue': 'ARPU',
                        'Revenue %': 'Revenue %',
                        'User %': 'User %'
                    })
                    
                    # Display table
                    st.dataframe(segment_metrics, use_container_width=True)
            else:
                st.info("Segment data is not properly formatted.")
        else:
            st.info("Segment data is not available.")
        
        # User clustering (bonus feature)
        st.subheader("User Clustering Analysis")
        
        # Check if we have the necessary data for clustering
        if 'active_users.csv' in datasets and 'revenue_by_segment.csv' in datasets:
            st.markdown("### Frequency vs. Revenue Clustering")
            
            # Create sample data for clustering demonstration
            # In a real implementation, this would use actual user-level data
            
            # Generate sample data based on segment information
            if 'revenue_by_segment.csv' in datasets:
                segment_df = datasets['revenue_by_segment.csv']
                
                if not segment_df.empty and 'Segment' in segment_df.columns and 'User_Count' in segment_df.columns and 'Average_Revenue' in segment_df.columns:
                    # Create synthetic user data based on segments
                    user_data = []
                    
                    for _, row in segment_df.iterrows():
                        segment = row['Segment']
                        user_count = int(row['User_Count'])
                        avg_revenue = float(row['Average_Revenue'])
                        
                        # Set frequency range based on segment
                        if segment == 'High spenders':
                            freq_mean, freq_std = 25, 5
                        elif segment == 'Medium spenders':
                            freq_mean, freq_std = 15, 5
                        elif segment == 'Low spenders':
                            freq_mean, freq_std = 8, 3
                        else:  # Non-spenders
                            freq_mean, freq_std = 3, 2
                        
                        # Generate users for this segment
                        for i in range(user_count):
                            # Add some randomness to the revenue
                            revenue = max(0, np.random.normal(avg_revenue, avg_revenue * 0.2))
                            frequency = max(1, np.random.normal(freq_mean, freq_std))
                            
                            user_data.append({
                                'User ID': f"{segment.replace(' ', '')}_User_{i}",
                                'Segment': segment,
                                'Frequency': frequency,
                                'Revenue': revenue
                            })
                    
                    # Create DataFrame
                    user_df = pd.DataFrame(user_data)
                    
                    # Perform K-means clustering
                    X = user_df[['Frequency', 'Revenue']].values
                    
                    # Determine optimal number of clusters (simplified)
                    n_clusters = 4  # We could use elbow method or silhouette score in a real implementation
                    
                    # Fit K-means
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    user_df['Cluster'] = kmeans.fit_predict(X)
                    
                    # Create scatter plot with clusters
                    fig = px.scatter(
                        user_df,
                        x='Frequency',
                        y='Revenue',
                        color='Cluster',
                        hover_data=['User ID', 'Segment'],
                        title='User Clustering: Frequency vs. Revenue',
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    # Add cluster centers
                    centers = kmeans.cluster_centers_
                    
                    for i, center in enumerate(centers):
                        fig.add_trace(go.Scatter(
                            x=[center[0]],
                            y=[center[1]],
                            mode='markers',
                            marker=dict(
                                color='black',
                                size=15,
                                symbol='x'
                            ),
                            name=f'Cluster {i} Center'
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title='Usage Frequency (Days per Month)',
                        yaxis_title='Revenue ($)',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster analysis
                    cluster_analysis = user_df.groupby('Cluster').agg({
                        'User ID': 'count',
                        'Frequency': 'mean',
                        'Revenue': 'mean'
                    }).reset_index()
                    
                    cluster_analysis = cluster_analysis.rename(columns={
                        'User ID': 'User Count',
                        'Frequency': 'Avg. Frequency',
                        'Revenue': 'Avg. Revenue'
                    })
                    
                    # Format columns
                    cluster_analysis['Avg. Frequency'] = cluster_analysis['Avg. Frequency'].apply(lambda x: f"{x:.1f}")
                    cluster_analysis['Avg. Revenue'] = cluster_analysis['Avg. Revenue'].apply(lambda x: f"${x:.2f}")
                    
                    # Display cluster analysis
                    st.dataframe(cluster_analysis, use_container_width=True)
                    
                    # Cluster interpretation
                    st.markdown("### Cluster Interpretation")
                    
                    # Calculate cluster characteristics
                    cluster_chars = []
                    
                    for cluster in range(n_clusters):
                        cluster_data = user_df[user_df['Cluster'] == cluster]
                        
                        avg_freq = cluster_data['Frequency'].mean()
                        avg_rev = cluster_data['Revenue'].mean()
                        user_count = len(cluster_data)
                        total_users = len(user_df)
                        user_pct = user_count / total_users * 100 if total_users > 0 else 0
                        
                        # Determine cluster type
                        if avg_freq > 20 and avg_rev > 50:
                            cluster_type = "High-value Power Users"
                            recommendation = "Offer exclusive content and premium features to maintain engagement."
                        elif avg_freq > 15 and avg_rev > 20:
                            cluster_type = "Regular Spenders"
                            recommendation = "Target with bundle offers and loyalty rewards to increase spending."
                        elif avg_freq > 10 and avg_rev < 20:
                            cluster_type = "Active Non-spenders"
                            recommendation = "Introduce targeted promotions to convert to paying users."
                        else:
                            cluster_type = "Casual Users"
                            recommendation = "Improve onboarding and early engagement to increase retention."
                        
                        cluster_chars.append({
                            'Cluster': cluster,
                            'Type': cluster_type,
                            'User %': f"{user_pct:.1f}%",
                            'Avg. Frequency': f"{avg_freq:.1f}",
                            'Avg. Revenue': f"${avg_rev:.2f}",
                            'Recommendation': recommendation
                        })
                    
                    # Display cluster characteristics
                    cluster_chars_df = pd.DataFrame(cluster_chars)
                    st.dataframe(cluster_chars_df, use_container_width=True)
                else:
                    st.info("Segment data is not properly formatted for clustering analysis.")
            else:
                st.info("Segment data is not available for clustering analysis.")
        else:
            st.info("Required data for clustering analysis is not available.")
    
    def render_behavioral_patterns_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Behavioral Patterns section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Behavioral Patterns")
        
        # Game mode preferences
        if 'game_mode_distribution.csv' in datasets:
            df = datasets['game_mode_distribution.csv']
            
            if not df.empty and 'Game_Mode' in df.columns and 'User_Count' in df.columns:
                # Create a bar chart for game mode distribution
                fig = px.bar(
                    df,
                    x='Game_Mode',
                    y='User_Count',
                    title='User Distribution by Game Mode',
                    color='Game_Mode',
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Game Mode',
                    yaxis_title='User Count',
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate percentages
                total_users = df['User_Count'].sum()
                df['Percentage'] = df['User_Count'] / total_users * 100 if total_users > 0 else 0
                
                # Format for display
                display_df = df.copy()
                display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.1f}%")
                
                # Display table
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("Game mode distribution data is not properly formatted.")
        else:
            st.info("Game mode distribution data is not available.")
        
        # Device preferences
        if 'device_distribution.csv' in datasets:
            df = datasets['device_distribution.csv']
            
            if not df.empty and 'Device_Type' in df.columns and 'User_Count' in df.columns:
                # Create a bar chart for device distribution
                fig = px.bar(
                    df,
                    x='Device_Type',
                    y='User_Count',
                    title='User Distribution by Device',
                    color='Device_Type',
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Device Type',
                    yaxis_title='User Count',
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate percentages
                total_users = df['User_Count'].sum()
                df['Percentage'] = df['User_Count'] / total_users * 100 if total_users > 0 else 0
                
                # Format for display
                display_df = df.copy()
                display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.1f}%")
                
                # Display table
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("Device distribution data is not properly formatted.")
        else:
            st.info("Device distribution data is not available.")
        
        # User journey funnel
        if 'funnel_analysis.csv' in datasets:
            df = datasets['funnel_analysis.csv']
            
            if not df.empty and 'Stage' in df.columns and 'Count' in df.columns:
                st.subheader("User Journey Analysis")
                
                # Create a funnel chart
                fig = go.Figure(go.Funnel(
                    y=df['Stage'],
                    x=df['Count'],
                    textinfo="value+percent initial",
                    marker=dict(color=px.colors.sequential.Viridis)
                ))
                
                # Update layout
                fig.update_layout(
                    title='User Journey Funnel',
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate drop-off rates
                if len(df) > 1:
                    drop_off_data = []
                    
                    for i in range(len(df) - 1):
                        current_stage = df['Stage'].iloc[i]
                        next_stage = df['Stage'].iloc[i + 1]
                        current_count = df['Count'].iloc[i]
                        next_count = df['Count'].iloc[i + 1]
                        
                        drop_off = current_count - next_count
                        drop_off_rate = (drop_off / current_count * 100) if current_count > 0 else 0
                        
                        drop_off_data.append({
                            'From Stage': current_stage,
                            'To Stage': next_stage,
                            'Drop-off Count': drop_off,
                            'Drop-off Rate': f"{drop_off_rate:.1f}%"
                        })
                    
                    # Find the biggest drop-off
                    biggest_drop_idx = 0
                    biggest_drop_rate = 0
                    
                    for i, data in enumerate(drop_off_data):
                        rate = float(data['Drop-off Rate'].replace('%', ''))
                        if rate > biggest_drop_rate:
                            biggest_drop_rate = rate
                            biggest_drop_idx = i
                    
                    # Highlight the biggest drop-off
                    if drop_off_data:
                        biggest_drop = drop_off_data[biggest_drop_idx]
                        st.markdown(f"**Biggest Drop-off:** {biggest_drop['Drop-off Rate']} between **{biggest_drop['From Stage']}** and **{biggest_drop['To Stage']}**")
                    
                    # Display drop-off table
                    st.dataframe(pd.DataFrame(drop_off_data), use_container_width=True)
            else:
                st.info("Funnel data is not properly formatted.")
        else:
            st.info("Funnel data is not available.")
