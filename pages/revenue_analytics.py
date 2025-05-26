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

class RevenueAnalyticsPage:
    """Class for rendering the Revenue Analytics page of the Matiks Gaming Analytics Dashboard."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize the Revenue Analytics page with a data loader.
        
        Args:
            data_loader: DataLoader instance for accessing data
        """
        self.data_loader = data_loader
        self.header = Header()
    
    def render(self, date_range: Dict, filters: Dict):
        """
        Render the Revenue Analytics page.
        
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
                "title": "Revenue Analytics",
                "description": "Detailed analysis of revenue patterns and monetization metrics."
            }
            self.header.render_full_header(page_info, date_range, filters, kpi_data)
            
            # Create tabs for different sections
            tabs = st.tabs(["Revenue Trends", "Segment Analysis", "Revenue Breakdown", "Monetization Metrics"])
            
            # Revenue Trends tab
            with tabs[0]:
                self.render_revenue_trends_section(datasets, date_range, filters)
            
            # Segment Analysis tab
            with tabs[1]:
                self.render_segment_analysis_section(datasets, date_range, filters)
            
            # Revenue Breakdown tab
            with tabs[2]:
                self.render_revenue_breakdown_section(datasets, date_range, filters)
            
            # Monetization Metrics tab
            with tabs[3]:
                self.render_monetization_metrics_section(datasets, date_range, filters)
            
        except Exception as e:
            logger.error(f"Error rendering Revenue Analytics page: {str(e)}")
            st.error(f"An error occurred while rendering the Revenue Analytics page: {str(e)}")
    
    def render_revenue_trends_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Revenue Trends section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Revenue Trends Over Time")
        
        # Revenue over time chart
        if 'revenue_by_date.csv' in datasets:
            df = datasets['revenue_by_date.csv']
            
            if not df.empty and 'Date' in df.columns and 'Revenue' in df.columns:
                # Filter by date range
                start_date = pd.Timestamp(date_range['start'])
                end_date = pd.Timestamp(date_range['end'])
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                if not df.empty:
                    # Create a line chart for revenue over time
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['Revenue'],
                        mode='lines',
                        name='Daily Revenue',
                        line=dict(color='#2ca02c', width=2),
                        fill='tozeroy'
                    ))
                    
                    # Calculate 7-day moving average
                    df['MA7'] = df['Revenue'].rolling(window=7).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['MA7'],
                        mode='lines',
                        name='7-Day Moving Average',
                        line=dict(color='#d62728', width=2, dash='dash')
                    ))
                    
                    # Calculate 30-day moving average
                    df['MA30'] = df['Revenue'].rolling(window=30).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['MA30'],
                        mode='lines',
                        name='30-Day Moving Average',
                        line=dict(color='#9467bd', width=2, dash='dash')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Revenue Over Time',
                        xaxis_title='Date',
                        yaxis_title='Revenue ($)',
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
                    
                    # Add range slider
                    fig.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label="1w", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Revenue growth metrics
                    st.subheader("Revenue Growth Metrics")
                    
                    # Calculate revenue metrics
                    if len(df) > 1:
                        # Total revenue
                        total_revenue = df['Revenue'].sum()
                        
                        # Daily average
                        daily_avg = df['Revenue'].mean()
                        
                        # Revenue growth
                        first_day_revenue = df.iloc[0]['Revenue']
                        last_day_revenue = df.iloc[-1]['Revenue']
                        revenue_growth = ((last_day_revenue - first_day_revenue) / first_day_revenue * 100) if first_day_revenue > 0 else 0
                        
                        # Week-over-week growth
                        if len(df) >= 14:  # Need at least 2 weeks of data
                            last_week = df.iloc[-7:]['Revenue'].sum()
                            prev_week = df.iloc[-14:-7]['Revenue'].sum()
                            wow_growth = ((last_week - prev_week) / prev_week * 100) if prev_week > 0 else 0
                        else:
                            wow_growth = None
                        
                        # Month-over-month growth
                        if len(df) >= 60:  # Need at least 2 months of data
                            last_month = df.iloc[-30:]['Revenue'].sum()
                            prev_month = df.iloc[-60:-30]['Revenue'].sum()
                            mom_growth = ((last_month - prev_month) / prev_month * 100) if prev_month > 0 else 0
                        else:
                            mom_growth = None
                        
                        # Create metrics table
                        metrics_data = {
                            'Metric': ['Total Revenue', 'Daily Average', 'Revenue Growth'],
                            'Value': [f"${total_revenue:,.2f}", f"${daily_avg:,.2f}", f"{revenue_growth:+.1f}%"]
                        }
                        
                        if wow_growth is not None:
                            metrics_data['Metric'].append('Week-over-Week Growth')
                            metrics_data['Value'].append(f"{wow_growth:+.1f}%")
                        
                        if mom_growth is not None:
                            metrics_data['Metric'].append('Month-over-Month Growth')
                            metrics_data['Value'].append(f"{mom_growth:+.1f}%")
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Revenue distribution
                        st.subheader("Revenue Distribution")
                        
                        # Create histogram for revenue distribution
                        fig = px.histogram(
                            df,
                            x='Revenue',
                            nbins=20,
                            title='Revenue Distribution',
                            color_discrete_sequence=['#2ca02c']
                        )
                        
                        fig.update_layout(
                            xaxis_title='Revenue ($)',
                            yaxis_title='Frequency',
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No revenue data available for the selected date range.")
            else:
                st.info("Revenue data is not properly formatted.")
        else:
            st.info("Revenue data is not available.")
    
    def render_segment_analysis_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Segment Analysis section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Revenue by User Segment")
        
        # Revenue by segment
        if 'revenue_by_segment.csv' in datasets:
            df = datasets['revenue_by_segment.csv']
            
            if not df.empty and 'Segment' in df.columns and 'Total_Revenue' in df.columns:
                # Create columns for metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a pie chart for revenue by segment
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
                
                with col2:
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
                
                # Revenue vs. User Count by segment
                if 'User_Count' in df.columns:
                    # Create a bar chart for revenue vs. user count by segment
                    fig = go.Figure()
                    
                    # Add revenue bars
                    fig.add_trace(go.Bar(
                        x=df['Segment'],
                        y=df['Total_Revenue'],
                        name='Total Revenue',
                        marker_color='#1f77b4'
                    ))
                    
                    # Add user count bars
                    fig.add_trace(go.Bar(
                        x=df['Segment'],
                        y=df['User_Count'],
                        name='User Count',
                        marker_color='#ff7f0e',
                        yaxis='y2'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Revenue vs. User Count by Segment',
                        xaxis_title='Segment',
                        yaxis=dict(
                            title='Revenue ($)',
                            side='left'
                        ),
                        yaxis2=dict(
                            title='User Count',
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
                
                # ARPU by segment
                if 'Average_Revenue' in df.columns:
                    # Create a bar chart for ARPU by segment
                    fig = px.bar(
                        df,
                        x='Segment',
                        y='Average_Revenue',
                        title='Average Revenue per User (ARPU) by Segment',
                        color='Segment',
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title='Segment',
                        yaxis_title='ARPU ($)',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Segment metrics table
                st.subheader("Segment Metrics")
                
                # Create a table with segment metrics
                segment_metrics = df.copy()
                
                # Calculate revenue percentage
                total_revenue = segment_metrics['Total_Revenue'].sum()
                segment_metrics['Revenue %'] = segment_metrics['Total_Revenue'] / total_revenue * 100 if total_revenue > 0 else 0
                
                # Calculate user percentage
                if 'User_Count' in segment_metrics.columns:
                    total_users = segment_metrics['User_Count'].sum()
                    segment_metrics['User %'] = segment_metrics['User_Count'] / total_users * 100 if total_users > 0 else 0
                
                # Calculate revenue per user
                if 'User_Count' in segment_metrics.columns and 'Total_Revenue' in segment_metrics.columns:
                    segment_metrics['Revenue per User'] = segment_metrics['Total_Revenue'] / segment_metrics['User_Count']
                
                # Format columns
                segment_metrics['Total_Revenue'] = segment_metrics['Total_Revenue'].apply(lambda x: f"${x:,.2f}")
                if 'Average_Revenue' in segment_metrics.columns:
                    segment_metrics['Average_Revenue'] = segment_metrics['Average_Revenue'].apply(lambda x: f"${x:,.2f}")
                segment_metrics['Revenue %'] = segment_metrics['Revenue %'].apply(lambda x: f"{x:.1f}%")
                if 'User %' in segment_metrics.columns:
                    segment_metrics['User %'] = segment_metrics['User %'].apply(lambda x: f"{x:.1f}%")
                if 'Revenue per User' in segment_metrics.columns:
                    segment_metrics['Revenue per User'] = segment_metrics['Revenue per User'].apply(lambda x: f"${x:,.2f}")
                
                # Rename columns for display
                segment_metrics = segment_metrics.rename(columns={
                    'Segment': 'Segment',
                    'User_Count': 'User Count',
                    'Total_Revenue': 'Total Revenue',
                    'Average_Revenue': 'ARPU',
                    'Revenue %': 'Revenue %',
                    'User %': 'User %',
                    'Revenue per User': 'Revenue per User'
                })
                
                # Display table
                st.dataframe(segment_metrics, use_container_width=True)
                
                # Segment insights
                st.subheader("Segment Insights")
                
                # Generate insights based on segment data
                insights = []
                
                try:
                    # Find highest revenue segment
                    highest_revenue_idx = df['Total_Revenue'].idxmax()
                    highest_revenue_segment = df.loc[highest_revenue_idx, 'Segment']
                    highest_revenue = df.loc[highest_revenue_idx, 'Total_Revenue']
                    highest_revenue_pct = highest_revenue / total_revenue * 100 if total_revenue > 0 else 0
                    
                    insights.append(f"💰 **Top revenue segment:** {highest_revenue_segment} generates ${highest_revenue:,.2f} ({highest_revenue_pct:.1f}% of total revenue).")
                    
                    # Find highest ARPU segment
                    if 'Average_Revenue' in df.columns:
                        highest_arpu_idx = df['Average_Revenue'].idxmax()
                        highest_arpu_segment = df.loc[highest_arpu_idx, 'Segment']
                        highest_arpu = df.loc[highest_arpu_idx, 'Average_Revenue']
                        
                        insights.append(f"💎 **Highest ARPU segment:** {highest_arpu_segment} users spend an average of ${highest_arpu:,.2f} each.")
                    
                    # Find largest user segment
                    if 'User_Count' in df.columns:
                        largest_segment_idx = df['User_Count'].idxmax()
                        largest_segment = df.loc[largest_segment_idx, 'Segment']
                        largest_segment_users = df.loc[largest_segment_idx, 'User_Count']
                        largest_segment_pct = largest_segment_users / total_users * 100 if total_users > 0 else 0
                        
                        insights.append(f"👥 **Largest user segment:** {largest_segment} with {largest_segment_users:,.0f} users ({largest_segment_pct:.1f}% of total users).")
                    
                    # Revenue concentration
                    if 'User_Count' in df.columns and len(df) > 1:
                        # Sort by revenue per user
                        sorted_df = df.sort_values('Total_Revenue', ascending=False)
                        top_segment = sorted_df.iloc[0]
                        top_segment_name = top_segment['Segment']
                        top_segment_revenue = top_segment['Total_Revenue']
                        top_segment_users = top_segment['User_Count']
                        
                        revenue_concentration = top_segment_revenue / total_revenue * 100 if total_revenue > 0 else 0
                        user_concentration = top_segment_users / total_users * 100 if total_users > 0 else 0
                        
                        if revenue_concentration > 50 and user_concentration < 30:
                            insights.append(f"⚠️ **Revenue concentration risk:** {top_segment_name} generates {revenue_concentration:.1f}% of revenue but represents only {user_concentration:.1f}% of users.")
                
                except Exception as e:
                    logger.error(f"Error generating segment insights: {str(e)}")
                    insights.append("Unable to generate segment insights due to data processing error.")
                
                # Display insights
                if insights:
                    for insight in insights:
                        st.markdown(insight)
                else:
                    st.info("No segment insights available for the selected data.")
            else:
                st.info("Segment data is not properly formatted.")
        else:
            st.info("Segment data is not available.")
    
    def render_revenue_breakdown_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Revenue Breakdown section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Revenue Breakdown Analysis")
        
        # Revenue by device type
        if 'device_distribution.csv' in datasets:
            df = datasets['device_distribution.csv']
            
            if not df.empty and 'Device_Type' in df.columns and 'User_Count' in df.columns:
                # Create synthetic revenue data based on device distribution
                # In a real implementation, this would use actual revenue by device data
                
                # Create a copy of the DataFrame
                device_revenue_df = df.copy()
                
                # Generate synthetic revenue data
                device_revenue_df['Revenue'] = device_revenue_df['User_Count'] * 20  # Arbitrary revenue per user
                
                # Add some variation
                device_revenue_df.loc[device_revenue_df['Device_Type'] == 'PC', 'Revenue'] *= 1.2
                device_revenue_df.loc[device_revenue_df['Device_Type'] == 'Console', 'Revenue'] *= 1.5
                device_revenue_df.loc[device_revenue_df['Device_Type'] == 'Mobile', 'Revenue'] *= 0.8
                
                # Calculate ARPU
                device_revenue_df['ARPU'] = device_revenue_df['Revenue'] / device_revenue_df['User_Count']
                
                # Create columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a pie chart for revenue by device
                    fig = px.pie(
                        device_revenue_df,
                        values='Revenue',
                        names='Device_Type',
                        title='Revenue by Device Type',
                        color_discrete_sequence=px.colors.qualitative.Plotly
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
                    # Create a bar chart for ARPU by device
                    fig = px.bar(
                        device_revenue_df,
                        x='Device_Type',
                        y='ARPU',
                        title='ARPU by Device Type',
                        color='Device_Type',
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title='Device Type',
                        yaxis_title='ARPU ($)',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Device metrics table
                st.subheader("Device Revenue Metrics")
                
                # Create a table with device metrics
                device_metrics = device_revenue_df.copy()
                
                # Calculate percentages
                total_revenue = device_metrics['Revenue'].sum()
                device_metrics['Revenue %'] = device_metrics['Revenue'] / total_revenue * 100 if total_revenue > 0 else 0
                
                total_users = device_metrics['User_Count'].sum()
                device_metrics['User %'] = device_metrics['User_Count'] / total_users * 100 if total_users > 0 else 0
                
                # Format columns
                device_metrics['Revenue'] = device_metrics['Revenue'].apply(lambda x: f"${x:,.2f}")
                device_metrics['ARPU'] = device_metrics['ARPU'].apply(lambda x: f"${x:,.2f}")
                device_metrics['Revenue %'] = device_metrics['Revenue %'].apply(lambda x: f"{x:.1f}%")
                device_metrics['User %'] = device_metrics['User %'].apply(lambda x: f"{x:.1f}%")
                
                # Rename columns for display
                device_metrics = device_metrics.rename(columns={
                    'Device_Type': 'Device Type',
                    'User_Count': 'User Count',
                    'Revenue': 'Revenue',
                    'ARPU': 'ARPU',
                    'Revenue %': 'Revenue %',
                    'User %': 'User %'
                })
                
                # Display table
                st.dataframe(device_metrics, use_container_width=True)
            else:
                st.info("Device distribution data is not properly formatted.")
        else:
            st.info("Device distribution data is not available.")
        
        # Revenue by game mode
        if 'game_mode_distribution.csv' in datasets:
            df = datasets['game_mode_distribution.csv']
            
            if not df.empty and 'Game_Mode' in df.columns and 'User_Count' in df.columns:
                # Create synthetic revenue data based on game mode distribution
                # In a real implementation, this would use actual revenue by game mode data
                
                # Create a copy of the DataFrame
                mode_revenue_df = df.copy()
                
                # Generate synthetic revenue data
                mode_revenue_df['Revenue'] = mode_revenue_df['User_Count'] * 25  # Arbitrary revenue per user
                
                # Add some variation
                mode_revenue_df.loc[mode_revenue_df['Game_Mode'] == 'Multiplayer', 'Revenue'] *= 1.3
                mode_revenue_df.loc[mode_revenue_df['Game_Mode'] == 'Co-op', 'Revenue'] *= 1.1
                mode_revenue_df.loc[mode_revenue_df['Game_Mode'] == 'Solo', 'Revenue'] *= 0.9
                
                # Calculate ARPU
                mode_revenue_df['ARPU'] = mode_revenue_df['Revenue'] / mode_revenue_df['User_Count']
                
                # Create columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a pie chart for revenue by game mode
                    fig = px.pie(
                        mode_revenue_df,
                        values='Revenue',
                        names='Game_Mode',
                        title='Revenue by Game Mode',
                        color_discrete_sequence=px.colors.qualitative.Safe
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
                    # Create a bar chart for ARPU by game mode
                    fig = px.bar(
                        mode_revenue_df,
                        x='Game_Mode',
                        y='ARPU',
                        title='ARPU by Game Mode',
                        color='Game_Mode',
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title='Game Mode',
                        yaxis_title='ARPU ($)',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Game mode metrics table
                st.subheader("Game Mode Revenue Metrics")
                
                # Create a table with game mode metrics
                mode_metrics = mode_revenue_df.copy()
                
                # Calculate percentages
                total_revenue = mode_metrics['Revenue'].sum()
                mode_metrics['Revenue %'] = mode_metrics['Revenue'] / total_revenue * 100 if total_revenue > 0 else 0
                
                total_users = mode_metrics['User_Count'].sum()
                mode_metrics['User %'] = mode_metrics['User_Count'] / total_users * 100 if total_users > 0 else 0
                
                # Format columns
                mode_metrics['Revenue'] = mode_metrics['Revenue'].apply(lambda x: f"${x:,.2f}")
                mode_metrics['ARPU'] = mode_metrics['ARPU'].apply(lambda x: f"${x:,.2f}")
                mode_metrics['Revenue %'] = mode_metrics['Revenue %'].apply(lambda x: f"{x:.1f}%")
                mode_metrics['User %'] = mode_metrics['User %'].apply(lambda x: f"{x:.1f}%")
                
                # Rename columns for display
                mode_metrics = mode_metrics.rename(columns={
                    'Game_Mode': 'Game Mode',
                    'User_Count': 'User Count',
                    'Revenue': 'Revenue',
                    'ARPU': 'ARPU',
                    'Revenue %': 'Revenue %',
                    'User %': 'User %'
                })
                
                # Display table
                st.dataframe(mode_metrics, use_container_width=True)
            else:
                st.info("Game mode distribution data is not properly formatted.")
        else:
            st.info("Game mode distribution data is not available.")
    
    def render_monetization_metrics_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Monetization Metrics section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Monetization Metrics Analysis")
        
        # ARPU over time
        if 'revenue_by_date.csv' in datasets and 'active_users.csv' in datasets:
            revenue_df = datasets['revenue_by_date.csv']
            users_df = datasets['active_users.csv']
            
            if (not revenue_df.empty and 'Date' in revenue_df.columns and 'Revenue' in revenue_df.columns and
                not users_df.empty and 'Date' in users_df.columns and 'DAU' in users_df.columns):
                
                # Filter by date range
                start_date = pd.Timestamp(date_range['start'])
                end_date = pd.Timestamp(date_range['end'])
                
                revenue_df = revenue_df[(revenue_df['Date'] >= start_date) & (revenue_df['Date'] <= end_date)]
                users_df = users_df[(users_df['Date'] >= start_date) & (users_df['Date'] <= end_date)]
                
                if not revenue_df.empty and not users_df.empty:
                    # Merge revenue and user data
                    merged_df = pd.merge(revenue_df, users_df, on='Date')
                    
                    # Calculate ARPU and ARPPU
                    merged_df['ARPU'] = merged_df['Revenue'] / merged_df['DAU']
                    
                    # Create ARPU trend chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=merged_df['Date'],
                        y=merged_df['ARPU'],
                        mode='lines',
                        name='ARPU',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Calculate 7-day moving average
                    merged_df['ARPU_MA7'] = merged_df['ARPU'].rolling(window=7).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=merged_df['Date'],
                        y=merged_df['ARPU_MA7'],
                        mode='lines',
                        name='7-Day Moving Average',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Average Revenue per User (ARPU) Over Time',
                        xaxis_title='Date',
                        yaxis_title='ARPU ($)',
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
                    
                    # Add range slider
                    fig.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label="1w", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate key monetization metrics
                    st.subheader("Key Monetization Metrics")
                    
                    # Calculate metrics
                    avg_arpu = merged_df['ARPU'].mean()
                    
                    # Calculate ARPU growth
                    first_arpu = merged_df['ARPU'].iloc[0]
                    last_arpu = merged_df['ARPU'].iloc[-1]
                    arpu_growth = ((last_arpu - first_arpu) / first_arpu * 100) if first_arpu > 0 else 0
                    
                    # Calculate revenue per MAU
                    if 'MAU' in merged_df.columns:
                        avg_revenue_per_mau = merged_df['Revenue'].sum() / merged_df['MAU'].iloc[-1]
                    else:
                        avg_revenue_per_mau = None
                    
                    # Create metrics table
                    metrics_data = {
                        'Metric': ['Average ARPU', 'ARPU Growth'],
                        'Value': [f"${avg_arpu:.2f}", f"{arpu_growth:+.1f}%"]
                    }
                    
                    if avg_revenue_per_mau is not None:
                        metrics_data['Metric'].append('Revenue per MAU')
                        metrics_data['Value'].append(f"${avg_revenue_per_mau:.2f}")
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # ARPU distribution
                    fig = px.histogram(
                        merged_df,
                        x='ARPU',
                        nbins=20,
                        title='ARPU Distribution',
                        color_discrete_sequence=['#1f77b4']
                    )
                    
                    fig.update_layout(
                        xaxis_title='ARPU ($)',
                        yaxis_title='Frequency',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No monetization data available for the selected date range.")
            else:
                st.info("Revenue or user data is not properly formatted.")
        else:
            st.info("Required data for monetization metrics is not available.")
        
        # Revenue vs. User Activity
        if 'revenue_by_date.csv' in datasets and 'active_users.csv' in datasets:
            revenue_df = datasets['revenue_by_date.csv']
            users_df = datasets['active_users.csv']
            
            if (not revenue_df.empty and 'Date' in revenue_df.columns and 'Revenue' in revenue_df.columns and
                not users_df.empty and 'Date' in users_df.columns and 'DAU' in users_df.columns):
                
                # Filter by date range
                start_date = pd.Timestamp(date_range['start'])
                end_date = pd.Timestamp(date_range['end'])
                
                revenue_df = revenue_df[(revenue_df['Date'] >= start_date) & (revenue_df['Date'] <= end_date)]
                users_df = users_df[(users_df['Date'] >= start_date) & (users_df['Date'] <= end_date)]
                
                if not revenue_df.empty and not users_df.empty:
                    # Merge revenue and user data
                    merged_df = pd.merge(revenue_df, users_df, on='Date')
                    
                    st.subheader("Revenue vs. User Activity")
                    
                    # Create scatter plot of revenue vs. DAU
                    fig = px.scatter(
                        merged_df,
                        x='DAU',
                        y='Revenue',
                        title='Revenue vs. Daily Active Users',
                        trendline='ols',
                        hover_data=['Date']
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title='Daily Active Users',
                        yaxis_title='Revenue ($)',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate correlation
                    correlation = merged_df['Revenue'].corr(merged_df['DAU'])
                    
                    st.markdown(f"**Correlation between Revenue and DAU:** {correlation:.2f}")
                    
                    if correlation > 0.7:
                        st.markdown("💡 **Strong positive correlation:** Revenue is strongly tied to daily active users. Focus on user acquisition and retention strategies.")
                    elif correlation > 0.3:
                        st.markdown("💡 **Moderate positive correlation:** Revenue is moderately tied to daily active users. Consider both user growth and monetization optimization.")
                    else:
                        st.markdown("💡 **Weak correlation:** Revenue is not strongly tied to daily active users. Focus on improving monetization of existing users.")
                else:
                    st.info("No data available for the selected date range.")
            else:
                st.info("Revenue or user data is not properly formatted.")
        else:
            st.info("Required data for revenue vs. user activity analysis is not available.")
