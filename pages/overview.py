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

class OverviewPage:
    """Class for rendering the Overview page of the Matiks Gaming Analytics Dashboard."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize the Overview page with a data loader.
        
        Args:
            data_loader: DataLoader instance for accessing data
        """
        self.data_loader = data_loader
        self.header = Header()
    
    def render(self, date_range: Dict, filters: Dict):
        """
        Render the Overview page.
        
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
                "title": "Dashboard Overview",
                "description": "Key metrics and insights for Matiks gaming platform."
            }
            self.header.render_full_header(page_info, date_range, filters, kpi_data)
            
            # Create tabs for different sections
            tabs = st.tabs(["User Activity", "Revenue", "Engagement", "Insights"])
            
            # User Activity tab
            with tabs[0]:
                self.render_user_activity_section(datasets, date_range, filters)
            
            # Revenue tab
            with tabs[1]:
                self.render_revenue_section(datasets, date_range, filters)
            
            # Engagement tab
            with tabs[2]:
                self.render_engagement_section(datasets, date_range, filters)
            
            # Insights tab
            with tabs[3]:
                self.render_insights_section(datasets, kpi_data)
            
        except Exception as e:
            logger.error(f"Error rendering Overview page: {str(e)}")
            st.error(f"An error occurred while rendering the Overview page: {str(e)}")
    
    def render_user_activity_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the User Activity section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Activity Trends")
        
        # DAU/WAU/MAU chart
        if 'active_users.csv' in datasets:
            df = datasets['active_users.csv']
            
            if not df.empty and 'Date' in df.columns:
                # Filter by date range
                start_date = pd.Timestamp(date_range['start'])
                end_date = pd.Timestamp(date_range['end'])
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                if not df.empty:
                    # Create a multi-line chart for DAU/WAU/MAU
                    fig = go.Figure()
                    
                    if 'DAU' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df['Date'],
                            y=df['DAU'],
                            mode='lines',
                            name='Daily Active Users',
                            line=dict(color='#1f77b4', width=2)
                        ))
                    
                    if 'WAU' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df['Date'],
                            y=df['WAU'],
                            mode='lines',
                            name='Weekly Active Users',
                            line=dict(color='#ff7f0e', width=2)
                        ))
                    
                    if 'MAU' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df['Date'],
                            y=df['MAU'],
                            mode='lines',
                            name='Monthly Active Users',
                            line=dict(color='#2ca02c', width=2)
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Active Users Over Time',
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
                    
                    # Stickiness chart
                    if 'Stickiness' in df.columns:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=df['Date'],
                            y=df['Stickiness'],
                            mode='lines',
                            name='Stickiness (DAU/MAU)',
                            line=dict(color='#9467bd', width=2),
                            fill='tozeroy'
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title='User Stickiness (DAU/MAU Ratio)',
                            xaxis_title='Date',
                            yaxis_title='Stickiness',
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
                else:
                    st.info("No user activity data available for the selected date range.")
            else:
                st.info("User activity data is not properly formatted.")
        else:
            st.info("User activity data is not available.")
        
        # Device distribution
        col1, col2 = st.columns(2)
        
        with col1:
            if 'device_distribution.csv' in datasets:
                df = datasets['device_distribution.csv']
                
                if not df.empty and 'Device_Type' in df.columns and 'User_Count' in df.columns:
                    # Create a pie chart for device distribution
                    fig = px.pie(
                        df,
                        values='User_Count',
                        names='Device_Type',
                        title='User Distribution by Device',
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
                else:
                    st.info("Device distribution data is not properly formatted.")
            else:
                st.info("Device distribution data is not available.")
        
        with col2:
            if 'game_mode_distribution.csv' in datasets:
                df = datasets['game_mode_distribution.csv']
                
                if not df.empty and 'Game_Mode' in df.columns and 'User_Count' in df.columns:
                    # Create a pie chart for game mode distribution
                    fig = px.pie(
                        df,
                        values='User_Count',
                        names='Game_Mode',
                        title='User Distribution by Game Mode',
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
                else:
                    st.info("Game mode distribution data is not properly formatted.")
            else:
                st.info("Game mode distribution data is not available.")
    
    def render_revenue_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Revenue section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Revenue Analysis")
        
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
                        name='Revenue',
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
                else:
                    st.info("No revenue data available for the selected date range.")
            else:
                st.info("Revenue data is not properly formatted.")
        else:
            st.info("Revenue data is not available.")
        
        # Revenue by segment
        if 'revenue_by_segment.csv' in datasets:
            df = datasets['revenue_by_segment.csv']
            
            if not df.empty and 'Segment' in df.columns and 'Total_Revenue' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a bar chart for revenue by segment
                    fig = px.bar(
                        df,
                        x='Segment',
                        y='Total_Revenue',
                        title='Revenue by User Segment',
                        color='Segment',
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title='User Segment',
                        yaxis_title='Total Revenue ($)',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Average_Revenue' in df.columns:
                        # Create a bar chart for ARPU by segment
                        fig = px.bar(
                            df,
                            x='Segment',
                            y='Average_Revenue',
                            title='Average Revenue per User by Segment',
                            color='Segment',
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        
                        # Update layout
                        fig.update_layout(
                            xaxis_title='User Segment',
                            yaxis_title='ARPU ($)',
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Average revenue data is not available.")
            else:
                st.info("Revenue by segment data is not properly formatted.")
        else:
            st.info("Revenue by segment data is not available.")
    
    def render_engagement_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Engagement section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Engagement")
        
        # User funnel
        if 'funnel_analysis.csv' in datasets:
            df = datasets['funnel_analysis.csv']
            
            if not df.empty and 'Stage' in df.columns and 'Count' in df.columns:
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
                
                # Calculate conversion rates between stages
                if len(df) > 1:
                    conversion_data = []
                    
                    for i in range(len(df) - 1):
                        from_stage = df['Stage'].iloc[i]
                        to_stage = df['Stage'].iloc[i + 1]
                        from_count = df['Count'].iloc[i]
                        to_count = df['Count'].iloc[i + 1]
                        
                        conversion_rate = (to_count / from_count * 100) if from_count > 0 else 0
                        
                        conversion_data.append({
                            'From Stage': from_stage,
                            'To Stage': to_stage,
                            'Conversion Rate': f"{conversion_rate:.1f}%"
                        })
                    
                    # Display conversion rates
                    st.markdown("### Stage-to-Stage Conversion Rates")
                    st.dataframe(pd.DataFrame(conversion_data), use_container_width=True)
            else:
                st.info("Funnel data is not properly formatted.")
        else:
            st.info("Funnel data is not available.")
        
        # Cohort retention heatmap
        if 'cohort_retention.csv' in datasets:
            df = datasets['cohort_retention.csv']
            
            if not df.empty and 'Cohort_Month' in df.columns:
                st.markdown("### Cohort Retention Analysis")
                
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
                else:
                    st.info("No retention data available in the cohort analysis.")
            else:
                st.info("Cohort data is not properly formatted.")
        else:
            st.info("Cohort data is not available.")
    
    def render_insights_section(self, datasets: Dict[str, pd.DataFrame], kpi_data: Dict[str, Dict]):
        """
        Render the Insights section.
        
        Args:
            datasets: Dictionary of DataFrames
            kpi_data: Dictionary of KPI data
        """
        st.subheader("Key Insights")
        
        # Generate insights based on the data
        insights = []
        
        try:
            # User activity insights
            if 'active_users.csv' in datasets and "Daily Active Users" in kpi_data:
                dau_change = kpi_data["Daily Active Users"]["change"]
                dau_value = kpi_data["Daily Active Users"]["value"]
                
                if dau_change > 5:
                    insights.append(f"📈 **Growing user base:** Daily active users increased by {dau_change:.1f}% to {dau_value:,.0f} users.")
                elif dau_change < -5:
                    insights.append(f"📉 **Declining user base:** Daily active users decreased by {abs(dau_change):.1f}% to {dau_value:,.0f} users.")
                
                if "Stickiness" in kpi_data:
                    stickiness = kpi_data["Stickiness"]["value"]
                    if stickiness > 0.3:
                        insights.append(f"🔄 **Strong user engagement:** Stickiness ratio of {stickiness:.2f} indicates users are returning frequently.")
                    elif stickiness < 0.2:
                        insights.append(f"⚠️ **Low user engagement:** Stickiness ratio of {stickiness:.2f} suggests users are not returning regularly.")
            
            # Revenue insights
            if 'revenue_by_date.csv' in datasets and "Total Revenue" in kpi_data:
                revenue_change = kpi_data["Total Revenue"]["change"]
                revenue_value = kpi_data["Total Revenue"]["value"]
                
                if revenue_change > 5:
                    insights.append(f"💰 **Revenue growth:** Total revenue increased by {revenue_change:.1f}% to ${revenue_value:,.2f}.")
                elif revenue_change < -5:
                    insights.append(f"📉 **Revenue decline:** Total revenue decreased by {abs(revenue_change):.1f}% to ${revenue_value:,.2f}.")
                
                if "ARPU" in kpi_data:
                    arpu = kpi_data["ARPU"]["value"]
                    arpu_change = kpi_data["ARPU"]["change"]
                    
                    if arpu_change > 5:
                        insights.append(f"💎 **Increasing monetization:** Average revenue per user increased by {arpu_change:.1f}% to ${arpu:.2f}.")
                    elif arpu_change < -5:
                        insights.append(f"⚠️ **Decreasing monetization:** Average revenue per user decreased by {abs(arpu_change):.1f}% to ${arpu:.2f}.")
            
            # Segment insights
            if 'revenue_by_segment.csv' in datasets:
                df = datasets['revenue_by_segment.csv']
                
                if not df.empty and 'Segment' in df.columns and 'Total_Revenue' in df.columns and 'User_Count' in df.columns:
                    # Find the highest revenue segment
                    highest_revenue_idx = df['Total_Revenue'].idxmax()
                    highest_revenue_segment = df.loc[highest_revenue_idx, 'Segment']
                    highest_revenue = df.loc[highest_revenue_idx, 'Total_Revenue']
                    highest_revenue_users = df.loc[highest_revenue_idx, 'User_Count']
                    
                    # Find the largest user segment
                    largest_segment_idx = df['User_Count'].idxmax()
                    largest_segment = df.loc[largest_segment_idx, 'Segment']
                    largest_segment_users = df.loc[largest_segment_idx, 'User_Count']
                    
                    insights.append(f"💰 **Key revenue segment:** {highest_revenue_segment} users ({highest_revenue_users:,.0f} users) generate ${highest_revenue:,.2f} in revenue.")
                    
                    if largest_segment != highest_revenue_segment:
                        insights.append(f"👥 **Largest user segment:** {largest_segment} is the largest user segment with {largest_segment_users:,.0f} users.")
            
            # Funnel insights
            if 'funnel_analysis.csv' in datasets:
                df = datasets['funnel_analysis.csv']
                
                if not df.empty and 'Stage' in df.columns and 'Count' in df.columns and len(df) >= 2:
                    # Find the biggest drop in the funnel
                    biggest_drop_idx = 0
                    biggest_drop_pct = 0
                    
                    for i in range(len(df) - 1):
                        current = df['Count'].iloc[i]
                        next_val = df['Count'].iloc[i + 1]
                        drop_pct = (1 - next_val / current) * 100 if current > 0 else 0
                        
                        if drop_pct > biggest_drop_pct:
                            biggest_drop_pct = drop_pct
                            biggest_drop_idx = i
                    
                    if biggest_drop_pct > 10:
                        from_stage = df['Stage'].iloc[biggest_drop_idx]
                        to_stage = df['Stage'].iloc[biggest_drop_idx + 1]
                        insights.append(f"🚧 **Funnel bottleneck:** {biggest_drop_pct:.1f}% drop-off between {from_stage} and {to_stage} stages.")
                    
                    # Overall funnel conversion
                    first_stage = df['Stage'].iloc[0]
                    last_stage = df['Stage'].iloc[-1]
                    first_count = df['Count'].iloc[0]
                    last_count = df['Count'].iloc[-1]
                    
                    conversion = (last_count / first_count * 100) if first_count > 0 else 0
                    insights.append(f"🔄 **Funnel conversion:** {conversion:.1f}% of users who {first_stage.lower()} complete the {last_stage.lower()} stage.")
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append("Unable to generate insights due to data processing error.")
        
        # Display insights
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.info("No insights available for the selected data.")
        
        # Recommendations
        st.subheader("Recommendations")
        
        # Generate recommendations based on insights
        recommendations = []
        
        try:
            # User engagement recommendations
            if 'active_users.csv' in datasets and kpi_data.get("Daily Active Users"):
                dau_change = kpi_data["Daily Active Users"]["change"]
                if dau_change < 0:
                    recommendations.append("🚀 **Boost user engagement:** Consider launching a limited-time event or promotion to re-engage users.")
            
            # Revenue recommendations
            if 'revenue_by_date.csv' in datasets and kpi_data.get("Total Revenue") and kpi_data.get("ARPU"):
                revenue_change = kpi_data["Total Revenue"]["change"]
                arpu = kpi_data["ARPU"]["value"]
                if revenue_change < 0:
                    recommendations.append("💸 **Increase monetization:** Review pricing strategy and consider introducing new premium features or bundles.")
                if arpu < 5:  # Arbitrary threshold for this example
                    recommendations.append("💰 **Improve ARPU:** Target high-value user segments with personalized offers to increase average revenue per user.")
            
            # Device-specific recommendations
            if 'device_distribution.csv' in datasets:
                df = datasets['device_distribution.csv']
                if not df.empty and 'Device_Type' in df.columns and 'User_Count' in df.columns:
                    devices = df['Device_Type'].tolist()
                    if 'Mobile' in devices:
                        mobile_users = df[df['Device_Type'] == 'Mobile']['User_Count'].values[0]
                        total_users = df['User_Count'].sum()
                        mobile_pct = (mobile_users / total_users * 100) if total_users > 0 else 0
                        if mobile_pct > 50:
                            recommendations.append(f"📱 **Mobile optimization:** With {mobile_pct:.1f}% of users on mobile, prioritize mobile experience improvements and features.")
            
            # Funnel recommendations
            if 'funnel_analysis.csv' in datasets:
                df = datasets['funnel_analysis.csv']
                if not df.empty and 'Stage' in df.columns and 'Count' in df.columns and len(df) >= 3:
                    # Find the biggest drop in the funnel
                    biggest_drop_idx = 0
                    biggest_drop_pct = 0
                    
                    for i in range(len(df) - 1):
                        current = df['Count'].iloc[i]
                        next_val = df['Count'].iloc[i + 1]
                        drop_pct = (1 - next_val / current) * 100 if current > 0 else 0
                        
                        if drop_pct > biggest_drop_pct:
                            biggest_drop_pct = drop_pct
                            biggest_drop_idx = i
                    
                    if biggest_drop_pct > 30:  # Arbitrary threshold
                        current_stage = df['Stage'].iloc[biggest_drop_idx]
                        next_stage = df['Stage'].iloc[biggest_drop_idx + 1]
                        recommendations.append(f"🔍 **Funnel optimization:** Address the {biggest_drop_pct:.1f}% drop-off between {current_stage} and {next_stage} stages.")
            
            # Add generic recommendations if we don't have enough specific ones
            if len(recommendations) < 2:
                recommendations.append("📊 **Data collection:** Enhance data collection to enable more personalized user experiences and targeted marketing.")
                recommendations.append("🔄 **A/B testing:** Implement A/B testing for new features to optimize user engagement and retention.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("Unable to generate recommendations due to data processing error.")
        
        # Display recommendations
        if recommendations:
            for recommendation in recommendations:
                st.markdown(recommendation)
        else:
            st.info("No recommendations available for the selected data.")
