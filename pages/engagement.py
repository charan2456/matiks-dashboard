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

class EngagementPage:
    """Class for rendering the Engagement page of the Matiks Gaming Analytics Dashboard."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize the Engagement page with a data loader.
        
        Args:
            data_loader: DataLoader instance for accessing data
        """
        self.data_loader = data_loader
        self.header = Header()
    
    def render(self, date_range: Dict, filters: Dict):
        """
        Render the Engagement page.
        
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
                "title": "User Engagement",
                "description": "Analysis of user engagement patterns and journey funnels."
            }
            self.header.render_full_header(page_info, date_range, filters, kpi_data)
            
            # Create tabs for different sections
            tabs = st.tabs(["Stickiness Analysis", "User Journey", "Engagement Patterns", "Retention"])
            
            # Stickiness Analysis tab
            with tabs[0]:
                self.render_stickiness_section(datasets, date_range, filters)
            
            # User Journey tab
            with tabs[1]:
                self.render_user_journey_section(datasets, date_range, filters)
            
            # Engagement Patterns tab
            with tabs[2]:
                self.render_engagement_patterns_section(datasets, date_range, filters)
            
            # Retention tab
            with tabs[3]:
                self.render_retention_section(datasets, date_range, filters)
            
        except Exception as e:
            logger.error(f"Error rendering Engagement page: {str(e)}")
            st.error(f"An error occurred while rendering the Engagement page: {str(e)}")
    
    def render_stickiness_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Stickiness Analysis section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Stickiness Analysis")
        
        # Stickiness trend
        if 'active_users.csv' in datasets:
            df = datasets['active_users.csv']
            
            if not df.empty and 'Date' in df.columns and 'Stickiness' in df.columns:
                # Filter by date range
                start_date = pd.Timestamp(date_range['start'])
                end_date = pd.Timestamp(date_range['end'])
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                if not df.empty:
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
                    
                    # Stickiness metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        current_stickiness = df['Stickiness'].iloc[-1]
                        st.metric("Current Stickiness", f"{current_stickiness:.2f}")
                    
                    with col2:
                        avg_stickiness = df['Stickiness'].mean()
                        st.metric("Average Stickiness", f"{avg_stickiness:.2f}")
                    
                    with col3:
                        if len(df) > 1:
                            first_stickiness = df['Stickiness'].iloc[0]
                            last_stickiness = df['Stickiness'].iloc[-1]
                            stickiness_change = ((last_stickiness - first_stickiness) / first_stickiness * 100) if first_stickiness > 0 else 0
                            st.metric("Stickiness Change", f"{stickiness_change:+.1f}%")
                        else:
                            st.metric("Stickiness Change", "N/A")
                    
                    # Stickiness interpretation
                    st.subheader("Stickiness Interpretation")
                    
                    if current_stickiness >= 0.3:
                        st.success("**Strong user engagement:** Users are returning frequently to the platform.")
                    elif current_stickiness >= 0.2:
                        st.info("**Moderate user engagement:** Users are returning regularly to the platform.")
                    else:
                        st.warning("**Low user engagement:** Users are not returning regularly to the platform.")
                    
                    # Stickiness recommendations
                    st.subheader("Recommendations")
                    
                    if current_stickiness < 0.2:
                        st.markdown("1. **Implement daily rewards:** Encourage users to return daily with progressive rewards.")
                        st.markdown("2. **Enhance notification system:** Send personalized notifications about new content or events.")
                        st.markdown("3. **Introduce daily challenges:** Create daily objectives that provide in-game benefits.")
                    elif current_stickiness < 0.3:
                        st.markdown("1. **Optimize reward frequency:** Adjust reward schedules to encourage more frequent visits.")
                        st.markdown("2. **Enhance social features:** Implement features that encourage user interaction.")
                        st.markdown("3. **Analyze user feedback:** Identify and address pain points in the user experience.")
                    else:
                        st.markdown("1. **Maintain engagement:** Continue current engagement strategies.")
                        st.markdown("2. **Optimize for retention:** Focus on long-term retention of highly engaged users.")
                        st.markdown("3. **Leverage community:** Encourage community building among highly engaged users.")
                else:
                    st.info("No stickiness data available for the selected date range.")
            else:
                st.info("Stickiness data is not properly formatted.")
        else:
            st.info("Stickiness data is not available.")
    
    def render_user_journey_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the User Journey section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Journey Analysis")
        
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
                        drop_off_rate = 100 - conversion_rate
                        
                        conversion_data.append({
                            'From Stage': from_stage,
                            'To Stage': to_stage,
                            'Conversion Rate': f"{conversion_rate:.1f}%",
                            'Drop-off Rate': f"{drop_off_rate:.1f}%",
                            'From Count': from_count,
                            'To Count': to_count
                        })
                    
                    # Display conversion rates
                    st.markdown("### Stage-to-Stage Conversion Rates")
                    
                    # Create a DataFrame for display
                    conversion_df = pd.DataFrame(conversion_data)
                    st.dataframe(conversion_df, use_container_width=True)
                    
                    # Find the biggest drop-off
                    biggest_drop_idx = 0
                    biggest_drop_rate = 0
                    
                    for i, data in enumerate(conversion_data):
                        rate = float(data['Drop-off Rate'].replace('%', ''))
                        if rate > biggest_drop_rate:
                            biggest_drop_rate = rate
                            biggest_drop_idx = i
                    
                    # Highlight the biggest drop-off
                    if conversion_data:
                        biggest_drop = conversion_data[biggest_drop_idx]
                        st.markdown(f"**Biggest Drop-off:** {biggest_drop['Drop-off Rate']} between **{biggest_drop['From Stage']}** and **{biggest_drop['To Stage']}**")
                    
                    # Funnel optimization recommendations
                    st.subheader("Funnel Optimization Recommendations")
                    
                    if biggest_drop_rate > 50:
                        st.markdown(f"1. **Critical bottleneck:** Address the {biggest_drop['Drop-off Rate']} drop-off between {biggest_drop['From Stage']} and {biggest_drop['To Stage']}.")
                        st.markdown(f"2. **User research:** Conduct user research to understand why users are dropping off at this stage.")
                        st.markdown(f"3. **A/B testing:** Test different approaches to improve conversion at this critical stage.")
                    elif biggest_drop_rate > 30:
                        st.markdown(f"1. **Significant drop-off:** Improve the {biggest_drop['Drop-off Rate']} conversion between {biggest_drop['From Stage']} and {biggest_drop['To Stage']}.")
                        st.markdown(f"2. **User experience review:** Review the user experience at this stage to identify friction points.")
                        st.markdown(f"3. **Targeted messaging:** Enhance communication to guide users through this transition.")
                    else:
                        st.markdown("1. **Overall funnel optimization:** Continue to monitor and optimize all stages of the funnel.")
                        st.markdown("2. **Incremental improvements:** Make small, continuous improvements to each stage of the funnel.")
                        st.markdown("3. **Benchmark analysis:** Compare your funnel metrics with industry benchmarks.")
            else:
                st.info("Funnel data is not properly formatted.")
        else:
            st.info("Funnel data is not available.")
    
    def render_engagement_patterns_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Engagement Patterns section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Engagement Patterns")
        
        # DAU/WAU/MAU ratios
        if 'active_users.csv' in datasets:
            df = datasets['active_users.csv']
            
            if not df.empty and 'Date' in df.columns and 'DAU' in df.columns and 'WAU' in df.columns and 'MAU' in df.columns:
                # Filter by date range
                start_date = pd.Timestamp(date_range['start'])
                end_date = pd.Timestamp(date_range['end'])
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                if not df.empty:
                    # Calculate DAU/WAU and WAU/MAU ratios
                    df['DAU_WAU_Ratio'] = df['DAU'] / df['WAU']
                    df['WAU_MAU_Ratio'] = df['WAU'] / df['MAU']
                    
                    # Create columns for charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create a line chart for DAU/WAU ratio
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=df['Date'],
                            y=df['DAU_WAU_Ratio'],
                            mode='lines',
                            name='DAU/WAU Ratio',
                            line=dict(color='#1f77b4', width=2),
                            fill='tozeroy'
                        ))
                        
                        # Calculate 7-day moving average
                        df['DAU_WAU_Ratio_MA7'] = df['DAU_WAU_Ratio'].rolling(window=7).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=df['Date'],
                            y=df['DAU_WAU_Ratio_MA7'],
                            mode='lines',
                            name='7-Day Moving Average',
                            line=dict(color='#ff7f0e', width=2, dash='dash')
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title='Daily to Weekly Active Users Ratio',
                            xaxis_title='Date',
                            yaxis_title='DAU/WAU Ratio',
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
                        # Create a line chart for WAU/MAU ratio
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=df['Date'],
                            y=df['WAU_MAU_Ratio'],
                            mode='lines',
                            name='WAU/MAU Ratio',
                            line=dict(color='#2ca02c', width=2),
                            fill='tozeroy'
                        ))
                        
                        # Calculate 7-day moving average
                        df['WAU_MAU_Ratio_MA7'] = df['WAU_MAU_Ratio'].rolling(window=7).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=df['Date'],
                            y=df['WAU_MAU_Ratio_MA7'],
                            mode='lines',
                            name='7-Day Moving Average',
                            line=dict(color='#d62728', width=2, dash='dash')
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title='Weekly to Monthly Active Users Ratio',
                            xaxis_title='Date',
                            yaxis_title='WAU/MAU Ratio',
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
                    
                    # Engagement metrics
                    st.subheader("Engagement Metrics")
                    
                    # Calculate current metrics
                    current_dau_wau = df['DAU_WAU_Ratio'].iloc[-1]
                    current_wau_mau = df['WAU_MAU_Ratio'].iloc[-1]
                    
                    # Calculate average metrics
                    avg_dau_wau = df['DAU_WAU_Ratio'].mean()
                    avg_wau_mau = df['WAU_MAU_Ratio'].mean()
                    
                    # Create metrics table
                    metrics_data = {
                        'Metric': ['DAU/WAU Ratio', 'WAU/MAU Ratio'],
                        'Current Value': [f"{current_dau_wau:.2f}", f"{current_wau_mau:.2f}"],
                        'Average Value': [f"{avg_dau_wau:.2f}", f"{avg_wau_mau:.2f}"]
                    }
                    
                    # Add change metrics if we have enough data
                    if len(df) > 1:
                        first_dau_wau = df['DAU_WAU_Ratio'].iloc[0]
                        first_wau_mau = df['WAU_MAU_Ratio'].iloc[0]
                        
                        dau_wau_change = ((current_dau_wau - first_dau_wau) / first_dau_wau * 100) if first_dau_wau > 0 else 0
                        wau_mau_change = ((current_wau_mau - first_wau_mau) / first_wau_mau * 100) if first_wau_mau > 0 else 0
                        
                        metrics_data['Change'] = [f"{dau_wau_change:+.1f}%", f"{wau_mau_change:+.1f}%"]
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Engagement interpretation
                    st.subheader("Engagement Interpretation")
                    
                    # DAU/WAU interpretation
                    if current_dau_wau >= 0.25:
                        st.success("**Strong daily engagement:** Users are engaging with the platform multiple days per week.")
                    elif current_dau_wau >= 0.15:
                        st.info("**Moderate daily engagement:** Users are engaging with the platform about once per week.")
                    else:
                        st.warning("**Low daily engagement:** Users are not engaging with the platform regularly throughout the week.")
                    
                    # WAU/MAU interpretation
                    if current_wau_mau >= 0.6:
                        st.success("**Strong weekly engagement:** Users are engaging with the platform multiple weeks per month.")
                    elif current_wau_mau >= 0.4:
                        st.info("**Moderate weekly engagement:** Users are engaging with the platform about twice per month.")
                    else:
                        st.warning("**Low weekly engagement:** Users are not engaging with the platform regularly throughout the month.")
                else:
                    st.info("No engagement data available for the selected date range.")
            else:
                st.info("Engagement data is not properly formatted.")
        else:
            st.info("Engagement data is not available.")
        
        # Game mode and device engagement
        col1, col2 = st.columns(2)
        
        with col1:
            # Game mode distribution
            if 'game_mode_distribution.csv' in datasets:
                df = datasets['game_mode_distribution.csv']
                
                if not df.empty and 'Game_Mode' in df.columns and 'User_Count' in df.columns:
                    # Create a pie chart for game mode distribution
                    fig = px.pie(
                        df,
                        values='User_Count',
                        names='Game_Mode',
                        title='User Engagement by Game Mode',
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
        
        with col2:
            # Device distribution
            if 'device_distribution.csv' in datasets:
                df = datasets['device_distribution.csv']
                
                if not df.empty and 'Device_Type' in df.columns and 'User_Count' in df.columns:
                    # Create a pie chart for device distribution
                    fig = px.pie(
                        df,
                        values='User_Count',
                        names='Device_Type',
                        title='User Engagement by Device',
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
    
    def render_retention_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Retention section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Retention Analysis")
        
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
                    
                    # Retention metrics
                    st.subheader("Retention Metrics")
                    
                    # Calculate key retention metrics
                    if len(retention_cols) > 0:
                        # First month retention (Month 1)
                        month_1_col = '1' if '1' in retention_cols else retention_cols[0]
                        month_1_retention = heatmap_data[month_1_col].mean()
                        
                        # Calculate metrics for display
                        metrics_data = {
                            'Metric': ['First Month Retention'],
                            'Value': [f"{month_1_retention:.1f}%"]
                        }
                        
                        # Add more metrics if we have enough data
                        if len(retention_cols) > 2:
                            # Month 3 retention
                            month_3_col = '3' if '3' in retention_cols else retention_cols[min(2, len(retention_cols) - 1)]
                            month_3_retention = heatmap_data[month_3_col].mean()
                            metrics_data['Metric'].append('Third Month Retention')
                            metrics_data['Value'].append(f"{month_3_retention:.1f}%")
                        
                        if len(retention_cols) > 5:
                            # Month 6 retention
                            month_6_col = '6' if '6' in retention_cols else retention_cols[min(5, len(retention_cols) - 1)]
                            month_6_retention = heatmap_data[month_6_col].mean()
                            metrics_data['Metric'].append('Sixth Month Retention')
                            metrics_data['Value'].append(f"{month_6_retention:.1f}%")
                        
                        # Display metrics
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Retention interpretation
                        st.subheader("Retention Interpretation")
                        
                        # First month retention interpretation
                        if month_1_retention >= 40:
                            st.success(f"**Strong first month retention:** {month_1_retention:.1f}% of users are still active after the first month.")
                        elif month_1_retention >= 25:
                            st.info(f"**Average first month retention:** {month_1_retention:.1f}% of users are still active after the first month.")
                        else:
                            st.warning(f"**Low first month retention:** Only {month_1_retention:.1f}% of users are still active after the first month.")
                        
                        # Retention recommendations
                        st.subheader("Retention Recommendations")
                        
                        if month_1_retention < 25:
                            st.markdown("1. **Improve onboarding:** Enhance the onboarding experience to better engage new users.")
                            st.markdown("2. **Early value delivery:** Ensure users experience value within the first few sessions.")
                            st.markdown("3. **Targeted communication:** Implement targeted email/push campaigns for new users.")
                        elif month_1_retention < 40:
                            st.markdown("1. **Optimize engagement:** Identify and optimize key engagement features.")
                            st.markdown("2. **Personalization:** Implement personalized experiences based on user behavior.")
                            st.markdown("3. **Feedback loop:** Establish a feedback loop with users to understand pain points.")
                        else:
                            st.markdown("1. **Maintain strengths:** Continue to leverage successful retention strategies.")
                            st.markdown("2. **Long-term engagement:** Focus on long-term engagement beyond the first month.")
                            st.markdown("3. **Community building:** Foster a community around your most engaged users.")
                else:
                    st.info("No retention data available in the cohort analysis.")
            else:
                st.info("Cohort data is not properly formatted.")
        else:
            st.info("Cohort data is not available.")
