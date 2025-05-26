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

class SegmentationPage:
    """Class for rendering the Segmentation page of the Matiks Gaming Analytics Dashboard."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize the Segmentation page with a data loader.
        
        Args:
            data_loader: DataLoader instance for accessing data
        """
        self.data_loader = data_loader
        self.header = Header()
    
    def render(self, date_range: Dict, filters: Dict):
        """
        Render the Segmentation page.
        
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
                "title": "User Segmentation",
                "description": "Analysis of user segments and clustering for targeted strategies."
            }
            self.header.render_full_header(page_info, date_range, filters, kpi_data)
            
            # Create tabs for different sections
            tabs = st.tabs(["Segment Analysis", "User Clustering", "Segment Comparison", "Targeting Strategies"])
            
            # Segment Analysis tab
            with tabs[0]:
                self.render_segment_analysis_section(datasets, date_range, filters)
            
            # User Clustering tab
            with tabs[1]:
                self.render_user_clustering_section(datasets, date_range, filters)
            
            # Segment Comparison tab
            with tabs[2]:
                self.render_segment_comparison_section(datasets, date_range, filters)
            
            # Targeting Strategies tab
            with tabs[3]:
                self.render_targeting_strategies_section(datasets, date_range, filters)
            
        except Exception as e:
            logger.error(f"Error rendering Segmentation page: {str(e)}")
            st.error(f"An error occurred while rendering the Segmentation page: {str(e)}")
    
    def render_segment_analysis_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Segment Analysis section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Segment Analysis")
        
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
    
    def render_user_clustering_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the User Clustering section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Clustering Analysis")
        
        # Check if we have the necessary data for clustering
        if 'revenue_by_segment.csv' in datasets:
            segment_df = datasets['revenue_by_segment.csv']
            
            if not segment_df.empty and 'Segment' in segment_df.columns and 'User_Count' in segment_df.columns and 'Average_Revenue' in segment_df.columns:
                st.markdown("### Frequency vs. Revenue Clustering")
                
                # Create synthetic user data based on segments for demonstration
                # In a real implementation, this would use actual user-level data
                
                # Generate sample data
                import numpy as np
                from sklearn.cluster import KMeans
                
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
                
                # Additional clustering visualization
                st.markdown("### Segment Distribution within Clusters")
                
                # Calculate segment distribution within clusters
                segment_cluster_dist = pd.crosstab(user_df['Cluster'], user_df['Segment'])
                
                # Convert to percentages
                segment_cluster_pct = segment_cluster_dist.div(segment_cluster_dist.sum(axis=1), axis=0) * 100
                
                # Create stacked bar chart
                fig = px.bar(
                    segment_cluster_pct.reset_index().melt(id_vars='Cluster', var_name='Segment', value_name='Percentage'),
                    x='Cluster',
                    y='Percentage',
                    color='Segment',
                    title='Segment Distribution within Clusters',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Cluster',
                    yaxis_title='Percentage (%)',
                    margin=dict(l=40, r=40, t=40, b=40),
                    barmode='stack'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Segment data is not properly formatted for clustering analysis.")
        else:
            st.info("Segment data is not available for clustering analysis.")
    
    def render_segment_comparison_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Segment Comparison section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Segment Comparison Analysis")
        
        # Revenue vs. User Count by segment
        if 'revenue_by_segment.csv' in datasets:
            df = datasets['revenue_by_segment.csv']
            
            if not df.empty and 'Segment' in df.columns and 'Total_Revenue' in df.columns and 'User_Count' in df.columns:
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
                
                # Revenue contribution vs. User contribution
                st.subheader("Revenue Contribution vs. User Contribution")
                
                # Calculate percentages
                df['Revenue_Pct'] = df['Total_Revenue'] / df['Total_Revenue'].sum() * 100
                df['User_Pct'] = df['User_Count'] / df['User_Count'].sum() * 100
                
                # Create a scatter plot
                fig = px.scatter(
                    df,
                    x='User_Pct',
                    y='Revenue_Pct',
                    size='Total_Revenue',
                    color='Segment',
                    hover_data=['Segment', 'Total_Revenue', 'User_Count', 'Average_Revenue'],
                    title='Revenue Contribution vs. User Contribution by Segment',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                # Add reference line (y=x)
                fig.add_trace(go.Scatter(
                    x=[0, 100],
                    y=[0, 100],
                    mode='lines',
                    name='Equal Contribution',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                # Update layout
                fig.update_layout(
                    xaxis_title='User Contribution (%)',
                    yaxis_title='Revenue Contribution (%)',
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.markdown("""
                **Interpretation:**
                - Segments above the dashed line contribute more to revenue than to user count (high value)
                - Segments below the dashed line contribute more to user count than to revenue (low value)
                - Bubble size represents total revenue contribution
                """)
            else:
                st.info("Segment data is not properly formatted for comparison analysis.")
        else:
            st.info("Segment data is not available for comparison analysis.")
    
    def render_targeting_strategies_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Targeting Strategies section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Segment Targeting Strategies")
        
        # Check if we have segment data
        if 'revenue_by_segment.csv' in datasets:
            df = datasets['revenue_by_segment.csv']
            
            if not df.empty and 'Segment' in df.columns:
                # Create a table of targeting strategies
                strategies = []
                
                # High spenders
                if 'High spenders' in df['Segment'].values:
                    strategies.append({
                        'Segment': 'High spenders',
                        'Goal': 'Retention and increased spending',
                        'Strategy': 'VIP program with exclusive content and early access to new features',
                        'KPIs': 'Retention rate, ARPU, Spending frequency'
                    })
                
                # Medium spenders
                if 'Medium spenders' in df['Segment'].values:
                    strategies.append({
                        'Segment': 'Medium spenders',
                        'Goal': 'Increase spending frequency and amount',
                        'Strategy': 'Loyalty rewards and bundle offers with good value perception',
                        'KPIs': 'Conversion to high spenders, Purchase frequency, Bundle adoption'
                    })
                
                # Low spenders
                if 'Low spenders' in df['Segment'].values:
                    strategies.append({
                        'Segment': 'Low spenders',
                        'Goal': 'Increase spending amount',
                        'Strategy': 'Entry-level premium offers and limited-time promotions',
                        'KPIs': 'Conversion to medium spenders, Promotion engagement, ARPU'
                    })
                
                # Non-spenders
                if 'Non-spenders' in df['Segment'].values:
                    strategies.append({
                        'Segment': 'Non-spenders',
                        'Goal': 'Convert to paying users',
                        'Strategy': 'Free trial of premium features and first-purchase incentives',
                        'KPIs': 'Conversion rate, First purchase value, Retention after purchase'
                    })
                
                # Display strategies
                if strategies:
                    st.dataframe(pd.DataFrame(strategies), use_container_width=True)
                
                # Targeting recommendations
                st.subheader("Targeting Recommendations")
                
                # Generate recommendations based on segment data
                recommendations = []
                
                try:
                    # Find highest revenue segment
                    highest_revenue_idx = df['Total_Revenue'].idxmax()
                    highest_revenue_segment = df.loc[highest_revenue_idx, 'Segment']
                    
                    recommendations.append(f"💎 **Focus on {highest_revenue_segment}:** This segment generates the most revenue. Prioritize retention strategies for these users.")
                    
                    # Find segment with highest potential
                    if 'User_Count' in df.columns and 'Average_Revenue' in df.columns:
                        # Calculate potential (simplified)
                        df['Potential'] = df['User_Count'] * (df['Average_Revenue'].max() - df['Average_Revenue'])
                        
                        highest_potential_idx = df['Potential'].idxmax()
                        highest_potential_segment = df.loc[highest_potential_idx, 'Segment']
                        
                        recommendations.append(f"🚀 **Growth opportunity in {highest_potential_segment}:** This segment has the highest revenue growth potential. Focus on increasing ARPU for these users.")
                    
                    # Find largest segment
                    if 'User_Count' in df.columns:
                        largest_segment_idx = df['User_Count'].idxmax()
                        largest_segment = df.loc[largest_segment_idx, 'Segment']
                        
                        if largest_segment != highest_revenue_segment:
                            recommendations.append(f"👥 **Convert {largest_segment}:** This is your largest user segment. Focus on converting these users to higher-value segments.")
                
                except Exception as e:
                    logger.error(f"Error generating targeting recommendations: {str(e)}")
                    recommendations.append("Unable to generate targeting recommendations due to data processing error.")
                
                # Display recommendations
                if recommendations:
                    for recommendation in recommendations:
                        st.markdown(recommendation)
                
                # Personalization strategies
                st.subheader("Personalization Strategies")
                
                st.markdown("""
                ### Content Personalization
                - **High-value users:** Exclusive content, early access, premium features
                - **Medium-value users:** Personalized recommendations, social features, achievement systems
                - **Low/Non-spenders:** Onboarding optimization, core gameplay focus, community integration
                
                ### Pricing Personalization
                - **High-value users:** Premium bundles, subscription options, collectible items
                - **Medium-value users:** Value bundles, limited-time offers, progression boosters
                - **Low/Non-spenders:** Entry-level purchases, trial offers, ad removal options
                
                ### Communication Personalization
                - **High-value users:** VIP communications, early announcements, exclusive events
                - **Medium-value users:** Feature highlights, community events, achievement celebrations
                - **Low/Non-spenders:** Onboarding guidance, feature discovery, social incentives
                """)
            else:
                st.info("Segment data is not properly formatted for targeting strategies.")
        else:
            st.info("Segment data is not available for targeting strategies.")
