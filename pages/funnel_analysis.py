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

class FunnelAnalysisPage:
    """Class for rendering the Funnel Analysis page of the Matiks Gaming Analytics Dashboard."""
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize the Funnel Analysis page with a data loader.
        
        Args:
            data_loader: DataLoader instance for accessing data
        """
        self.data_loader = data_loader
        self.header = Header()
    
    def render(self, date_range: Dict, filters: Dict):
        """
        Render the Funnel Analysis page.
        
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
                "title": "Funnel Analysis",
                "description": "Analysis of user journey through key conversion stages."
            }
            self.header.render_full_header(page_info, date_range, filters, kpi_data)
            
            # Create tabs for different sections
            tabs = st.tabs(["User Journey", "Conversion Analysis", "Funnel Optimization", "Insights"])
            
            # User Journey tab
            with tabs[0]:
                self.render_user_journey_section(datasets, date_range, filters)
            
            # Conversion Analysis tab
            with tabs[1]:
                self.render_conversion_analysis_section(datasets, date_range, filters)
            
            # Funnel Optimization tab
            with tabs[2]:
                self.render_funnel_optimization_section(datasets, date_range, filters)
            
            # Insights tab
            with tabs[3]:
                self.render_insights_section(datasets, date_range, filters)
            
        except Exception as e:
            logger.error(f"Error rendering Funnel Analysis page: {str(e)}")
            st.error(f"An error occurred while rendering the Funnel Analysis page: {str(e)}")
    
    def render_user_journey_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the User Journey section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("User Journey Funnel")
        
        # Funnel visualization
        if 'funnel_analysis.csv' in datasets:
            df = datasets['funnel_analysis.csv']
            
            if not df.empty and 'Stage' in df.columns and 'Count' in df.columns:
                # Create funnel chart
                fig = go.Figure(go.Funnel(
                    y=df['Stage'],
                    x=df['Count'],
                    textposition="inside",
                    textinfo="value+percent initial",
                    opacity=0.8,
                    marker={"color": ["#4C78A8", "#72B7B2", "#54A24B", "#EECA3B", "#F58518", "#E45756"]},
                    connector={"line": {"color": "royalblue", "width": 1}}
                ))
                
                # Update layout
                fig.update_layout(
                    title="User Journey Funnel",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate conversion rates
                conversion_rates = []
                
                for i in range(len(df) - 1):
                    from_stage = df['Stage'].iloc[i]
                    to_stage = df['Stage'].iloc[i + 1]
                    from_count = df['Count'].iloc[i]
                    to_count = df['Count'].iloc[i + 1]
                    
                    conversion_rate = (to_count / from_count * 100) if from_count > 0 else 0
                    drop_off_rate = 100 - conversion_rate
                    
                    conversion_rates.append({
                        'From Stage': from_stage,
                        'To Stage': to_stage,
                        'Conversion Rate': f"{conversion_rate:.1f}%",
                        'Drop-off Rate': f"{drop_off_rate:.1f}%"
                    })
                
                # Display conversion rates
                st.subheader("Stage-to-Stage Conversion Rates")
                st.dataframe(pd.DataFrame(conversion_rates), use_container_width=True)
                
                # Calculate overall funnel conversion
                first_count = df['Count'].iloc[0]
                last_count = df['Count'].iloc[-1]
                overall_conversion = (last_count / first_count * 100) if first_count > 0 else 0
                
                # Display overall conversion
                st.metric("Overall Funnel Conversion", f"{overall_conversion:.1f}%")
                
                # Find biggest drop-off
                biggest_drop_idx = 0
                biggest_drop_rate = 0
                
                for i in range(len(df) - 1):
                    from_count = df['Count'].iloc[i]
                    to_count = df['Count'].iloc[i + 1]
                    
                    drop_off_rate = (1 - to_count / from_count) * 100 if from_count > 0 else 0
                    
                    if drop_off_rate > biggest_drop_rate:
                        biggest_drop_rate = drop_off_rate
                        biggest_drop_idx = i
                
                if biggest_drop_idx < len(df) - 1:
                    from_stage = df['Stage'].iloc[biggest_drop_idx]
                    to_stage = df['Stage'].iloc[biggest_drop_idx + 1]
                    
                    st.warning(f"**Critical Bottleneck:** {biggest_drop_rate:.1f}% drop-off between **{from_stage}** and **{to_stage}** stages")
            else:
                st.info("Funnel data is not properly formatted.")
        else:
            st.info("Funnel data is not available.")
    
    def render_conversion_analysis_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Conversion Analysis section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Conversion Analysis")
        
        # Funnel visualization by segment
        if ('funnel_analysis.csv' in datasets and not datasets['funnel_analysis.csv'].empty and
            'revenue_by_segment.csv' in datasets and not datasets['revenue_by_segment.csv'].empty):
            
            funnel_df = datasets['funnel_analysis.csv']
            segment_df = datasets['revenue_by_segment.csv']
            
            if 'Stage' in funnel_df.columns and 'Count' in funnel_df.columns and 'Segment' in segment_df.columns:
                # Create synthetic funnel data by segment for demonstration
                # In a real implementation, this would use actual segment-level funnel data
                
                # Get segments
                segments = segment_df['Segment'].tolist()
                
                # Create synthetic data
                funnel_by_segment = []
                
                for stage_idx, row in funnel_df.iterrows():
                    stage = row['Stage']
                    total_count = row['Count']
                    
                    # Distribute users across segments based on segment distribution
                    for segment in segments:
                        # Adjust conversion rates based on segment
                        if segment == 'High spenders':
                            segment_factor = 1.5 if stage_idx > 1 else 1.1  # Better conversion for high spenders
                        elif segment == 'Medium spenders':
                            segment_factor = 1.2 if stage_idx > 1 else 1.05
                        elif segment == 'Low spenders':
                            segment_factor = 0.9 if stage_idx > 1 else 0.95
                        else:  # Non-spenders
                            segment_factor = 0.6 if stage_idx > 1 else 0.9
                        
                        # Calculate segment count
                        segment_count = int(total_count * segment_factor / len(segments))
                        
                        funnel_by_segment.append({
                            'Stage': stage,
                            'Segment': segment,
                            'Count': segment_count
                        })
                
                # Create DataFrame
                funnel_segment_df = pd.DataFrame(funnel_by_segment)
                
                # Create funnel chart by segment
                fig = px.funnel(
                    funnel_segment_df,
                    x='Count',
                    y='Stage',
                    color='Segment',
                    title="User Journey Funnel by Segment"
                )
                
                # Update layout
                fig.update_layout(
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate conversion rates by segment
                st.subheader("Conversion Rates by Segment")
                
                # Create columns for segment comparison
                cols = st.columns(len(segments))
                
                for i, segment in enumerate(segments):
                    segment_data = funnel_segment_df[funnel_segment_df['Segment'] == segment]
                    
                    if len(segment_data) > 1:
                        first_count = segment_data.iloc[0]['Count']
                        last_count = segment_data.iloc[-1]['Count']
                        overall_conversion = (last_count / first_count * 100) if first_count > 0 else 0
                        
                        with cols[i]:
                            st.metric(f"{segment}", f"{overall_conversion:.1f}%", help="Overall funnel conversion rate")
                
                # Conversion comparison
                st.subheader("Stage Conversion by Segment")
                
                # Create bar chart for conversion rates by stage and segment
                conversion_data = []
                
                for segment in segments:
                    segment_data = funnel_segment_df[funnel_segment_df['Segment'] == segment]
                    
                    for i in range(len(segment_data) - 1):
                        from_stage = segment_data.iloc[i]['Stage']
                        to_stage = segment_data.iloc[i + 1]['Stage']
                        from_count = segment_data.iloc[i]['Count']
                        to_count = segment_data.iloc[i + 1]['Count']
                        
                        conversion_rate = (to_count / from_count * 100) if from_count > 0 else 0
                        
                        conversion_data.append({
                            'Segment': segment,
                            'From Stage': from_stage,
                            'To Stage': to_stage,
                            'Conversion Rate': conversion_rate,
                            'Stage Transition': f"{from_stage} → {to_stage}"
                        })
                
                conversion_df = pd.DataFrame(conversion_data)
                
                # Create bar chart
                fig = px.bar(
                    conversion_df,
                    x='Stage Transition',
                    y='Conversion Rate',
                    color='Segment',
                    barmode='group',
                    title="Conversion Rates by Stage and Segment"
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Stage Transition",
                    yaxis_title="Conversion Rate (%)",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Funnel or segment data is not properly formatted.")
        else:
            st.info("Funnel or segment data is not available.")
    
    def render_funnel_optimization_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Funnel Optimization section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Funnel Optimization Opportunities")
        
        # Funnel optimization
        if 'funnel_analysis.csv' in datasets:
            df = datasets['funnel_analysis.csv']
            
            if not df.empty and 'Stage' in df.columns and 'Count' in df.columns:
                # Calculate drop-off rates
                drop_off_rates = []
                
                for i in range(len(df) - 1):
                    from_stage = df['Stage'].iloc[i]
                    to_stage = df['Stage'].iloc[i + 1]
                    from_count = df['Count'].iloc[i]
                    to_count = df['Count'].iloc[i + 1]
                    
                    drop_off_rate = (1 - to_count / from_count) * 100 if from_count > 0 else 0
                    drop_off_count = from_count - to_count
                    
                    drop_off_rates.append({
                        'From Stage': from_stage,
                        'To Stage': to_stage,
                        'Drop-off Rate': drop_off_rate,
                        'Drop-off Count': drop_off_count
                    })
                
                # Sort by drop-off count
                drop_off_rates = sorted(drop_off_rates, key=lambda x: x['Drop-off Count'], reverse=True)
                
                # Create optimization opportunities
                opportunities = []
                
                for i, drop_off in enumerate(drop_off_rates):
                    from_stage = drop_off['From Stage']
                    to_stage = drop_off['To Stage']
                    drop_off_rate = drop_off['Drop-off Rate']
                    drop_off_count = drop_off['Drop-off Count']
                    
                    # Generate optimization strategies based on stage
                    if from_stage == 'Visitor' and to_stage == 'Signup':
                        strategies = [
                            "Simplify signup process to reduce friction",
                            "Add social login options for one-click signup",
                            "Highlight value proposition more clearly on landing page",
                            "Implement exit-intent popup with signup incentive"
                        ]
                    elif from_stage == 'Signup' and to_stage == 'Onboarded':
                        strategies = [
                            "Streamline onboarding to essential steps only",
                            "Add progress indicators to show completion status",
                            "Implement guided tutorial with rewards",
                            "Send reminder emails to users who abandon onboarding"
                        ]
                    elif from_stage == 'Onboarded' and to_stage == 'Purchaser':
                        strategies = [
                            "Offer first-purchase discount or bonus",
                            "Implement limited-time offers to create urgency",
                            "Show social proof of other users making purchases",
                            "Improve visibility and appeal of purchasable items"
                        ]
                    elif from_stage == 'Purchaser' and to_stage == 'High Value':
                        strategies = [
                            "Create bundle offers with better value perception",
                            "Implement loyalty program with escalating rewards",
                            "Personalize offers based on past purchase behavior",
                            "Add exclusive content or features for higher spenders"
                        ]
                    else:
                        strategies = [
                            "Analyze user behavior to identify friction points",
                            "A/B test different user flows to optimize conversion",
                            "Implement targeted messaging for users at this stage",
                            "Add incentives to encourage progression to next stage"
                        ]
                    
                    # Calculate potential impact
                    # Assume a 10% improvement in conversion rate
                    current_conversion = 100 - drop_off_rate
                    improved_conversion = current_conversion * 1.1  # 10% improvement
                    additional_conversions = drop_off_count * 0.1  # 10% of drop-offs
                    
                    opportunities.append({
                        'Priority': i + 1,
                        'Bottleneck': f"{from_stage} → {to_stage}",
                        'Drop-off Rate': f"{drop_off_rate:.1f}%",
                        'Drop-off Count': int(drop_off_count),
                        'Potential Impact': f"+{additional_conversions:.0f} conversions",
                        'Optimization Strategies': strategies
                    })
                
                # Display opportunities
                for opportunity in opportunities:
                    with st.expander(f"Priority {opportunity['Priority']}: {opportunity['Bottleneck']} ({opportunity['Drop-off Rate']} drop-off)"):
                        st.markdown(f"**Current Drop-off:** {opportunity['Drop-off Count']} users")
                        st.markdown(f"**Potential Impact:** {opportunity['Potential Impact']} with 10% improvement")
                        
                        st.markdown("**Recommended Strategies:**")
                        for i, strategy in enumerate(opportunity['Optimization Strategies']):
                            st.markdown(f"{i+1}. {strategy}")
            else:
                st.info("Funnel data is not properly formatted.")
        else:
            st.info("Funnel data is not available.")
    
    def render_insights_section(self, datasets: Dict[str, pd.DataFrame], date_range: Dict, filters: Dict):
        """
        Render the Insights section.
        
        Args:
            datasets: Dictionary of DataFrames
            date_range: Dictionary with start and end dates
            filters: Dictionary of filters to apply
        """
        st.subheader("Funnel Analysis Insights")
        
        # Funnel insights
        if 'funnel_analysis.csv' in datasets:
            df = datasets['funnel_analysis.csv']
            
            if not df.empty and 'Stage' in df.columns and 'Count' in df.columns:
                # Calculate key metrics
                first_count = df['Count'].iloc[0]
                last_count = df['Count'].iloc[-1]
                overall_conversion = (last_count / first_count * 100) if first_count > 0 else 0
                
                # Calculate drop-off rates
                drop_off_rates = []
                
                for i in range(len(df) - 1):
                    from_stage = df['Stage'].iloc[i]
                    to_stage = df['Stage'].iloc[i + 1]
                    from_count = df['Count'].iloc[i]
                    to_count = df['Count'].iloc[i + 1]
                    
                    drop_off_rate = (1 - to_count / from_count) * 100 if from_count > 0 else 0
                    
                    drop_off_rates.append({
                        'from_stage': from_stage,
                        'to_stage': to_stage,
                        'drop_off_rate': drop_off_rate
                    })
                
                # Find biggest drop-off
                biggest_drop = max(drop_off_rates, key=lambda x: x['drop_off_rate']) if drop_off_rates else None
                
                # Generate insights
                insights = []
                
                if overall_conversion < 5:
                    insights.append("🔍 **Low overall conversion:** Less than 5% of visitors become high-value users, indicating significant optimization potential throughout the funnel.")
                elif overall_conversion > 15:
                    insights.append("🌟 **Strong overall conversion:** More than 15% of visitors become high-value users, indicating an effective user journey.")
                
                if biggest_drop and biggest_drop['drop_off_rate'] > 70:
                    insights.append(f"⚠️ **Critical bottleneck:** {biggest_drop['drop_off_rate']:.1f}% drop-off between {biggest_drop['from_stage']} and {biggest_drop['to_stage']} stages requires immediate attention.")
                
                # Add specific insights based on stages
                for drop_off in drop_off_rates:
                    if drop_off['from_stage'] == 'Visitor' and drop_off['to_stage'] == 'Signup' and drop_off['drop_off_rate'] > 60:
                        insights.append("🚪 **High visitor bounce rate:** The majority of visitors don't sign up, suggesting issues with the landing page or signup process.")
                    
                    if drop_off['from_stage'] == 'Signup' and drop_off['to_stage'] == 'Onboarded' and drop_off['drop_off_rate'] > 40:
                        insights.append("🔄 **Onboarding abandonment:** Many users abandon the onboarding process, indicating it may be too complex or time-consuming.")
                    
                    if drop_off['from_stage'] == 'Onboarded' and drop_off['to_stage'] == 'Purchaser' and drop_off['drop_off_rate'] > 80:
                        insights.append("💰 **Low purchase conversion:** The vast majority of onboarded users never make a purchase, suggesting issues with monetization strategy or value perception.")
                
                # Add generic insights if we don't have enough specific ones
                if len(insights) < 3:
                    insights.append("📊 **Funnel optimization opportunity:** Each stage of the user journey presents opportunities for incremental improvements in conversion rates.")
                    insights.append("🔄 **User journey complexity:** Multiple steps in the conversion process create compounding effects on overall conversion rates.")
                
                # Display insights
                for insight in insights:
                    st.markdown(insight)
                
                # Recommendations
                st.subheader("Recommendations")
                
                # Generate recommendations based on insights
                recommendations = []
                
                if overall_conversion < 10:
                    recommendations.append("🎯 **Holistic funnel review:** Conduct a comprehensive review of each funnel stage to identify and address friction points.")
                
                if biggest_drop:
                    if biggest_drop['from_stage'] == 'Visitor' and biggest_drop['to_stage'] == 'Signup':
                        recommendations.append("🚪 **Optimize landing page:** Simplify the signup process and clearly communicate value proposition to improve visitor-to-signup conversion.")
                    
                    if biggest_drop['from_stage'] == 'Signup' and biggest_drop['to_stage'] == 'Onboarded':
                        recommendations.append("🔄 **Streamline onboarding:** Reduce onboarding steps to essential elements only and add progress indicators to improve completion rates.")
                    
                    if biggest_drop['from_stage'] == 'Onboarded' and biggest_drop['to_stage'] == 'Purchaser':
                        recommendations.append("💰 **Enhance monetization:** Implement first-purchase incentives and improve visibility of purchasable items to increase conversion to paying users.")
                    
                    if biggest_drop['from_stage'] == 'Purchaser' and biggest_drop['to_stage'] == 'High Value':
                        recommendations.append("🌟 **Upgrade strategy:** Create clear upgrade paths with increasing value perception to convert more users to high-value status.")
                
                # Add generic recommendations if we don't have enough specific ones
                if len(recommendations) < 3:
                    recommendations.append("📱 **A/B testing program:** Implement systematic A/B testing at each funnel stage to continuously optimize conversion rates.")
                    recommendations.append("🔍 **User research:** Conduct user interviews and surveys to understand barriers to progression at each funnel stage.")
                    recommendations.append("📊 **Segment-specific funnels:** Develop tailored user journeys for different user segments to improve overall conversion rates.")
                
                # Display recommendations
                for recommendation in recommendations:
                    st.markdown(recommendation)
            else:
                st.info("Funnel data is not properly formatted.")
        else:
            st.info("Funnel data is not available.")
