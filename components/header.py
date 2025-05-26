import streamlit as st
import pandas as pd
import logging
from typing import Dict, Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Header:
    """
    Class to handle the header section with KPIs for the Matiks dashboard.
    """
    
    def __init__(self, title: str = "Matiks Gaming Analytics Dashboard"):
        """
        Initialize the header component.
        
        Args:
            title: Dashboard title
        """
        self.title = title
    
    def render_title(self):
        """Render the dashboard title."""
        st.title(self.title)
        st.markdown("""
        This interactive dashboard provides insights into user behavior, revenue patterns, 
        and engagement metrics for the Matiks gaming platform.
        """)
    
    def render_date_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        Render the selected date range.
        
        Args:
            start_date: Start date
            end_date: End date
        """
        st.markdown(f"**Data from:** {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}")
    
    def render_kpis(self, kpi_data: Dict[str, Dict]):
        """
        Render KPI metrics in a row of cards.
        
        Args:
            kpi_data: Dictionary of KPI data with format:
                {
                    "KPI Name": {
                        "value": current value,
                        "change": percentage change,
                        "trend": "up" or "down" or "neutral",
                        "format": format string (e.g., "{:,.0f}")
                    }
                }
        """
        try:
            # Create columns for KPIs
            cols = st.columns(len(kpi_data))
            
            # Render each KPI in its own column
            for i, (kpi_name, kpi_info) in enumerate(kpi_data.items()):
                with cols[i]:
                    # Format the value
                    format_str = kpi_info.get("format", "{:,.0f}")
                    value_str = format_str.format(kpi_info["value"])
                    
                    # Determine trend color and icon
                    trend = kpi_info.get("trend", "neutral")
                    if trend == "up":
                        trend_color = "green" if kpi_info["change"] >= 0 else "red"
                        trend_icon = "↑" if kpi_info["change"] >= 0 else "↓"
                    elif trend == "down":
                        trend_color = "green" if kpi_info["change"] <= 0 else "red"
                        trend_icon = "↓" if kpi_info["change"] <= 0 else "↑"
                    else:
                        trend_color = "gray"
                        trend_icon = "→"
                    
                    # Format the change percentage
                    change_str = f"{abs(kpi_info['change']):.1f}%"
                    
                    # Create the KPI card with custom CSS
                    st.markdown(f"""
                    <div style="border-radius: 5px; border: 1px solid #ddd; padding: 10px; text-align: center;">
                        <h3 style="margin: 0; color: #555;">{kpi_name}</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">{value_str}</p>
                        <p style="margin: 0; color: {trend_color};">
                            {trend_icon} {change_str}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error rendering KPIs: {str(e)}")
            st.error(f"Error displaying KPIs: {str(e)}")
    
    def render_filters_summary(self, filters: Dict):
        """
        Render a summary of applied filters.
        
        Args:
            filters: Dictionary of applied filters
        """
        active_filters = []
        
        if filters["device"] != "All Devices":
            active_filters.append(f"Device: {filters['device']}")
        
        if filters["mode"] != "All Modes":
            active_filters.append(f"Game Mode: {filters['mode']}")
        
        if filters["segment"] != "All Segments":
            active_filters.append(f"Segment: {filters['segment']}")
        
        if active_filters:
            st.markdown("**Active Filters:** " + " | ".join(active_filters))
        else:
            st.markdown("**Active Filters:** None (showing all data)")
    
    def render_page_header(self, page_title: str, description: str):
        """
        Render a header for a specific page.
        
        Args:
            page_title: Page title
            description: Page description
        """
        st.header(page_title)
        st.markdown(description)
        st.markdown("---")
    
    def render_section_header(self, section_title: str, section_description: Optional[str] = None):
        """
        Render a header for a section within a page.
        
        Args:
            section_title: Section title
            section_description: Optional section description
        """
        st.subheader(section_title)
        if section_description:
            st.markdown(section_description)
    
    def render_full_header(
        self, 
        page_info: Dict,
        date_range: Dict,
        filters: Dict,
        kpi_data: Optional[Dict[str, Dict]] = None
    ):
        """
        Render the complete header section including title, KPIs, and filters summary.
        
        Args:
            page_info: Dictionary with page title and description
            date_range: Dictionary with start and end dates
            filters: Dictionary of applied filters
            kpi_data: Optional dictionary of KPI data
        """
        # Render title and description
        self.render_page_header(page_info["title"], page_info["description"])
        
        # Render date range
        self.render_date_range(pd.Timestamp(date_range["start"]), pd.Timestamp(date_range["end"]))
        
        # Render KPIs if provided
        if kpi_data:
            st.markdown("---")
            self.render_kpis(kpi_data)
        
        # Render filters summary
        st.markdown("---")
        self.render_filters_summary(filters)
        st.markdown("---")
