import streamlit as st
import pandas as pd
import logging
from typing import Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Sidebar:
    """
    Class to handle the sidebar navigation and filters for the Matiks dashboard.
    """
    
    def __init__(self):
        """Initialize the sidebar component."""
        self.pages = {
            "Overview": "📊 Dashboard Overview",
            "User Analytics": "👥 User Analytics",
            "Revenue Analytics": "💰 Revenue Analytics",
            "Engagement": "🎮 Engagement Analytics",
            "Segmentation": "🔍 Segmentation Analysis",
            "Cohort Analysis": "📈 Cohort Analysis"
        }
    
    def create_sidebar(self) -> Dict:
        """
        Create and render the sidebar with navigation and filters.
        
        Returns:
            Dictionary containing selected options and filters
        """
        with st.sidebar:
            st.title("🎮 Matiks Analytics")
            
            # Add separator
            st.markdown("---")
            
            # Navigation
            st.subheader("Navigation")
            selected_page = st.radio(
                "Go to",
                options=list(self.pages.keys()),
                format_func=lambda x: self.pages[x],
                label_visibility="collapsed"
            )
            
            # Add separator
            st.markdown("---")
            
            # Date range filter
            st.subheader("Date Range")
            
            # Default date range (last 30 days)
            default_start_date = pd.Timestamp.now() - pd.Timedelta(days=30)
            default_end_date = pd.Timestamp.now()
            
            start_date = st.date_input(
                "Start Date",
                value=default_start_date,
                key="start_date"
            )
            
            end_date = st.date_input(
                "End Date",
                value=default_end_date,
                key="end_date"
            )
            
            # Validate date range
            if start_date > end_date:
                st.error("Error: End date must be after start date.")
                start_date = end_date - pd.Timedelta(days=1)
            
            # Add separator
            st.markdown("---")
            
            # Filters section
            st.subheader("Filters")
            
            # Device filter
            device_types = ["All Devices", "Mobile", "Tablet", "Desktop", "Console"]
            selected_device = st.selectbox(
                "Device Type",
                options=device_types,
                index=0
            )
            
            # Game mode filter
            game_modes = ["All Modes", "Casual", "Competitive", "Story", "Multiplayer", "Training"]
            selected_mode = st.selectbox(
                "Game Mode",
                options=game_modes,
                index=0
            )
            
            # User segment filter
            user_segments = ["All Segments", "New Users", "Casual Players", "Regular Players", "Power Users", "Paying Users"]
            selected_segment = st.selectbox(
                "User Segment",
                options=user_segments,
                index=0
            )
            
            # Add separator
            st.markdown("---")
            
            # Theme toggle
            st.subheader("Settings")
            theme = st.selectbox(
                "Theme",
                options=["Light", "Dark"],
                index=0
            )
            
            # About section
            with st.expander("About"):
                st.markdown("""
                **Matiks Gaming Analytics Dashboard**
                
                This dashboard provides insights into user behavior, revenue patterns, 
                and engagement metrics for the Matiks gaming platform.
                
                Version: 2.0
                """)
            
            # Help section
            with st.expander("Help"):
                st.markdown("""
                **How to use this dashboard:**
                
                1. Use the navigation radio buttons to switch between different views
                2. Set date range to analyze specific time periods
                3. Apply filters to focus on specific segments
                4. Hover over charts for detailed information
                5. Click on legends to show/hide data series
                """)
        
        # Return selected options
        return {
            "page": selected_page,
            "date_range": {
                "start": start_date,
                "end": end_date
            },
            "filters": {
                "device": selected_device,
                "mode": selected_mode,
                "segment": selected_segment
            },
            "settings": {
                "theme": theme
            }
        }
