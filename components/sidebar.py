import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Sidebar:
    """Class for creating and managing the sidebar in the Matiks Gaming Analytics Dashboard."""
    
    def __init__(self):
        """Initialize the Sidebar."""
        pass
    
    def create_sidebar(self) -> Dict:
        """
        Create the sidebar with navigation and filter options.
        
        Returns:
            Dictionary containing selected options
        """
        with st.sidebar:
            # Dashboard title and logo
            st.markdown("""
            <div style="text-align: center;">
                <h1 style="font-size: 1.5rem; margin-bottom: 0;">🎮 Matiks Gaming</h1>
                <p style="font-size: 0.8rem; margin-top: 0;">Analytics Dashboard</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Navigation
            st.subheader("Navigation")
            pages = [
                "Overview",
                "User Analytics",
                "Revenue Analytics",
                "Engagement",
                "Segmentation",
                "Cohort Analysis"
            ]
            selected_page = st.radio("", pages, index=0, label_visibility="collapsed")
            
            st.markdown("---")
            
            # Date range selection
            st.subheader("Date Range")
            
            # Predefined date ranges
            date_ranges = {
                "Last 7 Days": (datetime.now() - timedelta(days=7), datetime.now()),
                "Last 30 Days": (datetime.now() - timedelta(days=30), datetime.now()),
                "Last 90 Days": (datetime.now() - timedelta(days=90), datetime.now()),
                "Year to Date": (datetime(datetime.now().year, 1, 1), datetime.now()),
                "All Time": (datetime(2023, 1, 1), datetime.now()),
                "Custom": None
            }
            
            selected_range = st.selectbox("Select date range", list(date_ranges.keys()))
            
            if selected_range == "Custom":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start date", value=datetime.now() - timedelta(days=30))
                with col2:
                    end_date = st.date_input("End date", value=datetime.now())
            else:
                start_date, end_date = date_ranges[selected_range]
                if start_date and end_date:  # Convert to date for display
                    start_date = start_date.date()
                    end_date = end_date.date()
            
            st.markdown("---")
            
            # Filters
            st.subheader("Filters")
            
            # Device filter
            devices = ["All Devices", "Mobile", "PC", "Console"]
            selected_device = st.selectbox("Device", devices)
            
            # Game mode filter
            modes = ["All Modes", "Multiplayer", "Co-op", "Solo"]
            selected_mode = st.selectbox("Game Mode", modes)
            
            # User segment filter
            segments = ["All Segments", "Non-spenders", "Low spenders", "Medium spenders", "High spenders"]
            selected_segment = st.selectbox("User Segment", segments)
            
            st.markdown("---")
            
            # Settings
            st.subheader("Settings")
            
            # Theme toggle
            theme = st.selectbox("Theme", ["Light", "Dark"])
            
            # Chart type
            chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Area"])
            
            # Refresh data button
            if st.button("Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.experimental_rerun()
            
            # About section
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; font-size: 0.8rem;">
                <p>Matiks Gaming Analytics v2.0</p>
                <p>© 2025 Matiks Gaming</p>
            </div>
            """, unsafe_allow_html=True)
        
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
                "theme": theme,
                "chart_type": chart_type
            }
        }
