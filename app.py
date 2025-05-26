import streamlit as st
import sys
import os
import logging

# Add parent directory to path to import components and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.sidebar import Sidebar
from components.header import Header
from utils.data_loader import DataLoader
from pages.overview import OverviewPage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the Matiks Gaming Analytics Dashboard."""
    
    # Set page configuration
    st.set_page_config(
        page_title="Matiks Gaming Analytics Dashboard",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        # Initialize data loader
        data_loader = DataLoader(data_dir="data")
        
        # Initialize sidebar
        sidebar = Sidebar()
        
        # Render sidebar and get selected options
        sidebar_options = sidebar.create_sidebar()
        
        # Initialize pages
        overview_page = OverviewPage(data_loader)
        
        # Render selected page
        selected_page = sidebar_options["page"]
        
        if selected_page == "Overview":
            overview_page.render(sidebar_options["date_range"], sidebar_options["filters"])
        elif selected_page == "User Analytics":
            st.title("User Analytics")
            st.info("This page is under development. Please check back later.")
        elif selected_page == "Revenue Analytics":
            st.title("Revenue Analytics")
            st.info("This page is under development. Please check back later.")
        elif selected_page == "Engagement":
            st.title("Engagement Analytics")
            st.info("This page is under development. Please check back later.")
        elif selected_page == "Segmentation":
            st.title("Segmentation Analysis")
            st.info("This page is under development. Please check back later.")
        elif selected_page == "Cohort Analysis":
            st.title("Cohort Analysis")
            st.info("This page is under development. Please check back later.")
        else:
            st.error(f"Unknown page: {selected_page}")
    
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.markdown("### Troubleshooting Information")
        st.markdown("""
        If you're seeing this error, please try the following:
        
        1. Check that all data files are present in the `data` directory
        2. Verify that all required packages are installed
        3. Restart the application
        
        If the problem persists, please contact support.
        """)

if __name__ == "__main__":
    main()
