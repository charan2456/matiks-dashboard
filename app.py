import streamlit as st
from utils.data_loader import DataLoader
from components.sidebar import Sidebar
from pages.overview import OverviewPage
from pages.user_analytics import UserAnalyticsPage
from pages.revenue_analytics import RevenueAnalyticsPage
from pages.engagement import EngagementPage
from pages.segmentation import SegmentationPage
from pages.cohort_analysis import CohortAnalysisPage
from pages.funnel_analysis import FunnelAnalysisPage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Matiks Gaming Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        border-bottom: 2px solid #4c78a8;
    }
    div[data-testid="stSidebarNav"] li div a {
        margin-left: 1rem;
        padding: 1rem;
        width: 300px;
        border-radius: 0.5rem;
    }
    div[data-testid="stSidebarNav"] li div::focus-visible {
        background-color: rgba(151, 166, 195, 0.15);
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 5% 5% 5% 10%;
        border: 1px solid #eee;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #eee;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main function to run the Matiks Gaming Analytics Dashboard."""
    try:
        # Initialize data loader
        data_loader = DataLoader(data_dir="data")
        
        # Create sidebar
        sidebar = Sidebar()
        sidebar_options = sidebar.create_sidebar()
        
        # Initialize pages
        overview_page = OverviewPage(data_loader)
        user_analytics_page = UserAnalyticsPage(data_loader)
        revenue_analytics_page = RevenueAnalyticsPage(data_loader)
        engagement_page = EngagementPage(data_loader)
        segmentation_page = SegmentationPage(data_loader)
        cohort_analysis_page = CohortAnalysisPage(data_loader)
        funnel_analysis_page = FunnelAnalysisPage(data_loader)
        
        # Render selected page
        selected_page = sidebar_options["page"]
        date_range = sidebar_options["date_range"]
        filters = sidebar_options["filters"]
        
        if selected_page == "Overview":
            overview_page.render(date_range, filters)
        elif selected_page == "User Analytics":
            user_analytics_page.render(date_range, filters)
        elif selected_page == "Revenue Analytics":
            revenue_analytics_page.render(date_range, filters)
        elif selected_page == "Engagement":
            engagement_page.render(date_range, filters)
        elif selected_page == "Segmentation":
            segmentation_page.render(date_range, filters)
        elif selected_page == "Cohort Analysis":
            cohort_analysis_page.render(date_range, filters)
        elif selected_page == "Funnel Analysis":
            funnel_analysis_page.render(date_range, filters)
        
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        st.error(f"An error occurred in the application: {str(e)}")

if __name__ == "__main__":
    main()
