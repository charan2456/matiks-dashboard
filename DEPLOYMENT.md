# Matiks Dashboard - Deployment Instructions (Updated)

## Overview
This document provides instructions for deploying the Matiks Gaming Analytics Dashboard, a comprehensive Streamlit application for visualizing gaming analytics data.

## Prerequisites
- Python 3.11 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Important Updates
This version includes critical fixes for:
- Streamlit caching compatibility issues
- CSV column name normalization for robust data handling
- Enhanced error handling and logging

## Installation Options

### Option 1: Deploy on Streamlit Cloud (Recommended)

1. **Create a GitHub repository**
   - Create a new repository on GitHub
   - Upload all files from this package to the repository

2. **Sign up for Streamlit Cloud**
   - Go to [streamlit.io](https://streamlit.io) and sign up for a free account
   - Connect your GitHub account

3. **Deploy the dashboard**
   - In Streamlit Cloud, click "New app"
   - Select your repository, branch, and set the main file path to `app.py`
   - Click "Deploy"

4. **Access your dashboard**
   - Once deployed, Streamlit Cloud will provide a permanent URL for your dashboard
   - Share this URL with your team

### Option 2: Local Deployment

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

3. **Access the dashboard**
   - Open your browser and go to `http://localhost:8501`

## Project Structure

```
matiks-dashboard/
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── components/             # Reusable UI components
│   ├── sidebar.py          # Navigation and filter controls
│   └── header.py           # Dashboard header with KPIs
├── pages/                  # Dashboard pages
│   └── overview.py         # Main dashboard overview
├── utils/                  # Utility functions
│   ├── data_loader.py      # Data loading with caching (UPDATED)
│   └── chart_factory.py    # Chart creation utilities
└── data/                   # Data files
    ├── active_users.csv
    ├── revenue_by_date.csv
    ├── funnel_analysis.csv
    └── ...
```

## Data Handling

### CSV Column Naming
The dashboard now includes robust column name handling:
- Column names are automatically normalized to lowercase
- Special column mappings are applied (e.g., 'DAU' → 'daily_active_users')
- Date columns are automatically detected and converted

### Data Requirements
CSV files should include these columns (case-insensitive):
- active_users.csv: Date/date, DAU/dau, WAU/wau, MAU/mau
- revenue_by_date.csv: Date/date, Revenue/revenue
- Other files: See documentation for expected column formats

## Customization

### Adding New Pages
1. Create a new Python file in the `pages/` directory
2. Follow the pattern in `pages/overview.py`
3. Add the new page to the sidebar options in `components/sidebar.py`
4. Add the page rendering logic in `app.py`

### Modifying Data Sources
1. Add new CSV files to the `data/` directory
2. Update the `DataLoader` class in `utils/data_loader.py` if needed
3. Use the data in your dashboard pages

## Troubleshooting

### Common Issues

1. **Missing dependencies**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Data file errors**
   - Check that all CSV files are present in the `data/` directory
   - Verify CSV file format and column names
   - The dashboard now handles column name variations automatically

3. **Streamlit version compatibility**
   - This dashboard is tested with Streamlit 1.31.0
   - If using a different version, check for API changes

### Getting Help
If you encounter issues not covered here, please:
1. Check the Streamlit documentation: https://docs.streamlit.io/
2. Review the error messages in the terminal where Streamlit is running
3. Contact the dashboard provider for additional support

## Performance Optimization

For large datasets or high-traffic deployments:
1. Enable caching for computationally intensive operations
2. Consider using Streamlit's session state for storing user preferences
3. Optimize data loading with filters applied at the data source level

## Security Considerations

1. Do not store sensitive data in the repository
2. Use environment variables for configuration settings
3. Consider authentication if the dashboard contains confidential information
