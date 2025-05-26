import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChartFactory:
    """
    Factory class for creating various chart types for the Matiks dashboard.
    """
    
    @staticmethod
    def create_line_chart(
        df: pd.DataFrame, 
        x: str, 
        y: Union[str, List[str]], 
        title: str = "",
        color: Optional[str] = None,
        height: int = 400,
        template: str = "plotly_white",
        hover_data: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create a line chart using Plotly Express.
        
        Args:
            df: DataFrame containing the data
            x: Column name for x-axis
            y: Column name(s) for y-axis
            title: Chart title
            color: Column name for color differentiation
            height: Chart height in pixels
            template: Plotly template name
            hover_data: Additional columns to show in hover tooltip
            
        Returns:
            Plotly figure object
        """
        try:
            fig = px.line(
                df, 
                x=x, 
                y=y, 
                title=title,
                color=color,
                height=height,
                template=template,
                hover_data=hover_data
            )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title=x.replace('_', ' ').title(),
                yaxis_title=y[0].replace('_', ' ').title() if isinstance(y, list) else y.replace('_', ' ').title(),
                legend_title_text=color.replace('_', ' ').title() if color else None,
                hovermode="x unified"
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    @staticmethod
    def create_bar_chart(
        df: pd.DataFrame, 
        x: str, 
        y: str, 
        title: str = "",
        color: Optional[str] = None,
        orientation: str = 'v',
        height: int = 400,
        template: str = "plotly_white",
        hover_data: Optional[List[str]] = None,
        text: Optional[str] = None
    ) -> go.Figure:
        """
        Create a bar chart using Plotly Express.
        
        Args:
            df: DataFrame containing the data
            x: Column name for x-axis
            y: Column name for y-axis
            title: Chart title
            color: Column name for color differentiation
            orientation: 'v' for vertical bars, 'h' for horizontal bars
            height: Chart height in pixels
            template: Plotly template name
            hover_data: Additional columns to show in hover tooltip
            text: Column name for text displayed on bars
            
        Returns:
            Plotly figure object
        """
        try:
            fig = px.bar(
                df, 
                x=x, 
                y=y, 
                title=title,
                color=color,
                orientation=orientation,
                height=height,
                template=template,
                hover_data=hover_data,
                text=text
            )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title=x.replace('_', ' ').title(),
                yaxis_title=y.replace('_', ' ').title(),
                legend_title_text=color.replace('_', ' ').title() if color else None
            )
            
            # Format text display
            if text:
                fig.update_traces(texttemplate='%{text}', textposition='outside')
            
            return fig
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    @staticmethod
    def create_pie_chart(
        df: pd.DataFrame, 
        names: str, 
        values: str, 
        title: str = "",
        color: Optional[str] = None,
        height: int = 400,
        template: str = "plotly_white",
        hover_data: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create a pie chart using Plotly Express.
        
        Args:
            df: DataFrame containing the data
            names: Column name for slice names
            values: Column name for slice values
            title: Chart title
            color: Column name for color differentiation
            height: Chart height in pixels
            template: Plotly template name
            hover_data: Additional columns to show in hover tooltip
            
        Returns:
            Plotly figure object
        """
        try:
            fig = px.pie(
                df, 
                names=names, 
                values=values, 
                title=title,
                color=color,
                height=height,
                template=template,
                hover_data=hover_data
            )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                legend_title_text=names.replace('_', ' ').title()
            )
            
            # Add percentage to labels
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            return fig
        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    @staticmethod
    def create_funnel_chart(
        df: pd.DataFrame, 
        x: str, 
        y: str, 
        title: str = "",
        height: int = 400,
        template: str = "plotly_white",
        text: Optional[str] = None
    ) -> go.Figure:
        """
        Create a funnel chart using Plotly Graph Objects.
        
        Args:
            df: DataFrame containing the data
            x: Column name for stage names
            y: Column name for stage values
            title: Chart title
            height: Chart height in pixels
            template: Plotly template name
            text: Column name for text displayed on funnel segments
            
        Returns:
            Plotly figure object
        """
        try:
            # Sort dataframe by values in descending order for proper funnel
            df_sorted = df.sort_values(by=y, ascending=False)
            
            # Create funnel chart
            fig = go.Figure(go.Funnel(
                x=df_sorted[y],
                y=df_sorted[x],
                textposition="inside",
                textinfo="value+percent initial",
                opacity=0.8,
                marker={"color": ["#4B56D2", "#5C73E6", "#6D90FA", "#82C3EC", "#97F5EB", "#ACFCE3"]},
                connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}}
            ))
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                height=height,
                template=template
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating funnel chart: {str(e)}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    @staticmethod
    def create_heatmap(
        df: pd.DataFrame, 
        x: str, 
        y: str, 
        z: str,
        title: str = "",
        height: int = 400,
        template: str = "plotly_white",
        color_scale: str = "Blues"
    ) -> go.Figure:
        """
        Create a heatmap using Plotly Express.
        
        Args:
            df: DataFrame containing the data
            x: Column name for x-axis
            y: Column name for y-axis
            z: Column name for z-axis (values)
            title: Chart title
            height: Chart height in pixels
            template: Plotly template name
            color_scale: Color scale for heatmap
            
        Returns:
            Plotly figure object
        """
        try:
            # Create pivot table if needed
            if df.shape[1] == 3:  # If data is in long format
                pivot_df = df.pivot(index=y, columns=x, values=z)
                fig = px.imshow(
                    pivot_df,
                    title=title,
                    height=height,
                    color_continuous_scale=color_scale,
                    template=template,
                    labels=dict(x=x.replace('_', ' ').title(), y=y.replace('_', ' ').title(), color=z.replace('_', ' ').title())
                )
            else:  # If data is already in wide format
                fig = px.imshow(
                    df,
                    title=title,
                    height=height,
                    color_continuous_scale=color_scale,
                    template=template
                )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
            
            # Add text annotations
            fig.update_traces(text=df.values, texttemplate="%{text}")
            
            return fig
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    @staticmethod
    def create_scatter_chart(
        df: pd.DataFrame, 
        x: str, 
        y: str, 
        title: str = "",
        color: Optional[str] = None,
        size: Optional[str] = None,
        height: int = 400,
        template: str = "plotly_white",
        hover_data: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create a scatter chart using Plotly Express.
        
        Args:
            df: DataFrame containing the data
            x: Column name for x-axis
            y: Column name for y-axis
            title: Chart title
            color: Column name for color differentiation
            size: Column name for point size
            height: Chart height in pixels
            template: Plotly template name
            hover_data: Additional columns to show in hover tooltip
            
        Returns:
            Plotly figure object
        """
        try:
            fig = px.scatter(
                df, 
                x=x, 
                y=y, 
                title=title,
                color=color,
                size=size,
                height=height,
                template=template,
                hover_data=hover_data
            )
            
            # Customize layout
            fig.update_layout(
                title={
                    'text': title,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title=x.replace('_', ' ').title(),
                yaxis_title=y.replace('_', ' ').title(),
                legend_title_text=color.replace('_', ' ').title() if color else None
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating scatter chart: {str(e)}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
