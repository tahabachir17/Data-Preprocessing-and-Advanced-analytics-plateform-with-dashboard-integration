import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        logger.info("DataVisualizer initialized")
    
    def get_plot_suggestions(self, df: pd.DataFrame, x_col: str, y_col: str) -> List[str]:
        """Suggest appropriate plot types based on data types"""
        suggestions = []
        
        x_dtype = df[x_col].dtype
        y_dtype = df[y_col].dtype
        
        x_is_numeric = pd.api.types.is_numeric_dtype(x_dtype)
        y_is_numeric = pd.api.types.is_numeric_dtype(y_dtype)
        x_is_datetime = pd.api.types.is_datetime64_any_dtype(x_dtype)
        y_is_datetime = pd.api.types.is_datetime64_any_dtype(y_dtype)
        
        x_unique = df[x_col].nunique()
        y_unique = df[y_col].nunique()
        
        # Both numeric
        if x_is_numeric and y_is_numeric:
            suggestions.extend(["Scatter Plot", "Line Plot", "Area Plot"])
            if len(df) < 1000:
                suggestions.append("Bubble Plot")
        
        # One categorical, one numeric
        elif (x_is_numeric and not y_is_numeric) or (not x_is_numeric and y_is_numeric):
            suggestions.extend(["Bar Plot", "Box Plot", "Violin Plot"])
            if x_unique <= 20 or y_unique <= 20:
                suggestions.append("Strip Plot")
        
        # Both categorical
        elif not x_is_numeric and not y_is_numeric:
            suggestions.extend(["Bar Plot", "Heatmap"])
            if x_unique <= 10 and y_unique <= 10:
                suggestions.append("Contingency Heatmap")
        
        # Time series
        if x_is_datetime or y_is_datetime:
            suggestions.extend(["Time Series Plot", "Line Plot"])
        
        # Distribution plots for single variables
        if x_col == y_col or x_is_numeric:
            suggestions.extend(["Histogram", "Density Plot"])
        
        return list(set(suggestions)) if suggestions else ["Bar Plot", "Scatter Plot"]
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                           color_col: Optional[str] = None, size_col: Optional[str] = None,
                           title: Optional[str] = None) -> go.Figure:
        """Create an interactive scatter plot"""
        try:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                color=color_col,
                size=size_col,
                title=title or f"{y_col} vs {x_col}",
                template="plotly_white",
                hover_data=df.columns.tolist()
            )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                showlegend=bool(color_col),
                height=600
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            raise
    
    def create_line_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                        color_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create an interactive line plot"""
        try:
            fig = px.line(
                df, x=x_col, y=y_col,
                color=color_col,
                title=title or f"{y_col} over {x_col}",
                template="plotly_white",
                markers=True
            )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                showlegend=bool(color_col),
                height=600
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating line plot: {e}")
            raise
    
    def create_bar_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                       color_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create an interactive bar plot"""
        try:
            # Aggregate data if needed
            if pd.api.types.is_numeric_dtype(df[y_col]):
                agg_df = df.groupby(x_col)[y_col].sum().reset_index()
            else:
                agg_df = df[x_col].value_counts().reset_index()
                agg_df.columns = [x_col, 'count']
                y_col = 'count'
            
            fig = px.bar(
                agg_df, x=x_col, y=y_col,
                color=color_col,
                title=title or f"{y_col} by {x_col}",
                template="plotly_white"
            )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                showlegend=bool(color_col),
                height=600
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating bar plot: {e}")
            raise
    
    def create_histogram(self, df: pd.DataFrame, x_col: str, 
                        color_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create an interactive histogram"""
        try:
            fig = px.histogram(
                df, x=x_col,
                color=color_col,
                title=title or f"Distribution of {x_col}",
                template="plotly_white",
                marginal="box"
            )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title="Count",
                showlegend=bool(color_col),
                height=600
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            raise
    
    def create_box_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                       color_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create an interactive box plot"""
        try:
            fig = px.box(
                df, x=x_col, y=y_col,
                color=color_col,
                title=title or f"{y_col} distribution by {x_col}",
                template="plotly_white"
            )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                showlegend=bool(color_col),
                height=600
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating box plot: {e}")
            raise
    
    def create_violin_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                          color_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create an interactive violin plot"""
        try:
            fig = px.violin(
                df, x=x_col, y=y_col,
                color=color_col,
                title=title or f"{y_col} distribution by {x_col}",
                template="plotly_white",
                box=True
            )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                showlegend=bool(color_col),
                height=600
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating violin plot: {e}")
            raise
    
    def create_heatmap(self, df: pd.DataFrame, x_col: str, y_col: str,
                      value_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create an interactive heatmap"""
        try:
            if value_col:
                pivot_df = df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc='mean')
            else:
                # Create contingency table
                pivot_df = pd.crosstab(df[y_col], df[x_col])
            
            fig = px.imshow(
                pivot_df,
                title=title or f"Heatmap of {x_col} vs {y_col}",
                template="plotly_white",
                aspect="auto"
            )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=600
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            raise
    
    def create_density_plot(self, df: pd.DataFrame, x_col: str,
                           color_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create a density plot"""
        try:
            fig = go.Figure()
            
            if color_col and color_col in df.columns:
                for category in df[color_col].unique():
                    subset = df[df[color_col] == category]
                    fig.add_trace(go.Histogram(
                        x=subset[x_col],
                        histnorm='probability density',
                        name=str(category),
                        opacity=0.7
                    ))
            else:
                fig.add_trace(go.Histogram(
                    x=df[x_col],
                    histnorm='probability density',
                    name=x_col,
                    opacity=0.7
                ))
            
            fig.update_layout(
                title=title or f"Density Distribution of {x_col}",
                xaxis_title=x_col,
                yaxis_title="Density",
                template="plotly_white",
                height=600
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating density plot: {e}")
            raise
    
    def create_area_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                        color_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create an area plot"""
        try:
            fig = px.area(
                df, x=x_col, y=y_col,
                color=color_col,
                title=title or f"{y_col} over {x_col} (Area)",
                template="plotly_white"
            )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=600
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating area plot: {e}")
            raise
    
    def create_bubble_plot(self, df: pd.DataFrame, x_col: str, y_col: str, size_col: str,
                          color_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
        """Create a bubble plot"""
        try:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                size=size_col,
                color=color_col,
                title=title or f"Bubble Plot: {y_col} vs {x_col}",
                template="plotly_white",
                hover_data=df.columns.tolist()
            )
            
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=600
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating bubble plot: {e}")
            raise
    
    def create_correlation_heatmap(self, df: pd.DataFrame, title: Optional[str] = None) -> go.Figure:
        """Create a correlation heatmap for numeric columns"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise ValueError("No numeric columns found for correlation")
            
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(
                corr_matrix,
                title=title or "Correlation Matrix",
                template="plotly_white",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            
            fig.update_layout(height=600)
            
            return fig
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            raise
    
    def create_plot(self, df: pd.DataFrame, plot_type: str, x_col: str, y_col: str = None,
                   color_col: str = None, size_col: str = None, title: str = None) -> go.Figure:
        """Main method to create plots based on type"""
        plot_methods = {
            "Scatter Plot": self.create_scatter_plot,
            "Line Plot": self.create_line_plot,
            "Bar Plot": self.create_bar_plot,
            "Histogram": self.create_histogram,
            "Box Plot": self.create_box_plot,
            "Violin Plot": self.create_violin_plot,
            "Heatmap": self.create_heatmap,
            "Density Plot": self.create_density_plot,
            "Area Plot": self.create_area_plot,
            "Bubble Plot": self.create_bubble_plot,
            "Time Series Plot": self.create_line_plot,
            "Correlation Heatmap": self.create_correlation_heatmap
        }
        
        if plot_type not in plot_methods:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        method = plot_methods[plot_type]
        
        # Prepare arguments based on plot type
        if plot_type in ["Histogram", "Density Plot"]:
            return method(df, x_col, color_col=color_col, title=title)
        elif plot_type == "Correlation Heatmap":
            return method(df, title=title)
        elif plot_type == "Bubble Plot" and size_col:
            return method(df, x_col, y_col, size_col, color_col=color_col, title=title)
        else:
            return method(df, x_col, y_col, color_col=color_col, title=title)
    
    def export_plot_html(self, fig: go.Figure, filename: str = "plot.html") -> str:
        """Export plot as HTML string"""
        try:
            html_str = fig.to_html(include_plotlyjs='cdn')
            return html_str
        except Exception as e:
            logger.error(f"Error exporting plot to HTML: {e}")
            raise
    
    def export_plot_image(self, fig: go.Figure, format: str = "png", width: int = 1200, height: int = 600) -> bytes:
        """Export plot as image bytes"""
        try:
            img_bytes = fig.to_image(format=format, width=width, height=height)
            return img_bytes
        except Exception as e:
            logger.error(f"Error exporting plot to image: {e}")
            raise
    
    def get_basic_stats(self, df: pd.DataFrame, x_col: str, y_col: str = None) -> Dict:
        """Get basic statistics for the selected columns"""
        try:
            stats = {}
            
            # X column stats
            if pd.api.types.is_numeric_dtype(df[x_col]):
                stats[x_col] = {
                    'count': df[x_col].count(),
                    'mean': df[x_col].mean(),
                    'std': df[x_col].std(),
                    'min': df[x_col].min(),
                    'max': df[x_col].max(),
                    'missing': df[x_col].isnull().sum()
                }
            else:
                stats[x_col] = {
                    'count': df[x_col].count(),
                    'unique': df[x_col].nunique(),
                    'top': df[x_col].mode().iloc[0] if not df[x_col].mode().empty else None,
                    'missing': df[x_col].isnull().sum()
                }
            
            # Y column stats
            if y_col and y_col in df.columns:
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    stats[y_col] = {
                        'count': df[y_col].count(),
                        'mean': df[y_col].mean(),
                        'std': df[y_col].std(),
                        'min': df[y_col].min(),
                        'max': df[y_col].max(),
                        'missing': df[y_col].isnull().sum()
                    }
                else:
                    stats[y_col] = {
                        'count': df[y_col].count(),
                        'unique': df[y_col].nunique(),
                        'top': df[y_col].mode().iloc[0] if not df[y_col].mode().empty else None,
                        'missing': df[y_col].isnull().sum()
                    }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting basic stats: {e}")
            return {}