import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import io
import base64
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Dashboard:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        logger.info("Dashboard initialized")
    
    def render_column_selector(self, df: pd.DataFrame, key_prefix: str = "") -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        """Render column selectors with intelligent defaults"""
        
        # Get column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # X-axis selection
            x_options = all_cols
            x_default = None
            
            # Smart defaults for X-axis
            if datetime_cols:
                x_default = datetime_cols[0]
            elif categorical_cols:
                x_default = categorical_cols[0]
            elif numeric_cols:
                x_default = numeric_cols[0]
            
            x_col = st.selectbox(
                "📊 Select X-axis column:",
                x_options,
                index=x_options.index(x_default) if x_default else 0,
                key=f"{key_prefix}_x_col",
                help="Choose the column for the X-axis"
            )
        
        with col2:
            # Y-axis selection
            y_options = ["None"] + all_cols
            y_default = None
            
            # Smart defaults for Y-axis
            if len(numeric_cols) >= 2:
                y_default = [col for col in numeric_cols if col != x_col][0]
            elif len(numeric_cols) == 1 and numeric_cols[0] != x_col:
                y_default = numeric_cols[0]
            elif numeric_cols and x_col not in numeric_cols:
                y_default = numeric_cols[0]
            
            y_col = st.selectbox(
                "📈 Select Y-axis column:",
                y_options,
                index=y_options.index(y_default) if y_default else 0,
                key=f"{key_prefix}_y_col",
                help="Choose the column for the Y-axis (optional for some plot types)"
            )
            
            if y_col == "None":
                y_col = None
        
        # Additional options in expandable section
        with st.expander("🎨 Additional Options", expanded=False):
            col3, col4 = st.columns(2)
            
            with col3:
                # Color column
                color_options = ["None"] + categorical_cols + numeric_cols
                color_col = st.selectbox(
                    "🎨 Color by column:",
                    color_options,
                    key=f"{key_prefix}_color_col",
                    help="Optional: Color points/bars by this column"
                )
                if color_col == "None":
                    color_col = None
            
            with col4:
                # Size column (for bubble plots)
                size_options = ["None"] + numeric_cols
                size_col = st.selectbox(
                    "📏 Size by column:",
                    size_options,
                    key=f"{key_prefix}_size_col",
                    help="Optional: Size points by this column (for bubble plots)"
                )
                if size_col == "None":
                    size_col = None
        
        return x_col, y_col, color_col, size_col
    
    def render_plot_type_selector(self, df: pd.DataFrame, x_col: str, y_col: str, key_prefix: str = "") -> str:
        """Render plot type selector with suggestions"""
        
        # Get suggestions
        if y_col:
            suggestions = self.visualizer.get_plot_suggestions(df, x_col, y_col)
        else:
            suggestions = ["Histogram", "Density Plot", "Bar Plot"]
        
        # All available plot types
        all_plot_types = [
            "Scatter Plot", "Line Plot", "Bar Plot", "Histogram", 
            "Box Plot", "Violin Plot", "Heatmap", "Density Plot",
            "Area Plot", "Bubble Plot", "Time Series Plot", "Correlation Heatmap"
        ]
        
        # Organize plot types
        suggested_types = [pt for pt in suggestions if pt in all_plot_types]
        other_types = [pt for pt in all_plot_types if pt not in suggested_types]
        
        # Display suggestions
        if suggested_types:
            st.write("💡 **Suggested plot types based on your data:**")
            cols = st.columns(min(len(suggested_types), 4))
            for i, plot_type in enumerate(suggested_types[:4]):
                with cols[i]:
                    if st.button(f"📊 {plot_type}", key=f"{key_prefix}_suggest_{i}", use_container_width=True):
                        st.session_state[f"{key_prefix}_selected_plot"] = plot_type
        
        # Plot type selection
        default_plot = suggested_types[0] if suggested_types else "Scatter Plot"
        if f"{key_prefix}_selected_plot" in st.session_state:
            default_plot = st.session_state[f"{key_prefix}_selected_plot"]
        
        plot_type = st.selectbox(
            "📈 Select plot type:",
            suggested_types + ["---"] + other_types,
            index=all_plot_types.index(default_plot) if default_plot in all_plot_types else 0,
            key=f"{key_prefix}_plot_type",
            help="Choose the type of visualization"
        )
        
        return plot_type
    
    def render_plot_customization(self, key_prefix: str = "") -> Dict:
        """Render plot customization options"""
        
        with st.expander("🎛️ Plot Customization", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input(
                    "Plot Title:",
                    placeholder="Enter custom title (optional)",
                    key=f"{key_prefix}_title"
                )
                
                width = st.slider(
                    "Plot Width (px):",
                    min_value=600,
                    max_value=1600,
                    value=1200,
                    step=100,
                    key=f"{key_prefix}_width"
                )
            
            with col2:
                height = st.slider(
                    "Plot Height (px):",
                    min_value=400,
                    max_value=1000,
                    value=600,
                    step=50,
                    key=f"{key_prefix}_height"
                )
                
                template = st.selectbox(
                    "Plot Theme:",
                    ["plotly_white", "plotly_dark", "simple_white", "ggplot2", "seaborn", "plotly"],
                    key=f"{key_prefix}_template"
                )
        
        return {
            'title': title if title else None,
            'width': width,
            'height': height,
            'template': template
        }
    
    def render_data_filter(self, df: pd.DataFrame, key_prefix: str = "") -> pd.DataFrame:
        """Render data filtering options"""
        
        with st.expander("🔍 Data Filtering", expanded=False):
            st.write("Filter your data before plotting:")
            
            # Row sampling
            if len(df) > 1000:
                sample_data = st.checkbox(
                    f"Sample data (dataset has {len(df):,} rows)",
                    key=f"{key_prefix}_sample",
                    help="Sample data for better performance"
                )
                
                if sample_data:
                    sample_size = st.slider(
                        "Sample size:",
                        min_value=100,
                        max_value=min(10000, len(df)),
                        value=min(1000, len(df)),
                        key=f"{key_prefix}_sample_size"
                    )
                    df = df.sample(n=sample_size, random_state=42)
                    st.info(f"Using {len(df):,} sampled rows")
            
            # Column-based filtering
            filter_col = st.selectbox(
                "Filter by column:",
                ["None"] + df.columns.tolist(),
                key=f"{key_prefix}_filter_col"
            )
            
            if filter_col != "None":
                col_data = df[filter_col]
                
                if pd.api.types.is_numeric_dtype(col_data):
                    # Numeric filtering
                    min_val, max_val = float(col_data.min()), float(col_data.max())
                    filter_range = st.slider(
                        f"Filter {filter_col} range:",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"{key_prefix}_filter_range"
                    )
                    df = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]
                
                else:
                    # Categorical filtering
                    unique_values = col_data.unique().tolist()
                    selected_values = st.multiselect(
                        f"Select {filter_col} values:",
                        unique_values,
                        default=unique_values,
                        key=f"{key_prefix}_filter_values"
                    )
                    if selected_values:
                        df = df[df[filter_col].isin(selected_values)]
                
                st.info(f"Filtered data: {len(df):,} rows")
        
        return df
    
    def render_statistics_panel(self, df: pd.DataFrame, x_col: str, y_col: str = None):
        """Render statistics panel"""
        
        with st.expander("📊 Data Statistics", expanded=False):
            stats = self.visualizer.get_basic_stats(df, x_col, y_col)
            
            if stats:
                cols = st.columns(len(stats))
                
                for i, (col_name, col_stats) in enumerate(stats.items()):
                    with cols[i]:
                        st.write(f"**{col_name}**")
                        
                        for stat_name, stat_value in col_stats.items():
                            if isinstance(stat_value, (int, float)):
                                if stat_name in ['mean', 'std']:
                                    st.write(f"{stat_name.title()}: {stat_value:.3f}")
                                else:
                                    st.write(f"{stat_name.title()}: {stat_value:,.0f}")
                            else:
                                st.write(f"{stat_name.title()}: {stat_value}")
    
    def render_export_options(self, fig: go.Figure, plot_name: str = "plot"):
        """Render export options"""
        
        st.write("### 📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as HTML
            if st.button("📄 Export as HTML", use_container_width=True):
                html_str = self.visualizer.export_plot_html(fig)
                st.download_button(
                    label="💾 Download HTML",
                    data=html_str,
                    file_name=f"{plot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
        
        with col2:
            # Export as PNG
            if st.button("🖼️ Export as PNG", use_container_width=True):
                try:
                    img_bytes = self.visualizer.export_plot_image(fig, format="png")
                    st.download_button(
                        label="💾 Download PNG",
                        data=img_bytes,
                        file_name=f"{plot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error exporting PNG: {e}")
        
        with col3:
            # Export as PDF
            if st.button("📑 Export as PDF", use_container_width=True):
                try:
                    pdf_bytes = self.visualizer.export_plot_image(fig, format="pdf")
                    st.download_button(
                        label="💾 Download PDF",
                        data=pdf_bytes,
                        file_name=f"{plot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error exporting PDF: {e}")
    
    def render_plot_gallery(self, plots_history: List[Dict]):
        """Render a gallery of previously created plots"""
        
        if not plots_history:
            st.info("No plots created yet. Create your first visualization above!")
            return
        
        st.write("### 🖼️ Plot Gallery")
        
        # Display plots in a grid
        cols_per_row = 2
        for i in range(0, len(plots_history), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                plot_idx = i + j
                if plot_idx < len(plots_history):
                    plot_info = plots_history[plot_idx]
                    
                    with col:
                        st.write(f"**{plot_info.get('title', f'Plot {plot_idx + 1}')}**")
                        st.write(f"Type: {plot_info['plot_type']}")
                        st.write(f"Created: {plot_info['timestamp']}")
                        
                        # Show thumbnail or recreate plot
                        if 'figure' in plot_info:
                            st.plotly_chart(plot_info['figure'], use_container_width=True, key=f"gallery_plot_{plot_idx}")
    
    def create_comprehensive_dashboard(self, df: pd.DataFrame, key_prefix: str = "main") -> Optional[go.Figure]:
        """Create the main visualization dashboard"""
        
        if df is None or df.empty:
            st.warning("⚠️ No data available for visualization.")
            return None
        
        st.subheader("📊 Interactive Data Visualization")
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_cols)
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        st.markdown("---")
        
        # Filter data first
        filtered_df = self.render_data_filter(df, key_prefix)
        
        # Column selection
        x_col, y_col, color_col, size_col = self.render_column_selector(filtered_df, key_prefix)
        
        # Plot type selection
        plot_type = self.render_plot_type_selector(filtered_df, x_col, y_col, key_prefix)
        
        # Plot customization
        custom_options = self.render_plot_customization(key_prefix)
        
        # Validation
        if plot_type == "---":
            st.warning("Please select a valid plot type.")
            return None
        
        # Special validations for specific plot types
        if plot_type in ["Scatter Plot", "Line Plot", "Box Plot", "Violin Plot", "Heatmap", "Area Plot"] and not y_col:
            st.warning(f"{plot_type} requires both X and Y columns.")
            return None
        
        if plot_type == "Bubble Plot" and not size_col:
            st.warning("Bubble Plot requires a size column.")
            return None
        
        # Create the plot
        try:
            with st.spinner(f"Creating {plot_type.lower()}..."):
                fig = self.visualizer.create_plot(
                    filtered_df, 
                    plot_type, 
                    x_col, 
                    y_col=y_col,
                    color_col=color_col, 
                    size_col=size_col,
                    title=custom_options['title']
                )
                
                # Apply customizations
                fig.update_layout(
                    template=custom_options['template'],
                    height=custom_options['height'],
                    width=custom_options['width']
                )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_main_plot")
            
            # Show statistics
            self.render_statistics_panel(filtered_df, x_col, y_col)
            
            # Export options
            self.render_export_options(fig, f"{plot_type.replace(' ', '_').lower()}")
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating plot: {str(e)}")
            logger.error(f"Plot creation error: {e}")
            return None
    
    def create_comparison_dashboard(self, df: pd.DataFrame):
        """Create a dashboard for comparing multiple visualizations"""
        
        st.subheader("🔄 Comparison Dashboard")
        st.write("Compare different visualizations of the same data side by side.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Plot 1**")
            fig1 = self.create_comprehensive_dashboard(df, "comp1")
        
        with col2:
            st.write("**📈 Plot 2**")
            fig2 = self.create_comprehensive_dashboard(df, "comp2")
        
        return fig1, fig2
    
    def create_multi_variable_dashboard(self, df: pd.DataFrame):
        """Create dashboard for multi-variable analysis"""
        
        st.subheader("🔬 Multi-Variable Analysis")
        
        # Correlation matrix for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            st.write("### 📊 Correlation Analysis")
            
            if st.button("🔍 Generate Correlation Heatmap", type="primary"):
                try:
                    fig = self.visualizer.create_correlation_heatmap(numeric_df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export options for correlation heatmap
                    self.render_export_options(fig, "correlation_heatmap")
                    
                except Exception as e:
                    st.error(f"Error creating correlation heatmap: {e}")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.selectbox("Select a target column:", numeric_cols)
                
        if target_col:
            if st.button("Show Target Correlation Matrix"):
                try:
                    corr_matrix = df.corr()[[target_col]].sort_values(by=target_col, ascending=False)
                    st.write(f"### 📊 Correlation Matrix for Target: **{target_col}**")
                    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
                except Exception as e:
                    st.error(f"Error generating target correlation matrix: {str(e)}")
                
        
        # Pair plot simulation
        if len(numeric_df.columns) >= 2:
            st.write("### 📈 Pairwise Relationships")
            
            selected_vars = st.multiselect(
                "Select variables for pairwise analysis:",
                numeric_df.columns.tolist(),
                default=numeric_df.columns.tolist()[:4] if len(numeric_df.columns) >= 4 else numeric_df.columns.tolist()
            )
            
            if len(selected_vars) >= 2:
                if st.button("📊 Generate Pairwise Plots"):
                    # Create subplot for pairwise relationships
                    from plotly.subplots import make_subplots
                    
                    n_vars = len(selected_vars)
                    fig = make_subplots(
                        rows=n_vars, cols=n_vars,
                        subplot_titles=[f"{var1} vs {var2}" for var1 in selected_vars for var2 in selected_vars]
                    )
                    
                    for i, var1 in enumerate(selected_vars):
                        for j, var2 in enumerate(selected_vars):
                            if i == j:
                                # Diagonal: histogram
                                fig.add_trace(
                                    go.Histogram(x=df[var1], name=var1, showlegend=False),
                                    row=i+1, col=j+1
                                )
                            else:
                                # Off-diagonal: scatter plot
                                fig.add_trace(
                                    go.Scatter(x=df[var2], y=df[var1], mode='markers', 
                                             name=f"{var1} vs {var2}", showlegend=False),
                                    row=i+1, col=j+1
                                )
                    
                    fig.update_layout(height=800, title="Pairwise Variable Relationships")
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_plot_insights(self, df: pd.DataFrame, x_col: str, y_col: str = None):
        """Generate and display automated insights about the plot"""
        
        with st.expander("🧠 Automated Insights", expanded=False):
            insights = []
            
            # Basic data insights
            total_rows = len(df)
            insights.append(f"📊 Dataset contains **{total_rows:,}** data points")
            
            # X column insights
            if pd.api.types.is_numeric_dtype(df[x_col]):
                x_range = df[x_col].max() - df[x_col].min()
                insights.append(f"📈 {x_col} ranges from **{df[x_col].min():.2f}** to **{df[x_col].max():.2f}** (range: {x_range:.2f})")
                
                # Check for outliers
                Q1 = df[x_col].quantile(0.25)
                Q3 = df[x_col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[x_col] < Q1 - 1.5*IQR) | (df[x_col] > Q3 + 1.5*IQR)]
                if len(outliers) > 0:
                    insights.append(f"⚠️ Detected **{len(outliers)}** potential outliers in {x_col}")
            else:
                unique_vals = df[x_col].nunique()
                insights.append(f"🏷️ {x_col} has **{unique_vals}** unique categories")
            
            # Y column insights
            if y_col and pd.api.types.is_numeric_dtype(df[y_col]):
                correlation = df[x_col].corr(df[y_col]) if pd.api.types.is_numeric_dtype(df[x_col]) else None
                if correlation is not None:
                    corr_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
                    corr_direction = "positive" if correlation > 0 else "negative"
                    insights.append(f"🔗 **{corr_strength.title()} {corr_direction}** correlation between {x_col} and {y_col} (r = {correlation:.3f})")
            
            # Missing data insights
            missing_x = df[x_col].isnull().sum()
            if missing_x > 0:
                insights.append(f"❓ {x_col} has **{missing_x}** missing values ({missing_x/total_rows*100:.1f}%)")
            
            if y_col:
                missing_y = df[y_col].isnull().sum()
                if missing_y > 0:
                    insights.append(f"❓ {y_col} has **{missing_y}** missing values ({missing_y/total_rows*100:.1f}%)")
            
            # Display insights
            for insight in insights:
                st.write(f"• {insight}")
            
            if not insights:
                st.write("No specific insights available for this data selection.")