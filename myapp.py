import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import tempfile
import os
import logging
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Import your custom classes
from src.data_processing.cleaner import DataCleaner
from src.data_processing.loader import DataLoader
from src.data_processing.transformer import DataTransformer
from src.visualization.charts import DataVisualizer
from src.visualization.dashboard import Dashboard
from src.analytics.ml_models import MLModels
from src.analytics.statistical import StatisticalAnalyzer
from src.analytics.advanced_analytics import AdvancedAnalytics

# Configuration du logging pour Streamlit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Page configuration with professional styling
st.set_page_config(
    page_title="DataFlow Pro - Advanced Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #cce7ff;
        color: #004085;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #b3d9ff;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    .nav-button {
        width: 100%;
        margin: 0.2rem 0;
        background: white;
        border: 1px solid #ddd;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: left;
    }
    
    .nav-button:hover {
        background: #f0f0f0;
        border-color: #2a5298;
    }
    
    .stSelectbox > div > div {
        background-color: white;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    .plot-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'df_original': None,  # Original raw data
        'df_cleaned': None,   # Cleaned data - MAIN WORKING DATAFRAME
        'df_second': None,    # Second dataset for merging
        'df_grouped': None,   # Independent grouped data
        'df_merged': None,    # Independent merged data
        'df_encoded': None,   # Independent encoded data
        'loader': DataLoader(),
        'cleaner': DataCleaner(),
        'transformer': DataTransformer(),
        'visualizer': DataVisualizer(),
        'dashboard': None,
        'ml_models': MLModels(),
        'processing_logs': [],
        'plots_history': [],
        'current_plot': None,
        'plot_counter': 0,
        'current_page': 'data_pipeline',  # Default page
        'selected_target': None,
        'cleaning_applied': False,
        'grouping_applied': False,
        'merging_applied': False,
        'encoding_applied': False
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value
    
    if st.session_state.dashboard is None:
        st.session_state.dashboard = Dashboard(st.session_state.visualizer)

def add_log(message, level="INFO"):
    """Add a log entry to session state"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}"
    st.session_state.processing_logs.append(log_entry)
    
    # Keep only last 50 logs
    if len(st.session_state.processing_logs) > 50:
        st.session_state.processing_logs = st.session_state.processing_logs[-50:]

def save_plot_to_history(plot_info):
    """Save plot information to history"""
    plot_info['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.plots_history.append(plot_info)
    
    # Keep only last 20 plots
    if len(st.session_state.plots_history) > 20:
        st.session_state.plots_history = st.session_state.plots_history[-20:]

def create_download_link(df, filename, file_format='csv'):
    """Create download link for dataframes"""
    if file_format == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📄 Download CSV</a>'
    elif file_format == 'excel':
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">📊 Download Excel</a>'
    
    return href

def export_plot_as_html(fig, filename):
    """Export plotly figure as HTML"""
    html_string = fig.to_html(include_plotlyjs='cdn')
    b64 = base64.b64encode(html_string.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html">📊 Download Plot (HTML)</a>'
    return href

def compare_datasets(df_original, df_processed, title="Dataset Comparison"):
    """Compare original vs processed datasets"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Original Dataset**")
        st.metric("Rows", f"{len(df_original):,}")
        st.metric("Columns", f"{len(df_original.columns):,}")
        st.metric("Missing Values", f"{df_original.isnull().sum().sum():,}")
        st.metric("Memory (MB)", f"{df_original.memory_usage(deep=True).sum() / 1024**2:.1f}")
    
    with col2:
        st.write(f"**{title}**")
        st.metric("Rows", f"{len(df_processed):,}", delta=f"{len(df_processed) - len(df_original):,}")
        st.metric("Columns", f"{len(df_processed.columns):,}", delta=f"{len(df_processed.columns) - len(df_original.columns):,}")
        st.metric("Missing Values", f"{df_processed.isnull().sum().sum():,}", delta=f"{df_processed.isnull().sum().sum() - df_original.isnull().sum().sum():,}")
        st.metric("Memory (MB)", f"{df_processed.memory_usage(deep=True).sum() / 1024**2:.1f}", delta=f"{(df_processed.memory_usage(deep=True).sum() - df_original.memory_usage(deep=True).sum()) / 1024**2:.1f}")

def show_target_selection_regression_only():
    """Target selection for regression only"""
    st.subheader("🎯 Target Variable Selection (Regression Only)")
    
    # Display only numeric column information
    with st.expander("📊 Available Numeric Columns (Click to expand)", expanded=True):
        col_info = st.session_state.ml_models.get_available_targets(current_df)
        if not col_info.empty:
            st.dataframe(col_info, use_container_width=True)
            st.info("ℹ️ Only numeric columns are shown as this system performs regression only.")
        else:
            st.error("❌ No numeric columns found for regression.")
            return None
    
    # Target selection - only numeric columns
    numeric_columns = current_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        st.error("❌ No numeric columns available for regression target.")
        return None
    
    target_column = st.selectbox(
        "Select Target Column for Regression:",
        options=numeric_columns,
        index=len(numeric_columns) - 1,
        help="Choose the numeric column you want to predict (regression target)"
    )
    
    if target_column:
        target_info = current_df[target_column]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Unique Values", target_info.nunique())
        with col2:
            st.metric("Missing Values", target_info.isnull().sum())
        with col3:
            st.metric("Min Value", f"{target_info.min():.2f}")
        with col4:
            st.metric("Max Value", f"{target_info.max():.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean", f"{target_info.mean():.4f}")
        with col2:
            st.metric("Std Dev", f"{target_info.std():.4f}")
        
        # Show sample values
        st.write(f"**Sample values from '{target_column}':**")
        sample_values = target_info.dropna().head(10).tolist()
        st.write(", ".join([f"{val:.4f}" for val in sample_values]))
        
        # Validation warnings
        if target_info.nunique() < 3:
            st.error(f"❌ Target has only {target_info.nunique()} unique values. Need at least 3 for regression.")
            return None
        
        if target_info.std() == 0:
            st.error("❌ Target has no variation. Cannot perform regression.")
            return None
    
    return target_column

def show_ml_configuration_regression():
    """ML configuration for regression only"""
    st.subheader("⚙️ Regression Configuration")
    st.info("🤖 All parameters are automatically optimized for regression tasks.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Correlation Threshold", "0.01", help="Features with correlation < 0.01 will be removed")
    
    with col2:
        st.metric("Test Set Size", "20%", help="Automatically set to 20% of data")
    
    with col3:
        st.metric("Cross-Validation", "5-fold", help="Automatically set to 5-fold CV")
    
    with st.expander("🔧 Regression Models to be Trained", expanded=False):
        models_info = """
        **Linear Models:**
        - Linear Regression
        - Ridge Regression (L2 regularization)
        - Lasso Regression (L1 regularization)  
        - ElasticNet (L1 + L2 regularization)
        
        **Tree-based Models:**
        - Random Forest Regressor
        - Decision Tree Regressor
        - Gradient Boosting Regressor
        
        **Other Models:**
        - Support Vector Regression (SVR)
        - K-Nearest Neighbors Regressor
        - Neural Network (MLP Regressor)
        """
        st.markdown(models_info)

def create_model_summary_dataframe(model_scores, best_model_name):
    """Create model summary dataframe"""
    summary = []
    for name, info in model_scores.items():
        summary.append({
            'Model': name,
            'R² Score': round(info['mean_score'], 4),
            'CV Std': round(info['std_score'], 4),
            'Is Best': name == best_model_name
        })
    
    return pd.DataFrame(summary).sort_values('R² Score', ascending=False)

def display_regression_results():
    """Display results specifically for regression"""
    if hasattr(st.session_state, 'ml_results') and st.session_state.ml_results is not None:
        st.markdown("---")
        st.subheader("📊 Regression Results")
        
        results = st.session_state.ml_results
        
        # Quick summary at the top
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏆 Best Model", results['best_model_name'])
        with col2:
            st.metric("📊 Problem Type", "REGRESSION")
        with col3:
            cv_score = results['best_score']
            st.metric("📈 R² Score", f"{cv_score:.4f}")
        with col4:
            feature_count = len(st.session_state.ml_models.feature_names) if st.session_state.ml_models.feature_names else 0
            st.metric("🎯 Features Used", feature_count)
        
        # Results tabs for regression
        result_tabs = st.tabs([
            "🏆 Best Model", 
            "📈 Model Comparison", 
            "🎯 Feature Importance", 
            "📊 Prediction Analysis",
            "📋 Training Summary", 
            "🔮 Make Predictions"
        ])
        
        with result_tabs[0]:
            st.markdown("### 🏆 Best Performing Regression Model")
            
            best_model_info = f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4>🤖 {results['best_model_name']}</h4>
                <p><strong>Problem Type:</strong> Regression</p>
                <p><strong>R² Cross-Validation Score:</strong> {results['best_score']:.4f}</p>
                <p><strong>Dataset Used:</strong> {selected_dataset}</p>
            </div>
            """
            st.markdown(best_model_info, unsafe_allow_html=True)
            
            # Test set performance for regression
            if 'test_metrics' in results:
                st.write("### 🎯 Test Set Performance")
                
                metrics = results['test_metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                with col3:
                    st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                with col4:
                    mape = metrics.get('mape', 0)
                    st.metric("MAPE", f"{mape:.2f}%")
        
        with result_tabs[1]:
            st.write("### 📈 Regression Model Performance Comparison")
            
            if 'model_scores' in results:
                summary_df = create_model_summary_dataframe(results['model_scores'], results['best_model_name'])
                
                # Style the dataframe for regression
                styled_df = summary_df.style.highlight_max(subset=['R² Score'], color='lightgreen')
                st.dataframe(styled_df, use_container_width=True)
                
                # Performance comparison chart for regression
                fig = px.bar(
                    summary_df,
                    x='Model',
                    y='R² Score',
                    color='Is Best',
                    title=f"Regression Model R² Score Comparison ({selected_dataset})",
                    text='R² Score',
                    color_discrete_map={True: '#1f77b4', False: '#d62728'}
                )

                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=500,
                    showlegend=True,
                    legend=dict(title="Best Model"),
                    yaxis_title="R² Score"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with result_tabs[2]:
            st.write("### 🎯 Feature Importance Analysis")
            
            if 'feature_importance' in results and results['feature_importance'] is not None:
                importance_df = results['feature_importance']
                
                # Show top features
                st.write("**Top 15 Most Important Features for Regression:**")
                top_features = importance_df.head(15)
                st.dataframe(top_features, use_container_width=True)
                
                # Feature importance plot
                fig = px.bar(
                    top_features,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Feature Importance for Regression",
                    labels={'importance': 'Importance Score', 'feature': 'Features'},
                    color='importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with result_tabs[3]:
            st.write("### 📊 Prediction Analysis")
            
            if 'test_metrics' in results and 'predictions' in results['test_metrics']:
                predictions = results['test_metrics']['predictions']
                actual = st.session_state.ml_models.y_test
                
                # Prediction vs Actual scatter plot
                fig_scatter = px.scatter(
                    x=actual, 
                    y=predictions,
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    title="Predicted vs Actual Values"
                )
                
                # Add perfect prediction line
                min_val = min(min(actual), min(predictions))
                max_val = max(max(actual), max(predictions))
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[min_val, max_val], 
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Residuals plot
                residuals = results['test_metrics']['residuals']
                fig_residuals = px.scatter(
                    x=predictions,
                    y=residuals,
                    labels={'x': 'Predicted Values', 'y': 'Residuals'},
                    title="Residuals Plot"
                )
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residuals, use_container_width=True)
        
        with result_tabs[4]:
            st.write("### 📋 Training Summary")
            
            if 'training_summary' in results:
                summary = results['training_summary']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dataset Information:**")
                    st.write(f"- Target Column: {summary.get('target_column', 'N/A')}")
                    st.write(f"- Original Features: {summary.get('original_features', 'N/A')}")
                    st.write(f"- Final Features: {summary.get('final_features', 'N/A')}")
                    st.write(f"- Train Size: {summary.get('train_size', 'N/A')}")
                    st.write(f"- Test Size: {summary.get('test_size', 'N/A')}")
                
                with col2:
                    st.write("**Model Information:**")
                    st.write(f"- Best Model: {summary.get('best_model', 'N/A')}")
                    st.write(f"- Best Score: {summary.get('best_score', 'N/A'):.4f}")
                    
                    if summary.get('removed_features'):
                        st.write(f"- Removed Features: {len(summary['removed_features'])}")
                        with st.expander("Show removed features"):
                            st.write(summary['removed_features'])
        
        with result_tabs[5]:
            st.write("### 🔮 Make Predictions on New Data")
            
            st.info("📁 Upload a dataset without the target column to make predictions")
            
            # File uploader for prediction data
            prediction_file = st.file_uploader(
                "Choose a CSV or Excel file for predictions:",
                type=['csv', 'xlsx'],
                key="prediction_data",
                help="Upload data without the target column to get predictions"
            )
            
            if prediction_file is not None:
                try:
                    # Load prediction data
                    if prediction_file.name.endswith('.csv'):
                        pred_df = pd.read_csv(prediction_file)
                    else:
                        pred_df = pd.read_excel(prediction_file)
                    
                    st.success(f"✅ Prediction data loaded: {pred_df.shape[0]:,} rows × {pred_df.shape[1]} columns")
                    
                    # Show preview
                    st.write("**Data Preview:**")
                    st.dataframe(pred_df.head(10), use_container_width=True)
                    
                    # Check if target column is present (it shouldn't be)
                    target_col = st.session_state.ml_models.target_column
                    if target_col and target_col in pred_df.columns:
                        st.warning(f"⚠️ Target column '{target_col}' found in prediction data. It will be ignored.")
                        pred_df = pred_df.drop(columns=[target_col])
                    
                    # Make predictions button
                    if st.button("🚀 Generate Predictions", type="primary", key="make_predictions"):
                        with st.spinner("🔄 Making predictions..."):
                            try:
                                # Make predictions
                                predictions = st.session_state.ml_models.predict_new_data(pred_df)
                                
                                if predictions is not None:
                                    # Add predictions to the dataframe
                                    pred_df_with_predictions = pred_df.copy()
                                    pred_df_with_predictions['Predicted_' + target_col] = predictions
                                    
                                    st.success(f"✅ Predictions generated successfully!")
                                    
                                    # Show prediction statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Predictions Made", len(predictions))
                                    with col2:
                                        st.metric("Mean Prediction", f"{predictions.mean():.4f}")
                                    with col3:
                                        st.metric("Min Prediction", f"{predictions.min():.4f}")
                                    with col4:
                                        st.metric("Max Prediction", f"{predictions.max():.4f}")
                                    
                                    # Show results with predictions
                                    st.write("### 📊 Results with Predictions")
                                    st.dataframe(pred_df_with_predictions, use_container_width=True)
                                    
                                    # Download button for results
                                    csv = pred_df_with_predictions.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Predictions CSV",
                                        data=csv,
                                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                    
                                    # Prediction distribution plot
                                    fig_pred = px.histogram(
                                        x=predictions,
                                        title="Distribution of Predictions",
                                        labels={'x': 'Predicted Values', 'y': 'Frequency'},
                                        marginal="box"
                                    )
                                    st.plotly_chart(fig_pred, use_container_width=True)
                                    
                                else:
                                    st.error("❌ Failed to generate predictions. Please check your data.")
                                    
                            except Exception as e:
                                st.error(f"❌ Error making predictions: {str(e)}")
                                st.exception(e)
                
                except Exception as e:
                    st.error(f"❌ Error loading prediction data: {str(e)}")
            
            else:
                st.info("👆 Please upload a dataset to make predictions")
# Initialize session state
initialize_session_state()

# Main header
st.markdown("""
<div class="main-header">
    <h1>Data Preprocessing and Advanced Analysis Platform</h1>
    <p style="margin: 0; font-size: 1.1em;">Professional Data Processing & Analytics Platform</p>
    <p style="margin: 0; font-size: 0.9em; opacity: 0.8;">Transform • Analyze • Visualize • Predict</p>
</div>
""", unsafe_allow_html=True)

# Professional sidebar navigation with buttons
st.sidebar.markdown("### 🧭 Navigation Center")
st.sidebar.markdown("---")

# Navigation buttons
if st.sidebar.button("🔧 Data Processing Pipeline", use_container_width=True):
    st.session_state.current_page = "data_pipeline"

if st.sidebar.button("📊 Data Visualization Studio", use_container_width=True):
    st.session_state.current_page = "visualization"

if st.sidebar.button("🔬 Advanced Analytics", use_container_width=True):
    st.session_state.current_page = "analytics"

if st.sidebar.button("🤖 Predictive Modeling", use_container_width=True):
    st.session_state.current_page = "prediction"

# Get current page
page = st.session_state.current_page

# Sidebar data summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Available Datasets")

datasets_available = []
if st.session_state.df_cleaned is not None:
    datasets_available.append(f"✅ Cleaned: {st.session_state.df_cleaned.shape[0]:,} × {st.session_state.df_cleaned.shape[1]}")
if st.session_state.df_grouped is not None:
    datasets_available.append(f"📊 Grouped: {st.session_state.df_grouped.shape[0]:,} × {st.session_state.df_grouped.shape[1]}")
if st.session_state.df_merged is not None:
    datasets_available.append(f"🔗 Merged: {st.session_state.df_merged.shape[0]:,} × {st.session_state.df_merged.shape[1]}")
if st.session_state.df_encoded is not None:
    datasets_available.append(f"🔤 Encoded: {st.session_state.df_encoded.shape[0]:,} × {st.session_state.df_encoded.shape[1]}")

if datasets_available:
    for dataset in datasets_available:
        st.sidebar.markdown(f"• {dataset}")
else:
    st.sidebar.markdown("• No processed datasets yet")

st.sidebar.markdown("---")
if st.sidebar.button("📋 Data Overview", use_container_width=True):
    st.session_state.current_page = "data_overview"
st.sidebar.markdown("---")

# Quick actions in sidebar
if st.session_state.df_cleaned is not None or st.session_state.df_grouped is not None or st.session_state.df_merged is not None or st.session_state.df_encoded is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Quick Actions")
    
    if st.sidebar.button("🗑️ Reset All Data", use_container_width=True):
        for key in ['df_original', 'df_cleaned', 'df_second', 'df_grouped', 'df_merged', 'df_encoded']:
            st.session_state[key] = None
        for key in ['cleaning_applied', 'grouping_applied', 'merging_applied', 'encoding_applied']:
            st.session_state[key] = False
        st.session_state.plots_history = []
        st.sidebar.success("✅ All data reset!")
        st.rerun()

# PAGE 1: DATA PROCESSING PIPELINE
if page == "data_pipeline":
    st.header("🔧 Data Processing Pipeline")
    st.markdown("Complete data preparation workflow in one integrated interface")
    st.markdown("---")
    
    # Processing Logs in sidebar (condensed view)
    with st.sidebar.expander("Recent Logs", expanded=False):
        if st.session_state.processing_logs:
            for log in st.session_state.processing_logs[-5:]:  # Show last 5 logs
                st.text(log)
        else:
            st.text("No logs yet")

    # ===========================================
    # SECTION 1: DATA LOADING
    # ===========================================
    st.header("📁 Data Loading")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx'],
        help="Upload your dataset to get started"
    )
    
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            with st.spinner("Loading data..."):
                add_log(f"Loading file: {uploaded_file.name}")
                # Use your DataLoader class
                df = st.session_state.loader.loader(tmp_file_path)
                st.session_state.df_original = df
                
                add_log(f"Data loaded successfully - Shape: {df.shape}")
                st.success("✅ Data loaded successfully!")
                
                # Display basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show column info
                with st.expander("Column Information", expanded=False):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Null Count': df.isnull().sum(),
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            add_log(error_msg, "ERROR")
            st.error(error_msg)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    st.markdown("---")

    # ===========================================
    # SECTION 2: DATA CLEANING
    # ===========================================
    st.header("🧹 Data Cleaning")
    
    if st.session_state.df_original is None:
        st.warning("⚠️ Please load a dataset first from the Data Loading section above.")
    else:
        df = st.session_state.df_original.copy()
        
        # Data quality assessment
        st.subheader("Data Quality Assessment")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("🔍 Assess Data Quality", type="primary"):
                with st.spinner("Assessing data quality..."):
                    add_log("Starting data quality assessment")
                    try:
                        quality_report = st.session_state.cleaner.assess_data_quality(df)
                        st.session_state.quality_report = quality_report  # Store for display
                        
                    except Exception as e:
                        error_msg = f"Error during quality assessment: {str(e)}"
                        add_log(error_msg, "ERROR")
                        st.error(error_msg)
        
        # Display quality report if available
        if hasattr(st.session_state, 'quality_report'):
            quality_report = st.session_state.quality_report
            
            with col2:
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.write("**Dataset Shape:**", quality_report['shape'])
                    st.write("**Memory Usage:**", f"{quality_report['memory_usage'] / (1024**2):.2f} MB")
                    st.write("**Duplicates:**", quality_report['duplicates'])
                
                with subcol2:
                    st.write("**Numeric Columns:**", len(quality_report['numeric_columns']))
                    st.write("**Categorical Columns:**", len(quality_report['categorical_columns']))
            
            # Missing values breakdown
            if quality_report['missing_values'].sum() > 0:
                with st.expander("Missing Values Details", expanded=True):
                    missing_df = pd.DataFrame({
                        'Column': quality_report['missing_values'].index,
                        'Missing Count': quality_report['missing_values'].values,
                        'Missing %': (quality_report['missing_values'].values / df.shape[0] * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['Missing Count'] > 0]
                    st.dataframe(missing_df, use_container_width=True)
                    add_log(f"Found missing values in {len(missing_df)} columns")
            else:
                st.success("✅ No missing values found!")
                add_log("No missing values found")
        
        # Data cleaning options
        st.subheader("Clean Data")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            data_type = st.selectbox(
                "Select cleaning strategy:",
                ["statique", "frequence"],
                help="Choose the strategy for handling missing values"
            )
            
            # Show frequency input only when frequence is selected
            frequency_hours = 1  # default value
            if data_type == "frequence":
                frequency_hours = st.number_input(
                    "Interpolation frequency (hours):",
                    min_value=0.1,
                    max_value=24.0,
                    value=1.0,
                    step=0.1,
                    help="Time interval in hours for data interpolation (e.g., 1.0 for hourly, 0.5 for 30 minutes)"
                )
                st.info(f"Selected frequency: {frequency_hours} hour(s)")
            
            if st.button("🧹 Clean Data", type="primary"):
                with st.spinner("Cleaning data..."):
                    add_log(f"Starting data cleaning with strategy: {data_type}" + 
                           (f" (frequency: {frequency_hours}h)" if data_type == "frequence" else ""))
                    
                    try:
                        initial_shape = df.shape
                        # Pass the frequency_hours parameter to clean_data
                        cleaned_df = st.session_state.cleaner.clean_data(df, data_type, frequency_hours)
                        
                        # Store cleaned data and drop original
                        st.session_state.df_cleaned = cleaned_df
                        st.session_state.df_original = None  # Drop original as requested
                        st.session_state.cleaning_applied = True
                        st.session_state.cleaning_results = {
                            'initial_shape': initial_shape,
                            'final_shape': cleaned_df.shape,
                            'strategy': data_type,
                            'frequency': frequency_hours if data_type == "frequence" else None
                        }
                        
                        success_msg = f"✅ Data cleaned successfully using {data_type} strategy! Original data dropped."
                        if data_type == "frequence":
                            success_msg += f" (Interpolation frequency: {frequency_hours}h)"
                        
                        add_log(f"Data cleaning completed: {initial_shape} → {cleaned_df.shape}")
                        st.success(success_msg)
                        
                    except Exception as e:
                        error_msg = f"Error during cleaning: {str(e)}"
                        add_log(error_msg, "ERROR")
                        st.error(error_msg)
        
        # Show cleaning results if available
        if hasattr(st.session_state, 'cleaning_results'):
            results = st.session_state.cleaning_results
            
            with col2:
                st.subheader("Cleaning Results")
                st.write(f"**Strategy:** {results['strategy'].title()}")
                if results['frequency']:
                    st.write(f"**Interpolation Frequency:** {results['frequency']} hour(s)")
                
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("Original Rows", results['initial_shape'][0])
                    st.metric("Original Columns", results['initial_shape'][1])
                with subcol2:
                    st.metric("Cleaned Rows", results['final_shape'][0], delta=results['final_shape'][0] - results['initial_shape'][0])
                    st.metric("Cleaned Columns", results['final_shape'][1], delta=results['final_shape'][1] - results['initial_shape'][1])
                
                # Show cleaned data preview
                if st.session_state.df_cleaned is not None:
                    with st.expander("Cleaned Data Preview", expanded=False):
                        st.dataframe(st.session_state.df_cleaned.head(10), use_container_width=True)

    st.markdown("---")

    # ===========================================
    # SECTION 3: DATA TRANSFORMATION
    # ===========================================
    st.header("🔄 Data Transformation")
    
    if st.session_state.df_cleaned is None:
        st.warning("⚠️ Please clean your data first from the Data Cleaning section above.")
    else:
        current_df = st.session_state.df_cleaned.copy()
        st.info(f"Working with cleaned dataset: {current_df.shape[0]} rows × {current_df.shape[1]} columns")
        
        # Create transformation sections vertically
                

        # ===== CATEGORICAL ENCODING =====
        st.subheader("🔤 Categorical Data Encoding")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Show target selection for encoding
            target_col = st.selectbox(
                "Select target column (optional, for target encoding):",
                options=['None'] + current_df.columns.tolist(),
                help="Select target column for advanced encoding methods"
            )
            
            target_column = None if target_col == 'None' else target_col
            
            if st.button("🔤 Encode Categorical Data", type="primary", key="categorical_encode"):
                with st.spinner("Encoding categorical data..."):
                    try:
                        add_log("Starting categorical encoding on cleaned data")
                        
                        # Use the smart encoding function from transformer
                        pred_df_raw = st.session_state.transformer.smart_categorical_encoding(
                            current_df.copy(), target_col=target_column
                        )
                        
                        
                        # Store as independent encoded dataframe
                        st.session_state.df_encoded = pred_df_raw
                        st.session_state.encoding_applied = True
                        st.session_state.encoding_results = {
                            'original_shape': current_df.shape,
                            'encoded_shape': pred_df_raw.shape
                        }
                        
                        st.success(f"✅ Categorical encoding completed!")
                        add_log(f"Encoding completed: {current_df.shape} → {pred_df_raw.shape}")
                        
                        # Show conversion summary
                        numeric_cols = len(pred_df_raw.select_dtypes(include=[np.number]).columns)
                        total_cols = len(pred_df_raw.columns)
                        st.info(f"📊 Result: {numeric_cols}/{total_cols} columns are now numeric")
                        
                    except Exception as e:
                        error_msg = f"Error encoding data: {str(e)}"
                        add_log(error_msg, "ERROR")
                        st.error(error_msg)
        
        # Show encoding results
        if hasattr(st.session_state, 'encoding_results'):
            results = st.session_state.encoding_results
            with col2:
                st.subheader("Encoding Results")
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("Original columns", results['original_shape'][1])
                    st.metric("Original rows", results['original_shape'][0])
                with subcol2:
                    st.metric("Encoded columns", results['encoded_shape'][1], delta=results['encoded_shape'][1] - results['original_shape'][1])
                    st.metric("Encoded rows", results['encoded_shape'][0], delta=results['encoded_shape'][0] - results['original_shape'][0])
                
                if st.session_state.df_encoded is not None:
                    with st.expander("Encoded Data Preview", expanded=False):
                        st.dataframe(st.session_state.df_encoded.head(10), use_container_width=True)

        st.markdown("**---**")

        # ===== DATA GROUPING =====
        st.subheader("📊 Data Grouping")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            available_cols = current_df.columns.tolist()
            group_cols = st.multiselect(
                "Select columns to group by:",
                available_cols,
                help="Choose one or more columns for grouping",
                key="group_cols"
            )
            
            if group_cols:
                # Preview of grouping
                st.write(f"**Selected:** {group_cols}")
                
                if st.button("📊 Group Data", type="primary", key="group_data"):
                    with st.spinner("Grouping data..."):
                        try:
                            add_log(f"Starting data grouping by: {group_cols}")
                            grouped_df = st.session_state.transformer.dataframe_grouping(current_df.copy(), group_cols)
                            
                            # Store as independent grouped dataframe
                            st.session_state.df_grouped = grouped_df
                            st.session_state.grouping_applied = True
                            st.session_state.grouping_results = {
                                'original_shape': current_df.shape,
                                'grouped_shape': grouped_df.shape,
                                'group_cols': group_cols
                            }
                            
                            st.success("✅ Data grouped successfully!")
                            add_log(f"Grouping completed: {current_df.shape} → {grouped_df.shape}")
                            
                        except Exception as e:
                            error_msg = f"Error grouping data: {str(e)}"
                            add_log(error_msg, "ERROR")
                            st.error(error_msg)
            else:
                st.info("Select columns above to enable grouping")
        
        # Show grouping results
        if hasattr(st.session_state, 'grouping_results'):
            results = st.session_state.grouping_results
            with col2:
                st.subheader("Grouping Results")
                st.write(f"**Grouped by:** {results['group_cols']}")
                
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("Original rows", results['original_shape'][0])
                with subcol2:
                    st.metric("Grouped rows", results['grouped_shape'][0])
                
                if st.session_state.df_grouped is not None:
                    # Show sample group sizes
                    sample_groups = current_df.groupby(results['group_cols']).size().head(5)
                    st.write("**Sample group sizes:**")
                    st.dataframe(sample_groups.to_frame('count'))
                    
                    with st.expander("Grouped Data Preview", expanded=False):
                        st.dataframe(st.session_state.df_grouped.head(10), use_container_width=True)

        st.markdown("**---**")

        # ===== MERGE/CONCATENATE =====
        st.subheader("🔗 Merge or Concatenate DataFrames")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # File uploader for second dataset
            second_file = st.file_uploader(
                "Upload second dataset:",
                type=['csv', 'xlsx'],
                key="second_dataset_pipeline"
            )
            
            if second_file is not None:
                # Load second dataset
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{second_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(second_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    df2 = st.session_state.loader.loader(tmp_file_path)
                    st.session_state.df_second = df2
                    
                    st.success(f"✅ Second dataset loaded: {df2.shape}")
                    
                    # Operation selection
                    operation = st.radio("Select operation:", ["Merge", "Concatenate"], key="merge_operation")
                    
                    if operation == "Merge":
                        # Find common columns
                        common_cols = list(set(current_df.columns) & set(df2.columns))
                        
                        if common_cols:
                            merge_cols = st.multiselect("Merge on columns:", common_cols, key="merge_cols")
                            merge_method = st.selectbox("Merge method:", ["inner", "outer", "left", "right"], key="merge_method")
                            
                            if merge_cols and st.button("🔗 Merge DataFrames", key="merge_button"):
                                with st.spinner("Merging dataframes..."):
                                    try:
                                        add_log(f"Starting merge on columns: {merge_cols}")
                                        merged_df = st.session_state.transformer.dataframe_merging(
                                            current_df, df2, on=merge_cols, how=merge_method
                                        )
                                        
                                        # Store as independent merged dataframe
                                        st.session_state.df_merged = merged_df
                                        st.session_state.merging_applied = True
                                        st.session_state.merge_results = {
                                            'df1_shape': current_df.shape,
                                            'df2_shape': df2.shape,
                                            'merged_shape': merged_df.shape,
                                            'operation': f"Merge ({merge_method})"
                                        }
                                        
                                        st.success("✅ DataFrames merged successfully!")
                                        add_log(f"Merge completed: {current_df.shape} + {df2.shape} → {merged_df.shape}")
                                        
                                    except Exception as e:
                                        error_msg = f"Error merging dataframes: {str(e)}"
                                        add_log(error_msg, "ERROR")
                                        st.error(error_msg)
                        else:
                            st.warning("No common columns found for merging")
                    
                    else:  # Concatenate
                        axis = st.selectbox(
                            "Concatenation axis:",
                            [0, 1],
                            format_func=lambda x: "Rows (0)" if x == 0 else "Columns (1)",
                            key="concat_axis"
                        )
                        
                        if st.button("📎 Concatenate DataFrames", key="concat_button"):
                            with st.spinner("Concatenating dataframes..."):
                                try:
                                    add_log(f"Starting concatenation along axis {axis}")
                                    concat_df = st.session_state.transformer.dataframe_concat(
                                        current_df, df2, axis=axis
                                    )
                                    
                                    # Store as independent merged dataframe (concatenation result)
                                    st.session_state.df_merged = concat_df
                                    st.session_state.merging_applied = True
                                    st.session_state.merge_results = {
                                        'df1_shape': current_df.shape,
                                        'df2_shape': df2.shape,
                                        'merged_shape': concat_df.shape,
                                        'operation': f"Concatenate (axis={axis})"
                                    }
                                    
                                    st.success("✅ DataFrames concatenated successfully!")
                                    add_log(f"Concatenation completed: {current_df.shape} + {df2.shape} → {concat_df.shape}")
                                    
                                except Exception as e:
                                    error_msg = f"Error concatenating dataframes: {str(e)}"
                                    add_log(error_msg, "ERROR")
                                    st.error(error_msg)
                    
                except Exception as e:
                    error_msg = f"Error loading second dataset: {str(e)}"
                    add_log(error_msg, "ERROR")
                    st.error(error_msg)
                finally:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
        
        # Show merge/concatenate results
        if hasattr(st.session_state, 'merge_results'):
            results = st.session_state.merge_results
            with col2:
                st.subheader("Operation Results")
                st.write(f"**Operation:** {results['operation']}")
                
                subcol1, subcol2, subcol3 = st.columns(3)
                with subcol1:
                    st.metric("Dataset 1", f"{results['df1_shape'][0]}×{results['df1_shape'][1]}")
                with subcol2:
                    st.metric("Dataset 2", f"{results['df2_shape'][0]}×{results['df2_shape'][1]}")
                with subcol3:
                    st.metric("Result", f"{results['merged_shape'][0]}×{results['merged_shape'][1]}")
                
                if st.session_state.df_merged is not None:
                    with col2:
                        with st.expander("Second Dataset Preview", expanded=False):
                            st.dataframe(st.session_state.df_second.head(3), use_container_width=True)
                        
                        with st.expander("Merged/Concatenated Data Preview", expanded=False):
                            st.dataframe(st.session_state.df_merged.head(10), use_container_width=True)


# ===========================================
# DATA OVERVIEW PAGE (SEPARATE)
# ===========================================
elif page == "data_overview":
    st.header("📋 Data Overview")
    st.markdown("View and download all processed datasets")
    st.markdown("---")
    
    # Show all available processed datasets
    datasets_to_show = []
    
    if st.session_state.df_cleaned is not None:
        datasets_to_show.append(("Cleaned Data", st.session_state.df_cleaned))
    
    if st.session_state.df_grouped is not None:
        datasets_to_show.append(("Grouped Data", st.session_state.df_grouped))
    
    if st.session_state.df_merged is not None:
        datasets_to_show.append(("Merged Data", st.session_state.df_merged))
    
    if st.session_state.df_encoded is not None:
        datasets_to_show.append(("Encoded Data", st.session_state.df_encoded))
    
    if not datasets_to_show:
        st.warning("⚠️ No processed datasets available. Please process your data first in the Data Pipeline.")
        st.info("👈 Navigate to 'Data Pipeline' to load and process your data.")
    else:
        st.success(f"📊 {len(datasets_to_show)} processed dataset(s) available")
        
        for i, (dataset_name, dataset_df) in enumerate(datasets_to_show):
            st.subheader(f"📊 {dataset_name}")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{dataset_df.shape[0]:,}")
            with col2:
                st.metric("Columns", f"{dataset_df.shape[1]:,}")
            with col3:
                st.metric("Missing Values", f"{dataset_df.isnull().sum().sum():,}")
            with col4:
                st.metric("Memory (MB)", f"{dataset_df.memory_usage(deep=True).sum() / 1024**2:.1f}")
            
            # Data preview and info in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Data Preview:**")
                st.dataframe(dataset_df.head(10), use_container_width=True)
            
            with col2:
                st.write("**Column Types:**")
                dtype_counts = dataset_df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"• {dtype}: {count} columns")
                
                # Download button for each dataset
                csv = dataset_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {dataset_name}",
                    data=csv,
                    file_name=f"{dataset_name.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"download_{dataset_name.lower().replace(' ', '_')}",
                    use_container_width=True
                )
            
            if i < len(datasets_to_show) - 1:  # Don't add separator after last item
                st.markdown("---")


# ===========================================
# PROCESSING LOGS PAGE (SEPARATE)
# ===========================================
elif page == "processing_logs":
    st.header("📝 Processing Logs")
    st.markdown("View all processing activities and system messages")
    st.markdown("---")
    
    if st.session_state.processing_logs:
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ Clear All Logs", type="secondary"):
                st.session_state.processing_logs = []
                st.success("All logs cleared!")
                st.rerun()
        
        with col2:
            # Export logs
            logs_text = "\n".join(st.session_state.processing_logs)
            st.download_button(
                label="📥 Export Logs",
                data=logs_text,
                file_name="processing_logs.txt",
                mime="text/plain"
            )
        
        with col3:
            st.info(f"Total logs: {len(st.session_state.processing_logs)}")
        
        st.markdown("---")
        
        # Filter options
        filter_type = st.selectbox(
            "Filter logs by type:",
            ["All", "Errors", "Warnings", "Info"],
            key="log_filter"
        )
        
        # Display logs based on filter
        filtered_logs = []
        if filter_type == "All":
            filtered_logs = st.session_state.processing_logs
        elif filter_type == "Errors":
            filtered_logs = [log for log in st.session_state.processing_logs if "ERROR" in log]
        elif filter_type == "Warnings":
            filtered_logs = [log for log in st.session_state.processing_logs if "WARNING" in log]
        else:  # Info
            filtered_logs = [log for log in st.session_state.processing_logs if "ERROR" not in log and "WARNING" not in log]
        
        if filtered_logs:
            st.subheader(f"{filter_type} Logs ({len(filtered_logs)})")
            
            # Display logs in reverse order (newest first)
            for i, log in enumerate(reversed(filtered_logs)):
                with st.container():
                    if "ERROR" in log:
                        st.error(f"[{len(filtered_logs)-i}] {log}")
                    elif "WARNING" in log:
                        st.warning(f"[{len(filtered_logs)-i}] {log}")
                    else:
                        st.info(f"[{len(filtered_logs)-i}] {log}")
        else:
            st.info(f"No {filter_type.lower()} logs found.")
    
    else:
        st.info("📋 No processing logs available yet.")
        st.markdown("Logs will appear here as you process data in the Data Pipeline.")
        
        # Quick navigation
        if st.button("🚀 Go to Data Pipeline"):
            st.switch_page("data_pipeline")  # Assuming you have page switching functionality

# Data Visualization Section
elif page == "visualization":
    st.header("📊 Interactive Data Visualization")
    
    # Get available processed datasets for visualization
    available_datasets = []
    dataset_options = {}
    
    if st.session_state.df_cleaned is not None:
        available_datasets.append("Cleaned Data")
        dataset_options["Cleaned Data"] = st.session_state.df_cleaned
    
    if st.session_state.df_grouped is not None:
        available_datasets.append("Grouped Data")
        dataset_options["Grouped Data"] = st.session_state.df_grouped
    
    if st.session_state.df_merged is not None:
        available_datasets.append("Merged Data")
        dataset_options["Merged Data"] = st.session_state.df_merged
    
    if st.session_state.df_encoded is not None:
        available_datasets.append("Encoded Data")
        dataset_options["Encoded Data"] = st.session_state.df_encoded
    
    if not available_datasets:
        st.warning("⚠️ No processed datasets available for visualization. Please process your data first.")
    else:
        # Dataset selection for visualization
        selected_dataset = st.selectbox(
            "📊 Select dataset for visualization:",
            available_datasets,
            help="Choose which processed dataset to visualize"
        )
        
        current_df = dataset_options[selected_dataset]
        st.info(f"Using {selected_dataset}: {current_df.shape[0]:,} rows × {current_df.shape[1]} columns")
        
        # Visualization tabs
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🎨 Single Plot", "🔄 Comparison View", "📊 Plot Gallery"])
        
        with viz_tab1:
            st.subheader("🎨 Create Visualization")
            
            # Create main visualization dashboard
            try:
                fig = st.session_state.dashboard.create_comprehensive_dashboard(current_df, "main_viz")
                
                if fig is not None:
                    # Save plot to history
                    plot_info = {
                        'figure': fig,
                        'plot_type': st.session_state.get('main_viz_plot_type', 'Unknown'),
                        'title': fig.layout.title.text if fig.layout.title.text else "Untitled Plot",
                        'x_col': st.session_state.get('main_viz_x_col'),
                        'y_col': st.session_state.get('main_viz_y_col'),
                        'data_shape': current_df.shape,
                        'dataset_type': selected_dataset
                    }
                    
                    # Only add to history if it's a new plot (not just updating)
                    if st.button("💾 Save Plot to Gallery"):
                        save_plot_to_history(plot_info)
                        st.success("✅ Plot saved to gallery!")
                    
                    # Render insights
                    x_col = st.session_state.get('main_viz_x_col')
                    y_col = st.session_state.get('main_viz_y_col')
                    if x_col:
                        st.session_state.dashboard.render_plot_insights(current_df, x_col, y_col)
                    
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
        
        with viz_tab2:
            st.subheader("🔄 Side-by-Side Comparison")
            st.write("Compare different visualizations of the same data")
            
            try:
                fig1, fig2 = st.session_state.dashboard.create_comparison_dashboard(current_df)
                
                if fig1 is not None and fig2 is not None:
                    if st.button("💾 Save Comparison to Gallery"):
                        # Save both plots
                        for i, fig in enumerate([fig1, fig2], 1):
                            plot_info = {
                                'figure': fig,
                                'plot_type': f"Comparison {i}",
                                'title': f"Comparison Plot {i} ({selected_dataset})",
                                'data_shape': current_df.shape,
                                'dataset_type': selected_dataset
                            }
                            save_plot_to_history(plot_info)
                        
                        st.success("✅ Comparison plots saved to gallery!")
                        
            except Exception as e:
                st.error(f"Error creating comparison: {str(e)}")
        
        with viz_tab3:
            st.subheader("🖼️ Plot Gallery & History")
            
            if st.session_state.plots_history:
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    if st.button("🗑️ Clear Gallery", type="secondary"):
                        st.session_state.plots_history = []
                        st.success("Gallery cleared!")
                        st.rerun()
                    
                    st.write(f"**Total plots: {len(st.session_state.plots_history)}**")
                
                with col1:
                    # Display plots
                    st.session_state.dashboard.render_plot_gallery(st.session_state.plots_history)
                    
            else:
                st.info("📷 No plots in gallery yet. Create some visualizations to see them here!")

# Advanced Analytics Section
elif page == "analytics":
    st.header("🔬 Advanced Analytics")
    st.markdown("Statistical analysis, hypothesis testing, and data quality tools")
    st.markdown("---")
    
    # Initialize analyzers
    if 'stat_analyzer' not in st.session_state:
        st.session_state.stat_analyzer = StatisticalAnalyzer()
    if 'adv_analytics' not in st.session_state:
        st.session_state.adv_analytics = AdvancedAnalytics()
    
    # Get available processed datasets for analytics
    available_datasets = []
    dataset_options = {}
    
    if st.session_state.df_cleaned is not None:
        available_datasets.append("Cleaned Data")
        dataset_options["Cleaned Data"] = st.session_state.df_cleaned
    
    if st.session_state.df_grouped is not None:
        available_datasets.append("Grouped Data")
        dataset_options["Grouped Data"] = st.session_state.df_grouped
    
    if st.session_state.df_merged is not None:
        available_datasets.append("Merged Data")
        dataset_options["Merged Data"] = st.session_state.df_merged
    
    if st.session_state.df_encoded is not None:
        available_datasets.append("Encoded Data")
        dataset_options["Encoded Data"] = st.session_state.df_encoded
    
    if not available_datasets:
        st.warning("⚠️ No processed datasets available for analytics. Please process your data first.")
    else:
        # Dataset selection for analytics
        selected_dataset = st.selectbox(
            "🔬 Select dataset for analysis:",
            available_datasets,
            help="Choose which processed dataset to analyze"
        )
        
        current_df = dataset_options[selected_dataset]
        st.info(f"Analyzing {selected_dataset}: {current_df.shape[0]:,} rows × {current_df.shape[1]} columns")
        
        # Expanded Analytics tabs
        analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4, analytics_tab5, analytics_tab6 = st.tabs([
            "📊 Descriptive Stats", 
            "🔗 Correlation Analysis", 
            "📈 Distribution Analysis",
            "🧪 Hypothesis Testing",
            "🎯 Outlier Detection",
            "📋 Data Quality Report"
        ])
        
        # =============================================
        # TAB 1: DESCRIPTIVE STATISTICS
        # =============================================
        with analytics_tab1:
            st.subheader("📊 Descriptive Statistics")
            
            # Basic statistics
            st.write("**Numerical Columns Statistics:**")
            numeric_df = current_df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)
                
                # Extended stats
                with st.expander("📈 Extended Statistics (Skewness, Kurtosis, etc.)"):
                    ext_stats = st.session_state.adv_analytics.get_summary_statistics_extended(current_df)
                    if 'extended_statistics' in ext_stats:
                        st.dataframe(ext_stats['extended_statistics'], use_container_width=True)
            else:
                st.info("No numerical columns found")
            
            # Categorical statistics
            st.write("**Categorical Columns Statistics:**")
            categorical_df = current_df.select_dtypes(include=['object', 'category'])
            if not categorical_df.empty:
                cat_stats = []
                for col in categorical_df.columns:
                    cat_stats.append({
                        'Column': col,
                        'Unique Values': current_df[col].nunique(),
                        'Most Frequent': current_df[col].mode().iloc[0] if not current_df[col].mode().empty else 'N/A',
                        'Missing Values': current_df[col].isnull().sum()
                    })
                st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)
            else:
                st.info("No categorical columns found")
        
        # =============================================
        # TAB 2: CORRELATION ANALYSIS
        # =============================================
        with analytics_tab2:
            st.subheader("🔗 Correlation Analysis")
            
            numeric_df = current_df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr()
                
                # Correlation heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu_r"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # High correlation pairs
                st.subheader("High Correlation Pairs (|r| > 0.7)")
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': round(corr_val, 3)
                            })
                
                if high_corr_pairs:
                    st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
                else:
                    st.info("No high correlation pairs found")
                
                # Multicollinearity detection
                with st.expander("🔍 Multicollinearity Analysis"):
                    threshold = st.slider("Correlation threshold", 0.7, 0.99, 0.85, 0.05)
                    multi_result = st.session_state.adv_analytics.detect_multicollinearity(current_df, threshold)
                    if 'error' not in multi_result:
                        st.write(f"**High correlation pairs found:** {multi_result['n_high_correlations']}")
                        st.write(f"**Recommendation:** {multi_result['recommendation']}")
                        if multi_result['potentially_redundant_features']:
                            st.warning(f"Potentially redundant features: {', '.join(multi_result['potentially_redundant_features'])}")
            else:
                st.info("No numerical columns available for correlation analysis")
        
        # =============================================
        # TAB 3: DISTRIBUTION ANALYSIS
        # =============================================
        with analytics_tab3:
            st.subheader("📈 Distribution Analysis")
            
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols, key="dist_col")
                
                if selected_col:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig_hist = px.histogram(
                            current_df, 
                            x=selected_col, 
                            title=f"Distribution of {selected_col}",
                            marginal="box"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Box plot
                        fig_box = px.box(
                            current_df, 
                            y=selected_col, 
                            title=f"Box Plot of {selected_col}"
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Normality testing
                    st.subheader("🔬 Normality Tests")
                    normality_result = st.session_state.stat_analyzer.test_normality(current_df[selected_col])
                    
                    if 'error' not in normality_result:
                        col1, col2, col3 = st.columns(3)
                        
                        for i, (test_name, test_result) in enumerate(normality_result.get('tests', {}).items()):
                            if 'error' not in test_result:
                                with [col1, col2, col3][i % 3]:
                                    is_normal = test_result.get('is_normal', False)
                                    st.metric(
                                        test_name.replace('_', ' ').title(),
                                        f"p = {test_result['p_value']:.4f}",
                                        delta="Normal" if is_normal else "Not Normal",
                                        delta_color="normal" if is_normal else "inverse"
                                    )
                        
                        overall = normality_result.get('overall_conclusion', {})
                        if overall.get('is_normal') is not None:
                            if overall['is_normal']:
                                st.success(f"✅ Data appears normally distributed ({overall['normal_tests']}/{overall['total_tests']} tests passed)")
                            else:
                                st.warning(f"⚠️ Data does not appear normally distributed ({overall['normal_tests']}/{overall['total_tests']} tests passed)")
                    
                    # Skewness and Kurtosis
                    with st.expander("📊 Skewness & Kurtosis Analysis"):
                        sk_result = st.session_state.stat_analyzer.get_skewness_kurtosis(current_df[selected_col])
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Skewness", f"{sk_result['skewness']:.4f}")
                            st.caption(sk_result['skewness_interpretation'])
                        with col2:
                            st.metric("Kurtosis", f"{sk_result['kurtosis']:.4f}")
                            st.caption(sk_result['kurtosis_interpretation'])
            else:
                st.info("No numerical columns available for distribution analysis")
        
        # =============================================
        # TAB 4: HYPOTHESIS TESTING
        # =============================================
        with analytics_tab4:
            st.subheader("🧪 Hypothesis Testing")
            st.markdown("Perform statistical hypothesis tests on your data")
            
            test_type = st.selectbox(
                "Select Test Type:",
                ["One-Sample T-Test", "Two-Sample T-Test", "Paired T-Test", "ANOVA", "Chi-Square Test", "Mann-Whitney U", "Kruskal-Wallis"]
            )
            
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = current_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            st.markdown("---")
            
            if test_type == "One-Sample T-Test":
                st.write("**One-Sample T-Test**: Compare sample mean to a hypothesized population mean")
                
                if numeric_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        test_col = st.selectbox("Select column:", numeric_cols, key="onesamp_col")
                    with col2:
                        pop_mean = st.number_input("Hypothesized population mean:", value=0.0)
                    
                    alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05, 0.01, key="onesamp_alpha")
                    
                    if st.button("Run One-Sample T-Test", type="primary"):
                        result = st.session_state.stat_analyzer.one_sample_ttest(
                            current_df[test_col], pop_mean, alpha
                        )
                        
                        if 'error' not in result:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Sample Mean", f"{result['sample_mean']:.4f}")
                            with col2:
                                st.metric("t-Statistic", f"{result['t_statistic']:.4f}")
                            with col3:
                                st.metric("p-Value", f"{result['p_value']:.6f}")
                            with col4:
                                st.metric("Cohen's d", f"{result['cohens_d']:.4f}")
                            
                            if result['reject_null']:
                                st.success(f"✅ **Reject H₀**: {result['interpretation']}")
                            else:
                                st.info(f"ℹ️ **Fail to Reject H₀**: {result['interpretation']}")
                            
                            st.caption(f"Effect size: {result['effect_size_interpretation']}")
                        else:
                            st.error(result['error'])
                else:
                    st.warning("No numeric columns available")
            
            elif test_type == "Two-Sample T-Test":
                st.write("**Two-Sample T-Test**: Compare means between two independent groups")
                
                if numeric_cols and categorical_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        value_col = st.selectbox("Select numeric column:", numeric_cols, key="twosamp_val")
                    with col2:
                        group_col = st.selectbox("Select grouping column:", categorical_cols, key="twosamp_grp")
                    
                    groups = current_df[group_col].dropna().unique()
                    if len(groups) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            group1 = st.selectbox("Group 1:", groups, key="twosamp_g1")
                        with col2:
                            group2 = st.selectbox("Group 2:", [g for g in groups if g != group1], key="twosamp_g2")
                        
                        equal_var = st.checkbox("Assume equal variances (unchecked = Welch's t-test)", value=False)
                        alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05, 0.01, key="twosamp_alpha")
                        
                        if st.button("Run Two-Sample T-Test", type="primary"):
                            g1_data = current_df[current_df[group_col] == group1][value_col]
                            g2_data = current_df[current_df[group_col] == group2][value_col]
                            
                            result = st.session_state.stat_analyzer.two_sample_ttest(
                                g1_data, g2_data, alpha, equal_var
                            )
                            
                            if 'error' not in result:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(f"Mean ({group1})", f"{result['group1_mean']:.4f}")
                                with col2:
                                    st.metric(f"Mean ({group2})", f"{result['group2_mean']:.4f}")
                                with col3:
                                    st.metric("t-Statistic", f"{result['t_statistic']:.4f}")
                                with col4:
                                    st.metric("p-Value", f"{result['p_value']:.6f}")
                                
                                if result['reject_null']:
                                    st.success(f"✅ **Reject H₀**: {result['interpretation']}")
                                else:
                                    st.info(f"ℹ️ **Fail to Reject H₀**: {result['interpretation']}")
                                
                                st.caption(f"Effect size (Cohen's d = {result['cohens_d']:.4f}): {result['effect_size_interpretation']}")
                                
                                # Levene's test info
                                levene = result['levene_test']
                                if not levene['equal_variances']:
                                    st.warning(f"⚠️ Levene's test suggests unequal variances (p = {levene['p_value']:.4f}). Consider using Welch's t-test.")
                            else:
                                st.error(result['error'])
                    else:
                        st.warning("Need at least 2 groups for comparison")
                else:
                    st.warning("Need numeric and categorical columns for comparison")
            
            elif test_type == "Paired T-Test":
                st.write("**Paired T-Test**: Compare means from related samples (before/after)")
                
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        before_col = st.selectbox("Before/First measurement:", numeric_cols, key="paired_before")
                    with col2:
                        after_col = st.selectbox("After/Second measurement:", [c for c in numeric_cols if c != before_col], key="paired_after")
                    
                    alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05, 0.01, key="paired_alpha")
                    
                    if st.button("Run Paired T-Test", type="primary"):
                        result = st.session_state.stat_analyzer.paired_ttest(
                            current_df[before_col], current_df[after_col], alpha
                        )
                        
                        if 'error' not in result:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Before Mean", f"{result['before_mean']:.4f}")
                            with col2:
                                st.metric("After Mean", f"{result['after_mean']:.4f}")
                            with col3:
                                st.metric("Mean Difference", f"{result['mean_difference']:.4f}")
                            with col4:
                                st.metric("p-Value", f"{result['p_value']:.6f}")
                            
                            if result['reject_null']:
                                st.success(f"✅ **Reject H₀**: {result['interpretation']}")
                            else:
                                st.info(f"ℹ️ **Fail to Reject H₀**: {result['interpretation']}")
                        else:
                            st.error(result['error'])
                else:
                    st.warning("Need at least 2 numeric columns for paired test")
            
            elif test_type == "ANOVA":
                st.write("**One-Way ANOVA**: Compare means across multiple groups")
                
                if numeric_cols and categorical_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        value_col = st.selectbox("Select numeric column:", numeric_cols, key="anova_val")
                    with col2:
                        group_col = st.selectbox("Select grouping column:", categorical_cols, key="anova_grp")
                    
                    alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05, 0.01, key="anova_alpha")
                    
                    if st.button("Run ANOVA", type="primary"):
                        result = st.session_state.stat_analyzer.one_way_anova(
                            current_df, value_col, group_col, alpha
                        )
                        
                        if 'error' not in result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("F-Statistic", f"{result['f_statistic']:.4f}")
                            with col2:
                                st.metric("p-Value", f"{result['p_value']:.6f}")
                            with col3:
                                st.metric("η² (Eta-squared)", f"{result['eta_squared']:.4f}")
                            
                            if result['reject_null']:
                                st.success(f"✅ **Reject H₀**: {result['interpretation']}")
                            else:
                                st.info(f"ℹ️ **Fail to Reject H₀**: {result['interpretation']}")
                            
                            st.caption(f"Effect size: {result['effect_size_interpretation']}")
                            
                            # Group statistics
                            st.subheader("Group Statistics")
                            st.dataframe(pd.DataFrame(result['group_statistics']), use_container_width=True)
                        else:
                            st.error(result['error'])
                else:
                    st.warning("Need numeric and categorical columns for ANOVA")
            
            elif test_type == "Chi-Square Test":
                st.write("**Chi-Square Test of Independence**: Test association between categorical variables")
                
                if len(categorical_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        cat1 = st.selectbox("First categorical variable:", categorical_cols, key="chi_cat1")
                    with col2:
                        cat2 = st.selectbox("Second categorical variable:", [c for c in categorical_cols if c != cat1], key="chi_cat2")
                    
                    alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05, 0.01, key="chi_alpha")
                    
                    if st.button("Run Chi-Square Test", type="primary"):
                        result = st.session_state.stat_analyzer.chi_square_independence(
                            current_df, cat1, cat2, alpha
                        )
                        
                        if 'error' not in result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("χ² Statistic", f"{result['chi_square_statistic']:.4f}")
                            with col2:
                                st.metric("p-Value", f"{result['p_value']:.6f}")
                            with col3:
                                st.metric("Cramér's V", f"{result['cramers_v']:.4f}")
                            
                            if result['reject_null']:
                                st.success(f"✅ **Reject H₀**: {result['interpretation']}")
                            else:
                                st.info(f"ℹ️ **Fail to Reject H₀**: {result['interpretation']}")
                            
                            st.caption(f"Effect size: {result['effect_size_interpretation']}")
                        else:
                            st.error(result['error'])
                else:
                    st.warning("Need at least 2 categorical columns for chi-square test")
            
            elif test_type == "Mann-Whitney U":
                st.write("**Mann-Whitney U Test**: Non-parametric alternative to two-sample t-test")
                
                if numeric_cols and categorical_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        value_col = st.selectbox("Select numeric column:", numeric_cols, key="mwu_val")
                    with col2:
                        group_col = st.selectbox("Select grouping column:", categorical_cols, key="mwu_grp")
                    
                    groups = current_df[group_col].dropna().unique()
                    if len(groups) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            group1 = st.selectbox("Group 1:", groups, key="mwu_g1")
                        with col2:
                            group2 = st.selectbox("Group 2:", [g for g in groups if g != group1], key="mwu_g2")
                        
                        if st.button("Run Mann-Whitney U Test", type="primary"):
                            g1_data = current_df[current_df[group_col] == group1][value_col]
                            g2_data = current_df[current_df[group_col] == group2][value_col]
                            
                            result = st.session_state.stat_analyzer.mann_whitney_u(g1_data, g2_data)
                            
                            if 'error' not in result:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("U Statistic", f"{result['u_statistic']:.4f}")
                                with col2:
                                    st.metric("p-Value", f"{result['p_value']:.6f}")
                                with col3:
                                    st.metric("Rank-Biserial r", f"{result['rank_biserial_r']:.4f}")
                                
                                if result['reject_null']:
                                    st.success(f"✅ **Reject H₀**: {result['interpretation']}")
                                else:
                                    st.info(f"ℹ️ **Fail to Reject H₀**: {result['interpretation']}")
                            else:
                                st.error(result['error'])
                else:
                    st.warning("Need numeric and categorical columns for Mann-Whitney U test")
            
            elif test_type == "Kruskal-Wallis":
                st.write("**Kruskal-Wallis H Test**: Non-parametric alternative to one-way ANOVA")
                
                if numeric_cols and categorical_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        value_col = st.selectbox("Select numeric column:", numeric_cols, key="kw_val")
                    with col2:
                        group_col = st.selectbox("Select grouping column:", categorical_cols, key="kw_grp")
                    
                    if st.button("Run Kruskal-Wallis Test", type="primary"):
                        result = st.session_state.stat_analyzer.kruskal_wallis(
                            current_df, value_col, group_col
                        )
                        
                        if 'error' not in result:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("H Statistic", f"{result['h_statistic']:.4f}")
                            with col2:
                                st.metric("p-Value", f"{result['p_value']:.6f}")
                            
                            if result['reject_null']:
                                st.success(f"✅ **Reject H₀**: {result['interpretation']}")
                            else:
                                st.info(f"ℹ️ **Fail to Reject H₀**: {result['interpretation']}")
                            
                            st.subheader("Group Statistics (Medians)")
                            st.dataframe(pd.DataFrame(result['group_statistics']), use_container_width=True)
                        else:
                            st.error(result['error'])
                else:
                    st.warning("Need numeric and categorical columns for Kruskal-Wallis test")
        
        # =============================================
        # TAB 5: OUTLIER DETECTION
        # =============================================
        with analytics_tab5:
            st.subheader("🎯 Outlier Detection")
            st.markdown("Identify and analyze outliers in your data")
            
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    outlier_col = st.selectbox("Select column for outlier detection:", numeric_cols, key="outlier_col")
                with col2:
                    method = st.selectbox("Detection method:", ["IQR Method", "Z-Score Method"])
                
                if method == "IQR Method":
                    multiplier = st.slider("IQR multiplier:", 1.0, 3.0, 1.5, 0.1)
                    result = st.session_state.stat_analyzer.detect_outliers_iqr(current_df[outlier_col], multiplier)
                else:
                    threshold = st.slider("Z-score threshold:", 2.0, 4.0, 3.0, 0.1)
                    result = st.session_state.stat_analyzer.detect_outliers_zscore(current_df[outlier_col], threshold)
                
                if 'error' not in result:
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Outliers", result['n_outliers'])
                    with col2:
                        st.metric("Outlier %", f"{result['outlier_percentage']:.2f}%")
                    with col3:
                        st.metric("Lower Bound" if method == "IQR Method" else "Mean", 
                                 f"{result.get('lower_bound', result.get('mean', 0)):.4f}")
                    with col4:
                        st.metric("Upper Bound" if method == "IQR Method" else "Std Dev",
                                 f"{result.get('upper_bound', result.get('std', 0)):.4f}")
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Box plot with outliers highlighted
                        fig = px.box(current_df, y=outlier_col, title=f"Box Plot of {outlier_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Histogram with bounds
                        fig = px.histogram(current_df, x=outlier_col, title=f"Distribution with Outlier Bounds")
                        if method == "IQR Method":
                            fig.add_vline(x=result['lower_bound'], line_dash="dash", line_color="red", 
                                         annotation_text="Lower Bound")
                            fig.add_vline(x=result['upper_bound'], line_dash="dash", line_color="red",
                                         annotation_text="Upper Bound")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Outlier details
                    if result['n_outliers'] > 0:
                        with st.expander(f"📋 View {result['n_outliers']} Outlier Values"):
                            outlier_df = pd.DataFrame({
                                'Index': result['outlier_indices'][:50],  # Limit display
                                'Value': result['outlier_values'][:50]
                            })
                            st.dataframe(outlier_df, use_container_width=True)
                            if result['n_outliers'] > 50:
                                st.caption(f"Showing first 50 of {result['n_outliers']} outliers")
                else:
                    st.error(result['error'])
            else:
                st.info("No numerical columns available for outlier detection")
        
        # =============================================
        # TAB 6: DATA QUALITY REPORT
        # =============================================
        with analytics_tab6:
            st.subheader("📋 Data Quality Report")
            st.markdown("Comprehensive data quality assessment and profiling")
            
            if st.button("Generate Quality Report", type="primary"):
                with st.spinner("Analyzing data quality..."):
                    report = st.session_state.adv_analytics.generate_data_quality_report(current_df)
                    
                    # Quality Score Card
                    st.subheader("📊 Quality Score")
                    score = report['quality_score']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Overall Score", f"{score['overall']}/100", delta=f"Grade: {score['grade']}")
                    with col2:
                        st.metric("Completeness", f"{score['completeness']:.1f}%")
                    with col3:
                        st.metric("Uniqueness", f"{score['uniqueness']:.1f}%")
                    with col4:
                        issues = report['issues_summary']
                        st.metric("Columns with Issues", issues['columns_with_issues'])
                    
                    st.markdown("---")
                    
                    # Overview
                    st.subheader("📈 Dataset Overview")
                    overview = report['overview']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", f"{overview['total_rows']:,}")
                    with col2:
                        st.metric("Total Columns", overview['total_columns'])
                    with col3:
                        st.metric("Missing Values", f"{overview['total_missing']:,}")
                    with col4:
                        st.metric("Memory Usage", f"{overview['memory_usage_mb']:.2f} MB")
                    
                    # Column Types
                    col_types = report['column_types']
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Numeric", col_types['numeric'])
                    with cols[1]:
                        st.metric("Categorical", col_types['categorical'])
                    with cols[2]:
                        st.metric("Datetime", col_types['datetime'])
                    with cols[3]:
                        st.metric("Boolean", col_types['boolean'])
                    
                    st.markdown("---")
                    
                    # Column Analysis
                    st.subheader("🔍 Column-Level Analysis")
                    
                    col_analysis_df = pd.DataFrame(report['column_analysis'])
                    col_analysis_df['issues'] = col_analysis_df['issues'].apply(lambda x: ', '.join(x) if x else 'None')
                    
                    # Highlight issues
                    def highlight_issues(row):
                        if row['has_issues']:
                            return ['background-color: #ffcccc'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = col_analysis_df.style.apply(highlight_issues, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Issue Summary
                    if report['issues_summary']['columns_with_issues'] > 0:
                        st.warning(f"⚠️ {report['issues_summary']['columns_with_issues']} columns have potential issues")
                        
                        with st.expander("📋 Issues Details"):
                            for col_info in report['column_analysis']:
                                if col_info['issues']:
                                    st.write(f"**{col_info['column']}**: {', '.join(col_info['issues'])}")


if page == "prediction":
    st.header("🤖 Automated Regression Modeling")
    st.markdown("Complete regression pipeline with automated model selection and prediction capabilities")
    st.info("🎯 This system performs **regression only** - predicting continuous numeric values")
    st.markdown("---")
    
    # Initialize ML models if not already done
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = MLModels()
    
    # Add tabs for Training and Prediction
    main_tabs = st.tabs(["🎓 Train Model", "🔮 Make Predictions"])
    
    # ===========================================
    # TAB 1: TRAIN MODEL
    # ===========================================
    with main_tabs[0]:
        st.subheader("🎓 Train New Regression Model")
        
        # Get available processed datasets for ML
        available_datasets = []
        dataset_options = {}
        
        if st.session_state.df_cleaned is not None:
            available_datasets.append("Cleaned Data")
            dataset_options["Cleaned Data"] = st.session_state.df_cleaned
        
        if st.session_state.df_grouped is not None:
            available_datasets.append("Grouped Data")
            dataset_options["Grouped Data"] = st.session_state.df_grouped
        
        if st.session_state.df_merged is not None:
            available_datasets.append("Merged Data")
            dataset_options["Merged Data"] = st.session_state.df_merged
        
        if st.session_state.df_encoded is not None:
            available_datasets.append("Encoded Data")
            dataset_options["Encoded Data"] = st.session_state.df_encoded
        
        if not available_datasets:
            st.warning("⚠️ No processed datasets available for regression. Please process your data first.")
        else:
            # Dataset selection for ML
            selected_dataset = st.selectbox(
                "🤖 Select dataset for regression:",
                available_datasets,
                help="Choose which processed dataset to use for regression modeling"
            )
            
            current_df = dataset_options[selected_dataset]
            st.info(f"Using {selected_dataset} for regression: {current_df.shape[0]:,} rows × {current_df.shape[1]} columns")
            
            # Target Selection Section - FIXED
            st.subheader("🎯 Target Variable Selection (Regression Only)")
            
            # Display only numeric column information
            with st.expander("📊 Available Numeric Columns (Click to expand)", expanded=True):
                col_info = st.session_state.ml_models.get_available_targets(current_df)
                if not col_info.empty:
                    st.dataframe(col_info, use_container_width=True)
                    st.info("ℹ️ Only numeric columns are shown as this system performs regression only.")
                else:
                    st.error("❌ No numeric columns found for regression.")
                    st.stop()  # Stop execution if no numeric columns
            
            # Target selection - only numeric columns
            numeric_columns = current_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                st.error("❌ No numeric columns available for regression target.")
                st.stop()  # Stop execution if no numeric columns
            
            target_column = st.selectbox(
                "Select Target Column for Regression:",
                options=numeric_columns,
                index=len(numeric_columns) - 1,
                help="Choose the numeric column you want to predict (regression target)"
            )
            
            if target_column:
                target_info = current_df[target_column]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Unique Values", target_info.nunique())
                with col2:
                    st.metric("Missing Values", target_info.isnull().sum())
                with col3:
                    st.metric("Min Value", f"{target_info.min():.2f}")
                with col4:
                    st.metric("Max Value", f"{target_info.max():.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean", f"{target_info.mean():.4f}")
                with col2:
                    st.metric("Std Dev", f"{target_info.std():.4f}")
                
                # Show sample values
                st.write(f"**Sample values from '{target_column}':**")
                sample_values = target_info.dropna().head(10).tolist()
                st.write(", ".join([f"{val:.4f}" for val in sample_values]))
                
                # Validation warnings
                if target_info.nunique() < 3:
                    st.error(f"❌ Target has only {target_info.nunique()} unique values. Need at least 3 for regression.")
                    st.stop()
                
                if target_info.std() == 0:
                    st.error("❌ Target has no variation. Cannot perform regression.")
                    st.stop()
                
                st.markdown("---")
                
                # ML Configuration - SIMPLIFIED
                st.subheader("⚙️ Regression Configuration")
                st.info("🤖 We will use GridSearchCV to find the best parameters for each regression model.")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Correlation Threshold", "0.01", help="Features with correlation < 0.01 will be removed")
                
                with col2:
                    st.metric("Test Set Size", "20%", help="Automatically set to 20% of data")
                
                with col3:
                    st.metric("Cross-Validation", "5-fold", help="Automatically set to 5-fold CV")
                
                with st.expander("🔧 Regression Models to be Tuned", expanded=False):
                    models_info = """
                    **Linear Models:**
                    - Linear Regression (fit_intercept)
                    - Ridge Regression (alpha, solver)
                    - Lasso Regression (alpha, selection)  
                    
                    **Tree-based Models:**
                    - Random Forest Regressor (n_estimators, max_depth)
                    - Gradient Boosting Regressor (learning_rate, n_estimators)
                    
                    *All models will undergo hyperparameter optimization.*
                    """
                    st.markdown(models_info)
                
                st.markdown("---")
                
                # Run ML Pipeline - FIXED
                if st.button("🚀 Run Automated Regression Pipeline", use_container_width=True, type="primary"):
                    # Create progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner("🤖 Running automated regression pipeline with hyperparameter tuning..."):
                        try:
                            # Update progress
                            status_text.text("🔄 Preparing data for regression...")
                            progress_bar.progress(10)
                            
                            # Debug information
                            st.write(f"Debug: Using dataset '{selected_dataset}' with shape {current_df.shape}")
                            st.write(f"Debug: Target column '{target_column}' selected")
                            
                            # Update progress
                            status_text.text("🔄 Tuning models (this may take a moment)...")
                            progress_bar.progress(30)
                            
                            # Run the complete regression pipeline
                            results = st.session_state.ml_models.full_regression_pipeline(
                                df=current_df,
                                target_column=target_column,
                                correlation_threshold=0.01,
                                test_size=0.2,
                                cv_folds=5,
                                save_model=True
                            )
                            
                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("✅ Regression pipeline completed successfully!")
                            
                            # Store results
                            st.session_state.ml_results = results
                            st.session_state.selected_target = target_column
                            st.session_state.selected_dataset = selected_dataset
                            
                            # Remove progress elements
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success("🎉 Regression Pipeline completed successfully! Model saved for future predictions.")
                            
                            # Force a rerun to show results
                            st.rerun()
                            
                        except Exception as e:
                            progress_bar.empty()
                            status_text.empty()
                            st.error(f"❌ Error in regression pipeline: {str(e)}")
                            st.exception(e)
                            
                            # Show debug information
                            st.write("Debug information:")
                            st.write(f"- Dataset shape: {current_df.shape}")
                            st.write(f"- Target column: {target_column}")
                            st.write(f"- Target column exists: {target_column in current_df.columns}")
                            st.write(f"- Target column type: {current_df[target_column].dtype}")
                            st.write(f"- Target unique values: {current_df[target_column].nunique()}")
                            
            # Display Results - FIXED
            if hasattr(st.session_state, 'ml_results') and st.session_state.ml_results is not None:
                st.markdown("---")
                st.subheader("📊 Regression Results")
                
                results = st.session_state.ml_results
                selected_dataset = getattr(st.session_state, 'selected_dataset', 'Unknown')
                
                # Check for model bundle
                bundle_path = "regression_model_bundle.zip"
                if os.path.exists(bundle_path):
                    with open(bundle_path, "rb") as fp:
                        btn = st.download_button(
                            label="📥 Download Trained Model Bundle (For Prediction)",
                            data=fp,
                            file_name="regression_model_bundle.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                
                # Quick summary at the top
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🏆 Best Model", results['best_model_name'])
                with col2:
                    st.metric("📊 Problem Type", "REGRESSION")
                with col3:
                    cv_score = results['best_score']
                    st.metric("📈 R² Score", f"{cv_score:.4f}")
                with col4:
                    feature_count = len(st.session_state.ml_models.feature_names) if st.session_state.ml_models.feature_names else 0
                    st.metric("🎯 Features Used", feature_count)
                
                # Results tabs for regression
                result_tabs = st.tabs([
                    "🏆 Best Model", 
                    "📈 Model Comparison", 
                    "🎯 Feature Importance", 
                    "📊 Prediction Analysis",
                    "📋 Training Summary"
                ])
                
                with result_tabs[0]:
                    st.markdown("### 🏆 Best Performing Regression Model")
                    
                    best_model_info = f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <h4>🤖 {results['best_model_name']}</h4>
                        <p><strong>Problem Type:</strong> Regression</p>
                        <p><strong>R² Cross-Validation Score:</strong> {results['best_score']:.4f}</p>
                        <p><strong>Dataset Used:</strong> {selected_dataset}</p>
                    </div>
                    """
                    st.markdown(best_model_info, unsafe_allow_html=True)
                    
                    # Test set performance for regression
                    if 'test_metrics' in results:
                        st.write("### 🎯 Test Set Performance")
                        
                        metrics = results['test_metrics']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
                        with col2:
                            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                        with col3:
                            st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                        with col4:
                            mape = metrics.get('mape', 0)
                            st.metric("MAPE", f"{mape:.2f}%")
                
                with result_tabs[1]:
                    st.write("### 📈 Regression Model Performance Comparison")
                    
                    if 'model_scores' in results:
                        # Create model summary dataframe
                        summary = []
                        for name, info in results['model_scores'].items():
                            summary.append({
                                'Model': name,
                                'R² Score': round(info['mean_score'], 4),
                                'Best Params': str(info.get('best_params', 'N/A')),
                                'Is Best': name == results['best_model_name']
                            })
                        
                        summary_df = pd.DataFrame(summary).sort_values('R² Score', ascending=False)
                        
                        # Style the dataframe for regression
                        styled_df = summary_df.style.highlight_max(subset=['R² Score'], color='lightgreen')
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Performance comparison chart for regression
                        fig = px.bar(
                            summary_df,
                            x='Model',
                            y='R² Score',
                            color='Is Best',
                            title=f"Regression Model R² Score Comparison ({selected_dataset})",
                            text='R² Score',
                            color_discrete_map={True: '#1f77b4', False: '#d62728'}
                        )

                        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            height=500,
                            showlegend=True,
                            legend=dict(title="Best Model"),
                            yaxis_title="R² Score"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with result_tabs[2]:
                    st.write("### 🎯 Feature Importance Analysis")
                    
                    if 'feature_importance' in results and results['feature_importance'] is not None:
                        importance_df = results['feature_importance']
                        
                        # Show top features
                        st.write("**Top 15 Most Important Features for Regression:**")
                        top_features = importance_df.head(15)
                        st.dataframe(top_features, use_container_width=True)
                        
                        # Feature importance plot
                        fig = px.bar(
                            top_features,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Feature Importance for Regression",
                            labels={'importance': 'Importance Score', 'feature': 'Features'},
                            color='importance',
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(
                            yaxis={'categoryorder':'total ascending'},
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with result_tabs[3]:
                    st.write("### 📊 Prediction Analysis")
                    
                    if 'test_metrics' in results and 'predictions' in results['test_metrics']:
                        predictions = results['test_metrics']['predictions']
                        actual = st.session_state.ml_models.y_test
                        
                        # Prediction vs Actual scatter plot
                        fig_scatter = px.scatter(
                            x=actual, 
                            y=predictions,
                            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                            title="Predicted vs Actual Values"
                        )
                        
                        # Add perfect prediction line
                        min_val = min(min(actual), min(predictions))
                        max_val = max(max(actual), max(predictions))
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=[min_val, max_val], 
                                y=[min_val, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            )
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Residuals plot
                        residuals = results['test_metrics']['residuals']
                        fig_residuals = px.scatter(
                            x=predictions,
                            y=residuals,
                            labels={'x': 'Predicted Values', 'y': 'Residuals'},
                            title="Residuals Plot"
                        )
                        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_residuals, use_container_width=True)
                
                with result_tabs[4]:
                    st.write("### 📋 Training Summary")
                    
                    if 'training_summary' in results:
                        summary = results['training_summary']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Dataset Information:**")
                            st.write(f"- Target Column: {summary.get('target_column', 'N/A')}")
                            st.write(f"- Original Features: {summary.get('original_features', 'N/A')}")
                            st.write(f"- Final Features: {summary.get('final_features', 'N/A')}")
                            st.write(f"- Train Size: {summary.get('train_size', 'N/A')}")
                            st.write(f"- Test Size: {summary.get('test_size', 'N/A')}")
                        
                        with col2:
                            st.write("**Model Information:**")
                            st.write(f"- Best Model: {summary.get('best_model', 'N/A')}")
                            st.write(f"- Best Score: {summary.get('best_score', 'N/A'):.4f}")
                            
                            if summary.get('removed_features'):
                                st.write(f"- Removed Features: {len(summary['removed_features'])}")
                                with st.expander("Show removed features"):
                                    st.write(summary['removed_features'])
    
    # ===========================================
    # TAB 2: MAKE PREDICTIONS
    # ===========================================
    with main_tabs[1]:
        st.subheader("🔮 Predict on New Data")
        st.markdown("Upload a trained model bundle and a new dataset to generate predictions.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("1️⃣ Upload Model Bundle")
            uploaded_model = st.file_uploader(
                "Upload 'regression_model_bundle.zip'",
                type=['zip'],
                key="model_uploader"
            )
            
            if uploaded_model:
                # Save to temp file and load
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                    tmp_file.write(uploaded_model.getvalue())
                    model_path = tmp_file.name
                
                try:
                    if st.session_state.ml_models.load_model_from_zip(model_path):
                        st.success(f"✅ Model loaded: {st.session_state.ml_models.best_model_name}")
                        st.write(f"**Target:** {st.session_state.ml_models.target_column}")
                        st.write(f"**Best R²:** {st.session_state.ml_models.best_score:.4f}")
                        st.session_state.model_loaded = True
                    else:
                        st.error("❌ Failed to load model bundle.")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(model_path)
        
        with col2:
            st.info("2️⃣ Upload New Data")
            uploaded_data = st.file_uploader(
                "Upload new dataset (CSV/Excel)",
                type=['csv', 'xlsx'],
                key="prediction_data_uploader"
            )
            
            if uploaded_data:
                # Reuse loader logic
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_data.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_data.getvalue())
                    data_path = tmp_file.name
                
                try:
                    new_df = st.session_state.loader.loader(data_path)
                    st.success(f"✅ Data loaded: {new_df.shape[0]} rows × {new_df.shape[1]} columns")
                    st.session_state.new_prediction_df = new_df
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                finally:
                    os.unlink(data_path)

        st.markdown("---")
        
        # Predict Button
        if st.session_state.get('model_loaded') and st.session_state.get('new_prediction_df') is not None:
            if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                with st.spinner("Generating predictions..."):
                    try:
                        predictions = st.session_state.ml_models.predict_new_data(st.session_state.new_prediction_df)
                        
                        if predictions is not None:
                            # Add predictions to dataframe
                            result_df = st.session_state.new_prediction_df.copy()
                            target_col = st.session_state.ml_models.target_column
                            prediction_col = f"Predicted_{target_col}"
                            result_df[prediction_col] = predictions
                            
                            st.success("✅ Predictions generated successfully!")
                            
                            # Show results
                            st.subheader("📊 Prediction Results")
                            st.dataframe(result_df.head(10), use_container_width=True)
                            
                            # Download results
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions (CSV)",
                                data=csv,
                                file_name=f"predictions_{target_col}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Visualization of predictions distribution
                            st.subheader("📈 Predictions Distribution")
                            fig = px.histogram(
                                result_df, 
                                x=prediction_col,
                                title=f"Distribution of Predicted {target_col}",
                                marginal="box"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
        elif not st.session_state.get('model_loaded') and not st.session_state.get('new_prediction_df'):
             st.info("👈 Upload both a model and a dataset to start.")
                
                



   

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666; padding: 20px;'>
        <p>🚀 <strong>Complete Data Processing & Analytics Platform</strong></p>
        <p>Load • Clean • Transform • Group • Merge • Encode • Visualize • Predict</p>
        <p><em>Built with Streamlit, Plotly, and Pandas</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)