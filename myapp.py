import base64
import logging
import os
import tempfile
from datetime import datetime
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import lightweight custom classes (always needed)
from src.data_processing.cleaner import DataCleaner
from src.data_processing.loader import DataLoader
from src.data_processing.transformer import DataTransformer
from src.visualization.charts import DataVisualizer
from src.visualization.dashboard import Dashboard

# Heavy analytics modules — imported lazily when their pages are visited
# from src.analytics.ml_models import MLModels
# from src.analytics.statistical import StatisticalAnalyzer
# from src.analytics.advanced_analytics import AdvancedAnalytics

# Configuration du logging pour Streamlit
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Page configuration with professional styling
st.set_page_config(
    page_title="DataFlow Pro - Advanced Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional design
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        "df_original": None,  # Original raw data
        "df_cleaned": None,  # Cleaned data - MAIN WORKING DATAFRAME
        "df_second": None,  # Second dataset for merging
        "df_grouped": None,  # Independent grouped data
        "df_merged": None,  # Independent merged data
        "df_encoded": None,  # Independent encoded data
        "loader": DataLoader(),
        "cleaner": DataCleaner(),
        "transformer": DataTransformer(),
        "visualizer": DataVisualizer(),
        "dashboard": None,
        "ml_models": None,  # Created lazily when prediction page is visited
        "processing_logs": [],
        "plots_history": [],
        "current_plot": None,
        "plot_counter": 0,
        "current_page": "data_pipeline",  # Default page
        "selected_target": None,
        "uploaded_dataframes": {},
        "uploaded_dataset_order": [],
        "active_dataset_key": None,
        "processed_data": {},
        "model_diagnostics_cache": {},
        "processing_mode": "Apply to All",
        "cleaning_applied": False,
        "grouping_applied": False,
        "merging_applied": False,
        "encoding_applied": False,
        "target_encoder_bytes": None,
        "target_encoder_columns": [],
        "loaded_prediction_model": None,
        "loaded_target_encoder": None,
        "new_prediction_df": None,
        "model_loaded": False,
        "prediction_result_df": None,
        "prediction_col_name": None,
        "prediction_target_col": None,
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
    plot_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.plots_history.append(plot_info)

    # Keep only last 20 plots
    if len(st.session_state.plots_history) > 20:
        st.session_state.plots_history = st.session_state.plots_history[-20:]


def create_download_link(df, filename, file_format="csv"):
    """Create download link for dataframes"""
    if file_format == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📄 Download CSV</a>'
    elif file_format == "excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">📊 Download Excel</a>'

    return href


def export_plot_as_html(fig, filename):
    """Export plotly figure as HTML"""
    html_string = fig.to_html(include_plotlyjs="cdn")
    b64 = base64.b64encode(html_string.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html">📊 Download Plot (HTML)</a>'
    return href


@st.cache_resource(show_spinner=False)
def load_cached_target_encoder(encoder_bytes: bytes):
    """Deserialize and cache a fitted TargetEncoder for reuse across reruns."""
    return joblib.load(BytesIO(encoder_bytes))


def ensure_encoder_input_columns(dataframe, encoder):
    """Match inference input to the exact raw schema expected by the fitted encoder."""
    aligned_df = dataframe.copy()

    expected_columns = None

    if hasattr(encoder, "feature_names_in_"):
        expected_columns = list(encoder.feature_names_in_)
    elif hasattr(encoder, "ordinal_encoder") and hasattr(encoder.ordinal_encoder, "feature_names_in_"):
        expected_columns = list(encoder.ordinal_encoder.feature_names_in_)
    elif hasattr(encoder, "get_feature_names_in"):
        expected_columns = list(encoder.get_feature_names_in())
    else:
        expected_columns = aligned_df.columns.tolist()

    for column in expected_columns:
        if column not in aligned_df.columns:
            aligned_df[column] = np.nan

    aligned_df = aligned_df[expected_columns]

    return aligned_df


def align_prediction_features(dataframe, expected_features):
    """Match transformed inference data to the exact feature set used by the trained model."""
    aligned_df = dataframe.copy()

    for column in expected_features:
        if column not in aligned_df.columns:
            aligned_df[column] = 0

    aligned_df = aligned_df[expected_features]

    if aligned_df.isna().any().any():
        aligned_df = aligned_df.fillna(0)

    numeric_subset = aligned_df.select_dtypes(include=[np.number])
    if np.isinf(numeric_subset).any().any():
        aligned_df = aligned_df.replace([np.inf, -np.inf], 0)

    return aligned_df


def get_uploaded_file_extension(uploaded_file):
    """Return the lowercase extension for an uploaded file."""
    return os.path.splitext(uploaded_file.name)[1].lower()


def inspect_excel_sheets(uploaded_file):
    """Read workbook sheet names from an uploaded Excel file."""
    workbook = BytesIO(uploaded_file.getvalue())
    with pd.ExcelFile(workbook, engine="openpyxl") as excel_file:
        return excel_file.sheet_names


def build_sheet_preferences(uploaded_files):
    """Build UI controls for Excel sheet selection."""
    preferences = {}

    for uploaded_file in uploaded_files:
        extension = get_uploaded_file_extension(uploaded_file)
        if extension not in {".xlsx", ".xlsm"}:
            continue

        sheet_names = inspect_excel_sheets(uploaded_file)
        if len(sheet_names) <= 1:
            preferences[uploaded_file.name] = {
                "load_all": False,
                "selected_sheets": sheet_names,
            }
            continue

        with st.expander(f"Sheet selection: {uploaded_file.name}", expanded=False):
            st.caption(f"Detected {len(sheet_names)} sheets in `{uploaded_file.name}`.")
            load_all = st.checkbox(
                f"Load all sheets from {uploaded_file.name}",
                value=True,
                key=f"load_all_sheets_{uploaded_file.name}",
            )

            selected_sheets = sheet_names
            if not load_all:
                selected_sheets = st.multiselect(
                    f"Choose sheets from {uploaded_file.name}",
                    options=sheet_names,
                    default=sheet_names[:1],
                    key=f"sheet_selector_{uploaded_file.name}",
                )

            preferences[uploaded_file.name] = {
                "load_all": load_all,
                "selected_sheets": selected_sheets,
            }

    return preferences


def has_multi_sheet_uploads(uploaded_files):
    """Return True when at least one uploaded Excel workbook has multiple sheets."""
    for uploaded_file in uploaded_files:
        extension = get_uploaded_file_extension(uploaded_file)
        if extension not in {".xlsx", ".xlsm"}:
            continue
        if len(inspect_excel_sheets(uploaded_file)) > 1:
            return True
    return False


def load_uploaded_files(uploaded_files, sheet_preferences):
    """Load uploaded files and return flattened datasets keyed by filename and sheet."""
    loaded_dataframes = {}
    dataset_order = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        extension = get_uploaded_file_extension(uploaded_file)
        status_text.info(f"Loading `{uploaded_file.name}` ({index}/{len(uploaded_files)})")
        add_log(f"Loading file: {uploaded_file.name}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            load_kwargs = {}
            if extension in {".xlsx", ".xlsm"}:
                preference = sheet_preferences.get(uploaded_file.name, {})
                selected_sheets = preference.get("selected_sheets") or []
                load_all = preference.get("load_all", False)

                if not load_all and not selected_sheets:
                    raise ValueError("At least one sheet must be selected.")

                if load_all:
                    load_kwargs["load_all_sheets"] = True
                else:
                    load_kwargs["sheet_name"] = selected_sheets if len(selected_sheets) > 1 else selected_sheets[0]

            loaded = st.session_state.loader.load_data(tmp_file_path, **load_kwargs)

            if isinstance(loaded, dict):
                for sheet_name, dataframe in loaded.items():
                    dataset_key = f"{uploaded_file.name}_{sheet_name}"
                    loaded_dataframes[dataset_key] = dataframe
                    dataset_order.append(dataset_key)
                    add_log(f"Loaded sheet '{sheet_name}' from {uploaded_file.name} - Shape: {dataframe.shape}")
            else:
                dataset_key = uploaded_file.name
                loaded_dataframes[dataset_key] = loaded
                dataset_order.append(dataset_key)
                add_log(f"Loaded {uploaded_file.name} - Shape: {loaded.shape}")
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

        progress_bar.progress(index / len(uploaded_files))

    status_text.success(f"Loaded {len(dataset_order)} dataset(s) from {len(uploaded_files)} file(s).")
    return loaded_dataframes, dataset_order


def render_dataset_preview(dataset_key, dataframe):
    """Render a compact preview and profile for a loaded dataset."""
    with st.expander(f"Preview: {dataset_key}", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", dataframe.shape[0])
        with col2:
            st.metric("Columns", dataframe.shape[1])
        with col3:
            st.metric("Memory Usage", f"{dataframe.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

        st.dataframe(dataframe.head(10), use_container_width=True)

        col_info = pd.DataFrame(
            {
                "Column": dataframe.columns,
                "Data Type": dataframe.dtypes.astype(str),
                "Non-Null Count": dataframe.count(),
                "Null Count": dataframe.isnull().sum(),
                "Unique Values": [dataframe[col].nunique() for col in dataframe.columns],
            }
        )
        st.dataframe(col_info, use_container_width=True)


def sync_active_dataset():
    """Keep the active raw dataframe aligned with the selected uploaded dataset."""
    active_key = st.session_state.get("active_dataset_key")
    uploaded_dataframes = st.session_state.get("uploaded_dataframes", {})
    if active_key and active_key in uploaded_dataframes:
        st.session_state.df_original = uploaded_dataframes[active_key].copy()


def sync_processed_dataset_views():
    """Mirror the active processed dataset into legacy single-dataset session keys."""
    active_key = st.session_state.get("active_dataset_key")
    processed_data = st.session_state.get("processed_data", {})
    dataset_stages = processed_data.get(active_key)

    if not dataset_stages:
        return

    st.session_state.df_cleaned = dataset_stages.get("cleaned")
    st.session_state.df_encoded = dataset_stages.get("encoded")
    st.session_state.df_grouped = dataset_stages.get("grouped")
    st.session_state.cleaning_applied = dataset_stages.get("cleaned") is not None
    st.session_state.encoding_applied = dataset_stages.get("encoded") is not None
    st.session_state.grouping_applied = dataset_stages.get("grouped") is not None


def parse_column_rename_rules(raw_rules):
    """Parse newline-delimited old:new column rename rules."""
    rename_map = {}
    for line in raw_rules.splitlines():
        if ":" not in line:
            continue
        original, renamed = line.split(":", 1)
        original = original.strip()
        renamed = renamed.strip()
        if original and renamed:
            rename_map[original] = renamed
    return rename_map


def get_common_columns(datasets):
    """Return the sorted intersection of dataset columns."""
    if not datasets:
        return []

    dataset_values = list(datasets.values())
    common = set(dataset_values[0].columns)
    for dataframe in dataset_values[1:]:
        common &= set(dataframe.columns)
    return sorted(common)


def get_common_numeric_columns(datasets):
    """Return columns that are numeric across every dataset."""
    if not datasets:
        return []

    common_numeric = set(datasets[next(iter(datasets))].select_dtypes(include=[np.number]).columns)
    for dataframe in datasets.values():
        common_numeric &= set(dataframe.select_dtypes(include=[np.number]).columns)
    return sorted(common_numeric)


def render_cleaning_controls(dataframe, key_prefix):
    """Render cleaning controls for a dataset scope."""
    selectable_columns = dataframe.columns.tolist()
    st.markdown("**Cleaning**")
    enable_cleaning = st.checkbox(
        "Apply cleaning",
        value=True,
        key=f"{key_prefix}_enable_cleaning",
    )
    cleaning_strategy = st.selectbox(
        "Cleaning strategy",
        ["statique", "frequence"],
        key=f"{key_prefix}_cleaning_strategy",
    )
    frequency_lines = 1.0
    if cleaning_strategy == "frequence":
        frequency_lines = st.number_input(
            "Interpolation frequency (hours/rows setting)",
            min_value=0.1,
            max_value=24.0,
            value=1.0,
            step=0.1,
            key=f"{key_prefix}_frequency_lines",
        )

    rename_rules = st.text_area(
        "Column rename rules (one `old:new` pair per line)",
        value="",
        key=f"{key_prefix}_rename_rules",
    )

    return {
        "enable_cleaning": enable_cleaning,
        "cleaning_strategy": cleaning_strategy,
        "frequency_lines": frequency_lines,
        "rename_rules": rename_rules,
        "available_columns": selectable_columns,
    }


def render_encoding_controls(
    dataframe,
    key_prefix,
    selectable_columns=None,
    selectable_numeric_columns=None,
):
    """Render encoding controls for a dataset scope."""
    selectable_columns = selectable_columns or dataframe.columns.tolist()
    selectable_numeric_columns = selectable_numeric_columns or [
        column for column in dataframe.select_dtypes(include=[np.number]).columns if column in selectable_columns
    ]

    st.markdown("**Encoding**")
    enable_encoding = st.checkbox(
        "Apply encoding",
        value=False,
        key=f"{key_prefix}_enable_encoding",
    )
    encoding_strategy = st.selectbox(
        "Encoding strategy",
        ["smart", "one_hot", "label", "none"],
        key=f"{key_prefix}_encoding_strategy",
    )
    target_column = st.selectbox(
        "Target column (optional)",
        options=["None"] + selectable_numeric_columns,
        key=f"{key_prefix}_target_column",
    )

    return {
        "enable_encoding": enable_encoding and encoding_strategy != "none",
        "encoding_strategy": encoding_strategy,
        "target_column": None if target_column == "None" else target_column,
    }


def render_grouping_controls(
    dataframe,
    key_prefix,
    selectable_columns=None,
):
    """Render grouping controls for a dataset scope."""
    selectable_columns = selectable_columns or dataframe.columns.tolist()

    st.markdown("**Grouping / Aggregation**")
    enable_grouping = st.checkbox(
        "Apply grouping",
        value=False,
        key=f"{key_prefix}_enable_grouping",
    )
    group_columns = st.multiselect(
        "Group by columns",
        options=selectable_columns,
        key=f"{key_prefix}_group_columns",
    )
    aggregation_functions = st.multiselect(
        "Aggregation functions",
        options=["sum", "mean", "max", "min", "count", "std"],
        default=["sum", "mean"],
        key=f"{key_prefix}_aggregation_functions",
    )

    return {
        "enable_grouping": enable_grouping,
        "group_columns": group_columns,
        "aggregation_functions": aggregation_functions,
    }


def apply_encoding_strategy(dataframe, strategy, target_column=None):
    """Apply the selected categorical encoding strategy."""
    working_df = filter_zero_only_numeric_rows(dataframe)
    if strategy == "smart":
        return st.session_state.transformer.smart_categorical_encoding(
            working_df,
            target_col=target_column,
        )

    categorical_columns = working_df.select_dtypes(include=["object", "category"]).columns.tolist()

    if strategy == "one_hot":
        encoded_df = pd.get_dummies(working_df, columns=categorical_columns, dtype=int)
        bool_columns = encoded_df.select_dtypes(include="bool").columns
        if len(bool_columns) > 0:
            encoded_df[bool_columns] = encoded_df[bool_columns].astype(int)
        return encoded_df

    if strategy == "label":
        for column in categorical_columns:
            working_df[column] = pd.factorize(working_df[column].fillna("Unknown"), sort=True)[0]
        bool_columns = working_df.select_dtypes(include="bool").columns
        if len(bool_columns) > 0:
            working_df[bool_columns] = working_df[bool_columns].astype(int)
        return working_df

    return working_df


def filter_zero_only_numeric_rows(dataframe):
    """Drop rows where every non-boolean numeric column is exactly zero."""
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns

    if len(numeric_columns) == 0:
        return dataframe.copy()

    zero_only_mask = dataframe.loc[:, numeric_columns].eq(0).all(axis=1)
    return dataframe.loc[~zero_only_mask].reset_index(drop=True)


def apply_grouping_settings(dataframe, group_columns, aggregation_functions):
    """Group a dataframe using user-selected aggregation functions."""
    if not group_columns:
        return dataframe.copy()
    if not aggregation_functions:
        aggregation_functions = ["sum"]

    missing_columns = [column for column in group_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Grouping columns not found: {missing_columns}")

    datetime_columns = dataframe.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    numeric_columns = [
        column
        for column in dataframe.select_dtypes(include=[np.number]).columns
        if column not in set(group_columns) | set(datetime_columns)
    ]

    if not numeric_columns:
        return dataframe.groupby(group_columns).first().reset_index()

    agg_map = {column: aggregation_functions for column in numeric_columns}
    grouped = dataframe.groupby(group_columns).agg(agg_map).reset_index()
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = [
            "_".join(part for part in column if part).strip("_") if isinstance(column, tuple) else column
            for column in grouped.columns
        ]
    return grouped


def drop_selected_columns(dataframe, columns_to_drop):
    """Drop user-selected columns that exist in the dataframe."""
    if not columns_to_drop:
        return dataframe.copy()

    valid_columns = [column for column in columns_to_drop if column in dataframe.columns]
    if not valid_columns:
        return dataframe.copy()

    return dataframe.drop(columns=valid_columns)


def process_dataset_with_config(dataset_key, dataframe, config):
    """Run cleaning, encoding, and grouping for one dataset."""
    rename_map = parse_column_rename_rules(config["rename_rules"])
    working_df = dataframe.copy()
    cleaned_df = working_df.copy()

    if config["enable_cleaning"]:
        cleaned_df = st.session_state.cleaner.clean_data(
            working_df.copy(),
            config["cleaning_strategy"],
            config["frequency_lines"],
        )
        if rename_map:
            valid_renames = {old: new for old, new in rename_map.items() if old in cleaned_df.columns}
            if valid_renames:
                cleaned_df = cleaned_df.rename(columns=valid_renames)
    effective_target = config["target_column"]
    if effective_target in rename_map:
        effective_target = rename_map[effective_target]

    effective_group_columns = [rename_map.get(column, column) for column in config["group_columns"]]

    encoded_df = cleaned_df.copy()
    if config["enable_encoding"]:
        encoded_df = apply_encoding_strategy(
            cleaned_df.copy(),
            config["encoding_strategy"],
            target_column=effective_target,
        )

    grouped_df = cleaned_df.copy()
    if config["enable_grouping"] and effective_group_columns:
        grouped_df = apply_grouping_settings(
            cleaned_df.copy(),
            effective_group_columns,
            config["aggregation_functions"],
        )

    final_df = grouped_df.copy() if config["enable_grouping"] else encoded_df.copy()

    return {
        "raw": dataframe.copy(),
        "cleaned": cleaned_df.copy(),
        "encoded": encoded_df.copy() if config["enable_encoding"] else None,
        "grouped": grouped_df.copy() if config["enable_grouping"] else None,
        "final": final_df,
        "config": config,
        "summary": {
            "raw_shape": dataframe.shape,
            "cleaned_shape": cleaned_df.shape,
            "encoded_shape": encoded_df.shape if config["enable_encoding"] else None,
            "grouped_shape": grouped_df.shape if config["enable_grouping"] else None,
            "final_shape": final_df.shape,
        },
    }


def get_stage_input_dataframe(dataset_key, preferred_stage):
    """Return the best available in-memory dataframe for a processing stage."""
    processed_entry = st.session_state.get("processed_data", {}).get(dataset_key, {})
    if preferred_stage == "encoding":
        return (
            processed_entry.get("cleaned")
            or processed_entry.get("raw")
            or st.session_state.uploaded_dataframes[dataset_key]
        )
    if preferred_stage == "grouping":
        return (
            processed_entry.get("encoded")
            or processed_entry.get("cleaned")
            or processed_entry.get("raw")
            or st.session_state.uploaded_dataframes[dataset_key]
        )
    return processed_entry.get("raw") or st.session_state.uploaded_dataframes[dataset_key]


def process_cleaning_batch(config_map):
    """Apply only cleaning settings across datasets."""
    processed = st.session_state.get("processed_data", {}).copy()
    ordered_keys = st.session_state.get("uploaded_dataset_order", [])
    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, dataset_key in enumerate(ordered_keys, start=1):
        status_text.info(f"Cleaning `{dataset_key}` ({index}/{len(ordered_keys)})")
        raw_df = st.session_state.uploaded_dataframes[dataset_key].copy()
        config = config_map[dataset_key]
        rename_map = parse_column_rename_rules(config["rename_rules"])
        cleaned_df = raw_df.copy()

        if config["enable_cleaning"]:
            cleaned_df = st.session_state.cleaner.clean_data(
                raw_df.copy(),
                config["cleaning_strategy"],
                config["frequency_lines"],
            )
        if rename_map:
            valid_renames = {old: new for old, new in rename_map.items() if old in cleaned_df.columns}
            if valid_renames:
                cleaned_df = cleaned_df.rename(columns=valid_renames)
        existing = processed.get(dataset_key, {"raw": raw_df.copy()})
        existing["raw"] = raw_df.copy()
        existing["cleaned"] = cleaned_df.copy()
        existing["final"] = cleaned_df.copy()
        existing.setdefault("summary", {})
        existing["summary"]["raw_shape"] = raw_df.shape
        existing["summary"]["cleaned_shape"] = cleaned_df.shape
        processed[dataset_key] = existing
        progress_bar.progress(index / len(ordered_keys))

    status_text.success(f"Cleaned {len(ordered_keys)} dataset(s).")
    return processed


def process_encoding_batch(config_map):
    """Apply only encoding settings across datasets."""
    processed = st.session_state.get("processed_data", {}).copy()
    ordered_keys = st.session_state.get("uploaded_dataset_order", [])
    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, dataset_key in enumerate(ordered_keys, start=1):
        status_text.info(f"Encoding `{dataset_key}` ({index}/{len(ordered_keys)})")
        base_df = get_stage_input_dataframe(dataset_key, "encoding").copy()
        config = config_map[dataset_key]
        encoded_df = base_df.copy()

        if config["enable_encoding"]:
            encoded_df = apply_encoding_strategy(
                base_df.copy(),
                config["encoding_strategy"],
                target_column=config["target_column"],
            )

        existing = processed.get(dataset_key, {"raw": st.session_state.uploaded_dataframes[dataset_key].copy()})
        existing["encoded"] = encoded_df.copy() if config["enable_encoding"] else None
        existing["final"] = encoded_df.copy()
        existing.setdefault("summary", {})
        existing["summary"]["encoded_shape"] = encoded_df.shape if config["enable_encoding"] else None
        existing["summary"]["final_shape"] = encoded_df.shape
        processed[dataset_key] = existing
        progress_bar.progress(index / len(ordered_keys))

    status_text.success(f"Encoded {len(ordered_keys)} dataset(s).")
    return processed


def process_grouping_batch(config_map):
    """Apply only grouping settings across datasets."""
    processed = st.session_state.get("processed_data", {}).copy()
    ordered_keys = st.session_state.get("uploaded_dataset_order", [])
    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, dataset_key in enumerate(ordered_keys, start=1):
        status_text.info(f"Grouping `{dataset_key}` ({index}/{len(ordered_keys)})")
        base_df = get_stage_input_dataframe(dataset_key, "grouping").copy()
        config = config_map[dataset_key]
        grouped_df = base_df.copy()

        if config["enable_grouping"] and config["group_columns"]:
            grouped_df = apply_grouping_settings(
                base_df.copy(),
                config["group_columns"],
                config["aggregation_functions"],
            )

        existing = processed.get(dataset_key, {"raw": st.session_state.uploaded_dataframes[dataset_key].copy()})
        existing["grouped"] = grouped_df.copy() if config["enable_grouping"] else None
        existing["final"] = grouped_df.copy()
        existing.setdefault("summary", {})
        existing["summary"]["grouped_shape"] = grouped_df.shape if config["enable_grouping"] else None
        existing["summary"]["final_shape"] = grouped_df.shape
        processed[dataset_key] = existing
        progress_bar.progress(index / len(ordered_keys))

    status_text.success(f"Grouped {len(ordered_keys)} dataset(s).")
    return processed


def apply_post_cleaning_column_drop(dataset_key, columns_to_drop):
    """Apply column deletion to an already cleaned dataset and persist it."""
    processed = st.session_state.get("processed_data", {})
    if dataset_key not in processed or processed[dataset_key].get("cleaned") is None:
        raise ValueError(f"No cleaned dataset available for '{dataset_key}'.")

    cleaned_df = processed[dataset_key]["cleaned"].copy()
    updated_cleaned_df = drop_selected_columns(cleaned_df, columns_to_drop)
    processed[dataset_key]["cleaned"] = updated_cleaned_df
    processed[dataset_key]["final"] = updated_cleaned_df.copy()
    processed[dataset_key].setdefault("summary", {})
    processed[dataset_key]["summary"]["cleaned_shape"] = updated_cleaned_df.shape
    processed[dataset_key]["summary"]["final_shape"] = updated_cleaned_df.shape
    processed[dataset_key]["post_cleaning_dropped_columns"] = columns_to_drop
    st.session_state.processed_data = processed


def apply_post_cleaning_zero_only_row_removal(dataset_key, datetime_col=None):
    """Remove zero-only rows from an already cleaned dataset and persist it."""
    from src.analytics.ml_models import remove_zero_only_rows

    processed = st.session_state.get("processed_data", {})
    if dataset_key not in processed or processed[dataset_key].get("cleaned") is None:
        raise ValueError(f"No cleaned dataset available for '{dataset_key}'.")

    cleaned_df = processed[dataset_key]["cleaned"].copy()
    updated_cleaned_df = remove_zero_only_rows(cleaned_df, datetime_col=datetime_col)
    removed_count = len(cleaned_df) - len(updated_cleaned_df)

    processed[dataset_key]["cleaned"] = updated_cleaned_df
    processed[dataset_key]["final"] = updated_cleaned_df.copy()
    processed[dataset_key].setdefault("summary", {})
    processed[dataset_key]["summary"]["cleaned_shape"] = updated_cleaned_df.shape
    processed[dataset_key]["summary"]["final_shape"] = updated_cleaned_df.shape
    processed[dataset_key]["post_cleaning_zero_only_rows_removed"] = removed_count
    st.session_state.processed_data = processed
    return removed_count


def get_cleaned_dataset_columns(processed_data):
    """Return the common columns across cleaned datasets."""
    cleaned_datasets = [
        dataset_entry["cleaned"]
        for dataset_entry in processed_data.values()
        if dataset_entry.get("cleaned") is not None
    ]
    if not cleaned_datasets:
        return []

    common_columns = set(cleaned_datasets[0].columns)
    for dataframe in cleaned_datasets[1:]:
        common_columns &= set(dataframe.columns)
    return sorted(common_columns)


def process_multiple_datasets(config_map):
    """Apply batch processing configs to every uploaded dataset in memory."""
    processed = {}
    ordered_keys = st.session_state.get("uploaded_dataset_order", [])
    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, dataset_key in enumerate(ordered_keys, start=1):
        status_text.info(f"Processing `{dataset_key}` ({index}/{len(ordered_keys)})")
        dataset = st.session_state.uploaded_dataframes[dataset_key]
        processed[dataset_key] = process_dataset_with_config(dataset_key, dataset, config_map[dataset_key])
        progress_bar.progress(index / len(ordered_keys))
        add_log(f"Processed dataset: {dataset_key}")

    status_text.success(f"Processed {len(processed)} dataset(s).")
    return processed


def render_processed_results(processed_data):
    """Display final processed dataset previews."""
    st.subheader("Processed Dataset Preview")
    for dataset_key in st.session_state.get("uploaded_dataset_order", []):
        if dataset_key not in processed_data:
            continue
        dataset_result = processed_data[dataset_key]
        summary = dataset_result["summary"]
        with st.expander(f"Processed: {dataset_key}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Raw", f"{summary['raw_shape'][0]} x {summary['raw_shape'][1]}")
            with col2:
                st.metric("Cleaned", f"{summary['cleaned_shape'][0]} x {summary['cleaned_shape'][1]}")
            with col3:
                st.metric(
                    "Encoded",
                    (
                        f"{summary['encoded_shape'][0]} x {summary['encoded_shape'][1]}"
                        if summary["encoded_shape"] is not None
                        else "Not applied"
                    ),
                )
            with col4:
                st.metric("Final", f"{summary['final_shape'][0]} x {summary['final_shape'][1]}")

            st.dataframe(dataset_result["final"].head(10), use_container_width=True)


def compare_datasets(df_original, df_processed, title="Dataset Comparison"):
    """Compare original vs processed datasets"""
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Dataset**")
        st.metric("Rows", f"{len(df_original):,}")
        st.metric("Columns", f"{len(df_original.columns):,}")
        st.metric("Missing Values", f"{df_original.isnull().sum().sum():,}")
        st.metric("Memory (MB)", f"{df_original.memory_usage(deep=True).sum() / 1024**2:.1f}")

    with col2:
        st.write(f"**{title}**")
        st.metric("Rows", f"{len(df_processed):,}", delta=f"{len(df_processed) - len(df_original):,}")
        st.metric(
            "Columns",
            f"{len(df_processed.columns):,}",
            delta=f"{len(df_processed.columns) - len(df_original.columns):,}",
        )
        st.metric(
            "Missing Values",
            f"{df_processed.isnull().sum().sum():,}",
            delta=f"{df_processed.isnull().sum().sum() - df_original.isnull().sum().sum():,}",
        )
        st.metric(
            "Memory (MB)",
            f"{df_processed.memory_usage(deep=True).sum() / 1024**2:.1f}",
            delta=f"{(df_processed.memory_usage(deep=True).sum() - df_original.memory_usage(deep=True).sum()) / 1024**2:.1f}",
        )


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
        help="Choose the numeric column you want to predict (regression target)",
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
        summary.append(
            {
                "Model": name,
                "R² Score": round(info["mean_score"], 4),
                "CV Std": round(info["std_score"], 4),
                "Is Best": name == best_model_name,
            }
        )

    return pd.DataFrame(summary).sort_values("R² Score", ascending=False)


def render_regression_metric_cards(metrics):
    """Render a compact metric summary for a regression model."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
    with col2:
        st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
    with col3:
        st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
    with col4:
        st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")


def render_learning_curve_chart(learning_curve_data, model_name):
    """Render the RMSE learning curve for the selected model."""
    if not learning_curve_data:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=learning_curve_data["train_sizes"],
            y=learning_curve_data["train_rmse"],
            mode="lines+markers",
            name="Training RMSE",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=learning_curve_data["train_sizes"],
            y=learning_curve_data["validation_rmse"],
            mode="lines+markers",
            name="Validation RMSE",
        )
    )
    fig.update_layout(
        title=f"Learning Curve - {model_name}",
        xaxis_title="Training Set Size",
        yaxis_title="RMSE",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_validation_curve_chart(validation_curve_data, model_name):
    """Render the validation curve for the selected model."""
    if not validation_curve_data:
        return

    x_axis_name = validation_curve_data["param_name"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=validation_curve_data["param_values"],
            y=validation_curve_data["train_rmse"],
            mode="lines+markers",
            name="Training RMSE",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=validation_curve_data["param_values"],
            y=validation_curve_data["validation_rmse"],
            mode="lines+markers",
            name="Validation RMSE",
        )
    )
    fig.update_layout(
        title=f"Validation Curve - {model_name}",
        xaxis_title=x_axis_name,
        yaxis_title="RMSE",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_iteration_history_chart(iteration_history, model_name):
    """Render the boosting training history for the selected model."""
    if not iteration_history:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iteration_history["iterations"],
            y=iteration_history["train_rmse"],
            mode="lines",
            name="Training RMSE",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iteration_history["iterations"],
            y=iteration_history["validation_rmse"],
            mode="lines",
            name="Validation RMSE",
        )
    )
    fig.update_layout(
        title=f"Iteration History - {model_name}",
        xaxis_title="Number of Boosting Iterations",
        yaxis_title="RMSE",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def display_regression_results():
    """Display results specifically for regression"""
    if hasattr(st.session_state, "ml_results") and st.session_state.ml_results is not None:
        st.markdown("---")
        st.subheader("📊 Regression Results")

        results = st.session_state.ml_results

        # Quick summary at the top
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏆 Best Model", results["best_model_name"])
        with col2:
            st.metric("📊 Problem Type", "REGRESSION")
        with col3:
            cv_score = results["best_score"]
            st.metric("📈 R² Score", f"{cv_score:.4f}")
        with col4:
            feature_count = (
                len(st.session_state.ml_models.feature_names) if st.session_state.ml_models.feature_names else 0
            )
            st.metric("🎯 Features Used", feature_count)

        # Results tabs for regression
        result_tabs = st.tabs(
            [
                "🏆 Best Model",
                "📈 Model Comparison",
                "🎯 Feature Importance",
                "📊 Prediction Analysis",
                "📋 Training Summary",
                "🔮 Make Predictions",
            ]
        )

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
            if "test_metrics" in results:
                st.write("### 🎯 Test Set Performance")

                metrics = results["test_metrics"]

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                with col3:
                    st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                with col4:
                    mape = metrics.get("mape", 0)
                    st.metric("MAPE", f"{mape:.2f}%")

        with result_tabs[1]:
            st.write("### 📈 Regression Model Performance Comparison")

            if "model_scores" in results:
                summary_df = create_model_summary_dataframe(results["model_scores"], results["best_model_name"])

                # Style the dataframe for regression
                styled_df = summary_df.style.highlight_max(subset=["R² Score"], color="lightgreen")
                st.dataframe(styled_df, use_container_width=True)

                # Performance comparison chart for regression
                fig = px.bar(
                    summary_df,
                    x="Model",
                    y="R² Score",
                    color="Is Best",
                    title=f"Regression Model R² Score Comparison ({selected_dataset})",
                    text="R² Score",
                    color_discrete_map={True: "#1f77b4", False: "#d62728"},
                )

                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=500,
                    showlegend=True,
                    legend=dict(title="Best Model"),
                    yaxis_title="R² Score",
                )
                st.plotly_chart(fig, use_container_width=True)

        with result_tabs[2]:
            st.write("### 🎯 Feature Importance Analysis")

            if "feature_importance" in results and results["feature_importance"] is not None:
                importance_df = results["feature_importance"]

                # Show top features
                st.write("**Top 15 Most Important Features for Regression:**")
                top_features = importance_df.head(15)
                st.dataframe(top_features, use_container_width=True)

                # Feature importance plot
                fig = px.bar(
                    top_features,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Feature Importance for Regression",
                    labels={"importance": "Importance Score", "feature": "Features"},
                    color="importance",
                    color_continuous_scale="viridis",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
                st.plotly_chart(fig, use_container_width=True)

        with result_tabs[3]:
            st.write("### 📊 Prediction Analysis")

            if "test_metrics" in results and "predictions" in results["test_metrics"]:
                predictions = results["test_metrics"]["predictions"]
                actual = st.session_state.ml_models.y_test

                # Prediction vs Actual scatter plot
                fig_scatter = px.scatter(
                    x=actual,
                    y=predictions,
                    labels={"x": "Actual Values", "y": "Predicted Values"},
                    title="Predicted vs Actual Values",
                )

                # Add perfect prediction line
                min_val = min(min(actual), min(predictions))
                max_val = max(max(actual), max(predictions))
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="Perfect Prediction",
                        line=dict(color="red", dash="dash"),
                    )
                )

                st.plotly_chart(fig_scatter, use_container_width=True)

                # Residuals plot
                residuals = results["test_metrics"]["residuals"]
                fig_residuals = px.scatter(
                    x=predictions,
                    y=residuals,
                    labels={"x": "Predicted Values", "y": "Residuals"},
                    title="Residuals Plot",
                )
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residuals, use_container_width=True)

        with result_tabs[4]:
            st.write("### 📋 Training Summary")

            if "training_summary" in results:
                summary = results["training_summary"]

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

                    if summary.get("removed_features"):
                        st.write(f"- Removed Features: {len(summary['removed_features'])}")
                        with st.expander("Show removed features"):
                            st.write(summary["removed_features"])

        with result_tabs[5]:
            st.write("### 🔮 Make Predictions on New Data")

            st.info("📁 Upload a dataset without the target column to make predictions")

            # File uploader for prediction data
            prediction_file = st.file_uploader(
                "Choose a CSV or Excel file for predictions:",
                type=["csv", "xlsx", "xlsm"],
                key="prediction_data",
                help="Upload data without the target column to get predictions",
            )

            if prediction_file is not None:
                try:
                    # Load prediction data
                    if prediction_file.name.endswith(".csv"):
                        pred_df = pd.read_csv(prediction_file)
                    else:
                        pred_df = pd.read_excel(prediction_file, engine="openpyxl")

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
                                    pred_df_with_predictions["Predicted_" + target_col] = predictions

                                    st.success("✅ Predictions generated successfully!")

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
                                        mime="text/csv",
                                    )

                                    # Prediction distribution plot
                                    fig_pred = px.histogram(
                                        x=predictions,
                                        title="Distribution of Predictions",
                                        labels={"x": "Predicted Values", "y": "Frequency"},
                                        marginal="box",
                                    )
                                    st.plotly_chart(fig_pred, use_container_width=True)

                                else:
                                    st.error("❌ Failed to generate predictions. Please check your data.")

                            except Exception as e:
                                st.error(f"❌ Error making predictions: {e!s}")
                                st.exception(e)

                except Exception as e:
                    st.error(f"❌ Error loading prediction data: {e!s}")

            else:
                st.info("👆 Please upload a dataset to make predictions")


# Initialize session state
initialize_session_state()

# Main header
st.markdown(
    """
<div class="main-header">
    <h1>Data Preprocessing and Advanced Analysis Platform</h1>
    <p style="margin: 0; font-size: 1.1em;">Professional Data Processing & Analytics Platform</p>
    <p style="margin: 0; font-size: 0.9em; opacity: 0.8;">Transform • Analyze • Visualize • Predict</p>
</div>
""",
    unsafe_allow_html=True,
)

# Professional sidebar navigation with buttons
st.sidebar.markdown("### 🧭 Navigation Center")
st.sidebar.markdown("---")

# Navigation buttons
if st.sidebar.button("🔧 Data Processing Pipeline", use_container_width=True):
    st.session_state.current_page = "data_pipeline"

if st.sidebar.button("📊 Data Visualization Studio", use_container_width=True):
    st.session_state.current_page = "visualization"

if st.sidebar.button("🔬 Exploratory Analytics", use_container_width=True):
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
    datasets_available.append(
        f"✅ Cleaned: {st.session_state.df_cleaned.shape[0]:,} × {st.session_state.df_cleaned.shape[1]}"
    )
if st.session_state.df_grouped is not None:
    datasets_available.append(
        f"📊 Grouped: {st.session_state.df_grouped.shape[0]:,} × {st.session_state.df_grouped.shape[1]}"
    )
if st.session_state.df_merged is not None:
    datasets_available.append(
        f"🔗 Merged: {st.session_state.df_merged.shape[0]:,} × {st.session_state.df_merged.shape[1]}"
    )
if st.session_state.df_encoded is not None:
    datasets_available.append(
        f"🔤 Encoded: {st.session_state.df_encoded.shape[0]:,} × {st.session_state.df_encoded.shape[1]}"
    )

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
if (
    st.session_state.df_cleaned is not None
    or st.session_state.df_grouped is not None
    or st.session_state.df_merged is not None
    or st.session_state.df_encoded is not None
):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Quick Actions")

    if st.sidebar.button("🗑️ Reset All Data", use_container_width=True):
        for key in [
            "df_original",
            "df_cleaned",
            "df_second",
            "df_grouped",
            "df_merged",
            "df_encoded",
            "active_dataset_key",
        ]:
            st.session_state[key] = None
        st.session_state.uploaded_dataframes = {}
        st.session_state.uploaded_dataset_order = []
        st.session_state.processed_data = {}
        for key in ["cleaning_applied", "grouping_applied", "merging_applied", "encoding_applied"]:
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

    uploaded_files = st.file_uploader(
        "Choose one or more CSV or Excel files",
        type=["csv", "xlsx", "xlsm"],
        accept_multiple_files=True,
        help="Upload one or more datasets to get started. Multi-sheet Excel files are supported.",
    )

    sheet_preferences = {}
    if uploaded_files:
        if has_multi_sheet_uploads(uploaded_files):
            st.info("Select which Excel sheets to load before starting the batch import.")
        sheet_preferences = build_sheet_preferences(uploaded_files)

        if st.button("Load Selected Files", type="primary", use_container_width=True):
            try:
                loaded_dataframes, dataset_order = load_uploaded_files(uploaded_files, sheet_preferences)
                st.session_state.uploaded_dataframes = loaded_dataframes
                st.session_state.uploaded_dataset_order = dataset_order
                st.session_state.processed_data = {}
                st.session_state.df_cleaned = None
                st.session_state.df_grouped = None
                st.session_state.df_merged = None
                st.session_state.df_encoded = None
                st.session_state.cleaning_applied = False
                st.session_state.grouping_applied = False
                st.session_state.merging_applied = False
                st.session_state.encoding_applied = False

                if dataset_order:
                    if st.session_state.active_dataset_key not in loaded_dataframes:
                        st.session_state.active_dataset_key = dataset_order[0]
                    sync_active_dataset()
                    sync_processed_dataset_views()
                    st.success(f"Loaded {len(dataset_order)} dataset(s) from {len(uploaded_files)} uploaded file(s).")
                else:
                    st.warning("No datasets were loaded from the selected files.")
            except Exception as e:
                error_msg = f"Error loading uploaded files: {e!s}"
                add_log(error_msg, "ERROR")
                st.error(error_msg)

    uploaded_datasets = st.session_state.get("uploaded_dataframes", {})
    uploaded_dataset_order = st.session_state.get("uploaded_dataset_order", [])

    if uploaded_datasets:
        st.subheader("Loaded Datasets")
        selected_active_dataset = st.selectbox(
            "Choose the active dataset for the preprocessing pipeline:",
            options=uploaded_dataset_order,
            index=(
                uploaded_dataset_order.index(st.session_state.active_dataset_key)
                if st.session_state.active_dataset_key in uploaded_dataset_order
                else 0
            ),
            key="active_dataset_selector",
        )

        if selected_active_dataset != st.session_state.active_dataset_key:
            st.session_state.active_dataset_key = selected_active_dataset
        sync_active_dataset()
        sync_processed_dataset_views()

        st.caption(
            "The active dataset is copied into st.session_state.df_original so the existing cleaning "
            "and transformation steps continue to work."
        )

        for dataset_key in uploaded_dataset_order:
            render_dataset_preview(dataset_key, uploaded_datasets[dataset_key])

    st.markdown("---")

    # ===========================================
    # SECTION 2: DATA CLEANING
    # ===========================================
    st.header("🧹 Data Cleaning")

    if len(uploaded_datasets) > 1:
        st.subheader("Batch Processing Mode")
        st.session_state.processing_mode = st.radio(
            "How should we apply batch settings?",
            options=["Apply to All", "Configure Individually"],
            horizontal=True,
        )

        if st.session_state.processing_mode == "Apply to All":
            reference_key = st.session_state.active_dataset_key or uploaded_dataset_order[0]
            reference_df = uploaded_datasets[reference_key]
            global_config = render_cleaning_controls(reference_df, "global_cleaning")

            if st.button("Run Batch Cleaning", type="primary", key="run_global_batch_cleaning"):
                config_map = {dataset_key: global_config for dataset_key in uploaded_dataset_order}
                try:
                    st.session_state.processed_data = process_cleaning_batch(config_map)
                    sync_processed_dataset_views()
                    st.success("Batch cleaning completed with global settings.")
                except Exception as e:
                    error_msg = f"Batch cleaning failed: {e!s}"
                    add_log(error_msg, "ERROR")
                    st.error(error_msg)

        else:
            individual_configs = {}
            for dataset_key in uploaded_dataset_order:
                dataset_df = uploaded_datasets[dataset_key]
                with st.expander(dataset_key, expanded=False):
                    individual_configs[dataset_key] = render_cleaning_controls(
                        dataset_df,
                        f"individual_cleaning_{dataset_key}",
                    )

            if st.button("Run Individual Batch Cleaning", type="primary", key="run_individual_batch_cleaning"):
                try:
                    st.session_state.processed_data = process_cleaning_batch(individual_configs)
                    sync_processed_dataset_views()
                    st.success("Batch cleaning completed with individual dataset settings.")
                except Exception as e:
                    error_msg = f"Individual batch cleaning failed: {e!s}"
                    add_log(error_msg, "ERROR")
                    st.error(error_msg)

        if st.session_state.processed_data:
            render_processed_results(st.session_state.processed_data)
            st.markdown("**Post-Cleaning Zero-Only Row Removal**")

            cleaned_dataset_keys = [
                dataset_key
                for dataset_key in uploaded_dataset_order
                if st.session_state.processed_data.get(dataset_key, {}).get("cleaned") is not None
            ]

            if cleaned_dataset_keys:
                if st.session_state.processing_mode == "Apply to All":
                    if st.button(
                        "Apply Zero-Only Row Removal to Cleaned Datasets", key="apply_global_zero_only_row_removal"
                    ):
                        removal_messages = []
                        for dataset_key in cleaned_dataset_keys:
                            removed_count = apply_post_cleaning_zero_only_row_removal(dataset_key)
                            removal_messages.append(f"{dataset_key}: {removed_count}")
                        sync_processed_dataset_views()
                        st.success(
                            "Zero-only row removal applied to all cleaned datasets. " + " | ".join(removal_messages)
                        )
                    st.caption(
                        "This action checks only numeric columns in the cleaned dataset and ignores datetime/date/time and categorical columns."
                    )
                else:
                    for dataset_key in cleaned_dataset_keys:
                        with st.expander(f"Remove zero-only rows: {dataset_key}", expanded=False):
                            if st.button("Apply Zero-Only Row Removal", key=f"apply_zero_only_rows_{dataset_key}"):
                                removed_count = apply_post_cleaning_zero_only_row_removal(dataset_key)
                                sync_processed_dataset_views()
                                st.success(
                                    f"Updated cleaned dataset: {dataset_key}. Removed {removed_count} zero-only rows."
                                )
                            st.caption(
                                "Checks only numeric columns and ignores datetime/date/time and categorical columns."
                            )

            st.markdown("**Post-Cleaning Column Deletion**")

            if cleaned_dataset_keys:
                if st.session_state.processing_mode == "Apply to All":
                    common_cleaned_columns = get_cleaned_dataset_columns(st.session_state.processed_data)
                    global_drop_columns = st.multiselect(
                        "Delete columns from all cleaned datasets:",
                        options=common_cleaned_columns,
                        default=[],
                        key="global_post_cleaning_drop_columns",
                    )
                    if st.button("Apply Column Deletion to Cleaned Datasets", key="apply_global_cleaned_drop"):
                        for dataset_key in cleaned_dataset_keys:
                            apply_post_cleaning_column_drop(dataset_key, global_drop_columns)
                        sync_processed_dataset_views()
                        st.success("Column deletion applied to all cleaned datasets.")
                else:
                    for dataset_key in cleaned_dataset_keys:
                        cleaned_df = st.session_state.processed_data[dataset_key]["cleaned"]
                        with st.expander(f"Delete columns: {dataset_key}", expanded=False):
                            dataset_drop_columns = st.multiselect(
                                f"Columns to delete from {dataset_key}",
                                options=cleaned_df.columns.tolist(),
                                default=[],
                                key=f"post_clean_drop_{dataset_key}",
                            )
                            if st.button("Apply Column Deletion", key=f"apply_post_clean_drop_{dataset_key}"):
                                apply_post_cleaning_column_drop(dataset_key, dataset_drop_columns)
                                sync_processed_dataset_views()
                                st.success(f"Updated cleaned dataset: {dataset_key}")

    elif st.session_state.df_original is None:
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
                        error_msg = f"Error during quality assessment: {e!s}"
                        add_log(error_msg, "ERROR")
                        st.error(error_msg)

        # Display quality report if available
        if hasattr(st.session_state, "quality_report"):
            quality_report = st.session_state.quality_report

            with col2:
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.write("**Dataset Shape:**", quality_report["shape"])
                    st.write("**Memory Usage:**", f"{quality_report['memory_usage'] / (1024**2):.2f} MB")
                    st.write("**Duplicates:**", quality_report["duplicates"])

                with subcol2:
                    st.write("**Numeric Columns:**", len(quality_report["numeric_columns"]))
                    st.write("**Categorical Columns:**", len(quality_report["categorical_columns"]))

            # Missing values breakdown
            if quality_report["missing_values"].sum() > 0:
                with st.expander("Missing Values Details", expanded=True):
                    missing_df = pd.DataFrame(
                        {
                            "Column": quality_report["missing_values"].index,
                            "Missing Count": quality_report["missing_values"].values,
                            "Missing %": (quality_report["missing_values"].values / df.shape[0] * 100).round(2),
                        }
                    )
                    missing_df = missing_df[missing_df["Missing Count"] > 0]
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
                help="Choose the strategy for handling missing values",
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
                    help="Time interval in hours for data interpolation (e.g., 1.0 for hourly, 0.5 for 30 minutes)",
                )
                st.info(f"Selected frequency: {frequency_hours} hour(s)")

            if st.button("🧹 Clean Data", type="primary"):
                with st.spinner("Cleaning data..."):
                    add_log(
                        f"Starting data cleaning with strategy: {data_type}"
                        + (f" (frequency: {frequency_hours}h)" if data_type == "frequence" else "")
                    )

                    try:
                        initial_shape = df.shape
                        # Pass the frequency_hours parameter to clean_data
                        cleaned_df = st.session_state.cleaner.clean_data(df, data_type, frequency_hours)

                        # Store cleaned data and drop original
                        st.session_state.df_cleaned = cleaned_df
                        st.session_state.df_original = None  # Drop original as requested
                        st.session_state.cleaning_applied = True
                        st.session_state.cleaning_results = {
                            "initial_shape": initial_shape,
                            "final_shape": cleaned_df.shape,
                            "strategy": data_type,
                            "frequency": frequency_hours if data_type == "frequence" else None,
                        }

                        success_msg = f"✅ Data cleaned successfully using {data_type} strategy! Original data dropped."
                        if data_type == "frequence":
                            success_msg += f" (Interpolation frequency: {frequency_hours}h)"

                        add_log(f"Data cleaning completed: {initial_shape} → {cleaned_df.shape}")
                        st.success(success_msg)

                    except Exception as e:
                        error_msg = f"Error during cleaning: {e!s}"
                        add_log(error_msg, "ERROR")
                        st.error(error_msg)

        # Show cleaning results if available
        if hasattr(st.session_state, "cleaning_results"):
            results = st.session_state.cleaning_results

            with col2:
                st.subheader("Cleaning Results")
                st.write(f"**Strategy:** {results['strategy'].title()}")
                if results["frequency"]:
                    st.write(f"**Interpolation Frequency:** {results['frequency']} hour(s)")

                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("Original Rows", results["initial_shape"][0])
                    st.metric("Original Columns", results["initial_shape"][1])
                with subcol2:
                    st.metric(
                        "Cleaned Rows",
                        results["final_shape"][0],
                        delta=results["final_shape"][0] - results["initial_shape"][0],
                    )
                    st.metric(
                        "Cleaned Columns",
                        results["final_shape"][1],
                        delta=results["final_shape"][1] - results["initial_shape"][1],
                    )

                # Show cleaned data preview
                if st.session_state.df_cleaned is not None:
                    columns_to_drop_after_cleaning = st.multiselect(
                        "Delete columns from the cleaned dataset:",
                        options=st.session_state.df_cleaned.columns.tolist(),
                        default=[],
                        key="single_post_cleaning_drop_columns",
                    )
                    if st.button("Apply Column Deletion to Cleaned Dataset", key="apply_single_post_cleaning_drop"):
                        st.session_state.df_cleaned = drop_selected_columns(
                            st.session_state.df_cleaned,
                            columns_to_drop_after_cleaning,
                        )
                        st.session_state.cleaning_results["final_shape"] = st.session_state.df_cleaned.shape
                        st.session_state.cleaning_results["dropped_columns"] = columns_to_drop_after_cleaning
                        st.success("Column deletion applied to the cleaned dataset.")

                    if results.get("dropped_columns"):
                        st.write(f"**Dropped Columns:** {results['dropped_columns']}")

                    with st.expander("Cleaned Data Preview", expanded=False):
                        st.dataframe(st.session_state.df_cleaned.head(10), use_container_width=True)

    st.markdown("---")

    # ===========================================
    # SECTION 3: DATA TRANSFORMATION
    # ===========================================
    st.header("🔄 Data Transformation")

    if len(uploaded_datasets) > 1:
        st.subheader("Categorical Encoding")
        if st.session_state.processing_mode == "Apply to All":
            common_columns = get_common_columns(uploaded_datasets)
            common_numeric_columns = get_common_numeric_columns(uploaded_datasets)
            reference_key = st.session_state.active_dataset_key or uploaded_dataset_order[0]
            reference_df = get_stage_input_dataframe(reference_key, "encoding")
            if common_columns:
                st.caption("Global encoding uses only columns shared across all datasets.")
            encoding_config = render_encoding_controls(
                reference_df,
                "global_encoding",
                selectable_columns=common_columns or reference_df.columns.tolist(),
                selectable_numeric_columns=common_numeric_columns,
            )
            if st.button("Run Batch Encoding", type="primary", key="run_global_batch_encoding"):
                config_map = {dataset_key: encoding_config for dataset_key in uploaded_dataset_order}
                try:
                    st.session_state.processed_data = process_encoding_batch(config_map)
                    sync_processed_dataset_views()
                    st.success("Batch encoding completed with global settings.")
                except Exception as e:
                    error_msg = f"Batch encoding failed: {e!s}"
                    add_log(error_msg, "ERROR")
                    st.error(error_msg)
        else:
            individual_encoding_configs = {}
            for dataset_key in uploaded_dataset_order:
                dataset_df = get_stage_input_dataframe(dataset_key, "encoding")
                with st.expander(f"Encoding: {dataset_key}", expanded=False):
                    individual_encoding_configs[dataset_key] = render_encoding_controls(
                        dataset_df,
                        f"individual_encoding_{dataset_key}",
                    )
            if st.button("Run Individual Batch Encoding", type="primary", key="run_individual_batch_encoding"):
                try:
                    st.session_state.processed_data = process_encoding_batch(individual_encoding_configs)
                    sync_processed_dataset_views()
                    st.success("Batch encoding completed with individual dataset settings.")
                except Exception as e:
                    error_msg = f"Individual batch encoding failed: {e!s}"
                    add_log(error_msg, "ERROR")
                    st.error(error_msg)

        st.markdown("**---**")
        st.subheader("Data Grouping")
        if st.session_state.processing_mode == "Apply to All":
            common_columns = get_common_columns(uploaded_datasets)
            reference_key = st.session_state.active_dataset_key or uploaded_dataset_order[0]
            reference_df = get_stage_input_dataframe(reference_key, "grouping")
            if common_columns:
                st.caption("Global grouping uses only columns shared across all datasets.")
            grouping_config = render_grouping_controls(
                reference_df,
                "global_grouping",
                selectable_columns=common_columns or reference_df.columns.tolist(),
            )
            if st.button("Run Batch Grouping", type="primary", key="run_global_batch_grouping"):
                config_map = {dataset_key: grouping_config for dataset_key in uploaded_dataset_order}
                try:
                    st.session_state.processed_data = process_grouping_batch(config_map)
                    sync_processed_dataset_views()
                    st.success("Batch grouping completed with global settings.")
                except Exception as e:
                    error_msg = f"Batch grouping failed: {e!s}"
                    add_log(error_msg, "ERROR")
                    st.error(error_msg)
        else:
            individual_grouping_configs = {}
            for dataset_key in uploaded_dataset_order:
                dataset_df = get_stage_input_dataframe(dataset_key, "grouping")
                with st.expander(f"Grouping: {dataset_key}", expanded=False):
                    individual_grouping_configs[dataset_key] = render_grouping_controls(
                        dataset_df,
                        f"individual_grouping_{dataset_key}",
                    )
            if st.button("Run Individual Batch Grouping", type="primary", key="run_individual_batch_grouping"):
                try:
                    st.session_state.processed_data = process_grouping_batch(individual_grouping_configs)
                    sync_processed_dataset_views()
                    st.success("Batch grouping completed with individual dataset settings.")
                except Exception as e:
                    error_msg = f"Individual batch grouping failed: {e!s}"
                    add_log(error_msg, "ERROR")
                    st.error(error_msg)

        if st.session_state.processed_data:
            render_processed_results(st.session_state.processed_data)
    elif st.session_state.df_cleaned is None:
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
                options=["None"] + current_df.columns.tolist(),
                help="Select target column for advanced encoding methods",
            )

            target_column = None if target_col == "None" else target_col

            if st.button("🔤 Encode Categorical Data", type="primary", key="categorical_encode"):
                with st.spinner("Encoding categorical data..."):
                    try:
                        add_log("Starting categorical encoding on cleaned data")

                        # Use the smart encoding function from transformer
                        filtered_df = filter_zero_only_numeric_rows(current_df)
                        pred_df_raw = st.session_state.transformer.smart_categorical_encoding(
                            filtered_df.copy(), target_col=target_column
                        )

                        st.session_state.target_encoder_bytes = None
                        st.session_state.target_encoder_columns = []

                        if target_column is not None:
                            fitted_target_encoder, encoded_columns = st.session_state.transformer.fit_target_encoder(
                                filtered_df.copy(),
                                target_col=target_column,
                            )
                            if fitted_target_encoder is not None:
                                encoder_buffer = BytesIO()
                                joblib.dump(fitted_target_encoder, encoder_buffer)
                                st.session_state.target_encoder_bytes = encoder_buffer.getvalue()
                                st.session_state.target_encoder_columns = encoded_columns
                                st.session_state.loaded_target_encoder = load_cached_target_encoder(
                                    st.session_state.target_encoder_bytes
                                )

                        # Store as independent encoded dataframe
                        st.session_state.df_encoded = pred_df_raw
                        st.session_state.encoding_applied = True
                        st.session_state.encoding_results = {
                            "original_shape": current_df.shape,
                            "encoded_shape": pred_df_raw.shape,
                        }

                        st.success("✅ Categorical encoding completed!")
                        add_log(f"Encoding completed: {current_df.shape} → {pred_df_raw.shape}")

                        # Show conversion summary
                        numeric_cols = len(pred_df_raw.select_dtypes(include=[np.number]).columns)
                        total_cols = len(pred_df_raw.columns)
                        st.info(f"📊 Result: {numeric_cols}/{total_cols} columns are now numeric")

                    except Exception as e:
                        error_msg = f"Error encoding data: {e!s}"
                        add_log(error_msg, "ERROR")
                        st.error(error_msg)

        # Show encoding results
        if hasattr(st.session_state, "encoding_results"):
            results = st.session_state.encoding_results
            with col2:
                st.subheader("Encoding Results")
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("Original columns", results["original_shape"][1])
                    st.metric("Original rows", results["original_shape"][0])
                with subcol2:
                    st.metric(
                        "Encoded columns",
                        results["encoded_shape"][1],
                        delta=results["encoded_shape"][1] - results["original_shape"][1],
                    )
                    st.metric(
                        "Encoded rows",
                        results["encoded_shape"][0],
                        delta=results["encoded_shape"][0] - results["original_shape"][0],
                    )

                if st.session_state.df_encoded is not None:
                    with st.expander("Encoded Data Preview", expanded=False):
                        st.dataframe(st.session_state.df_encoded.head(10), use_container_width=True)
                    if st.session_state.get("target_encoder_bytes"):
                        encoded_columns = st.session_state.get("target_encoder_columns", [])
                        if encoded_columns:
                            st.caption(f"Saved TargetEncoder columns: {', '.join(encoded_columns)}")
                        st.download_button(
                            label="📥 Download Fitted TargetEncoder",
                            data=st.session_state.target_encoder_bytes,
                            file_name="target_encoder.pkl",
                            mime="application/octet-stream",
                            key="download_target_encoder_artifact",
                        )

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
                key="group_cols",
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
                                "original_shape": current_df.shape,
                                "grouped_shape": grouped_df.shape,
                                "group_cols": group_cols,
                            }

                            st.success("✅ Data grouped successfully!")
                            add_log(f"Grouping completed: {current_df.shape} → {grouped_df.shape}")

                        except Exception as e:
                            error_msg = f"Error grouping data: {e!s}"
                            add_log(error_msg, "ERROR")
                            st.error(error_msg)
            else:
                st.info("Select columns above to enable grouping")

        # Show grouping results
        if hasattr(st.session_state, "grouping_results"):
            results = st.session_state.grouping_results
            with col2:
                st.subheader("Grouping Results")
                st.write(f"**Grouped by:** {results['group_cols']}")

                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("Original rows", results["original_shape"][0])
                with subcol2:
                    st.metric("Grouped rows", results["grouped_shape"][0])

                if st.session_state.df_grouped is not None:
                    # Show sample group sizes
                    sample_groups = current_df.groupby(results["group_cols"]).size().head(5)
                    st.write("**Sample group sizes:**")
                    st.dataframe(sample_groups.to_frame("count"))

                    with st.expander("Grouped Data Preview", expanded=False):
                        st.dataframe(st.session_state.df_grouped.head(10), use_container_width=True)

        st.markdown("**---**")

        # ===== MERGE/CONCATENATE =====
        st.subheader("🔗 Merge or Concatenate DataFrames")

        col1, col2 = st.columns([1, 2])

        with col1:
            # File uploader for second dataset
            second_file = st.file_uploader(
                "Upload second dataset:", type=["csv", "xlsx", "xlsm"], key="second_dataset_pipeline"
            )

            if second_file is not None:
                # Load second dataset
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{second_file.name.split('.')[-1]}"
                ) as tmp_file:
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
                            merge_method = st.selectbox(
                                "Merge method:", ["inner", "outer", "left", "right"], key="merge_method"
                            )

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
                                            "df1_shape": current_df.shape,
                                            "df2_shape": df2.shape,
                                            "merged_shape": merged_df.shape,
                                            "operation": f"Merge ({merge_method})",
                                        }

                                        st.success("✅ DataFrames merged successfully!")
                                        add_log(
                                            f"Merge completed: {current_df.shape} + {df2.shape} → {merged_df.shape}"
                                        )

                                    except Exception as e:
                                        error_msg = f"Error merging dataframes: {e!s}"
                                        add_log(error_msg, "ERROR")
                                        st.error(error_msg)
                        else:
                            st.warning("No common columns found for merging")

                    else:  # Concatenate
                        axis = st.selectbox(
                            "Concatenation axis:",
                            [0, 1],
                            format_func=lambda x: "Rows (0)" if x == 0 else "Columns (1)",
                            key="concat_axis",
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
                                        "df1_shape": current_df.shape,
                                        "df2_shape": df2.shape,
                                        "merged_shape": concat_df.shape,
                                        "operation": f"Concatenate (axis={axis})",
                                    }

                                    st.success("✅ DataFrames concatenated successfully!")
                                    add_log(
                                        f"Concatenation completed: {current_df.shape} + {df2.shape} → {concat_df.shape}"
                                    )

                                except Exception as e:
                                    error_msg = f"Error concatenating dataframes: {e!s}"
                                    add_log(error_msg, "ERROR")
                                    st.error(error_msg)

                except Exception as e:
                    error_msg = f"Error loading second dataset: {e!s}"
                    add_log(error_msg, "ERROR")
                    st.error(error_msg)
                finally:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

        # Show merge/concatenate results
        if hasattr(st.session_state, "merge_results"):
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
                    use_container_width=True,
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
                label="📥 Export Logs", data=logs_text, file_name="processing_logs.txt", mime="text/plain"
            )

        with col3:
            st.info(f"Total logs: {len(st.session_state.processing_logs)}")

        st.markdown("---")

        # Filter options
        filter_type = st.selectbox("Filter logs by type:", ["All", "Errors", "Warnings", "Info"], key="log_filter")

        # Display logs based on filter
        filtered_logs = []
        if filter_type == "All":
            filtered_logs = st.session_state.processing_logs
        elif filter_type == "Errors":
            filtered_logs = [log for log in st.session_state.processing_logs if "ERROR" in log]
        elif filter_type == "Warnings":
            filtered_logs = [log for log in st.session_state.processing_logs if "WARNING" in log]
        else:  # Info
            filtered_logs = [
                log for log in st.session_state.processing_logs if "ERROR" not in log and "WARNING" not in log
            ]

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
            help="Choose which processed dataset to visualize",
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
                        "figure": fig,
                        "plot_type": st.session_state.get("main_viz_plot_type", "Unknown"),
                        "title": fig.layout.title.text if fig.layout.title.text else "Untitled Plot",
                        "x_col": st.session_state.get("main_viz_x_col"),
                        "y_col": st.session_state.get("main_viz_y_col"),
                        "data_shape": current_df.shape,
                        "dataset_type": selected_dataset,
                    }

                    # Only add to history if it's a new plot (not just updating)
                    if st.button("💾 Save Plot to Gallery"):
                        save_plot_to_history(plot_info)
                        st.success("✅ Plot saved to gallery!")

                    # Render insights
                    x_col = st.session_state.get("main_viz_x_col")
                    y_col = st.session_state.get("main_viz_y_col")
                    if x_col:
                        st.session_state.dashboard.render_plot_insights(current_df, x_col, y_col)

            except Exception as e:
                st.error(f"Error creating visualization: {e!s}")

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
                                "figure": fig,
                                "plot_type": f"Comparison {i}",
                                "title": f"Comparison Plot {i} ({selected_dataset})",
                                "data_shape": current_df.shape,
                                "dataset_type": selected_dataset,
                            }
                            save_plot_to_history(plot_info)

                        st.success("✅ Comparison plots saved to gallery!")

            except Exception as e:
                st.error(f"Error creating comparison: {e!s}")

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
    st.header("🔬 Exploratory Analytics")
    st.markdown("Statistical, correlation and distribution analysis and data quality tools")
    st.markdown("---")

    # Lazy imports — only loaded when user visits this page
    from src.analytics.advanced_analytics import AdvancedAnalytics
    from src.analytics.statistical import StatisticalAnalyzer

    # Initialize analyzers
    if "stat_analyzer" not in st.session_state:
        st.session_state.stat_analyzer = StatisticalAnalyzer()
    if "adv_analytics" not in st.session_state:
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
            "🔬 Select dataset for analysis:", available_datasets, help="Choose which processed dataset to analyze"
        )

        current_df = dataset_options[selected_dataset]
        st.info(f"Analyzing {selected_dataset}: {current_df.shape[0]:,} rows × {current_df.shape[1]} columns")

        # Expanded Analytics tabs
        analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab5, analytics_tab6 = st.tabs(
            [
                "📊 Descriptive Stats",
                "🔗 Correlation Analysis",
                "📈 Distribution Analysis",
                "🎯 Outlier Detection",
                "📋 Data Quality Report",
            ]
        )

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
                    if "extended_statistics" in ext_stats:
                        st.dataframe(ext_stats["extended_statistics"], use_container_width=True)
            else:
                st.info("No numerical columns found")

            # Categorical statistics
            st.write("**Categorical Columns Statistics:**")
            categorical_df = current_df.select_dtypes(include=["object", "category"])
            if not categorical_df.empty:
                cat_stats = []
                for col in categorical_df.columns:
                    cat_stats.append(
                        {
                            "Column": col,
                            "Unique Values": current_df[col].nunique(),
                            "Most Frequent": (
                                current_df[col].mode().iloc[0] if not current_df[col].mode().empty else "N/A"
                            ),
                            "Missing Values": current_df[col].isnull().sum(),
                        }
                    )
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
                    color_continuous_scale="RdBu_r",
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

                # High correlation pairs
                st.subheader("High Correlation Pairs (|r| > 0.7)")
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append(
                                {
                                    "Variable 1": corr_matrix.columns[i],
                                    "Variable 2": corr_matrix.columns[j],
                                    "Correlation": round(corr_val, 3),
                                }
                            )

                if high_corr_pairs:
                    st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
                else:
                    st.info("No high correlation pairs found")

                # Multicollinearity detection
                with st.expander("🔍 Multicollinearity Analysis"):
                    threshold = st.slider("Correlation threshold", 0.7, 0.99, 0.85, 0.05)
                    multi_result = st.session_state.adv_analytics.detect_multicollinearity(current_df, threshold)
                    if "error" not in multi_result:
                        st.write(f"**High correlation pairs found:** {multi_result['n_high_correlations']}")
                        st.write(f"**Recommendation:** {multi_result['recommendation']}")

                        # Show the high correlation pairs table
                        if multi_result["high_correlation_pairs"]:
                            pairs_df = pd.DataFrame(multi_result["high_correlation_pairs"])
                            st.dataframe(pairs_df, use_container_width=True)

                        if multi_result["potentially_redundant_features"]:
                            st.warning(
                                f"Potentially redundant features: {', '.join(multi_result['potentially_redundant_features'])}"
                            )
                    else:
                        st.error(multi_result["error"])
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
                            current_df, x=selected_col, title=f"Distribution of {selected_col}", marginal="box"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    with col2:
                        # Box plot
                        fig_box = px.box(current_df, y=selected_col, title=f"Box Plot of {selected_col}")
                        st.plotly_chart(fig_box, use_container_width=True)

                    # Skewness and Kurtosis
                    with st.expander("📊 Skewness & Kurtosis Analysis"):
                        sk_result = st.session_state.stat_analyzer.get_skewness_kurtosis(current_df[selected_col])
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Skewness", f"{sk_result['skewness']:.4f}")
                            st.caption(sk_result["skewness_interpretation"])
                        with col2:
                            st.metric("Kurtosis", f"{sk_result['kurtosis']:.4f}")
                            st.caption(sk_result["kurtosis_interpretation"])
            else:
                st.info("No numerical columns available for distribution analysis")

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

                if "error" not in result:
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Outliers", result["n_outliers"])
                    with col2:
                        st.metric("Outlier %", f"{result['outlier_percentage']:.2f}%")
                    with col3:
                        st.metric(
                            "Lower Bound" if method == "IQR Method" else "Mean",
                            f"{result.get('lower_bound', result.get('mean', 0)):.4f}",
                        )
                    with col4:
                        st.metric(
                            "Upper Bound" if method == "IQR Method" else "Std Dev",
                            f"{result.get('upper_bound', result.get('std', 0)):.4f}",
                        )

                    # Visualization
                    col1, col2 = st.columns(2)

                    with col1:
                        # Box plot with outliers highlighted
                        fig = px.box(current_df, y=outlier_col, title=f"Box Plot of {outlier_col}")
                        st.plotly_chart(fig, use_container_width=True, key="outlier_box_plot")

                    with col2:
                        # Histogram with bounds
                        fig = px.histogram(current_df, x=outlier_col, title="Distribution with Outlier Bounds")
                        if method == "IQR Method":
                            fig.add_vline(
                                x=result["lower_bound"],
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Lower Bound",
                            )
                            fig.add_vline(
                                x=result["upper_bound"],
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Upper Bound",
                            )
                        st.plotly_chart(fig, use_container_width=True, key="outlier_hist_plot")

                    # Outlier details
                    if result["n_outliers"] > 0:
                        with st.expander(f"📋 View {result['n_outliers']} Outlier Values"):
                            outlier_df = pd.DataFrame(
                                {
                                    "Index": result["outlier_indices"][:50],  # Limit display
                                    "Value": result["outlier_values"][:50],
                                }
                            )
                            st.dataframe(outlier_df, use_container_width=True)
                            if result["n_outliers"] > 50:
                                st.caption(f"Showing first 50 of {result['n_outliers']} outliers")
                else:
                    st.error(result["error"])
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
                    score = report["quality_score"]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Overall Score", f"{score['overall']}/100", delta=f"Grade: {score['grade']}")
                    with col2:
                        st.metric("Completeness", f"{score['completeness']:.1f}%")
                    with col3:
                        st.metric("Uniqueness", f"{score['uniqueness']:.1f}%")
                    with col4:
                        issues = report["issues_summary"]
                        st.metric("Columns with Issues", issues["columns_with_issues"])

                    st.markdown("---")

                    # Overview
                    st.subheader("📈 Dataset Overview")
                    overview = report["overview"]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", f"{overview['total_rows']:,}")
                    with col2:
                        st.metric("Total Columns", overview["total_columns"])
                    with col3:
                        st.metric("Missing Values", f"{overview['total_missing']:,}")
                    with col4:
                        st.metric("Memory Usage", f"{overview['memory_usage_mb']:.2f} MB")

                    # Column Types
                    col_types = report["column_types"]
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Numeric", col_types["numeric"])
                    with cols[1]:
                        st.metric("Categorical", col_types["categorical"])
                    with cols[2]:
                        st.metric("Datetime", col_types["datetime"])
                    with cols[3]:
                        st.metric("Boolean", col_types["boolean"])

                    st.markdown("---")

                    # Column Analysis
                    st.subheader("🔍 Column-Level Analysis")

                    col_analysis_df = pd.DataFrame(report["column_analysis"])
                    col_analysis_df["issues"] = col_analysis_df["issues"].apply(lambda x: ", ".join(x) if x else "None")

                    # Highlight issues
                    def highlight_issues(row):
                        if row["has_issues"]:
                            return ["background-color: #ffcccc"] * len(row)
                        return [""] * len(row)

                    styled_df = col_analysis_df.style.apply(highlight_issues, axis=1)
                    st.dataframe(styled_df, use_container_width=True)

                    # Issue Summary
                    if report["issues_summary"]["columns_with_issues"] > 0:
                        st.warning(f"⚠️ {report['issues_summary']['columns_with_issues']} columns have potential issues")

                        with st.expander("📋 Issues Details"):
                            for col_info in report["column_analysis"]:
                                if col_info["issues"]:
                                    st.write(f"**{col_info['column']}**: {', '.join(col_info['issues'])}")


elif page == "prediction":
    st.header("🤖 Automated Regression Modeling")
    st.markdown("Complete regression pipeline with automated model selection and prediction capabilities")
    st.info("🎯 This system performs **regression only** - predicting continuous numeric values")
    st.markdown("---")

    # Lazy import — only loaded when user visits this page
    from src.analytics.ml_models import MLModels, SmartCategoricalEncoder

    # Initialize ML models if not already done
    if (
        "ml_models" not in st.session_state
        or st.session_state.ml_models is None
        or getattr(st.session_state.ml_models, "api_version", 0) < 2
    ):
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
                help="Choose which processed dataset to use for regression modeling",
            )

            current_df = dataset_options[selected_dataset]
            training_df = current_df
            raw_training_df = None
            categorical_preprocessor = None

            if selected_dataset == "Encoded Data" and st.session_state.df_cleaned is not None:
                raw_training_df = st.session_state.df_cleaned.copy()
                categorical_preprocessor = SmartCategoricalEncoder()
            if selected_dataset == "Encoded Data" and raw_training_df is not None:
                st.caption(
                    "This run will also fit and save a raw-to-prediction pipeline so inference matches training."
                )
            st.info(
                f"Using {selected_dataset} for regression: {current_df.shape[0]:,} rows × {current_df.shape[1]} columns"
            )

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
                help="Choose the numeric column you want to predict (regression target)",
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
                    st.error(
                        f"❌ Target has only {target_info.nunique()} unique values. Need at least 3 for regression."
                    )
                    st.stop()

                if target_info.std() == 0:
                    st.error("❌ Target has no variation. Cannot perform regression.")
                    st.stop()

                st.markdown("---")

                # ML Configuration - SIMPLIFIED
                st.subheader("⚙️ Regression Configuration")
                all_regression_models = list(st.session_state.ml_models.get_regression_models().keys())
                training_scope = st.radio(
                    "Choose model training scope:",
                    ["Train all available models", "Train specific selected models"],
                    horizontal=True,
                    key="regression_training_scope",
                )

                if training_scope == "Train all available models":
                    selected_models = all_regression_models
                    search_strategy = "random"
                    strategy_label = "RandomizedSearchCV"
                    st.info(
                        "🤖 Training all available regression models uses RandomizedSearchCV to keep tuning responsive."
                    )
                else:
                    selected_models = st.multiselect(
                        "Select regression models to train:",
                        options=all_regression_models,
                        default=all_regression_models[:2],
                        key="selected_regression_models",
                    )

                    if not selected_models:
                        search_strategy = None
                        strategy_label = None
                        st.warning("Select at least one regression model to continue.")
                    elif len(selected_models) <= 2:
                        search_strategy = "grid"
                        strategy_label = "GridSearchCV"
                        st.info(
                            "🤖 With two or fewer selected models, the app uses GridSearchCV for exhaustive tuning."
                        )
                    else:
                        search_strategy = "random"
                        strategy_label = "RandomizedSearchCV"
                        st.info(
                            "🤖 With more than two selected models, the app switches to RandomizedSearchCV for faster exploration."
                        )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Correlation Threshold", "0.01", help="Features with correlation < 0.01 will be removed")

                with col2:
                    st.metric("Test Set Size", "20%", help="Automatically set to 20% of data")

                with col3:
                    st.metric("Cross-Validation", "5-fold", help="Automatically set to 5-fold CV")

                remove_all_zero_rows = st.checkbox(
                    "Supprimer automatiquement les lignes entièrement remplies de zéros avant l'entraînement",
                    value=False,
                    key="regression_drop_all_zero_rows",
                    help="Supprime uniquement les lignes dont toutes les valeurs sont égales à 0 avant le pipeline de standardisation et d'entraînement.",
                )

                if False and remove_all_zero_rows:
                    _, all_zero_row_count = st.session_state.ml_models.remove_all_zero_rows(training_df)
                    st.caption(
                        f"Lignes entièrement à zéro détectées dans le dataset d'entraînement : {all_zero_row_count}"
                    )

                with st.expander("🔧 Regression Models to be Tuned", expanded=False):
                    st.write("Selected models:")
                    for model_name in selected_models:
                        st.write(f"- {model_name}")

                    if selected_models and strategy_label:
                        st.caption(
                            f"Hyperparameter strategy: {strategy_label}. "
                            f"{'All available models are included.' if training_scope == 'Train all available models' else 'The strategy adapts to the number of selected models.'}"
                        )

                st.markdown("---")

                # Run ML Pipeline - FIXED
                if st.button("🚀 Run Automated Regression Pipeline", use_container_width=True, type="primary"):
                    if not selected_models or not search_strategy:
                        st.warning("Select at least one regression model before running the pipeline.")
                        st.stop()

                    # Create progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    with st.spinner(
                        f"🤖 Running automated regression pipeline with {strategy_label} hyperparameter tuning..."
                    ):
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
                                df=training_df,
                                target_column=target_column,
                                correlation_threshold=0.01,
                                test_size=0.2,
                                cv_folds=5,
                                save_model=True,
                                selected_models=selected_models,
                                search_strategy=search_strategy,
                                random_search_iterations=10,
                                raw_input_df=raw_training_df,
                                categorical_preprocessor=categorical_preprocessor,
                            )

                            # Update progress
                            progress_bar.progress(100)
                            status_text.text("✅ Regression pipeline completed successfully!")

                            # Store results
                            st.session_state.ml_results = results
                            st.session_state.model_diagnostics_cache = {}
                            st.session_state.selected_target = target_column
                            st.session_state.selected_dataset = selected_dataset

                            # Remove progress elements
                            progress_bar.empty()
                            status_text.empty()

                            st.success(
                                "🎉 Regression Pipeline completed successfully! Model saved for future predictions."
                            )

                            # Force a rerun to show results
                            st.rerun()

                        except Exception as e:
                            progress_bar.empty()
                            status_text.empty()
                            st.error(f"❌ Error in regression pipeline: {e!s}")
                            st.exception(e)

                            # Show debug information
                            st.write("Debug information:")
                            st.write(f"- Dataset shape: {current_df.shape}")
                            st.write(f"- Target column: {target_column}")
                            st.write(f"- Target column exists: {target_column in current_df.columns}")
                            st.write(f"- Target column type: {current_df[target_column].dtype}")
                            st.write(f"- Target unique values: {current_df[target_column].nunique()}")

            # Display Results - FIXED
            if hasattr(st.session_state, "ml_results") and st.session_state.ml_results is not None:
                st.markdown("---")
                st.subheader("📊 Regression Results")

                results = st.session_state.ml_results
                selected_dataset = getattr(st.session_state, "selected_dataset", "Unknown")

                # Check for model bundle
                bundle_path = "regression_model_bundle.zip"
                if os.path.exists(bundle_path):
                    with open(bundle_path, "rb") as fp:
                        btn = st.download_button(
                            label="📥 Download Trained Model Bundle (For Prediction)",
                            data=fp,
                            file_name="regression_model_bundle.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )

                # Quick summary at the top
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🏆 Best Model", results["best_model_name"])
                with col2:
                    st.metric("📊 Problem Type", "REGRESSION")
                with col3:
                    cv_score = results["best_score"]
                    st.metric("📈 R² Score", f"{cv_score:.4f}")
                with col4:
                    feature_count = (
                        len(st.session_state.ml_models.feature_names) if st.session_state.ml_models.feature_names else 0
                    )
                    st.metric("🎯 Features Used", feature_count)

                # Results tabs for regression
                result_tabs = st.tabs(
                    [
                        "🏆 Best Model",
                        "📈 Model Comparison",
                        "🎯 Feature Importance",
                        "📊 Prediction Analysis",
                        "📋 Training Summary",
                    ]
                )

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
                    if "test_metrics" in results:
                        st.write("### 🎯 Test Set Performance")

                        metrics = results["test_metrics"]

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
                        with col2:
                            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                        with col3:
                            st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                        with col4:
                            mape = metrics.get("mape", 0)
                            st.metric("MAPE", f"{mape:.2f}%")

                with result_tabs[1]:
                    st.write("### 📈 Regression Model Performance Comparison")

                    if "model_scores" in results:
                        # Create model summary dataframe
                        summary = []
                        for name, info in results["model_scores"].items():
                            summary.append(
                                {
                                    "Model": name,
                                    "R² Score": round(info["mean_score"], 4),
                                    "Best Params": str(info.get("best_params", "N/A")),
                                    "Is Best": name == results["best_model_name"],
                                }
                            )

                        summary_df = pd.DataFrame(summary).sort_values("R² Score", ascending=False)

                        # Style the dataframe for regression
                        styled_df = summary_df.style.highlight_max(subset=["R² Score"], color="lightgreen")
                        st.dataframe(styled_df, use_container_width=True)

                        # Performance comparison chart for regression
                        fig = px.bar(
                            summary_df,
                            x="Model",
                            y="R² Score",
                            color="Is Best",
                            title=f"Regression Model R² Score Comparison ({selected_dataset})",
                            text="R² Score",
                            color_discrete_map={True: "#1f77b4", False: "#d62728"},
                        )

                        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            height=500,
                            showlegend=True,
                            legend=dict(title="Best Model"),
                            yaxis_title="R² Score",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with result_tabs[2]:
                    st.write("### 🎯 Feature Importance Analysis")

                    if "feature_importance" in results and results["feature_importance"] is not None:
                        importance_df = results["feature_importance"]

                        # Show top features
                        st.write("**Top 15 Most Important Features for Regression:**")
                        top_features = importance_df.head(15)
                        st.dataframe(top_features, use_container_width=True)

                        # Feature importance plot
                        fig = px.bar(
                            top_features,
                            x="importance",
                            y="feature",
                            orientation="h",
                            title="Feature Importance for Regression",
                            labels={"importance": "Importance Score", "feature": "Features"},
                            color="importance",
                            color_continuous_scale="viridis",
                        )
                        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
                        st.plotly_chart(fig, use_container_width=True)

                with result_tabs[3]:
                    st.write("### ?? Prediction Analysis")

                    if results.get("model_scores"):
                        trained_model_names = list(results["model_scores"].keys())
                        current_default_model = results.get("best_model_name", trained_model_names[0])

                        st.info(
                            f"Default selection: `{current_default_model}` is currently used as the best model. "
                            "Use the selector below to explore every trained model, and override the default if another one looks more robust."
                        )

                        selected_analysis_model = st.selectbox(
                            "Select a trained model to analyze:",
                            options=trained_model_names,
                            index=trained_model_names.index(current_default_model),
                            key="prediction_analysis_model_selector",
                        )

                        selected_model_info = results["model_scores"][selected_analysis_model]
                        selected_metrics = results.get("all_test_metrics", {}).get(
                            selected_analysis_model
                        ) or results.get("test_metrics", {})

                        overview_col, action_col = st.columns([3, 2])
                        with overview_col:
                            render_regression_metric_cards(selected_metrics)
                        with action_col:
                            st.write("**Selection Status**")
                            st.write(f"- Default best model: {current_default_model}")
                            st.write(f"- Analyzing: {selected_analysis_model}")
                            st.write(
                                f"- Search strategy: {selected_model_info.get('search_strategy', results.get('training_summary', {}).get('search_strategy', 'N/A'))}"
                            )

                            if selected_analysis_model != current_default_model:
                                if st.button(
                                    f"Use {selected_analysis_model} for prediction/export",
                                    key=f"override_model_{selected_analysis_model}",
                                    use_container_width=True,
                                ):
                                    st.session_state.ml_models.set_active_model(
                                        selected_analysis_model,
                                        results["model_scores"],
                                    )
                                    results["best_model_name"] = selected_analysis_model
                                    results["best_model"] = results["model_scores"][selected_analysis_model]["model"]
                                    results["best_score"] = results["model_scores"][selected_analysis_model][
                                        "mean_score"
                                    ]
                                    if (
                                        "all_test_metrics" in results
                                        and selected_analysis_model in results["all_test_metrics"]
                                    ):
                                        results["test_metrics"] = results["all_test_metrics"][selected_analysis_model]
                                    results["feature_importance"] = st.session_state.ml_models.get_feature_importance(
                                        model=results["best_model"],
                                        feature_names=st.session_state.ml_models.feature_names,
                                    )
                                    results["training_summary"]["best_model"] = selected_analysis_model
                                    results["training_summary"]["best_score"] = results["best_score"]
                                    st.session_state.ml_results = results
                                    st.session_state.ml_models.save_model_and_preprocessors()
                                    st.success(
                                        f"{selected_analysis_model} is now the active model for predictions and bundle export."
                                    )
                                    st.rerun()
                            else:
                                st.success("Current default best model is selected.")

                        comparison_rows = []
                        for model_name, metrics in results.get("all_test_metrics", {}).items():
                            comparison_rows.append(
                                {
                                    "Model": model_name,
                                    "R? Test": round(metrics.get("r2_score", 0), 4),
                                    "RMSE Test": round(metrics.get("rmse", 0), 4),
                                    "MAE Test": round(metrics.get("mae", 0), 4),
                                    "MAPE Test (%)": round(metrics.get("mape", 0), 2),
                                    "Default Best": model_name == current_default_model,
                                }
                            )
                        if comparison_rows:
                            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

                        predictions = selected_metrics.get("predictions")
                        actual = st.session_state.ml_models.y_test
                        if predictions is not None and actual is not None:
                            actual_series = (
                                actual.reset_index(drop=True) if hasattr(actual, "reset_index") else pd.Series(actual)
                            )
                            prediction_series = pd.Series(predictions)

                            fig_scatter = px.scatter(
                                x=actual_series,
                                y=prediction_series,
                                labels={"x": "Actual Values", "y": "Predicted Values"},
                                title=f"Predicted vs Actual Values - {selected_analysis_model}",
                            )

                            min_val = min(actual_series.min(), prediction_series.min())
                            max_val = max(actual_series.max(), prediction_series.max())
                            fig_scatter.add_trace(
                                go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode="lines",
                                    name="Perfect Prediction",
                                    line=dict(color="red", dash="dash"),
                                )
                            )

                            st.plotly_chart(fig_scatter, use_container_width=True)

                            residual_series = pd.Series(selected_metrics.get("residuals"))
                            fig_residuals = px.scatter(
                                x=prediction_series,
                                y=residual_series,
                                labels={"x": "Predicted Values", "y": "Residuals"},
                                title=f"Residuals Plot - {selected_analysis_model}",
                            )
                            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_residuals, use_container_width=True)

                        cache_key = f"{selected_dataset}::{selected_analysis_model}"
                        if cache_key not in st.session_state.model_diagnostics_cache:
                            with st.spinner(f"Generating diagnostic curves for {selected_analysis_model}..."):
                                st.session_state.model_diagnostics_cache[cache_key] = (
                                    st.session_state.ml_models.compile_model_diagnostics(
                                        selected_analysis_model,
                                        selected_model_info,
                                        cv_folds=results.get("training_summary", {}).get("cv_folds", 5),
                                    )
                                )

                        diagnostics = st.session_state.model_diagnostics_cache.get(cache_key, {})
                        render_learning_curve_chart(diagnostics.get("learning_curve"), selected_analysis_model)

                        validation_curve = diagnostics.get("validation_curve")
                        if validation_curve:
                            render_validation_curve_chart(validation_curve, selected_analysis_model)

                        iteration_history = diagnostics.get("iteration_history")
                        if iteration_history:
                            render_iteration_history_chart(iteration_history, selected_analysis_model)

                        if not validation_curve and selected_analysis_model == "Linear Regression":
                            st.caption(
                                "Linear Regression exposes only the learning curve because it has no regularization hyperparameter or iterative training history."
                            )

                with result_tabs[4]:
                    st.write("### 📋 Training Summary")

                    if "training_summary" in results:
                        summary = results["training_summary"]

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Dataset Information:**")
                            st.write(f"- Target Column: {summary.get('target_column', 'N/A')}")
                            st.write(f"- Original Features: {summary.get('original_features', 'N/A')}")
                            st.write(f"- Final Features: {summary.get('final_features', 'N/A')}")
                            st.write(f"- Train Size: {summary.get('train_size', 'N/A')}")
                            st.write(f"- Test Size: {summary.get('test_size', 'N/A')}")
                            st.write(
                                f"- All-Zero Row Filter: {'Enabled' if summary.get('drop_all_zero_rows') else 'Disabled'}"
                            )
                            st.write(f"- Removed All-Zero Rows: {summary.get('removed_all_zero_rows', 0)}")

                        with col2:
                            st.write("**Model Information:**")
                            st.write(f"- Best Model: {summary.get('best_model', 'N/A')}")
                            st.write(f"- Best Score: {summary.get('best_score', 'N/A'):.4f}")

                            if summary.get("removed_features"):
                                st.write(f"- Removed Features: {len(summary['removed_features'])}")
                                with st.expander("Show removed features"):
                                    st.write(summary["removed_features"])

    # ===========================================
    # TAB 2: MAKE PREDICTIONS
    # ===========================================
    with main_tabs[1]:
        st.subheader("🔮 Predict on New Data")
        st.markdown(
            "Upload a trained model and a new dataset. The fitted TargetEncoder is loaded automatically from Streamlit cache after preprocessing."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("1️⃣ Upload Model Bundle")
            uploaded_model = st.file_uploader(
                "Upload 'regression_model_bundle.zip'", type=["zip"], key="model_uploader"
            )

            if uploaded_model:
                # Save to temp file and load
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                    tmp_file.write(uploaded_model.getvalue())
                    model_path = tmp_file.name

                try:
                    if st.session_state.ml_models.load_model_from_zip(model_path):
                        st.session_state.loaded_prediction_model = (
                            st.session_state.ml_models.full_prediction_pipeline or st.session_state.ml_models.best_model
                        )
                        st.success(f"✅ Model loaded: {st.session_state.ml_models.best_model_name}")
                        st.write(f"**Target:** {st.session_state.ml_models.target_column}")
                        st.write(f"**Best R²:** {st.session_state.ml_models.best_score:.4f}")
                        st.write(f"**Features:** {len(st.session_state.ml_models.feature_names)}")
                        st.session_state.model_loaded = True
                    else:
                        st.error("❌ Failed to load model bundle.")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(model_path)

            uploaded_direct_model = st.file_uploader(
                "Or upload a direct model artifact (.pkl/.joblib)", type=["pkl", "joblib"], key="direct_model_uploader"
            )

            if uploaded_direct_model:
                model_suffix = os.path.splitext(uploaded_direct_model.name)[1] or ".pkl"
                with tempfile.NamedTemporaryFile(delete=False, suffix=model_suffix) as tmp_file:
                    tmp_file.write(uploaded_direct_model.getvalue())
                    direct_model_path = tmp_file.name

                try:
                    st.session_state.loaded_prediction_model = joblib.load(direct_model_path)
                    st.session_state.model_loaded = True
                    st.success("✅ Direct model artifact loaded successfully.")
                except Exception as e:
                    st.session_state.loaded_prediction_model = None
                    st.session_state.model_loaded = False
                    st.error(f"Error loading direct model artifact: {e}")
                finally:
                    os.unlink(direct_model_path)

        with col2:
            st.info("2️⃣ Upload New Data")
            uploaded_data = st.file_uploader(
                "Upload new dataset (CSV/Excel)", type=["csv", "xlsx", "xlsm"], key="prediction_data_uploader"
            )

            if uploaded_data:
                # Reuse loader logic
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{uploaded_data.name.split('.')[-1]}"
                ) as tmp_file:
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

        with col3:
            st.info("3️⃣ Load Fitted TargetEncoder")
            uploaded_encoder = st.file_uploader(
                "Upload 'target_encoder.pkl' or reuse the encoder saved in this session",
                type=["pkl", "joblib"],
                key="target_encoder_uploader",
            )

            if uploaded_encoder:
                encoder_suffix = os.path.splitext(uploaded_encoder.name)[1] or ".pkl"
                with tempfile.NamedTemporaryFile(delete=False, suffix=encoder_suffix) as tmp_file:
                    tmp_file.write(uploaded_encoder.getvalue())
                    encoder_path = tmp_file.name

                try:
                    st.session_state.loaded_target_encoder = joblib.load(encoder_path)
                    st.success("✅ Fitted TargetEncoder loaded successfully.")
                except Exception as e:
                    st.session_state.loaded_target_encoder = None
                    st.error(f"Error loading encoder: {e}")
                finally:
                    os.unlink(encoder_path)
            elif st.session_state.get("target_encoder_bytes"):
                try:
                    st.session_state.loaded_target_encoder = joblib.load(BytesIO(st.session_state.target_encoder_bytes))
                    st.success("✅ Using fitted TargetEncoder saved from the Data Preprocessing section.")
                    encoded_columns = st.session_state.get("target_encoder_columns", [])
                    if encoded_columns:
                        st.write(f"**Encoded columns:** {', '.join(encoded_columns)}")
                except Exception as e:
                    st.session_state.loaded_target_encoder = None
                    st.error(f"Error restoring session TargetEncoder: {e}")

            if st.session_state.get("target_encoder_bytes"):
                try:
                    st.session_state.loaded_target_encoder = load_cached_target_encoder(
                        st.session_state.target_encoder_bytes
                    )
                    encoded_columns = st.session_state.get("target_encoder_columns", [])
                    st.info("Cached TargetEncoder from Data Preprocessing will be used automatically for prediction.")
                    if encoded_columns:
                        st.caption(f"Cached encoded columns: {', '.join(encoded_columns)}")
                except Exception as e:
                    st.session_state.loaded_target_encoder = None
                    st.error(f"Error restoring cached TargetEncoder: {e}")

        st.markdown("---")

        # Predict Button
        if (
            st.session_state.get("model_loaded")
            and st.session_state.get("new_prediction_df") is not None
            and st.session_state.get("loaded_prediction_model") is not None
        ):
            if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                with st.spinner("Generating predictions..."):
                    try:
                        new_data = st.session_state.new_prediction_df.copy()
                        loaded_model = st.session_state.loaded_prediction_model

                        if loaded_model is None:
                            raise ValueError("No trained model artifact is currently loaded.")

                        target_col = st.session_state.ml_models.target_column
                        if target_col and target_col in new_data.columns:
                            new_data = new_data.drop(columns=[target_col])

                        if hasattr(loaded_model, "named_steps") and "model" in loaded_model.named_steps:
                            predictions = loaded_model.predict(new_data)
                        else:
                            from sklearn.pipeline import Pipeline
                            from sklearn.preprocessing import FunctionTransformer

                            loaded_encoder = st.session_state.loaded_target_encoder
                            if loaded_encoder is None:
                                raise ValueError(
                                    "This model bundle requires the cached TargetEncoder, but none is available."
                                )

                            new_data = ensure_encoder_input_columns(new_data, loaded_encoder)
                            expected_feature_names = st.session_state.ml_models.feature_names or []

                            prediction_pipeline = Pipeline(
                                steps=[
                                    ("encoder", loaded_encoder),
                                    (
                                        "align_features",
                                        FunctionTransformer(
                                            align_prediction_features,
                                            kw_args={"expected_features": expected_feature_names},
                                            validate=False,
                                        ),
                                    ),
                                    ("model", loaded_model),
                                ]
                            )
                            predictions = prediction_pipeline.predict(new_data)

                        if predictions is not None:
                            # Add predictions to the ORIGINAL dataframe (including target if present)
                            result_df = st.session_state.new_prediction_df.copy()
                            prediction_target = target_col or "target"
                            prediction_col = f"Predicted_{prediction_target}"
                            result_df[prediction_col] = predictions

                            # Store in session state so results survive reruns
                            st.session_state.prediction_result_df = result_df
                            st.session_state.prediction_col_name = prediction_col
                            st.session_state.prediction_target_col = prediction_target

                            st.success("✅ Predictions generated successfully!")
                        else:
                            st.error("❌ Failed to generate predictions. Check the logs for details.")

                    except Exception as e:
                        st.error(f"Error generating predictions: {e!s}")
                        st.exception(e)
        elif (
            not st.session_state.get("model_loaded")
            or st.session_state.get("loaded_prediction_model") is None
            or st.session_state.get("new_prediction_df") is None
        ):
            st.info("👈 Upload a model bundle and a dataset to start predictions.")

        # Display prediction results (persisted in session state)
        if st.session_state.get("prediction_result_df") is not None:
            result_df = st.session_state.prediction_result_df
            prediction_col = st.session_state.get("prediction_col_name", "Predictions")
            target_col = st.session_state.get("prediction_target_col", "target")

            st.subheader("📊 Prediction Results")

            # Show prediction statistics
            pred_values = result_df[prediction_col]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Predictions Made", len(pred_values))
            with col2:
                st.metric("Mean Prediction", f"{pred_values.mean():.4f}")
            with col3:
                st.metric("Min Prediction", f"{pred_values.min():.4f}")
            with col4:
                st.metric("Max Prediction", f"{pred_values.max():.4f}")

            # Show results table
            st.dataframe(result_df, use_container_width=True)

            # Download results as CSV
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv,
                file_name=f"predictions_{target_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Visualization of predictions distribution
            st.subheader("📈 Predictions Distribution")
            fig = px.histogram(
                result_df, x=prediction_col, title=f"Distribution of Predicted {target_col}", marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)


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
    unsafe_allow_html=True,
)
