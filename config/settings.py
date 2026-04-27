"""
Application Settings
====================
Centralized configuration for the DataFlow Pro platform.
All constants, feature flags, and tunable parameters live here.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# ───────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# ───────────────────────────────────────────────────
# Application Metadata
# ───────────────────────────────────────────────────
APP_NAME = "DataFlow Pro"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Advanced Data Preprocessing & Analytics Platform"
APP_ICON = "📈"

# ───────────────────────────────────────────────────
# Environment
# ───────────────────────────────────────────────────
ENV = os.getenv("APP_ENV", "development")  # development | staging | production
DEBUG = ENV == "development"

# ───────────────────────────────────────────────────
# File Upload
# ───────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

# ───────────────────────────────────────────────────
# Data Processing Defaults
# ───────────────────────────────────────────────────
NUMERIC_CONVERSION_THRESHOLD = 0.7   # % of values that must be numeric-like
MISSING_VALUE_STRATEGIES = [
    "drop_rows",
    "drop_columns",
    "fill_mean",
    "fill_median",
    "fill_mode",
    "fill_zero",
    "interpolate",
]
MAX_LOG_ENTRIES = 50
MAX_PLOT_HISTORY = 20

# ───────────────────────────────────────────────────
# Machine Learning Defaults
# ───────────────────────────────────────────────────
ML_TEST_SIZE = 0.20
ML_CV_FOLDS = 5
ML_RANDOM_STATE = 42
ML_CORRELATION_THRESHOLD = 0.01
ML_HIGH_CARDINALITY_THRESHOLD = 10
ML_DROP_THRESHOLD = 0.90

# ───────────────────────────────────────────────────
# Visualization
# ───────────────────────────────────────────────────
DEFAULT_CHART_HEIGHT = 500
COLOR_PALETTE = [
    "#1e3c72", "#2a5298", "#3b82f6",
    "#10b981", "#f59e0b", "#ef4444",
    "#8b5cf6", "#ec4899", "#06b6d4",
]

# ───────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOGS_DIR / "app.log"
LOG_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per log file
LOG_BACKUP_COUNT = 3               # keep 3 rotated copies


@dataclass(frozen=True)
class AppSettings:
    """Typed application settings for runtime dependency injection."""

    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    logs_dir: Path = LOGS_DIR
    env: str = ENV
    debug: bool = DEBUG
    max_upload_size_mb: int = MAX_UPLOAD_SIZE_MB
    numeric_conversion_threshold: float = NUMERIC_CONVERSION_THRESHOLD
    ml_test_size: float = ML_TEST_SIZE
    ml_cv_folds: int = ML_CV_FOLDS
    ml_random_state: int = ML_RANDOM_STATE
    log_level: str = LOG_LEVEL
    log_file: Path = LOG_FILE


def get_settings() -> AppSettings:
    """Return typed settings while keeping backward-compatible module constants."""
    return AppSettings()
