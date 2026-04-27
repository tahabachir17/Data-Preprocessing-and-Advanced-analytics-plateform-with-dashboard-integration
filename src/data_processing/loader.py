"""Data loading services for tabular sources."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from config.settings import ALLOWED_EXTENSIONS, NUMERIC_CONVERSION_THRESHOLD

logger = logging.getLogger(__name__)


class DataLoader:
    """Load CSV and Excel files with lightweight header detection."""

    HEADER_KEYWORDS = (
        "date",
        "timestamp",
        "datetime",
        "time",
        "jour",
        "journee",
        "annee",
        "year",
        "mois",
        "month",
        "semaine",
        "week",
    )

    def find_keyword_rows(self, df: pd.DataFrame, keywords: Iterable[str]) -> list[int]:
        """Find row indices containing any keyword, case-insensitively."""
        pattern = re.compile("|".join(map(re.escape, keywords)), flags=re.IGNORECASE)
        mask = pd.Series(False, index=df.index)

        for column in df.columns:
            column_matches = df[column].astype(str).str.contains(pattern, na=False)
            mask |= column_matches

        return df[mask].index.tolist()

    def _convert_object_columns_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert mostly numeric object columns to numeric dtype."""
        converted = df.copy()

        for column in converted.select_dtypes(include=["object"]).columns:
            sample = converted[column].dropna().astype(str)
            total_count = len(sample)
            if total_count == 0:
                continue

            numeric_count = sample.str.fullmatch(r"[-+]?\d*\.?\d+").sum()
            percentage_numeric = numeric_count / total_count

            if percentage_numeric > NUMERIC_CONVERSION_THRESHOLD:
                logger.info(
                    "Converting column '%s' to numeric (%s numeric-like values)",
                    column,
                    f"{percentage_numeric:.0%}",
                )
                converted[column] = pd.to_numeric(converted[column], errors="coerce")

        return converted

    def _validate_path(self, filepath: str | Path) -> Path:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        extension = path.suffix.lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError("Unsupported file format. Please use .csv or .xlsx files.")

        return path

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        index_col = 0 if "Unnamed: 0" in df.columns else None
        if index_col is not None:
            df = df.set_index(df.columns[0])

        mask = self.find_keyword_rows(df.head(20), self.HEADER_KEYWORDS)
        if mask:
            df = pd.read_csv(path, header=mask[0], index_col=index_col)

        return df

    def _load_excel(self, path: Path) -> pd.DataFrame:
        df = pd.read_excel(path)
        mask = self.find_keyword_rows(df.head(20), self.HEADER_KEYWORDS)
        if mask:
            df = pd.read_excel(path, header=mask[0] + 1)
        return df

    def load_data(self, filepath: str | Path) -> pd.DataFrame:
        """Load a supported tabular file into a dataframe."""
        path = self._validate_path(filepath)
        logger.info("Loading data from '%s'", path)

        try:
            if path.suffix.lower() == ".csv":
                df = self._load_csv(path)
            else:
                df = self._load_excel(path)
        except Exception as exc:
            logger.exception("Failed to load file '%s'", path)
            raise ValueError(f"Failed to load file '{path.name}': {exc}") from exc

        df = df.loc[:, ~df.columns.astype(str).str.lower().str.contains("unnamed")]
        return self._convert_object_columns_to_numeric(df)

    def loader(self, filepath: str | Path) -> pd.DataFrame:
        """Backward-compatible alias for the original public API."""
        return self.load_data(filepath)
