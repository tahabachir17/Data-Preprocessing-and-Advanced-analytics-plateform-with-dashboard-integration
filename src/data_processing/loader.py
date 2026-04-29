"""Data loading services for tabular sources."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from config.settings import ALLOWED_EXTENSIONS, NUMERIC_CONVERSION_THRESHOLD

logger = logging.getLogger(__name__)
SheetSelection = str | int | list[str] | list[int] | None
LoadedData = pd.DataFrame | dict[str, pd.DataFrame]


class DataLoader:
    """Load CSV and Excel files with lightweight header and sheet detection."""

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
            raise ValueError(
                "Unsupported file format. Please use .csv, .xlsx, .xls, or .xlsm files."
            )

        return path

    def _get_excel_engine(self, path: Path) -> str | None:
        if path.suffix.lower() in {".xlsx", ".xlsm"}:
            return "openpyxl"
        return None

    def get_excel_sheet_names(self, filepath: str | Path) -> list[str]:
        """Return the sheet names for a supported Excel workbook."""
        path = self._validate_path(filepath)
        if path.suffix.lower() == ".csv":
            raise ValueError("Sheet inspection is only available for Excel files.")

        with pd.ExcelFile(path, engine=self._get_excel_engine(path)) as workbook:
            return workbook.sheet_names

    def _post_process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.loc[:, ~df.columns.astype(str).str.lower().str.contains("unnamed")]
        return self._convert_object_columns_to_numeric(cleaned)

    def _post_process_loaded_data(self, data: LoadedData) -> LoadedData:
        if isinstance(data, dict):
            return {
                sheet_name: self._post_process_dataframe(sheet_df)
                for sheet_name, sheet_df in data.items()
            }
        return self._post_process_dataframe(data)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        index_col = 0 if "Unnamed: 0" in df.columns else None
        if index_col is not None:
            df = df.set_index(df.columns[0])

        mask = self.find_keyword_rows(df.head(20), self.HEADER_KEYWORDS)
        if mask:
            df = pd.read_csv(path, header=mask[0], index_col=index_col)

        return df

    def _load_excel_sheet(self, path: Path, sheet_name: str | int) -> pd.DataFrame:
        engine = self._get_excel_engine(path)
        df = pd.read_excel(path, sheet_name=sheet_name, engine=engine)
        mask = self.find_keyword_rows(df.head(20), self.HEADER_KEYWORDS)
        if mask:
            df = pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=mask[0] + 1,
                engine=engine,
            )
        return df

    def _load_excel(
        self,
        path: Path,
        sheet_name: SheetSelection = 0,
        load_all_sheets: bool = False,
    ) -> LoadedData:
        available_sheets = self.get_excel_sheet_names(path)
        if load_all_sheets or sheet_name is None:
            selected_sheets: list[str | int] = list(available_sheets)
        elif isinstance(sheet_name, list):
            selected_sheets = sheet_name
        else:
            selected_sheets = [sheet_name]

        for selected_sheet in selected_sheets:
            if isinstance(selected_sheet, str) and selected_sheet not in available_sheets:
                raise ValueError(
                    f"Sheet '{selected_sheet}' not found in '{path.name}'. "
                    f"Available sheets: {available_sheets}"
                )

        loaded_sheets = {
            str(selected_sheet): self._load_excel_sheet(path, selected_sheet)
            for selected_sheet in selected_sheets
        }

        if not load_all_sheets and sheet_name is not None and not isinstance(sheet_name, list):
            return next(iter(loaded_sheets.values()))

        return loaded_sheets

    def load_data(
        self,
        filepath: str | Path,
        *,
        sheet_name: SheetSelection = 0,
        load_all_sheets: bool = False,
    ) -> LoadedData:
        """Load a supported tabular file into a dataframe or sheet mapping."""
        path = self._validate_path(filepath)
        logger.info("Loading data from '%s'", path)

        try:
            if path.suffix.lower() == ".csv":
                data = self._load_csv(path)
            else:
                data = self._load_excel(
                    path,
                    sheet_name=sheet_name,
                    load_all_sheets=load_all_sheets,
                )
        except Exception as exc:
            logger.exception("Failed to load file '%s'", path)
            raise ValueError(f"Failed to load file '{path.name}': {exc}") from exc

        return self._post_process_loaded_data(data)

    def load_batch(
        self,
        filepaths: Iterable[str | Path],
        *,
        sheet_names: dict[str | Path, SheetSelection] | None = None,
        load_all_sheets: bool = False,
        skip_errors: bool = False,
    ) -> dict[str, LoadedData]:
        """Load multiple files, optionally with per-file Excel sheet selection."""
        loaded_data: dict[str, LoadedData] = {}
        failures: dict[str, str] = {}
        sheet_names = sheet_names or {}

        for filepath in filepaths:
            path = Path(filepath)
            requested_sheet = (
                sheet_names.get(filepath)
                if filepath in sheet_names
                else sheet_names.get(str(path), sheet_names.get(path))
            )

            try:
                loaded_data[str(path)] = self.load_data(
                    path,
                    sheet_name=requested_sheet if requested_sheet is not None else 0,
                    load_all_sheets=load_all_sheets,
                )
            except Exception as exc:
                failures[str(path)] = str(exc)
                if skip_errors:
                    logger.warning("Skipping '%s' during batch load: %s", path, exc)

        if failures and not skip_errors:
            error_summary = "; ".join(
                f"{path}: {message}" for path, message in failures.items()
            )
            raise ValueError(f"Batch load failed for one or more files: {error_summary}")

        return loaded_data

    def loader(
        self,
        filepath: str | Path,
        *,
        sheet_name: SheetSelection = 0,
        load_all_sheets: bool = False,
    ) -> LoadedData:
        """Backward-compatible alias for the original public API."""
        return self.load_data(
            filepath,
            sheet_name=sheet_name,
            load_all_sheets=load_all_sheets,
        )
