"""Shared dataframe utilities used across ingestion and transformation modules."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


def normalize_column_name(column_name: object) -> str:
    """Return a normalized, snake_case-friendly column name."""
    normalized = (
        str(column_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("%", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_")
        .replace("\\", "_")
    )
    return "_".join(filter(None, normalized.split("_")))


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names."""
    standardized = df.copy()
    standardized.columns = [normalize_column_name(col) for col in standardized.columns]
    return standardized


def parse_time_value(value: object) -> str:
    """Parse a time-like value into ``HH:MM:SS`` when possible."""
    text = str(value).strip()
    match = re.match(r"^(\d{1,2})\s*[Hh:]\s*(\d{0,2})$", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        return f"{hour:02d}:{minute:02d}:00"

    if text.isdigit():
        numeric_time = int(text)
        if numeric_time <= 24:
            return f"{numeric_time:02d}:00:00"
        hour = numeric_time // 100
        minute = numeric_time % 100
        return f"{hour:02d}:{minute:02d}:00"

    return text


def merge_date_and_time_columns(
    df: pd.DataFrame,
    date_keywords: Iterable[str] = ("date", "datetime", "timestamp", "jour", "journee"),
    time_keywords: Iterable[str] = ("heure", "hour", "time"),
) -> pd.DataFrame:
    """Merge matching date and time columns into a single ``date_time`` column."""
    merged = df.copy()

    date_col = None
    for column in merged.columns:
        if any(keyword in column.lower() for keyword in date_keywords):
            candidate = pd.to_datetime(merged[column], errors="coerce")
            if candidate.notna().any():
                merged[column] = candidate
                date_col = column
                break

    if date_col is None:
        return merged

    time_col = None
    for column in merged.columns:
        if column == date_col:
            continue
        if any(keyword in column.lower() for keyword in time_keywords):
            time_col = column
            break

    if time_col is None:
        return merged

    date_part = merged[date_col].dt.date.astype(str)
    time_part = merged[time_col].map(parse_time_value)
    merged["date_time"] = pd.to_datetime(date_part + " " + time_part, errors="coerce")
    return merged.drop(columns=[date_col, time_col])
