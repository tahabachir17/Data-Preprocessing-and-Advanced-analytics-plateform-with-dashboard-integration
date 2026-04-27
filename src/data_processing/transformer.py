"""Data transformation utilities for grouping, merging, and encoding."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.dataframe_utils import merge_date_and_time_columns, standardize_columns

logger = logging.getLogger(__name__)


class DataTransformer:
    """Reusable dataframe transformation service."""

    def __init__(self) -> None:
        logger.info("DataTransformer initialized")

    def smart_categorical_encoding(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
        high_card_threshold: int = 10,
        drop_threshold: float = 0.9,
    ) -> pd.DataFrame:
        """
        Encode categorical columns using one-hot or target encoding.

        Columns with near-unique identifiers are dropped to avoid leakage and
        sparse feature blow-up.
        """
        df_encoded = standardize_columns(df)
        categorical_columns = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()

        if not categorical_columns:
            bool_columns = df_encoded.select_dtypes(include="bool").columns
            if len(bool_columns) > 0:
                df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)
            return df_encoded

        import category_encoders as ce

        if target_col and target_col in df_encoded.columns:
            df_encoded[target_col] = df_encoded[target_col].fillna(df_encoded[target_col].mean())

        processed_columns: set[str] = set()
        for column in categorical_columns:
            if column not in df_encoded.columns or column in processed_columns:
                continue

            unique_count = df_encoded[column].nunique(dropna=True)
            total_count = len(df_encoded[column].dropna())
            unique_ratio = unique_count / total_count if total_count > 0 else 0.0

            if unique_ratio > drop_threshold:
                logger.info(
                    "Dropping high-cardinality column '%s' (%s/%s unique values)",
                    column,
                    unique_count,
                    total_count,
                )
                df_encoded = df_encoded.drop(columns=[column])
                processed_columns.add(column)
                continue

            if unique_count <= high_card_threshold:
                logger.info("One-hot encoding '%s'", column)
                dummies = pd.get_dummies(df_encoded[column], prefix=column, dtype=int)
                dummies = dummies.loc[:, ~dummies.columns.isin(df_encoded.columns)]
                df_encoded = df_encoded.drop(columns=[column])
                if not dummies.empty:
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                processed_columns.add(column)
                continue

            if target_col and target_col in df_encoded.columns:
                logger.info("Target encoding '%s'", column)
                encoder = ce.TargetEncoder(cols=[column])
                df_encoded[column] = encoder.fit_transform(df_encoded[column], df_encoded[target_col])
                processed_columns.add(column)
                continue

            logger.info("Falling back to one-hot encoding for '%s' without target column", column)
            dummies = pd.get_dummies(df_encoded[column], prefix=column, dtype=int)
            dummies = dummies.loc[:, ~dummies.columns.isin(df_encoded.columns)]
            df_encoded = df_encoded.drop(columns=[column])
            if not dummies.empty:
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
            processed_columns.add(column)

        if df_encoded.columns.duplicated().any():
            duplicate_columns = df_encoded.columns[df_encoded.columns.duplicated()].tolist()
            logger.warning("Duplicate columns detected after encoding: %s", duplicate_columns)
            df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]

        bool_columns = df_encoded.select_dtypes(include="bool").columns
        if len(bool_columns) > 0:
            df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)

        return df_encoded

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge date/time columns into a single datetime feature when possible."""
        logger.info("Starting data transformation")
        transformed = merge_date_and_time_columns(df)
        if "date_time" in transformed.columns:
            logger.info("Created merged 'date_time' column")
        else:
            logger.info("No mergeable date/time column pair detected")
        return transformed

    def dataframe_grouping(self, df: pd.DataFrame, group_cols: str | list[str]) -> pd.DataFrame:
        """Group a dataframe and aggregate numeric columns safely."""
        logger.info("Starting dataframe grouping by: %s", group_cols)
        grouping_columns = [group_cols] if isinstance(group_cols, str) else list(group_cols)

        missing_columns = [column for column in grouping_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Grouping columns not found in dataframe: {missing_columns}")

        has_datetime_group_col = any(pd.api.types.is_datetime64_any_dtype(df[column]) for column in grouping_columns)
        datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
        excluded_columns = set(grouping_columns) | set(datetime_cols)
        numeric_columns = [
            column for column in df.select_dtypes(include="number").columns if column not in excluded_columns
        ]

        if not numeric_columns:
            logger.warning("No numeric columns available for aggregation")
            return df.groupby(grouping_columns).first().reset_index()

        if has_datetime_group_col:
            agg_funcs = {column: ["mean", "max", "std"] for column in numeric_columns}
        else:
            agg_funcs = {column: ["sum", "mean", "max"] for column in numeric_columns}

        grouped = df.groupby(grouping_columns).agg(agg_funcs).reset_index()
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [
                "_".join(column).strip("_") if column[1] else column[0] for column in grouped.columns.values
            ]

        logger.info("Grouping completed. Shape: %s -> %s", df.shape, grouped.shape)
        return grouped

    def dataframe_merging(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        on: str | list[str],
        how: str = "inner",
    ) -> pd.DataFrame:
        """Merge two dataframes with validation."""
        logger.info("Starting dataframe merge on columns: %s, method: %s", on, how)
        merge_columns = [on] if isinstance(on, str) else list(on)
        supported_methods = {"left", "right", "inner", "outer"}
        if how not in supported_methods:
            raise ValueError(f"Unsupported merge method '{how}'. Expected one of {sorted(supported_methods)}.")

        missing_in_df1 = [column for column in merge_columns if column not in df1.columns]
        missing_in_df2 = [column for column in merge_columns if column not in df2.columns]
        if missing_in_df1:
            raise ValueError(f"Merge columns not found in first dataframe: {missing_in_df1}")
        if missing_in_df2:
            raise ValueError(f"Merge columns not found in second dataframe: {missing_in_df2}")

        merged = pd.merge(df1, df2, on=merge_columns, how=how)
        logger.info("Merge completed. Shapes: %s + %s -> %s", df1.shape, df2.shape, merged.shape)
        return merged

    def dataframe_concat(self, df1: pd.DataFrame, df2: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
        """Concatenate two dataframes along the requested axis."""
        logger.info("Starting dataframe concatenation along axis %s", axis)
        if axis not in {0, 1}:
            raise ValueError("axis must be 0 or 1")

        ignore_index = axis == 0
        concatenated = pd.concat([df1, df2], axis=axis, ignore_index=ignore_index)
        logger.info("Concatenation completed. Shapes: %s + %s -> %s", df1.shape, df2.shape, concatenated.shape)
        return concatenated
