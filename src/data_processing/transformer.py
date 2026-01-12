import pandas as pd
import numpy as np
from typing import Union
import logging
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    def __init__(self):
        logger.info("DataTransformer initialized")

    def smart_categorical_encoding(self, df, target_col=None, high_card_threshold=10, drop_threshold=0.9):
        """
        Automatically chooses the best encoding for each categorical column:
        - Drops columns with more than 90% unique values
        - Uses One-Hot Encoding for low-cardinality columns
        - Uses Target Encoding for high-cardinality columns (if target provided)
        - Falls back to One-Hot Encoding if no target provided
        
        Parameters:
        - drop_threshold: float, drops columns if unique_values/total_values > threshold (default: 0.9)
        """
        def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
            """Standardize column names"""
            print("Standardizing column names...")
            df.columns = [
                '_'.join(filter(None, str(col).strip().lower()
                    .replace(' ', '_')
                    .replace('.', '')
                    .replace('%', '')
                    .replace('(', '')
                    .replace(')', '')
                    .replace('[', '')
                    .replace(']', '')
                    .replace('/', '_')
                    .replace('\\', '_')
                    .split('_')))
                for col in df.columns
            ]
            print(f"New column names: {df.columns.tolist()}")
            return df
        
        df_encoded = df.copy()
        df_encoded = _standardize_columns(df_encoded)
        cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fill target column if provided
        if target_col and target_col in df_encoded.columns:
            df_encoded[target_col] = df_encoded[target_col].fillna(df_encoded[target_col].mean())
        
        # Track already processed columns to avoid duplicates
        processed_columns = set()

        for col in cat_cols:
            # Skip if column was already processed or doesn't exist anymore
            if col not in df_encoded.columns or col in processed_columns:
                continue
                
            unique_count = df_encoded[col].nunique()
            total_count = len(df_encoded[col].dropna())
            unique_ratio = unique_count / total_count if total_count > 0 else 0

            # 1️⃣ Drop high cardinality columns (>90% unique values)
            if unique_ratio > drop_threshold:
                print(f"🗑️ Dropping '{col}' - too many unique values ({unique_count}/{total_count} = {unique_ratio:.2%})")
                df_encoded = df_encoded.drop(columns=[col])
                processed_columns.add(col)
                continue

            # 2️⃣ Low-cardinality → One-Hot Encoding
            if unique_count <= high_card_threshold:
                print(f"✅ Using One-Hot Encoding for '{col}' ({unique_count} unique values)")
                dummies = pd.get_dummies(df_encoded[col], prefix=col, dtype=int)
                
                # Check for duplicate columns before concatenation
                existing_cols = set(df_encoded.columns)
                new_dummy_cols = []
                for dummy_col in dummies.columns:
                    if dummy_col in existing_cols:
                        print(f"⚠️ Column '{dummy_col}' already exists, skipping duplicate")
                    else:
                        new_dummy_cols.append(dummy_col)
                
                if new_dummy_cols:
                    df_encoded = df_encoded.drop(columns=[col])
                    df_encoded = pd.concat([df_encoded, dummies[new_dummy_cols]], axis=1)
                else:
                    # If all dummy columns already exist, just drop the original
                    df_encoded = df_encoded.drop(columns=[col])
                
                processed_columns.add(col)

            # 3️⃣ High-cardinality → Target Encoding (if target available), otherwise One-Hot
            elif target_col and target_col in df_encoded.columns:
                print(f"🎯 Using Target Encoding for '{col}' ({unique_count} unique values)")
                encoder = ce.TargetEncoder(cols=[col])
                df_encoded[col] = encoder.fit_transform(df_encoded[col], df_encoded[target_col])
                processed_columns.add(col)
            else:
                # Fallback to One-Hot Encoding if no target provided
                print(f"✅ Using One-Hot Encoding for '{col}' ({unique_count} unique values) - No target provided")
                dummies = pd.get_dummies(df_encoded[col], prefix=col, dtype=int)
                
                # Check for duplicate columns before concatenation
                existing_cols = set(df_encoded.columns)
                new_dummy_cols = []
                for dummy_col in dummies.columns:
                    if dummy_col in existing_cols:
                        print(f"⚠️ Column '{dummy_col}' already exists, skipping duplicate")
                    else:
                        new_dummy_cols.append(dummy_col)
                
                if new_dummy_cols:
                    df_encoded = df_encoded.drop(columns=[col])
                    df_encoded = pd.concat([df_encoded, dummies[new_dummy_cols]], axis=1)
                else:
                    # If all dummy columns already exist, just drop the original
                    df_encoded = df_encoded.drop(columns=[col])
                
                processed_columns.add(col)
        
        # Final check for any remaining duplicates
        if len(df_encoded.columns) != len(set(df_encoded.columns)):
            duplicate_cols = df_encoded.columns[df_encoded.columns.duplicated()].tolist()
            print(f"⚠️ Warning: Duplicate columns detected: {duplicate_cols}")
            # Remove duplicates by keeping the first occurrence
            df_encoded = df_encoded.loc[:, ~df_encoded.columns.duplicated()]

        bool_cols = df.select_dtypes(include='bool').columns

        # Convert True/False to 1/0
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
        
        return df_encoded
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform date columns to datetime format"""
        logger.info("Starting data transformation")
        df_transformed = df.copy()
        date_columns_found = []
        
        # Look for date columns with various names
        date_keywords = ['date', 'datetime', 'timestamp', 'time', 'jour', 'journee']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                try:
                    logger.info(f"Transforming column '{col}' to datetime")
                    df_transformed[col] = pd.to_datetime(df_transformed[col], errors='coerce')
                    date_columns_found.append(col)
                except Exception as e:
                    logger.warning(f"Could not convert column '{col}' to datetime: {str(e)}")
        
        if date_columns_found:
            logger.info(f"Successfully transformed {len(date_columns_found)} date columns: {date_columns_found}")
        else:
            logger.warning("No date columns found to transform")
        
        return df_transformed
    
    def dataframe_grouping(self, df: pd.DataFrame, group_cols: Union[str, list]) -> pd.DataFrame:
        """
        Groups the DataFrame by specified columns and aggregates numerical columns.
        """
        logger.info(f"Starting dataframe grouping by: {group_cols}")
        
        # Ensure group_cols is a list
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        # Validate that group columns exist
        missing_cols = [col for col in group_cols if col not in df.columns]
        if missing_cols:
            error_msg = f"Grouping columns not found in dataframe: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Check if any groupby column is datetime
            has_datetime_group_col = any(
                pd.api.types.is_datetime64_any_dtype(df[col]) for col in group_cols
            )

            # Columns eligible for aggregation (exclude group_cols and datetime columns)
            datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
            exclude_cols = set(group_cols) | set(datetime_cols)
            numeric_cols = [col for col in df.select_dtypes(include='number').columns 
                          if col not in exclude_cols]

            if not numeric_cols:
                logger.warning("No numeric columns available for aggregation")
                return df.groupby(group_cols).first().reset_index()

            # Define aggregation functions
            if has_datetime_group_col:
                logger.info("Detected datetime in groupby columns → Using [mean, max, std]")
                agg_funcs = {col: ['mean', 'max', 'std'] for col in numeric_cols}
            else:
                logger.info("No datetime in groupby columns → Using [sum, mean, max]")
                agg_funcs = {col: ['sum', 'mean', 'max'] for col in numeric_cols}

            # Perform grouping and aggregation
            df_grouped = df.groupby(group_cols).agg(agg_funcs).reset_index()
            
            # Flatten column names if multi-level
            if isinstance(df_grouped.columns, pd.MultiIndex):
                df_grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                    for col in df_grouped.columns.values]
            
            logger.info(f"Grouping completed. Shape: {df.shape} → {df_grouped.shape}")
            return df_grouped
            
        except Exception as e:
            logger.error(f"Error during grouping: {str(e)}")
            raise
    
    def dataframe_merging(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                         on: Union[str, list], how: str = 'inner') -> pd.DataFrame:
        """
        Merges two DataFrames on specified columns.
        """
        logger.info(f"Starting dataframe merge on columns: {on}, method: {how}")
        
        # Ensure 'on' is a list
        if isinstance(on, str):
            on = [on]
        
        # Validate merge columns exist in both dataframes
        missing_in_df1 = [col for col in on if col not in df1.columns]
        missing_in_df2 = [col for col in on if col not in df2.columns]
        
        if missing_in_df1:
            error_msg = f"Merge columns not found in first dataframe: {missing_in_df1}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if missing_in_df2:
            error_msg = f"Merge columns not found in second dataframe: {missing_in_df2}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            merged_df = pd.merge(df1, df2, on=on, how=how)
            logger.info(f"Merge completed. Shapes: {df1.shape} + {df2.shape} → {merged_df.shape}")
            return merged_df
        except Exception as e:
            logger.error(f"Error during merge: {str(e)}")
            raise
    
    def dataframe_concat(self, df1: pd.DataFrame, df2: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
        """
        Concatenates two DataFrames along the specified axis.
        """
        logger.info(f"Starting dataframe concatenation along axis {axis}")
        
        try:
            concatenated_df = pd.concat([df1, df2], axis=axis, ignore_index=True)
            logger.info(f"Concatenation completed. Shapes: {df1.shape} + {df2.shape} → {concatenated_df.shape}")
            return concatenated_df
        except Exception as e:
            logger.error(f"Error during concatenation: {str(e)}")
            raise