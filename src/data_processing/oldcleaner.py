import pandas as pd
import numpy as np

import re


class DataCleaner:
    def __init__(self):
        self.cleaning_report = {}

    def assess_data_quality(self, df: pd.DataFrame) -> dict:
        """Comprehensive data quality assessment"""
        return {
            'shape': df.shape,
            'missing_values': df.isnull().sum(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
        }

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Optimizing data types...")
        initial_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)

        for col in df.columns:
            col_dtype = df[col].dtype

            if col_dtype == 'int64':
                if df[col].min() >= np.iinfo(np.int8).min and df[col].max() <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif df[col].min() >= np.iinfo(np.int16).min and df[col].max() <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            elif col_dtype == 'float64':
                if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

            elif col_dtype == 'object':
                invalid_pattern = r'(\d+\.\D)|(\d+\.\.+\d+)|(\d+\/\.\d+)'

                mask_invalid = df[col].astype(str).str.contains(invalid_pattern, regex=True, na=False)
                invalid_count = mask_invalid.sum()
                if invalid_count > 0:
                    print(f"⚠️ Column '{col}' has {invalid_count} invalid numeric-like string values. Dropping {invalid_count} rows.")
                    df = df[~mask_invalid]


        final_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"Memory usage optimized from {initial_memory:.2f} MB to {final_memory:.2f} MB")

        return df

    def _handle_missing_values(self, df: pd.DataFrame, data_type: str = "statique") -> pd.DataFrame:
        """FIXED: Proper missing value handling with actual modifications"""
        
        # Make a copy to avoid modifying the original
        df_cleaned = df.copy()
        
        if data_type == "frequence":
            print("🔄 Handling missing values using FREQUENCE strategy (date-based grouping)")
            exclude_keywords = ['heure', 'hour', 'time', 'date', 'timestamp', 'poste', 'shift', 'team', 'station', 'lab', 'mois', 'semaine']
            numeric_cols = [col for col in df_cleaned.select_dtypes(include=[np.number]).columns if not any(k in col.lower() for k in exclude_keywords)]

            def find_column_by_keywords(keywords):
                for col in df_cleaned.columns:
                    if any(k in col.lower() for k in keywords):
                        return col
                return None

            date_col = find_column_by_keywords(['date', 'timestamp'])
            if not date_col or not numeric_cols:
                print("⚠️ Missing date or numeric columns. Falling back to STATIQUE strategy.")
                return self._handle_missing_values(df_cleaned, "statique")

            grouped = df_cleaned.groupby(date_col, group_keys=False)

            def keep_group(group):
                numeric_data = group[numeric_cols]
                non_zero = (numeric_data > 0).any().any()
                has_data = numeric_data.notna().any().any()
                return non_zero or (has_data and not (numeric_data.fillna(0) == 0).all().all())

            before = len(df_cleaned)
            df_cleaned = grouped.filter(keep_group)
            after = len(df_cleaned)
            print(f"🧹 Removed {before - after} rows from date groups with all-zero or missing measurements.")

        elif data_type == "statique":
            print("🔄 Handling missing values using STATIQUE strategy (<10% missing = drop rows, >=10% = impute)")
            total_rows = len(df_cleaned)
            rows_dropped = 0
            missing_info = df_cleaned.isnull().sum()
            columns_with_missing = missing_info[missing_info > 0]

            if columns_with_missing.empty:
                print("✅ No missing values found.")
                return df_cleaned

            print(f"📊 Found {len(columns_with_missing)} columns with missing values:")
            
            for col in columns_with_missing.index:
                missing_count = missing_info[col]
                missing_pct = (missing_count / total_rows) * 100
                print(f"   • '{col}': {missing_count}/{total_rows} missing ({missing_pct:.1f}%)")

                #if missing_pct < 10:
                    # Drop rows with missing values for this column
                #    before_drop = len(df_cleaned)
                #    df_cleaned = df_cleaned.dropna(subset=[col])
                #    dropped = before_drop - len(df_cleaned)
                #    rows_dropped += dropped
                #    print(f"     ➡️ Dropped {dropped} rows (missing < 10%)")

                #else:
                    # Impute missing values
                if df_cleaned[col].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]:
                    # For numeric columns, use mean
                    mean_val = df_cleaned[col].mean()
                    df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                    print(f"     ➡️ Imputed {missing_count} values with mean: {mean_val:.3f}")
                else:
                        # For categorical columns, use mode
                    mode_vals = df_cleaned[col].mode()
                    if not mode_vals.empty:
                        mode_val = mode_vals[0]
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                        print(f"     ➡️ Imputed {missing_count} values with mode: '{mode_val}'")
                    else:
                        # If no mode found, use "Unknown"
                        df_cleaned[col] = df_cleaned[col].fillna("Unknown")
                        print(f"     ➡️ Imputed {missing_count} values with 'Unknown'")

            # Final summary
            final_missing = df_cleaned.isnull().sum().sum()
            print(f"📈 SUMMARY:")
            print(f"   • Total rows dropped: {rows_dropped}")
            print(f"   • Remaining missing values: {final_missing}")
            print(f"   • Final shape: {df_cleaned.shape}")
            
            # Verify the cleaning worked
            if final_missing > 0:
                remaining_missing = df_cleaned.isnull().sum()
                remaining_cols = remaining_missing[remaining_missing > 0]
            
                print(f"⚠️ WARNING: Still have missing values in: {remaining_cols.to_dict()}")
            def drop_columns_with_long_text(df, max_length=50):
                """
                Drop columns where ANY cell has text length greater than max_length
                
                Parameters:
                df (pd.DataFrame): Input dataframe
                max_length (int): Maximum allowed character length (default: 20)
                
                Returns:
                pd.DataFrame: Dataframe with long-text columns removed
                """
                
                original_columns = df.columns.tolist()
                columns_to_keep = []
                dropped_columns = []
                
                for col in df.columns:
                    # Convert column to string and check max length
                    max_len_in_col = df[col].astype(str).str.len().max()
                    
                    if max_len_in_col <= max_length:
                        columns_to_keep.append(col)
                    else:
                        dropped_columns.append(col)
                        print(f"Dropping column '{col}' - max text length: {max_len_in_col}")
                
                # Create filtered dataframe
                df_filtered = df[columns_to_keep].copy()
                
                # Show results
                print(f"\nOriginal columns: {len(original_columns)}")
                print(f"Columns dropped: {len(dropped_columns)}")
                print(f"Remaining columns: {len(columns_to_keep)}")
                
                if dropped_columns:
                    print(f"Dropped columns: {dropped_columns}")
                
                return df_filtered
            df_cleaned = drop_columns_with_long_text(df_cleaned)
            keywords = ['id', 'temp', 'test', 'unnamed', 'runtime','rank','plateform']

            # SAFE VERSION - only drop columns that actually exist
            columns_to_drop = [col for col in df_cleaned.columns if any(keyword.lower() in col.lower() for keyword in keywords)]

            print(f"Available columns: {list(df_cleaned.columns)}")
            print(f"Columns to drop: {columns_to_drop}")

            if columns_to_drop:
                df_cleaned.drop(columns=columns_to_drop, inplace=True)
                print(f"Successfully dropped {len(columns_to_drop)} columns")

            # Get list of columns that can be converted
            numeric_cols = []
            for col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_cols.append(col)
                except:
                    pass

            # Convert only those columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].apply(pd.to_numeric)

        url_pattern = re.compile(r'(https?://|www\.)', re.IGNORECASE)
        threshold=0.7
        for col in df.columns:
            if df[col].dtype == object:
                total_rows = len(df[col].dropna())
                if total_rows == 0:
                    continue
                link_count = df[col].dropna().astype(str).apply(lambda x: bool(url_pattern.search(x))).sum()
                link_ratio = link_count / total_rows
                
                if link_ratio >= threshold:
                    df_cleaned.drop(columns=[col], inplace=True)
            
        
        
        return df_cleaned
    
        
    def clean_data(self, df: pd.DataFrame, data_type: str = "statique") -> pd.DataFrame:
        """Main cleaning pipeline - FIXED to ensure changes are applied"""
        print("\n=== STARTING DATA CLEANING PIPELINE ===")
        print(f"Initial shape: {df.shape}")
        print(f"Initial missing values: {df.isnull().sum().sum()}")
        
        # Start with a copy of the original data
        df_cleaned = df.copy()
        
        # Step 1: Remove duplicates
        initial_len = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        duplicates_removed = initial_len - len(df_cleaned)
        if duplicates_removed > 0:
            print(f"✅ Removed {duplicates_removed} duplicate rows.")
        else:
            print("✅ No duplicate rows found.")

        # Step 2: Standardize columns
        df_cleaned = self._standardize_columns(df_cleaned)
        
        # Step 3: Optimize data types
        df_cleaned = self._optimize_dtypes(df_cleaned)
        
        # Step 4: Handle missing values (MAIN FIX)
        df_cleaned = self._handle_missing_values(df_cleaned, data_type)

        # Final verification
        final_missing = df_cleaned.isnull().sum().sum()
        print(f"\n=== CLEANING PIPELINE COMPLETED ===")
        print(f"Final shape: {df_cleaned.shape}")
        print(f"Final missing values: {final_missing}")
        print(f"Data cleaning {'SUCCESS' if final_missing < df.isnull().sum().sum() else 'PARTIAL'}")
        
        return df_cleaned