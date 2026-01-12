import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta


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

    def _can_convert_to_numeric(self, series: pd.Series, threshold: float = 0.8) -> bool:
        """
        Check if a series can be meaningfully converted to numeric.
        
        Parameters:
        series: pandas Series to check
        threshold: minimum proportion of values that should be convertible to numeric
        
        Returns:
        bool: True if series can be converted to numeric, False otherwise
        """
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(series):
            return True
            
        # Get non-null values
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return False
            
        # Try to convert to numeric and count successful conversions
        converted_series = pd.to_numeric(non_null_series, errors='coerce')
        successful_conversions = converted_series.notna().sum()
        conversion_rate = successful_conversions / len(non_null_series)
        
        # Additional checks for truly categorical data
        unique_values = non_null_series.unique()
        
        # If most values are text-like (contain letters), likely categorical
        if len(unique_values) > 0:
            text_like_count = 0
            for val in unique_values[:min(10, len(unique_values))]:  # Check first 10 unique values
                val_str = str(val).strip()
                # Check if value contains letters (excluding common numeric patterns)
                if re.search(r'[a-zA-Z]', val_str) and not re.match(r'^\d+[eE][+-]?\d+$', val_str):
                    text_like_count += 1
            
            text_like_ratio = text_like_count / min(10, len(unique_values))
            if text_like_ratio > 0.5:  # More than half contain letters
                print(f"   ⚠️ Column appears to be categorical (text-like ratio: {text_like_ratio:.2f})")
                return False
        
        # Check if it's likely a categorical column with many unique string values
        if conversion_rate < threshold:
            sample_values = non_null_series.head(5).tolist()
            print(f"   ⚠️ Low conversion rate ({conversion_rate:.2f}), sample values: {sample_values}")
            return False
            
        return True

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency - FIXED VERSION"""
        print("Optimizing data types...")
        initial_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        exclude_keywords = ['heure', 'date', 'time', 'day', 'month', 'year', 'ligne', 'poste', 'labo']

        # First pass: Identify columns that can be converted to numeric
        convertible_columns = []
        for col in df.columns:
            # Skip if column name contains excluded keywords
            if any(keyword.lower() in col.lower() for keyword in exclude_keywords):
                print(f"   ⏭️ Skipping '{col}' (contains excluded keyword)")
                continue
                
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                convertible_columns.append(col)
                continue
                
            # Test if column can be converted to numeric
            print(f"   🔍 Testing numeric conversion for '{col}'...")
            if self._can_convert_to_numeric(df[col]):
                convertible_columns.append(col)
                print(f"   ✅ '{col}' can be converted to numeric")
            else:
                print(f"   ❌ '{col}' will remain as categorical/object")

        # Second pass: Convert identified columns to numeric
        print(f"\n📊 Converting {len(convertible_columns)} columns to numeric...")
        for col in convertible_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"   ✅ '{col}': {original_dtype} → {df[col].dtype}")
        
        

        # Third pass: Optimize numeric column sizes
        print("\n🔧 Optimizing numeric column sizes...")
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_dtype = df[col].dtype
                
                # Handle invalid numeric-like patterns before optimization
                if col_dtype == 'object':
                    invalid_pattern = r'(\d+\.\D)|(\d+\.\.+\d+)|(\d+\/\.\d+)'
                    mask_invalid = df[col].astype(str).str.contains(invalid_pattern, regex=True, na=False)
                    invalid_count = mask_invalid.sum()
                    if invalid_count > 0:
                        print(f"   ⚠️ Column '{col}' has {invalid_count} invalid numeric-like string values. Setting to NaN.")
                        df.loc[mask_invalid, col] = np.nan

                # Optimize integer types
                if col_dtype == 'int64':
                    if df[col].min() >= np.iinfo(np.int8).min and df[col].max() <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                        print(f"   📉 '{col}': int64 → int8")
                    elif df[col].min() >= np.iinfo(np.int16).min and df[col].max() <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                        print(f"   📉 '{col}': int64 → int16")
                    elif df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                        print(f"   📉 '{col}': int64 → int32")

                # Optimize float types
                elif col_dtype == 'float64':
                    if df[col].min() >= np.finfo(np.float32).min and df[col].max() <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                        print(f"   📉 '{col}': float64 → float32")

        final_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"\n💾 Memory usage optimized from {initial_memory:.2f} MB to {final_memory:.2f} MB")
        
        # Summary of data types
        print(f"\n📋 Final data types summary:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"   • Numeric columns: {len(numeric_cols)}")
        print(f"   • Categorical columns: {len(categorical_cols)}")
        
        return df

    def diagnose_interpolation_issues(self, df: pd.DataFrame) -> None:
        """Diagnostic des problèmes d'interpolation"""
        print("\n🔍 === DIAGNOSTIC DES DONNÉES ===")
        
        # Trouver colonnes de date
        date_cols = [col for col in df.columns if any(k in col.lower() for k in ['date', 'time', 'datetime'])]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"📅 Colonnes de date détectées: {date_cols}")
        print(f"🔢 Colonnes numériques: {len(numeric_cols)}")
        
        # Analyser la distribution des NaN
        for col in numeric_cols[:5]:  # Analyser seulement les 5 premières
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                print(f"\n📊 Colonne '{col}':")
                print(f"   • Total NaN: {missing_count}/{len(df)} ({missing_count/len(df)*100:.1f}%)")
                
                # Trouver les "trous" de NaN consécutifs
                is_nan = df[col].isnull()
                nan_groups = (is_nan != is_nan.shift()).cumsum()[is_nan]
                
                if len(nan_groups) > 0:
                    nan_group_sizes = nan_groups.value_counts().sort_index()
                    max_consecutive = nan_group_sizes.max() if len(nan_group_sizes) > 0 else 0
                    print(f"   • Plus grand trou consécutif: {max_consecutive} valeurs")
                    print(f"   • Nombre de groupes de NaN: {len(nan_group_sizes)}")
                
                # Première et dernière valeur valide
                first_valid_idx = df[col].first_valid_index()
                last_valid_idx = df[col].last_valid_index()
                print(f"   • Première valeur valide: index {first_valid_idx}")
                print(f"   • Dernière valeur valide: index {last_valid_idx}")

    def _interpolate_by_row_frequency(self, df: pd.DataFrame, frequency_lines: int = 8) -> pd.DataFrame:
        """
        FIXED: Row-based interpolation with proper group handling
        
        Parameters:
        df: Input DataFrame
        frequency_lines: Number of lines per group (default 8, user-provided)
        
        Logic:
        - Group every X lines together
        - Within each group, if ANY valid value exists, use the median for ALL rows in that group
        - If group has NO valid values, use 0
        """
        print(f"🔄 ROW-BASED interpolation with frequency = {frequency_lines} lines per group")
        
        df_result = df.copy()

        # Keywords to exclude from numeric processing
        exclude_keywords = [
            'heure', 'hour', 'time', 'date', 'timestamp',
            'poste', 'shift', 'team', 'station', 'lab',
            'mois', 'semaine', 'id', 'rank', 'index'
        ]

        # Get numeric columns excluding those matching keywords
        numeric_cols = [
            col for col in df_result.select_dtypes(include=[np.number]).columns
            if not any(k in col.lower() for k in exclude_keywords)
        ]

        if not numeric_cols:
            print("⚠️ No numeric columns found for interpolation.")
            return df_result

        print(f"📊 Processing {len(numeric_cols)} numeric columns: {numeric_cols}")

        # Create row group assignments based on user-provided frequency_lines
        total_rows = len(df_result)
        row_groups = df_result.index // frequency_lines

        print(f"📈 Total rows: {total_rows}")
        print(f"📦 Number of groups: {row_groups.max() + 1}")
        print(f"📋 Rows per group: {frequency_lines}")

        # Process each numeric column separately
        for col in numeric_cols:
            print(f"🔧 Processing column '{col}'...")
            
            initial_missing = df_result[col].isnull().sum()
            print(f"   • Initial missing values: {initial_missing}/{total_rows} ({initial_missing/total_rows*100:.1f}%)")

            # Fill NaNs group by group
            for group_num in row_groups.unique():
                group_mask = (row_groups == group_num)
                group_data = df_result.loc[group_mask, col]
                valid_values = group_data.dropna()

                if len(valid_values) > 0:
                    group_value = valid_values.median()
                    filled_count = group_data.isnull().sum()
                    if filled_count > 0:
                        print(f"     📦 Group {group_num}: Using {group_value:.3f} (from {len(valid_values)} valid values)")
                else:
                    group_value = 0
                    filled_count = group_data.isnull().sum()
                    if filled_count > 0:
                        print(f"     📦 Group {group_num}: No valid values, using 0")

                if filled_count > 0:
                    df_result.loc[group_mask, col] = df_result.loc[group_mask, col].fillna(group_value)

            # Check column after fill
            final_missing = df_result[col].isnull().sum()
            if final_missing == 0:
                print(f"   ✅ Column '{col}': ALL NaN values eliminated!")
            else:
                print(f"   ❌ Column '{col}': {final_missing} NaN values still remain!")

        # Final summary
        final_total_missing = df_result[numeric_cols].isnull().sum().sum()
        print(f"✅ ROW-BASED INTERPOLATION COMPLETED:")
        print(f"   • Final missing values: {final_total_missing}")
        print(f"   • Shape: {df_result.shape}")

        return df_result



    def _handle_missing_values(self, df: pd.DataFrame, data_type: str = "statique", frequency_lines: int = 8) -> pd.DataFrame:
        """
        FIXED: Updated missing value handling with row-based frequency
        
        Parameters:
        - frequency_lines: number of rows per group instead of hours
        """
        
        df_cleaned = df.copy()
        
        if data_type == "frequence":
            print("🔄 Handling missing values using ROW-BASED FREQUENCE strategy")
            
            # Convert specific columns if they exist
            if 'cao_bouillie' in df_cleaned.columns:
                df_cleaned['cao_bouillie'] = pd.to_numeric(df_cleaned['cao_bouillie'], errors='coerce')
            
            exclude_keywords = ['heure', 'hour', 'time', 'date', 'timestamp', 'poste', 'shift', 'team', 'station', 'lab', 'mois', 'semaine']
            numeric_cols = [col for col in df_cleaned.select_dtypes(include=[np.number]).columns if not any(k in col.lower() for k in exclude_keywords)]

            # Find date column for filtering
            def find_column_by_keywords(keywords):
                for col in df_cleaned.columns:
                    if any(k in col.lower() for k in keywords):
                        return col
                return None
            
            date_col = find_column_by_keywords(['date', 'timestamp'])
            if not date_col or not numeric_cols:
                print("⚠️ Missing date or numeric columns. Falling back to STATIQUE strategy.")
                return self._handle_missing_values(df_cleaned, "statique")

            # Filter out bad date groups
            if date_col in df_cleaned.columns:
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

            # Apply row-based interpolation
            print(f"🎯 Application de l'interpolation par fréquence de {frequency_lines} lignes")
            df_cleaned = self._interpolate_by_row_frequency(df_cleaned, frequency_lines)
            
            # Check remaining missing values
            missing_after_interpolation = df_cleaned.isnull().sum().sum()
            print(f"📊 Valeurs manquantes après interpolation: {missing_after_interpolation}")
            
            if missing_after_interpolation > 0:
                print("⚠️ Il reste des valeurs manquantes après interpolation, application du nettoyage de secours...")
                
                # Backup cleaning logic
                for col in numeric_cols:
                    if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
                        missing_count = df_cleaned[col].isnull().sum()
                        mean_val = df_cleaned[col].mean()
                        if pd.notna(mean_val):
                            df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                            print(f"   • Colonne '{col}': {missing_count} valeurs remplies avec moyenne {mean_val:.3f}")
                        else:
                            df_cleaned[col] = df_cleaned[col].fillna(0)
                            print(f"   • Colonne '{col}': {missing_count} valeurs remplies avec 0")
                
                # Categorical columns
                categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
                for col in categorical_cols:
                    if df_cleaned[col].isnull().sum() > 0:
                        missing_count = df_cleaned[col].isnull().sum()
                        df_cleaned[col] = df_cleaned[col].fillna("Unknown")
                        print(f"   • Colonne '{col}': {missing_count} valeurs remplies avec 'Unknown'")
                        
            else:
                print("✅ Interpolation réussie ! Toutes les valeurs manquantes ont été remplies.")
            
        elif data_type == "statique":
            # Static handling
            print("🔄 Handling missing values using STATIQUE strategy")
            total_rows = len(df_cleaned)
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

            final_missing = df_cleaned.isnull().sum().sum()
            print(f"📈 STATIC CLEANING SUMMARY:")
            print(f"   • Remaining missing values: {final_missing}")
            print(f"   • Final shape: {df_cleaned.shape}")

            # Additional cleaning for static data
            def drop_columns_with_long_text(df, max_length=50):
                original_columns = df.columns.tolist()
                columns_to_keep = []
                dropped_columns = []
                
                for col in df.columns:
                    max_len_in_col = df[col].astype(str).str.len().max()
                    
                    if max_len_in_col <= max_length:
                        columns_to_keep.append(col)
                    else:
                        dropped_columns.append(col)
                        print(f"Dropping column '{col}' - max text length: {max_len_in_col}")
                
                df_filtered = df[columns_to_keep].copy()
                
                print(f"\nOriginal columns: {len(original_columns)}")
                print(f"Columns dropped: {len(dropped_columns)}")
                print(f"Remaining columns: {len(columns_to_keep)}")
                
                return df_filtered

            df_cleaned = drop_columns_with_long_text(df_cleaned)
            
            # Drop unwanted columns for static data
            keywords = ['id', 'temp', 'test', 'unnamed', 'runtime','rank','plateform']
            columns_to_drop = [col for col in df_cleaned.columns if any(keyword.lower() in col.lower() for keyword in keywords)]

            if columns_to_drop:
                df_cleaned.drop(columns=columns_to_drop, inplace=True)
                print(f"Successfully dropped {len(columns_to_drop)} columns")

            # Remove URL columns
            url_pattern = re.compile(r'(https?://|www\.)', re.IGNORECASE)
            threshold = 0.7
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype == object:
                    total_rows = len(df_cleaned[col].dropna())
                    if total_rows == 0:
                        continue
                    link_count = df_cleaned[col].dropna().astype(str).apply(lambda x: bool(url_pattern.search(x))).sum()
                    link_ratio = link_count / total_rows
                    
                    if link_ratio >= threshold:
                        df_cleaned.drop(columns=[col], inplace=True)
        
        return df_cleaned


    def clean_data(self, df: pd.DataFrame, data_type: str = "statique", frequency_lines: int = 8) -> pd.DataFrame:
        """
        Main cleaning pipeline with row-based frequency parameter
        
        Parameters:
        - frequency_lines: number of rows per group instead of hours
        """
        print("\n=== STARTING DATA CLEANING PIPELINE ===")
        print(f"Initial shape: {df.shape}")
        print(f"Initial missing values: {df.isnull().sum().sum()}")
        print(f"Data type: {data_type}")
        if data_type == "frequence":
            print(f"Interpolation frequency: {frequency_lines} lines per group")
        
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
        
        # Step 4: Handle missing values with frequency parameter
        df_cleaned = self._handle_missing_values(df_cleaned, data_type, frequency_lines)

        # Final verification
        final_missing = df_cleaned.isnull().sum().sum()
        print(f"\n=== CLEANING PIPELINE COMPLETED ===")
        print(f"Final shape: {df_cleaned.shape}")
        print(f"Final missing values: {final_missing}")
        print(f"Data cleaning {'SUCCESS' if final_missing < df.isnull().sum().sum() else 'PARTIAL'}")
        
        return df_cleaned