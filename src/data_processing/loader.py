# In src/data_processing/loader.py
import pandas as pd
import streamlit as st
from typing import Union, Optional
import re

class DataLoader:
    
    def find_keyword_rows(self,df, keywords):
        """Find row indices containing any keyword (case-insensitive)"""
        pattern = re.compile('|'.join(map(re.escape, keywords)), flags=re.IGNORECASE)
        mask = pd.Series(False, index=df.index)
        
        for col in df.columns:
            # Convert to string and check for matches
            col_matches = df[col].astype(str).str.lower().str.contains(pattern, na=False)
            mask |= col_matches
        
        return df[mask].index.tolist()
    
    def _convert_object_columns_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts object-type columns to numeric only if the majority of values are numeric-like.
        Non-convertible values are coerced to NaN.
        """
        print("🔍 Converting object columns with mostly numeric values...")

        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().astype(str)
            numeric_count = sample.apply(lambda x: x.replace('.', '', 1).isdigit()).sum()
            total_count = len(sample)

            if total_count == 0:
                continue

            percentage_numeric = numeric_count / total_count

            if percentage_numeric > 0.7:  # Only convert if >70% values look numeric
                print(f"   ➡️ Converting column '{col}' to numeric ({percentage_numeric:.0%} numeric-like values)")
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def loader(self,filepath):
        if filepath.endswith('.csv'):
            preview_cols = pd.read_csv(filepath, nrows=0).columns
            index_col = 0 if 'Unnamed: 0' in preview_cols else None
        
            df = pd.read_csv(filepath, index_col=index_col)
            mask = self.find_keyword_rows(df, ['date', 'timestamp', 'datetime', 'time', 'jour', 'journee', 'annee', 'year', 'mois', 'month', 'semaine', 'week'])

            if mask:  # if at least one match was found
                df = pd.read_csv(filepath, header=mask[0], index_col=index_col)
        
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
            mask = self.find_keyword_rows(df, ['date', 'timestamp', 'datetime', 'time', 'jour', 'journee', 'annee', 'year', 'mois', 'month', 'semaine', 'week'])

            if mask:
                df = pd.read_excel(filepath, header=mask[0] + 1)

        else:
            raise ValueError("Unsupported file format. Please use .csv or .xlsx files.")
        
        df = df.loc[:, ~df.columns.str.lower().str.contains('unnamed')]


        return df