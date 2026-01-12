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

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        print("Optimizing data types...")
        initial_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        exclude_keywords = ['heure', 'date', 'time', 'day', 'month', 'year', 'ligne', 'poste', 'labo']

        for col in df.columns:
            # Check if column name contains any excluded keyword (case-insensitive)
            if not any(keyword.lower() in col.lower() for keyword in exclude_keywords):
                df[col] = pd.to_numeric(df[col], errors='coerce') 

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
                # Remove invalid numeric-like patterns
                invalid_pattern = r'(\d+\.\D)|(\d+\.\.+\d+)|(\d+\/\.\d+)'
                mask_invalid = df[col].astype(str).str.contains(invalid_pattern, regex=True, na=False)
                invalid_count = mask_invalid.sum()
                if invalid_count > 0:
                    print(f"⚠️ Column '{col}' has {invalid_count} invalid numeric-like string values. Dropping {invalid_count} rows.")
                    df = df[~mask_invalid]

        final_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"Memory usage optimized from {initial_memory:.2f} MB to {final_memory:.2f} MB")
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

    def _interpolate_by_frequency(self, df: pd.DataFrame, frequency_hours: int = 1) -> pd.DataFrame:
        """
        ENHANCED: Interpolation des données par fréquence avec gestion complète des NaN
        """
        print(f"🔄 Interpolation par fréquence de {frequency_hours} heure(s)")
        
        # Trouver les colonnes de date et heure avec priorités
        date_col = None
        heure_col = None
        
        # Priorité pour les colonnes de date
        date_keywords = ['date_c', 'date', 'timestamp', 'datetime']
        for keyword in date_keywords:
            for col in df.columns:
                if col.lower() == keyword or keyword in col.lower():
                    date_col = col
                    break
            if date_col:
                break
        
        # Chercher colonne heure si pas trouvée dans date
        heure_keywords = ['heure', 'hour', 'time']
        for keyword in heure_keywords:
            for col in df.columns:
                if keyword in col.lower() and (col.lower() != date_col.lower() if date_col else True):
                    heure_col = col
                    break
            if heure_col:
                break
        
        if not date_col:
            print("⚠️ Aucune colonne de date trouvée. Interpolation impossible.")
            return df
        
        df_interpolated = df.copy()
        has_separate_heure = False
        
        try:
            # Cas 1: Colonne date + colonne heure séparées
            if heure_col and heure_col in df_interpolated.columns:
                print(f"📅 Mode SÉPARÉ - Date: '{date_col}', Heure: '{heure_col}'")
                has_separate_heure = True
                
                # Nettoyer et convertir la colonne heure
                def convert_heure_to_time(heure_str):
                    """Convertit différents formats d'heure en format standard"""
                    if pd.isna(heure_str):
                        return "00:00:00"
                    
                    heure_str = str(heure_str).upper().strip()
                    
                    # Format "06H00" -> "06:00:00"
                    if 'H' in heure_str:
                        parts = heure_str.replace('H', ':')
                        if ':' in parts:
                            time_parts = parts.split(':')
                            if len(time_parts) == 2:
                                return f"{time_parts[0].zfill(2)}:{time_parts[1].zfill(2)}:00"
                        else:
                            return f"{parts.zfill(2)}:00:00"
                    
                    # Format "06:00" ou "6:00"
                    elif ':' in heure_str:
                        time_parts = heure_str.split(':')
                        if len(time_parts) == 2:
                            return f"{time_parts[0].zfill(2)}:{time_parts[1].zfill(2)}:00"
                        elif len(time_parts) == 3:
                            return f"{time_parts[0].zfill(2)}:{time_parts[1].zfill(2)}:{time_parts[2].zfill(2)}"
                    
                    # Format numérique simple "6", "14", etc.
                    elif heure_str.isdigit():
                        return f"{heure_str.zfill(2)}:00:00"
                    
                    return "00:00:00"
                
                # Convertir les heures
                df_interpolated['heure_converted'] = df_interpolated[heure_col].apply(convert_heure_to_time)
                
                # Combiner date et heure
                df_interpolated['datetime_combined'] = pd.to_datetime(
                    df_interpolated[date_col].astype(str) + ' ' + df_interpolated['heure_converted'].astype(str),
                    errors='coerce'
                )
                
                datetime_col = 'datetime_combined'
                
            # Cas 2: Colonne date contient déjà l'heure complète
            else:
                print(f"📅 Mode INTÉGRÉ - Date avec heure: '{date_col}'")
                datetime_col = date_col
                
                # Vérifier le format de la colonne date
                sample_value = df_interpolated[date_col].dropna().iloc[0] if len(df_interpolated[date_col].dropna()) > 0 else None
                print(f"📊 Exemple de valeur: {sample_value}")
                
                # Convertir en datetime avec gestion des différents formats
                if not pd.api.types.is_datetime64_any_dtype(df_interpolated[date_col]):
                    # Essayer différents formats
                    formats_to_try = [
                        '%Y-%m-%d %H:%M:%S',
                        '%d/%m/%Y %H:%M:%S',
                        '%Y-%m-%d %H:%M',
                        '%d/%m/%Y %H:%M',
                        '%Y-%m-%d',
                        '%d/%m/%Y'
                    ]
                    
                    converted = False
                    for fmt in formats_to_try:
                        try:
                            df_interpolated[datetime_col] = pd.to_datetime(df_interpolated[date_col], format=fmt)
                            print(f"✅ Format détecté: {fmt}")
                            converted = True
                            break
                        except:
                            continue
                    
                    if not converted:
                        # Dernière tentative avec infer_datetime_format
                        df_interpolated[datetime_col] = pd.to_datetime(df_interpolated[date_col], errors='coerce', infer_datetime_format=True)
            
            # Vérifier s'il y a des dates invalides
            invalid_dates = df_interpolated[datetime_col].isnull().sum()
            if invalid_dates > 0:
                print(f"⚠️ {invalid_dates} dates invalides trouvées. Suppression de ces lignes.")
                df_interpolated = df_interpolated.dropna(subset=[datetime_col])
            
            if len(df_interpolated) == 0:
                print("❌ Aucune donnée valide après nettoyage des dates.")
                return df
            
            # Trier par datetime
            df_interpolated = df_interpolated.sort_values(by=datetime_col).reset_index(drop=True)
            
            # Définir l'index comme la colonne de datetime
            df_interpolated.set_index(datetime_col, inplace=True)
            
            # Créer une nouvelle fréquence basée sur les heures
            if frequency_hours >= 1:
                freq_str = f'{int(frequency_hours)}H'
            else:
                # Pour les fractions d'heure, convertir en minutes
                minutes = int(frequency_hours * 60)
                freq_str = f'{minutes}min'
            
            # Obtenir les colonnes numériques pour l'interpolation
            exclude_keywords = ['id', 'rank', 'index', 'moi', 'dat', 'semai', 'poste', 'heure', 'datetime']
            numeric_cols = df_interpolated.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if not any(k in col.lower() for k in exclude_keywords)]
            
            if not numeric_cols:
                print("⚠️ Aucune colonne numérique trouvée pour l'interpolation.")
                return df_interpolated.reset_index()
            
            print(f"📊 Colonnes numériques à interpoler: {numeric_cols}")
            
            # Créer un index complet avec la fréquence demandée
            start_date = df_interpolated.index.min()
            end_date = df_interpolated.index.max()
            full_index = pd.date_range(start=start_date, end=end_date, freq=freq_str)
            
            print(f"📅 Période: {start_date} à {end_date}")
            print(f"📈 Points de données originaux: {len(df_interpolated)}")
            print(f"📈 Points après interpolation: {len(full_index)}")
            
            # Reindex avec le nouvel index complet
            df_reindexed = df_interpolated.reindex(full_index)
            
            # INTERPOLATION AMÉLIORÉE - Gestion complète des NaN
            for col in numeric_cols:
                if col in df_reindexed.columns:
                    print(f"🔧 Traitement colonne '{col}'...")
                    
                    # Statistiques avant traitement
                    initial_missing = df_reindexed[col].isnull().sum()
                    total_values = len(df_reindexed[col])
                    missing_pct = initial_missing / total_values * 100
                    print(f"   • Valeurs manquantes initiales: {initial_missing}/{total_values} ({missing_pct:.1f}%)")
                    
                    # CAS 1: Série complètement vide ou presque
                    if missing_pct > 95:
                        print(f"   ⚠️ Série quasi-vide ({missing_pct:.1f}%), remplissage avec médiane globale...")
                        global_median = df_reindexed[col].median()
                        if pd.notna(global_median):
                            df_reindexed[col] = df_reindexed[col].fillna(global_median)
                        else:
                            # Si même pas de médiane, utiliser 0 ou une valeur par défaut
                            df_reindexed[col] = df_reindexed[col].fillna(0)
                            print(f"   ➡️ Aucune valeur valide, remplissage avec 0")
                            
                    # CAS 2: Beaucoup de valeurs manquantes (80-95%)
                    elif missing_pct > 80:
                        print(f"   ⚠️ Beaucoup de valeurs manquantes ({missing_pct:.1f}%), stratégie hybride...")
                        
                        # Obtenir les indices avec des valeurs valides
                        valid_indices = df_reindexed[col].dropna().index
                        
                        if len(valid_indices) > 0:
                            # Calculer médiane sur les valeurs existantes
                            median_val = df_reindexed[col].median()
                            
                            # Étape 1: Remplir les zones aux extrémités
                            first_valid = valid_indices[0]
                            last_valid = valid_indices[-1]
                            
                            # Avant la première valeur valide
                            mask_before = df_reindexed.index < first_valid
                            df_reindexed.loc[mask_before, col] = df_reindexed.loc[mask_before, col].fillna(median_val)
                            
                            # Après la dernière valeur valide  
                            mask_after = df_reindexed.index > last_valid
                            df_reindexed.loc[mask_after, col] = df_reindexed.loc[mask_after, col].fillna(median_val)
                            
                            print(f"   ✅ Pré-remplissage extrémités avec médiane: {median_val:.3f}")
                    
                    # Étape principale: Interpolation linéaire
                    print("   🔄 Application interpolation linéaire...")
                    df_reindexed[col] = df_reindexed[col].interpolate(
                        method='linear', 
                        limit_direction='both',
                        limit=None  # Pas de limite sur le nombre de valeurs à interpoler
                    )
                    
                    # Étape de sécurité 1: Forward fill puis Backward fill (VERSION CORRIGÉE)
                    remaining_nan_1 = df_reindexed[col].isnull().sum()
                    if remaining_nan_1 > 0:
                        print(f"   🔄 {remaining_nan_1} NaN restants, application forward/backward fill...")
                        # NOUVELLE SYNTAXE pour pandas récent
                        df_reindexed[col] = df_reindexed[col].ffill().bfill()
                    
                    # Étape de sécurité 2: Remplissage avec médiane
                    remaining_nan_2 = df_reindexed[col].isnull().sum()
                    if remaining_nan_2 > 0:
                        print(f"   🔄 {remaining_nan_2} NaN encore restants, remplissage médiane...")
                        median_val = df_reindexed[col].median()
                        if pd.notna(median_val):
                            df_reindexed[col] = df_reindexed[col].fillna(median_val)
                            print(f"   ➡️ Remplissage avec médiane: {median_val:.3f}")
                    
                    # Étape de sécurité 3: Dernier recours - valeur constante
                    remaining_nan_3 = df_reindexed[col].isnull().sum()
                    if remaining_nan_3 > 0:
                        print(f"   ⚠️ {remaining_nan_3} NaN ENCORE présents, dernier recours...")
                        
                        # Essayer avec la moyenne des valeurs non-NaN de la colonne
                        if df_reindexed[col].notna().any():
                            fallback_val = df_reindexed[col].mean()
                        else:
                            # Si vraiment aucune valeur, utiliser 0
                            fallback_val = 0
                        
                        df_reindexed[col] = df_reindexed[col].fillna(fallback_val)
                        print(f"   ➡️ Dernier remplissage avec: {fallback_val:.3f}")
                    
                    # Vérification finale
                    final_missing = df_reindexed[col].isnull().sum()
                    if final_missing == 0:
                        print(f"   ✅ Colonne '{col}' : TOUS les NaN éliminés !")
                    else:
                        print(f"   ❌ Colonne '{col}' : {final_missing} NaN PERSISTENT - PROBLÈME CRITIQUE")
                        
                        # Debug : montrer quelques valeurs pour diagnostic
                        nan_positions = df_reindexed[df_reindexed[col].isnull()].index[:3]
                        print(f"      Positions NaN persistants: {list(nan_positions)}")
            
            # Pour les colonnes non-numériques (VERSION CORRIGÉE)
            categorical_cols = df_reindexed.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                if col not in ['heure_converted']:  # Éviter colonnes temporaires
                    missing_count = df_reindexed[col].isnull().sum()
                    if missing_count > 0:
                        print(f"🔤 Traitement colonne catégorielle '{col}': {missing_count} NaN")
                        # NOUVELLE SYNTAXE pour pandas récent  
                        df_reindexed[col] = df_reindexed[col].ffill().bfill()
                        
                        # Si encore des NaN, remplir avec "Unknown"
                        remaining = df_reindexed[col].isnull().sum()
                        if remaining > 0:
                            df_reindexed[col] = df_reindexed[col].fillna("Unknown")
                            print(f"   ➡️ {remaining} valeurs remplies avec 'Unknown'")
            
            # Reset l'index pour revenir au format original
            df_result = df_reindexed.reset_index()
            df_result.rename(columns={'index': 'datetime'}, inplace=True)
            
            # Recréer les colonnes selon le format original
            if has_separate_heure:
                # Cas: colonnes Date et Heure séparées
                df_result[date_col] = df_result['datetime'].dt.date
                df_result[heure_col] = df_result['datetime'].dt.strftime('%H') + 'H' + df_result['datetime'].dt.strftime('%M')
                
                # Supprimer les colonnes temporaires
                columns_to_drop = ['heure_converted', 'datetime_combined']
                for col in columns_to_drop:
                    if col in df_result.columns:
                        df_result.drop(columns=[col], inplace=True)
                        
                # Supprimer la colonne datetime si elle n'existait pas à l'origine
                if 'datetime' in df_result.columns and 'datetime' not in df.columns:
                    df_result.drop(columns=['datetime'], inplace=True)
                    
            else:
                # Cas: colonne date unique (renommer selon l'original)
                df_result.rename(columns={'datetime': date_col}, inplace=True)
            
            # Statistiques finales
            interpolated_points = len(df_result) - len(df_interpolated)
            final_total_missing = df_result.isnull().sum().sum()
            print(f"✅ Interpolation terminée:")
            print(f"   • {interpolated_points} nouveaux points de données ajoutés")
            print(f"   • Forme finale: {df_result.shape}")
            print(f"   • Valeurs manquantes finales: {final_total_missing}")
            print(f"   • Colonnes finales: {list(df_result.columns)}")
            
            return df_result
            
        except Exception as e:
            print(f"❌ Erreur lors de l'interpolation: {str(e)}")
            import traceback
            traceback.print_exc()
            return df

    def _handle_missing_values(self, df: pd.DataFrame, data_type: str = "statique", frequency_hours: int = 1) -> pd.DataFrame:
        """FULLY FIXED: Proper missing value handling with complete NaN elimination"""
        
        df_cleaned = df.copy()
        
        if data_type == "frequence":
            print("🔄 Handling missing values using FREQUENCE strategy (date-based interpolation)")
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

            # Application de l'interpolation par fréquence
            print(f"🎯 Application de l'interpolation par fréquence de {frequency_hours} heure(s)")
            df_cleaned = self._interpolate_by_frequency(df_cleaned, frequency_hours)
            
            # Vérification post-interpolation
            missing_after_interpolation = df_cleaned.isnull().sum().sum()
            print(f"📊 Valeurs manquantes après interpolation: {missing_after_interpolation}")
            
            # Nettoyage de secours si nécessaire
            if missing_after_interpolation > 0:
                print("⚠️ Il reste des valeurs manquantes après interpolation, application du nettoyage de secours...")
                
                exclude_keywords = ['heure', 'hour', 'time', 'date', 'timestamp', 'poste', 'shift', 'team', 'station', 'lab', 'mois', 'semaine', 'datetime']
                numeric_cols = [col for col in df_cleaned.select_dtypes(include=[np.number]).columns 
                            if not any(k in col.lower() for k in exclude_keywords)]
                
                # Remplissage final pour colonnes numériques
                for col in numeric_cols:
                    if df_cleaned[col].isnull().sum() > 0:
                        missing_count = df_cleaned[col].isnull().sum()
                        mean_val = df_cleaned[col].mean()
                        if pd.notna(mean_val):
                            df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                            print(f"   • Colonne '{col}': {missing_count} valeurs remplies avec moyenne {mean_val:.3f}")
                        else:
                            df_cleaned[col] = df_cleaned[col].fillna(0)
                            print(f"   • Colonne '{col}': {missing_count} valeurs remplies avec 0")
                
                # Remplissage final pour colonnes catégorielles
                categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
                for col in categorical_cols:
                    if df_cleaned[col].isnull().sum() > 0:
                        missing_count = df_cleaned[col].isnull().sum()
                        df_cleaned[col] = df_cleaned[col].fillna("Unknown")
                        print(f"   • Colonne '{col}': {missing_count} valeurs remplies avec 'Unknown'")
                        
            else:
                print("✅ Interpolation réussie ! Toutes les valeurs manquantes ont été remplies.")
            
        elif data_type == "statique":
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

            # Final summary for static
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

            # Convert numeric columns
            numeric_cols = []
            for col in df_cleaned.columns:
                try:
                    pd.to_numeric(df_cleaned[col], errors='raise')
                    numeric_cols.append(col)
                except:
                    pass

            if numeric_cols:
                df_cleaned[numeric_cols] = df_cleaned[numeric_cols].apply(pd.to_numeric)

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
    
    def clean_data(self, df: pd.DataFrame, data_type: str = "statique", frequency_hours: int = 1) -> pd.DataFrame:
        """
        Main cleaning pipeline - ENHANCED with frequency interpolation
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        data_type (str): "statique" or "frequence"
        frequency_hours (int): Hours frequency for interpolation (only for "frequence" type)
        """
        print("\n=== STARTING DATA CLEANING PIPELINE ===")
        print(f"Initial shape: {df.shape}")
        print(f"Initial missing values: {df.isnull().sum().sum()}")
        print(f"Data type: {data_type}")
        if data_type == "frequence":
            print(f"Interpolation frequency: {frequency_hours} hour(s)")
        
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
        df_cleaned = self._handle_missing_values(df_cleaned, data_type, frequency_hours)

        # Final verification
        final_missing = df_cleaned.isnull().sum().sum()
        print(f"\n=== CLEANING PIPELINE COMPLETED ===")
        print(f"Final shape: {df_cleaned.shape}")
        print(f"Final missing values: {final_missing}")
        print(f"Data cleaning {'SUCCESS' if final_missing < df.isnull().sum().sum() else 'PARTIAL'}")
        
        return df_cleaned