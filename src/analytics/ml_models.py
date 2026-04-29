import pandas as pd
import numpy as np
import warnings
import joblib
import os
import zipfile
import shutil
import tempfile

from config.settings import ML_CV_FOLDS, ML_RANDOM_STATE

warnings.filterwarnings('ignore')


def remove_zero_only_rows(df: pd.DataFrame, datetime_col: str = None) -> pd.DataFrame:
    """
    Remove rows where all non-datetime columns contain only zeros.

    A row is considered "zero-only" if every non-datetime column
    contains either 0, 0.0, or NaN — with no meaningful values.
    """

    original_count = len(df)

    # Step 1: Identify columns to ignore
    cols_to_ignore = set()

    if datetime_col and datetime_col in df.columns:
        cols_to_ignore.add(datetime_col)

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            cols_to_ignore.add(col)
        elif col.lower() in {
            'datetime', 'date', 'time', 'timestamp',
            'date_time', 'created_at', 'updated_at'
        }:
            cols_to_ignore.add(col)

    # Step 2: Select only numeric columns to check
    numeric_cols = [
        col for col in df.select_dtypes(include='number').columns
        if col not in cols_to_ignore
    ]

    if not numeric_cols:
        print("Warning: No numeric columns found to check. Returning original DataFrame.")
        return df.copy()

    # Step 3: Build the zero-only mask
    numeric_data = df[numeric_cols]
    zero_or_nan = numeric_data.apply(lambda col: (col == 0) | col.isna())
    zero_only_mask = zero_or_nan.all(axis=1)

    # Step 4: Filter out zero-only rows
    clean_df = df[~zero_only_mask].reset_index(drop=True)

    removed_count = original_count - len(clean_df)

    # Step 5: Report
    print("Zero-Only Row Removal Report")
    print(f"   Columns checked  : {numeric_cols}")
    print(f"   Columns ignored  : {list(cols_to_ignore)}")
    print(f"   Original rows    : {original_count}")
    print(f"   Rows removed     : {removed_count}")
    print(f"   Remaining rows   : {len(clean_df)}")

    return clean_df


class SmartCategoricalEncoder:
    """Reproduce the app's smart categorical encoding as a reusable fitted transformer."""

    def __init__(self):
        self.encoder = None
        self.categorical_columns_ = []
        self.numeric_columns_ = []
        self.feature_names_ = []

    def fit(self, X, y=None):
        from category_encoders import OneHotEncoder

        frame = X.copy()
        self.numeric_columns_ = frame.select_dtypes(include=[np.number, 'bool']).columns.tolist()
        self.categorical_columns_ = [
            col for col in frame.columns if col not in self.numeric_columns_
        ]

        if self.categorical_columns_:
            self.encoder = OneHotEncoder(
                cols=self.categorical_columns_,
                handle_missing='value',
                handle_unknown='value',
                use_cat_names=True,
                return_df=True,
            )
            transformed = self.encoder.fit_transform(frame, y)
            self.feature_names_ = transformed.columns.tolist()
        else:
            self.encoder = None
            self.feature_names_ = frame.columns.tolist()

        return self

    def transform(self, X):
        frame = X.copy()

        if self.encoder is not None:
            transformed = self.encoder.transform(frame)
        else:
            transformed = frame

        for column in self.feature_names_:
            if column not in transformed.columns:
                transformed[column] = 0

        transformed = transformed[self.feature_names_]
        return transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class RegressionFeaturePreprocessor:
    """Mirror the model module's numeric scaling, feature filtering, and categorical encoding."""

    def __init__(self, correlation_threshold=0.01):
        self.correlation_threshold = correlation_threshold
        self.scaler = None
        self.label_encoders_ = {}
        self.scaler_feature_names_ = []
        self.numerical_features_ = []
        self.categorical_features_ = []
        self.removed_features_ = []
        self.feature_names_ = []

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        frame = X.copy()
        y_series = pd.Series(y, index=frame.index)
        self.numerical_features_ = frame.select_dtypes(include=[np.number, 'bool']).columns.tolist()
        self.categorical_features_ = [
            col for col in frame.columns if col not in self.numerical_features_
        ]

        scaled = frame.copy()
        if self.numerical_features_:
            self.scaler = StandardScaler()
            self.scaler_feature_names_ = list(self.numerical_features_)

            for column in self.numerical_features_:
                if np.isinf(scaled[column]).any():
                    median_value = scaled[column].replace([np.inf, -np.inf], np.nan).median()
                    scaled[column] = scaled[column].replace([np.inf, -np.inf], median_value)
                if scaled[column].isna().any():
                    scaled[column] = scaled[column].fillna(scaled[column].median())

            scaled[self.numerical_features_] = self.scaler.fit_transform(
                scaled[self.numerical_features_]
            )

            correlations = scaled[self.numerical_features_].corrwith(y_series).abs().fillna(0)
            self.removed_features_ = correlations[
                correlations <= self.correlation_threshold
            ].index.tolist()
            scaled = scaled.drop(columns=self.removed_features_, errors='ignore')
            self.numerical_features_ = [
                feature for feature in self.numerical_features_
                if feature not in self.removed_features_
            ]

        active_categorical_features = [
            col for col in self.categorical_features_ if col in scaled.columns
        ]
        encoders = {}
        for column in active_categorical_features:
            encoder = LabelEncoder()
            scaled[column] = scaled[column].fillna('missing_value').astype(str)
            scaled[column] = encoder.fit_transform(scaled[column])
            encoders[column] = encoder

        self.label_encoders_ = encoders
        scaled = self._finalize_frame(scaled)
        self.feature_names_ = scaled.columns.tolist()
        return self

    def transform(self, X):
        frame = X.copy()

        if self.removed_features_:
            frame = frame.drop(columns=self.removed_features_, errors='ignore')

        if self.scaler is not None and self.scaler_feature_names_:
            scaler_df = pd.DataFrame(0, index=frame.index, columns=self.scaler_feature_names_)
            for column in self.scaler_feature_names_:
                if column in frame.columns:
                    values = frame[column]
                    if np.isinf(values).any():
                        values = values.replace([np.inf, -np.inf], np.nan)
                    scaler_df[column] = values.fillna(values.median())

            scaled_values = self.scaler.transform(scaler_df[self.scaler_feature_names_])
            scaled_df = pd.DataFrame(
                scaled_values, index=frame.index, columns=self.scaler_feature_names_
            )
            for column in self.numerical_features_:
                if column in scaled_df.columns:
                    frame[column] = scaled_df[column]

        for column, encoder in self.label_encoders_.items():
            if column not in frame.columns:
                frame[column] = 'missing_value'
            values = frame[column].fillna('missing_value').astype(str)
            known_classes = set(encoder.classes_)
            default_class = encoder.classes_[0]
            values = values.apply(lambda value: value if value in known_classes else default_class)
            frame[column] = encoder.transform(values)

        frame = self._finalize_frame(frame)
        for column in self.feature_names_:
            if column not in frame.columns:
                frame[column] = 0
        return frame[self.feature_names_].astype(np.float64)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def _finalize_frame(self, dataframe):
        finalized = dataframe.copy()
        numeric_subset = finalized.select_dtypes(include=[np.number, 'bool']).columns
        if len(numeric_subset) > 0:
            finalized[numeric_subset] = finalized[numeric_subset].replace([np.inf, -np.inf], np.nan)

        for column in finalized.columns:
            if not np.issubdtype(finalized[column].dtype, np.number):
                finalized[column] = pd.to_numeric(finalized[column], errors='coerce')

        return finalized.fillna(0)


class MLModels:
    def __init__(self):
        self.api_version = 2
        self.scaler = None  # Created lazily in scale_numerical_features()
        self.label_encoders = {}  # Store multiple label encoders
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_column = None
        self.feature_names = None
        self.removed_features = []
        self.numerical_features = []
        self.categorical_features = []
        self.full_prediction_pipeline = None
        self.model_save_path = "best_regression_model.pkl"
        self.scaler_save_path = "regression_scaler.pkl"
        self.encoders_save_path = "regression_encoders.pkl"
        self.metadata_save_path = "model_metadata.pkl"
        self.full_pipeline_save_path = "full_regression_pipeline.pkl"
        self.zip_save_path = "regression_model_bundle.zip"
        self.n_jobs = max(1, int(os.getenv("ML_N_JOBS", "1")))

    def get_all_zero_row_mask(self, df, datetime_col=None):
        """
        Return a mask aligned with remove_zero_only_rows for UI previews.
        """
        cols_to_ignore = set()

        if datetime_col and datetime_col in df.columns:
            cols_to_ignore.add(datetime_col)

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                cols_to_ignore.add(col)
            elif col.lower() in {
                'datetime', 'date', 'time', 'timestamp',
                'date_time', 'created_at', 'updated_at'
            }:
                cols_to_ignore.add(col)

        numeric_cols = [
            col for col in df.select_dtypes(include='number').columns
            if col not in cols_to_ignore
        ]

        if not numeric_cols:
            return pd.Series(False, index=df.index)

        numeric_data = df[numeric_cols]
        zero_or_nan = numeric_data.apply(lambda col: (col == 0) | col.isna())
        return zero_or_nan.all(axis=1)

    def remove_all_zero_rows(self, df, datetime_col=None):
        """
        Remove rows using the shared zero-only filtering helper.
        """
        filtered_df = remove_zero_only_rows(df, datetime_col=datetime_col)
        removed_count = len(df) - len(filtered_df)
        return filtered_df, removed_count
        
    def get_available_targets(self, df):
        """
        Return available numeric target columns for regression selection
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_info = []
        
        for col in numeric_columns:
            col_info = {
                'column': col,
                'dtype': str(df[col].dtype),
                'unique_values': df[col].nunique(),
                'missing_values': df[col].isnull().sum(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'sample_values': df[col].dropna().head(3).tolist()
            }
            columns_info.append(col_info)
        
        return pd.DataFrame(columns_info)
    
    def scale_numerical_features(self, X):
        """
        Scale only numerical features (int and float)
        """
        from sklearn.preprocessing import StandardScaler

        if self.scaler is None:
            self.scaler = StandardScaler()

        X_scaled = X.copy()
        
        # Identify numerical and categorical features
        self.numerical_features = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()
        self.categorical_features = X.select_dtypes(exclude=[np.number, 'bool']).columns.tolist()
        
        print(f"Numerical features to scale ({len(self.numerical_features)}): {self.numerical_features}")
        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
        if self.numerical_features:
            # Handle infinite and missing values
            for col in self.numerical_features:
                if np.isinf(X[col]).any():
                    print(f"Warning: Infinite values found in {col}, replacing with median")
                    X_scaled[col] = X_scaled[col].replace([np.inf, -np.inf], X_scaled[col].median())
                
                if X[col].isna().any():
                    print(f"Warning: NaN values found in {col}, filling with median")
                    X_scaled[col] = X_scaled[col].fillna(X_scaled[col].median())
            
            # Scale numerical features
            try:
                X_scaled[self.numerical_features] = self.scaler.fit_transform(X_scaled[self.numerical_features])
                print("Numerical features scaled successfully")
            except Exception as e:
                print(f"Error scaling numerical features: {e}")
        
        return X_scaled

    def encode_categorical_features(self, X):
        """
        Encode categorical features using label encoding with error handling
        """
        from sklearn.preprocessing import LabelEncoder

        X_encoded = X.copy()
        
        for col in self.categorical_features:
            try:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Handle missing values first
                X_encoded[col] = X_encoded[col].fillna('missing_value')
                X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col].astype(str))
                print(f"Successfully encoded categorical feature: {col}")
            except Exception as e:
                print(f"Error encoding {col}: {e}")
                # If encoding fails, drop the column
                X_encoded = X_encoded.drop(columns=[col])
                if col in self.categorical_features:
                    self.categorical_features.remove(col)
        
        return X_encoded
    
    def remove_zero_correlation_features(self, X, y, correlation_threshold=0.01):
        """
        Remove features with very low correlation to target after scaling
        """
        self.removed_features = []
        
        if not self.numerical_features:
            print("No numerical features found for correlation analysis")
            return X
        
        try:
            # Calculate correlations only for numerical features
            numeric_X = X[self.numerical_features]
            correlations = numeric_X.corrwith(pd.Series(y, index=X.index)).abs()
            
            # Handle NaN correlations
            correlations = correlations.fillna(0)
            
            # Find features with correlation <= threshold
            features_to_remove = correlations[correlations <= correlation_threshold].index.tolist()
            
            if features_to_remove:
                print(f"Removing {len(features_to_remove)} numerical features with correlation <= {correlation_threshold}")
                print(f"Removed features: {features_to_remove}")
                X_filtered = X.drop(columns=features_to_remove)
                self.removed_features = features_to_remove
                
                # Update numerical features list
                self.numerical_features = [f for f in self.numerical_features if f not in features_to_remove]
            else:
                X_filtered = X
                print("No features removed based on correlation threshold")
                
        except Exception as e:
            print(f"Error in correlation analysis: {e}")
            X_filtered = X
        
        return X_filtered

    def prepare_data_pipeline(
        self,
        df,
        target_column,
        correlation_threshold=0.01,
        drop_all_zero_rows=False,
    ):
        """
        Complete data preparation pipeline for regression
        """
        print(f"Starting regression data preparation pipeline...")
        print(f"Input data shape: {df.shape}")

        working_df = df.copy()
        removed_all_zero_rows = 0

        if drop_all_zero_rows:
            working_df, removed_all_zero_rows = self.remove_all_zero_rows(working_df)
            print(f"Shape after all-zero row removal: {working_df.shape}")
        
        if target_column not in working_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        self.target_column = target_column
        
        # Separate features and target
        X = working_df.drop(columns=[target_column]).copy()
        y = working_df[target_column].copy()
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target missing values: {y.isnull().sum()}")
        
        # Remove rows where target is missing
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        if len(y) == 0:
            raise ValueError("No valid target values found after removing missing values")
        
        # Check if target is suitable for regression
        if y.nunique() < 3:
            raise ValueError(f"Target has only {y.nunique()} unique values. Not suitable for regression.")
        
        if y.std() == 0:
            raise ValueError("Target has no variation (all values are the same). Cannot perform regression.")
        
        print(f"After removing missing targets - X: {X.shape}, y: {y.shape}")
        
        # Step 1: Scale numerical features
        print("Step 1: Scaling numerical features...")
        X_scaled = self.scale_numerical_features(X)
        
        # Step 2: Remove low correlation features
        print("Step 2: Removing low correlation features...")
        X_filtered = self.remove_zero_correlation_features(X_scaled, y, correlation_threshold)
        
        # Step 3: Encode categorical features
        print("Step 3: Encoding categorical features...")
        X_final = self.encode_categorical_features(X_filtered)
        
        # Final data validation
        if X_final.shape[1] == 0:
            raise ValueError("No features remaining after preprocessing")
        
        if X_final.isna().any().any():
            print("Warning: NaN values still present, filling with zeros")
            X_final = X_final.fillna(0)
        
        if np.isinf(X_final.select_dtypes(include=[np.number])).any().any():
            print("Warning: Infinite values still present, replacing with zeros")
            X_final = X_final.replace([np.inf, -np.inf], 0)
        
        # Final safety: convert ALL columns to numeric (catches bool, object leftovers)
        for col in X_final.columns:
            if not np.issubdtype(X_final[col].dtype, np.number):
                try:
                    X_final[col] = pd.to_numeric(X_final[col], errors='coerce').fillna(0)
                    print(f"Safety-converted non-numeric column '{col}' to numeric")
                except Exception:
                    print(f"Warning: Dropping unconvertible column '{col}'")
                    X_final = X_final.drop(columns=[col])
        
        # Store final feature names
        self.feature_names = X_final.columns.tolist()
        print(f"Final feature count: {len(self.feature_names)}")
        print(f"Final feature names: {self.feature_names[:10]}...")
        
        return X_final, y, removed_all_zero_rows
    
    def split_data(self, X, y, test_size=0.2, random_state=ML_RANDOM_STATE):
        """
        Split data for regression
        """
        from sklearn.model_selection import train_test_split

        print(f"Splitting data with test_size={test_size}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Data split completed: Train {self.X_train.shape}, Test {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_regression_models(self):
        """
        Get regression models with hyperparameter grids for tuning
        """
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

        models = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False]
                }
            },
            'Ridge Regression': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
                }
            },
            'Lasso Regression': {
                'model': Lasso(max_iter=2000),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'selection': ['cyclic', 'random']
                }
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }
        return models

    def _resolve_cv_folds(self, n_samples, cv_folds):
        """Reduce CV folds safely for small datasets."""
        if n_samples < cv_folds:
            return max(2, n_samples // 2)
        return cv_folds

    def get_model_diagnostic_config(self, model_name):
        """Return diagnostic plot configuration for each regression model."""
        model_configs = self.get_regression_models()
        base_params = model_configs.get(model_name, {}).get('params', {})

        config = {
            'validation_param': None,
            'validation_values': None,
            'supports_iteration_history': False,
        }

        if model_name in {'Ridge Regression', 'Lasso Regression'}:
            config['validation_param'] = 'alpha'
            config['validation_values'] = base_params.get('alpha', [0.1, 1.0, 10.0])
        elif model_name == 'Random Forest':
            config['validation_param'] = 'max_depth'
            config['validation_values'] = base_params.get('max_depth', [None, 10, 20])
        elif model_name == 'Gradient Boosting':
            config['validation_param'] = 'learning_rate'
            config['validation_values'] = base_params.get('learning_rate', [0.01, 0.1, 0.2])
            config['supports_iteration_history'] = True

        return config

    def generate_learning_curve_data(self, model, X, y, cv_folds=ML_CV_FOLDS):
        """Generate RMSE learning curve data for a regression model."""
        from sklearn.base import clone
        from sklearn.model_selection import learning_curve

        X_frame = X.astype(np.float64)
        y_array = y.values.astype(np.float64) if hasattr(y, 'values') else np.array(y, dtype=np.float64)
        cv = self._resolve_cv_folds(len(X_frame), cv_folds)

        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=clone(model),
            X=X_frame,
            y=y_array,
            cv=cv,
            train_sizes=np.linspace(0.2, 1.0, 5),
            scoring='neg_mean_squared_error',
            n_jobs=self.n_jobs,
        )

        return {
            'train_sizes': train_sizes.tolist(),
            'train_rmse': np.sqrt(-train_scores.mean(axis=1)).tolist(),
            'validation_rmse': np.sqrt(-validation_scores.mean(axis=1)).tolist(),
        }

    def generate_validation_curve_data(self, model_name, best_params, X, y, cv_folds=ML_CV_FOLDS):
        """Generate RMSE validation curve data when the model has a meaningful hyperparameter."""
        from sklearn.base import clone
        from sklearn.model_selection import validation_curve

        diagnostic_config = self.get_model_diagnostic_config(model_name)
        param_name = diagnostic_config.get('validation_param')
        param_values = diagnostic_config.get('validation_values')

        if not param_name or not param_values:
            return None

        models = self.get_regression_models()
        if model_name not in models:
            return None

        estimator = clone(models[model_name]['model'])
        fixed_params = {
            key: value
            for key, value in (best_params or {}).items()
            if key != param_name
        }
        if fixed_params:
            estimator.set_params(**fixed_params)

        X_frame = X.astype(np.float64)
        y_array = y.values.astype(np.float64) if hasattr(y, 'values') else np.array(y, dtype=np.float64)
        cv = self._resolve_cv_folds(len(X_frame), cv_folds)

        train_scores, validation_scores = validation_curve(
            estimator=estimator,
            X=X_frame,
            y=y_array,
            param_name=param_name,
            param_range=param_values,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=self.n_jobs,
        )

        return {
            'param_name': param_name,
            'param_values': [str(value) for value in param_values],
            'train_rmse': np.sqrt(-train_scores.mean(axis=1)).tolist(),
            'validation_rmse': np.sqrt(-validation_scores.mean(axis=1)).tolist(),
        }

    def generate_iteration_history(self, model_name, model, X_train, y_train, X_test, y_test):
        """Generate per-iteration train/test RMSE for boosting models."""
        from sklearn.metrics import mean_squared_error

        if model_name != 'Gradient Boosting' or not hasattr(model, 'staged_predict'):
            return None

        train_predictions = list(model.staged_predict(X_train.astype(np.float64)))
        test_predictions = list(model.staged_predict(X_test.astype(np.float64)))

        iterations = list(range(1, len(train_predictions) + 1))
        train_rmse = [
            float(np.sqrt(mean_squared_error(y_train, prediction)))
            for prediction in train_predictions
        ]
        validation_rmse = [
            float(np.sqrt(mean_squared_error(y_test, prediction)))
            for prediction in test_predictions
        ]

        return {
            'iterations': iterations,
            'train_rmse': train_rmse,
            'validation_rmse': validation_rmse,
        }

    def evaluate_all_trained_models(self, model_scores, X_test, y_test):
        """Evaluate every trained model on the held-out test set."""
        all_metrics = {}
        for model_name, model_info in model_scores.items():
            metrics, _ = self.evaluate_model(model_info['model'], X_test, y_test)
            all_metrics[model_name] = metrics
        return all_metrics

    def compile_model_diagnostics(self, model_name, model_info, cv_folds=ML_CV_FOLDS):
        """Compile model-specific diagnostics using stored train/test data."""
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Training and test data are not available for diagnostics.")

        model = model_info['model']
        best_params = model_info.get('best_params', {})

        diagnostics = {
            'learning_curve': self.generate_learning_curve_data(
                model, self.X_train, self.y_train, cv_folds=cv_folds
            ),
            'validation_curve': self.generate_validation_curve_data(
                model_name, best_params, self.X_train, self.y_train, cv_folds=cv_folds
            ),
            'iteration_history': self.generate_iteration_history(
                model_name, model, self.X_train, self.y_train, self.X_test, self.y_test
            ),
            'config': self.get_model_diagnostic_config(model_name),
        }
        return diagnostics

    def set_active_model(self, model_name, model_scores):
        """Allow the UI to override which trained model is treated as active/best."""
        if model_name not in model_scores:
            raise ValueError(f"Model '{model_name}' is not available in the trained results.")

        self.best_model_name = model_name
        self.best_model = model_scores[model_name]['model']
        self.best_score = model_scores[model_name]['mean_score']
    
    def train_and_select_best_model(
        self,
        X_train,
        y_train,
        cv_folds=ML_CV_FOLDS,
        selected_model_names=None,
        search_strategy="grid",
        random_search_iterations=10,
    ):
        """
        Train selected regression models using GridSearchCV or RandomizedSearchCV.
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        models = self.get_regression_models()
        if selected_model_names:
            models = {
                name: config
                for name, config in models.items()
                if name in selected_model_names
            }

        if not models:
            raise ValueError("No regression models selected for training.")

        model_scores = {}
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Target statistics: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        
        # Adjust CV folds if dataset is too small
        if X_train.shape[0] < cv_folds:
            cv_folds = max(2, X_train.shape[0] // 2)
            print(f"Reduced CV folds to {cv_folds} due to small dataset")
        
        # Train all models and evaluate
        successful_models = 0
        for name, config in models.items():
            try:
                print(f"Tuning {name}...")
                model = config['model']
                params = config['params']
                
                if search_strategy == "random":
                    total_combinations = 1
                    for values in params.values():
                        total_combinations *= len(values)
                    n_iter = min(random_search_iterations, total_combinations)
                    model_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=params,
                        n_iter=n_iter,
                        cv=cv_folds,
                        scoring='r2',
                        n_jobs=self.n_jobs,
                        verbose=1,
                        random_state=ML_RANDOM_STATE
                    )
                else:
                    model_search = GridSearchCV(
                        estimator=model,
                        param_grid=params,
                        cv=cv_folds,
                        scoring='r2',
                        n_jobs=self.n_jobs,
                        verbose=1
                    )
                
                X_train_frame = X_train.astype(np.float64)
                y_train_array = y_train.values.astype(np.float64) if hasattr(y_train, 'values') else np.array(y_train, dtype=np.float64)
                model_search.fit(X_train_frame, y_train_array)
                
                best_cv_score = model_search.best_score_
                best_params = model_search.best_params_
                best_estimator = model_search.best_estimator_
                std_score = float(model_search.cv_results_['std_test_score'][model_search.best_index_])
                
                if np.isnan(best_cv_score):
                    print(f"Warning: {name} produced invalid CV scores")
                    continue
                
                model_scores[name] = {
                    'mean_score': best_cv_score,
                    'std_score': std_score,
                    'best_params': best_params,
                    'model': best_estimator,
                    'search_strategy': search_strategy,
                }
                
                successful_models += 1
                print(f"{name}: Best R² = {best_cv_score:.4f} with params {best_params} ✓")
                
            except Exception as e:
                import traceback
                print(f"Error training {name}: {str(e)}")
                print(traceback.format_exc())
                continue
        
        print(f"Successfully trained {successful_models} out of {len(models)} models")
        
        if successful_models == 0:
            raise ValueError("No models were successfully trained!")
        
        # Find best model
        self.best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['mean_score'])
        self.best_model = model_scores[self.best_model_name]['model']
        self.best_score = model_scores[self.best_model_name]['mean_score']
        
        print(f"\nBest performing model: {self.best_model_name}")
        print(f"Best R² score: {self.best_score:.4f}")
        
        return model_scores
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Comprehensive model evaluation for regression
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        try:
            predictions = model.predict(X_test)
            
            # Calculate regression metrics
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE (Mean Absolute Percentage Error) safely
            mape = 0
            if (y_test != 0).any():
                mape = np.mean(np.abs((y_test - predictions) / np.where(y_test != 0, y_test, 1))) * 100
            
            metrics = {
                'r2_score': r2,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'predictions': predictions,
                'residuals': y_test - predictions
            }
            
            return metrics, predictions
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {'r2_score': 0.0, 'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}, []

    def get_feature_importance(self, model=None, feature_names=None):
        """
        Extract feature importance with error handling
        """
        try:
            if model is None:
                model = self.best_model
            if feature_names is None:
                feature_names = self.feature_names
            
            if feature_names is None or model is None:
                return None
                
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                return importance_df
                
            elif hasattr(model, 'coef_'):
                # For linear models
                if len(model.coef_.shape) > 1:
                    importance_values = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importance_values = np.abs(model.coef_)
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_values.flatten()
                }).sort_values('importance', ascending=False)
                return importance_df
            else:
                return None
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            return None

    def save_model_and_preprocessors(self):
        """
        Save the best model and preprocessors to a ZIP bundle
        """
        try:
            # Create a temporary directory to store files before zipping
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, self.model_save_path)
                scaler_path = os.path.join(temp_dir, self.scaler_save_path)
                encoders_path = os.path.join(temp_dir, self.encoders_save_path)
                metadata_path = os.path.join(temp_dir, self.metadata_save_path)
                full_pipeline_path = os.path.join(temp_dir, self.full_pipeline_save_path)
                
                # Save artifacts
                joblib.dump(self.best_model, model_path)
                joblib.dump(self.scaler, scaler_path)
                joblib.dump(self.label_encoders, encoders_path)
                
                metadata = {
                    'best_model_name': self.best_model_name,
                    'best_score': self.best_score,
                    'target_column': self.target_column,
                    'feature_names': self.feature_names,
                    'numerical_features': self.numerical_features,
                    'categorical_features': self.categorical_features,
                    'removed_features': self.removed_features
                }
                joblib.dump(metadata, metadata_path)
                
                # Create ZIP file
                with zipfile.ZipFile(self.zip_save_path, 'w') as zipf:
                    zipf.write(model_path, self.model_save_path)
                    zipf.write(scaler_path, self.scaler_save_path)
                    zipf.write(encoders_path, self.encoders_save_path)
                    zipf.write(metadata_path, self.metadata_save_path)
                    if self.full_prediction_pipeline is not None:
                        joblib.dump(self.full_prediction_pipeline, full_pipeline_path)
                        zipf.write(full_pipeline_path, self.full_pipeline_save_path)
                
                print(f"Model bundle saved to cross-platform zip: {self.zip_save_path}")
                return self.zip_save_path
            
        except Exception as e:
            print(f"Error saving model bundle: {e}")
            return None

    def load_model_from_zip(self, zip_path):
        """
        Load model and preprocessors from a ZIP file
        """
        try:
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"Zip file not found: {zip_path}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    zipf.extractall(temp_dir)
                
                # Load components
                self.best_model = joblib.load(os.path.join(temp_dir, self.model_save_path))
                self.scaler = joblib.load(os.path.join(temp_dir, self.scaler_save_path))
                self.label_encoders = joblib.load(os.path.join(temp_dir, self.encoders_save_path))
                
                metadata = joblib.load(os.path.join(temp_dir, self.metadata_save_path))
                self.best_model_name = metadata['best_model_name']
                self.best_score = metadata['best_score']
                self.target_column = metadata['target_column']
                self.feature_names = metadata['feature_names']
                self.numerical_features = metadata['numerical_features']
                self.categorical_features = metadata['categorical_features']
                self.removed_features = metadata['removed_features']
                full_pipeline_bundle_path = os.path.join(temp_dir, self.full_pipeline_save_path)
                if os.path.exists(full_pipeline_bundle_path):
                    self.full_prediction_pipeline = joblib.load(full_pipeline_bundle_path)
                else:
                    self.full_prediction_pipeline = None
                
                print("Model bundle loaded successfully")
                return True
                
        except Exception as e:
            print(f"Error loading model bundle: {e}")
            return False

    def predict_new_data(self, new_df):
        """
        Make predictions on new data using the saved model
        """
        try:
            if self.best_model is None:
                raise ValueError("No model loaded. Please load a model first.")
            
            print(f"Processing new data for prediction: {new_df.shape}")
            
            # Prepare the new data using the same pipeline
            X_new = new_df.copy()
            
            # Remove target column if present in new data
            if self.target_column and self.target_column in X_new.columns:
                print(f"Removing target column '{self.target_column}' from prediction data")
                X_new = X_new.drop(columns=[self.target_column])
            
            # Remove features that were removed during training
            for col in self.removed_features:
                if col in X_new.columns:
                    X_new = X_new.drop(columns=[col])
            
            # Drop any extra columns not in the expected feature set
            # (keeps only columns that are in feature_names, numerical_features, or categorical_features)
            expected_cols = set(self.feature_names) | set(self.numerical_features) | set(self.categorical_features)
            extra_cols = [col for col in X_new.columns if col not in expected_cols]
            if extra_cols:
                print(f"Dropping {len(extra_cols)} extra columns not used during training: {extra_cols}")
                X_new = X_new.drop(columns=extra_cols)
            
            # Handle missing values for numerical features
            for col in self.numerical_features:
                if col in X_new.columns:
                    if X_new[col].isna().any():
                        X_new[col] = X_new[col].fillna(X_new[col].median())
                    if np.isinf(X_new[col]).any():
                        X_new[col] = X_new[col].replace([np.inf, -np.inf], X_new[col].median())
            
            # Scale numerical features using only the features the scaler was fitted on
            if self.scaler is not None and self.numerical_features:
                # Get the feature names the scaler was fitted on
                scaler_features = list(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else self.numerical_features
                # Only transform columns that exist in both the new data and the scaler
                available_numeric = [col for col in scaler_features if col in X_new.columns]
                if available_numeric:
                    # Build a DataFrame with all scaler features, filling missing ones with 0
                    scaler_df = pd.DataFrame(0, index=X_new.index, columns=scaler_features)
                    for col in available_numeric:
                        scaler_df[col] = X_new[col]
                    scaled_values = self.scaler.transform(scaler_df)
                    scaled_df = pd.DataFrame(scaled_values, index=X_new.index, columns=scaler_features)
                    # Only copy back columns that exist in X_new
                    for col in available_numeric:
                        X_new[col] = scaled_df[col]
            
            # Encode categorical features
            for col in self.categorical_features:
                if col in X_new.columns and col in self.label_encoders:
                    try:
                        # Handle missing values
                        X_new[col] = X_new[col].fillna('missing_value')
                        # Handle unseen categories
                        X_new[col] = X_new[col].astype(str)
                        
                        # Transform known categories, assign a default for unknown ones
                        encoder = self.label_encoders[col]
                        unique_values = X_new[col].unique()
                        known_classes = set(encoder.classes_)
                        
                        for value in unique_values:
                            if value not in known_classes:
                                # Assign the most common encoded value for unknown categories
                                X_new[col] = X_new[col].replace(value, encoder.classes_[0])
                        
                        X_new[col] = encoder.transform(X_new[col])
                    except Exception as e:
                        print(f"Error encoding {col}: {e}")
                        # Drop problematic columns
                        X_new = X_new.drop(columns=[col])
            
            # Ensure we have all the features the model expects
            missing_features = []
            for feature in self.feature_names:
                if feature not in X_new.columns:
                    missing_features.append(feature)
                    X_new[feature] = 0  # Add missing feature with default value
            
            if missing_features:
                print(f"Warning: Missing features filled with zeros: {missing_features}")
            
            # Reorder columns to match training data (and drop any extras)
            X_new = X_new[self.feature_names]
            
            # Handle any remaining issues
            if X_new.isna().any().any():
                X_new = X_new.fillna(0)
            
            if np.isinf(X_new.select_dtypes(include=[np.number])).any().any():
                X_new = X_new.replace([np.inf, -np.inf], 0)
            
            # Make predictions
            predictions = self.best_model.predict(X_new.astype(np.float64))
            
            print(f"Predictions completed for {len(predictions)} samples")
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            import traceback
            traceback.print_exc()
            return None

    def full_regression_pipeline(self, df, target_column, correlation_threshold=0.01,
                                test_size=0.2, cv_folds=5, save_model=True,
                                selected_models=None, search_strategy="grid",
                                random_search_iterations=10, raw_input_df=None,
                                categorical_preprocessor=None,
                                drop_all_zero_rows=False):
        """
        Complete automated regression pipeline
        """
        from sklearn.pipeline import Pipeline

        print("=== AUTOMATED REGRESSION PIPELINE ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Target column: {target_column}")
        print()
        
        try:
            # Data preparation
            X_final, y_final, removed_all_zero_rows = self.prepare_data_pipeline(
                df,
                target_column,
                correlation_threshold,
                drop_all_zero_rows=drop_all_zero_rows,
            )
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X_final, y_final, test_size, random_state=42)
            
            # Train models and select best
            model_scores = self.train_and_select_best_model(
                X_train,
                y_train,
                cv_folds,
                selected_model_names=selected_models,
                search_strategy=search_strategy,
                random_search_iterations=random_search_iterations,
            )
            
            # Evaluate best model on test set
            print(f"\n=== FINAL EVALUATION ON TEST SET ===")
            test_metrics, test_predictions = self.evaluate_model(
                self.best_model, X_test, y_test
            )
            
            print(f"Best Model: {self.best_model_name}")
            print(f"Test Set Performance:")
            print(f"  R² Score: {test_metrics['r2_score']:.4f}")
            print(f"  RMSE: {test_metrics['rmse']:.4f}")
            print(f"  MAE: {test_metrics['mae']:.4f}")
            print(f"  MAPE: {test_metrics['mape']:.2f}%")

            self.full_prediction_pipeline = None
            if raw_input_df is not None and categorical_preprocessor is not None:
                try:
                    raw_features = raw_input_df.drop(columns=[target_column], errors='ignore').copy()
                    X_train_raw = raw_features.loc[X_train.index]
                    fitted_categorical_preprocessor = categorical_preprocessor.fit(X_train_raw, y_train)
                    X_train_encoded = fitted_categorical_preprocessor.transform(X_train_raw)

                    regression_preprocessor = RegressionFeaturePreprocessor(
                        correlation_threshold=correlation_threshold
                    )
                    X_train_pipeline = regression_preprocessor.fit_transform(
                        X_train_encoded,
                        y_train,
                    )

                    if list(X_train_pipeline.columns) == list(X_train.columns):
                        self.full_prediction_pipeline = Pipeline(
                            steps=[
                                ('categorical_encoder', fitted_categorical_preprocessor),
                                ('regression_preprocessor', regression_preprocessor),
                                ('model', self.best_model),
                            ]
                        )
                        print("Saved a full raw-to-prediction pipeline for inference.")
                    else:
                        print(
                            "Skipping raw prediction pipeline save because feature alignment "
                            "did not match the trained regression inputs."
                        )
                except Exception as pipeline_error:
                    print(
                        "Warning: could not build the optional raw prediction pipeline: "
                        f"{pipeline_error}"
                    )
            
            # Save model if requested
            if save_model:
                self.save_model_and_preprocessors()
            
            return {
                'best_model': self.best_model,
                'best_model_name': self.best_model_name,
                'best_score': self.best_score,
                'test_metrics': test_metrics,
                'all_test_metrics': self.evaluate_all_trained_models(model_scores, X_test, y_test),
                'model_scores': model_scores,
                'feature_importance': self.get_feature_importance(),
                'training_summary': {
                    'target_column': self.target_column,
                    'original_features': len(self.feature_names) + len(self.removed_features),
                    'final_features': len(self.feature_names),
                    'removed_features': self.removed_features,
                    'numerical_features': self.numerical_features,
                    'categorical_features': self.categorical_features,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'best_model': self.best_model_name,
                    'best_score': self.best_score,
                    'selected_models': list(model_scores.keys()),
                    'search_strategy': search_strategy,
                    'cv_folds': cv_folds,
                    'drop_all_zero_rows': drop_all_zero_rows,
                    'removed_all_zero_rows': removed_all_zero_rows,
                }
            }
                
        except Exception as e:
            print(f"Error in regression pipeline: {str(e)}")
            raise
