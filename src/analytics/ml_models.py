import pandas as pd
import numpy as np
import warnings
import joblib
import os
import zipfile
import shutil
import tempfile
warnings.filterwarnings('ignore')

class MLModels:
    def __init__(self):
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
        self.model_save_path = "best_regression_model.pkl"
        self.scaler_save_path = "regression_scaler.pkl"
        self.encoders_save_path = "regression_encoders.pkl"
        self.metadata_save_path = "model_metadata.pkl"
        self.zip_save_path = "regression_model_bundle.zip"
        
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

    def prepare_data_pipeline(self, df, target_column, correlation_threshold=0.01):
        """
        Complete data preparation pipeline for regression
        """
        print(f"Starting regression data preparation pipeline...")
        print(f"Input data shape: {df.shape}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        self.target_column = target_column
        
        # Separate features and target
        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()
        
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
        
        return X_final, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
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
    
    def train_and_select_best_model(self, X_train, y_train, cv_folds=5):
        """
        Train all regression models using GridSearchCV and return only the best performing one
        """
        from sklearn.model_selection import GridSearchCV

        models = self.get_regression_models()
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
                
                # Perform Grid Search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=-1,  # Use all available cores
                    verbose=1
                )
                
                # Ensure numpy float64 to avoid sklearn dtype issues
                X_train_array = X_train.values.astype(np.float64)
                y_train_array = y_train.values.astype(np.float64) if hasattr(y_train, 'values') else np.array(y_train, dtype=np.float64)
                grid_search.fit(X_train_array, y_train_array)
                
                best_cv_score = grid_search.best_score_
                best_params = grid_search.best_params_
                best_estimator = grid_search.best_estimator_
                
                if np.isnan(best_cv_score):
                    print(f"Warning: {name} produced invalid CV scores")
                    continue
                
                model_scores[name] = {
                    'mean_score': best_cv_score,
                    'best_params': best_params,
                    'model': best_estimator
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
            
            # Handle missing values for numerical features
            for col in self.numerical_features:
                if col in X_new.columns:
                    if X_new[col].isna().any():
                        X_new[col] = X_new[col].fillna(X_new[col].median())
                    if np.isinf(X_new[col]).any():
                        X_new[col] = X_new[col].replace([np.inf, -np.inf], X_new[col].median())
            
            # Scale numerical features
            if self.numerical_features:
                available_numeric = [col for col in self.numerical_features if col in X_new.columns]
                if available_numeric:
                    X_new[available_numeric] = self.scaler.transform(X_new[available_numeric])
            
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
            
            # Remove features that were removed during training
            for col in self.removed_features:
                if col in X_new.columns:
                    X_new = X_new.drop(columns=[col])
            
            # Ensure we have all the features the model expects
            missing_features = []
            for feature in self.feature_names:
                if feature not in X_new.columns:
                    missing_features.append(feature)
                    X_new[feature] = 0  # Add missing feature with default value
            
            if missing_features:
                print(f"Warning: Missing features filled with zeros: {missing_features}")
            
            # Reorder columns to match training data
            X_new = X_new[self.feature_names]
            
            # Handle any remaining issues
            if X_new.isna().any().any():
                X_new = X_new.fillna(0)
            
            if np.isinf(X_new.select_dtypes(include=[np.number])).any().any():
                X_new = X_new.replace([np.inf, -np.inf], 0)
            
            # Make predictions
            predictions = self.best_model.predict(X_new)
            
            print(f"Predictions completed for {len(predictions)} samples")
            return predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None

    def full_regression_pipeline(self, df, target_column, correlation_threshold=0.01, 
                                test_size=0.2, cv_folds=5, save_model=True):
        """
        Complete automated regression pipeline
        """
        print("=== AUTOMATED REGRESSION PIPELINE ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Target column: {target_column}")
        print()
        
        try:
            # Data preparation
            X_final, y_final = self.prepare_data_pipeline(df, target_column, correlation_threshold)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X_final, y_final, test_size, random_state=42)
            
            # Train models and select best
            model_scores = self.train_and_select_best_model(X_train, y_train, cv_folds)
            
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
            
            # Save model if requested
            if save_model:
                self.save_model_and_preprocessors()
            
            return {
                'best_model': self.best_model,
                'best_model_name': self.best_model_name,
                'best_score': self.best_score,
                'test_metrics': test_metrics,
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
                    'best_score': self.best_score
                }
            }
                
        except Exception as e:
            print(f"Error in regression pipeline: {str(e)}")
            raise