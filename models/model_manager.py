"""
Model Manager - single responsibility for ML model operations
Now saves models to the data folder
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import joblib
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

class ModelManager:
    """Manages all ML model operations"""
    
    def __init__(self, config: Dict[str, Any], log_queue=None):
        self.config = config
        self.log_queue = log_queue
        self.model = None
        self.feature_columns = None
        self.model_path = config['model']['save_path']
        self.features_path = config['model']['features_save_path']
        
        # Get params with defaults for backward compatibility
        self.n_estimators = config['model'].get('n_estimators', 100)
        self.max_depth = config['model'].get('max_depth', 10)
        self.min_samples_split = config['model'].get('min_samples_split', 2)
        self.min_samples_leaf = config['model'].get('min_samples_leaf', 1)
        self.random_state = config['model'].get('random_state', 42)
        
        # Ensure data directory exists
        self._ensure_data_directory()
    
    def _log(self, message: str, level: str = 'info'):
        """Log message to UI log panel or fallback to print"""
        if self.log_queue:
            self.log_queue.put(message, level)
        else:
            print(message)
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        data_dirs = [
            os.path.dirname(self.model_path),
            os.path.dirname(self.features_path)
        ]
        for data_dir in data_dirs:
            if data_dir and not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
                self._log(f"Created directory: {data_dir}")
    
    def engineer_features(self, df: pd.DataFrame, for_prediction: bool = False) -> pd.DataFrame:
        """Create predictive features from raw data with validation.
        Args:
            for_prediction: If True, keeps the last row (no target needed for prediction).
        """
        if df.empty:
            self._log("⚠ Cannot engineer features: DataFrame is empty", 'warning')
            return pd.DataFrame()
        
        # Check required columns
        required_cols = ['daily_return', 'price']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            self._log(f"⚠ Missing required columns: {missing}", 'warning')
            return pd.DataFrame()
        
        # Check minimum data points (need enough for lags and indicators)
        if len(df) < 50:
            self._log(f"⚠ Insufficient data: {len(df)} records (need at least 50)", 'warning')
            return pd.DataFrame()
        
        df = df.copy()
        
        # Target: 1 if next day positive, else 0
        df['target'] = (df['daily_return'].shift(-1) > 0).astype(int)
        
        # Remove last row only for training (no target available)
        if not for_prediction:
            df = df.iloc[:-1]
        
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
        
        # Futures features
        if 'asx_futures' in df.columns:
            df['asx_futures_return'] = df['asx_futures'].pct_change()
            df['futures_premium'] = df['asx_futures'] / df['price'] - 1
        
        if 'sp500_futures' in df.columns:
            df['sp500_futures_return'] = df['sp500_futures'].pct_change()
        
        # Volatility
        if 'vix' in df.columns:
            df['vix_change'] = df['vix'].pct_change()
            df['vix_level'] = df['vix']
        
        # Commodity returns
        for commodity in ['gold', 'copper', 'oil', 'iron_ore_proxy']:
            if commodity in df.columns:
                df[f'{commodity}_return'] = df[commodity].pct_change()
        
        # Currency
        if 'audusd' in df.columns:
            df['audusd_return'] = df['audusd'].pct_change()
        
        # Technical indicators
        df['ema12'] = df['price'].ewm(span=12).mean()
        df['ema26'] = df['price'].ewm(span=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Drop NaN rows from feature engineering
        initial_rows = len(df)
        df.dropna(inplace=True)
        rows_dropped = initial_rows - len(df)
        
        if rows_dropped > 0:
            self._log(f"⚠ Dropped {rows_dropped} rows with NaN values", 'warning')
        
        if df.empty:
            self._log("⚠ No valid data after feature engineering", 'warning')
            return pd.DataFrame()
        
        self._log(f"✓ Feature engineering complete: {len(df)} rows, {len(df.columns)} columns", 'success')
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (exclude target and identifiers)"""
        exclude_cols = ['target', 'daily_return', 'price']
        return [col for col in df.columns if col not in exclude_cols]
    
    def save_feature_names(self, feature_names: List[str]) -> None:
        """Save feature names separately for better model viewing"""
        feature_names_path = self.features_path.replace('features.pkl', 'feature_names.txt')
        try:
            with open(feature_names_path, 'w') as f:
                for i, name in enumerate(feature_names, 1):
                    f.write(f"{i:3d}. {name}\n")
            self._log(f"✓ Feature names saved to {feature_names_path}", 'success')
        except Exception as e:
            self._log(f"⚠ Failed to save feature names: {e}", 'warning')
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train Random Forest model on the provided DataFrame"""
        result = {
            "success": True,
            "train_accuracy": 0,
            "test_accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "feature_importance": None,
            "message": ""
        }
        
        try:
            self.feature_columns = self.get_feature_columns(df)
            X = df[self.feature_columns]
            y = df['target']
            
            # Save feature names for viewer
            self.save_feature_names(self.feature_columns)
            
            # Check if we have enough data
            if len(X) < 100:
                result["success"] = False
                result["message"] = f"Insufficient data: {len(X)} rows (need at least 100)"
                return result
            
            # Train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Initialize and train
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            result["train_accuracy"] = accuracy_score(y_train, y_train_pred)
            result["test_accuracy"] = accuracy_score(y_test, y_test_pred)
            
            # Only calculate additional metrics if we have both classes in test set
            if len(np.unique(y_test)) > 1:
                result["precision"] = precision_score(y_test, y_test_pred, average='binary', zero_division=0)
                result["recall"] = recall_score(y_test, y_test_pred, average='binary', zero_division=0)
                result["f1_score"] = f1_score(y_test, y_test_pred, average='binary', zero_division=0)
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            result["feature_importance"] = importance.head(10).to_dict('records')
            
            # Ensure directory exists before saving
            self._ensure_data_directory()
            
            # Save model and features
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.feature_columns, self.features_path)
            result["message"] = f"Model saved to {self.model_path}"
            
            self._log(f"✓ Model training complete. Test accuracy: {result['test_accuracy']:.3f}", 'success')
            
        except Exception as e:
            result["success"] = False
            result["message"] = str(e)
            self._log(f"✗ Model training failed: {e}", 'error')
        
        return result
    
    def train_with_cv(self, df: pd.DataFrame, cv_folds: int = 5) -> Dict[str, Any]:
        """Train with cross-validation for more robust evaluation"""
        result = self.train(df)
        
        if result["success"] and self.model is not None:
            try:
                X = df[self.feature_columns]
                y = df['target']
                
                # Perform cross-validation
                cv_scores = cross_val_score(self.model, X, y, cv=cv_folds)
                result["cv_mean"] = cv_scores.mean()
                result["cv_std"] = cv_scores.std()
                result["cv_scores"] = cv_scores.tolist()
                
                self._log(f"✓ Cross-validation complete. Mean: {result['cv_mean']:.3f} (±{result['cv_std']:.3f})", 'success')
                
            except Exception as e:
                result["cv_error"] = str(e)
                self._log(f"⚠ Cross-validation failed: {e}", 'warning')
        
        return result
    
    def load_model(self) -> bool:
        """Load pre-trained model if exists"""
        if os.path.exists(self.model_path) and os.path.exists(self.features_path):
            try:
                self.model = joblib.load(self.model_path)
                self.feature_columns = joblib.load(self.features_path)
                self._log(f"✓ Model loaded from {self.model_path}", 'success')
                return True
            except Exception as e:
                self._log(f"✗ Failed to load model: {e}", 'error')
                return False
        return False
    
    def save_model_with_version(self, version: str = None) -> str:
        """Save model with version tag"""
        if self.model is None:
            self._log("⚠ No model to save", 'warning')
            return ""
        
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save with version in filename
        versioned_model_path = self.model_path.replace('.pkl', f'_{version}.pkl')
        versioned_features_path = self.features_path.replace('.pkl', f'_{version}.pkl')
        
        try:
            joblib.dump(self.model, versioned_model_path)
            joblib.dump(self.feature_columns, versioned_features_path)
            self._log(f"✓ Model saved with version: {version}", 'success')
            return version
        except Exception as e:
            self._log(f"✗ Failed to save versioned model: {e}", 'error')
            return ""
    
    def predict(self, df: pd.DataFrame) -> Optional[float]:
        """
        Predict probability for the latest row
        Returns probability of positive return
        """
        if self.model is None or self.feature_columns is None:
            if not self.load_model():
                self._log("⚠ No trained model available. Please train first.", 'warning')
                return None
        
        try:
            # Check if all required features are present
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                self._log(f"⚠ Missing features for prediction: {missing_features}", 'warning')
                return None
            
            latest = df.iloc[[-1]][self.feature_columns]
            
            # Check for NaN values
            if latest.isna().any().any():
                self._log("⚠ Latest data contains NaN values", 'warning')
                return None
            
            prob = self.model.predict_proba(latest)[0][1]
            return prob
            
        except Exception as e:
            self._log(f"✗ Prediction failed: {e}", 'error')
            return None
    
    def get_feature_importance_data(self, top_n: int = 20) -> Dict:
        """Get feature importance data formatted for visualization"""
        if self.model is None:
            if not self.load_model():
                return {}
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return {
            'features': importance['feature'].tolist(),
            'importance': importance['importance'].tolist()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            "model_loaded": self.model is not None,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "model_path": self.model_path,
            "features_path": self.features_path,
            "config": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state
            }
        }
        
        # Add model parameters if model is loaded
        if self.model is not None:
            info["model_params"] = {
                "n_classes": len(self.model.classes_),
                "n_features": self.model.n_features_in_,
                "tree_count": len(self.model.estimators_)
            }
            
            # Add feature importance summary
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                info["feature_importance"] = {
                    "mean": float(np.mean(importances)),
                    "std": float(np.std(importances)),
                    "max": float(np.max(importances)),
                    "min": float(np.min(importances))
                }
        
        # Add file info if model exists
        if os.path.exists(self.model_path):
            info["model_file_size"] = os.path.getsize(self.model_path)
            info["model_modified"] = datetime.fromtimestamp(
                os.path.getmtime(self.model_path)
            ).strftime('%Y-%m-%d %H:%M:%S')
        
        return info
