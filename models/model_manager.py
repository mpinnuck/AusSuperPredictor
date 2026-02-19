"""
Model Manager - single responsibility for ML model operations
Now saves models to the data folder
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from typing import Dict, Any, Optional, Tuple

class ModelManager:
    """Manages all ML model operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_columns = None
        self.model_path = config['model']['save_path']
        self.features_path = config['model']['features_save_path']
        
        # Ensure data directory exists
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        data_dirs = [
            os.path.dirname(self.model_path),
            os.path.dirname(self.features_path)
        ]
        for data_dir in data_dirs:
            if data_dir and not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
                print(f"Created directory: {data_dir}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create predictive features from raw data"""
        if df.empty:
            return pd.DataFrame()
            
        df = df.copy()
        
        # Target: 1 if next day positive, else 0
        df['target'] = (df['daily_return'].shift(-1) > 0).astype(int)
        
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
        
        df.dropna(inplace=True)
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature columns (exclude target and identifiers)"""
        exclude_cols = ['target', 'daily_return', 'price']
        return [col for col in df.columns if col not in exclude_cols]
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train Random Forest model on the provided DataFrame"""
        result = {
            "success": True,
            "train_accuracy": 0,
            "test_accuracy": 0,
            "feature_importance": None,
            "message": ""
        }
        
        try:
            self.feature_columns = self.get_feature_columns(df)
            X = df[self.feature_columns]
            y = df['target']
            
            # Train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Initialize and train
            self.model = RandomForestClassifier(
                n_estimators=self.config['model']['n_estimators'],
                max_depth=self.config['model']['max_depth'],
                random_state=self.config['model']['random_state'],
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_acc = accuracy_score(y_train, self.model.predict(X_train))
            test_acc = accuracy_score(y_test, self.model.predict(X_test))
            
            result["train_accuracy"] = train_acc
            result["test_accuracy"] = test_acc
            
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
            
        except Exception as e:
            result["success"] = False
            result["message"] = str(e)
        
        return result
    
    def load_model(self) -> bool:
        """Load pre-trained model if exists"""
        if os.path.exists(self.model_path) and os.path.exists(self.features_path):
            self.model = joblib.load(self.model_path)
            self.feature_columns = joblib.load(self.features_path)
            return True
        return False
    
    def predict(self, df: pd.DataFrame) -> Optional[float]:
        """
        Predict probability for the latest row
        Returns probability of positive return
        """
        if self.model is None or self.feature_columns is None:
            if not self.load_model():
                return None
        
        try:
            latest = df.iloc[[-1]][self.feature_columns]
            prob = self.model.predict_proba(latest)[0][1]
            return prob
        except Exception:
            return None
