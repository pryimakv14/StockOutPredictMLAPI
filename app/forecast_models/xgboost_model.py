import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from xgboost import XGBRegressor
from app.forecast_models.base import BaseForecastModel

logger = logging.getLogger(__name__)

class XGBoostForecastModel(BaseForecastModel):
    """XGBoost implementation of the forecast model interface."""
    
    def __init__(self, **kwargs):
        """Initialize XGBoost with hyperparameters."""
        # Extract XGBoost-specific params, use defaults for Prophet-like params
        xgb_params = {
            'n_estimators': kwargs.pop('n_estimators', 100),
            'max_depth': kwargs.pop('max_depth', 6),
            'learning_rate': kwargs.pop('learning_rate', 0.1),
            'subsample': kwargs.pop('subsample', 0.8),
            'colsample_bytree': kwargs.pop('colsample_bytree', 0.8),
            **kwargs  # Any remaining kwargs
        }
        self.model = XGBRegressor(**xgb_params)
        self.history: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[list] = None
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime column."""
        features = df.copy()
        features['ds'] = pd.to_datetime(features['ds'])
        features['year'] = features['ds'].dt.year
        features['month'] = features['ds'].dt.month
        features['day'] = features['ds'].dt.day
        features['dayofweek'] = features['ds'].dt.dayofweek
        features['dayofyear'] = features['ds'].dt.dayofyear
        # Week of year (simple calculation)
        features['week'] = (features['dayofyear'] - 1) // 7 + 1
        features['quarter'] = features['ds'].dt.quarter
        
        # Cyclical encoding for seasonality
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['dayofweek_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)
        
        # Lag features (if enough data)
        if len(features) > 7:
            features['lag_7'] = features['y'].shift(7)
            features['lag_30'] = features['y'].shift(30)
        
        # Rolling averages
        if len(features) > 7:
            features['rolling_mean_7'] = features['y'].rolling(window=7, min_periods=1).mean()
            features['rolling_mean_30'] = features['y'].rolling(window=30, min_periods=1).mean()
        
        return features
    
    def fit(self, df: pd.DataFrame) -> None:
        """Train the XGBoost model."""
        self.history = df.copy()
        features_df = self._create_features(df)
        
        # Select feature columns (exclude 'ds' and 'y')
        self.feature_columns = [col for col in features_df.columns 
                               if col not in ['ds', 'y']]
        
        X = features_df[self.feature_columns].fillna(0)
        y = features_df['y'].fillna(0)
        
        self.model.fit(X, y)
    
    def predict(self, periods: int, last_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Generate future predictions using recursive approach for lag features."""
        if self.history is None:
            raise ValueError("Model must be fitted before prediction")
        
        if last_date is None:
            last_date = self.history['ds'].max()
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        # Start with history for recursive prediction
        working_df = self.history.copy()
        predictions = []
        
        # Recursive prediction: predict one step at a time to properly handle lags
        for i, future_date in enumerate(future_dates):
            # Create features for this future date
            temp_df = pd.concat([
                working_df,
                pd.DataFrame({'ds': [future_date], 'y': [0]})  # Placeholder y
            ], ignore_index=True)
            
            features_df = self._create_features(temp_df)
            # Get features for the last row (the future date)
            future_features = features_df.tail(1)[self.feature_columns].fillna(0)
            
            # Predict
            pred = self.model.predict(future_features)[0]
            pred = max(0, pred)  # Ensure non-negative
            predictions.append(pred)
            
            # Add predicted value to working dataframe for next iteration's lags
            working_df = pd.concat([
                working_df,
                pd.DataFrame({'ds': [future_date], 'y': [pred]})
            ], ignore_index=True)
        
        result = pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions
        })
        
        return result
    
    def get_last_training_date(self) -> pd.Timestamp:
        """Get the last date from training data."""
        if self.history is None or self.history.empty:
            raise ValueError("Model history is not available")
        return self.history['ds'].max().normalize().tz_localize(None)
    
    def get_model_name(self) -> str:
        return "XGBoost"

