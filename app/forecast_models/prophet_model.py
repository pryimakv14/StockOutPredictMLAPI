import pandas as pd
import logging
from typing import Dict, Any, Optional
from prophet import Prophet
from app.forecast_models.base import BaseForecastModel

logger = logging.getLogger(__name__)

class ProphetForecastModel(BaseForecastModel):
    """Prophet implementation of the forecast model interface."""
    
    def __init__(self, **kwargs):
        """Initialize Prophet with hyperparameters."""
        self.model = Prophet(**kwargs)
        self.history: Optional[pd.DataFrame] = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """Train the Prophet model."""
        self.model.fit(df)
        self.history = df.copy()
    
    def predict(self, periods: int, last_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Generate future predictions."""
        if self.history is None:
            raise ValueError("Model must be fitted before prediction")
        
        future_df = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future_df)
        
        # Filter to only future dates
        if last_date is None:
            last_date = self.history['ds'].max()
        
        forecast['ds'] = pd.to_datetime(forecast['ds']).dt.tz_localize(None)
        future_forecast = forecast[forecast['ds'] > last_date].copy()
        
        return future_forecast[['ds', 'yhat']].copy()
    
    def get_last_training_date(self) -> pd.Timestamp:
        """Get the last date from training data."""
        if self.history is None or self.history.empty:
            raise ValueError("Model history is not available")
        return self.history['ds'].max().normalize().tz_localize(None)
    
    def get_model_name(self) -> str:
        return "Prophet"
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying Prophet model."""
        return getattr(self.model, name)

