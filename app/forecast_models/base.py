from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class BaseForecastModel(ABC):
    """Abstract base class for all forecasting models."""
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """
        Train the model on historical data.
        Args:
            df: DataFrame with columns 'ds' (datetime) and 'y' (target)
        """
        pass
    
    @abstractmethod
    def predict(self, periods: int, last_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Generate future predictions.
        Args:
            periods: Number of periods to forecast
            last_date: Last date in training data (optional, for date generation)
        Returns:
            DataFrame with columns 'ds' (datetime) and 'yhat' (predicted values)
        """
        pass
    
    @abstractmethod
    def get_last_training_date(self) -> pd.Timestamp:
        """Get the last date from training data."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the model type."""
        pass

