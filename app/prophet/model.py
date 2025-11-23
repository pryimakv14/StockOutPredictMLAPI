import pandas as pd
from typing import Dict, Any
from prophet import Prophet
from app.model_interface import BaseModel

class ProphetModel(BaseModel):
    """Prophet implementation of the BaseModel interface."""
    
    def __init__(self, **kwargs):
        self._prophet = Prophet(**kwargs)
        self._history = None
    
    def fit(self, df: pd.DataFrame) -> None:
        self._prophet.fit(df)
        self._history = df
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._prophet.predict(df)
    
    def make_future_dataframe(self, periods: int) -> pd.DataFrame:
        return self._prophet.make_future_dataframe(periods=periods)
    
    @property
    def history(self) -> pd.DataFrame:
        if self._history is not None:
            return self._history

        prophet_history = getattr(self._prophet, 'history', None)
        if prophet_history is not None:
            return prophet_history
        return pd.DataFrame()
    
    @property
    def model_name(self) -> str:
        return "Prophet"
    
    @staticmethod
    def get_model_name() -> str:
        return "Prophet"
    
    def __getattr__(self, name):
        return getattr(self._prophet, name)

