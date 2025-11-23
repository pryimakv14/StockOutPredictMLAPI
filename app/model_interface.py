from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for forecasting models."""
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train the model on the provided data."""
        pass
    
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for the provided dates."""
        pass
    
    @abstractmethod
    def make_future_dataframe(self, periods: int) -> pd.DataFrame:
        """Create a future dataframe for forecasting."""
        pass
    
    @property
    @abstractmethod
    def history(self) -> pd.DataFrame:
        """Return the training data history."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model implementation."""
        pass

