from typing import Dict, Any
from app.forecast_models.base import BaseForecastModel
from app.forecast_models.prophet_model import ProphetForecastModel
from app.forecast_models.xgboost_model import XGBoostForecastModel

def create_model(model_type: str = "prophet", **kwargs) -> BaseForecastModel:
    """
    Factory function to create forecast models.
    
    Args:
        model_type: Type of model ('prophet' or 'xgboost')
        **kwargs: Model-specific hyperparameters
    
    Returns:
        Instance of BaseForecastModel
    """
    model_type_lower = model_type.lower()
    
    if model_type_lower == "prophet":
        return ProphetForecastModel(**kwargs)
    elif model_type_lower == "xgboost":
        return XGBoostForecastModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'prophet', 'xgboost'")

