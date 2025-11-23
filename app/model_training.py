import pandas as pd
from typing import Dict, Any
from app.model_factory import get_training_function

def perform_hyperparameter_tuning(df: pd.DataFrame, horizon_days: int = 30) -> Dict[str, Any]:
    training_func = get_training_function()
    return training_func(df, horizon_days)

