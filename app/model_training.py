import pandas as pd
import numpy as np
import logging
import random
from typing import Dict, Any
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error
from app.config import N_RANDOM_SEARCH_ITERATIONS
from app.forecast_models.factory import create_model

logger = logging.getLogger(__name__)


def perform_hyperparameter_tuning(df: pd.DataFrame, model_type: str = "prophet", horizon_days: int = 30) -> Dict[str, Any]:
    model_type_lower = model_type.lower()
    
    if model_type_lower == "prophet":
        param_grid = {
            'changepoint_prior_scale': [0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0, 20.0, 30.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'holidays_prior_scale': [0.1, 1.0, 5.0, 10.0, 20.0, 30.0],
            'yearly_seasonality': [True, False],
            'weekly_seasonality': [True, False],
            'daily_seasonality': [True, False]
        }
    elif model_type_lower == "xgboost":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'prophet', 'xgboost'")
    
    best_params = {}
    best_mae = float('inf')

    horizon_str = f'{horizon_days} days'
    initial_days = max(horizon_days * 3, 90)
    initial_str = f'{initial_days} days'
    period_str = f'{horizon_days // 2} days'

    if len(df) < (initial_days + (horizon_days // 2)):
        logger.warning(f"Skipping hyperparameter tuning for SKU. "
                        f"Not enough data ({len(df)} rows) for CV with initial={initial_str}.")
        return {
            "status": "skipped_tuning",
            "message": f"Skipped tuning due to insufficient data. Using default parameters.",
            "best_parameters": {} 
        }

    logger.info(f"Starting RANDOM SEARCH tuning with {N_RANDOM_SEARCH_ITERATIONS} combinations...")

    tested_params_list = []
    for _ in range(N_RANDOM_SEARCH_ITERATIONS):
        params = {}
        for key, values in param_grid.items():
            params[key] = random.choice(values)
        
        if params not in tested_params_list:
             tested_params_list.append(params)
    
    logger.info(f"Generated {len(tested_params_list)} unique random combinations for tuning.")

    for params in tested_params_list:
        try:
            if model_type_lower == "prophet":
                # Use Prophet's built-in cross-validation
                m = Prophet(**params).fit(df)
                df_cv = cross_validation(m, initial=initial_str, period=period_str, horizon=horizon_str,
                                         parallel="processes")
                df_p = performance_metrics(df_cv, rolling_window=1)
                mae = df_p['mae'].values[0]
            else:  # xgboost
                # Simple time-series cross-validation for XGBoost
                max_historical_date = df['ds'].max()
                cutoff_date = max_historical_date - pd.Timedelta(days=horizon_days)
                train_df = df[df['ds'] <= cutoff_date].copy()
                test_df = df[df['ds'] > cutoff_date].copy()
                
                if len(train_df) < 14 or len(test_df) < 1:
                    continue
                
                m = create_model(model_type=model_type, **params)
                m.fit(train_df)
                
                # Predict on test period
                test_predictions = m.predict(periods=len(test_df), last_date=train_df['ds'].max())
                test_predictions = test_predictions.merge(
                    test_df[['ds', 'y']], on='ds', how='inner'
                )
                
                if len(test_predictions) == 0:
                    continue
                
                mae = mean_absolute_error(test_predictions['y'], test_predictions['yhat'])

            if mae < best_mae:
                best_mae = mae
                best_params = params
        except Exception as e:
            logger.warning(f"Error during CV with params {params}: {e}")
            continue

    if not best_params:
        logger.error("Hyperparameter tuning failed for all combinations. Using defaults.")
        best_params = {}
        status_msg = "tuning_failed"
    else:
        logger.info(f"Tuning complete. Best MAE: {best_mae}. Best params: {best_params}")
        status_msg = "tuning_success"

    return {
        "status": status_msg,
        "best_parameters": best_params,
        "best_cross_validation_mae": best_mae
    }

