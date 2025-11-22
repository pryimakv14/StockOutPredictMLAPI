import pandas as pd
import logging
import random
from typing import Dict, Any
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from app.config import N_RANDOM_SEARCH_ITERATIONS

logger = logging.getLogger(__name__)


def perform_hyperparameter_tuning(df: pd.DataFrame, horizon_days: int = 30) -> Dict[str, Any]:
    param_grid = {
        'changepoint_prior_scale': [0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0, 20.0, 30.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'holidays_prior_scale': [0.1, 1.0, 5.0, 10.0, 20.0, 30.0],
        'yearly_seasonality': [True, False],
        'weekly_seasonality': [True, False],
        'daily_seasonality': [True, False]
    }
    
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
            m = Prophet(**params).fit(df)
            df_cv = cross_validation(m, initial=initial_str, period=period_str, horizon=horizon_str,
                                     parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            mae = df_p['mae'].values[0]

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

