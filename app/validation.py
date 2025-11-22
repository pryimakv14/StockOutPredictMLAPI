import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from fastapi import HTTPException, status
from sklearn.metrics import mean_absolute_error, mean_squared_error
from app.data_handler import get_product_data
from app.forecast_models.factory import create_model

logger = logging.getLogger(__name__)

async def run_period_accuracy_validation(
        sku: str,
        test_period_days: int = 30,
        model_type: str = "prophet",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = 'additive',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        **kwargs
) -> Dict[str, Any]:

    logger.info(f"Running PERIOD accuracy validation for SKU: {sku} with test period: {test_period_days} days")

    if test_period_days < 30:
        msg = "Test period must be at least 30 days for this validation."
        logger.error(msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    df_daily = get_product_data(sku)
    if df_daily is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No usable historical data found for SKU {sku}.")

    df_daily['ds'] = pd.to_datetime(df_daily['ds']).dt.tz_localize(None)
    df_daily = df_daily.sort_values('ds')

    try:
        max_historical_date = df_daily['ds'].max()
        cutoff_date = max_historical_date - pd.Timedelta(days=test_period_days)
        train_df = df_daily[df_daily['ds'] <= cutoff_date].copy()
        test_df = df_daily[df_daily['ds'] > cutoff_date].copy()

        if train_df.empty:
            raise ValueError(f"No training data available. Cutoff date: {cutoff_date.date()}")
        if len(test_df) < 1:
            raise ValueError(
                f"No test data available. Cutoff date: {cutoff_date.date()}, "
                f"Max date in data: {max_historical_date.date()}, "
                f"Requested test period: {test_period_days} days"
            )
        if len(train_df) < 14:
            raise ValueError(
                f"Not enough training data: {len(train_df)} days (needs >= 14)"
            )
        
        logger.info(f"Split data: Train until {cutoff_date.date()} ({len(train_df)} days), "
                     f"Test from {test_df['ds'].min().date()} to {test_df['ds'].max().date()} ({len(test_df)} days).")

    except Exception as e:
        logger.error(f"Error splitting data or calculating actuals for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Failed to split data/calculate actuals: {e}")

    try:
        # Build model parameters based on model type
        model_params = {}
        if model_type.lower() == "prophet":
            model_params = {
                'changepoint_prior_scale': changepoint_prior_scale,
                'seasonality_prior_scale': seasonality_prior_scale,
                'holidays_prior_scale': holidays_prior_scale,
                'seasonality_mode': seasonality_mode,
                'yearly_seasonality': yearly_seasonality,
                'weekly_seasonality': weekly_seasonality,
                'daily_seasonality': daily_seasonality
            }
        else:  # xgboost or other models
            # Use any additional kwargs for XGBoost parameters
            model_params = kwargs
        
        model = create_model(model_type=model_type, **model_params)
        model.fit(train_df)
        
        params_used_log = str(model_params)
        logger.info(f"Trained validation model ({model_type}) for SKU {sku} with params: {params_used_log}")
                     
    except Exception as e:
        logger.error(f"Error training temporary validation model for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Validation failed during training: {e}")

    try:
        # Use the abstract interface for prediction
        last_train_date = train_df['ds'].max()
        forecast = model.predict(periods=len(test_df), last_date=last_train_date)
        forecast['ds'] = pd.to_datetime(forecast['ds']).dt.tz_localize(None)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)

        comparison_df = test_df[['ds', 'y']].copy().sort_values('ds')
        comparison_df = comparison_df.merge(
            forecast[['ds', 'yhat']], 
            on='ds', 
            how='inner'
        )
        
        if comparison_df.empty:
            raise ValueError("No matching dates between test data and forecast predictions")
        
        predicted_list = [float(round(p, 2)) for p in comparison_df['yhat']]
        actual_list = [float(round(a, 2)) for a in comparison_df['y']]
        
        if len(actual_list) == 0 or len(predicted_list) == 0:
            raise ValueError("No data available for metrics calculation")
        
        actual_array = np.array(actual_list)
        predicted_array = np.array(predicted_list)
        
        mae = mean_absolute_error(actual_array, predicted_array)
        rmse = np.sqrt(mean_squared_error(actual_array, predicted_array))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            actual_array_no_zeros = np.where(actual_array == 0, 1e-9, actual_array)
            mape = np.mean(np.abs((actual_array - predicted_array) / actual_array_no_zeros)) * 100
            if np.isinf(mape) or np.isnan(mape):
                mape = None
            else:
                mape = float(round(mape, 2))
        
        mbe = np.mean(predicted_array - actual_array)
        mbe = float(round(mbe, 2))
        
        ss_res = np.sum((actual_array - predicted_array) ** 2)
        ss_tot = np.sum((actual_array - np.mean(actual_array)) ** 2)
        if ss_tot != 0:
            r_squared = 1 - (ss_res / ss_tot)
            r_squared = float(round(r_squared, 4))
        else:
            r_squared = None
        
        logger.info(f"Generated daily results for SKU {sku}: {len(predicted_list)} days. "
                     f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape}%, MBE={mbe:.2f}")

    except Exception as e:
        logger.error(f"Error predicting for validation (SKU {sku}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Validation failed during prediction: {e}")

    return {
        "sku": sku,
        "model_type": model_type,
        "validation_period_info": f"Training data until {cutoff_date.date()}, "
                                  f"Testing on historical data from {test_df['ds'].min().date()} to {test_df['ds'].max().date()} ({len(test_df)} days).",
        "parameters_used": model_params,
        "predicted": predicted_list,
        "actual": actual_list,
        "metrics": {
            "mae": float(round(mae, 2)),
            "rmse": float(round(rmse, 2)),
            "mape": mape,
            "mbe": mbe,
            "r_squared": r_squared
        }
    }

