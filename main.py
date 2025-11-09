import pandas as pd
from fastapi import FastAPI, HTTPException, Path, Query, status, Body, UploadFile, File
from typing import Dict, Any, List, Optional
from datetime import datetime
import joblib
import os
import logging
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import itertools
import random
from pydantic import BaseModel, Field
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = os.getenv("MODEL_DIR", "trained_models")
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "data/sales_history.csv")

N_RANDOM_SEARCH_ITERATIONS = 60

os.makedirs(MODEL_DIR, exist_ok=True)
data_dir_for_upload = os.path.dirname(DATA_FILE_PATH)
if data_dir_for_upload:
    os.makedirs(data_dir_for_upload, exist_ok=True)

app = FastAPI(
    title="Magento Sales Predictor API",
    description="API for training ML models (with hyperparameter tuning), "
                "predicting future sales, and validating model accuracy.",
    version="0.3.0"
)


def get_product_data(sku: str) -> pd.DataFrame | None:
    """Loads CSV, filters by SKU, aggregates daily, and prepares for Prophet."""
    try:
        logging.info(f"Attempting to read data file: {DATA_FILE_PATH}")
        if not os.path.exists(DATA_FILE_PATH):
            logging.error(f"Data file not found at {DATA_FILE_PATH}")
            return None
        df_all = pd.read_csv(DATA_FILE_PATH, names=['sku', 'qty_ordered', 'created_at'], header=None)
        df_all['sku'] = df_all['sku'].astype(str).str.strip().str.lower()
        logging.info(f"Successfully read data file. Total rows: {len(df_all)}")
    except pd.errors.EmptyDataError:
        logging.warning(f"Data file {DATA_FILE_PATH} is empty.")
        return None
    except Exception as e:
        logging.error(f"Error reading CSV file {DATA_FILE_PATH}: {e}", exc_info=True)
        return None

    sku_cleaned = str(sku).strip().lower()
    df_product_raw = df_all[df_all['sku'] == sku_cleaned].copy()

    if df_product_raw.empty:
        logging.warning(f"No data found for SKU: {sku} in the CSV.")
        return None

    try:
        df_product_raw['created_at'] = pd.to_datetime(df_product_raw['created_at']).dt.tz_localize(None)
        df_daily = df_product_raw.set_index('created_at').resample('D')['qty_ordered'].sum().reset_index()
        df_daily.rename(columns={"created_at": "ds", "qty_ordered": "y"}, inplace=True)
        df_daily = df_daily.sort_values('ds')
        logging.info(f"Aggregated data for SKU {sku}. Found {len(df_daily)} daily records.")
    except Exception as e:
        logging.error(f"Error processing data for SKU {sku}: {e}", exc_info=True)
        return None

    if len(df_daily) < 14:
        logging.warning(f"Not enough aggregated data points ({len(df_daily)}) for SKU: {sku}. Need at least 14.")
        return None

    return df_daily


def perform_hyperparameter_tuning(df: pd.DataFrame, horizon_days: int = 30) -> Dict[str, Any]:
    """
    Performs RANDOM SEARCH hyperparameter tuning using Prophet's cross-validation.
    """
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
        logging.warning(f"Skipping hyperparameter tuning for SKU. "
                        f"Not enough data ({len(df)} rows) for CV with initial={initial_str}.")
        return {
            "status": "skipped_tuning",
            "message": f"Skipped tuning due to insufficient data. Using default parameters.",
            "best_parameters": {} 
        }

    logging.info(f"Starting RANDOM SEARCH tuning with {N_RANDOM_SEARCH_ITERATIONS} combinations...")

    tested_params_list = []
    for _ in range(N_RANDOM_SEARCH_ITERATIONS):
        params = {}
        for key, values in param_grid.items():
            params[key] = random.choice(values)
        
        if params not in tested_params_list:
             tested_params_list.append(params)
    
    logging.info(f"Generated {len(tested_params_list)} unique random combinations for tuning.")

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
            logging.warning(f"Error during CV with params {params}: {e}")
            continue

    if not best_params:
        logging.error("Hyperparameter tuning failed for all combinations. Using defaults.")
        best_params = {}
        status_msg = "tuning_failed"
    else:
        logging.info(f"Tuning complete. Best MAE: {best_mae}. Best params: {best_params}")
        status_msg = "tuning_success"

    return {
        "status": status_msg,
        "best_parameters": best_params,
        "best_cross_validation_mae": best_mae
    }


def predict_stock_duration(model: Prophet, current_stock: int, forecast_horizon_days: int = 365) -> Dict[str, Any]:
    """
    Predicts the number of days until the current stock runs out
    based on the model's forecast.
    """
    try:
        if model.history is None or model.history.empty:
            logging.error("Model history is not available. Cannot determine last training date.")
            return {
                "days_of_stock_remaining": "Error", 
                "error_message": "Model history is empty.",
                "predicted_out_of_stock_date": None,
                "last_training_date": None
            }

        last_training_date = model.history['ds'].max().normalize().tz_localize(None)
        
        future_df = model.make_future_dataframe(periods=forecast_horizon_days)
        forecast = model.predict(future_df)
        
        forecast['ds'] = forecast['ds'].dt.tz_localize(None)
        future_forecast = forecast[forecast['ds'] > last_training_date].copy()
        
        future_forecast['yhat'] = future_forecast['yhat'].clip(lower=0)
        
        if future_forecast.empty:
            logging.warning(f"No future forecast data could be generated beyond {last_training_date.date()}.")
            return {
                "days_of_stock_remaining": "Error", 
                "error_message": "No future forecast data could be generated.",
                "predicted_out_of_stock_date": None,
                "last_training_date": last_training_date.strftime('%Y-%m-%d')
            }

        remaining_stock = float(current_stock)
        days_counted = 0
        
        for _, row in future_forecast.iterrows():
            days_counted += 1
            predicted_sales_this_day = row['yhat']
            
            remaining_stock -= predicted_sales_this_day
            
            if remaining_stock <= 0:
                logging.info(f"Stock for SKU depleted on (relative) day {days_counted}. "
                             f"Predicted OOS date: {row['ds'].date()}. "
                             f"Remaining stock: {remaining_stock:.2f}")
                return {
                    "days_of_stock_remaining": days_counted,
                    "predicted_out_of_stock_date": row['ds'].strftime('%Y-%m-%d'),
                    "last_training_date": last_training_date.strftime('%Y-%m-%d'),
                    "error_message": None
                }
        
        logging.info(f"Stock did not run out within the {forecast_horizon_days} day horizon.")
        return {
            "days_of_stock_remaining": forecast_horizon_days, 
            "predicted_out_of_stock_date": f"More than {forecast_horizon_days} days",
            "last_training_date": last_training_date.strftime('%Y-%m-%d'),
            "error_message": f"Stock did not run out within {forecast_horizon_days} days."
        }

    except Exception as e:
        logging.error(f"Error during stock duration prediction: {e}", exc_info=True)
        return {
            "days_of_stock_remaining": "Error", 
            "error_message": str(e),
            "predicted_out_of_stock_date": None,
            "last_training_date": None
        }


class ValidationRequest(BaseModel):
    test_period_days: int = Field(30, ge=30, description="Number of days for the test set (must be >= 30)")
    changepoint_prior_scale: float = Field(0.05, description="Flexibility of the trend")
    seasonality_prior_scale: float = Field(10.0, description="Strength of seasonality")
    holidays_prior_scale: float = Field(10.0, description="Strength of holiday effects")
    seasonality_mode: str = Field('additive', description="Seasonality mode ('additive' or 'multiplicative')")
    yearly_seasonality: bool = Field(True, description="Enable yearly seasonality")
    weekly_seasonality: bool = Field(True, description="Enable weekly seasonality")
    daily_seasonality: bool = Field(False, description="Enable daily seasonality (default False for daily data)")


class OptionalHyperparameters(BaseModel):
    changepoint_prior_scale: Optional[float] = Field(None, description="Flexibility of the trend")
    seasonality_prior_scale: Optional[float] = Field(None, description="Strength of seasonality")
    holidays_prior_scale: Optional[float] = Field(None, description="Strength of holiday effects")
    seasonality_mode: Optional[str] = Field(None, description="Seasonality mode ('additive' or 'multiplicative')")
    yearly_seasonality: Optional[bool] = Field(None, description="Enable yearly seasonality")
    weekly_seasonality: Optional[bool] = Field(None, description="Enable weekly seasonality")
    daily_seasonality: Optional[bool] = Field(None, description="Enable daily seasonality")


async def run_period_accuracy_validation(
        sku: str,
        test_period_days: int = 30,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = 'additive',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False
) -> Dict[str, Any]:

    logging.info(f"Running PERIOD accuracy validation for SKU: {sku} with test period: {test_period_days} days")

    if test_period_days < 30:
        msg = "Test period must be at least 30 days for this validation."
        logging.error(msg)
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
        
        logging.info(f"Split data: Train until {cutoff_date.date()} ({len(train_df)} days), "
                     f"Test from {test_df['ds'].min().date()} to {test_df['ds'].max().date()} ({len(test_df)} days).")

    except Exception as e:
        logging.error(f"Error splitting data or calculating actuals for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Failed to split data/calculate actuals: {e}")

    try:
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        
        model.fit(train_df)
        params_used_log = (
            f"changepoint_prior_scale={changepoint_prior_scale}, "
            f"seasonality_prior_scale={seasonality_prior_scale}, "
            f"holidays_prior_scale={holidays_prior_scale}, "
            f"seasonality_mode={seasonality_mode}, "
            f"yearly={yearly_seasonality}, weekly={weekly_seasonality}, daily={daily_seasonality}"
        )
        logging.info(f"Trained validation model for SKU {sku} with params: {params_used_log}")
                     
    except Exception as e:
        logging.error(f"Error training temporary validation model for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Validation failed during training: {e}")

    try:
        future_df_test_period = test_df[['ds']].copy().sort_values('ds')
        forecast = model.predict(future_df_test_period)
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
        
        logging.info(f"Generated daily results for SKU {sku}: {len(predicted_list)} days. "
                     f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape}%, MBE={mbe:.2f}")

    except Exception as e:
        logging.error(f"Error predicting for validation (SKU {sku}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Validation failed during prediction: {e}")

    return {
        "sku": sku,
        "validation_period_info": f"Training data until {cutoff_date.date()}, "
                                  f"Testing on historical data from {test_df['ds'].min().date()} to {test_df['ds'].max().date()} ({len(test_df)} days).",
        "parameters_used": {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "holidays_prior_scale": holidays_prior_scale,
            "seasonality_mode": seasonality_mode,
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality
        },
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


@app.post('/train/{sku}', status_code=status.HTTP_200_OK, summary="Train Model for SKU")
async def train_model_endpoint(
        sku: str = Path(..., description="The product SKU to train the model for."),
        params: Optional[OptionalHyperparameters] = Body(None, description="Optional: Specific hyperparameters to use. If omitted, random search tuning will be performed.")
):
    """
    Triggers model training for a specific SKU.
    - If a request body with hyperparameters is provided, it uses them directly (skips tuning).
    - If no request body (or null) is provided, it performs random search hyperparameter tuning.
    """
    logging.info(f"Received training request for SKU: {sku}")
    df_daily = get_product_data(sku)
    if df_daily is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No usable training data found for SKU {sku} at '{DATA_FILE_PATH}'. "
                   f"Ensure CSV exists and has enough data."
        )

    training_info = {}
    best_params = {}

    if params is None:
        logging.info(f"No specific parameters provided for SKU {sku}. Running random search tuning...")
        tuning_results_dict = perform_hyperparameter_tuning(df_daily, horizon_days=30)
        
        best_params = tuning_results_dict.get("best_parameters", {})
        training_info = tuning_results_dict
        logging.info(f"Tuning complete for SKU {sku}. Best params found: {best_params}")
    
    else:
        logging.info(f"Specific parameters provided for SKU {sku}. Skipping tuning.")
        
        best_params = params.model_dump(exclude_unset=True, exclude_none=True)
        
        training_info = {
            "status": "skipped_tuning_params_provided",
            "parameters_used": best_params,
            "message": "Trained using parameters provided in the request."
        }
        logging.info(f"Training SKU {sku} with provided params: {best_params}")

    try:
        prophet_args = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False 
        }
        prophet_args.update(best_params)

        model = Prophet(**prophet_args)
        
        model.fit(df_daily)
        model_path = os.path.join(MODEL_DIR, f"{sku}.joblib")
        joblib.dump(model, model_path)
        logging.info(f"Model for SKU {sku} saved successfully to {model_path}")
    except Exception as e:
        logging.error(f"Error during model training or saving for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Model training failed: {e}")

    return {
        "status": "success",
        "sku": sku,
        "message": "Model trained successfully.",
        "model_saved_at": model_path,
        "training_info": training_info
    }


@app.get('/predict/{sku}', summary="Predict Stock Duration (Days Left)")
async def predict_sales_endpoint(
        sku: str = Path(..., description="The product SKU to predict sales for."),
        current_stock: int = Query(..., description="The current number of items in stock.", ge=0),
        forecast_horizon_days: int = Query(365, description="How many days into the future to look for stock depletion.", ge=30, le=1095)
):
    """
    Predicts the number of days until the stock for a given SKU runs out,
    based on the provided `current_stock`.
    
    The prediction is relative to the *end of the training data*
    (i.e., the last date in `sales_history.csv` for this SKU).
    """
    logging.info(f"Received stock duration prediction request for SKU: {sku} with stock: {current_stock}")
    model_path = os.path.join(MODEL_DIR, f"{sku}.joblib")

    if not os.path.exists(model_path):
        logging.warning(f"Prediction failed: No trained model found for SKU {sku} at {model_path}.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No trained model found for SKU {sku}.")

    try:
        model = joblib.load(model_path)
        logging.info(f"Loaded trained model for SKU: {sku} from {model_path}")

        duration_results = predict_stock_duration(
            model=model, 
            current_stock=current_stock, 
            forecast_horizon_days=forecast_horizon_days
        )

    except Exception as e:
        logging.error(f"Prediction failed for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to load model or run prediction: {e}")

    return {
        "sku": sku,
        "current_stock_provided": current_stock,
        "forecast_horizon_checked": forecast_horizon_days,
        "prediction_engine": "Prophet",
        **duration_results
    }


@app.post('/validate-period-accuracy/{sku}',
          summary="Validate Model Accuracy (Daily Results)",
          response_model=Dict[str, Any])
async def validate_period_accuracy_endpoint(
        sku: str = Path(..., description="The product SKU to validate accuracy for."),
        request: ValidationRequest = Body(..., description="Validation parameters including test period days and hyperparameters")
) -> Dict[str, Any]:
    logging.info(f"Received validation request for SKU: {sku} with parameters: {request}")
    validation_results = await run_period_accuracy_validation(
        sku=sku,
        test_period_days=request.test_period_days,
        changepoint_prior_scale=request.changepoint_prior_scale,
        seasonality_prior_scale=request.seasonality_prior_scale,
        holidays_prior_scale=request.holidays_prior_scale,
        seasonality_mode=request.seasonality_mode,
        yearly_seasonality=request.yearly_seasonality,
        weekly_seasonality=request.weekly_seasonality,
        daily_seasonality=request.daily_seasonality
    )

    return validation_results


@app.post('/upload-data', 
          status_code=status.HTTP_200_OK, 
          summary="Upload Sales History CSV File")
async def upload_data_endpoint(
        file: UploadFile = File(..., description="CSV file with sales history. Expected columns: sku, qty_ordered, created_at. Header row is optional.")
):
    """
    Accepts a CSV file and saves it to the configured data file path.
    """
    logging.info(f"Received file upload request: {file.filename}")
    
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV file (.csv extension)"
        )
    
    try:
        contents = await file.read()
        
        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            with open(tmp_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip().lower()
                has_header = any(col in first_line for col in ['sku', 'qty_ordered', 'created_at', 'qty', 'quantity'])
            
            if has_header:
                test_df = pd.read_csv(tmp_path, names=['sku', 'qty_ordered', 'created_at'], header=0)
                logging.info("Detected header row in CSV, will skip it when saving")
            else:
                test_df = pd.read_csv(tmp_path, names=['sku', 'qty_ordered', 'created_at'], header=None)
                logging.info("No header detected, reading as data")
            
            if test_df.empty:
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="CSV file is empty or contains only header"
                )
            
            numeric_mask = pd.to_numeric(test_df['qty_ordered'], errors='coerce').notna()
            if not numeric_mask.all():
                test_df = test_df[numeric_mask].copy()
                logging.warning(f"Filtered out {len(numeric_mask) - numeric_mask.sum()} non-numeric rows (likely headers)")
            
            if test_df.empty:
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="CSV file contains no valid data rows after filtering"
                )
            
            try:
                test_df['qty_ordered'] = pd.to_numeric(test_df['qty_ordered'], errors='raise')
                test_df['created_at'] = pd.to_datetime(test_df['created_at'], errors='raise')
            except Exception as e:
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"CSV validation failed: Invalid data types. {str(e)}"
                )
            
            test_df[['sku', 'qty_ordered', 'created_at']].to_csv(
                DATA_FILE_PATH, 
                index=False, 
                header=False
            )
            
            os.unlink(tmp_path)
            
            row_count = len(test_df)
            unique_skus = test_df['sku'].nunique()
            
            logging.info(f"Successfully uploaded and saved CSV file. "
                         f"Rows: {row_count}, Unique SKUs: {unique_skus}, "
                         f"Saved to: {DATA_FILE_PATH}")
            
            return {
                "status": "success",
                "message": "CSV file uploaded and saved successfully",
                "file_path": DATA_FILE_PATH,
                "rows_count": row_count,
                "unique_skus": unique_skus,
                "date_range": {
                    "start": test_df['created_at'].min().isoformat(),
                    "end": test_df['created_at'].max().isoformat()
                } if not test_df.empty else None
            }
            
        except pd.errors.EmptyDataError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV file appears to be empty or invalid"
            )
        except pd.errors.ParserError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV parsing error: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload and save file: {str(e)}"
        )


@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check")
async def health_check():
    """Simple health check endpoint."""
    logging.debug("Health check endpoint called.")
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    if not os.path.exists(DATA_FILE_PATH):
        logging.warning(
            f"LOCAL DEV: Data file not found at '{DATA_FILE_PATH}'. "
            f"Create dummy data or adjust path/env var.")

    logging.info("Starting Uvicorn server for local development on http://0.0.0.0:5000")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
    