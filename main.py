import pandas as pd
from fastapi import FastAPI, HTTPException, Path, Query, status, Body, UploadFile, File
from typing import Dict, Any, List
from datetime import datetime
import joblib
import os
import logging
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import itertools
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = os.getenv("MODEL_DIR", "trained_models")
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "data/sales_history.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI(
    title="Magento Sales Predictor API",
    description="API for training ML models (with hyperparameter tuning), "
                "predicting future sales, and validating model accuracy.",
    version="0.1.0"
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
        # Convert 'created_at', making it timezone-naive for consistency
        df_product_raw['created_at'] = pd.to_datetime(df_product_raw['created_at']).dt.tz_localize(None)
        df_daily = df_product_raw.set_index('created_at').resample('D')['qty_ordered'].sum().reset_index()
        df_daily.rename(columns={"created_at": "ds", "qty_ordered": "y"}, inplace=True)
        df_daily = df_daily.sort_values('ds')  # Ensure sorted
        logging.info(f"Aggregated data for SKU {sku}. Found {len(df_daily)} daily records.")
    except Exception as e:
        logging.error(f"Error processing data for SKU {sku}: {e}", exc_info=True)
        return None

    if len(df_daily) < 14:  # Check after aggregation
        logging.warning(f"Not enough aggregated data points ({len(df_daily)}) for SKU: {sku}. Need at least 14.")
        return None

    return df_daily


def perform_hyperparameter_tuning(df: pd.DataFrame, horizon_days: int = 30) -> Dict[str, Any]:
    """
    Performs hyperparameter tuning using Prophet's cross-validation.
    Returns a dictionary with the best parameters and a status message.
    """
    # Define a simple grid of parameters to search
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [1.0, 5.0, 10.0, 20.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'holidays_prior_scale': [1.0, 5.0, 10.0, 20.0],
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

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

    logging.info(f"Starting hyperparameter tuning with {len(all_params)} combinations...")

    for params in all_params:
        try:
            m = Prophet(**params, daily_seasonality=True, weekly_seasonality=True).fit(df)
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
        best_params = {
            'holidays_prior_scale': 1.0,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'additive'
        }
        status_msg = "tuning_failed"
    else:
        logging.info(f"Tuning complete. Best MAE: {best_mae}. Best params: {best_params}")
        status_msg = "tuning_success"

    return {
        "status": status_msg,
        "best_parameters": best_params,
        "best_cross_validation_mae": best_mae
    }


def predict_sales_next_periods(model: Prophet, forecast_horizon: int = 30) -> Dict[str, Any]:
    """
    Predicts the total quantity sold over the next 7, 15, and 30 day periods.
    Casts results to float for JSON compatibility.
    """
    results = {
        "predicted_total_sales_next_7_days": "Error",
        "predicted_total_sales_next_15_days": "Error",
        "predicted_total_sales_next_30_days": "Error"
    }
    actual_horizon = max(30, forecast_horizon)

    try:
        future = model.make_future_dataframe(periods=actual_horizon)
        forecast = model.predict(future)

        today = pd.Timestamp.now().normalize().tz_localize(None)
        forecast['ds'] = forecast['ds'].dt.tz_localize(None)
        future_forecast = forecast[forecast['ds'] > today].copy()
        future_forecast['yhat'] = future_forecast['yhat'].clip(lower=0)

        date_7_days_end = today + pd.Timedelta(days=7)
        date_15_days_end = today + pd.Timedelta(days=15)
        date_30_days_end = today + pd.Timedelta(days=30)

        sales_next_7 = future_forecast[future_forecast['ds'] <= date_7_days_end]['yhat'].sum()
        sales_next_15 = future_forecast[future_forecast['ds'] <= date_15_days_end]['yhat'].sum()
        sales_next_30 = future_forecast[future_forecast['ds'] <= date_30_days_end]['yhat'].sum()

        results["predicted_total_sales_next_7_days"] = float(round(sales_next_7, 2))
        results["predicted_total_sales_next_15_days"] = float(round(sales_next_15, 2))
        results["predicted_total_sales_next_30_days"] = float(round(sales_next_30, 2))
        logging.info(f"Predicted total sales: Next 7d={sales_next_7:.2f}, "
                     f"Next 15d={sales_next_15:.2f}, Next 30d={sales_next_30:.2f}")

    except Exception as e:
        logging.error(f"Error during future sales prediction: {e}", exc_info=True)

    return results


# Add Pydantic model for request body
class ValidationRequest(BaseModel):
    test_period_days: int = Field(30, ge=30, description="Number of days for the test set (must be >= 30)")
    changepoint_prior_scale: float = Field(0.05, description="Flexibility of the trend")
    seasonality_prior_scale: float = Field(10.0, description="Strength of seasonality")
    holidays_prior_scale: float = Field(10.0, description="Strength of holiday effects")
    seasonality_mode: str = Field('additive', description="Seasonality mode ('additive' or 'multiplicative')")


async def run_period_accuracy_validation(
        sku: str,
        test_period_days: int = 30,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        seasonality_mode: str = 'additive'
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

    # Split data - use historical validation approach
    try:
        # Get the maximum date from historical data
        max_historical_date = df_daily['ds'].max()
        cutoff_date = max_historical_date - pd.Timedelta(days=test_period_days)
        
        # Train on all data BEFORE cutoff
        train_df = df_daily[df_daily['ds'] <= cutoff_date].copy()
        # Test on all data AFTER cutoff (still historical data)
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

    # Train model using provided hyperparameters
    try:
        model = Prophet(
            daily_seasonality=True, 
            weekly_seasonality=True,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode
        )
        model.fit(train_df)
        logging.info(f"Trained validation model for SKU {sku} with params: "
                    f"changepoint_prior_scale={changepoint_prior_scale}, "
                    f"seasonality_prior_scale={seasonality_prior_scale}, "
                    f"holidays_prior_scale={holidays_prior_scale}, "
                    f"seasonality_mode={seasonality_mode}")
    except Exception as e:
        logging.error(f"Error training temporary validation model for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Validation failed during training: {e}")

    # Generate predictions for historical test period dates only
    try:
        # Use exact historical dates from test_df - no future dates
        future_df_test_period = test_df[['ds']].copy().sort_values('ds')
        
        # Predict only for these historical dates
        forecast = model.predict(future_df_test_period)
        forecast['ds'] = pd.to_datetime(forecast['ds']).dt.tz_localize(None)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)

        # Merge forecast with actuals on exact date matches
        comparison_df = test_df[['ds', 'y']].copy().sort_values('ds')
        comparison_df = comparison_df.merge(
            forecast[['ds', 'yhat']], 
            on='ds', 
            how='inner'  # Use inner join to ensure we only compare dates that exist in both
        )
        
        if comparison_df.empty:
            raise ValueError("No matching dates between test data and forecast predictions")
        
        # Build the daily results
        predicted_list = [float(round(p, 2)) for p in comparison_df['yhat']]
        actual_list = [float(round(a, 2)) for a in comparison_df['y']]
        
        # Calculate metrics
        if len(actual_list) == 0 or len(predicted_list) == 0:
            raise ValueError("No data available for metrics calculation")
        
        actual_array = np.array(actual_list)
        predicted_array = np.array(predicted_list)
        
        mae = mean_absolute_error(actual_array, predicted_array)
        rmse = np.sqrt(mean_squared_error(actual_array, predicted_array))
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((actual_array - predicted_array) / actual_array)) * 100
            if np.isinf(mape) or np.isnan(mape):
                mape = None
            else:
                mape = float(round(mape, 2))
        
        # Mean Bias Error (MBE)
        mbe = np.mean(predicted_array - actual_array)
        mbe = float(round(mbe, 2))
        
        # Calculate R-squared
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
            "seasonality_mode": seasonality_mode
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
        sku: str = Path(..., description="The product SKU to train the model for.")
):
    """
    Triggers model training for a specific SKU using data from the configured CSV file.
    Performs hyperparameter tuning on the *entire* dataset before saving the final model.
    """
    logging.info(f"Received training request for SKU: {sku}")
    df_daily = get_product_data(sku)
    if df_daily is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No usable training data found for SKU {sku} at '{DATA_FILE_PATH}'. "
                   f"Ensure CSV exists and has enough data."
        )

    tuning_results = perform_hyperparameter_tuning(df_daily, horizon_days=30)
    best_params = tuning_results.get("best_parameters", {})
    logging.info(f"Training final model for SKU {sku} with params: {best_params}")

    try:
        model = Prophet(
            daily_seasonality=True, weekly_seasonality=True,
            **best_params  # Unpack the best parameters
        )
        model.fit(df_daily)
        model_path = os.path.join(MODEL_DIR, f"{sku}.joblib")
        joblib.dump(model, model_path)
        logging.info(f"Model for SKU {sku} saved successfully to {model_path}")
    except Exception as e:
        logging.error(f"Error during model training or saving for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Model training failed: {e}")

    return {
        "status": "success",
        "sku": sku,  # Fixed: Use the actual SKU variable
        "message": "Model trained successfully.",
        "model_saved_at": model_path,
        "tuning_results": tuning_results
    }


@app.get('/predict/{sku}', summary="Predict Future Sales for SKU")
async def predict_sales_endpoint(
        sku: str = Path(..., description="The product SKU to predict sales for.")
):
    """
    Predicts total sales for the next 7, 15, and 30 days.
    Requires a pre-trained model for the SKU (ideally trained via the /train endpoint).
    """
    logging.info(f"Received future sales prediction request for SKU: {sku}")
    model_path = os.path.join(MODEL_DIR, f"{sku}.joblib")

    if not os.path.exists(model_path):
        logging.warning(f"Prediction failed: No trained model found for SKU {sku} at {model_path}.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No trained model found for SKU {sku}.")

    try:
        model = joblib.load(model_path)
        logging.info(f"Loaded trained model for SKU: {sku} from {model_path}")

        sales_results = predict_sales_next_periods(model)

    except Exception as e:
        logging.error(f"Prediction failed for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to load model or run prediction: {e}")

    return {
        "sku": sku,  # Fixed: Use the actual SKU variable
        **sales_results,  # Unpack predicted sales for 7/15/30 days
        "prediction_engine": "Prophet"
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
        seasonality_mode=request.seasonality_mode
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
    The CSV should have columns: sku, qty_ordered, created_at.
    Header row is optional - if present, it will be detected and skipped.
    The saved file will have no header row to match the existing format.
    If the file exists, it will be overwritten.
    """
    logging.info(f"Received file upload request: {file.filename}")
    
    # Validate file extension
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV file (.csv extension)"
        )
    
    try:
        # Ensure data directory exists
        data_dir = os.path.dirname(DATA_FILE_PATH)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            logging.info(f"Created data directory: {data_dir}")
        
        # Read the uploaded file content
        contents = await file.read()
        
        # Validate CSV structure by trying to read it
        try:
            # Save to temporary location first to validate
            import tempfile
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            # Detect if CSV has a header row by checking first line
            with open(tmp_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip().lower()
                # Check if first line looks like a header (contains expected column names)
                has_header = any(col in first_line for col in ['sku', 'qty_ordered', 'created_at', 'qty', 'quantity'])
            
            # Try reading with header detection
            if has_header:
                # Read with header, then we'll save without header
                test_df = pd.read_csv(tmp_path, names=['sku', 'qty_ordered', 'created_at'], header=0)
                logging.info("Detected header row in CSV, will skip it when saving")
            else:
                # Read without header (matches existing format)
                test_df = pd.read_csv(tmp_path, names=['sku', 'qty_ordered', 'created_at'], header=None)
                logging.info("No header detected, reading as data")
            
            # Basic validation - check if columns exist and have data
            if test_df.empty:
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="CSV file is empty or contains only header"
                )
            
            # Validate column types - skip rows that might still be header-like
            # Filter out rows where qty_ordered is not numeric (e.g., header text)
            numeric_mask = pd.to_numeric(test_df['qty_ordered'], errors='coerce').notna()
            if not numeric_mask.all():
                # Some rows failed numeric conversion - likely header rows
                test_df = test_df[numeric_mask].copy()
                logging.warning(f"Filtered out {len(numeric_mask) - numeric_mask.sum()} non-numeric rows (likely headers)")
            
            if test_df.empty:
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="CSV file contains no valid data rows after filtering"
                )
            
            # Validate column types on cleaned data
            try:
                test_df['qty_ordered'] = pd.to_numeric(test_df['qty_ordered'], errors='raise')
                test_df['created_at'] = pd.to_datetime(test_df['created_at'], errors='raise')
            except Exception as e:
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"CSV validation failed: Invalid data types. {str(e)}"
                )
            
            # Save the cleaned data WITHOUT header to match existing format
            test_df[['sku', 'qty_ordered', 'created_at']].to_csv(
                DATA_FILE_PATH, 
                index=False, 
                header=False  # No header row in saved file
            )
            
            # Clean up temp file
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
