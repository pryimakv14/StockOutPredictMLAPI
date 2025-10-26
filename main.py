import pandas as pd
from fastapi import FastAPI, HTTPException, Path, Query, status
from typing import Dict, Any
from datetime import datetime
import joblib
import os
import logging
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import numpy as np

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = os.getenv("MODEL_DIR", "trained_models")
# Default path assumes it's run from within the python_service dir locally,
# or /app/data when run via Docker with the volume mount.
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "data/sales_history.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI(
    title="Magento Sales Predictor API",
    description="API for training ML models, predicting future sales, and validating model accuracy.",
    version="0.0.4" # Incremented version
)

# --- Helper Functions ---

def get_product_data(sku: str) -> pd.DataFrame | None:
    """Loads CSV, filters by SKU, aggregates daily, and prepares for Prophet."""
    try:
        logging.info(f"Attempting to read data file: {DATA_FILE_PATH}")
        if not os.path.exists(DATA_FILE_PATH):
             logging.error(f"Data file not found at {DATA_FILE_PATH}")
             return None
        df_all = pd.read_csv(DATA_FILE_PATH)
        logging.info(f"Successfully read data file. Total rows: {len(df_all)}")
    except pd.errors.EmptyDataError:
        logging.warning(f"Data file {DATA_FILE_PATH} is empty.")
        return None
    except Exception as e:
        logging.error(f"Error reading CSV file {DATA_FILE_PATH}: {e}", exc_info=True)
        return None

    sku_cleaned = str(sku).strip().lower()
    df_product_raw = df_all[df_all['sku'].astype(str).str.strip().str.lower() == sku_cleaned].copy()

    if df_product_raw.empty:
        logging.warning(f"No data found for SKU: {sku} in the CSV.")
        return None

    try:
        # Convert 'created_at', making it timezone-naive for consistency
        df_product_raw['created_at'] = pd.to_datetime(df_product_raw['created_at']).dt.tz_localize(None)
        df_daily = df_product_raw.set_index('created_at').resample('D')['qty_ordered'].sum().reset_index()
        df_daily.rename(columns={"created_at": "ds", "qty_ordered": "y"}, inplace=True)
        df_daily = df_daily.sort_values('ds') # Ensure sorted
        logging.info(f"Aggregated data for SKU {sku}. Found {len(df_daily)} daily records.")
    except Exception as e:
        logging.error(f"Error processing data for SKU {sku}: {e}", exc_info=True)
        return None

    if len(df_daily) < 14: # Check after aggregation
        logging.warning(f"Not enough aggregated data points ({len(df_daily)}) for SKU: {sku}. Need at least 14.")
        return None

    return df_daily


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

        today = pd.Timestamp.now().normalize().tz_localize(None) # Use timezone-naive today
        forecast['ds'] = forecast['ds'].dt.tz_localize(None) # Ensure forecast ds is naive
        future_forecast = forecast[forecast['ds'] > today].copy()
        future_forecast['yhat'] = future_forecast['yhat'].clip(lower=0)

        date_7_days_end = today + pd.Timedelta(days=7)
        date_15_days_end = today + pd.Timedelta(days=15)
        date_30_days_end = today + pd.Timedelta(days=30)

        # Sum daily predictions (yhat) within each period
        sales_next_7 = future_forecast[future_forecast['ds'] <= date_7_days_end]['yhat'].sum()
        sales_next_15 = future_forecast[future_forecast['ds'] <= date_15_days_end]['yhat'].sum()
        sales_next_30 = future_forecast[future_forecast['ds'] <= date_30_days_end]['yhat'].sum()

        # Cast to float for JSON serialization
        results["predicted_total_sales_next_7_days"] = float(round(sales_next_7, 2))
        results["predicted_total_sales_next_15_days"] = float(round(sales_next_15, 2))
        results["predicted_total_sales_next_30_days"] = float(round(sales_next_30, 2))
        logging.info(f"Predicted total sales: Next 7d={sales_next_7:.2f}, Next 15d={sales_next_15:.2f}, Next 30d={sales_next_30:.2f}")

    except Exception as e:
        logging.error(f"Error during future sales prediction: {e}", exc_info=True)
        # Keep default "Error" values

    return results


async def run_period_accuracy_validation(
    sku: str,
    test_period_days: int = 60,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    holidays_prior_scale: float = 10.0,
    seasonality_mode: str = 'additive'
) -> Dict[str, Any]:
    """
    Performs time-series cross-validation comparing predicted vs actual sales totals
    for 7, 15, and 30 day periods within the test set. Casts results for JSON.
    """
    logging.info(f"Running PERIOD accuracy validation for SKU: {sku} with test period: {test_period_days} days")

    if test_period_days < 30:
        msg = "Test period must be at least 30 days for this validation."
        logging.error(msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    df_daily = get_product_data(sku)
    if df_daily is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No usable historical data found for SKU {sku}.")

    df_daily['ds'] = pd.to_datetime(df_daily['ds']).dt.tz_localize(None)
    df_daily = df_daily.sort_values('ds')

    # Split data
    try:
        cutoff_date = df_daily['ds'].max() - pd.Timedelta(days=test_period_days)
        train_df = df_daily[df_daily['ds'] <= cutoff_date].copy()
        test_df = df_daily[df_daily['ds'] > cutoff_date].copy()

        if train_df.empty or len(test_df) < 30 or len(train_df) < 14:
             raise ValueError(f"Not enough data to split: Train {len(train_df)}, Test {len(test_df)} (needs >=30)")
        logging.info(f"Split data: Train until {cutoff_date.date()}, Test {len(test_df)} days.")

        start_test_date = test_df['ds'].min()
        test_end_7 = start_test_date + pd.Timedelta(days=6)
        test_end_15 = start_test_date + pd.Timedelta(days=14)
        test_end_30 = start_test_date + pd.Timedelta(days=29)

        actual_sales_7 = test_df[test_df['ds'] <= test_end_7]['y'].sum()
        actual_sales_15 = test_df[test_df['ds'] <= test_end_15]['y'].sum()
        actual_sales_30 = test_df[test_df['ds'] <= test_end_30]['y'].sum()

    except Exception as e:
        logging.error(f"Error splitting data or calculating actuals for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to split data/calculate actuals: {e}")

    # Train temporary model
    try:
        model = Prophet(
            daily_seasonality=True, weekly_seasonality=True,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode
        )
        model.fit(train_df)
    except Exception as e:
        logging.error(f"Error training temporary validation model for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Validation failed during training: {e}")

    # Make predictions for the test period
    try:
        future_df_test_period = test_df[['ds']].copy()
        forecast = model.predict(future_df_test_period)
        forecast['ds'] = forecast['ds'].dt.tz_localize(None)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)

        predicted_sales_7 = forecast[forecast['ds'] <= test_end_7]['yhat'].sum()
        predicted_sales_15 = forecast[forecast['ds'] <= test_end_15]['yhat'].sum()
        predicted_sales_30 = forecast[forecast['ds'] <= test_end_30]['yhat'].sum()

    except Exception as e:
        logging.error(f"Error predicting for validation (SKU {sku}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Validation failed during prediction: {e}")

    # Calculate Period Metrics
    try:
        ae_7 = abs(actual_sales_7 - predicted_sales_7)
        ape_7 = (ae_7 / actual_sales_7 * 100) if actual_sales_7 != 0 else np.inf
        ae_15 = abs(actual_sales_15 - predicted_sales_15)
        ape_15 = (ae_15 / actual_sales_15 * 100) if actual_sales_15 != 0 else np.inf
        ae_30 = abs(actual_sales_30 - predicted_sales_30)
        ape_30 = (ae_30 / actual_sales_30 * 100) if actual_sales_30 != 0 else np.inf

        # Cast results to float/str for JSON serialization
        metrics = {
            "7_day_actual": float(round(actual_sales_7, 2)),
            "7_day_predicted": float(round(predicted_sales_7, 2)),
            "7_day_abs_error": float(round(ae_7, 2)),
            "7_day_perc_error": float(round(ape_7, 2)) if ape_7 != np.inf else "Inf/NA (Actual=0)",

            "15_day_actual": float(round(actual_sales_15, 2)),
            "15_day_predicted": float(round(predicted_sales_15, 2)),
            "15_day_abs_error": float(round(ae_15, 2)),
            "15_day_perc_error": float(round(ape_15, 2)) if ape_15 != np.inf else "Inf/NA (Actual=0)",

            "30_day_actual": float(round(actual_sales_30, 2)),
            "30_day_predicted": float(round(predicted_sales_30, 2)),
            "30_day_abs_error": float(round(ae_30, 2)),
            "30_day_perc_error": float(round(ape_30, 2)) if ape_30 != np.inf else "Inf/NA (Actual=0)",
        }
        logging.info(f"Period validation metrics for SKU {sku}: {metrics}")
    except Exception as e:
        logging.error(f"Error calculating period metrics for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Validation failed during metric calculation: {e}")

    # Return results
    return {
        "sku": sku,
        "validation_period_info": f"Training data until {cutoff_date.date()}, Testing on {len(test_df)} subsequent days.",
        "parameters_used": {
             "changepoint_prior_scale": changepoint_prior_scale,
             "seasonality_prior_scale": seasonality_prior_scale,
             "holidays_prior_scale": holidays_prior_scale,
             "seasonality_mode": seasonality_mode
        },
        "period_metrics": metrics
    }


# --- API Endpoints ---

@app.post('/train/{sku}', status_code=status.HTTP_200_OK, summary="Train Model for SKU")
async def train_model_endpoint(
    sku: str = Path(..., description="The product SKU to train the model for.")
):
    """
    Triggers model training for a specific SKU using data from the configured CSV file.
    """
    logging.info(f"Received training request for SKU: {sku}")
    df_daily = get_product_data(sku)
    if df_daily is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No usable training data found for SKU {sku} at '{DATA_FILE_PATH}'. Ensure CSV exists and has enough data."
        )
    try:
        model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        model.fit(df_daily)
        model_path = os.path.join(MODEL_DIR, f"{sku}.joblib")
        joblib.dump(model, model_path)
        logging.info(f"Model for SKU {sku} saved successfully to {model_path}")
    except Exception as e:
         logging.error(f"Error during model training or saving for SKU {sku}: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Model training failed: {e}")
    return {"status": "success", "sku": sku, "message": "Model trained successfully.", "model_saved_at": model_path}


@app.get('/predict/{sku}', summary="Predict Future Sales for SKU")
async def predict_sales_endpoint( # Renamed endpoint function
    sku: str = Path(..., description="The product SKU to predict sales for.")
    # Removed current_stock parameter
):
    """
    Predicts total sales for the next 7, 15, and 30 days.
    Requires a pre-trained model for the SKU.
    """
    logging.info(f"Received future sales prediction request for SKU: {sku}")
    model_path = os.path.join(MODEL_DIR, f"{sku}.joblib")

    if not os.path.exists(model_path):
        logging.warning(f"Prediction failed: No trained model found for SKU {sku} at {model_path}.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No trained model found for SKU {sku}.")

    try:
        model = joblib.load(model_path)
        logging.info(f"Loaded trained model for SKU: {sku} from {model_path}")
        
        # Only get future sales predictions
        sales_results = predict_sales_next_periods(model)

    except Exception as e:
        logging.error(f"Prediction failed for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to load model or run prediction: {e}")

    # Return only the relevant results
    return {
        "sku": sku,
        **sales_results, # Unpack predicted sales for 7/15/30 days
        "prediction_engine": "Prophet"
    }


@app.post('/validate-period-accuracy/{sku}', summary="Validate Model Accuracy (Period Totals)")
async def validate_period_accuracy_endpoint(
    sku: str = Path(..., description="The product SKU to validate accuracy for."),
    test_period_days: int = Query(60, description="Number of days for the test set (must be >= 30).", ge=30),
    changepoint_prior_scale: float = Query(0.05, description="Flexibility of the trend."),
    seasonality_prior_scale: float = Query(10.0, description="Strength of seasonality."),
    holidays_prior_scale: float = Query(10.0, description="Strength of holiday effects."),
    seasonality_mode: str = Query('additive', description="Seasonality mode ('additive' or 'multiplicative').")
) -> Dict[str, Any]:
    """
    Performs time-series cross-validation comparing predicted vs actual sales totals
    for 7, 15, and 30 day periods within the test set.
    """
    validation_results = await run_period_accuracy_validation(
        sku=sku, test_period_days=test_period_days,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        seasonality_mode=seasonality_mode
    )
    if "error" in validation_results:
         pass
             
    return validation_results


@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check")
async def health_check():
    """Simple health check endpoint."""
    logging.debug("Health check endpoint called.")
    return {"status": "healthy"}


# --- Main execution block for local testing ---
if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(DATA_FILE_PATH):
        logging.warning(f"LOCAL DEV: Data file not found at '{DATA_FILE_PATH}'. Create dummy data or adjust path/env var.")
        # Optional: Add dummy data creation here if desired for local testing

    logging.info("Starting Uvicorn server for local development on http://0.0.0.0:5000")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)