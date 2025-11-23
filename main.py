import os
import logging
import tempfile
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Path, Query, status, Body, UploadFile, File
from typing import Dict, Any, Optional

from app.config import MODEL_DIR, DATA_FILE_PATH
from app.models import ValidationRequest, OptionalHyperparameters
from app.data_handler import get_product_data
from app.model_training import perform_hyperparameter_tuning
from app.predictions import predict_stock_duration
from app.validation import run_period_accuracy_validation
from app.model_factory import create_model, get_model_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Magento Sales Predictor API",
    description="API for training ML models (with hyperparameter tuning), "
                "predicting future sales, and validating model accuracy.",
    version="0.3.0"
)


@app.post('/train/{sku}', status_code=status.HTTP_200_OK, summary="Train Model for SKU")
async def train_model_endpoint(
        sku: str = Path(..., description="The product SKU to train the model for."),
        params: Optional[OptionalHyperparameters] = Body(None, description="Optional: Specific hyperparameters to use. If omitted, random search tuning will be performed.")
):
    logger.info(f"Received training request for SKU: {sku}")
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
        logger.info(f"No specific parameters provided for SKU {sku}. Running random search tuning...")
        tuning_results_dict = perform_hyperparameter_tuning(df_daily, horizon_days=30)
        
        best_params = tuning_results_dict.get("best_parameters", {})
        training_info = tuning_results_dict
        logger.info(f"Tuning complete for SKU {sku}. Best params found: {best_params}")
    
    else:
        logger.info(f"Specific parameters provided for SKU {sku}. Skipping tuning.")
        
        best_params = params.model_dump(exclude_unset=True, exclude_none=True)
        
        training_info = {
            "status": "skipped_tuning_params_provided",
            "parameters_used": best_params,
            "message": "Trained using parameters provided in the request."
        }
        logger.info(f"Training SKU {sku} with provided params: {best_params}")

    try:
        model = create_model(params=best_params)
        
        model.fit(df_daily)
        model_path = os.path.join(MODEL_DIR, f"{sku}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model for SKU {sku} saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Error during model training or saving for SKU {sku}: {e}", exc_info=True)
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
    logger.info(f"Received stock duration prediction request for SKU: {sku} with stock: {current_stock}")
    model_path = os.path.join(MODEL_DIR, f"{sku}.joblib")

    if not os.path.exists(model_path):
        logger.warning(f"Prediction failed: No trained model found for SKU {sku} at {model_path}.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No trained model found for SKU {sku}.")

    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded trained model for SKU: {sku} from {model_path}")

        duration_results = predict_stock_duration(
            model=model, 
            current_stock=current_stock, 
            forecast_horizon_days=forecast_horizon_days
        )

    except Exception as e:
        logger.error(f"Prediction failed for SKU {sku}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to load model or run prediction: {e}")

    return {
        "sku": sku,
        "current_stock_provided": current_stock,
        "forecast_horizon_checked": forecast_horizon_days,
        "prediction_engine": get_model_name(),
        **duration_results
    }


@app.post('/validate-period-accuracy/{sku}',
          summary="Validate Model Accuracy (Daily Results)",
          response_model=Dict[str, Any])
async def validate_period_accuracy_endpoint(
        sku: str = Path(..., description="The product SKU to validate accuracy for."),
        request: ValidationRequest = Body(..., description="Validation parameters including test period days and hyperparameters")
) -> Dict[str, Any]:
    logger.info(f"Received validation request for SKU: {sku} with parameters: {request}")
    # Convert request to dict and pass as kwargs to maintain abstraction
    request_dict = request.model_dump(exclude_unset=False)
    test_period_days = request_dict.pop('test_period_days')
    validation_results = await run_period_accuracy_validation(
        sku=sku,
        test_period_days=test_period_days,
        **request_dict
    )

    return validation_results


@app.post('/upload-data', 
          status_code=status.HTTP_200_OK, 
          summary="Upload Sales History CSV File")
async def upload_data_endpoint(
        file: UploadFile = File(..., description="CSV file with sales history. Expected columns: sku, qty_ordered, created_at. Header row is optional.")
):
    logger.info(f"Received file upload request: {file.filename}")
    
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
                logger.info("Detected header row in CSV, will skip it when saving")
            else:
                test_df = pd.read_csv(tmp_path, names=['sku', 'qty_ordered', 'created_at'], header=None)
                logger.info("No header detected, reading as data")
            
            if test_df.empty:
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="CSV file is empty or contains only header"
                )
            
            numeric_mask = pd.to_numeric(test_df['qty_ordered'], errors='coerce').notna()
            if not numeric_mask.all():
                test_df = test_df[numeric_mask].copy()
                logger.warning(f"Filtered out {len(numeric_mask) - numeric_mask.sum()} non-numeric rows (likely headers)")
            
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
            
            logger.info(f"Successfully uploaded and saved CSV file. "
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
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload and save file: {str(e)}"
        )


@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check")
async def health_check():
    logger.debug("Health check endpoint called.")
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    if not os.path.exists(DATA_FILE_PATH):
        logger.warning(
            f"LOCAL DEV: Data file not found at '{DATA_FILE_PATH}'. "
            f"Create dummy data or adjust path/env var.")

    logger.info("Starting Uvicorn server for local development on http://0.0.0.0:5000")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
