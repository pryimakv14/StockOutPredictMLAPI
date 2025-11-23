import logging
from typing import Dict, Any
from app.model_interface import BaseModel

logger = logging.getLogger(__name__)

def predict_stock_duration(model: BaseModel, current_stock: int, forecast_horizon_days: int = 365) -> Dict[str, Any]:
    try:
        if model.history is None or model.history.empty:
            logger.error("Model history is not available. Cannot determine last training date.")
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
            logger.warning(f"No future forecast data could be generated beyond {last_training_date.date()}.")
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
                logger.info(f"Stock for SKU depleted on (relative) day {days_counted}. "
                             f"Predicted OOS date: {row['ds'].date()}. "
                             f"Remaining stock: {remaining_stock:.2f}")
                return {
                    "days_of_stock_remaining": days_counted,
                    "predicted_out_of_stock_date": row['ds'].strftime('%Y-%m-%d'),
                    "last_training_date": last_training_date.strftime('%Y-%m-%d'),
                    "error_message": None
                }
        
        logger.info(f"Stock did not run out within the {forecast_horizon_days} day horizon.")
        return {
            "days_of_stock_remaining": forecast_horizon_days, 
            "predicted_out_of_stock_date": f"More than {forecast_horizon_days} days",
            "last_training_date": last_training_date.strftime('%Y-%m-%d'),
            "error_message": f"Stock did not run out within {forecast_horizon_days} days."
        }

    except Exception as e:
        logger.error(f"Error during stock duration prediction: {e}", exc_info=True)
        return {
            "days_of_stock_remaining": "Error", 
            "error_message": str(e),
            "predicted_out_of_stock_date": None,
            "last_training_date": None
        }
