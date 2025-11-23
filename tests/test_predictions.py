import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from app.predictions import predict_stock_duration

class TestPredictStockDuration:
    
    def test_predict_stock_duration_success(self):
        mock_model = Mock()
        history_dates = pd.date_range('2024-01-01', periods=30, freq='D')
        mock_model.history = pd.DataFrame({
            'ds': history_dates,
            'y': [10] * 30
        })
        
        last_date = history_dates.max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=10, freq='D')
        
        all_dates = pd.concat([pd.Series(history_dates), pd.Series(future_dates)]).reset_index(drop=True)
        mock_forecast = pd.DataFrame({
            'ds': all_dates,
            'yhat': [10] * 40
        })
        
        mock_model.make_future_dataframe = Mock(return_value=pd.DataFrame({'ds': all_dates}))
        mock_model.predict = Mock(return_value=mock_forecast)
        
        result = predict_stock_duration(mock_model, current_stock=50, forecast_horizon_days=10)
        
        assert result['days_of_stock_remaining'] == 5
        assert result['error_message'] is None
        assert result['predicted_out_of_stock_date'] is not None
        assert result['last_training_date'] is not None
    
    def test_predict_stock_duration_empty_history(self):
        mock_model = Mock()
        mock_model.history = None
        
        result = predict_stock_duration(mock_model, current_stock=100)
        
        assert result['days_of_stock_remaining'] == "Error"
        assert result['error_message'] == "Model history is empty."
        assert result['predicted_out_of_stock_date'] is None
        assert result['last_training_date'] is None
    
    def test_predict_stock_duration_stock_not_depleted(self):
        mock_model = Mock()
        history_dates = pd.date_range('2024-01-01', periods=30, freq='D')
        mock_model.history = pd.DataFrame({
            'ds': history_dates,
            'y': [1] * 30
        })
        
        last_date = history_dates.max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=10, freq='D')
        
        all_dates = pd.concat([pd.Series(history_dates), pd.Series(future_dates)]).reset_index(drop=True)
        mock_forecast = pd.DataFrame({
            'ds': all_dates,
            'yhat': [1] * 40
        })
        
        mock_model.make_future_dataframe = Mock(return_value=pd.DataFrame({'ds': all_dates}))
        mock_model.predict = Mock(return_value=mock_forecast)
        
        result = predict_stock_duration(mock_model, current_stock=1000, forecast_horizon_days=10)
        
        assert result['days_of_stock_remaining'] == 10
        assert "More than" in result['predicted_out_of_stock_date']
        assert result['error_message'] is not None
    
    def test_predict_stock_duration_negative_predictions_clipped(self):
        mock_model = Mock()
        history_dates = pd.date_range('2024-01-01', periods=30, freq='D')
        mock_model.history = pd.DataFrame({
            'ds': history_dates,
            'y': [10] * 30
        })
        
        last_date = history_dates.max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=5, freq='D')
        
        all_dates = pd.concat([pd.Series(history_dates), pd.Series(future_dates)]).reset_index(drop=True)
        mock_forecast = pd.DataFrame({
            'ds': all_dates,
            'yhat': [10] * 30 + [-5, 20, 15, 10, 5]
        })
        
        mock_model.make_future_dataframe = Mock(return_value=pd.DataFrame({'ds': all_dates}))
        mock_model.predict = Mock(return_value=mock_forecast)
        
        result = predict_stock_duration(mock_model, current_stock=50, forecast_horizon_days=5)
        
        assert result['days_of_stock_remaining'] != "Error"
        assert result['error_message'] is None
    
    def test_predict_stock_duration_empty_forecast(self):
        mock_model = Mock()
        mock_model.history = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=30, freq='D'),
            'y': [10] * 30
        })
        
        mock_forecast = pd.DataFrame({
            'ds': mock_model.history['ds'],
            'yhat': [10] * 30
        })
        
        mock_model.make_future_dataframe = Mock(return_value=pd.DataFrame({'ds': mock_forecast['ds']}))
        mock_model.predict = Mock(return_value=mock_forecast)
        
        result = predict_stock_duration(mock_model, current_stock=100)
        
        assert result['days_of_stock_remaining'] == "Error"
        assert "No future forecast data" in result['error_message']
    
    def test_predict_stock_duration_exception_handling(self):
        mock_model = Mock()
        mock_model.history = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=30, freq='D'),
            'y': [10] * 30
        })
        mock_model.make_future_dataframe = Mock(side_effect=Exception("Test error"))
        
        result = predict_stock_duration(mock_model, current_stock=100)
        
        assert result['days_of_stock_remaining'] == "Error"
        assert "Test error" in result['error_message']
