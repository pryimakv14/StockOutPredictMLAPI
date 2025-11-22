import pytest
import pandas as pd
from unittest.mock import patch, Mock, AsyncMock
from fastapi import HTTPException
from app.validation import run_period_accuracy_validation


class TestRunPeriodAccuracyValidation:
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe with enough data for validation"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = [10 + i % 7 for i in range(100)]
        return pd.DataFrame({
            'ds': dates,
            'y': values
        })
    
    @pytest.mark.asyncio
    async def test_validation_insufficient_test_period(self):
        """Test that validation fails with test period < 30 days"""
        with patch('app.validation.get_product_data', return_value=pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=50, freq='D'),
            'y': [10] * 50
        })):
            with pytest.raises(HTTPException) as exc_info:
                await run_period_accuracy_validation('SKU1', test_period_days=20)
            
            assert exc_info.value.status_code == 400
    
    @pytest.mark.asyncio
    async def test_validation_no_data_found(self):
        """Test handling when no data is found for SKU"""
        with patch('app.validation.get_product_data', return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                await run_period_accuracy_validation('SKU1', test_period_days=30)
            
            assert exc_info.value.status_code == 404
    
    @pytest.mark.asyncio
    async def test_validation_insufficient_training_data(self):
        """Test handling when training data is insufficient"""
        df = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=10, freq='D'),
            'y': [10] * 10
        })
        
        with patch('app.validation.get_product_data', return_value=df):
            with pytest.raises(HTTPException) as exc_info:
                await run_period_accuracy_validation('SKU1', test_period_days=30)
            
            assert exc_info.value.status_code == 400
    
    @pytest.mark.asyncio
    @patch('app.validation.Prophet')
    async def test_validation_success(self, mock_prophet, sample_dataframe):
        """Test successful validation"""
        def mock_predict(future_df):
            return pd.DataFrame({
                'ds': future_df['ds'].values,
                'yhat': [10.5] * len(future_df)
            })
        
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=mock_predict)
        mock_prophet.return_value = mock_model
        
        with patch('app.validation.get_product_data', return_value=sample_dataframe):
            result = await run_period_accuracy_validation(
                'SKU1',
                test_period_days=30,
                changepoint_prior_scale=0.05
            )
        
        assert 'sku' in result
        assert 'metrics' in result
        assert 'predicted' in result
        assert 'actual' in result
        assert 'mae' in result['metrics']
        assert 'rmse' in result['metrics']
    
    @pytest.mark.asyncio
    @patch('app.validation.Prophet')
    async def test_validation_metrics_calculation(self, mock_prophet, sample_dataframe):
        """Test that validation calculates all required metrics"""
        def mock_predict(future_df):
            return pd.DataFrame({
                'ds': future_df['ds'].values,
                'yhat': [10.5] * len(future_df)
            })
        
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=mock_predict)
        mock_prophet.return_value = mock_model
        
        with patch('app.validation.get_product_data', return_value=sample_dataframe):
            result = await run_period_accuracy_validation('SKU1', test_period_days=30)
        
        metrics = result['metrics']
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'mbe' in metrics
        assert 'r_squared' in metrics
    
    @pytest.mark.asyncio
    @patch('app.validation.Prophet')
    async def test_validation_parameters_used(self, mock_prophet, sample_dataframe):
        """Test that validation uses provided parameters"""
        def mock_predict(future_df):
            return pd.DataFrame({
                'ds': future_df['ds'].values,
                'yhat': [10.5] * len(future_df)
            })
        
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=mock_predict)
        mock_prophet.return_value = mock_model
        
        with patch('app.validation.get_product_data', return_value=sample_dataframe):
            result = await run_period_accuracy_validation(
                'SKU1',
                test_period_days=30,
                changepoint_prior_scale=0.1,
                seasonality_prior_scale=5.0,
                seasonality_mode='multiplicative'
            )
        
        params_used = result['parameters_used']
        assert params_used['changepoint_prior_scale'] == 0.1
        assert params_used['seasonality_prior_scale'] == 5.0
        assert params_used['seasonality_mode'] == 'multiplicative'

