import pytest
import pandas as pd
from unittest.mock import patch, Mock, AsyncMock
from fastapi import HTTPException
from app.validation import run_period_accuracy_validation


class TestRunPeriodAccuracyValidation:
    
    @pytest.fixture
    def sample_dataframe(self):
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = [10 + i % 7 for i in range(100)]
        return pd.DataFrame({
            'ds': dates,
            'y': values
        })
    
    @pytest.mark.asyncio
    async def test_validation_insufficient_test_period(self):
        mock_validation_func = AsyncMock(side_effect=HTTPException(status_code=400, detail="Test period must be at least 30 days"))
        
        with patch('app.validation.get_validation_function', return_value=mock_validation_func):
            with pytest.raises(HTTPException) as exc_info:
                await run_period_accuracy_validation('SKU1', test_period_days=20)
            
            assert exc_info.value.status_code == 400
    
    @pytest.mark.asyncio
    async def test_validation_no_data_found(self):
        mock_validation_func = AsyncMock(side_effect=HTTPException(status_code=404, detail="No data found"))
        
        with patch('app.validation.get_validation_function', return_value=mock_validation_func):
            with pytest.raises(HTTPException) as exc_info:
                await run_period_accuracy_validation('SKU1', test_period_days=30)
            
            assert exc_info.value.status_code == 404
    
    @pytest.mark.asyncio
    async def test_validation_insufficient_training_data(self):
        mock_validation_func = AsyncMock(side_effect=HTTPException(status_code=400, detail="Insufficient training data"))
        
        with patch('app.validation.get_validation_function', return_value=mock_validation_func):
            with pytest.raises(HTTPException) as exc_info:
                await run_period_accuracy_validation('SKU1', test_period_days=30)
            
            assert exc_info.value.status_code == 400
    
    @pytest.mark.asyncio
    async def test_validation_success(self, sample_dataframe):
        mock_result = {
            'sku': 'SKU1',
            'metrics': {
                'mae': 2.5,
                'rmse': 3.0,
                'mape': 5.0,
                'mbe': 0.5,
                'r_squared': 0.95
            },
            'predicted': [10.5, 11.0, 10.8],
            'actual': [10, 11, 11],
            'parameters_used': {}
        }
        mock_validation_func = AsyncMock(return_value=mock_result)
        
        with patch('app.validation.get_validation_function', return_value=mock_validation_func):
            result = await run_period_accuracy_validation('SKU1', test_period_days=30, param1=0.05)
        
        assert 'sku' in result
        assert 'metrics' in result
        assert 'predicted' in result
        assert 'actual' in result
        assert 'mae' in result['metrics']
        assert 'rmse' in result['metrics']
        mock_validation_func.assert_called_once()
        call_kwargs = mock_validation_func.call_args[1]
        assert call_kwargs['test_period_days'] == 30
        assert call_kwargs['param1'] == 0.05
    
    @pytest.mark.asyncio
    async def test_validation_metrics_calculation(self, sample_dataframe):
        mock_result = {
            'sku': 'SKU1',
            'metrics': {
                'mae': 2.5,
                'rmse': 3.0,
                'mape': 5.0,
                'mbe': 0.5,
                'r_squared': 0.95
            },
            'predicted': [10.5, 11.0],
            'actual': [10, 11]
        }
        mock_validation_func = AsyncMock(return_value=mock_result)
        
        with patch('app.validation.get_validation_function', return_value=mock_validation_func):
            result = await run_period_accuracy_validation('SKU1', test_period_days=30)
        
        metrics = result['metrics']
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'mbe' in metrics
        assert 'r_squared' in metrics
    
    @pytest.mark.asyncio
    async def test_validation_hyperparameters_passed_through(self, sample_dataframe):
        mock_result = {
            'sku': 'SKU1',
            'metrics': {'mae': 2.5},
            'predicted': [10.5],
            'actual': [10],
            'parameters_used': {'param1': 0.1, 'param2': 5.0}
        }
        mock_validation_func = AsyncMock(return_value=mock_result)
        
        with patch('app.validation.get_validation_function', return_value=mock_validation_func):
            result = await run_period_accuracy_validation(
                'SKU1',
                test_period_days=30,
                param1=0.1,
                param2=5.0,
                param3='value'
            )
        
        call_kwargs = mock_validation_func.call_args[1]
        assert call_kwargs['param1'] == 0.1
        assert call_kwargs['param2'] == 5.0
        assert call_kwargs['param3'] == 'value'
