import pytest
import pandas as pd
from unittest.mock import patch, Mock, MagicMock
from app.model_training import perform_hyperparameter_tuning


class TestPerformHyperparameterTuning:
    
    @pytest.fixture
    def sample_dataframe(self):
        dates = pd.date_range('2023-01-01', periods=150, freq='D')
        values = [10 + i % 7 for i in range(150)]
        return pd.DataFrame({
            'ds': dates,
            'y': values
        })
    
    @pytest.fixture
    def insufficient_dataframe(self):
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        values = [10] * 50
        return pd.DataFrame({
            'ds': dates,
            'y': values
        })
    
    def test_hyperparameter_tuning_insufficient_data(self, insufficient_dataframe):
        mock_tuning_func = Mock(return_value={
            'status': 'skipped_tuning',
            'message': 'Skipped tuning due to insufficient data. Using default parameters.',
            'best_parameters': {}
        })
        
        with patch('app.model_training.get_training_function', return_value=mock_tuning_func):
            result = perform_hyperparameter_tuning(insufficient_dataframe, horizon_days=30)
        
        assert result['status'] == 'skipped_tuning'
        assert 'insufficient data' in result['message'].lower()
        assert result['best_parameters'] == {}
        mock_tuning_func.assert_called_once_with(insufficient_dataframe, 30)
    
    def test_hyperparameter_tuning_success(self, sample_dataframe):
        mock_tuning_func = Mock(return_value={
            'status': 'tuning_success',
            'best_parameters': {'param1': 0.1, 'param2': 5.0},
            'best_cross_validation_mae': 2.5
        })
        
        with patch('app.model_training.get_training_function', return_value=mock_tuning_func):
            result = perform_hyperparameter_tuning(sample_dataframe, horizon_days=30)
        
        assert result['status'] == 'tuning_success'
        assert 'best_parameters' in result
        assert 'best_cross_validation_mae' in result
        mock_tuning_func.assert_called_once_with(sample_dataframe, 30)
    
    def test_hyperparameter_tuning_all_fail(self, sample_dataframe):
        mock_tuning_func = Mock(return_value={
            'status': 'tuning_failed',
            'best_parameters': {}
        })
        
        with patch('app.model_training.get_training_function', return_value=mock_tuning_func):
            result = perform_hyperparameter_tuning(sample_dataframe, horizon_days=30)
        
        assert result['status'] == 'tuning_failed'
        assert result['best_parameters'] == {}
    
    def test_hyperparameter_tuning_returns_best_parameters(self, sample_dataframe):
        mock_tuning_func = Mock(return_value={
            'status': 'tuning_success',
            'best_parameters': {'param1': 0.1},
            'best_cross_validation_mae': 2.0
        })
        
        with patch('app.model_training.get_training_function', return_value=mock_tuning_func):
            result = perform_hyperparameter_tuning(sample_dataframe, horizon_days=30)
        
        if result['status'] == 'tuning_success':
            assert result['best_cross_validation_mae'] == 2.0
            assert 'param1' in result['best_parameters']
    
    def test_hyperparameter_tuning_horizon_passed_through(self, sample_dataframe):
        mock_tuning_func = Mock(return_value={
            'status': 'tuning_success',
            'best_parameters': {},
            'best_cross_validation_mae': 2.0
        })
        
        with patch('app.model_training.get_training_function', return_value=mock_tuning_func):
            result = perform_hyperparameter_tuning(sample_dataframe, horizon_days=60)
        
        mock_tuning_func.assert_called_once_with(sample_dataframe, 60)

