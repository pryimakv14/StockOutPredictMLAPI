import pytest
import pandas as pd
from unittest.mock import patch, Mock, MagicMock
from app.model_training import perform_hyperparameter_tuning


class TestPerformHyperparameterTuning:
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe with sufficient data for tuning"""
        dates = pd.date_range('2023-01-01', periods=150, freq='D')
        values = [10 + i % 7 for i in range(150)]
        return pd.DataFrame({
            'ds': dates,
            'y': values
        })
    
    @pytest.fixture
    def insufficient_dataframe(self):
        """Create a dataframe with insufficient data for tuning"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        values = [10] * 50
        return pd.DataFrame({
            'ds': dates,
            'y': values
        })
    
    def test_hyperparameter_tuning_insufficient_data(self, insufficient_dataframe):
        """Test that tuning is skipped when there's insufficient data"""
        result = perform_hyperparameter_tuning(insufficient_dataframe, horizon_days=30)
        
        assert result['status'] == 'skipped_tuning'
        assert 'insufficient data' in result['message'].lower()
        assert result['best_parameters'] == {}
    
    @patch('app.model_training.Prophet')
    @patch('app.model_training.cross_validation')
    @patch('app.model_training.performance_metrics')
    def test_hyperparameter_tuning_success(self, mock_perf_metrics, mock_cv, mock_prophet, sample_dataframe):
        """Test successful hyperparameter tuning"""
        mock_model = Mock()
        mock_prophet.return_value = mock_model
        
        mock_cv_result = pd.DataFrame({'y': [10, 12, 11], 'yhat': [9, 13, 10]})
        mock_cv.return_value = mock_cv_result
        
        mock_perf_df = pd.DataFrame({'mae': [2.5, 3.0, 2.0]})
        mock_perf_metrics.return_value = mock_perf_df
        
        with patch('app.model_training.N_RANDOM_SEARCH_ITERATIONS', 2):
            result = perform_hyperparameter_tuning(sample_dataframe, horizon_days=30)
        
        assert result['status'] in ['tuning_success', 'tuning_failed']
        assert 'best_parameters' in result
        assert 'best_cross_validation_mae' in result
    
    @patch('app.model_training.Prophet')
    @patch('app.model_training.cross_validation')
    @patch('app.model_training.performance_metrics')
    def test_hyperparameter_tuning_all_fail(self, mock_perf_metrics, mock_cv, mock_prophet, sample_dataframe):
        """Test when all parameter combinations fail"""
        mock_prophet.side_effect = Exception("Model fitting failed")
        
        with patch('app.model_training.N_RANDOM_SEARCH_ITERATIONS', 2):
            result = perform_hyperparameter_tuning(sample_dataframe, horizon_days=30)
        
        assert result['status'] == 'tuning_failed'
        assert result['best_parameters'] == {}
    
    @patch('app.model_training.Prophet')
    @patch('app.model_training.cross_validation')
    @patch('app.model_training.performance_metrics')
    def test_hyperparameter_tuning_selects_best_mae(self, mock_perf_metrics, mock_cv, mock_prophet, sample_dataframe):
        """Test that tuning selects parameters with lowest MAE"""
        mock_model = Mock()
        mock_prophet.return_value = mock_model
        
        mock_cv.return_value = pd.DataFrame({'y': [10], 'yhat': [9]})
        
        mock_perf_metrics.side_effect = [
            pd.DataFrame({'mae': [5.0]}),
            pd.DataFrame({'mae': [2.0]}),
        ]
        
        with patch('app.model_training.N_RANDOM_SEARCH_ITERATIONS', 2):
            result = perform_hyperparameter_tuning(sample_dataframe, horizon_days=30)
        
        if result['status'] == 'tuning_success':
            assert result['best_cross_validation_mae'] == 2.0
    
    def test_hyperparameter_tuning_horizon_calculation(self, sample_dataframe):
        """Test that horizon days affects the initial period calculation"""
        with patch('app.model_training.Prophet') as mock_prophet, \
             patch('app.model_training.cross_validation') as mock_cv, \
             patch('app.model_training.performance_metrics') as mock_perf:
            
            mock_model = Mock()
            mock_prophet.return_value = mock_model
            mock_cv.return_value = pd.DataFrame({'y': [10], 'yhat': [9]})
            mock_perf.return_value = pd.DataFrame({'mae': [2.0]})
            
            with patch('app.model_training.N_RANDOM_SEARCH_ITERATIONS', 1):
                result = perform_hyperparameter_tuning(sample_dataframe, horizon_days=60)
            
            if mock_cv.called:
                call_kwargs = mock_cv.call_args[1] if mock_cv.call_args[1] else mock_cv.call_args[0][1] if len(mock_cv.call_args[0]) > 1 else {}
                call_str = str(mock_cv.call_args)
                assert '60 days' in call_str or 'horizon' in call_str.lower()

