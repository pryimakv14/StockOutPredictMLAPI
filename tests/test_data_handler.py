import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import patch
from app.data_handler import get_product_data


class TestGetProductData:
    
    def test_get_product_data_success(self):
        test_data = """SKU1,5,2024-01-01
SKU1,3,2024-01-01
SKU1,2,2024-01-02
SKU1,4,2024-01-03
SKU1,1,2024-01-04
SKU1,2,2024-01-05
SKU1,3,2024-01-06
SKU1,1,2024-01-07
SKU1,2,2024-01-08
SKU1,3,2024-01-09
SKU1,1,2024-01-10
SKU1,2,2024-01-11
SKU1,3,2024-01-12
SKU1,1,2024-01-13
SKU1,2,2024-01-14
SKU2,1,2024-01-01"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(test_data)
            tmp_path = tmp_file.name
        
        try:
            with patch('app.data_handler.DATA_FILE_PATH', tmp_path):
                result = get_product_data('SKU1')
                
                assert result is not None
                assert isinstance(result, pd.DataFrame)
                assert 'ds' in result.columns
                assert 'y' in result.columns
                assert len(result) >= 14
        finally:
            os.unlink(tmp_path)
    
    def test_get_product_data_file_not_found(self):
        with patch('app.data_handler.DATA_FILE_PATH', '/nonexistent/path/data.csv'):
            result = get_product_data('SKU1')
            assert result is None
    
    def test_get_product_data_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with patch('app.data_handler.DATA_FILE_PATH', tmp_path):
                result = get_product_data('SKU1')
                assert result is None
        finally:
            os.unlink(tmp_path)
    
    def test_get_product_data_sku_not_found(self):
        test_data = """SKU1,5,2024-01-01
SKU2,3,2024-01-01"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(test_data)
            tmp_path = tmp_file.name
        
        try:
            with patch('app.data_handler.DATA_FILE_PATH', tmp_path):
                result = get_product_data('SKU999')
                assert result is None
        finally:
            os.unlink(tmp_path)
    
    def test_get_product_data_insufficient_data(self):
        test_data = '\n'.join([f"SKU1,1,2024-01-{i:02d}" for i in range(1, 10)])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(test_data)
            tmp_path = tmp_file.name
        
        try:
            with patch('app.data_handler.DATA_FILE_PATH', tmp_path):
                result = get_product_data('SKU1')
                assert result is None
        finally:
            os.unlink(tmp_path)
    
    def test_get_product_data_case_insensitive(self):
        import pandas as pd
        dates = pd.date_range('2024-01-01', periods=45, freq='D')
        test_data = '\n'.join([
            f"sku1,{i},{dates[i].strftime('%Y-%m-%d')}" for i in range(15)
        ] + [
            f"SKU1,{i},{dates[i+15].strftime('%Y-%m-%d')}" for i in range(15)
        ] + [
            f"sku1,{i},{dates[i+30].strftime('%Y-%m-%d')}" for i in range(15)
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(test_data)
            tmp_path = tmp_file.name
        
        try:
            with patch('app.data_handler.DATA_FILE_PATH', tmp_path):
                result = get_product_data('SKU1')
                assert result is not None
                assert len(result) >= 14
        finally:
            os.unlink(tmp_path)
    
    def test_get_product_data_daily_aggregation(self):
        test_data = '\n'.join([
            "SKU1,5,2024-01-01",
            "SKU1,3,2024-01-01",
            "SKU1,2,2024-01-01",
            "SKU1,1,2024-01-02"
        ] + [f"SKU1,1,2024-01-{i:02d}" for i in range(3, 15)])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(test_data)
            tmp_path = tmp_file.name
        
        try:
            with patch('app.data_handler.DATA_FILE_PATH', tmp_path):
                result = get_product_data('SKU1')
                assert result is not None
                jan_1_data = result[result['ds'] == pd.to_datetime('2024-01-01')]
                assert len(jan_1_data) == 1
                assert jan_1_data['y'].values[0] == 10
        finally:
            os.unlink(tmp_path)
