# Unit Tests

This directory contains unit tests for the key functions in the application.

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run a specific test file:
```bash
pytest tests/test_data_handler.py
```

To run with verbose output:
```bash
pytest tests/ -v
```

To run with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Files

- `test_data_handler.py` - Tests for data loading and processing functions
- `test_predictions.py` - Tests for stock duration prediction functions
- `test_model_training.py` - Tests for hyperparameter tuning
- `test_validation.py` - Tests for model validation functions

## Test Coverage

The tests cover:
- Success cases
- Error handling
- Edge cases (empty data, insufficient data, etc.)
- Data validation
- Parameter handling

