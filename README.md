# Magento Sales Predictor API

A FastAPI-based REST API for training machine learning models, predicting future sales, and validating model accuracy. This project uses Facebook's Prophet library for time series forecasting to predict stock duration for products based on historical sales data.

## Features

- **Model Training**: Train Prophet models for specific product SKUs with automatic hyperparameter tuning
- **Stock Duration Prediction**: Predict how many days until stock depletion based on current inventory
- **Model Validation**: Validate model accuracy using historical data with configurable test periods
- **Data Upload**: Upload sales history CSV files via REST API
- **Hyperparameter Tuning**: Automatic random search for optimal model parameters
- **Modular Architecture**: Factory pattern implementation allowing easy extension to other forecasting models

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd test
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
export MODEL_DIR="trained_models"  # Default: trained_models
export DATA_FILE_PATH="data/sales_history.csv"  # Default: data/sales_history.csv
```

Or create a `.env` file:
```
MODEL_DIR=trained_models
DATA_FILE_PATH=data/sales_history.csv
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── data_handler.py        # Data loading and processing
│   ├── model_factory.py       # Model creation factory
│   ├── model_interface.py     # Abstract base class for models
│   ├── model_training.py      # Hyperparameter tuning logic
│   ├── models.py              # Pydantic models for API requests
│   ├── predictions.py         # Prediction logic
│   ├── validation.py          # Model validation logic
│   └── prophet/               # Prophet-specific implementations
│       ├── __init__.py
│       ├── model.py
│       ├── models.py
│       ├── training.py
│       └── validation.py
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_data_handler.py
│   ├── test_model_training.py
│   ├── test_predictions.py
│   └── test_validation.py
├── main.py                    # FastAPI application entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage

### Starting the Server

Run the development server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

The API will be available at `http://localhost:5000`

### API Documentation

Once the server is running, you can access:
- **Interactive API docs**: `http://localhost:5000/docs` (Swagger UI)
- **Alternative docs**: `http://localhost:5000/redoc` (ReDoc)

### API Endpoints

#### 1. Upload Sales Data
**POST** `/upload-data`

Upload a CSV file containing sales history. Expected columns:
- `sku`: Product SKU identifier
- `qty_ordered`: Quantity ordered
- `created_at`: Order date (datetime format)

**Example:**
```bash
curl -X POST "http://localhost:5000/upload-data" \
  -F "file=@sales_history.csv"
```

#### 2. Train Model
**POST** `/train/{sku}`

Train a Prophet model for a specific SKU. Supports automatic hyperparameter tuning or manual parameter specification.

**With automatic tuning:**
```bash
curl -X POST "http://localhost:5000/train/PRODUCT-SKU-123"
```

**With specific parameters:**
```bash
curl -X POST "http://localhost:5000/train/PRODUCT-SKU-123" \
  -H "Content-Type: application/json" \
  -d '{
    "yearly_seasonality": true,
    "weekly_seasonality": true,
    "daily_seasonality": false
  }'
```

#### 3. Predict Stock Duration
**GET** `/predict/{sku}`

Predict how many days until stock depletion for a given SKU and current stock level.

**Parameters:**
- `sku`: Product SKU (path parameter)
- `current_stock`: Current inventory count (query parameter, required)
- `forecast_horizon_days`: Maximum days to forecast (query parameter, default: 365, range: 30-1095)

**Example:**
```bash
curl "http://localhost:5000/predict/PRODUCT-SKU-123?current_stock=100&forecast_horizon_days=365"
```

#### 4. Validate Model Accuracy
**POST** `/validate-period-accuracy/{sku}`

Validate model accuracy by testing on historical data.

**Example:**
```bash
curl -X POST "http://localhost:5000/validate-period-accuracy/PRODUCT-SKU-123" \
  -H "Content-Type: application/json" \
  -d '{
    "test_period_days": 30,
    "yearly_seasonality": true,
    "weekly_seasonality": true
  }'
```

#### 5. Health Check
**GET** `/health`

Check if the API is running.

```bash
curl "http://localhost:5000/health"
```

## Testing

Run the test suite:
```bash
pytest
```

Run tests with verbose output:
```bash
pytest -v
```

Run specific test file:
```bash
pytest tests/test_predictions.py
```

## Configuration

The application uses environment variables for configuration:

- `MODEL_DIR`: Directory where trained models are saved (default: `trained_models`)
- `DATA_FILE_PATH`: Path to the sales history CSV file (default: `data/sales_history.csv`)
- `N_RANDOM_SEARCH_ITERATIONS`: Number of iterations for hyperparameter tuning (default: 60, set in `config.py`)

## Data Format

The sales history CSV file should have the following format:
- **No header row** (or header will be automatically detected and skipped)
- Columns: `sku`, `qty_ordered`, `created_at`
- `created_at` should be in a parseable datetime format

Example:
```
PRODUCT-SKU-123,5,2024-01-15 10:30:00
PRODUCT-SKU-123,3,2024-01-16 14:20:00
PRODUCT-SKU-456,10,2024-01-15 09:15:00
```

## Model Details

This project uses **Facebook Prophet** for time series forecasting. Prophet is designed for business time series data and handles:
- Trend changes
- Seasonality (yearly, weekly, daily)
- Holiday effects
- Missing data and outliers

The implementation includes:
- Automatic hyperparameter tuning using random search
- Support for custom hyperparameters
- Model persistence using joblib
- Validation framework for accuracy assessment

## Development

### Adding New Models

The project uses a factory pattern, making it easy to add new forecasting models:

1. Create a new model class inheriting from `app.model_interface.BaseModel`
2. Implement all abstract methods
3. Update `app.model_factory.py` to use the new model
4. Add model-specific training and validation logic
