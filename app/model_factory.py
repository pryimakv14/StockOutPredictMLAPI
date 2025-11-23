import logging
from typing import Dict, Any, Optional, Callable, Type
from pydantic import BaseModel
import pandas as pd
from app.model_interface import BaseModel as ModelInterface
from app.prophet.model import ProphetModel
from app.prophet.training import perform_hyperparameter_tuning as prophet_perform_hyperparameter_tuning
from app.prophet.validation import run_period_accuracy_validation as prophet_run_period_accuracy_validation
from app.prophet.models import ValidationRequest as ProphetValidationRequest, OptionalHyperparameters as ProphetOptionalHyperparameters

logger = logging.getLogger(__name__)

def create_model(params: Optional[Dict[str, Any]] = None) -> ModelInterface:
    default_params = {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False
    }
    
    if params:
        default_params.update(params)
    
    return ProphetModel(**default_params)


def get_model_name() -> str:
    return ProphetModel.get_model_name()


def get_training_function() -> Callable:
    return prophet_perform_hyperparameter_tuning


def get_validation_function() -> Callable:
    return prophet_run_period_accuracy_validation


def get_validation_request_class() -> Type[BaseModel]:
    return ProphetValidationRequest


def get_optional_hyperparameters_class() -> Type[BaseModel]:
    return ProphetOptionalHyperparameters

