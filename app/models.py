from pydantic import BaseModel, Field
from typing import Optional, Type
from app.model_factory import get_validation_request_class, get_optional_hyperparameters_class

ValidationRequest: Type[BaseModel] = get_validation_request_class()
OptionalHyperparameters: Type[BaseModel] = get_optional_hyperparameters_class()
