from pydantic import BaseModel, Field
from typing import Optional


class ValidationRequest(BaseModel):
    test_period_days: int = Field(30, ge=30, description="Number of days for the test set (must be >= 30)")
    changepoint_prior_scale: float = Field(0.05, description="Flexibility of the trend")
    seasonality_prior_scale: float = Field(10.0, description="Strength of seasonality")
    holidays_prior_scale: float = Field(10.0, description="Strength of holiday effects")
    seasonality_mode: str = Field('additive', description="Seasonality mode ('additive' or 'multiplicative')")
    yearly_seasonality: bool = Field(True, description="Enable yearly seasonality")
    weekly_seasonality: bool = Field(True, description="Enable weekly seasonality")
    daily_seasonality: bool = Field(False, description="Enable daily seasonality (default False for daily data)")


class OptionalHyperparameters(BaseModel):
    changepoint_prior_scale: Optional[float] = Field(None, description="Flexibility of the trend")
    seasonality_prior_scale: Optional[float] = Field(None, description="Strength of seasonality")
    holidays_prior_scale: Optional[float] = Field(None, description="Strength of holiday effects")
    seasonality_mode: Optional[str] = Field(None, description="Seasonality mode ('additive' or 'multiplicative')")
    yearly_seasonality: Optional[bool] = Field(None, description="Enable yearly seasonality")
    weekly_seasonality: Optional[bool] = Field(None, description="Enable weekly seasonality")
    daily_seasonality: Optional[bool] = Field(None, description="Enable daily seasonality")

