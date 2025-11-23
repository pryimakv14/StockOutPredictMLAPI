from typing import Dict, Any
from app.model_factory import get_validation_function

async def run_period_accuracy_validation(
        sku: str,
        test_period_days: int = 30,
        **hyperparameters
) -> Dict[str, Any]:
    validation_func = get_validation_function()
    return await validation_func(sku, test_period_days=test_period_days, **hyperparameters)
