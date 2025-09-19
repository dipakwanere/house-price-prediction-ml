from typing import Dict, Any

from pydantic import BaseModel, Field


class KaggleHouseFeatures(BaseModel):
    """Flexible schema to accept raw Kaggle competition features.

    We keep it generic (Dict[str, Any]) because the dataset has many mixed types
    and optional values; preprocessing handles types and missing values.
    """

    features: Dict[str, Any] = Field(..., description="Raw Kaggle features dictionary")


class PredictionResponse(BaseModel):
    price: float = Field(..., description="Predicted SalePrice in USD")
