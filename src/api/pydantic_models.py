from pydantic import BaseModel
from typing import List

class CustomerFeatures(BaseModel):
    features: List[float]  # match the input feature size of your model

class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int