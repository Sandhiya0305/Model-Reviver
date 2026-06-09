from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_version: str
