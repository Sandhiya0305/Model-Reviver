import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI

import config as cfg
from api.schemas import PredictionRequest, PredictionResponse
from model_loader import get_model, load_model

_model_lock = threading.Lock()


def _log_prediction(features: list[float], prediction: int, confidence: float, version: str):
    log_path = Path(cfg.PRODUCTION_LOG_PATH)
    row = {
        **{f"feature_{i}": features[i] for i in range(len(features))},
        "prediction": prediction,
        "confidence": confidence,
        "model_version": version,
        "timestamp": pd.Timestamp.now(),
    }
    header = not log_path.exists()
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=header, index=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="ModelReviver", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    _, version = get_model()
    return {"status": "ok", "model_version": version}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    model, model_version = get_model()
    if model is None:
        load_model()
        model, model_version = get_model()
    if model is None:
        return PredictionResponse(prediction=-1, confidence=0.0, model_version="none")

    features_np = np.array(req.features, dtype="float32").reshape(1, -1)
    with _model_lock:
        with torch.no_grad():
            tensor = torch.tensor(features_np)
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            conf = float(torch.max(probs).item())

    _log_prediction(req.features, pred, conf, model_version)
    return PredictionResponse(prediction=pred, confidence=conf, model_version=model_version)
