import json
import threading

import joblib
import mlflow
import torch

import config as cfg

_model = None
_model_version = "none"
_model_lock = threading.Lock()

CHAMPION_ALIAS = "champion"


def load_model():
    global _model, _model_version
    try:
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        try:
            mvd = client.get_model_version_by_alias(cfg.MODEL_REGISTRY_NAME, CHAMPION_ALIAS)
            uri = f"runs:/{mvd.run_id}/model"
            _model = mlflow.pytorch.load_model(uri)
            _model.eval()
            _model_version = str(mvd.version)
            return
        except Exception:
            # No alias set yet; try latest version
            versions = client.search_model_versions(f"name='{cfg.MODEL_REGISTRY_NAME}'")
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
                uri = f"runs:/{latest.run_id}/model"
                _model = mlflow.pytorch.load_model(uri)
                _model.eval()
                _model_version = str(latest.version)
                return
    except Exception as e:
        print(f"MLflow load failed: {e}")

    if cfg.MODEL_PKL_PATH.exists():
        artifact = joblib.load(cfg.MODEL_PKL_PATH)
        _model = artifact["model"]
        _model.eval()
        _model_version = "local"
        print("Loaded local fallback model")


def get_model():
    global _model
    if _model is None:
        load_model()
    return _model, _model_version


def hot_swap_model():
    with _model_lock:
        load_model()
    print(f"Model hot-swapped to version {_model_version}")
