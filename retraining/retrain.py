import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd

import config as cfg
from data.generator import generate_production_batch
from training.train import train_model
from monitoring.drift_detector import detect_drift


def _get_previous_metrics() -> dict:
    if cfg.METADATA_PATH.exists():
        with open(cfg.METADATA_PATH) as f:
            return json.load(f).get("metrics", {})
    return {}


def _promote_to_champion(model_name: str, version: str):
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name=model_name, version=version, alias="champion")


def retrain():
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ModelReviver")

    if not cfg.REFERENCE_DATA_PATH.exists():
        print("No reference data found, skipping retrain")
        return

    ref_df = pd.read_csv(cfg.REFERENCE_DATA_PATH)

    # Generate drifted synthetic data for retraining (simulates new real-world data)
    # Use a moderate drift step to reflect the distribution shift
    drifted = generate_production_batch(
        batch_size=cfg.REFERENCE_SIZE // 2,
        drift_step=0.04,
        batch_index=100,
    )

    combined = pd.concat([ref_df, drifted], ignore_index=True).drop_duplicates()

    if len(combined) < 100:
        print(f"Too few samples ({len(combined)}) to retrain")
        return

    print(f"Retraining on {len(combined)} samples (ref={len(ref_df)}, new={len(drifted)})")
    metrics, model = train_model(combined, return_model=True)

    prev_metrics = _get_previous_metrics()
    prev_f1 = prev_metrics.get("f1", 0.0)
    new_f1 = metrics.get("f1", 0.0)

    if new_f1 >= prev_f1 - 0.02:
        print(f"Validation passed: F1 {new_f1:.4f} >= previous {prev_f1:.4f}")

        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{cfg.MODEL_REGISTRY_NAME}'")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            _promote_to_champion(cfg.MODEL_REGISTRY_NAME, latest.version)

        joblib.dump({"model": model, "features": cfg.N_FEATURES}, cfg.MODEL_PKL_PATH)
        with open(cfg.METADATA_PATH, "w") as f:
            f.write(json.dumps({
                "version": str(latest.version) if versions else "unknown",
                "metrics": metrics,
                "prev_metrics": prev_metrics,
            }))

        from model_loader import hot_swap_model as _hot_swap
        _hot_swap()

        print(f"Retraining complete. New F1: {new_f1:.4f}")
    else:
        print(f"Validation failed: F1 {new_f1:.4f} < previous {prev_f1:.4f} — keeping old model")

    # Evaluate post-retraining drift
    drift_result = detect_drift(ref_df, drifted)
    print(f"Post-retraining drift score (drifted vs ref): {drift_result['drift_score']:.3f}")
