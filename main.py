import threading

import pandas as pd
import uvicorn
from api.main import app
from data.generator import generate_reference_data
from training.train import train_model

import config as cfg


def bootstrap():
    if not cfg.REFERENCE_DATA_PATH.exists():
        print("Generating reference data...")
        df = generate_reference_data()
        df.to_csv(cfg.REFERENCE_DATA_PATH, index=False)
        print(f"Reference data saved to {cfg.REFERENCE_DATA_PATH}")

    if not cfg.MODEL_PKL_PATH.exists():
        import mlflow
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("ModelReviver")
        print("Training initial model...")
        df = generate_reference_data()
        train_model(df)


def start_monitor():
    try:
        from monitoring.monitor import run_monitor_loop
        ref_df = pd.read_csv(cfg.REFERENCE_DATA_PATH)
        t = threading.Thread(target=run_monitor_loop, args=(ref_df,), daemon=True)
        t.start()
        print("Drift monitor background thread started")
    except Exception as e:
        print(f"Monitor failed to start: {e}")


if __name__ == "__main__":
    bootstrap()
    start_monitor()
    uvicorn.run(app, host="0.0.0.0", port=8000)
