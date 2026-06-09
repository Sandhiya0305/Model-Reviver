import json
import time
from pathlib import Path

import pandas as pd

import config as cfg
from monitoring.drift_detector import detect_drift


def run_monitor_loop(reference_df: pd.DataFrame):
    print("Monitor loop started")
    while True:
        time.sleep(cfg.CHECK_INTERVAL_SECONDS)

        log_path = Path(cfg.PRODUCTION_LOG_PATH)
        if not log_path.exists():
            continue

        df = pd.read_csv(log_path)
        if len(df) < cfg.WINDOW_SIZE:
            continue

        current_window = df.tail(cfg.WINDOW_SIZE)

        result = detect_drift(reference_df, current_window)
        print(f"Drift check: score={result['drift_score']:.3f}, detected={result['drift_detected']}")

        if result["drift_detected"]:
            print("Drift detected — triggering retraining")
            from retraining.retrain import retrain
            try:
                retrain()
            except Exception as e:
                print(f"Retraining failed: {e}")
