"""Simulates production data with increasing drift and sends it to the API."""

import time
import requests

from data.generator import generate_production_batch

API_URL = "http://localhost:8000/predict"


def simulate(batches: int = 12, batch_size: int = 50, delay: float = 2.0):
    print(f"Simulating {batches} batches of {batch_size} requests each...")
    for i in range(batches):
        drift = min(i * 0.01, 1.0)
        batch = generate_production_batch(batch_size, drift_step=drift, batch_index=i * 10)
        feature_cols = [c for c in batch.columns if c.startswith("feature_")]
        for _, row in batch.iterrows():
            features = row[feature_cols].tolist()
            try:
                requests.post(API_URL, json={"features": features}, timeout=2)
            except requests.ConnectionError:
                print("API not reachable — start the server first")
                return
        print(f"  Batch {i + 1}: drift_step={drift:.2f} — sent {batch_size} requests")
        time.sleep(delay)

    print("Simulation complete.")


if __name__ == "__main__":
    simulate()
