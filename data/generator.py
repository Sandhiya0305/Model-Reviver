import numpy as np
import pandas as pd

from config import N_FEATURES, REFERENCE_SIZE, PRODUCTION_BATCH_SIZE, RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)


def _true_label(features: np.ndarray) -> np.ndarray:
    weights = np.linspace(0.5, 1.5, N_FEATURES)
    logits = features @ weights + rng.normal(0, 0.1, size=len(features))
    return (logits > logits.mean()).astype(np.int64)


def generate_reference_data(n: int = REFERENCE_SIZE) -> pd.DataFrame:
    features = rng.normal(loc=0.0, scale=1.0, size=(n, N_FEATURES))
    labels = _true_label(features)
    columns = [f"feature_{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(features, columns=columns)
    df["label"] = labels
    df["timestamp"] = pd.Timestamp.now()
    return df


def generate_production_batch(
    batch_size: int = PRODUCTION_BATCH_SIZE,
    drift_step: float = 0.0,
    batch_index: int = 0,
) -> pd.DataFrame:
    loc = drift_step * min(batch_index, 100)
    features = rng.normal(loc=loc, scale=1.0, size=(batch_size, N_FEATURES))
    labels = _true_label(features)
    columns = [f"feature_{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(features, columns=columns)
    df["label"] = labels
    df["timestamp"] = pd.Timestamp.now()
    return df
