# ModelReviver — Implementation Plan

## Overview

ModelReviver is a self-healing MLOps platform that continuously monitors deployed ML models, detects data/concept drift, automatically retrains on fresh data, validates, and redeploys improved versions — all without human intervention.

---

## Dataset Strategy

Use a **synthetic tabular data generator** with built-in drift injection. This gives us:

- Controllable drift onset (trigger retraining on demand)
- No external download dependencies
- Full reproducibility
- Ground-truth labels for validation

The generator produces a **reference dataset** (training) and then a **production data stream** whose feature distribution gradually shifts over time (e.g., mean of Feature_A drifts from 0.0 → 2.0 over 5000 samples).

---

## Project Structure

```
modelreviver/
├── config.py                      # Central config (paths, thresholds, MLflow URI)
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container build
├── docker-compose.yml             # Multi-service orchestration
│
├── data/
│   ├── generator.py               # Synthetic data generator with drift
│   ├── reference_data.csv         # Generated reference (training) dataset
│   └── production_log.csv         # Accumulated production predictions
│
├── training/
│   ├── train.py                   # Train PyTorch model, log to MLflow
│   └── evaluate.py                # Evaluate model on holdout set
│
├── models/
│   ├── metadata.json              # Active model metadata
│   └── model.pkl                  # Local fallback model artifact
│
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app with /predict endpoint
│   └── schemas.py                 # Pydantic request/response models
│
├── monitoring/
│   ├── drift_detector.py          # Evidently AI drift analysis
│   └── monitor.py                 # Background scheduler for drift checks
│
└── retraining/
    └── retrain.py                 # Auto-retrain on drift, register & deploy
```

---

## Implementation Phases

### Phase 1 — Foundation

| Step | File | Description |
|------|------|-------------|
| 1.1 | `requirements.txt` | Pin `torch`, `pandas`, `scikit-learn`, `fastapi`, `uvicorn`, `evidently`, `mlflow`, `pydantic`, `joblib`, `python-multipart` |
| 1.2 | `config.py` | Central config class: `DRIFT_THRESHOLD=0.5`, `WINDOW_SIZE=500`, `MLFLOW_TRACKING_URI=file:./mlruns`, `MODEL_REGISTRY_NAME="ModelReviver"`, `PRODUCTION_LOG_PATH`, `REFERENCE_DATA_PATH`, etc. |
| 1.3 | `data/generator.py` | Function `generate_reference_data(n=5000) -> pd.DataFrame` producing clean data with ~5 numerical features + binary label. Function `generate_production_batch(batch_size=100, drift_step=0.0) -> pd.DataFrame` that progressively shifts feature mean by `drift_step`. |

### Phase 2 — Training Pipeline

| Step | File | Description |
|------|------|-------------|
| 2.1 | `training/train.py` | CLI: reads reference data, trains a PyTorch MLP (2 hidden layers, ReLU, Adam), logs to MLflow (params, metrics, model artifact), registers in MLflow Model Registry under name `"ModelReviver"`, also pickles to `models/model.pkl` as fallback |
| 2.2 | `training/evaluate.py` | Takes model + test DataLoader, computes accuracy, precision, recall, F1. Returns dict of metrics. Called by `train.py` after training and by `retrain.py` during validation gate. |

### Phase 3 — Prediction Service

| Step | File | Description |
|------|------|-------------|
| 3.1 | `api/main.py` | FastAPI app with startup event that loads model from MLflow registry (latest production stage) or falls back to `model.pkl`. `POST /predict` accepts features, runs inference, appends to `production_log.csv`, returns prediction + confidence. Background task: optional periodic drift check. |
| 3.2 | `api/schemas.py` | `PredictionRequest(BaseModel)` with `features: list[float]`, `PredictionResponse(BaseModel)` with `prediction: int`, `confidence: float`, `model_version: str`. |

### Phase 4 — Monitoring & Drift Detection

| Step | File | Description |
|------|------|-------------|
| 4.1 | `monitoring/drift_detector.py` | Uses Evidently AI `DataDriftPreset` on column `features`. Compares reference data vs. recent production window (last `WINDOW_SIZE` rows). Returns `drift_score` (share of drifted features) and boolean `drift_detected`. |
| 4.2 | `monitoring/monitor.py` | Runs in a background thread. Every `CHECK_INTERVAL` seconds (or every N predictions): reads production log, runs drift detector, logs result. If drift detected and above threshold, calls `retrain()` from retraining module. |

### Phase 5 — Retraining & Auto-Redeployment

| Step | File | Description |
|------|------|-------------|
| 5.1 | `retraining/retrain.py` | Fetches recent production data (features + logged predictions as pseudo-labels or true labels if available). Combines with reference data. Retrains model. Runs evaluation. If metrics >= previous model's metrics → registers new version in MLflow, transitions it to "Production" stage, hot-swaps the API's in-memory model, saves to `model.pkl`. If metrics degrade → logs warning, does NOT deploy. |

### Phase 6 — Containerization & Orchestration

| Step | File | Description |
|------|------|-------------|
| 6.1 | `Dockerfile` | Multi-stage: python:3.11-slim, install deps, copy code, expose 8000, run `uvicorn api.main:app --host 0.0.0.0 --port 8000` |
| 6.2 | `docker-compose.yml` | Two services: `api` (FastAPI + monitor thread) and `mlflow` (MLflow tracking server with `mlflow server` on port 5000) |

---

## End-to-End Closed Loop Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. train.py generates reference data, trains model,        │
│     registers in MLflow                                     │
│                     │                                       │
│                     ▼                                       │
│  2. api/main.py starts → loads model from MLflow registry   │
│                     │                                       │
│                     ▼                                       │
│  3. POST /predict → inference → log to production_log.csv   │
│                     │                                       │
│                     ▼                                       │
│  4. Background monitor wakes up periodically:               │
│     a. Reads last WINDOW_SIZE rows from production log      │
│     b. Compares vs reference data using Evidently AI        │
│     c. Drift score > threshold?                             │
│          ├─ No → sleep, continue monitoring                 │
│          └─ Yes → trigger retrain.py                        │
│                     │                                       │
│                     ▼                                       │
│  5. retrain.py:                                             │
│     a. Pull recent (drifted) production data                │
│     b. Retrain model on old + new data                      │
│     c. Evaluate vs holdout                                  │
│     d. Metrics improved?                                    │
│          ├─ No → abort, keep old model                      │
│          └─ Yes → register new version in MLflow            │
│                     │                                       │
│                     ▼                                       │
│  6. API detects new production-stage model → hot-swap       │
│     in memory → subsequent predictions use new model        │
│                     │                                       │
│                     ▼                                       │
│  7. Return to step 3 (loop continues indefinitely)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

1. **Synthetic data over real data** — enables deterministic, repeatable drift scenarios for demo & testing.
2. **Background thread for monitoring** — avoids needing a separate scheduler (cron, Airflow) in v1; keeps the deploy simple.
3. **MLflow Model Registry stages** — `"Production"` stage denotes the active model; the API polls for the latest production-stage model.
4. **Validation gate** — never deploy a model that performs worse than the current one, even if drift is detected.
5. **Thread-safe model hot-swap** — use a `threading.Lock` around the in-memory model reference so predictions are never served with a half-swapped model.

---

## Success Metrics (from README)

| Metric | How We Measure |
|--------|---------------|
| Drift Detection Accuracy | Evidently AI drift score vs. known injected drift |
| Model Accuracy Improvement | Compare eval metrics before vs. after retraining |
| Retraining Success Rate | % of retraining cycles that pass validation gate |
| Deployment Time | Time from drift detection → new model serving |
| Prediction Latency | p50/p99 inference time |
| System Availability | Uptime of FastAPI health endpoint |

---

## Future Enhancements (Post-v1)

- **Kubernetes** — deploy `api` + `mlflow` + `monitor` as separate pods
- **Kafka** — stream predictions as events for async drift analysis
- **Prometheus / Grafana** — expose drift metrics as Prometheus gauges, dashboard
- **Multi-model support** — manage multiple model registry names / endpoints
- **Explainable AI** — SHAP / LIME explanations alongside predictions
- **Cloud deployment** — AWS ECS / GCP Cloud Run
- **Auto-scaling** — scale API replicas based on prediction latency / QPS
