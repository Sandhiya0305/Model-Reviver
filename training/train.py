import json
import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from training.evaluate import evaluate


class MLP(nn.Module):
    def __init__(self, n_features: int = cfg.N_FEATURES, hidden_dim: int = cfg.HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_loaders(df: pd.DataFrame, test_size: float):
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].values.astype("float32")
    y = df["label"].values.astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=cfg.RANDOM_SEED
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, X_train, y_train, X_test, y_test


def train_model(
    df: pd.DataFrame,
    model_name: str = cfg.MODEL_REGISTRY_NAME,
    return_model: bool = False,
) -> tuple[dict, nn.Module | None]:
    train_loader, test_loader, X_train, y_train, X_test, y_test = _build_loaders(df, cfg.TEST_SPLIT)

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    for epoch in range(cfg.TRAINING_EPOCHS):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    metrics = evaluate(model, test_loader)

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": cfg.TRAINING_EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "learning_rate": cfg.LEARNING_RATE,
            "hidden_dim": cfg.HIDDEN_DIM,
            "n_features": cfg.N_FEATURES,
            "train_size": len(X_train),
            "test_size": len(X_test),
        })
        mlflow.log_metrics(metrics)

        input_example = X_train[:5]
        signature = mlflow.models.infer_signature(
            model_input=X_train,
            model_output=model(torch.tensor(X_test[:5])).detach().numpy(),
        )

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )

    # Save local fallback
    joblib.dump({"model": model, "features": cfg.N_FEATURES}, cfg.MODEL_PKL_PATH)
    with open(cfg.METADATA_PATH, "w") as f:
        json.dump({"version": "latest", "metrics": metrics}, f)

    print(f"Training complete. Metrics: {metrics}")
    return metrics, model if return_model else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(cfg.REFERENCE_DATA_PATH))
    args = parser.parse_args()

    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ModelReviver")

    df = pd.read_csv(args.data)
    train_model(df)


if __name__ == "__main__":
    main()
