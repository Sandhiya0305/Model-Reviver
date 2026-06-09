import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> dict:
    device = next(model.parameters()).device
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1) if outputs.shape[1] > 1 else (outputs.squeeze() > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
