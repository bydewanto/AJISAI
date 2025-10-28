# ajisai/trainer/eval_utils.py

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_model(encoder, classifier, dataloader, device):
    """
    Evaluate encoder + classifier on validation data.
    """
    encoder.eval()
    classifier.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            feats = encoder(x)
            if feats.ndim > 2:
                feats = torch.flatten(feats, start_dim=1)

            logits = classifier(feats)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return acc, f1


def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, F1, and confusion matrix directly from numpy arrays.
    """
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}
