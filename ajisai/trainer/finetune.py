# ajisai/trainer/finetune.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ajisai.trainer.eval_utils import compute_metrics


def finetune_fewshot(
    encoder: nn.Module,
    classifier: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 50,
):
    """
    Fine-tune pretrained encoder for few-shot classification.

    Args:
        encoder: Frozen or trainable encoder.
        classifier: Simple classifier (e.g., LogisticRegression, Linear layer).
        train_loader: Dataloader for few-shot samples.
        val_loader: Validation dataloader.
    """
    encoder.to(device)
    classifier.to(device)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        encoder.eval()
        classifier.train()

        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Few-shot Epoch {epoch}"):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                feats = encoder(x)
                if feats.ndim > 2:
                    feats = torch.flatten(feats, start_dim=1)

            preds = classifier(feats)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        acc, f1 = evaluate_model(encoder, classifier, val_loader, device)
        print(
            f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% | F1: {f1:.2f}"
        )

    return classifier
