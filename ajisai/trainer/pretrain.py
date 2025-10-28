# ajisai/trainer/pretrain.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ajisai.losses import InfoNCELoss, AjisaiDualLoss
from ajisai.utils import logger, checkpoint


def pretrain_ssl(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 100,
    loss_fn: str = "ajisai",
    temperature: float = 0.5,
    alpha: float = 1.0,
    log_interval: int = 10,
    save_path: str = "checkpoints/ajisai_ssl.pth",
):
    """
    Pretrain Ajisai dual-encoder model using self-supervised learning.

    Args:
        model: AjisaiDualEncoder model.
        dataloader: Dataloader providing (view1, view2) pairs.
        optimizer: Optimizer (Adam, LARS, etc.).
        device: Computation device.
        epochs: Number of training epochs.
        loss_fn: "infonce" or "ajisai".
    """
    model.to(device)
    model.train()

    if loss_fn.lower() == "infonce":
        criterion = InfoNCELoss(temperature)
    else:
        criterion = AjisaiDualLoss(temperature, alpha)

    log = logger.get_logger("Ajisai Pretrain")
    log.info("Starting Ajisai SSL pretraining...")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        for batch_idx, (x_a, x_b) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            x_a, x_b = x_a.to(device), x_b.to(device)

            optimizer.zero_grad()
            z_a, z_b = model(x_a, x_b)
            loss = criterion(z_a, z_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % log_interval == 0:
                log.info(f"Epoch [{epoch}/{epochs}] | Step [{batch_idx}] | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        log.info(f"Epoch [{epoch}/{epochs}] | Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint.save_model(model, optimizer, epoch, avg_loss, save_path)

    log.info("Pretraining completed successfully.")
    return model
