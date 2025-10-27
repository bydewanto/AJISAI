# ajisai/losses/infonce.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Standard InfoNCE loss for contrastive self-supervised learning.
    Computes similarity between positive and negative pairs.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: Tensor of shape [batch_size, dim] from view 1.
            z_j: Tensor of shape [batch_size, dim] from view 2.
        Returns:
            Scalar InfoNCE loss.
        """
        batch_size = z_i.size(0)

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature

        # Labels are indices of positives along the diagonal
        labels = torch.arange(batch_size, device=z_i.device)

        # Cross-entropy over similarity matrix
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_j = F.cross_entropy(sim_matrix.T, labels)

        return (loss_i + loss_j) / 2
