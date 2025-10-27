# ajisai/losses/ajisai_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AjisaiDualLoss(nn.Module):
    """
    Ajisai Dual-Encoder Contrastive Loss.
    Encourages aligned representations between encoder A and encoder B
    while maintaining separation between distinct samples.
    """

    def __init__(self, temperature: float = 0.5, alpha: float = 1.0):
        """
        Args:
            temperature: Scaling factor for similarity logits.
            alpha: Weight for balancing symmetric loss terms.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_a: Embeddings from encoder A [batch_size, dim].
            z_b: Embeddings from encoder B [batch_size, dim].
        Returns:
            Scalar Ajisai dual loss.
        """
        batch_size = z_a.size(0)

        # Normalize both views
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)

        # Cross-similarity between encoders
        sim_ab = torch.matmul(z_a, z_b.T) / self.temperature
        sim_ba = torch.matmul(z_b, z_a.T) / self.temperature

        # Positive pairs are along the diagonal
        labels = torch.arange(batch_size, device=z_a.device)

        # Compute symmetric loss
        loss_ab = F.cross_entropy(sim_ab, labels)
        loss_ba = F.cross_entropy(sim_ba, labels)

        loss = (loss_ab + self.alpha * loss_ba) / (1 + self.alpha)

        return loss
