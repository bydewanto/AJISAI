# ajisai/models/ajisai.py

import torch
import torch.nn as nn
from .base_encoder import load_base_encoder
from .projector import ProjectionHead


class AjisaiDualEncoder(nn.Module):
    """
    Ajisai Dual-Encoder Self-Supervised Model.
    Encodes two augmented views (or teacher-student pairs) and projects them
    into a shared latent space for contrastive training.
    """

    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = False,
        feature_dim: int = 512,
        proj_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()

        # Two encoders (could be weight-shared or independent)
        self.encoder_a = load_base_encoder(arch, pretrained, feature_dim)
        self.encoder_b = load_base_encoder(arch, pretrained, feature_dim)

        # Projection heads
        self.projector_a = ProjectionHead(feature_dim, hidden_dim, proj_dim)
        self.projector_b = ProjectionHead(feature_dim, hidden_dim, proj_dim)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor):
        """
        Args:
            x_a: First augmented view [B, C, H, W]
            x_b: Second augmented view [B, C, H, W]
        Returns:
            z_a, z_b: Projected embeddings (normalized)
        """
        # Encode both views
        h_a = self.encoder_a(x_a)
        h_b = self.encoder_b(x_b)

        # Flatten in case of CNN feature maps
        if h_a.ndim > 2:
            h_a = torch.flatten(h_a, start_dim=1)
        if h_b.ndim > 2:
            h_b = torch.flatten(h_b, start_dim=1)

        # Project to latent space
        z_a = self.projector_a(h_a)
        z_b = self.projector_b(h_b)

        return z_a, z_b
