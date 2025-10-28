# ajisai/models/projector.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Simple 2-layer MLP projection head for contrastive learning.
    (Used in SimCLR, AMDIM, Ajisai, etc.)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128, use_bn: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.normalize(x, dim=1)
        return x
