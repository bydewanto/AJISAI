# ajisai/models/base_encoder.py

import torch
import torch.nn as nn
from torchvision import models


def load_base_encoder(arch: str = "resnet18", pretrained: bool = False, output_dim: int = 512) -> nn.Module:
    """
    Loads a CNN or ViT backbone encoder.

    Args:
        arch: Encoder architecture name ("resnet18", "resnet50", "vit_b_16", etc.)
        pretrained: Whether to use ImageNet pretraining.
        output_dim: Output feature dimension from the encoder.

    Returns:
        nn.Module: Encoder network without classification head.
    """
    arch = arch.lower()

    if "resnet" in arch:
        model = getattr(models, arch)(weights="IMAGENET1K_V1" if pretrained else None)
        modules = list(model.children())[:-1]  # remove fc layer
        encoder = nn.Sequential(*modules)
        feature_dim = model.fc.in_features

    elif "vit" in arch:
        from torchvision.models.vision_transformer import vit_b_16, vit_b_32
        if arch == "vit_b_16":
            model = vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
        elif arch == "vit_b_32":
            model = vit_b_32(weights="IMAGENET1K_V1" if pretrained else None)
        else:
            raise ValueError(f"Unsupported ViT architecture: {arch}")
        encoder = model
        feature_dim = model.heads.head.in_features
        model.heads = nn.Identity()

    else:
        raise ValueError(f"Unsupported encoder architecture: {arch}")

    # Add optional projection layer to adjust output dim
    if output_dim != feature_dim:
        projection = nn.Linear(feature_dim, output_dim)
        encoder = nn.Sequential(encoder, nn.Flatten(), projection)
    else:
        encoder = nn.Sequential(encoder, nn.Flatten())

    return encoder
