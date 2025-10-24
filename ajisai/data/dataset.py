import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class AMDIMDataset(Dataset):
    """Dataset for SSL pretraining (AMDIM/Ajisai-style)."""
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


class FewShotDataset(Dataset):
    """Dataset for few-shot classification evaluation."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.669, 0.691, 0.685],
                std=[0.203, 0.193, 0.264]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]
