import os
import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import random
from pathlib import Path
from shutil import copy2

# Set your dataset path here
data_root = Path("/kaggle/input/kengo-k423/MangoLeafBD Dataset")
output_root = Path("/kaggle/working/fewshot_dataset")
support_dir = output_root / "support"
query_dir = output_root / "query"

# Create folders
for folder in [support_dir, query_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# Get list of classes with enough images
classes = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
eligible_classes = [cls for cls in classes if len(list((data_root / cls).glob("*"))) >= 6]

# Sample 5 classes (5-way)
selected_classes = random.sample(eligible_classes, 5)

fewshot_split = {}
for cls in selected_classes:
    class_path = data_root / cls
    images = sorted([f for f in class_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    sampled = random.sample(images, 6)  # 1 support, 5 query

    fewshot_split[cls] = {
        "support": sampled[0].name,
        "query": [img.name for img in sampled[1:]]
    }

    # Save support image
    (support_dir / cls).mkdir(parents=True, exist_ok=True)
    copy2(sampled[0], support_dir / cls / sampled[0].name)

    # Save query images
    (query_dir / cls).mkdir(parents=True, exist_ok=True)
    for img in sampled[1:]:
        copy2(img, query_dir / cls / img.name)

# Print the split info
import json
print(json.dumps(fewshot_split, indent=2))

# === Dataset ===
data_path = "/kaggle/input/kengo-k423/MangoLeafBD Dataset"
classes = sorted(os.listdir(data_path))

def create_splits(path, test_size=0.3, seed=42):
    image_paths, labels = [], []
    for label, class_name in enumerate(classes):
        class_path = os.path.join(path, class_name)
        for img_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_name))
            labels.append(label)
    return image_paths, labels

X_all, y_all = create_splits(data_path)

# === AMDIM-style Transform (Improved) ===
class AMDIMTransform:
    def __init__(self):
        self.global_crop = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.669, 0.691, 0.685],
                                std = [0.203, 0.193, 0.264])
        ])
        self.local_crop = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.3, 0.6)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.669, 0.691, 0.685],
                                std = [0.203, 0.193, 0.264])
        ])

    def __call__(self, img):
        return self.global_crop(img), self.local_crop(img)

# === AMDIM Dataset ===
class AMDIMDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)

# === Mutual Info Loss ===
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_global, z_local):
        batch_size = z_global.size(0)
        z_global = F.normalize(z_global, dim=1)
        z_local = F.normalize(z_local, dim=1)
        logits = torch.mm(z_global, z_local.t()) / self.temperature
        labels = torch.arange(batch_size).to(z_global.device)
        return F.cross_entropy(logits, labels)

# === AMDIM Model with Stronger Projection ===
class AMDIM(nn.Module):
    def __init__(self, global_encoder, local_encoder):
        super().__init__()
        self.global_encoder = global_encoder
        self.local_encoder = local_encoder
        self.global_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128)
        )
        self.local_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128)
        )

    def forward(self, global_x, local_x):
        g_feat = self.global_encoder(global_x).view(global_x.size(0), -1)
        l_feat = self.local_encoder(local_x).view(local_x.size(0), -1)
        g_out = self.global_head(g_feat)
        l_out = self.local_head(l_feat)
        return g_out, l_out

# === Init Model ===
resnet_global = models.resnet50(weights=None)
resnet_local = models.resnet18(weights=None)
backbone_g = nn.Sequential(*list(resnet_global.children())[:-1])
backbone_l = nn.Sequential(*list(resnet_local.children())[:-1])
model = AMDIM(backbone_g, backbone_l).cuda()

transform = AMDIMTransform()
dataset = AMDIMDataset(X_all, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * 50)
criterion = InfoNCELoss()

# === Train AMDIM ===
def train_amdim(model, dataloader, optimizer, scheduler, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for global_x, local_x in dataloader:
            global_x, local_x = global_x.cuda(), local_x.cuda()
            g_out, l_out = model(global_x, local_x)
            loss = criterion(g_out, l_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(dataloader):.4f}")

train_amdim(model, dataloader, optimizer, scheduler, criterion, epochs=50)

# mango
# Dataset Mean (RGB): [0.669, 0.691, 0.685]
# Dataset Std (RGB): [0.203, 0.193, 0.264]
# 

from pathlib import Path
import random
from shutil import copy2

src_dir = Path("/kaggle/input/kengo-k423/MangoLeafBD Dataset")
support_dir = Path("/kaggle/working/fewshot_dataset/support")
query_dir = Path("/kaggle/working/fewshot_dataset/query")

support_dir.mkdir(parents=True, exist_ok=True)
query_dir.mkdir(parents=True, exist_ok=True)

# Pick 5 classes with enough images (≥10 images per class recommended)
classes = sorted([cls.name for cls in src_dir.iterdir() if cls.is_dir() and len(list(cls.glob("*"))) >= 20])
selected_classes = random.sample(classes, 5)

fewshot_split = {}

for cls in selected_classes:
    images = list((src_dir / cls).glob("*"))
    selected_imgs = random.sample(images, 20)  # 5 for support, 15 for query

    # Save support
    (support_dir / cls).mkdir(parents=True, exist_ok=True)
    for img in selected_imgs[:5]:
        copy2(img, support_dir / cls / img.name)

    # Save query
    (query_dir / cls).mkdir(parents=True, exist_ok=True)
    for img in selected_imgs[5:]:
        copy2(img, query_dir / cls / img.name)

    fewshot_split[cls] = {
        "support": [img.name for img in selected_imgs[:5]],
        "query": [img.name for img in selected_imgs[5:]]
    }

import json
print(json.dumps(fewshot_split, indent=2))

from pathlib import Path
import random
from shutil import copy2

src_dir = Path("/kaggle/input/kengo-k423/MangoLeafBD Dataset")
support_dir = Path("/kaggle/working/fewshot_dataset/support")
query_dir = Path("/kaggle/working/fewshot_dataset/query")

support_dir.mkdir(parents=True, exist_ok=True)
query_dir.mkdir(parents=True, exist_ok=True)

# Pick 5 classes with enough images (≥10 images per class recommended)
classes = sorted([cls.name for cls in src_dir.iterdir() if cls.is_dir() and len(list(cls.glob("*"))) >= 20])
selected_classes = random.sample(classes, 5)

fewshot_split = {}

for cls in selected_classes:
    images = list((src_dir / cls).glob("*"))
    selected_imgs = random.sample(images, 20)  # 5 for support, 15 for query

    # Save support
    (support_dir / cls).mkdir(parents=True, exist_ok=True)
    for img in selected_imgs[:5]:
        copy2(img, support_dir / cls / img.name)

    # Save query
    (query_dir / cls).mkdir(parents=True, exist_ok=True)
    for img in selected_imgs[5:]:
        copy2(img, query_dir / cls / img.name)

    fewshot_split[cls] = {
        "support": [img.name for img in selected_imgs[:5]],
        "query": [img.name for img in selected_imgs[5:]]
    }

import json
print(json.dumps(fewshot_split, indent=2))

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import torch

# === Few-Shot Dataset Class ===
class FewShotDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.669, 0.691, 0.685],
                                std = [0.203, 0.193, 0.264])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

# === Helper: Extract image paths and numeric labels from folder ===
def get_fewshot_image_paths(base_dir):
    paths, labels, label_map = [], [], {}
    base_dir = Path(base_dir)
    for idx, class_dir in enumerate(sorted(base_dir.iterdir())):
        if class_dir.is_dir():
            label_map[idx] = class_dir.name
            for img_path in class_dir.glob("*"):
                paths.append(str(img_path))
                labels.append(idx)
    return paths, labels, label_map

# === Helper: Feature extraction ===
def extract_features(encoder, dataloader):
    encoder.eval()
    feats, lbls = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.cuda()
            feat = encoder(x).view(x.size(0), -1)
            feats.append(feat.cpu())
            lbls.append(y)
    return torch.cat(feats), torch.cat(lbls)

# === Load Few-Shot Datasets ===
support_paths, support_labels, label_map = get_fewshot_image_paths("/kaggle/working/fewshot_dataset/support")
query_paths, query_labels, _ = get_fewshot_image_paths("/kaggle/working/fewshot_dataset/query")

support_loader = DataLoader(FewShotDataset(support_paths, support_labels), batch_size=32)
query_loader = DataLoader(FewShotDataset(query_paths, query_labels), batch_size=32)

# === Extract Features from Ajisai's Global Encoder ===
encoder = model.global_encoder.cuda()
X_support, y_support = extract_features(encoder, support_loader)
X_query, y_query = extract_features(encoder, query_loader)

# === Train + Evaluate Logistic Regression ===
clf = LogisticRegression(max_iter=5000)
clf.fit(X_support, y_support)
y_pred = clf.predict(X_query)

# === Report Metrics ===
acc = accuracy_score(y_query, y_pred)
precision = precision_score(y_query, y_pred, average='macro', zero_division=0)
recall = recall_score(y_query, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_query, y_pred, average='macro', zero_division=0)

print("\n🔍 Few-Shot Evaluation Results")
print(f"Accuracy      : {acc:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"Class Mapping : {label_map}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === Confusion Matrix ===
cm = confusion_matrix(y_query, y_pred)
class_names = [label_map[i] for i in sorted(label_map.keys())]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()