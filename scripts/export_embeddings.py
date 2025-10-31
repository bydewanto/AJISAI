import torch
import os
from utils.config_loader import load_config
from models.ajisai import AjisaiModel
from data.datasets import FewShotDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def export_embeddings(config_path="./experiments/mango_fewshot/config.yaml"):
    config = load_config(config_path)

    device = torch.device(config[   "training"]["device"])
    print(f"🔍 Loading model on {device}...")

    model = AjisaiModel(encoder_name=config["model"]["encoder"],
                        projector_dim=config["model"]["projector_dim"])
    checkpoint_path = os.path.join(config["logging"]["save_dir"], "best_model.pth")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval().to(device)

    dataset = FewShotDataset(config["dataset"]["val_dir"], config["dataset"]["image_size"])
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_features, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting embeddings"):
            feats = model.encoder(imgs.to(device))
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    embeddings = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    np.save(os.path.join(config["logging"]["save_dir"], "embeddings.npy"), embeddings)
    np.save(os.path.join(config["logging"]["save_dir"], "labels.npy"), labels)

    print("✅ Embeddings exported successfully!")

if __name__ == "__main__":
    export_embeddings()
