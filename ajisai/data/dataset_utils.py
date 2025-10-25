import os
import random
from pathlib import Path
from shutil import copy2

def create_splits(data_path, min_images=6):
    """Create train/test splits for pretraining."""
    classes = sorted([d.name for d in Path(data_path).iterdir() if d.is_dir()])
    eligible = [cls for cls in classes if len(list((Path(data_path) / cls).glob('*'))) >= min_images]
    return eligible


def create_fewshot_split(src_dir, output_dir, n_way=5, n_support=5, n_query=15):
    """
    Randomly split dataset into few-shot format:
    - n_way classes
    - n_support support images per class
    - n_query query images per class
    """
    src_dir = Path(src_dir)
    support_dir = Path(output_dir) / "support"
    query_dir = Path(output_dir) / "query"

    support_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted([cls.name for cls in src_dir.iterdir() if cls.is_dir() and len(list(cls.glob('*'))) >= (n_support + n_query)])
    selected = random.sample(classes, n_way)

    fewshot_split = {}
    for cls in selected:
        images = list((src_dir / cls).glob("*"))
        selected_imgs = random.sample(images, n_support + n_query)

        # Copy to support
        (support_dir / cls).mkdir(parents=True, exist_ok=True)
        for img in selected_imgs[:n_support]:
            copy2(img, support_dir / cls / img.name)

        # Copy to query
        (query_dir / cls).mkdir(parents=True, exist_ok=True)
        for img in selected_imgs[n_support:]:
            copy2(img, query_dir / cls / img.name)

        fewshot_split[cls] = {
            "support": [img.name for img in selected_imgs[:n_support]],
            "query": [img.name for img in selected_imgs[n_support:]]
        }

    return fewshot_split
