# Ajisai — Dual‑Encoder Self‑Supervised Learning

...existing code...

## Overview

Ajisai is a small PyTorch framework for Ajisai/AMDIM-style dual-encoder self-supervised learning (SSL) with utilities for few-shot evaluation and embedding export.

Core ideas:

- Train two encoders to produce aligned latent embeddings for different views of the same image.
- Use contrastive losses (InfoNCE / Ajisai dual loss).
- Extract embeddings and evaluate few-shot classification with a simple classifier.

## Quick links (open these files/symbols)

- Entry point: [main.py](main.py)
- Pretraining loop: [`ajisai.trainer.pretrain_ssl`](ajisai/trainer/pretrain.py) — [ajisai/trainer/pretrain.py](ajisai/trainer/pretrain.py)
- Few‑shot fine‑tuning: [`ajisai.trainer.finetune_fewshot`](ajisai/trainer/finetune.py) — [ajisai/trainer/finetune.py](ajisai/trainer/finetune.py)
- Evaluation helpers: [`ajisai.trainer.eval_utils.evaluate_model`](ajisai/trainer/eval_utils.py), [`ajisai.trainer.eval_utils.compute_metrics`](ajisai/trainer/eval_utils.py) — [ajisai/trainer/eval_utils.py](ajisai/trainer/eval_utils.py)

Data & augmentation:

- Transform: [`ajisai.data.AjisaiTransform`](ajisai/data/transform.py) — [ajisai/data/transform.py](ajisai/data/transform.py)
- Datasets: [`ajisai.data.AMDIMDataset`](ajisai/data/dataset.py), [`ajisai.data.FewShotDataset`](ajisai/data/dataset.py) — [ajisai/data/dataset.py](ajisai/data/dataset.py)
- Few‑shot split helper: [`ajisai.data.dataset_utils.create_fewshot_split`](ajisai/data/dataset_utils.py) — [ajisai/data/dataset_utils.py](ajisai/data/dataset_utils.py)

Models & projection:

- Dual encoder: [`ajisai.models.ajisai.AjisaiDualEncoder`](ajisai/models/ajisai.py) — [ajisai/models/ajisai.py](ajisai/models/ajisai.py)
- Backbone loader: [`ajisai.models.base_encoder.load_base_encoder`](ajisai/models/base_encoder.py) — [ajisai/models/base_encoder.py](ajisai/models/base_encoder.py)
- Projection head: [`ajisai.models.projector.ProjectionHead`](ajisai/models/projector.py) — [ajisai/models/projector.py](ajisai/models/projector.py)

Losses:

- InfoNCE: [`ajisai.losses.InfoNCELoss`](ajisai/losses/infonce.py) — [ajisai/losses/infonce.py](ajisai/losses/infonce.py)
- Ajisai dual loss: [`ajisai.losses.AjisaiDualLoss`](ajisai/losses/ajisai_loss.py) — [ajisai/losses/ajisai_loss.py](ajisai/losses/ajisai_loss.py)

Extras:

- Example / notebook: [ajisai.py](ajisai.py)
- Scripts: [scripts/train_ajisai.sh](scripts/train_ajisai.sh), [scripts/export_embeddings.py](scripts/export_embeddings.py)
- Requirements: [requirements.txt](requirements.txt)