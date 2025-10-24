Ajisai/
│
├── ajisai/                        # Core Python package
│   ├── __init__.py
│   │
│   ├── config/                    # Configuration files (YAML or JSON)
│   │   ├── amdim_config.yaml
│   │   ├── simclr_config.yaml
│   │   └── fewshot_config.yaml
│   │
│   ├── data/                      # Data loading and augmentation
│   │   ├── __init__.py
│   │   ├── datasets.py            # FewShotDataset, AMDIMDataset
│   │   ├── transforms.py          # AjisaiTransform (SimCLR-style)
│   │   └── dataset_utils.py       # Split, resize, normalization, etc.
│   │
│   ├── models/                    # Model architectures
│   │   ├── __init__.py
│   │   ├── ajisai.py              # Dual-encoder model definition
│   │   ├── projector.py           # Projection head (MLP)
│   │   └── base_encoder.py        # CNN / ViT backbone loaders
│   │
│   ├── losses/                    # SSL objectives
│   │   ├── __init__.py
│   │   ├── infonce.py             # InfoNCE Loss
│   │   └── ajisai_loss.py         # Custom dual-encoder loss
│   │
│   ├── trainer/                   # Training & evaluation logic
│   │   ├── __init__.py
│   │   ├── pretrain.py            # SSL training loop
│   │   ├── finetune.py            # Fine-tuning or few-shot
│   │   └── eval_utils.py          # Accuracy, F1, confusion matrix
│   │
│   ├── utils/                     # Helper functions
│   │   ├── seed.py
│   │   ├── logger.py
│   │   ├── checkpoint.py
│   │   └── visualization.py
│   │
│   └── main.py                    # Unified entry point (calls pretrain or finetune)
│
├── experiments/                   # Store experiment configs & logs
│   ├── mango_fewshot/
│   │   ├── config.yaml
│   │   └── results.json
│   └── rice_fewshot/
│       └── config.yaml
│
├── notebooks/                     # Interactive demos or analysis
│   ├── Ajisai_demo.ipynb
│   └── Visualization.ipynb
│
├── scripts/                       # CLI / automation scripts
│   ├── train_ajisai.sh
│   ├── evaluate_fewshot.sh
│   └── export_embeddings.py
│
├── requirements.txt
├── setup.py                       # (optional) for pip install -e .
├── README.md
└── LICENSE
