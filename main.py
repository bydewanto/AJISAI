import argparse
import torch
import yaml
import os
from utils.seed import set_seed
from utils.logger import get_logger
from trainer.pretrain import pretrain_ssl
from trainer.finetune import finetune_fewshot
from scripts.export_embeddings import export_embeddings

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="🌸 Ajisai SSL Framework Entry Point")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "finetune", "export"], help="Run mode")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    # Prepare logging directory
    log_dir = config.get("logging", {}).get("save_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(os.path.join(log_dir, "ajisai.log"))

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Mode selection
    if args.mode == "pretrain":
        logger.info("Starting pretraining 🌱")
        pretrain_ssl(config, device, logger)

    elif args.mode == "finetune":
        logger.info("Starting few-shot fine-tuning 🌼")
        finetune_fewshot(config, device, logger)

    elif args.mode == "export":
        logger.info("Exporting embeddings 💠")
        export_embeddings(config_path=args.config)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    logger.info("✨ Task completed successfully!")

if __name__ == "__main__":
    main()
