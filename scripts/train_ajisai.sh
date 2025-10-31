#!/bin/bash
# ===========================
# Ajisai SSL Training Script
# ===========================

# Default configuration file
CONFIG_PATH="./experiments/mango_fewshot/config.yaml"

# Allow custom config override
if [ ! -z "$1" ]; then
  CONFIG_PATH=$1
fi

echo "🚀 Starting Ajisai pretraining using config: $CONFIG_PATH"

# Activate environment (optional, depends on user)
# source venv/bin/activate

python main.py --config $CONFIG_PATH --mode pretrain

echo "✅ Pretraining finished."

# example usage:
# bash scripts/train_ajisai.sh
# # or for another experiment
# bash scripts/train_ajisai.sh ./experiments/rice_fewshot/config.yaml
