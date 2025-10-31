#!/bin/bash
# ===========================
# Ajisai Few-Shot Evaluation Script
# ===========================

CONFIG_PATH="./experiments/mango_fewshot/config.yaml"

if [ ! -z "$1" ]; then
  CONFIG_PATH=$1
fi

echo "🧠 Running few-shot evaluation for: $CONFIG_PATH"

python main.py --config $CONFIG_PATH --mode finetune

echo "🎯 Few-shot evaluation complete."

# example usage:
# bash scripts/evaluate_fewshot.sh
