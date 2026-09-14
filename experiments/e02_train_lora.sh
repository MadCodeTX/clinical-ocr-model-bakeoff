#!/usr/bin/env bash
# QLoRA distillation of Qwen2.5-VL-3B on the synthetic degraded clinical corpus.
source "$(dirname "$0")/_common.sh"
if [ -d checkpoints/qwen25vl3b-lora ]; then echo "adapter exists, done"; exit 0; fi
CUDA_VISIBLE_DEVICES=1 train-venv/bin/python3 train_student.py \
  --out checkpoints/qwen25vl3b-lora --epochs 1
