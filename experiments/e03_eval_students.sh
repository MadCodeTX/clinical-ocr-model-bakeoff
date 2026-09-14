#!/usr/bin/env bash
# Evaluate base Qwen2.5-VL-3B and the LoRA student on ClinOCR-Bench (one per GPU).
source "$(dirname "$0")/_common.sh"
set -o pipefail
cleanup_containers
CUDA_VISIBLE_DEVICES=0 train-venv/bin/python3 eval_hf.py --out results/qwen25vl3b-base &
p0=$!
p1=0
if [ -d checkpoints/qwen25vl3b-lora ]; then
  CUDA_VISIBLE_DEVICES=1 train-venv/bin/python3 eval_hf.py \
    --out results/qwen25vl3b-lora --adapter checkpoints/qwen25vl3b-lora &
  p1=$!
fi
wait $p0; r0=$?
[ "$p1" != "0" ] && { wait $p1; r1=$?; } || r1=0
score qwen25vl3b-base
[ -d results/qwen25vl3b-lora ] && score qwen25vl3b-lora
[ $r0 -eq 0 ] && [ $r1 -eq 0 ]
