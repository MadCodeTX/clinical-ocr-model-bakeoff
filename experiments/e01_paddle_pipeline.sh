#!/usr/bin/env bash
# PaddleOCR-VL full pipeline: PP-DocLayoutV2 layout (+ orientation classify,
# unwarping) -> per-region VLM calls against a vLLM server -> markdown.
source "$(dirname "$0")/_common.sh"
set -o pipefail
cleanup_containers
bash serve_model.sh paddleocr-vl PaddlePaddle/PaddleOCR-VL 0
paddle-venv/bin/python3 eval_paddle_pipeline.py \
  --out results/paddleocr-vl-pipeline --server http://localhost:8000/v1
cleanup_containers
score paddleocr-vl-pipeline
