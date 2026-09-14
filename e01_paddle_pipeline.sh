#!/usr/bin/env bash
# PaddleOCR-VL full pipeline: PP-DocLayoutV2 layout (+ orientation classify,
# unwarping) -> per-region VLM calls against a vLLM server -> markdown.
source "$(dirname "$0")/_common.sh"
set -o pipefail
cleanup_containers
# the pipeline resolves the served name itself (e.g. "PaddleOCR-VL-1.6-0.9B"),
# so expose the same weights under several aliases
# NOTE: layout + VLM recognition only. The optional doc-orientation
# classification sub-model crashes the CPU CV worker on ~30% of these scans
# (verified: 7/24 failures with it on, 0/24 with it off), so it is disabled.
EXTRA_SERVED="PaddleOCR-VL-1.6-0.9B PaddleOCR-VL" \
  bash serve_model.sh paddleocr-vl PaddlePaddle/PaddleOCR-VL 0
paddle-venv/bin/python3 eval_paddle_pipeline.py \
  --out results/paddleocr-vl-pipeline --server http://localhost:8000/v1 \
  --no-orient --no-unwarp
cleanup_containers
score paddleocr-vl-pipeline
