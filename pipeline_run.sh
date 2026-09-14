#!/usr/bin/env bash
set -e
cd ~/ocr-bench
bash serve_model.sh paddleocr-vl PaddlePaddle/PaddleOCR-VL 0
paddle-venv/bin/python3 eval_paddle_pipeline.py --out results/paddleocr-vl-pipeline --server http://localhost:8000/v1
docker rm -f ocr-paddleocr-vl
