#!/usr/bin/env bash
set -e
cd ~/ocr-bench
bash serve_model.sh olmocr-2 allenai/olmOCR-2-7B-1025 1
.venv/bin/python3 label_synth.py --endpoint http://localhost:8001 --prompt "Extract the contents. [Markdown]." --n 200 --out results/teacher-labels
docker rm -f ocr-olmocr-2
