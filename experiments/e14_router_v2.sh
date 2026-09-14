#!/usr/bin/env bash
# Router variants: escalate PaddleOCR-VL -> olmOCR-2, and dots.mocr -> olmOCR-2.
source "$(dirname "$0")/_common.sh"
.venv/bin/python3 - <<'PY'
import sys
sys.path.insert(0, "/home/nick/ocr-bench")
from router import main
r1 = main("paddleocr-vl", "olmocr-2", 0.20, "/home/nick/ocr-bench/results/router-paddle-olmocr")
r2 = main("dots-mocr", "olmocr-2", 0.20, "/home/nick/ocr-bench/results/router-dots-olmocr")
print("router variants done", r1, r2)
PY
