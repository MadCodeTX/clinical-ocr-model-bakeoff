#!/usr/bin/env bash
# One-shot env setup on bigbox.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q datasets pillow rapidfuzz

# tesseract baseline (needs sudo; skip if already present)
if ! command -v tesseract >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq tesseract-ocr
fi
tesseract --version | head -1
.venv/bin/python -c "import datasets, PIL, rapidfuzz; print('python deps OK')"
