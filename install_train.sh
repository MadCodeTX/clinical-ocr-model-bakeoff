#!/usr/bin/env bash
set -x
cd ~/ocr-bench
python3 -m venv train-venv
. train-venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu126
pip install -q transformers peft accelerate qwen-vl-utils pillow numpy scikit-learn
python -c "import torch,transformers,peft,sklearn;print(\"TRAIN ENV OK\", torch.__version__, transformers.__version__)"
