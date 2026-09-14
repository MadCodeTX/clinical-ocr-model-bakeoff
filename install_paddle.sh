#!/usr/bin/env bash
set -x
cd ~/ocr-bench
python3 -m venv paddle-venv
. paddle-venv/bin/activate
pip install -q --upgrade pip
pip install -q paddlepaddle
pip install -q paddleocr
python -c "import paddle,paddleocr;print(\"PADDLE ENV OK\", paddle.__version__, paddleocr.__version__)" 2>&1 | tail -3
