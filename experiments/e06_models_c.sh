#!/usr/bin/env bash
# DeepSeek-OCR (3B MoE, throughput-oriented) on GPU0; best-effort MonkeyOCRv2
# (0.7B, Apache-2.0, claims open-source SOTA on MDPBench) on GPU1.
# MonkeyOCRv2 ships its own inference stack (not vLLM-serveable), so this is an
# integration probe: clone, install, discover the entry point, run a small slice.
source "$(dirname "$0")/_common.sh"
set -o pipefail
cleanup_containers

bash run_model.sh deepseek-ocr deepseek-ai/DeepSeek-OCR 0 "Free OCR." \
  > logs/deepseek-ocr.log 2>&1 &
d=$!

(
  set -x
  mkdir -p third_party && cd third_party
  [ -d MonkeyOCRv2 ] || git clone --depth 1 \
    https://github.com/Yuliang-Liu/MonkeyOCRv2 MonkeyOCRv2
  cd MonkeyOCRv2
  ls -la
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  # MonkeyOCRv2 has no root requirements.txt; deps live per-component, and
  # download_model.py needs huggingface_hub regardless.
  .venv/bin/pip install -q huggingface_hub
  [ -f parsing/requirements.txt ] && .venv/bin/pip install -q -r parsing/requirements.txt
  echo "--- entry points:"
  find . -maxdepth 3 -name "*.py" | head -30
  echo "--- model download:"
  if [ -d model_weight/MonkeyOCRv2-B-Parsing ]; then
    echo "weights already present"
  else
    .venv/bin/python download_model.py -n MonkeyOCRv2-B-Parsing
  fi
  exit 1   # weights fetch OK; scoring still needs a bespoke runner (see PLAN.md)
) > logs/monkeyocrv2.log 2>&1 &
m=$!

wait $d; rd=$?
wait $m; rm_=$?
cleanup_containers
if [ $rm_ -ne 0 ]; then
  echo "MonkeyOCRv2 probe ended rc=$rm_ - see logs/monkeyocrv2.log (expected: needs bespoke runner)"
fi
[ $rd -eq 0 ]
