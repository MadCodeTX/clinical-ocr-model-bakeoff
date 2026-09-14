#!/usr/bin/env bash
# Two more mid-size models in parallel: dots.ocr (predecessor, 3B) and
# Qwen2.5-VL-7B-Instruct (mid-size general VLM reference).
source "$(dirname "$0")/_common.sh"
set -o pipefail
cleanup_containers
bash run_model.sh dots-ocr rednote-hilab/dots.ocr 0 \
  "Extract the text content from this image." > logs/dots-ocr.log 2>&1 &
a=$!
bash run_model.sh qwen25vl7b Qwen/Qwen2.5-VL-7B-Instruct 1 \
  "Extract the text content from this image." > logs/qwen25vl7b.log 2>&1 &
b=$!
wait $a; ra=$?
wait $b; rb=$?
cleanup_containers
[ $ra -eq 0 ] && [ $rb -eq 0 ]
