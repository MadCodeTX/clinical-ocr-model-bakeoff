#!/usr/bin/env bash
# Prompt sensitivity: same models, different prompts. PaddleOCR-VL with a
# natural-language instruction instead of its "OCR:" element prompt; dots.mocr
# with "OCR:" instead of its document prompt.
source "$(dirname "$0")/_common.sh"
set -o pipefail
cleanup_containers
bash run_model.sh paddleocr-vl-promptb PaddlePaddle/PaddleOCR-VL 0 \
  "Extract all text from this document image in natural reading order." \
  > logs/paddleocr-vl-promptb.log 2>&1 &
a=$!
bash run_model.sh dots-mocr-promptocr rednote-hilab/dots.mocr 1 "OCR:" \
  > logs/dots-mocr-promptocr.log 2>&1 &
b=$!
wait $a; ra=$?
wait $b; rb=$?
cleanup_containers
[ $ra -eq 0 ] && [ $rb -eq 0 ]
