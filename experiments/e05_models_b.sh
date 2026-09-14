#!/usr/bin/env bash
# Chandra OCR 2 (5B, quality reference; OpenRAIL-M research-only licence) and
# Nanonets-OCR2-3B (forms/signatures), in parallel.
source "$(dirname "$0")/_common.sh"
set -o pipefail
cleanup_containers
bash run_model.sh chandra-2 datalab-to/chandra-ocr-2 0 \
  "Convert this page to markdown." > logs/chandra-2.log 2>&1 &
a=$!
bash run_model.sh nanonets-ocr2-3b nanonets/Nanonets-OCR2-3B 1 \
  "Convert this document to markdown." > logs/nanonets-ocr2-3b.log 2>&1 &
b=$!
wait $a; ra=$?
wait $b; rb=$?
cleanup_containers
[ $ra -eq 0 ] && [ $rb -eq 0 ]
