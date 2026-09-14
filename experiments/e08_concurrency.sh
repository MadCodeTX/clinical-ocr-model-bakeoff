#!/usr/bin/env bash
# Throughput curve: dots.mocr at concurrency 1, 4 and 16 (the main run used 8).
# Full 328-doc eval set each time so numbers are directly comparable.
source "$(dirname "$0")/_common.sh"
set -o pipefail
cleanup_containers
for c in 1 4 16; do
  CONCURRENCY=$c bash run_model.sh "dots-mocr-c$c" rednote-hilab/dots.mocr 0 \
    "Extract the text content from this image." > "logs/dots-mocr-c$c.log" 2>&1
  cleanup_containers
done
