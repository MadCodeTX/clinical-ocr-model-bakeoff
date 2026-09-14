#!/usr/bin/env bash
# Return the box to its original state: free the GPUs and restart the user's
# Qwen3.8-27B vLLM deployment that was stopped before benchmarking.
source "$(dirname "$0")/_common.sh"
cleanup_containers
if docker ps -a --format '{{.Names}}' | grep -q '^vllm-Qwen3.8-27B-FP8$'; then
  docker start vllm-Qwen3.8-27B-FP8 && echo "restarted original vLLM deployment"
  sleep 30
else
  echo "original container not found; GPUs left idle"
fi
docker ps --format '{{.Names}}: {{.Status}}'
