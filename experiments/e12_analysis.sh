#!/usr/bin/env bash
# Extra analyses: per-template breakdown, CER vs image-quality correlates,
# worst-document inspection, failure taxonomy.
source "$(dirname "$0")/_common.sh"
.venv/bin/python3 analysis_extras.py
