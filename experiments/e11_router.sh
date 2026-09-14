#!/usr/bin/env bash
# Escalation router (cheap = PaddleOCR-VL, expensive = dots.mocr).
source "$(dirname "$0")/_common.sh"
.venv/bin/python3 router.py
