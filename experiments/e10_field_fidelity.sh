#!/usr/bin/env bash
# Field-level fidelity: recall/precision of dates, MRNs, accession codes, lab
# decimals, phone numbers per model (CER hides identifier corruption).
source "$(dirname "$0")/_common.sh"
.venv/bin/python3 field_fidelity.py
