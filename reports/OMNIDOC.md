# OmniDocBench generalisation (A3)

The ClinOCR-Bench ranking re-measured on 1651 real PDF pages (OmniDocBench, Apache-2.0, research use only). Synthetic content with real degradation -> real documents with real layouts.

**Caveat:** OmniDocBench is saturated and its rigid metrics penalise semantically-correct formatting, so compare the *ranking* across models, not the absolute CER against reports/REPORT.md.

| model | median CER | mean CER | excl-runaway | runaways | n docs |
|---|---|---|---|---|---|
| qwen38-27b | 0.7368 | 0.8716 | 0.7140 | 202 | 1649 |

