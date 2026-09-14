# CANE OCR Model Bake-off (bigbox, Sept 2026)

Goal: replace/augment Tesseract and avoid Azure Layout ($0.01/page) for Care
Everywhere scanned-document parsing. Benchmark open-weights OCR VLMs of varying
size on real-artifact clinical scans, measure quality + throughput on 2x4090.

## Test data
**ClinOCR-Bench** (MIT license) — 328 eval docs (test split) across 6 artifact
subsets: normal, handwriting, poor, rotated, tables, mixed. 16 clinical
templates (referral letters, pathology reports, fax cover sheets, lab reports),
degraded with real-world artifacts (crumpling, low-res, rotation), with
human-audited ground truth. No PHI — synthetic content, real degradation.
Source: huggingface.co/datasets/Daniele0025/ClinOCR-Bench (arXiv 2607.03650)

## Models (size-magnitude ladder, all vLLM-servable)
| # | Model | Params | License | Prompt |
|---|-------|--------|---------|--------|
| 1 | ibm-granite/granite-docling-258M | 0.26B | Apache-2.0 | "Convert this page to docling." |
| 2 | PaddlePaddle/PaddleOCR-VL | 0.9B | Apache-2.0 | "OCR:" |
| 3 | rednote-hilab/dots.mocr | 3B | MIT | prompt_ocr ("Extract the text content from this image.") |
| 4 | allenai/olmOCR-2-7B-1025 | 8B | Apache-2.0 | "Extract the contents. [Markdown]." |
| 0 | Tesseract v5 (baseline) | - | Apache-2.0 | CLI, psm 3 |

Excluded: Chandra OCR 2 (OpenRAIL-M, restricted to orgs <$2M revenue),
MinerU 2.5 (AGPL), MonkeyOCRv2 (own inference stack, not vLLM-serveable yet —
future run), DeepSeek-OCR (weaker benchmark scores than the above at similar size).

## Metrics
- **CER** (character error rate) after normalization: lowercase, strip
  markdown/XML markup, collapse whitespace. Mean + median over 328 docs,
  plus per-subset breakdown (which artifact kills which model).
- **pages/sec** sustained at concurrency 8 on one RTX 4090.
- Errors/exceptions counted separately.

## Hardware
bigbox: 2x RTX 4090 24GB, i9-14900K, 94GB RAM. vLLM v0.27.1 in docker,
one model per GPU, two models in parallel per round.

## Reproduce
```
bash setup_env.sh          # venv, python deps, tesseract
python3 export_data.py     # pulls ClinOCR-Bench test split -> data/clinocr/
bash run_tesseract.sh
bash run_model.sh granite-docling ibm-granite/granite-docling-258M 0 "Convert this page to docling." &
bash run_model.sh paddleocr-vl PaddlePaddle/PaddleOCR-VL 1 "OCR:" & wait
bash run_model.sh dots-mocr rednote-hilab/dots.mocr 0 "Extract the text content from this image." &
bash run_model.sh olmocr-2 allenai/olmOCR-2-7B-1025 1 "Extract the contents. [Markdown]." & wait
python3 report.py          # combine summaries
```
