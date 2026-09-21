---
name: ocr-bench
description: Drive the clinical OCR bake-off harness on bigbox (~/ocr-bench) — check status/GPUs, read the leaderboard and REPORT.md, launch model evals on ClinOCR-Bench, and publish results to GitHub. Use whenever the user asks about OCR model comparison results, wants to test a document-parsing model, or asks to update/publish the bake-off report.
---

# OCR bake-off harness

Everything lives in `/home/nick/ocr-bench` on bigbox. The `ocr_*` tools
(registered by the `ocr-bench` extension) already know how to reach it — from
bigbox directly, or over `ssh bigbox` from the Mac. Prefer the tools over raw
shell.

## Tools

| tool | use for |
|---|---|
| `ocr_status` | first call: queue state, recent runs + exit codes, GPU memory, containers |
| `ocr_leaderboard` | scored models sorted by CER; `query` filters (e.g. `lora`, `medreal`, `dots`) |
| `ocr_report` | the full write-up `reports/REPORT.md` (leaderboard, failure analysis, router, distillation, cost) |
| `ocr_run` | evaluate a new model: name, hf_id, prompt, gpu (0/1), concurrency → background; poll `ocr_status` |
| `ocr_publish` | regenerate REPORT.md and push to GitHub |

## Conventions

- A run writes `results/<name>/summary.json` (+ `per_doc.json`,
  `predictions.jsonl`, `vllm.log`). The leaderboard includes **every**
  `results/*/summary.json`, so pick descriptive run names.
- GPU etiquette: check `ocr_status` first; one served model per GPU
  (`--gpu 0` / `--gpu 1`). Runaway containers are named `ocr-*` and can be
  removed with `docker rm -f ocr-<name>`.
- Prompts matter: use each model's recommended prompt when known
  (PaddleOCR-VL: `OCR:`; olmOCR-2: `Extract the contents. [Markdown].`;
  dots.mocr: `Extract the text content from this image.`;
  granite-docling: `Convert this page to docling.`).
- Eval set is fixed (ClinOCR-Bench test, 328 docs) so numbers are comparable
  across everything in `results/`. Don't re-run the same name — pick a new one.
- After runs finish, call `ocr_publish` so the GitHub report is current.

## Deeper analysis

- per-document CER: `results/<name>/per_doc.json`
- field fidelity (dates/MRNs/codes): `reports/field_fidelity.json`
- routing study: `results/router*/summary.json`
- worst pages: `reports/worst_docs.md`
- train a student: `train_student.py` (QLoRA Qwen2.5-VL-3B on synthetic docs)
- repo: https://github.com/MadCodeTX/clinical-ocr-model-bakeoff
