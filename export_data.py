#!/usr/bin/env python3
"""Export ClinOCR-Bench (MIT licensed) to images + jsonl for the eval harness.

https://huggingface.co/datasets/Daniele0025/ClinOCR-Bench
6 subsets x 64 docs = 384 total; we use the `test` split (328 docs, exemplars held out).
"""
import json
import os

from datasets import load_dataset
from PIL import Image

SUBSETS = ["normal", "handwriting", "poor", "rotated", "tables", "mixed"]
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "clinocr")
os.makedirs(OUT, exist_ok=True)

rows = []
for subset in SUBSETS:
    ds = load_dataset("Daniele0025/ClinOCR-Bench", subset)["test"]
    img_dir = os.path.join(OUT, "images", subset)
    os.makedirs(img_dir, exist_ok=True)
    for item in ds:
        doc_id = item["doc_id"]
        img_path = os.path.join(img_dir, f"{doc_id}.jpg")
        if not os.path.exists(img_path):
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path, "JPEG", quality=92)
        rows.append({
            "doc_id": doc_id,
            "subset": subset,
            "template": item["template"],
            "image": os.path.relpath(img_path, ROOT),
            "ground_truth": item["ground_truth"],
        })
    print(f"[{subset}] exported {len(ds)} docs")

with open(os.path.join(OUT, "eval.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"Total: {len(rows)} eval documents -> {OUT}/eval.jsonl")
