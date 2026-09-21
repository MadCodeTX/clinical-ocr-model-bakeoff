#!/usr/bin/env python3
"""Sample a large real-document corpus from RVL-CDIP (chainyo/rvl-cdip) for the
data-scale experiments. These are real scanned business/administrative documents
(letter, form, invoice, handwritten, report, ...) -- a scale proxy for clinical
scans when we only have a frontier teacher to label them.

Usage: python3 build_rvl.py --n 6000 --out data/rvl
"""
import argparse
import json
import os

from datasets import load_dataset
from PIL import Image

CLASSES = ["letter", "form", "email", "handwritten", "advertisement",
           "scientific_report", "scientific_publication", "specification",
           "file_folder", "news_article", "budget", "invoice",
           "presentation", "questionnaire", "resume", "memo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default="data/rvl")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    per_class = max(1, args.n // len(CLASSES))
    manifest, got = [], 0
    # one streaming pass, keeping up to per_class per class
    counts = {i: 0 for i in range(len(CLASSES))}
    for row in load_dataset("chainyo/rvl-cdip", split="train", streaming=True):
        lab = row.get("label")
        if lab is None or counts.get(lab, 0) >= per_class:
            continue
        img = row.get("image")
        if img is None:
            continue
        doc_id = f"rvl_{got:05d}"
        path = os.path.join(img_dir, doc_id + ".jpg")
        if not os.path.exists(path):
            (img.convert("RGB") if img.mode != "RGB" else img).save(path, "JPEG", quality=90)
        manifest.append({"doc_id": doc_id, "source": CLASSES[lab] if lab < len(CLASSES) else str(lab),
                         "image": os.path.relpath(path, os.path.dirname(args.out) or ".")})
        counts[lab] += 1
        got += 1
        if got >= args.n:
            break
        if got % 500 == 0:
            print(f"  {got} saved", flush=True)
    with open(os.path.join(args.out, "manifest.jsonl"), "w") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    print(f"saved {len(manifest)} pages -> {args.out}/manifest.jsonl")
    print("per class:", {CLASSES[k]: v for k, v in counts.items() if v})


if __name__ == "__main__":
    main()