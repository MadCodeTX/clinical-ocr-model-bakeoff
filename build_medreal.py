#!/usr/bin/env python3
"""Assemble a corpus of real medical-document scans from public HF datasets.

Downloads page images to data/medreal/images/<source>/ and writes
data/medreal/manifest.jsonl with {doc_id, source, image, ref_gt?}.

`ref_gt` is whatever ground truth the source already ships (full page text for
the historical scans, structured JSON for the prescriptions). It is kept for
cross-checking the DeepSeek labels -- the training/eval label itself is the
transcription minted by label_openrouter.py.

Usage: python3 build_medreal.py [--per-source N]
"""
import argparse
import json
import os

from datasets import load_dataset
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "medreal")

# name, HF dataset, split, default cap, how to render ref_gt (or None)
SOURCES = [
    ("noisy-med", "hmnshudhmn24/noisy-medical-document-images-ocr", "train", 400, None),
    ("prescription", "Technoculture/medical-prescriptions", "train", 200, "json"),
    ("india-hist", "davanstrien/india-medical-ocr-test", "train", 50, "text"),
    ("medform", "Ronysalem/medical-forms-dataset", "train", 8, "gt"),
]


def ref_gt(row, kind):
    if kind is None:
        return None
    if kind == "json":
        try:
            d = row.get("json")
            d = json.loads(d) if isinstance(d, str) else d
            return json.dumps(d, indent=1) if isinstance(d, dict) else str(d)
        except Exception:
            return None
    if kind == "text":
        return row.get("text")
    if kind == "gt":
        g = row.get("ground_truth")
        return json.dumps(g) if not isinstance(g, str) else g
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=0, help="override every cap")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for name, ds_id, split, cap, gt_kind in SOURCES:
        if args.per_source:
            cap = args.per_source
        img_dir = os.path.join(OUT, "images", name)
        os.makedirs(img_dir, exist_ok=True)
        got = 0
        try:
            ds = load_dataset(ds_id, split=split, streaming=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] SKIP: cannot load {ds_id}: {e}")
            continue
        for i, row in enumerate(ds):
            if got >= cap:
                break
            img = row.get("image")
            if img is None:
                continue
            doc_id = f"{name}_{i:04d}"
            path = os.path.join(img_dir, f"{doc_id}.jpg")
            if not os.path.exists(path):
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(path, "JPEG", quality=92)
            manifest.append({
                "doc_id": doc_id,
                "source": name,
                "image": os.path.relpath(path, ROOT),
                "ref_gt": ref_gt(row, gt_kind),
            })
            got += 1
        print(f"[{name}] {got} pages from {ds_id}")

    with open(os.path.join(OUT, "manifest.jsonl"), "w") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    print(f"total {len(manifest)} pages -> {OUT}/manifest.jsonl")


if __name__ == "__main__":
    main()