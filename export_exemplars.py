#!/usr/bin/env python3
"""Export the ClinOCR-Bench *train* split as one-shot exemplars and attach the
exemplar references (homo_id / hetero_id) to the existing eval set.

ClinOCR-Bench ships 8 training exemplars per subset (one per template) and every
test document references two of them:
  * homo_id   -- exemplar from the same template (clean version of this form)
  * hetero_id -- exemplar from a different template in the same subset

The existing `eval.jsonl` is rewritten in place but only gains fields: the
doc_id / subset / template / image / ground_truth values are copied verbatim so
every previously-scored run stays comparable.

Usage: python3 export_exemplars.py
"""
import json
import os

from datasets import load_dataset

SUBSETS = ["normal", "handwriting", "poor", "rotated", "tables", "mixed"]
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "clinocr")


def main():
    eval_path = os.path.join(OUT, "eval.jsonl")
    with open(eval_path) as f:
        eval_rows = [json.loads(l) for l in f]

    # doc_id -> {homo_id, hetero_id, sample, template}
    ref = {}
    exemplars = []
    for subset in SUBSETS:
        ds = load_dataset("Daniele0025/ClinOCR-Bench", subset)
        img_dir = os.path.join(OUT, "images", subset)
        os.makedirs(img_dir, exist_ok=True)
        for split in ("train", "test"):
            if split not in ds:
                continue
            for item in ds[split]:
                if split == "test":
                    ref[item["doc_id"]] = {
                        "homo_id": item.get("homo_id") or "",
                        "hetero_id": item.get("hetero_id") or "",
                        "sample": item.get("sample"),
                    }
                    continue
                doc_id = item["doc_id"]
                img_path = os.path.join(img_dir, f"{doc_id}.jpg")
                if not os.path.exists(img_path):
                    img = item["image"]
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(img_path, "JPEG", quality=92)
                exemplars.append({
                    "doc_id": doc_id,
                    "subset": subset,
                    "template": item["template"],
                    "image": os.path.relpath(img_path, ROOT),
                    "ground_truth": item["ground_truth"],
                })
        print(f"[{subset}] test refs + train exemplars exported")

    with open(os.path.join(OUT, "exemplars.jsonl"), "w") as f:
        for r in exemplars:
            f.write(json.dumps(r) + "\n")

    n_linked = 0
    for row in eval_rows:
        r = ref.get(row["doc_id"])
        if r:
            row.update(r)
            if r["homo_id"] and r["hetero_id"]:
                n_linked += 1
    with open(eval_path, "w") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")

    print(f"eval.jsonl: {len(eval_rows)} rows, "
          f"{n_linked} with both exemplars linked")
    print(f"exemplars.jsonl: {len(exemplars)} exemplars")


if __name__ == "__main__":
    main()
