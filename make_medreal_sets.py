#!/usr/bin/env python3
"""Turn the DeepSeek-labelled medreal corpus into eval + training files.

  data/medreal/eval.jsonl   doc_id, subset=<source>, image, ground_truth (DeepSeek), ref_gt?
  data/medreal/labels.jsonl id, image, text (DeepSeek), handwriting=False   [for train_student.py]

Also reports per-source stats and, where the source shipped independent ground
truth (india-hist full text), the CER between it and the DeepSeek label -- a
sanity check on label trustworthiness that does not rely on DeepSeek itself.
"""
import json
import os
import re

from rapidfuzz.distance import Levenshtein

ROOT = os.path.dirname(os.path.abspath(__file__))
MED = os.path.join(ROOT, "data", "medreal")


def norm(t):
    t = (t or "").lower()
    t = re.sub(r"<[^>]{1,40}>", " ", t)
    t = "".join(ch if ch not in set("#*_`|>[]~") else " " for ch in t)
    return re.sub(r"\s+", " ", t).strip()


def cer(a, b):
    if not a:
        return 0.0 if not b else 1.0
    if not b:
        return 1.0
    return min(Levenshtein.distance(a, b) / len(a), 2.0)


def main():
    manifest = [json.loads(l) for l in open(os.path.join(MED, "manifest.jsonl"))]
    preds = {json.loads(l)["doc_id"]: json.loads(l)
             for l in open(os.path.join(ROOT, "results", "medreal-deepseek", "predictions.jsonl"))}

    eval_rows, label_rows = [], []
    per_source = {}
    empty = errored = 0
    ref_cers = []
    for m in manifest:
        p = preds.get(m["doc_id"])
        if not p:
            continue
        text = (p.get("prediction") or "").strip()
        if p.get("error"):
            errored += 1
        if not text:
            empty += 1
            continue
        eval_rows.append({"doc_id": m["doc_id"], "subset": m["source"],
                          "source": m["source"], "image": m["image"],
                          "ground_truth": text, "ref_gt": m.get("ref_gt")})
        label_rows.append({"id": m["doc_id"], "image": m["image"], "text": text,
                           "handwriting": False})
        per_source[m["source"]] = per_source.get(m["source"], 0) + 1
        if m.get("ref_gt"):
            ref_cers.append(cer(norm(m["ref_gt"]), norm(text)))

    with open(os.path.join(MED, "eval.jsonl"), "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(MED, "labels.jsonl"), "w") as f:
        for r in label_rows:
            f.write(json.dumps(r) + "\n")

    print(f"eval.jsonl: {len(eval_rows)} pages, labels.jsonl: {len(label_rows)}")
    print("per source:", per_source)
    print(f"empty={empty} errored={errored}")
    if ref_cers:
        print(f"DeepSeek vs independent GT (n={len(ref_cers)}): "
              f"mean CER {sum(ref_cers)/len(ref_cers):.4f}, "
              f"median {sorted(ref_cers)[len(ref_cers)//2]:.4f}")


if __name__ == "__main__":
    main()