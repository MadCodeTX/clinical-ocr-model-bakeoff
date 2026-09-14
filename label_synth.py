#!/usr/bin/env python3
"""Teacher labeling: run a teacher VLM over the synthetic corpus and measure how
closely its labels match the exact ground truth (i.e., the label noise you
inherit if you distill from teacher output on otherwise-unlabeled scans).

Usage: python3 label_synth.py --endpoint http://localhost:8000 --n 200 \
    --out results/teacher-labels
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cli import cer, normalize, predict_one  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="ocr")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "synth", "labels.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)][:args.n]
    items = []
    for r in rows:
        img = r["image"]
        if not os.path.exists(img):  # labels use path relative to data/
            img = os.path.join(ROOT, "data", r["image"])
        items.append({"doc_id": r["id"], "subset": "synth", "image": img,
                      "ground_truth": r["text"], "handwriting": r["handwriting"]})

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    preds = []
    with ThreadPoolExecutor(args.concurrency) as ex:
        futs = [ex.submit(predict_one, args.endpoint, args.model, args.prompt,
                          args.max_tokens, it) for it in items]
        for i, f in enumerate(futs):
            preds.append(f.result())
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(items)}", flush=True)
    wall = time.time() - t0

    gt = {it["doc_id"]: it for it in items}
    cers, hw_cers, print_cers = [], [], []
    with open(os.path.join(args.out, "predictions.jsonl"), "w") as f:
        for p in preds:
            it = gt[p["doc_id"]]
            c = cer(normalize(it["ground_truth"]), normalize(p["prediction"]))
            p["gt_cer"] = round(c, 4)
            f.write(json.dumps(p) + "\n")
            cers.append(c)
            (hw_cers if it["handwriting"] else print_cers).append(c)

    summary = {
        "teacher": "olmOCR-2-7B",
        "n": len(preds),
        "label_noise_cer_vs_exact_gt": round(sum(cers) / len(cers), 4),
        "median": round(sorted(cers)[len(cers) // 2], 4),
        "printed_cer": round(sum(print_cers) / max(len(print_cers), 1), 4),
        "handwriting_font_cer": round(sum(hw_cers) / max(len(hw_cers), 1), 4),
        "wall_time_s": round(wall, 1),
        "pages_per_sec": round(len(preds) / wall, 3),
    }
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
