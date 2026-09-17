#!/usr/bin/env bash
# Baseline: Tesseract OCR on the same eval set, same CER scoring.
set -euo pipefail
cd "$(dirname "$0")"

OUT=${OUT:-results/tesseract}
mkdir -p "$OUT"
.venv/bin/python3 - "$OUT" <<'EOF'
import json, os, subprocess, sys, time

sys.path.insert(0, ".")
from eval_cli import normalize, cer  # reuse scoring

out_dir = sys.argv[1]
root = os.getcwd()
data_path = os.path.join(os.environ.get("DATA", "data/clinocr"), "eval.jsonl")
items = [json.loads(l) for l in open(data_path)]
t0 = time.time()
rows = []
for i, it in enumerate(items):
    img = os.path.join(root, it["image"])
    t = time.time()
    try:
        r = subprocess.run(["tesseract", img, "stdout", "--psm", "3"],
                           capture_output=True, text=True, timeout=120)
        text = r.stdout
    except Exception as e:
        text = ""
    rows.append({"doc_id": it["doc_id"], "subset": it["subset"],
                 "latency_s": round(time.time() - t, 2), "prediction": text,
                 "error": None})
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(items)}")
wall = time.time() - t0
with open(f"{out_dir}/predictions.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

gt = {it["doc_id"]: it for it in items}
persub = {}
for r in rows:
    c = cer(normalize(gt[r["doc_id"]]["ground_truth"]), normalize(r["prediction"]))
    persub.setdefault(r["subset"], []).append(c)
cers = [cer(normalize(gt[r["doc_id"]]["ground_truth"]), normalize(r["prediction"])) for r in rows]
summary = {
    "model": "tesseract",
    "n_docs": len(items),
    "n_errors": 0,
    "mean_cer": round(sum(cers) / len(cers), 4),
    "median_cer": round(sorted(cers)[len(cers)//2], 4),
    "per_subset": {s: {"mean_cer": round(sum(v)/len(v), 4), "n": len(v)} for s, v in sorted(persub.items())},
    "wall_time_s": round(wall, 1),
    "pages_per_sec": round(len(items)/wall, 3),
    "avg_latency_s": round(sum(r["latency_s"] for r in rows)/len(rows), 2),
}
json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=1)
per_doc = [{"doc_id": r["doc_id"], "subset": r["subset"], "error": r["error"],
            "cer": round(c, 4)} for r, c in zip(rows, cers)]
json.dump(per_doc, open(f"{out_dir}/per_doc.json", "w"), indent=1)
print(json.dumps(summary, indent=2))
EOF
