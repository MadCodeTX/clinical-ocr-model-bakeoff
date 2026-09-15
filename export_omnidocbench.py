#!/usr/bin/env python3
"""Export OmniDocBench (1651 real PDF pages) to the ClinOCR-Bench eval schema.

https://huggingface.co/datasets/opendatalab/OmniDocBench  (Apache-2.0,
research use only -- flag this if results feed a commercial decision)

Exports `data/omnidoc/eval.jsonl` with the same fields the harness already
understands -- doc_id, subset, image, ground_truth -- plus language / layout /
data_source for slicing. `subset` is OmniDocBench's own attribute tag
(v1.5 / equation_hard / layout_hard / table_hard), so per-subset breakdowns work
unchanged. Reading order follows each block's `order`; tables emit their HTML
(which eval_cli.normalize() reduces to cell text), equations their LaTeX.

This is a *generalisation* check, not a difficulty check: OmniDocBench is
saturated and its rigid metrics punish semantically-correct formatting. Compare
models against each other on it, not against ClinOCR-Bench numbers.

Usage: python3 export_omnidocbench.py [--limit N] [--skip-images]
"""
import argparse
import json
import os

from huggingface_hub import hf_hub_download, snapshot_download

REPO = "opendatalab/OmniDocBench"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "omnidoc")

# Blocks we transcribe, and which field carries their content.
TEXT_CATS = {
    "text_block", "title", "header", "footer", "reference", "list_group",
    "page_number", "table_caption", "figure_caption", "equation_caption",
    "page_footnote", "table_footnote", "figure_footnote", "equation_explanation",
    "code_txt", "code_txt_caption",
}
SKIP_CATS = {c for c in (
    "text_mask", "chart_mask", "table_mask", "unknown_mask", "need_mask",
    "organic_chemical_formula_mask", "algorithm_mask", "abandon", "figure",
)}


def block_text(b):
    ct = b.get("category_type")
    if ct in SKIP_CATS or b.get("ignore"):
        return ""
    if ct in ("equation_isolated", "equation_semantic"):
        return b.get("latex") or ""
    if ct == "table":
        return b.get("html") or b.get("text") or ""
    if ct in TEXT_CATS:
        return b.get("text") or ""
    return b.get("text") or b.get("latex") or b.get("html") or ""


def page_ground_truth(page):
    blocks = sorted(page.get("layout_dets", []),
                    key=lambda b: (b.get("order") is None, b.get("order") or 0))
    parts = [t for t in (block_text(b) for b in blocks) if t and t.strip()]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    gt_path = hf_hub_download(REPO, "OmniDocBench.json", repo_type="dataset",
                              local_dir=OUT)
    if not args.skip_images:
        print("downloading images (~1.45 GB)...")
        snapshot_download(REPO, repo_type="dataset", local_dir=OUT,
                          allow_patterns=["images/*"])

    pages = json.load(open(gt_path))
    if args.limit:
        pages = pages[:args.limit]

    rows = []
    for page in pages:
        info = page["page_info"]
        attr = info.get("page_attribute", {})
        name = info["image_path"]
        img_path = os.path.join(OUT, "images", name)
        gt = page_ground_truth(page)
        if not gt:
            continue
        rows.append({
            "doc_id": os.path.splitext(name)[0],
            "subset": attr.get("subset", "?"),
            "language": attr.get("language"),
            "layout": attr.get("layout"),
            "data_source": attr.get("data_source"),
            "image": os.path.relpath(img_path, ROOT),
            "ground_truth": gt,
        })

    if not args.skip_images:
        missing = [r for r in rows if not os.path.exists(os.path.join(ROOT, r["image"]))]
        if missing:
            print(f"WARNING: {len(missing)} images missing on disk")

    with open(os.path.join(OUT, "eval.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    print(f"exported {len(rows)} pages -> {OUT}/eval.jsonl")
    print("subsets:", Counter(r["subset"] for r in rows).most_common())
    print("languages:", Counter(r["language"] for r in rows).most_common())


if __name__ == "__main__":
    main()
