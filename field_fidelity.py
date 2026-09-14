#!/usr/bin/env python3
"""Field-level fidelity: do clinically load-bearing tokens survive OCR?

CER treats all characters equally, but for Care Everywhere extraction the things
that break downstream logic are identifiers and values: dates, MRNs, accession
codes, lab decimals, phone numbers. This computes set-based recall/precision of
those token categories per model.

Writes reports/field_fidelity.json and prints a table.
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))

PATTERNS = {
    "dates": re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    "ids_5to8": re.compile(r"\b\d{5,8}\b"),
    "decimals": re.compile(r"\b\d+\.\d+\b"),
    "codes": re.compile(r"\b[A-Z]{1,4}\d{2,}[-A-Z0-9]*\b"),
    "phones": re.compile(r"\(?\d{3}\)?[ \-]?\d{3}[- ]\d{4}"),
    "mrn_prefix": re.compile(r"\b[A-Z]{2}\d{2}[- ]?\d{2}[- ]?\d{2}[A-Z]?\b"),
}


def toks(text):
    """Lowercased, whitespace-collapsed so spacing differences don't count."""
    t = re.sub(r"\s+", " ", text or "")
    return t


def extract(text):
    t = toks(text)
    return {k: PATTERNS[k].findall(t.upper()) for k in PATTERNS}


def prf(ref_counts, hyp_counts):
    tp = sum(min(hyp_counts.get(k, 0), v) for k, v in ref_counts.items())
    ref_total = sum(ref_counts.values())
    hyp_total = sum(hyp_counts.values())
    rec = tp / ref_total if ref_total else float("nan")
    prec = tp / hyp_total if hyp_total else float("nan")
    return tp, ref_total, hyp_total, rec, prec


def main():
    gt = {json.loads(l)["doc_id"]: json.loads(l)
          for l in open(os.path.join(ROOT, "data", "clinocr", "eval.jsonl"))}

    out = {}
    for preds_path in sorted(glob.glob(os.path.join(ROOT, "results", "*", "predictions.jsonl"))):
        model = os.path.basename(os.path.dirname(preds_path))
        rows = [json.loads(l) for l in open(preds_path)]
        agg = {k: {"tp": 0, "ref": 0, "hyp": 0} for k in PATTERNS}
        n_docs = 0
        for r in rows:
            g = gt.get(r["doc_id"])
            if not g:
                continue
            n_docs += 1
            ref = extract(g["ground_truth"])
            hyp = extract(r.get("prediction") or "")
            for k in PATTERNS:
                rc, hc = {}, {}
                for tok in ref[k]:
                    rc[tok] = rc.get(tok, 0) + 1
                for tok in hyp[k]:
                    hc[tok] = hc.get(tok, 0) + 1
                tp, rt, ht, _, _ = prf(rc, hc)
                agg[k]["tp"] += tp
                agg[k]["ref"] += rt
                agg[k]["hyp"] += ht
        stats = {}
        for k, v in agg.items():
            rec = v["tp"] / v["ref"] if v["ref"] else float("nan")
            prec = v["tp"] / v["hyp"] if v["hyp"] else float("nan")
            f1 = (2 * rec * prec / (rec + prec)) if rec and prec and (rec + prec) else 0.0
            stats[k] = {"recall": round(rec, 4), "precision": round(prec, 4) if prec == prec else None,
                        "f1": round(f1, 4), "n_ref": v["ref"]}
        overall_tp = sum(v["tp"] for v in agg.values())
        overall_ref = sum(v["ref"] for v in agg.values())
        out[model] = {
            "n_docs": n_docs,
            "overall_fields": {
                "recall": round(overall_tp / overall_ref, 4) if overall_ref else None,
                "n_ref": overall_ref,
            },
            "by_type": stats,
        }

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "field_fidelity.json"), "w") as f:
        json.dump(out, f, indent=1)

    order = list(PATTERNS)
    print(f"{'model':<26}{'fields':>8}" + "".join(f"{k:>10}" for k in order))
    print("-" * 90)
    for m, s in sorted(out.items(), key=lambda kv: -(kv[1]["overall_fields"]["recall"] or 0)):
        print(f"{m:<26}{s['overall_fields']['recall']:>8.3f}"
              + "".join(f"{s['by_type'][k]['recall']:>10.3f}" for k in order))
    print("\n(recall of ground-truth field tokens; higher is better)")


if __name__ == "__main__":
    main()
