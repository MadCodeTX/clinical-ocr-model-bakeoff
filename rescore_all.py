#!/usr/bin/env python3
"""Recompute the runaway-robust metrics for every scored run, from per_doc.json.

Mean CER on this benchmark is unstable: a page the model cannot read can send
it into a repetition loop, pinning that document at the CER cap of 2.0, and
which pages do that varies between identical runs. Two runs of the same adapter
at the same configuration gave mean CER 0.2844 and 0.3072 -- while their medians
were 0.0944 and 0.0954.

This backfills n_runaway and mean_cer_excl_runaway into existing summaries so
older runs can be compared on the stable statistics without re-running them.
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    updated = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", "per_doc.json"))):
        name = os.path.basename(os.path.dirname(p))
        sp = os.path.join(os.path.dirname(p), "summary.json")
        if not os.path.exists(sp):
            continue
        try:
            docs = json.load(open(p))
            cers = [d["cer"] for d in docs if "cer" in d]
        except (json.JSONDecodeError, TypeError):
            continue
        if not cers:
            continue
        summary = json.load(open(sp))
        kept = [c for c in cers if c < 2.0]
        summary["n_runaway"] = sum(1 for c in cers if c >= 2.0)
        summary["mean_cer_excl_runaway"] = round(sum(kept) / len(kept), 4) if kept else None
        json.dump(summary, open(sp, "w"), indent=1)
        updated.append((name, summary))

    print(f"{'run':<34}{'mean':>9}{'median':>9}{'excl-run':>10}{'runaway':>9}")
    print("-" * 71)
    for name, s in sorted(updated, key=lambda kv: kv[1].get("median_cer") or 9):
        m = s.get("mean_cer")
        med = s.get("median_cer")
        ex = s.get("mean_cer_excl_runaway")
        n = s.get("n_runaway")
        if m is None:
            continue
        print(f"{name:<34}{m:>9.4f}{med:>9.4f}"
              f"{(f'{ex:.4f}' if ex is not None else '--'):>10}{n:>9}")


if __name__ == "__main__":
    main()
