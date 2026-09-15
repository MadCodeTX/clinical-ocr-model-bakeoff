#!/usr/bin/env python3
"""Generate the four overnight study reports from results/ summaries.

Deterministic and safe with partial results -- each report is written only from
the runs that exist, and says so. Run any time; the overnight driver calls it
after every stage.

  reports/NOISE_FLOOR.md   -- +- for every metric from n=5 repeats (A1)
  reports/CORPUS_SWEEP.md  -- median CER vs teacher-corpus size (B2)
  reports/ONESHOT.md       -- zero vs homogeneous vs heterogeneous (A2)
  reports/OMNIDOC.md       -- do the rankings transfer to real pages (A3)
"""
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")


def load(name):
    p = os.path.join(ROOT, "results", name, "summary.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return None


def fmt(x, nd=4):
    return "--" if x is None else f"{x:.{nd}f}"


def write_noise_floor():
    metrics = ["median_cer", "mean_cer", "mean_cer_excl_runaway", "n_runaway"]
    families = [("dots.mocr (base, GPU0)", "noise-dotsmocr-{}"),
                ("Qwen2.5-VL-3B + LoRA (student, GPU1)", "noise-student-{}")]
    L = ["# Noise floor (A1)", "",
         "Independent repeats of the *same* model at the *same* config, so every "
         "spread here is measurement noise, not a modelling difference.", ""]
    measured = {}
    for label, pat in families:
        runs = [load(pat.format(i)) for i in range(1, 6)]
        runs = [r for r in runs if r]
        L += [f"## {label}  (n={len(runs)})", ""]
        if not runs:
            L += ["_no runs found_", ""]
            continue
        L += ["| metric | mean | median | stdev | min | max | range |",
              "|---|---|---|---|---|---|---|"]
        for m in metrics:
            vals = [r.get(m) for r in runs if r.get(m) is not None]
            if not vals:
                continue
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            L.append(f"| {m} | {fmt(st.mean(vals))} | {fmt(st.median(vals))} | "
                     f"{fmt(sd)} | {fmt(min(vals))} | {fmt(max(vals))} | "
                     f"{fmt(max(vals) - min(vals))} |")
            measured.setdefault(m, []).append(sd)
        L.append("")
    L += ["## The noise floor to quote", "",
          "Any later difference must clear these to count as a finding.", ""]
    if measured:
        L += ["| metric | worst-case stdev across families |",
              "|---|---|"]
        for m, sds in measured.items():
            L.append(f"| {m} | +/-{fmt(max(sds))} |")
        L.append("")
    L += ["Reminder from METHODOLOGY: **rank on `median_cer`**; the mean is "
          "carried by a non-deterministic runaway tail and is far noisier.", ""]
    with open(os.path.join(REPORTS, "NOISE_FLOOR.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return measured


def write_corpus_sweep():
    sizes = [800, 2400, 4800]
    L = ["# Corpus-size sweep (B2)", "",
         "Does more distilled data help the small student, or is it a capacity "
         "limit? Same teacher (Qwen3.8-27B), same recipe, corpus size varying.", ""]
    for tag, label in (("3b", "Qwen2.5-VL-3B"), ("7b", "Qwen2.5-VL-7B")):
        rows = [(n, load(f"sweep-{tag}-{n}")) for n in sizes]
        rows = [(n, r) for n, r in rows if r]
        L += [f"## {label}", ""]
        if not rows:
            L += ["_no runs found_", ""]
            continue
        L += ["| corpus docs | median CER | mean CER | excl-runaway | runaways | wall s |",
              "|---|---|---|---|---|---|"]
        for n, r in rows:
            L.append(f"| {n} | {fmt(r.get('median_cer'))} | {fmt(r.get('mean_cer'))} "
                     f"| {fmt(r.get('mean_cer_excl_runaway'))} | {r.get('n_runaway')} "
                     f"| {r.get('wall_time_s')} |")
        L.append("")
        L += ["Per-subset median CER is in each `results/sweep-%s-*/summary.json` "
              "(`per_subset`)." % tag, ""]
    L += ["**Reading it:** a flat curve at the largest size is a capacity limit; "
          "a curve still bending is a data limit and the earlier 3B null result "
          "was premature. Check every delta against reports/NOISE_FLOOR.md.", ""]
    with open(os.path.join(REPORTS, "CORPUS_SWEEP.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def write_oneshot():
    models = [
        ("dots.mocr", "dots-mocr", "oneshot-dotsmocr"),
        ("Qwen2.5-VL-7B", "qwen25vl7b", "oneshot-qwen25vl7b"),
        ("Qwen3.8-27B", "qwen38-27b", "oneshot-qwen38"),
    ]
    L = ["# One-shot exemplar regimes (A2)", "",
         "Does showing the model one clean exemplar of the same form fix the "
         "degraded read? `homo` = same template as the query; `hetero` = a "
         "different template in the same subset. Zero-shot numbers are the "
         "existing full eval runs.", ""]
    L += ["| model | regime | median CER | mean CER | excl-runaway | runaways |",
          "|---|---|---|---|---|---|"]
    for label, zero_name, one_prefix in models:
        cells = [("zero", load(zero_name)),
                 ("homo", load(one_prefix + "-homo")),
                 ("hetero", load(one_prefix + "-hetero"))]
        for regime, r in cells:
            if not r:
                L.append(f"| {label} | {regime} | -- | -- | -- | -- |")
                continue
            L.append(f"| {label} | {regime} | {fmt(r.get('median_cer'))} "
                     f"| {fmt(r.get('mean_cer'))} | {fmt(r.get('mean_cer_excl_runaway'))} "
                     f"| {r.get('n_runaway')} |")
    L += ["", "**Reading it:** if homo/hetero close the gap on degraded subsets "
          "(`poor`, `rotated`, `mixed`), one-shot is a cheaper production lever "
          "than fine-tuning. Whether homo beats hetero tells you if the gain is "
          "template-specific or just 'here is what a good answer looks like'.", ""]
    with open(os.path.join(REPORTS, "ONESHOT.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def write_omnidoc():
    names = sorted(os.path.basename(os.path.dirname(p)) for p in
                   glob.glob(os.path.join(ROOT, "results", "omnidoc-*", "summary.json")))
    L = ["# OmniDocBench generalisation (A3)", "",
         "The ClinOCR-Bench ranking re-measured on 1651 real PDF pages "
         "(OmniDocBench, Apache-2.0, research use only). Synthetic content with "
         "real degradation -> real documents with real layouts.", "",
         "**Caveat:** OmniDocBench is saturated and its rigid metrics penalise "
         "semantically-correct formatting, so compare the *ranking* across "
         "models, not the absolute CER against reports/REPORT.md.", ""]
    if not names:
        L += ["_no runs found_", ""]
    else:
        L += ["| model | median CER | mean CER | excl-runaway | runaways | n docs |",
              "|---|---|---|---|---|---|"]
        rows = [(n, load(n)) for n in names]
        rows.sort(key=lambda kv: kv[1].get("median_cer") or 9)
        for n, r in rows:
            L.append(f"| {n.replace('omnidoc-', '')} | {fmt(r.get('median_cer'))} "
                     f"| {fmt(r.get('mean_cer'))} | {fmt(r.get('mean_cer_excl_runaway'))} "
                     f"| {r.get('n_runaway')} | {r.get('n_docs')} |")
        L.append("")
    with open(os.path.join(REPORTS, "OMNIDOC.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    os.makedirs(REPORTS, exist_ok=True)
    write_noise_floor()
    write_corpus_sweep()
    write_oneshot()
    write_omnidoc()
    print("wrote NOISE_FLOOR.md CORPUS_SWEEP.md ONESHOT.md OMNIDOC.md")


if __name__ == "__main__":
    main()
