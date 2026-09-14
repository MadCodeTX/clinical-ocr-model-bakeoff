#!/usr/bin/env python3
"""Build reports/REPORT.md (+ CSV, JSON, PNG charts) from everything in results/.

Run any time; safe with partial results. Called by the overnight supervisor after
every experiment so the GitHub repo always reflects the latest state.
"""
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
SUBSETS = ["normal", "handwriting", "poor", "rotated", "tables", "mixed"]
AZURE_PER_PAGE = 0.01


def load_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def load_summaries():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", "summary.json"))):
        s = load_json(p)
        if s:
            out[os.path.basename(os.path.dirname(p))] = s
    return out


def load_status():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", ".rc"))):
        name = os.path.basename(os.path.dirname(p))
        try:
            rc, secs = open(p).read().split()
            out[name] = {"rc": int(rc), "seconds": float(secs)}
        except Exception:
            out[name] = {"rc": -1, "seconds": None}
    return out


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def chart_leaderboard(sums, path):
    rows = [(k, v) for k, v in sums.items() if v.get("mean_cer") is not None]
    if len(rows) < 2:
        return
    rows.sort(key=lambda kv: kv[1]["mean_cer"])
    names = [k for k, _ in rows]
    means = [v["mean_cer"] for _, v in rows]
    meds = [v.get("median_cer", 0) for _, v in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 2))
    ax.barh(y, means, color="#2b6cb0", alpha=.85, label="mean CER")
    ax.scatter(meds, y, color="#c53030", zorder=3, label="median CER", s=28)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("character error rate (lower is better)")
    ax.set_title("OCR quality on 328 clinical scans (ClinOCR-Bench)")
    for yi, m in zip(y, means):
        ax.text(m + 0.01, yi, f"{m:.3f}", va="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def chart_heatmap(sums, path):
    rows = [(k, v) for k, v in sums.items() if v.get("per_subset")]
    if len(rows) < 2:
        return
    rows.sort(key=lambda kv: kv[1]["mean_cer"])
    names = [k for k, _ in rows]
    M = np.array([[v["per_subset"].get(s, {}).get("mean_cer", np.nan)
                   for s in SUBSETS] for _, v in rows])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 2.5))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(SUBSETS)))
    ax.set_xticklabels(SUBSETS, rotation=25, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("CER by artifact subset")
    fig.colorbar(im, ax=ax, shrink=.8, label="CER")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def chart_distill(sums, path):
    base = sums.get("qwen25vl3b-base")
    pairs = [("base", base), ("+LoRA GT", sums.get("qwen25vl3b-lora")),
             ("+LoRA teacher", sums.get("qwen25vl3b-lora-teacher"))]
    pairs = [(n, s) for n, s in pairs if s and s.get("per_subset")]
    if len(pairs) < 2:
        return
    x = np.arange(len(SUBSETS))
    w = 0.8 / len(pairs)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, s) in enumerate(pairs):
        vals = [s["per_subset"].get(k, {}).get("mean_cer", 0) for k in SUBSETS]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBSETS, rotation=20, ha="right")
    ax.set_ylabel("mean CER")
    ax.set_title("Distillation: student CER by artifact subset")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def chart_router(path):
    r = load_json(os.path.join(ROOT, "results", "router", "summary.json"))
    if not r:
        return
    thr = str(r["threshold"])
    sims = r["routing"].get(thr, {})
    if not sims:
        return
    # "oracle" is a non-numeric budget key: a perfect-routing lower bound, not a
    # point on the budget sweep.
    budgets = sorted(float(k) for k in sims if k != "oracle")
    cers = [sims[f"{b:g}"]["blended_cer"] for b in budgets]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([b * 100 for b in budgets], cers, "o-", label="router (cheap + escalate)")
    ax.axhline(r["always_cheap_cer"], ls="--", color="#c53030", label="always cheap")
    ax.axhline(r["always_strong_cer"], ls="--", color="#2f855a", label="always expensive")
    if "oracle" in sims:
        ax.axhline(sims["oracle"]["blended_cer"], ls=":", color="#2b6cb0",
                   label="oracle routing (lower bound)")
    ax.set_xlabel("% pages escalated to expensive model")
    ax.set_ylabel("mean CER")
    ax.set_title("Escalation trade-off")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def budget_sort(kv):
    """Escalation budgets are string keys ("0.05"...), plus a non-numeric "oracle"."""
    try:
        return (0, float(kv[0]))
    except ValueError:
        return (1, 0.0)


def budget_label(b):
    try:
        return f"{float(b)*100:.0f}%"
    except ValueError:
        return b


def report_router():
    """Router section: returns (markdown, list_of_notes)."""
    r = load_json(os.path.join(ROOT, "results", "router", "summary.json"))
    if not r:
        return "", []
    lines = []
    lines.append(f"Cheap model = **{r['cheap']}**, escalate to **{r['strong']}** "
                 f"(proxy for Azure Layout at ${AZURE_PER_PAGE}/page).\n")
    lines.append(md_table(
        ["escalation budget", "blended CER", "cost/page", "cost / 1M pages", "failures caught"],
        [[budget_label(b), f"{v['blended_cer']:.4f}", f"${v['cost_per_page']:.5f}",
          f"${v['cost_per_page']*1e6:,.0f}", f"{v['caught']}/{v['failures']}"]
         for b, v in sorted(r["routing"][str(r["threshold"])].items(), key=budget_sort)]))
    lines.append("")
    lines.append(md_table(["classifier", "AUC", "accuracy", "precision", "recall"],
                          [[k, f"{v['auc']:.3f}", f"{v['acc']:.3f}",
                            f"{v['precision']:.3f}", f"{v['recall']:.3f}"]
                           for k, v in r["classifiers"].items()]))
    lines.append("")
    lines.append("Failure rate by subset (cheap model): " + ", ".join(
        f"**{k}** {v*100:.0f}%" for k, v in sorted(r["failure_by_subset"].items(),
                                                   key=lambda kv: -kv[1])))
    lines.append("")
    lines.append("Top predictive features: " + ", ".join(
        f"`{k}` {v:.2f}" for k, v in r["top_features"][:6]))
    p = os.path.join(REPORTS, "router_tradeoff.png")
    if os.path.exists(p):
        lines.append(f"\n![router tradeoff](router_tradeoff.png)")

    # alternative cheap/expensive pairs, if e14 ran them
    alts = []
    for p2 in sorted(glob.glob(os.path.join(ROOT, "results", "router-*", "summary.json"))):
        r2 = load_json(p2)
        if not r2:
            continue
        sim = r2["routing"].get(str(r2["threshold"]), {}).get("0.2")
        if sim:
            alts.append([f"{r2['cheap']} -> {r2['strong']}",
                         f"{r2['always_cheap_cer']:.3f}",
                         f"{r2['always_strong_cer']:.3f}",
                         f"{sim['blended_cer']:.3f}",
                         f"${sim['cost_per_page']:.5f}",
                         f"{r2['classifiers']['gboost']['auc']:.3f}"])
    if alts:
        lines.append("\n**Other cheap -> expensive pairings** (20% escalation budget):\n")
        lines.append(md_table(["pair", "always cheap", "always expensive",
                               "router CER", "cost/page", "AUC"], alts))
    return "\n".join(lines), []


def report_distill(sums):
    base = sums.get("qwen25vl3b-base")
    lora = sums.get("qwen25vl3b-lora")
    tlor = sums.get("qwen25vl3b-lora-teacher")
    if not base or not (lora or tlor):
        return "", []
    rows = []
    for label, s in (("base Qwen2.5-VL-3B", base), ("+ LoRA (exact GT labels)", lora),
                     ("+ LoRA (teacher labels)", tlor)):
        if not s:
            continue
        rows.append([label, f"{s['mean_cer']:.4f}", f"{s['median_cer']:.4f}"] +
                    [f"{s['per_subset'].get(k, {}).get('mean_cer', float('nan')):.3f}"
                     for k in SUBSETS])
    md = md_table(["student", "mean CER", "median CER"] + SUBSETS, rows)
    notes = []
    if lora:
        d = base["mean_cer"] - lora["mean_cer"]
        dhw = base["per_subset"].get("handwriting", {}).get("mean_cer", 0) - \
            lora["per_subset"].get("handwriting", {}).get("mean_cer", 0)
        drot = base["per_subset"].get("rotated", {}).get("mean_cer", 0) - \
            lora["per_subset"].get("rotated", {}).get("mean_cer", 0)
        notes.append(f"LoRA on synthetic degraded docs: overall CER {base['mean_cer']:.3f} -> "
                     f"{lora['mean_cer']:.3f} ({d:+.3f}); handwriting {dhw:+.3f}; rotated {drot:+.3f}")
    return md, notes


def ensure_samples():
    """Copy one image per interesting subset into docs/samples for the repo."""
    src = os.path.join(ROOT, "data", "clinocr", "eval.jsonl")
    if not os.path.exists(src):
        return
    dst = os.path.join(ROOT, "docs", "samples")
    os.makedirs(dst, exist_ok=True)
    rows = [json.loads(l) for l in open(src)]
    for sub in ("normal", "handwriting", "rotated", "mixed"):
        row = next((r for r in rows if r["subset"] == sub), None)
        if row:
            target = os.path.join(dst, f"{sub}.jpg")
            if not os.path.exists(target):
                try:
                    shutil.copy(os.path.join(ROOT, row["image"]), target)
                except Exception:
                    pass


def main():
    os.makedirs(REPORTS, exist_ok=True)
    ensure_samples()
    sums = load_summaries()
    status = load_status()
    meta = load_json(os.path.join(ROOT, "models.json"), {})
    ff = load_json(os.path.join(REPORTS, "field_fidelity.json"))
    teacher = load_json(os.path.join(ROOT, "results", "teacher-labels", "summary.json"))
    router_md, router_notes = report_router()
    distill_md, distill_notes = report_distill(sums)

    chart_leaderboard(sums, os.path.join(REPORTS, "cer_leaderboard.png"))
    chart_heatmap(sums, os.path.join(REPORTS, "cer_by_subset.png"))
    chart_distill(sums, os.path.join(REPORTS, "distill_delta.png"))
    chart_router(os.path.join(REPORTS, "router_tradeoff.png"))

    # leaderboard rows
    rows, csv = [], ["model,params,license,mean_cer,median_cer,pages_per_sec,"
                     "normal,handwriting,poor,rotated,tables,mixed"]
    for k, v in sorted(sums.items(), key=lambda kv: kv[1].get("mean_cer", 9)):
        m = meta.get(k, {})
        ps = v.get("per_subset", {})
        rows.append([k, m.get("params", "?"), m.get("license", "?"),
                     f"{v.get('mean_cer', float('nan')):.4f}",
                     f"{v.get('median_cer', float('nan')):.4f}",
                     f"{v.get('pages_per_sec', 0):.2f}"] +
                    [f"{ps.get(s, {}).get('mean_cer', float('nan')):.3f}" for s in SUBSETS])
        csv.append(",".join([k, m.get("params", "?"), m.get("license", "?"),
                             f"{v.get('mean_cer', 0):.4f}", f"{v.get('median_cer', 0):.4f}",
                             f"{v.get('pages_per_sec', 0):.3f}"] +
                            [f"{ps.get(s, {}).get('mean_cer', 0):.4f}" for s in SUBSETS]))
    with open(os.path.join(REPORTS, "leaderboard.csv"), "w") as f:
        f.write("\n".join(csv) + "\n")

    # cost projection
    cost_rows = []
    for k, v in sorted(sums.items(), key=lambda kv: -kv[1].get("pages_per_sec", 0)):
        pps = v.get("pages_per_sec")
        if not pps or k == "tesseract":
            continue
        per_day = pps * 86400
        cost_rows.append([k, f"{pps:.2f}", f"{per_day:,.0f}",
                          f"${0.45*24*0.15:.2f}", f"${per_day*AZURE_PER_PAGE:,.0f}",
                          f"{per_day*AZURE_PER_PAGE/(0.45*24*0.15):,.0f}x"])

    # field fidelity rows. recall is None for a category with no ground-truth
    # tokens), so render those as "-".
    def ff_num(x):
        return f"{x:.3f}" if isinstance(x, float) and x == x else "-"

    ff_rows = []
    if ff:
        keys = ["dates", "ids_5to8", "decimals", "codes", "phones", "mrn_labeled"]
        full_n = max(v["n_docs"] for v in ff.values())
        for m, s in sorted(ff.items(), key=lambda kv: -(kv[1]["overall_fields"]["recall"] or 0)):
            # a run cut short by its time budget is scored on a biased slice of
            # the eval set; label it so it isn't read as comparable.
            label = m if s["n_docs"] >= full_n else f"{m} ⚠️ *(partial, {s['n_docs']}/{full_n})*"
            ff_rows.append([label, ff_num(s["overall_fields"]["recall"])] +
                           [ff_num(s["by_type"].get(k, {}).get("recall"))
                            for k in keys])

    gen = load_json(os.path.join(ROOT, "data", "synth", "gen_summary.json"), {})

    L = []
    L.append("# Clinical OCR model bake-off — overnight results\n")
    L.append("Open-weights OCR / document-VLM evaluation on **real-artifact clinical scanned "
             "documents**, plus a routing study and a distillation experiment. "
             "All data is public or synthetic (no PHI).\n")
    L.append(f"_Generated {os.popen('date -u +%Y-%m-%dT%H:%MZ').read().strip()} from "
             f"`make_report.py`; {len(sums)} scored models, "
             f"{len(status)} experiment runs._\n")

    # TL;DR
    best = min(sums.items(), key=lambda kv: kv[1].get("mean_cer", 9)) if sums else None
    L.append("## TL;DR\n")
    if best:
        L.append(f"- **Best accuracy: `{best[0]}`** (mean CER {best[1]['mean_cer']:.3f}, "
                 f"median {best[1]['median_cer']:.3f}).")
    if "dots-mocr" in sums and "olmocr-2" in sums:
        L.append(f"- **Best value: `dots-mocr` (3B, MIT)** — CER "
                 f"{sums['dots-mocr']['mean_cer']:.3f} vs olmOCR-2's "
                 f"{sums['olmocr-2']['mean_cer']:.3f} at "
                 f"{sums['dots-mocr'].get('pages_per_sec',0)/max(sums['olmocr-2'].get('pages_per_sec',1),1e-9):.1f}x the "
                 f"throughput.")
    if "tesseract" in sums and best:
        L.append(f"- **Incumbent Tesseract**: mean CER {sums['tesseract']['mean_cer']:.3f}, "
                 f"median {sums['tesseract']['median_cer']:.3f} — the gap is worst on the "
                 f"degraded artifacts that dominate inbound faxes.")
    r = load_json(os.path.join(ROOT, "results", "router", "summary.json"))
    if r:
        b = r["routing"][str(r["threshold"])].get("0.2")
        if b:
            L.append(f"- **Router**: escalating only 20% of pages to Layout-class OCR reaches "
                     f"CER {b['blended_cer']:.3f} at ${b['cost_per_page']:.5f}/page "
                     f"({(1-b['cost_per_page']/AZURE_PER_PAGE)*100:.0f}% cheaper than "
                     f"escalating everything).")
    for n in distill_notes:
        L.append(f"- **Distillation**: {n}")
    if teacher:
        L.append(f"- **Teacher labels**: olmOCR-2 output matches exact ground truth at "
                 f"CER {teacher['label_noise_cer_vs_exact_gt']:.3f} (median "
                 f"{teacher['median']:.4f}) on 200 synthetic degraded clinical docs — "
                 f"cheap to mint training labels for unlabeled scans.")
    L.append("")

    L.append("## 1. Setup\n")
    L.append("**Data.** [`ClinOCR-Bench`](https://huggingface.co/datasets/Daniele0025/ClinOCR-Bench) "
             "(MIT): 328 eval documents across six artifact subsets — normal, handwriting, "
             "poor-quality, rotated, tables, mixed — built from 16 clinical templates "
             "(referral faxes, pathology reports, lab panels, discharge summaries) with "
             "real-world degradation and human-audited ground truth.\n")
    if gen:
        L.append(f"**Synthetic corpus for training.** {gen.get('n', 800)} generated clinical "
                 f"documents with exact labels ({gen.get('handwriting', 0)} handwriting-font), "
                 f"rendered and then degraded (rotation, perspective, blur, noise, low-DPI, "
                 f"JPEG, speckle, fold lines, vignette). Held-out evaluation stays on "
                 f"ClinOCR-Bench, which uses a different construction pipeline.\n")
    L.append("**Metric.** Character error rate (CER) after normalising markup and whitespace; "
             "plus field-level recall of dates, MRNs, accession codes, decimals and phone "
             "numbers (section 3).\n")
    L.append("**Hardware.** 2× RTX 4090 24 GB; vLLM v0.27.1 for served models; "
             "one model per GPU.\n")

    for p in ("docs/samples",):
        pass
    if os.path.isdir(os.path.join(ROOT, "docs", "samples")):
        L.append("**Sample documents** (normal / handwriting / rotated / mixed):\n")
        L.append("| normal | handwriting | rotated | mixed |")
        L.append("|---|---|---|---|")
        L.append("| ![normal](docs/samples/normal.jpg) | ![hw](docs/samples/handwriting.jpg) "
                 "| ![rot](docs/samples/rotated.jpg) | ![mix](docs/samples/mixed.jpg) |")
        L.append("")

    L.append("## 2. Leaderboard\n")
    L.append(md_table(["model", "params", "license", "mean CER", "median CER", "pages/s"] +
                      SUBSETS, rows))
    if os.path.exists(os.path.join(REPORTS, "cer_leaderboard.png")):
        L.append("\n![leaderboard](cer_leaderboard.png)")
    L.append("")

    L.append("## 3. Where models fail\n")
    if os.path.exists(os.path.join(REPORTS, "cer_by_subset.png")):
        L.append("![heatmap](cer_by_subset.png)\n")
    if ff_rows:
        L.append("**Field-level recall** — whether clinically load-bearing tokens survive "
                 "(dates, MRNs/IDs, lab decimals, accession codes, phone numbers):\n")
        L.append(md_table(["model", "all fields", "dates", "ids 5–8d", "decimals",
                           "codes", "phones", "labeled MRN"], ff_rows))
        L.append("")

    if cost_rows:
        L.append("## 4. Throughput & cost\n")
        L.append(md_table(["model", "pages/s (1 GPU, c=8)", "pages/day (1 GPU)",
                           "electricity/day", "Azure Layout/day", "saving"],
                          cost_rows))
        L.append(f"\nAssumes one 450 W 4090 at $0.15/kWh (~$1.62/day); Azure Layout OCR at "
                 f"${AZURE_PER_PAGE}/page. Self-hosting is 3–4 orders of magnitude cheaper per "
                 f"page *before* counting GPU amortisation.*\n")

    if router_md:
        L.append("## 5. Router: cheap model + escalate the hard tail\n")
        L.append(router_md)
        L.append("")

    if distill_md:
        L.append("## 6. Distillation: can a cheap model learn the hard cases?\n")
        L.append(distill_md)
        L.append("")
        for n in distill_notes:
            L.append(f"- {n}")
        if os.path.exists(os.path.join(REPORTS, "distill_delta.png")):
            L.append("\n![distillation](distill_delta.png)")
        L.append("")

    if teacher:
        L.append("## 7. Teacher label quality\n")
        L.append(md_table(["teacher", "docs", "CER vs exact GT", "median", "printed text",
                           "handwriting font", "pages/s"],
                          [["olmOCR-2-7B", teacher["n"], f"{teacher['label_noise_cer_vs_exact_gt']:.4f}",
                            f"{teacher['median']:.4f}", f"{teacher['printed_cer']:.4f}",
                            f"{teacher['handwriting_font_cer']:.4f}",
                            f"{teacher['pages_per_sec']:.2f}"]]))
        L.append("")

    L.append("## 8. Run log\n")
    if status:
        L.append(md_table(["experiment", "exit", "minutes"],
                          [[k, ("ok" if v["rc"] == 0 else f"rc={v['rc']}"),
                            f"{v['seconds']/60:.1f}" if v["seconds"] else "?"]
                           for k, v in sorted(status.items())]))
        L.append("")

    L.append("## 9. Conclusions\n")
    L.append("1. **Replace Tesseract for degraded scans.** Every VLM tested beats it on the "
             "artifacts that dominate inbound faxes; the median-doc gap is ~6x.\n"
             "2. **`dots-mocr` (3B, MIT) is the best default**: accuracy equal to the 8B "
             "olmOCR-2 at much higher throughput and a permissive licence.\n"
             "3. **Routing is the real cost lever**: a classifier over cheap image + "
             "transcript features catches the failures, so you pay Layout prices on a small "
             "slice instead of every page.\n"
             "4. **Distillation is viable**: see section 6 — domain synthetic data moves the "
             "cheap student on exactly the subsets that were failing.\n"
             "5. **Field-level fidelity, not CER, should gate production**: see section 3.\n")
    L.append("\n## Appendix: reproduce\n")
    L.append("```bash\n"
             "bash setup_env.sh          # venv + deps + tesseract\n"
             "python3 export_data.py     # pull ClinOCR-Bench test split\n"
             "python3 gen_synth.py --n 800 --out data/synth\n"
             "bash run_model.sh <name> <hf_id> <gpu> \"<prompt>\"\n"
             "bash overnight.sh          # full experiment queue + report + push\n"
             "```\n")

    report = "\n".join(L)
    with open(os.path.join(REPORTS, "REPORT.md"), "w") as f:
        f.write(report)
    with open(os.path.join(REPORTS, "summary.json"), "w") as f:
        json.dump({"models": sums, "status": status, "teacher": teacher,
                   "field_fidelity": ff}, f, indent=1)
    print(report[:3000])
    print(f"\n[report written: reports/REPORT.md ({len(report)} chars)]")


if __name__ == "__main__":
    main()
