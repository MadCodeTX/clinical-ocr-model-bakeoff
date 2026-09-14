#!/usr/bin/env python3
"""Escalation router: decide per page whether to trust the cheap OCR model or
escalate to an expensive one (Azure Layout proxy, $0.01/page).

Only inference-time features are used: image statistics + cheap-model output
statistics. Labels for training come from measured per-doc CER (supervision you
can produce offline with a modest amount of ground truth, then apply at scale).

Writes results/router/summary.json (consumed by make_report.py).
"""
import json
import os

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
AZURE = 0.01


def image_features(path):
    im = Image.open(path).convert("L")
    w, h = im.size
    small = np.asarray(im.resize((512, max(64, int(512 * h / w)))), dtype=np.float32)
    lap = (-4 * small + np.roll(small, 1, 0) + np.roll(small, -1, 0)
           + np.roll(small, 1, 1) + np.roll(small, -1, 1))
    gw, gh = np.diff(small, axis=1), np.diff(small, axis=0)
    dark = (small < 128).astype(np.float32)
    d = np.abs(np.diff(small, axis=1))
    cb = d[:, 7::8].mean() if d.shape[1] > 8 else 0.0
    cw = d[:, 3::8].mean() if d.shape[1] > 8 else 0.0
    darkb = (small < 100).astype(np.float32)
    nb = (np.roll(darkb, 1, 0) + np.roll(darkb, -1, 0) + np.roll(darkb, 1, 1)
          + np.roll(darkb, -1, 1))
    rowv = float(dark.sum(1).var())
    best = 0.0
    for ang in (-8, -4, 4, 8):
        r = np.asarray(Image.fromarray(small.astype(np.uint8)).rotate(
            ang, resample=Image.BILINEAR, fillcolor=255), dtype=np.float32)
        best = max(best, float((r < 128).astype(np.float32).sum(1).var()))
    return {
        "width": w, "height": h, "aspect": w / h, "mpx": w * h / 1e6,
        "blur": float(lap.var()), "edge": float(np.mean(np.abs(gw)) + np.mean(np.abs(gh))),
        "ink": float((small < 128).mean()), "rowvar": rowv,
        "colvar": float(dark.sum(0).var()),
        "blockiness": float(cb / (cw + 1e-6)),
        "speckle": float(((darkb > 0) & (nb == 0)).mean()),
        "skewness": float(best / (rowv + 1e-6)),
    }


def pred_features(pred):
    p = (pred or "").strip()
    n = len(p)
    alnum = sum(c.isalnum() for c in p)
    toks = p.split()
    rep = 0.0
    if len(toks) > 20:
        grams = [" ".join(toks[i:i + 4]) for i in range(len(toks) - 4)]
        if grams:
            _, counts = np.unique(grams, return_counts=True)
            rep = float(counts.max() / len(grams))
    return {"pred_len": n, "non_alnum": 1 - (alnum / n) if n else 0.0,
            "avg_tok_len": (alnum / len(toks)) if toks else 0.0, "rep4": rep,
            "has_table": float("|" in p),
            "newlines": float(p.count("\n") / max(n, 1) * 100), "empty": float(n == 0)}


def load_preds(name):
    return {json.loads(l)["doc_id"]: json.loads(l)
            for l in open(os.path.join(ROOT, "results", name, "predictions.jsonl"))}


def main(cheap="paddleocr-vl", strong="dots-mocr", threshold=0.20,
         out=os.path.join(ROOT, "results", "router")):
    gt = {json.loads(l)["doc_id"]: json.loads(l)
          for l in open(os.path.join(ROOT, "data", "clinocr", "eval.jsonl"))}
    per_doc = {d["doc_id"]: d for d in
               json.load(open(os.path.join(ROOT, "results", cheap, "per_doc.json")))}
    strong_doc = {d["doc_id"]: d for d in
                  json.load(open(os.path.join(ROOT, "results", strong, "per_doc.json")))}
    cheap_preds = load_preds(cheap)

    ids = sorted(per_doc)
    X, names = [], None
    for i in ids:
        f = image_features(os.path.join(ROOT, gt[i]["image"]))
        f.update(pred_features(cheap_preds[i].get("prediction")))
        names = list(f)
        X.append([f[k] for k in names])
    X = np.asarray(X, dtype=np.float32)
    cer_cheap = np.array([per_doc[i]["cer"] for i in ids])
    cer_strong = np.array([strong_doc[i]["cer"] for i in ids])

    result = {
        "cheap": cheap, "strong": strong, "n": len(ids), "threshold": threshold,
        "always_cheap_cer": float(cer_cheap.mean()),
        "always_strong_cer": float(cer_strong.mean()),
        "always_strong_cost_per_page": AZURE,
        "classifiers": {}, "routing": {}, "top_features": [],
        "failure_by_subset": {},
        "cer_by_subset": {s: {} for s in sorted(set(gt[i]["subset"] for i in ids))},
    }
    for s in result["cer_by_subset"]:
        m = [j for j, i in enumerate(ids) if gt[i]["subset"] == s]
        result["cer_by_subset"][s] = {
            "cheap": float(cer_cheap[m].mean()), "strong": float(cer_strong[m].mean()),
            "n": len(m)}

    y = (cer_cheap > threshold).astype(int)
    proba_gb = None
    for mname, clf in [
        ("logreg", make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, class_weight="balanced"))),
        ("gboost", GradientBoostingClassifier(random_state=0)),
    ]:
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        result["classifiers"][mname] = {
            "auc": float(roc_auc_score(y, proba)),
            "acc": float(accuracy_score(y, proba > 0.5)),
            "precision": float(precision_score(y, proba > 0.5, zero_division=0)),
            "recall": float(recall_score(y, proba > 0.5, zero_division=0)),
        }
        if mname == "gboost":
            proba_gb = proba

    sims = {}
    for budget in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        k = int(len(ids) * budget)
        order = np.argsort(-proba_gb)
        esc = np.zeros(len(ids), dtype=bool)
        esc[order[:k]] = True
        blended = np.where(esc, cer_strong, cer_cheap)
        sims[f"{budget:g}"] = {
            "blended_cer": float(blended.mean()),
            "cost_per_page": float(esc.mean() * AZURE),
            "caught": int((esc & (cer_cheap > threshold)).sum()),
            "failures": int((cer_cheap > threshold).sum()),
        }
    esc = cer_cheap > threshold
    sims["oracle"] = {
        "blended_cer": float(np.where(esc, cer_strong, cer_cheap).mean()),
        "cost_per_page": float(esc.mean() * AZURE),
        "caught": int(esc.sum()), "failures": int(esc.sum())}
    result["routing"][str(threshold)] = sims

    gb = GradientBoostingClassifier(random_state=0).fit(X, y)
    result["top_features"] = sorted(zip(names, [float(v) for v in gb.feature_importances_]),
                                    key=lambda kv: -kv[1])

    subs = {}
    for i in ids:
        subs.setdefault(gt[i]["subset"], []).append(int(cer_cheap[ids.index(i)] > threshold))
    result["failure_by_subset"] = {k: float(np.mean(v)) for k, v in sorted(subs.items())}

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(result, f, indent=1)

    # also expose escalation decisions for inspection
    order = np.argsort(-proba_gb)
    top = [{"doc_id": ids[j], "subset": gt[ids[j]]["subset"],
            "prob_fail": round(float(proba_gb[j]), 3), "cer_cheap": float(cer_cheap[j]),
            "cer_strong": float(cer_strong[j])} for j in order[:40]]
    with open(os.path.join(out, "top_escalations.json"), "w") as f:
        json.dump(top, f, indent=1)

    print(json.dumps({k: v for k, v in result.items()
                      if k in ("cheap", "strong", "always_cheap_cer", "always_strong_cer",
                               "classifiers", "failure_by_subset")}, indent=1))
    print("throughput-matched budgets:", json.dumps(result["routing"][str(threshold)], indent=1)
          if threshold == 0.20 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
