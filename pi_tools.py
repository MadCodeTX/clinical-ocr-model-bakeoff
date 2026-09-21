#!/usr/bin/env python3
"""CLI facade over the OCR bake-off harness, designed to be called by the
pi extension (ocr-bench.ts) but perfectly usable by humans.

Subcommands:
  status                          supervisor/queue/GPU/git state
  leaderboard [--query SUB] [-n N]
  report   [--chars N]            print reports/REPORT.md (truncated)
  run      --name N --hf-id ID --prompt P [--gpu 0] [--concurrency 8]
                                  launch a model eval in the background
  publish                          regenerate report + git commit/push
"""
import argparse
import glob
import json
import os
import shlex
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "bin", "python3")
AZURE = 0.01


def sh(cmd, timeout=120):
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                          timeout=timeout)


def cmd_status(_):
    out = []
    done = os.path.exists(os.path.join(ROOT, "logs", "overnight.done"))
    out.append(f"overnight queue: {'COMPLETE' if done else 'running/stopped'}")
    log = os.path.join(ROOT, "logs", "supervisor.log")
    if os.path.exists(log):
        out.append("--- supervisor.log (tail) ---")
        out += open(log).read().strip().splitlines()[-6:]
    rcs = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", ".rc")),
                    key=os.path.getmtime, reverse=True)[:12]:
        name = os.path.basename(os.path.dirname(p))
        rcs.append(f"  {name}: rc={open(p).read().strip()}")
    out.append("--- recent experiment results ---")
    out += rcs or ["  (none)"]
    g = sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader")
    out.append("--- GPUs ---\n" + (g.stdout.strip() or "n/a"))
    c = sh("docker ps --format '{{.Names}}: {{.Status}}' | grep -E 'ocr|vllm' | head -6")
    out.append("--- containers ---\n" + (c.stdout.strip() or "  (none)"))
    gi = sh(f"git -C {shlex.quote(ROOT)} log --oneline -3")
    out.append("--- git ---\n" + gi.stdout.strip())
    print("\n".join(out))


def cmd_leaderboard(args):
    meta = {}
    try:
        meta = json.load(open(os.path.join(ROOT, "models.json")))
    except Exception:
        pass
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", "summary.json"))):
        try:
            s = json.load(open(p))
        except Exception:
            continue
        name = os.path.basename(os.path.dirname(p))
        if args.query and args.query.lower() not in name.lower():
            continue
        m = meta.get(name, {})
        ps = s.get("per_subset") or {}
        rows.append((s.get("mean_cer", 9), name, m.get("params", "?"),
                     s.get("median_cer"), s.get("pages_per_sec"),
                     s.get("n_docs"), ps.get("handwriting", {}).get("mean_cer"),
                     ps.get("rotated", {}).get("mean_cer"),
                     ps.get("tables", {}).get("mean_cer")))
    rows.sort(key=lambda r: (r[0] if r[0] is not None else 9))
    print(f"{'model':<32}{'params':<12}{'CER':>8}{'med':>8}{'pg/s':>7}{'n':>5}"
          f"{'hw':>7}{'rot':>7}{'tbl':>7}")
    print("-" * 96)
    for r in rows[:args.limit]:
        f = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "  -"  # noqa: E731
        print(f"{r[1]:<32}{r[2]:<12}{f(r[0]):>8}{f(r[3]):>8}"
              f"{(f'{r[4]:.2f}' if r[4] else '-'):>7}{r[5] or 0:>5}"
              f"{f(r[6]):>7}{f(r[7]):>7}{f(r[8]):>7}")
    print(f"\n{len(rows)} scored runs (query={args.query!r}, showing {min(len(rows), args.limit)})")
    print("full write-up: reports/REPORT.md | per-doc: results/<name>/per_doc.json")


def cmd_report(args):
    p = os.path.join(ROOT, "reports", "REPORT.md")
    if not os.path.exists(p):
        print("REPORT.md not found - run the publish command first")
        return
    text = open(p).read()
    if args.chars and len(text) > args.chars:
        print(text[:args.chars])
        print(f"\n[... truncated, {len(text) - args.chars} more chars; "
              f"full file: reports/REPORT.md]")
    else:
        print(text)


def cmd_run(args):
    log = os.path.join(ROOT, "logs", f"run-{args.name}.log")
    cmd = (f"cd {shlex.quote(ROOT)} && mkdir -p logs results && "
           f"CONCURRENCY={args.concurrency} nohup setsid bash run_model.sh "
           f"{shlex.quote(args.name)} {shlex.quote(args.hf_id)} {args.gpu} "
           f"{shlex.quote(args.prompt)} > {shlex.quote(log)} 2>&1 < /dev/null & echo started")
    r = sh(cmd)
    print(r.stdout.strip() or r.stderr.strip())
    print(f"name: {args.name}\nlog: {log}\npoll: pi_tools.py status | tail the log")


def cmd_publish(_):
    r = sh(f"cd {shlex.quote(ROOT)} && {shlex.quote(PY)} make_report.py "
           f"> logs/report.log 2>&1; echo rc=$?")
    print("make_report", r.stdout.strip())
    g = sh(f"cd {shlex.quote(ROOT)} && git add -A && "
           f"git diff --cached --quiet || git commit -qm 'results update via pi'; "
           f"git push origin main 2>&1 | tail -2; git log --oneline -1")
    print(g.stdout.strip() or g.stderr.strip())


def main():
    ap = argparse.ArgumentParser(prog="pi_tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    lb = sub.add_parser("leaderboard")
    lb.add_argument("--query", default="")
    lb.add_argument("-n", "--limit", type=int, default=25)
    rp = sub.add_parser("report")
    rp.add_argument("--chars", type=int, default=0)
    rn = sub.add_parser("run")
    rn.add_argument("--name", required=True)
    rn.add_argument("--hf-id", required=True)
    rn.add_argument("--prompt", required=True)
    rn.add_argument("--gpu", type=int, default=0)
    rn.add_argument("--concurrency", type=int, default=8)
    sub.add_parser("publish")
    args = ap.parse_args()
    {"status": cmd_status, "leaderboard": cmd_leaderboard, "report": cmd_report,
     "run": cmd_run, "publish": cmd_publish}[args.cmd](args)


if __name__ == "__main__":
    main()
