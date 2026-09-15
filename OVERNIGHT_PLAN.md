# Overnight OCR run — plan and handover

**Repo:** `~/ocr-bench` (working repo → `github.com/MadCodeTX/clinical-ocr-model-bakeoff`)
**Publishable results get promoted to:** `~/clinical-ocr-bench` → `github.com/MadCodeTX/clinical-ocr-bench`
**Hardware:** bigbox, 2× RTX 4090 24 GB
**Budget:** ~10 hours × 2 GPUs = ~20 GPU-hours
**Written:** 2026-09-14

---

## 0. Read this first — three things that will waste your night

### 0.1 Do NOT run your own agent against the local vLLM model

The resident container `vllm-Qwen3.8-27B-FP8` on port 8000 is Nick's local Claude Code
backend **and** it occupies both GPUs. This plan requires stopping it.

If you are running through `~/.local/bin/lclaude` or anything else pointed at
`localhost:8000` or LiteLLM on `:4000`, **stopping that container kills your own
backend mid-run.** Use a cloud-backed agent (Codex, or Claude Code on Anthropic's
API) for this work.

Stop it at the start, restore it at the end:

```bash
docker stop vllm-Qwen3.8-27B-FP8            # frees both GPUs
# ... overnight work ...
bash experiments/e99_restore_service.sh      # brings it back, waits for readiness
```

### 0.2 Mean CER is not a valid metric here — rank on median

Two runs of the *same model at the same config* scored mean CER 0.2844 and 0.3072.
A non-deterministic tail of repetition loops pins ~7% of documents at the CER cap
of 2.0 and carries ~46% of total error.

- **Rank on `median_cer`.** Noise ±0.0010.
- **Mean CER noise is ±0.023.** Never report a mean-CER difference below ~0.03 as real.
- `n_runaway` and `mean_cer_excl_runaway` are in every summary (`score_preds.py`).
- Backfill older runs with `rescore_all.py`.

### 0.3 Two settings silently corrupt LoRA student results

| setting | why |
|---|---|
| `LORA_TOWER=1` | vLLM applies LoRA to the language model only and **silently ignores** vision-tower adapter weights (192 of 696 tensors; 133 `no matching PunicaWrapper` warnings). Without this you serve a different model than you trained. |
| `MAX_PIXELS=802816` | `train_student.py` trains at `1024*28*28`; Qwen2.5-VL serves at `12845056` by default — 16×. An adapter served outside its training resolution scores *worse than the base model*. |

Both are defaults in `run_lora_model.sh`. **Do not change them without pinning the
training resolution to match.**

Full write-up: `~/clinical-ocr-bench/docs/METHODOLOGY.md`.

### 0.4 `run_model.sh` and `run_lora_model.sh` are single-GPU only

They take one GPU ordinal and derive the port from it (`PORT=$((8000 + GPU))`).
Only `serve_model.sh` handles a GPU list. Anything needing both cards (the 27B)
must be served with `serve_model.sh` and evaluated by pointing `eval_cli.py` at
the endpoint directly. The generalised runner in `~/clinical-ocr-bench/bench/`
handles `GPUS=0,1` natively if you prefer to borrow it.

---

## 1. Where things stand

### Established (margins well outside noise)

| finding | evidence |
|---|---|
| Qwen3.8-27B is a tier above everything else | median CER 0.0224 vs 0.0513 next best; field recall 0.910; 1 runaway doc |
| Every VLM beats Tesseract on degraded scans | median 0.02–0.10 vs 0.444 |
| Qwen3.8 is a 5× better label source than olmOCR-2 | label noise 0.0121 vs 0.0621 CER |
| Distillation helps the 7B student, not the 3B | 7B median 0.0720 → 0.0464; 3B 0.0649 → 0.0669 (null) |
| Exact ground truth is the *worst* teacher | flat-text labels regress `tables` 0.080 → 0.164 |
| Fine-tuning increases runaway generation | 7B 9 → 17 documents, every config |
| Image resolution beats fine-tuning | base 3B: 0.2919 → 0.2461 mean from pixels alone |

### Open questions this run should answer

1. **Is the 3B/7B distillation split a capacity limit or a data limit?** Only 800
   training documents were used. Untested.
2. **Do the conclusions hold on real documents?** ClinOCR-Bench is synthetic
   content with real degradation. Everything so far is on that one corpus.
3. **What is the actual noise floor?** Measured from n=2. Needs n≥5.
4. **Can runaway generation be suppressed?** It is the single largest error
   source. `repetition_penalty=1.05` made it *worse* (0.1994 → 0.2105, runaways
   5 → 8) — clinical documents repeat legitimately. Other levers untested.

---

## 2. Getting more data

Four sources, ordered by value per hour of work.

### 2.1 One-shot exemplars — free, no download, 3× the eval conditions ⭐

**This is the highest-value item and it was missed entirely.** ClinOCR-Bench ships
a `train` split of 56 exemplar documents (one per template) and every eval
document references two of them:

- **homogeneous** exemplar — same template as the query
- **heterogeneous** exemplar — different template, same subset

That enables three regimes on data already on disk: **zero-shot** (all we have
done), **one-shot homogeneous**, **one-shot heterogeneous**.

Real question: does showing a model one clean example of the same form fix the
degraded read? If one-shot closes the gap, that is a cheaper production lever than
fine-tuning.

**Verified 2026-09-14: the current export does NOT carry exemplar fields.**
`data/clinocr/eval.jsonl` has only `doc_id, ground_truth, image, subset, template`,
and `export_data.py` pulls only `['test']`. So step one is a re-export that also
pulls the `train` split and carries the exemplar reference fields through:

```bash
.venv/bin/python3 -c "
import json
r=[json.loads(l) for l in open('data/clinocr/eval.jsonl')][0]
print(sorted(r.keys()))"   # confirm before/after
```
Keep the existing fields and filenames identical so previously-scored runs stay
comparable — add fields, do not rename any.

### 2.2 Scale the synthetic corpus — free, unlimited, directly answers Q1

`gen_synth.py` already produces genuinely multi-artifact documents (mean ~5
degradations/doc, 37% with 6+). Generating 2400–4800 is just CPU time.

```bash
.venv/bin/python3 gen_synth.py --n 4800 --out data/synth_large --seed 11
```
Cost: ~25 min CPU. This is the input to the corpus-size sweep.

### 2.3 OmniDocBench — 1651 real pages ⭐

- **Size:** 1651 PDF pages, 10 document types, 5 layouts, 5 languages
- **Has:** blur / watermark / colourful-background attribute tags → sliceable like our subsets
- **Licence:** Apache-2.0, **research only, not for commercial use** — fine for
  benchmarking, must be flagged if results inform a commercial decision
- **GT:** JSON block-level + markdown conversion tools → CER-compatible
- **Source:** `opendatalab/OmniDocBench` on HuggingFace

Caveat worth knowing: it is considered **saturated** — top models exceed 94%. Its
value here is *generalisation*, not difficulty: do our rankings survive on real
documents? Note also that its rigid metrics penalise semantically-correct
formatting differences — the same trap we hit with markdown normalisation.

### 2.4 Lower priority / needs new scorers

| source | size | why not tonight |
|---|---|---|
| **olmOCR-Bench** (ODC-BY-1.0) | 1403 PDFs, 7010 unit tests | Binary unit-test scoring, not CER. Genuinely better methodology — immune to the formatting traps — but needs a new scorer. Good follow-up project. |
| **RxHandBD** | 5578 handwritten prescription word crops | Word-level, not full-page. Targets our weakest subset but needs a different harness. |
| **OCRBench v2** | 10k QA pairs | QA-based, not full-page transcription. |

---

## 3. The run plan

Two GPUs run independently. **Lane A = GPU 0, Lane B = GPU 1.** Nothing in Lane A
depends on Lane B except where noted.

Start both lanes after `docker stop vllm-Qwen3.8-27B-FP8`.

### Lane A (GPU 0) — measurement validity + generalisation

#### A1. Noise floor, properly measured (~1.5 h)

The whole study's error bars rest on n=2. Fix that.

```bash
for i in 1 2 3 4 5; do
  GPU=0 bash run_model.sh noise-dotsmocr-$i rednote-hilab/dots.mocr 0 \
    "Extract the text content from this image."
done
for i in 1 2 3 4 5; do
  LORA_TOWER=1 MAX_PIXELS=802816 bash run_lora_model.sh noise-student-$i \
    checkpoints/qwen25vl3b-lora 0
done
```

Then report **mean, median, and standard deviation of each metric across the 5
repeats**, for `median_cer`, `mean_cer`, `mean_cer_excl_runaway`, `n_runaway`.

**Deliverable:** `reports/NOISE_FLOOR.md` stating the ± for each metric. Every
later claim in the project must be checked against it.

#### A2. One-shot exemplar regimes (~2.5 h)

Extend `eval_cli.py` with `--shot {zero,homo,hetero}`. For one-shot, prepend the
exemplar image + its ground-truth text as a prior user/assistant turn, then the
query image.

Run three regimes × three models (dots.mocr, Qwen2.5-VL-7B, Qwen3.8-27B). Qwen3.8
needs both GPUs — schedule it after Lane B's minting finishes, or drop it.

**Deliverable:** does one-shot close the degraded-scan gap, and does the
homogeneous/heterogeneous difference matter?

#### A3. OmniDocBench integration (~3 h)

```bash
.venv/bin/python3 export_omnidocbench.py     # write this: HF -> data/omnidoc/eval.jsonl
```
Match the existing schema exactly — `doc_id`, `subset` (use their attribute tags),
`image`, `ground_truth` — and everything downstream works unchanged.

Then run the top 5 by median CER: Qwen3.8-27B, Qwen2.5-VL-7B, Qwen2.5-VL-3B,
dots.mocr, olmOCR-2. Budget 30–60 min each; 1651 pages is 5× our corpus.

**Deliverable:** does the ClinOCR-Bench ranking hold on real documents?

### Lane B (GPU 1) — does more data fix the 3B?

#### B1. Generate + mint (~2.5 h)

```bash
.venv/bin/python3 gen_synth.py --n 4800 --out data/synth_large --seed 11   # CPU, ~25 min
```

Minting 4800 labels with Qwen3.8 needs the 27B served, which wants **both GPUs**.
Coordinate: pause Lane A, serve Qwen3.8 across GPUs 0+1, mint, then release.

`serve_model.sh` accepts a GPU **list** for tensor parallelism (added
2026-09-14 — it previously took a single ordinal and would have silently served
the 27B on one card):

```bash
bash serve_model.sh qwen38-27b Qwen/Qwen3.8-27B-FP8 0,1   # TP=2, port 8000
.venv/bin/python3 label_synth.py --endpoint http://localhost:8000 \
  --model Qwen/Qwen3.8-27B-FP8 --teacher-name qwen38-27b \
  --prompt "Extract the text content from this image." \
  --data data/synth_large/labels.jsonl --n 4800 --concurrency 8 \
  --out results/teacher-labels-q38-4800
```
At the measured 0.489 pages/s this is **~2.7 h**. If that is too much of the
night, mint 2400 (~1.4 h) — still a 3× increase over the current corpus.

```bash
.venv/bin/python3 build_teacher_set.py \
  --preds results/teacher-labels-q38-4800/predictions.jsonl \
  --labels data/synth_large/labels.jsonl \
  --out data/synth_large/teacher_labels_q38.jsonl
```

#### B2. Corpus-size sweep (~4 h)

The actual answer to Q1. Train both students at three corpus sizes:

```bash
for n in 800 2400 4800; do
  head -$n data/synth_large/teacher_labels_q38.jsonl > /tmp/tl_$n.jsonl
  CUDA_VISIBLE_DEVICES=1 train-venv/bin/python3 train_student.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct --data /tmp/tl_$n.jsonl \
    --out checkpoints/q38-3b-$n --epochs 1
  LORA_TOWER=1 MAX_PIXELS=802816 GPU=1 \
    bash run_lora_model.sh sweep-3b-$n checkpoints/q38-3b-$n 1
done
```
Repeat for `Qwen/Qwen2.5-VL-7B-Instruct` if time allows (7B training ≈ 14 min per
800 docs, so 4800 ≈ 84 min).

**Deliverable:** a curve of median CER vs training-corpus size for both students.
If the 3B curve is still flat at 4800, it is a capacity limit. If it bends, it was
a data limit and the earlier null result was premature.

#### B3. Runaway mitigation (~1 h, only if time remains)

`repetition_penalty` failed. Untested levers, cheapest first:

1. **`no_repeat_ngram_size`** (vLLM supports it) — blocks exact n-gram loops
   without penalising legitimate repeated *tokens*. Most promising: the failure is
   `" D. D. D. D."`, an n-gram loop, not token frequency.
2. **Tighter `max_tokens`** — cap at ~2× the longest ground truth. Truncates the
   loop rather than preventing it; cheap and may recover most of the CER.
3. **Post-hoc loop detection** — flag outputs whose length far exceeds the corpus
   norm and score them as failures rather than as text. This is arguably the
   *correct* reporting fix regardless: a runaway is a failed document, not a bad
   transcription.

Measure against `qwen25vl7b` (5 runaways) and `paddleocr-vl` (50).

---

## 4. Priority if you fall behind

Drop from the bottom:

1. **A1 noise floor** — everything else is uninterpretable without it
2. **B1+B2 corpus sweep** — the main open research question
3. **A2 one-shot** — cheapest new signal, no downloads
4. **A3 OmniDocBench** — most work, and its generalisation value survives to another night
5. **B3 runaway mitigation** — nice to have

---

## 5. Deliverables

Commit to `~/ocr-bench` as you go, one commit per stage:

- `reports/NOISE_FLOOR.md` — ± for every metric, from n=5
- `reports/CORPUS_SWEEP.md` — median CER vs corpus size, both students
- `reports/ONESHOT.md` — zero vs homo vs hetero
- `reports/OMNIDOC.md` — does the ranking transfer to real documents
- `results/<run>/` — summaries + per_doc for every run
- Updated `reports/REPORT.md` via `make_report.py`

**Report medians with the measured ±. Do not report a mean-CER difference under
0.03 as a finding.**

Promote only settled results to `~/clinical-ocr-bench`, and keep that repo's
methodology docs in sync if any of the guardrails in §0 turn out to be wrong.

---

## 6. Agent setup status (checked 2026-09-14)

### Codex — installed, **not ready**

```
binary    /usr/local/bin/codex  (npm @openai/codex@0.77.0)
version   0.77.0  — latest is 0.142.5, ~65 releases behind
auth      NO ~/.codex/auth.json, and OPENAI_API_KEY is unset
last use  2026-07-04 (per ~/.codex/log/codex-tui.log)
config    no config.toml — all defaults
```

**Before handing over:**
```bash
npm install -g @openai/codex@latest    # 0.77.0 is very old
codex login                            # or export OPENAI_API_KEY=...
codex --version && ls ~/.codex/auth.json
```

Codex is cloud-backed, so it is **safe for this plan** — stopping the local vLLM
container will not affect it.

### "Pi" coding agent — **not found**

Nothing named `pi` is installed: no binary on `PATH`, no `~/.pi`, `~/.config/pi`,
no npm global, nothing in `~/.local/bin`. What is there:

| tool | what it is |
|---|---|
| `~/.local/bin/oc` | opencode wrapper, auto-detects a local OpenAI-compatible server |
| `~/.local/bin/lclaude` | Claude Code wired to local llama-server (Qwen3.6-27B) |
| `~/.local/bin/tiny-agents` | HF tiny-agents |
| `~/.local/bin/claude` | Claude Code |

If "Pi" is something else, point me at it. **Note that `oc` and `lclaude` are both
backed by local model servers**, which makes them unsuitable for this plan — see
§0.1.

This plan is written to be agent-agnostic: every step is a shell command against
scripts already in the repo, so any agent with shell access can execute it.

---

## 7. Running it unattended

```bash
tmux new -s overnight
docker stop vllm-Qwen3.8-27B-FP8
# lane A
nohup bash -c 'set -x; ...lane A commands...' > logs/laneA.log 2>&1 &
# lane B
nohup bash -c 'set -x; ...lane B commands...' > logs/laneB.log 2>&1 &
```

`overnight.sh` + `watchdog.sh` already implement a supervisor with per-experiment
time budgets and automatic report regeneration — adapt `experiments/queue.txt`
rather than writing a new runner. Note from the last run: budget generously,
`e13` was killed at its 3600 s limit with the eval 235/328 done.

**Morning:** `bash experiments/e99_restore_service.sh` and confirm
`curl -s localhost:8000/v1/models` returns the model before leaving it.
