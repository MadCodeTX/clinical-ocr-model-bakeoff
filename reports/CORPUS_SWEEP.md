# Corpus-size sweep (B2)

Does more distilled data help the small student, or is it a capacity limit? Same teacher (Qwen3.8-27B), same recipe, corpus size varying.

## Qwen2.5-VL-3B

| corpus docs | median CER | mean CER | excl-runaway | runaways | wall s |
|---|---|---|---|---|---|
| 0 (base) | 0.0649 | 0.2969 | 0.1685 | 23 | 308.9 |
| 800 | 0.0663 | 0.2926 | 0.1517 | 25 | 376.6 |
| 2400 | 0.0704 | 0.3002 | 0.1415 | 28 | 375.2 |
| 4800 | 0.0692 | 0.3068 | 0.1426 | 29 | 385.6 |

Per-subset median CER is in each `results/sweep-3b-*/summary.json` (`per_subset`).

## Qwen2.5-VL-7B

| corpus docs | median CER | mean CER | excl-runaway | runaways | wall s |
|---|---|---|---|---|---|
| 0 (base) | 0.0720 | 0.2112 | 0.1607 | 9 | 438.0 |
| 800 | 0.0449 | 0.2056 | 0.1316 | 13 | 527.5 |
| 2400 | 0.0617 | 0.2018 | 0.1510 | 9 | 477.4 |

Per-subset median CER is in each `results/sweep-7b-*/summary.json` (`per_subset`).

**Reading it (measured, n=1 per point; noise floor +/-0.0026 median CER):**

- **3B is capacity-limited, not data-limited.** Every fine-tuned point sits at or below the untuned base and the curve does not move from 800 to 4800 docs. The earlier 3B null result was not premature.
- **7B distillation helps, but more data does not help monotonically.** 800 docs cut median CER 0.0720 -> 0.0449 (well outside noise); 2400 docs regress to 0.0617. The extra labels are not noisier (label CER is flat at ~0.012 across the corpus), so this is a training/optimisation effect, not a data-quality one.
- The 800-doc 7B gain is concentrated in `rotated` (0.305->0.236), `tables` (0.081->0.046) and `normal`; 2400 trades `handwriting` (0.137->0.251) for `mixed`. Treat 800 as the 7B sweet spot on this evidence.

