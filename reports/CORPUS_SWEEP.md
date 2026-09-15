# Corpus-size sweep (B2)

Does more distilled data help the small student, or is it a capacity limit? Same teacher (Qwen3.8-27B), same recipe, corpus size varying.

## Qwen2.5-VL-3B

| corpus docs | median CER | mean CER | excl-runaway | runaways | wall s |
|---|---|---|---|---|---|
| 800 | 0.2265 | 0.4701 | 0.2814 | 36 | 445.0 |
| 2400 | 0.3416 | 0.5298 | 0.3926 | 28 | 418.7 |
| 4800 | 0.3119 | 0.5139 | 0.3643 | 30 | 422.1 |

Per-subset median CER is in each `results/sweep-3b-*/summary.json` (`per_subset`).

## Qwen2.5-VL-7B

| corpus docs | median CER | mean CER | excl-runaway | runaways | wall s |
|---|---|---|---|---|---|
| 800 | 0.1903 | 0.3777 | 0.2724 | 20 | 612.6 |

Per-subset median CER is in each `results/sweep-7b-*/summary.json` (`per_subset`).

**Reading it:** a flat curve at the largest size is a capacity limit; a curve still bending is a data limit and the earlier 3B null result was premature. Check every delta against reports/NOISE_FLOOR.md.

