# Corpus-size sweep (B2)

Does more distilled data help the small student, or is it a capacity limit? Same teacher (Qwen3.8-27B), same recipe, corpus size varying.

## Qwen2.5-VL-3B

| corpus docs | median CER | mean CER | excl-runaway | runaways | wall s |
|---|---|---|---|---|---|
| 800 | 0.2265 | 0.4701 | 0.2814 | 36 | 445.0 |

Per-subset median CER is in each `results/sweep-3b-*/summary.json` (`per_subset`).

## Qwen2.5-VL-7B

_no runs found_

**Reading it:** a flat curve at the largest size is a capacity limit; a curve still bending is a data limit and the earlier 3B null result was premature. Check every delta against reports/NOISE_FLOOR.md.

