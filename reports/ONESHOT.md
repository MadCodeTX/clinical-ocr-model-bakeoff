# One-shot exemplar regimes (A2)

Does showing the model one clean exemplar of the same form fix the degraded read? `homo` = same template as the query; `hetero` = a different template in the same subset. Zero-shot numbers are the existing full eval runs.

| model | regime | median CER | mean CER | excl-runaway | runaways |
|---|---|---|---|---|---|
| dots.mocr | zero | 0.0747 | 0.2126 | 0.1736 | 7 |
| dots.mocr | homo | 1.0000 | 1.0000 | 1.0000 | 0 |
| dots.mocr | hetero | 1.0000 | 1.0000 | 1.0000 | 0 |
| Qwen2.5-VL-7B | zero | 0.0513 | 0.1994 | 0.1716 | 5 |
| Qwen2.5-VL-7B | homo | 0.0206 | 0.1637 | 0.1524 | 2 |
| Qwen2.5-VL-7B | hetero | 0.0513 | 0.2335 | 0.1722 | 11 |
| Qwen3.8-27B | zero | 0.0224 | 0.1059 | 0.1001 | 1 |
| Qwen3.8-27B | homo | -- | -- | -- | -- |
| Qwen3.8-27B | hetero | -- | -- | -- | -- |

**Reading it:** if homo/hetero close the gap on degraded subsets (`poor`, `rotated`, `mixed`), one-shot is a cheaper production lever than fine-tuning. Whether homo beats hetero tells you if the gain is template-specific or just 'here is what a good answer looks like'.

