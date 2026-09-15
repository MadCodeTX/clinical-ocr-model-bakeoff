# Noise floor (A1)

Independent repeats of the *same* model at the *same* config, so every spread here is measurement noise, not a modelling difference.

## dots.mocr (base, GPU0)  (n=3)

| metric | mean | median | stdev | min | max | range |
|---|---|---|---|---|---|---|
| median_cer | 0.0749 | 0.0747 | 0.0004 | 0.0747 | 0.0754 | 0.0007 |
| mean_cer | 0.2162 | 0.2167 | 0.0024 | 0.2135 | 0.2183 | 0.0048 |
| mean_cer_excl_runaway | 0.1677 | 0.1680 | 0.0012 | 0.1664 | 0.1688 | 0.0024 |
| n_runaway | 8.6667 | 9.0000 | 0.5774 | 8.0000 | 9.0000 | 1.0000 |

## Qwen2.5-VL-3B + LoRA (student, GPU1)  (n=3)

| metric | mean | median | stdev | min | max | range |
|---|---|---|---|---|---|---|
| median_cer | 0.0973 | 0.0962 | 0.0026 | 0.0954 | 0.1003 | 0.0049 |
| mean_cer | 0.3002 | 0.2955 | 0.0096 | 0.2938 | 0.3112 | 0.0174 |
| mean_cer_excl_runaway | 0.1720 | 0.1711 | 0.0055 | 0.1670 | 0.1779 | 0.0109 |
| n_runaway | 23.0000 | 23.0000 | 1.0000 | 22.0000 | 24.0000 | 2.0000 |

## The noise floor to quote

Any later difference must clear these to count as a finding.

| metric | worst-case stdev across families |
|---|---|
| median_cer | +/-0.0026 |
| mean_cer | +/-0.0096 |
| mean_cer_excl_runaway | +/-0.0055 |
| n_runaway | +/-1.0000 |

Reminder from METHODOLOGY: **rank on `median_cer`**; the mean is carried by a non-deterministic runaway tail and is far noisier.

