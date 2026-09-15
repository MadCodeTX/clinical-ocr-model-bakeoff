# Noise floor (A1)

Independent repeats of the *same* model at the *same* config, so every spread here is measurement noise, not a modelling difference.

## dots.mocr (base, GPU0)  (n=5)

| metric | mean | median | stdev | min | max | range |
|---|---|---|---|---|---|---|
| median_cer | 0.0750 | 0.0747 | 0.0005 | 0.0747 | 0.0757 | 0.0010 |
| mean_cer | 0.2174 | 0.2183 | 0.0025 | 0.2135 | 0.2201 | 0.0066 |
| mean_cer_excl_runaway | 0.1671 | 0.1680 | 0.0019 | 0.1642 | 0.1688 | 0.0046 |
| n_runaway | 9.0000 | 9.0000 | 0.7071 | 8.0000 | 10.0000 | 2.0000 |

## Qwen2.5-VL-3B + LoRA (student, GPU1)  (n=5)

| metric | mean | median | stdev | min | max | range |
|---|---|---|---|---|---|---|
| median_cer | 0.0966 | 0.0962 | 0.0022 | 0.0947 | 0.1003 | 0.0056 |
| mean_cer | 0.2968 | 0.2955 | 0.0102 | 0.2832 | 0.3112 | 0.0280 |
| mean_cer_excl_runaway | 0.1732 | 0.1723 | 0.0046 | 0.1670 | 0.1779 | 0.0109 |
| n_runaway | 22.2000 | 23.0000 | 1.9235 | 19.0000 | 24.0000 | 5.0000 |

## The noise floor to quote

Any later difference must clear these to count as a finding.

| metric | worst-case stdev across families |
|---|---|
| median_cer | +/-0.0022 |
| mean_cer | +/-0.0102 |
| mean_cer_excl_runaway | +/-0.0046 |
| n_runaway | +/-1.9235 |

Reminder from METHODOLOGY: **rank on `median_cer`**; the mean is carried by a non-deterministic runaway tail and is far noisier.

