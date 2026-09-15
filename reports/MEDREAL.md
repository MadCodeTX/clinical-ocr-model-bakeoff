# Real medical scans (medreal)

Public medical-document scans assembled from HuggingFace datasets and labelled with a vision LLM (DeepSeek-v4.1-flash) using the same prompt as training. Sources: noisy-med (400 patient statements), prescription (200), india-hist (46 real 1878 scans), medform (5).

Labels: 655 pages, 0 errors, $0.00092/page ($0.6004 total).

**Caveat:** the new-set ground truth *is* the DeepSeek label, so a student distilled on it scores well there partly by construction. The old set (human-audited ClinOCR-Bench GT) is the honest test.

## Old set — ClinOCR-Bench (independent GT, 328 docs)

| model | median CER | mean CER | excl-runaway | runaways |
|---|---|---|---|---|
| 7B base | 0.0720 | 0.2112 | 0.1607 | 9 |
| 7B, trained on old synthetic 800 | 0.0449 | 0.2056 | 0.1316 | 13 |
| 7B, trained on old 800 + new real | 0.0450 | 0.1971 | 0.1462 | 9 |

## New set — medreal (DeepSeek GT, ~651 pages)

| model | median CER | mean CER | excl-runaway | runaways |
|---|---|---|---|---|
| 7B base | 0.0957 | 0.1635 | 0.1521 | 4 |
| dots.mocr | -- | -- | -- | -- |
| 7B, trained on old 800 + new real | 0.0821 | 0.1414 | 0.1270 | 5 |

