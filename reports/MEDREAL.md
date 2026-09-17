# Real medical scans (medreal)

Public medical-document scans assembled from HuggingFace datasets and labelled with a vision LLM (DeepSeek-v4.1-flash) using the same prompt as training. Sources: noisy-med (400 patient statements), prescription (200), india-hist (46 real 1878 scans), medform (5).

Labels: 655 pages, 0 errors, $0.00092/page ($0.6004 total).

**Caveat:** the new-set ground truth *is* the DeepSeek label, so a student distilled on it scores well there partly by construction. The old set (human-audited ClinOCR-Bench GT) is the honest test of accuracy; the new-set column mostly measures agreement with DeepSeek on real scans.

## Every model, old vs new (median CER, lower is better)

| model | old ClinOCR median | old mean | new medreal median | new mean | new runaways |
|---|---|---|---|---|---|
| Tesseract v5 (incumbent) | 0.4440 | 0.4744 | 0.2946 | 0.2958 | None |
| granite-docling (0.26B) | 0.8710 | 0.8665 | 0.3168 | 0.3817 | 13 |
| PaddleOCR-VL (0.9B) | 0.1005 | 0.4298 | 0.2416 | 0.5879 | 140 |
| dots.mocr (3B) | 0.0747 | 0.2126 | 0.1726 | 0.1985 | 2 |
| dots.ocr (3B) | 0.0761 | 0.2613 | 2.0000 | 1.1386 | 344 |
| olmOCR-2 (8B) | 0.0796 | 0.2076 | 0.2570 | 0.2755 | 3 |
| Qwen2.5-VL-3B | 0.0616 | 0.2919 | 0.1961 | 0.2324 | 7 |
| Qwen2.5-VL-7B | 0.0513 | 0.1994 | 0.1134 | 0.1782 | 4 |
| Chandra OCR 2 (5B) | 0.7134 | 0.8641 | 0.9256 | 0.9197 | 5 |
| DeepSeek-OCR (3B MoE) | 0.3007 | 0.7717 | 0.3240 | 0.3767 | 11 |
| Nanonets-OCR2 (3B) | 2.0000 | 1.8010 | 2.0000 | 1.9672 | 621 |
| Qwen3.8-27B (TP=2) | 0.0224 | 0.1059 | 0.0799 | 0.1448 | 6 |

## Distilled students (7B)

| model | old ClinOCR median | new medreal median |
|---|---|---|
| 7B base (matched res) | 0.0720 | 0.0957 |
| 7B, old synthetic 800 only | 0.0449 | 0.1194 |
| 7B, old 800 + new real | 0.0450 | 0.0821 |

