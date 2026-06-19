# QAPE Baseline Results Summary

This document summarizes the baseline performance of local quantized models on our benchmark datasets.

| Model | Task | Metric | Score (%) |
| :--- | :--- | :--- | :--- |
| `gemma2:2b` | `gsm8k` | Accuracy | 54.00% |
| `gemma2:2b` | `svamp` | Accuracy | 60.00% |
| `gemma2:2b` | `pii` | F1 Score | 33.67% |
| `gemma2:2b` | `financial` | F1 Score | 72.26% |
| `llama3.2:latest` | `gsm8k` | Accuracy | 68.00% |
| `llama3.2:latest` | `svamp` | Accuracy | 78.00% |
| `llama3.2:latest` | `pii` | F1 Score | 24.49% |
| `llama3.2:latest` | `financial` | F1 Score | 62.28% |

## 🧬 Optimized Prompt Results (Limit=50, 40% Val / 60% Test Split)

| Target Model | Task | Meta-Model | Baseline Test Score | Evolved Test Score | Improvement |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `llama3.2:latest` | `pii` | `gemma2:2b` | 27.15% | 24.29% | -2.86% |
| `llama3.2:latest` | `financial` | `gemma2:2b` | 55.81% | 64.94% | **+9.14%** |

