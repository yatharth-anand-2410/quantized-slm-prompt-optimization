# QAPE Baseline Results Summary

This document summarizes the baseline performance of local quantized models on our benchmark datasets.

| Model | Task | Metric | Score (%) |
| :--- | :--- | :--- | :--- |
| `gemma2:2b` | `gsm8k` | Accuracy | 54.00% |
| `gemma2:2b` | `svamp` | Accuracy | 60.00% |
| `gemma2:2b` | `pii` | F1 Score | 33.67% |
| `gemma2:2b` | `financial` | F1 Score | 72.26% |
| `gemma2:2b-instruct-q8_0` | `gsm8k` | Accuracy | 56.00% |
| `gemma2:2b-instruct-q8_0` | `svamp` | Accuracy | 60.00% |
| `gemma2:2b-instruct-q8_0` | `pii` | F1 Score | 33.71% |
| `gemma2:2b-instruct-q8_0` | `financial` | F1 Score | 65.61% |
| `gemma2:2b-instruct-fp16` | `gsm8k` | Accuracy | 44.00% |
| `gemma2:2b-instruct-fp16` | `svamp` | Accuracy | 56.00% |
| `gemma2:2b-instruct-fp16` | `pii` | F1 Score | 34.27% |
| `gemma2:2b-instruct-fp16` | `financial` | F1 Score | 66.05% |
| `llama3.2:latest` | `gsm8k` | Accuracy | 68.00% |
| `llama3.2:latest` | `svamp` | Accuracy | 78.00% |
| `llama3.2:latest` | `pii` | F1 Score | 24.49% |
| `llama3.2:latest` | `financial` | F1 Score | 62.28% |
| `llama3.2:3b-instruct-q8_0` | `gsm8k` | Accuracy | 48.00% |
| `llama3.2:3b-instruct-q8_0` | `svamp` | Accuracy | 76.00% |
| `llama3.2:3b-instruct-q8_0` | `pii` | F1 Score | 19.04% |
| `llama3.2:3b-instruct-q8_0` | `financial` | F1 Score | 60.12% |
| `llama3.2:3b-instruct-fp16` | `gsm8k` | Accuracy | 78.00% |
| `llama3.2:3b-instruct-fp16` | `svamp` | Accuracy | 76.00% |
| `llama3.2:3b-instruct-fp16` | `pii` | F1 Score | 23.53% |
| `llama3.2:3b-instruct-fp16` | `financial` | F1 Score | 60.40% |

## 🧬 Optimized Prompt Results (Limit=50, 40% Val / 60% Test Split)

| Target Model | Task | Meta-Model | Baseline Test Score | Evolved Test Score | Improvement |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `llama3.2:latest` | `financial` | `gemma2:2b` | 55.81% | 64.94% | **+9.14%** |
| `llama3.2:latest` | `gsm8k` | `gemma2:2b` | 76.67% | 76.67% | +0.00% |
| `llama3.2:latest` | `pii` | `gemma2:2b` | 27.15% | 24.29% | -2.86% |
