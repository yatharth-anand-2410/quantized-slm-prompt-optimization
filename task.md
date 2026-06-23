# 📋 QAPE Project Task Board

Track the status of the Quantization-Aware Prompt Evolution (QAPE) research tasks.

---

## 🚀 Current Status: Phase 2 (Experimental Run - Baseline & Degradation)

| Phase | Description | Target Date | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Literature Review & Setup | June 17, 2026 | ✅ Completed |
| **Phase 2** | Experimental Run - Baseline & Degradation | June 30, 2026 | 🔄 In Progress |
| **Phase 3** | Prompt Optimization Loop | July 14, 2026 | ⏳ Scheduled |
| **Phase 4** | Drafting EMNLP Short Paper | July 28, 2026 | ⏳ Scheduled |
| **Phase 5** | Polish & Submission | August 2, 2026 | ⏳ Scheduled |

---

## 📝 Detailed Task Checklist

### Phase 1: Lit Review & Setup (Completed)
- [x] Literature review on precision drift and prompt optimization.
- [x] Local environment setup (Ollama local inference).
- [x] Dataset downloader script and fetching benchmarks (GSM8K, SVAMP, Cleanlab PII & Financial NER).

### Phase 2: Baseline & Degradation (In Progress)
- [x] Run baseline evaluations for `gemma2:2b` (4-bit).
- [x] Run baseline evaluations for `llama3.2:latest` (3B, 4-bit).
- [x] Run baseline evaluations for `gemma2:2b-instruct-q8_0` (8-bit).
- [x] Run baseline evaluations for `llama3.2:3b-instruct-q8_0` (8-bit).
- [ ] Run baseline evaluations for remaining low-precision models (e.g., Q3/Q2 if applicable).
- [x] Document precision-induced degradation observations.

### Phase 3: Prompt Optimization Loop (Scheduled)
- [x] Implement genetic prompt mutation algorithm (`src/optimize.py`).
- [x] Run prompt evolution for `llama3.2:latest` on Financial task.
- [x] Run prompt evolution for `llama3.2:latest` on PII extraction task.
- [x] Run prompt evolution on GSM8K math task.
- [ ] Run prompt evolution on SVAMP math task.
- [ ] Perform cross-precision transfer evaluations (e.g., testing evolved Q4 prompts on FP16 and vice-versa).
- [ ] Compile performance recovery statistics.

### Phase 4 & 5: Paper Drafting & Submission (Scheduled)
- [ ] Setup EMNLP LaTeX template.
- [ ] Draft core sections (Abstract, Intro, Methods, Results, Discussion).
- [ ] Generate prompt transfer matrix visualization.
- [ ] Final proofread and submission.
