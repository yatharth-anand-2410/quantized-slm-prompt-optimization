# Quantized SLM Prompt Optimization (QAPE)

This repository contains the official implementation of **QAPE: Quantization-Aware Prompt Evolution for Quantized Small Language Models**, a genetic algorithm-based prompt optimization framework designed to recover performance loss in quantized Small Language Models (SLMs) without weight fine-tuning or retraining.

---

## 🔬 Core Findings
1. **Performance Recovery:** QAPE recovers up to **+25.02%** absolute F1 score on structured extraction (Financial NER) on the sub-1B `Qwen-2.5-0.5B` model, and **+10.00%** accuracy on math reasoning (SVAMP) under 4-bit (Q4) quantization.
2. **Precision-Dependent Specialization:** On mathematical reasoning tasks (GSM8K and SVAMP), prompts evolved directly on quantized weights specialize to the quantization-induced noise grid. Transferring them across precision boundaries (e.g., 4-bit to FP16) introduces a minor performance drop (6.67% to 13.33%) compared to direct optimization.
3. **Format Resilience:** On structured information extraction, prompts evolved under low-precision targets act as strong structural anchors, providing excellent transferability (and even performance gains) when executed on full-precision models.

---

## 📂 Directory Structure
* `src/`
  * `evaluate.py` - Base evaluation runner supporting GSM8K, SVAMP, PII, and Financial NER.
  * `optimize.py` - Genetic prompt mutation loop and evolution controller.
  * `cross_precision_transfer.py` - Evaluator for prompt transferability across precision boundaries (Q4 vs. FP16).
  * `plot_degradation.py` - Script to compile and plot model precision degradation curves.
  * `plot_transfer_matrix.py` - Script to generate the $2 \times 2$ prompt transfer matrix heatmaps.
  * `check_latex.py` - Automated syntax auditor for ACL/EMNLP LaTeX documents.
* `data/` - Curation subsets for GSM8K, SVAMP, PII, and Financial NER.
* `results/` - Evaluation outputs, checkpoints, and compiled plots.
* `paper/` - LaTeX source files and style documents for EMNLP short paper submission.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ and [Ollama](https://ollama.com/) installed on your system.

### 2. Install Dependencies
We recommend using `uv` for fast package management, but standard `pip` works as well:
```bash
# Using uv
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### 3. Pull Target Models
Start your local Ollama daemon and pull the target evaluations and meta-models:
```bash
# Pull evaluation targets
ollama pull llama3.2:latest              # Q4 Quantized
ollama pull llama3.2:3b-instruct-fp16     # Full Precision (FP16)
ollama pull qwen2.5:0.5b                 # Q4 Quantized

# Pull meta-model for prompt mutations
ollama pull gemma2:2b                    # Q4 Quantized
```

---

## 🚀 Running Experiments

### 1. Baseline Evaluation
To evaluate the base models using initial prompts across tasks:
```bash
python3 src/evaluate.py --model llama3.2:latest --task gsm8k --limit 30
```

### 2. Prompt Evolution
To run the QAPE genetic prompt optimization loop:
```bash
python3 src/optimize.py --model llama3.2:latest --task financial --meta-model gemma2:2b --generations 5 --pop-size 4
```

### 3. Cross-Precision Transfer Matrix
To evaluate cross-precision prompt transferability between Q4 and FP16:
```bash
python3 src/cross_precision_transfer.py
```

### 4. Plotting & Verification
To regenerate the degradation curves and EMNLP transfer heatmaps:
```bash
# Generate precision degradation plots
uv run --with matplotlib --with numpy src/plot_degradation.py

# Generate cross-precision transfer matrices
uv run --with matplotlib --with numpy src/plot_transfer_matrix.py

# Verify LaTeX draft syntax and citations
python3 src/check_latex.py
```
