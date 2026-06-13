# Quantized SLM Prompt Optimization

This project investigates **Quantization-Aware Prompt Evolution (QAPE)** as a zero-training-compute alternative to fine-tuning for Small Language Models (SLMs) on consumer hardware (e.g., M3 MacBook Air).

## 🔬 Core Objectives
1. **Benchmark Degradation:** Measure how post-training quantization (e.g., FP16 vs Q8 vs Q4 vs Q3) degrades structured reasoning and tool-calling accuracy in sub-8B models.
2. **Reflective Prompt Evolution:** Run genetic prompt optimization (like GEPA) directly on the quantized model to recover performance.
3. **Cross-Model Resilience:** Evaluate if prompts evolved on a quantized model outperform prompts evolved on full-precision models when both are run at low precision.

## 🛠️ Local Stack
* **Models:** `Llama-3.2-3B-Instruct`, `Gemma-2-2B-IT`, `Phi-3.5-mini-3.8B`.
* **Inference:** Ollama or MLX-LM.
* **Orchestration:** DSPy.

## 📂 Directory Structure
* `README.md` - Project description and research roadmap.
* `src/` - Python scripts for agent setup and optimization loops (to be added).
* `data/` - Training/evaluation datasets (e.g., GSM8K subsets).
* `results/` - Accuracies and prompt evolution traces.
