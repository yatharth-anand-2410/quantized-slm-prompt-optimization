# QAPE Literature Review & Novelty Analysis

This document compiles key insights from recent literature in **LLM precision drift** and **prompt optimization**, establishing the scientific foundation and novelty of **Quantization-Aware Prompt Evolution (QAPE)**.

---

## 🔑 Key Academic Papers

### 1. PrecisionDiff (April 2026)
*   **Title:** *Hidden Reliability Risks in Large Language Models: Systematic Identification of Precision-Induced Output Disagreements*
*   **Key Findings:**
    *   Quantization (e.g., FP16 $\to$ INT8/INT4) introduces systematic, non-linear shifts in the model’s internal representations.
    *   Standard prompts that are highly optimized for FP16 models frequently experience **precision-induced behavioral disagreements** when executed at low precision.
    *   These disagreements degrade instruction-following, safety boundaries (leading to jailbreaks), and reasoning consistency.
*   **How QAPE builds on this:** PrecisionDiff *identifies* the discrepancies between precisions but does not propose a method to fix or align them. QAPE serves as the **compensatory framework**: by evolving prompts directly on the quantized model, we attempt to align its low-precision output distribution back to the high-precision baseline.

### 2. FOZO (2024/2025)
*   **Title:** *FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation*
*   **Key Findings:**
    *   Demonstrates that prompt optimization (specifically zeroth-order gradients) can adapt models at test-time under memory and compute constraints.
    *   Validates that prompt optimization remains effective even when running on quantized (INT8) models.
*   **How QAPE builds on this:** FOZO focuses on soft prompt tuning and continuous parameter optimization under memory constraints. QAPE, in contrast, focuses on **discrete prompt optimization** (natural language instructions and few-shot exemplars) which is:
    1.  Fully black-box (doesn't require gradient access, making it compatible with APIs and standard local engines like Ollama).
    2.  Completely zero-training-compute (no backpropagation or weight/embedding modification).
    3.  Human-interpretable and easily transferable.

---

## 💡 Scientific Novelty of QAPE

Our proposed methodology focuses on **Cross-Precision Generalization ($RQ_3$)**, which has not been systematically mapped in literature:
1.  **Quantization-Aware Fitness:** Standard discrete prompt optimizers (like DSPy or OPRO) assume a static, full-precision model. QAPE is the first systematic evaluation of optimizing discrete instructions using the *quantized model's fitness landscape* as the feedback loop.
2.  **Cross-Precision Transfer Matrix:** We will construct a transferability matrix mapping prompts optimized at precision $A$ and evaluated at precision $B$.
    *   *Conventional wisdom:* A prompt optimized on a larger/better model (FP16) should always be better.
    *   *Our hypothesis:* A prompt optimized specifically on the target quantized model ($P_{Q4}$) will outperform $P_{FP16}$ when evaluated on the $Q4$ model because it adapts to the model's altered quantization activation pathways (e.g., avoiding terms that trigger quantized attention errors).

---

## 📊 Task Sensitivity Hypotheses (For Phase 1 Selection)

Based on literature, we hypothesize the following sensitivity levels across tasks:

1.  **JSON / Schema Adherence (Highest Sensitivity):**
    *   *Why:* Quantization limits the precision of token representations, making the model more likely to drift away from strict syntax requirements (brackets, quotes).
    *   *QAPE Potential:* High. Optimization can find prompts that use simplified structures or explicit XML-like anchors that are more resilient to quantization drift.
2.  **Mathematical Reasoning - GSM8K (High Sensitivity):**
    *   *Why:* Multi-step reasoning requires coherent accumulation of context. Quantization errors accumulate with each generated token in the chain-of-thought (CoT).
    *   *QAPE Potential:* High. Optimization can discover simpler CoT structures or instructions that restrict the model's vocabulary to more robust, high-frequency mathematical tokens.
3.  **Conversational Tool-Calling (Medium Sensitivity):**
    *   *Why:* Tool-calling relies on parsing function arguments. 
    *   *QAPE Potential:* Medium. Formatting constraints can be optimized.
