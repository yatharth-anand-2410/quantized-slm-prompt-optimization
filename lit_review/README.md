# Literature Review & Task Selection Notes

This directory houses research notes, paper summaries, and candidate tasks for **Quantization-Aware Prompt Evolution (QAPE)**.

---

## 📚 Core Research Themes & Target Papers

We categorize the relevant literature into three primary areas:

### 1. LLM Quantization Limits & Behavioral Drift
How does quantization affect LLM representations, logic, and output consistency?

*   **PrecisionDiff: Detecting Precision-Induced Behavioral Disagreements in LLMs (2024)**
    *   *Key Takeaway:* Standard prompts on FP16 models fail to produce identical or even semantic-equivalent outputs when evaluated on lower-precision equivalents. 
    *   *Relevance:* Establishes that quantization changes the output distribution, justifying why prompts must be "evolved" specifically for the target precision.
*   **The Impact of Quantization on LLM Reasoning (Various Papers, 2023-2025)**
    *   *Key Takeaway:* Quantization below 4-bit (e.g., Q3, Q2) causes severe degradation in multi-step reasoning tasks. It degrades entropy levels in attention layers.

### 2. Automatic Prompt Optimization (APO)
What are the state-of-the-art frameworks for searching/evolving prompts?

*   **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Stanford, 2023-2024)**
    *   *Key Takeaway:* DSPy compiles pipelines by optimizing prompts (e.g., via `BootstrapFewShot`, `MIPROv2`) using a training dataset.
    *   *Relevance:* We can run DSPy optimization loops using a quantized model (e.g., Llama-3.2-3B-Q4) as the target LLM.
*   **OPRO: Optimization by PROmpting (DeepMind, 2023)**
    *   *Key Takeaway:* LLMs act as meta-optimizers to generate new prompts based on historical trial-and-error performance logs.
    *   *Relevance:* Suggests that a larger model (e.g., Llama-3-8B) can be used to optimize prompts for a smaller, quantized model (e.g., Llama-3.2-3B-Q4).
*   **GEPA: Genetic Evolutionary Prompt Algorithms**
    *   *Key Takeaway:* Evolutionary prompt mutation based on fitness functions yields robust generalization.

### 3. Task Sensitivity to Quantization
Which tasks show the most dramatic degradation under quantization?

*   **Structured Output (JSON/Tool-Calling) vs. Formatting Burden:**
    *   *Key Takeaway:* Smaller models (<8B) running at low precision (Q4, Q3) frequently violate JSON syntax or miss function arguments.
    *   *Relevance:* This makes structured output one of the most high-impact targets for QAPE, as recovering syntax compliance has immediate commercial value.

---

## 🎯 Candidate Tasks for QAPE Benchmarking

We need tasks that show **high degradation under quantization** (so there is a gap to recover) but are **computational feasible** to run in an evolutionary loop on an M3 Mac (16GB RAM).

| Task Name | Dataset Source | Quantization Sensitivity | Evaluation Complexity | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Mathematical Reasoning** | GSM8K (subsets) | High (Q4/Q3 show math drift) | Medium (regex parser on final answer) | Solve word math problems. Requires chain-of-thought (CoT). |
| **JSON Schema Adherence** | Synthetic JSON extraction | Extremely High (Q3/Q4 lose structure) | Easy (parseable JSON validation) | Transform messy medical/legal records into strict JSON schemas. |
| **Tool/Function Calling** | ToolBench or custom API-calling | Very High (syntax errors, missed params) | Medium (exact match on tool arguments) | Select the correct API and format the parameters under system constraints. |
| **Multi-Hop QA** | HotpotQA | Medium | High (exact match / F1 score on answer) | Retrieve and reason over multiple context blocks to answer a question. |

---

## 📝 Next Steps for Lit Review & Task Selection
1. **Identify the exact target tasks:** We will test a small sample (e.g., 50 examples) of GSM8K and JSON Extraction on `Llama-3.2-3B` at FP16, Q8, Q4, and Q3 to see which task exhibits the clearest degradation.
2. **Draft the Related Work section:** Start summarizing the findings on precision drift and prompt optimization.
