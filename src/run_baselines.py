import subprocess
import json
import os

models = [
    "gemma2:2b", 
    "gemma2:2b-instruct-q8_0", 
    "llama3.2:latest", 
    "llama3.2:3b-instruct-q8_0"
]
tasks = ["gsm8k", "svamp", "pii", "financial"]
limit = 50

os.makedirs("results", exist_ok=True)

for model in models:
    for task in tasks:
        model_clean = model.replace(":", "-")
        output_file = f"results/baseline_{model_clean}_{task}.json"
        if os.path.exists(output_file):
            print(f"Skipping {model} on {task} (already evaluated).")
            continue
            
        print(f"\n==========================================")
        print(f"Running evaluation for {model} on {task}...")
        print(f"==========================================")
        cmd = [
            "python3", "src/evaluate.py",
            "--model", model,
            "--task", task,
            "--limit", str(limit),
            "--output", output_file
        ]
        subprocess.run(cmd)

# Summarize results into a markdown table
summary_markdown = """# QAPE Baseline Results Summary

This document summarizes the baseline performance of local quantized models on our benchmark datasets.

| Model | Task | Metric | Score (%) |
| :--- | :--- | :--- | :--- |
"""

for model in models:
    for task in tasks:
        model_clean = model.replace(":", "-")
        output_file = f"results/baseline_{model_clean}_{task}.json"
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                data = json.load(f)
                summary_markdown += f"| `{model}` | `{task}` | {data['metric']} | {data['score_percentage']:.2f}% |\n"

# Scan and compile any optimized prompt results dynamically
optimized_markdown = """
## 🧬 Optimized Prompt Results (Limit=50, 40% Val / 60% Test Split)

| Target Model | Task | Meta-Model | Baseline Test Score | Evolved Test Score | Improvement |
| :--- | :--- | :--- | :---: | :---: | :---: |
"""

has_optimized = False
for filename in sorted(os.listdir("results")):
    if filename.startswith("optimized_") and filename.endswith(".json"):
        with open(os.path.join("results", filename), "r") as f:
            data = json.load(f)
            baseline = f"{data['baseline_test_score']:.2f}%"
            evolved = f"{data['evolved_test_score']:.2f}%"
            imp = f"{data['improvement']:+.2f}%"
            if data['improvement'] > 0:
                imp = f"**{imp}**"
            optimized_markdown += f"| `{data['target_model']}` | `{data['task']}` | `{data['meta_model']}` | {baseline} | {evolved} | {imp} |\n"
            has_optimized = True

if has_optimized:
    summary_markdown += optimized_markdown

with open("results/README.md", "w") as f:
    f.write(summary_markdown)

print("\n🎉 All baseline runs complete! Summary saved to results/README.md.")
