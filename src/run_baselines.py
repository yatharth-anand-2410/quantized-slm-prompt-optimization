import subprocess
import json
import os

models = ["gemma2:2b", "llama3.2:latest"]
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

with open("results/README.md", "w") as f:
    f.write(summary_markdown)

print("\n🎉 All baseline runs complete! Summary saved to results/README.md.")
