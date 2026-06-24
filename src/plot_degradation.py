import os
import json
import re
import matplotlib.pyplot as plt

def main():
    results_dir = "results"
    
    # We want to compile scores for:
    # Model families: Gemma-2-2B, Llama-3.2-3B
    # Precisions: FP16 (16 bits), Q8 (8 bits), Q4 (4 bits)
    # Tasks: gsm8k, svamp, pii, financial
    
    tasks = ["gsm8k", "svamp", "pii", "financial"]
    precisions = ["FP16", "Q8", "Q4"]
    precision_bits = {"FP16": 16, "Q8": 8, "Q4": 4}
    
    # Structure to hold scores: model_family -> task -> precision -> score
    data = {
        "Gemma-2-2B": {task: {p: None for p in precisions} for task in tasks},
        "Llama-3.2-3B": {task: {p: None for p in precisions} for task in tasks}
    }
    
    # Regex to parse baseline file names
    # Examples:
    # baseline_gemma2-2b_gsm8k.json -> Gemma-2-2B, Q4, gsm8k
    # baseline_gemma2-2b-instruct-q8_0_gsm8k.json -> Gemma-2-2B, Q8, gsm8k
    # baseline_gemma2-2b-instruct-fp16_gsm8k.json -> Gemma-2-2B, FP16, gsm8k
    # baseline_llama3.2-latest_gsm8k.json -> Llama-3.2-3B, Q4, gsm8k
    # baseline_llama3.2-3b-instruct-q8_0_gsm8k.json -> Llama-3.2-3B, Q8, gsm8k
    # baseline_llama3.2-3b-instruct-fp16_gsm8k.json -> Llama-3.2-3B, FP16, gsm8k
    
    for filename in os.listdir(results_dir):
        if not filename.startswith("baseline_") or not filename.endswith(".json") or filename == "baseline_results.json":
            continue
            
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r") as f:
            try:
                res_data = json.load(f)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
                
        model_name = res_data.get("model", "")
        task_name = res_data.get("task", "")
        score = res_data.get("score_percentage", None)
        
        if score is None or task_name not in tasks:
            continue
            
        # Classify model family and precision
        family = None
        precision = None
        
        if "gemma2:2b" in model_name or "gemma2-2b" in model_name:
            family = "Gemma-2-2B"
            if "fp16" in model_name:
                precision = "FP16"
            elif "q8_0" in model_name:
                precision = "Q8"
            else:
                precision = "Q4"
        elif "llama3.2" in model_name or "llama3.2-latest" in model_name or "llama3.2-3b" in model_name:
            family = "Llama-3.2-3B"
            if "fp16" in model_name:
                precision = "FP16"
            elif "q8_0" in model_name:
                precision = "Q8"
            else:
                precision = "Q4"
                
        if family and precision:
            data[family][task_name][precision] = score

    # Print markdown table
    print("# Compiled Precision Degradation Metrics\n")
    print("| Model Family | Task | FP16 Score | Q8 Score | Q4 Score | Degradation (FP16 -> Q4) |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: |")
    
    for family in sorted(data.keys()):
        for task in tasks:
            scores = data[family][task]
            fp16 = f"{scores['FP16']:.2f}%" if scores['FP16'] is not None else "N/A"
            q8 = f"{scores['Q8']:.2f}%" if scores['Q8'] is not None else "N/A"
            q4 = f"{scores['Q4']:.2f}%" if scores['Q4'] is not None else "N/A"
            
            if scores['FP16'] is not None and scores['Q4'] is not None:
                deg = f"{scores['Q4'] - scores['FP16']:+.2f}%"
            else:
                deg = "N/A"
                
            print(f"| {family} | `{task}` | {fp16} | {q8} | {q4} | {deg} |")
            
    # Plot curves using matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Precision Degradation Curves (FP16 -> Q8 -> Q4)", fontsize=16, fontweight='bold')
    
    colors = {"Gemma-2-2B": "#d9381e", "Llama-3.2-3B": "#1a73e8"}
    markers = {"Gemma-2-2B": "o", "Llama-3.2-3B": "s"}
    
    for idx, task in enumerate(tasks):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(f"Task: {task.upper()}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Precision (effective bits)")
        ax.set_ylabel("Score (%)")
        ax.set_xticks([16, 8, 4])
        ax.set_xticklabels(["FP16 (16-bit)", "Q8_0 (8-bit)", "Q4_0 (4-bit)"])
        ax.grid(True, linestyle="--", alpha=0.6)
        
        for family in data.keys():
            scores = data[family][task]
            x_vals = []
            y_vals = []
            
            # Map precisions to bits
            for p in ["FP16", "Q8", "Q4"]:
                if scores[p] is not None:
                    x_vals.append(precision_bits[p])
                    y_vals.append(scores[p])
            
            if x_vals:
                ax.plot(x_vals, y_vals, label=family, color=colors[family], marker=markers[family], linewidth=2.5, markersize=8)
                # Annotate data points
                for x, y in zip(x_vals, y_vals):
                    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
                    
        ax.legend(loc="lower left" if task in ["pii", "financial"] else "lower right")
        ax.set_ylim(-5, 105)
        
    plt.tight_layout()
    
    # Save the plot
    output_plot_path = os.path.join(results_dir, "precision_degradation.png")
    plt.savefig(output_plot_path, dpi=150)
    print(f"\n📈 Saved degradation plot to {output_plot_path}")
    
    # Save to artifacts directory as well
    artifacts_dir = "/Users/stokome/.gemini/antigravity/brain/2ffb461d-a19b-46d1-bcdf-4ae324805895"
    if os.path.exists(artifacts_dir):
        artifact_plot_path = os.path.join(artifacts_dir, "precision_degradation.png")
        plt.savefig(artifact_plot_path, dpi=150)
        print(f"💼 Saved degradation plot copy to artifacts: {artifact_plot_path}")

if __name__ == "__main__":
    main()
