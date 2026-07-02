import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    results_json = "results/cross_precision_transfer.json"
    
    if not os.path.exists(results_json):
        print(f"Error: {results_json} not found.")
        return
        
    with open(results_json, "r") as f:
        data = json.load(f)
        
    results = data.get("results", {})
    
    tasks = ["gsm8k", "svamp", "pii", "financial"]
    model_q4 = "llama3.2:latest"
    model_fp16 = "llama3.2:3b-instruct-fp16"
    
    # Setup subplots: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    fig.suptitle("QAPE Cross-Precision Prompt Transfer Matrices (Llama 3.2 3B)", fontsize=15, fontweight='bold', y=0.96)
    
    # Source / Target labels
    source_labels = ["Evolved Q4 Prompt", "Evolved FP16 Prompt"]
    target_labels = ["Q4 Model", "FP16 Model"]
    
    # Custom color maps per task for visual distinction or clean paper styling
    cmap = plt.cm.Blues
    
    for idx, task in enumerate(tasks):
        ax = axes[idx // 2, idx % 2]
        task_data = results.get(task, {})
        
        # Extract scores
        # Baseline scores for reference in the subtitle/annotation
        q4_baseline = task_data.get(model_q4, {}).get("baseline_prompt", 0.0)
        fp16_baseline = task_data.get(model_fp16, {}).get("baseline_prompt", 0.0)
        
        # Grid of:
        # [ [ Q4-prompt-on-Q4-model,  Q4-prompt-on-FP16-model  ],
        #   [ FP16-prompt-on-Q4-model, FP16-prompt-on-FP16-model ] ]
        q4_on_q4 = task_data.get(model_q4, {}).get("evolved_q4_prompt", 0.0)
        q4_on_fp16 = task_data.get(model_fp16, {}).get("evolved_q4_prompt", 0.0)
        
        fp16_on_q4 = task_data.get(model_q4, {}).get("evolved_fp16_prompt", 0.0)
        fp16_on_fp16 = task_data.get(model_fp16, {}).get("evolved_fp16_prompt", 0.0)
        
        matrix = np.array([
            [q4_on_q4, q4_on_fp16],
            [fp16_on_q4, fp16_on_fp16]
        ])
        
        # Draw heatmap
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100)
        
        # Add labels
        ax.set_xticks(np.arange(len(target_labels)))
        ax.set_yticks(np.arange(len(source_labels)))
        ax.set_xticklabels(target_labels, fontsize=10, fontweight='bold')
        ax.set_yticklabels(source_labels, fontsize=10, fontweight='bold')
        
        # Set titles
        ax.set_title(f"Task: {task.upper()}", fontsize=12, fontweight='bold', pad=10)
        
        # Add annotations inside the heatmaps
        for i in range(len(source_labels)):
            for j in range(len(target_labels)):
                score = matrix[i, j]
                # Determine text color based on background intensity
                text_color = "white" if score > 50 else "black"
                ax.text(j, i, f"{score:.2f}%", ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')
                
        # Draw baseline references as minor labels at the bottom of the subplots
        baseline_text = f"Baseline Prompts:\nQ4 Model: {q4_baseline:.2f}% | FP16 Model: {fp16_baseline:.2f}%"
        ax.text(0.5, -0.22, baseline_text, transform=ax.transAxes, ha="center", va="center", fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#d0d0d0", lw=1))
        
        # Adjust visual frames
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(matrix.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(matrix.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the PNG plot
    output_png = "results/prompt_transfer_matrix.png"
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"📈 Saved transfer matrix plot to {output_png}")
    
    # Save the PDF plot for paper draft
    output_pdf = "paper/prompt_transfer_matrix.pdf"
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"📄 Saved paper PDF figure to {output_pdf}")
    
    # Save a copy to the artifacts directory
    artifacts_dir = "/Users/stokome/.gemini/antigravity/brain/2ffb461d-a19b-46d1-bcdf-4ae324805895"
    if os.path.exists(artifacts_dir):
        artifact_png = os.path.join(artifacts_dir, "prompt_transfer_matrix.png")
        plt.savefig(artifact_png, dpi=150, bbox_inches='tight')
        print(f"💼 Saved copy to artifacts: {artifact_png}")

if __name__ == "__main__":
    main()
