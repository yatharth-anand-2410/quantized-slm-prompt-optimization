import os
import subprocess

model = "qwen2.5:0.5b"
meta_model = "gemma2:2b"
tasks = ["gsm8k", "svamp", "pii", "financial"]
limit = 50
generations = 3
pop_size = 4

os.makedirs("results", exist_ok=True)

for task in tasks:
    output_file = f"results/optimized_qwen2.5-0.5b_{task}.json"
    if os.path.exists(output_file):
        print(f"Skipping optimization for {model} on {task} (already completed).")
        continue
        
    print(f"\n==========================================")
    print(f"Running prompt optimization for {model} on {task}...")
    print(f"==========================================")
    cmd = [
        "python3", "src/optimize.py",
        "--target-model", model,
        "--meta-model", meta_model,
        "--task", task,
        "--limit", str(limit),
        "--generations", str(generations),
        "--pop-size", str(pop_size),
        "--output", output_file
    ]
    subprocess.run(cmd)

# Re-run run_baselines.py to compile all baseline and optimized results into results/README.md
print("\n📝 Compiling results into README...")
subprocess.run(["python3", "src/run_baselines.py"])
print("🎉 All optimizations and compilation complete!")
