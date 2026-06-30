import subprocess
import sys

def main():
    commands = [
        ["python3", "-u", "src/optimize.py", "--target-model", "llama3.2:3b-instruct-fp16", "--meta-model", "gemma2:2b", "--task", "svamp", "--generations", "3", "--pop-size", "4", "--limit", "50", "--output", "results/optimized_llama3.2-fp16_svamp.json"],
        ["python3", "-u", "src/optimize.py", "--target-model", "llama3.2:3b-instruct-fp16", "--meta-model", "gemma2:2b", "--task", "pii", "--generations", "3", "--pop-size", "4", "--limit", "50", "--output", "results/optimized_llama3.2-fp16_pii.json"],
        ["python3", "-u", "src/optimize.py", "--target-model", "llama3.2:3b-instruct-fp16", "--meta-model", "gemma2:2b", "--task", "financial", "--generations", "3", "--pop-size", "4", "--limit", "50", "--output", "results/optimized_llama3.2-fp16_financial.json"]
    ]
    
    for idx, cmd in enumerate(commands):
        task_name = cmd[8]
        print(f"\n==========================================")
        print(f"🚀 Starting optimization for task: {task_name.upper()} ({idx+1}/{len(commands)})")
        print(f"==========================================")
        
        # We use subprocess.run and pipe stdout/stderr in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            
        process.wait()
        if process.returncode != 0:
            print(f"❌ Task {task_name} failed with return code {process.returncode}")
            sys.exit(process.returncode)
        print(f"✅ Finished task: {task_name}")

if __name__ == "__main__":
    main()
