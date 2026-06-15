#!/bin/bash

# ==========================================
# 🤖 QAPE Daily Automation Script
# ==========================================

# Navigate to the repository directory
REPO_DIR="/Users/stokome/Desktop/Personal/quantized-slm-prompt-optimization"
cd "$REPO_DIR" || exit

echo "🕒 Starting Daily QAPE Research Automation: $(date)"

# 1. Ensure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚙️ Starting Ollama server..."
    /opt/homebrew/bin/ollama serve > /dev/null 2>&1 &
    sleep 5
fi

# 2. Run baseline evaluations (if models are pulled)
echo "📊 Running evaluations..."
if /opt/homebrew/bin/ollama list | grep -q "gemma2:2b"; then
    python3 src/evaluate.py --model gemma2:2b --task json >> results/evaluation_log.txt 2>&1
    python3 src/evaluate.py --model gemma2:2b --task math >> results/evaluation_log.txt 2>&1
fi

# 3. Sync changes to GitHub
echo "🐙 Syncing changes with GitHub..."
git add .

# Check if there are changes to commit
if ! git diff-index --quiet HEAD --; then
    COMMIT_MSG="auto: daily research evaluation update - $(date +'%Y-%m-%d %H:%M')"
    git commit -m "$COMMIT_MSG"
    git push origin main
    echo "✅ Changes committed and pushed: '$COMMIT_MSG'"
else
    echo "ℹ️ No changes detected. Repository is up-to-date."
fi

echo "🏁 Daily automation cycle complete."
