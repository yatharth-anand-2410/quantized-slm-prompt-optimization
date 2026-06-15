import argparse
import json
import re
import urllib.request
import urllib.error

# ==========================================
# 📊 Test Datasets
# ==========================================

# 1. JSON Extraction Dataset (Messy text to strict JSON)
JSON_DATASET = [
    {
        "input": "Yesterday I met a developer named Alice Smith. She is 28 years old and lives in Seattle. Her core skills are Python, Docker, and Kubernetes.",
        "target": {"name": "Alice Smith", "age": 28, "city": "Seattle", "skills": ["Python", "Docker", "Kubernetes"]}
    },
    {
        "input": "We have a new client. John Doe, age 45, based in Chicago. He is fluent in Java and Go.",
        "target": {"name": "John Doe", "age": 45, "city": "Chicago", "skills": ["Java", "Go"]}
    },
    {
        "input": "Resume profile: Maria Garcia. Location: Miami, Florida. Age: 34. Key strengths: SQL, Tableau, PowerBI, Excel.",
        "target": {"name": "Maria Garcia", "age": 34, "city": "Miami", "skills": ["SQL", "Tableau", "PowerBI", "Excel"]}
    },
    {
        "input": "Contact info: David Kim, 22. Currently living in Boston. Skills: JavaScript, React, Node.js.",
        "target": {"name": "David Kim", "age": 22, "city": "Boston", "skills": ["JavaScript", "React", "Node.js"]}
    },
    {
        "input": "Employee database entry: Sarah Connor, 39, Los Angeles. Proficient in C++, Rust, and Assembly.",
        "target": {"name": "Sarah Connor", "age": 39, "city": "Los Angeles", "skills": ["C++", "Rust", "Assembly"]}
    }
]

# 2. Mathematical Reasoning Dataset (GSM8K Subset)
MATH_DATASET = [
    {
        "input": "Weng earns $12 an hour for baby-sitting. Yesterday, she baby-sat for 5 hours. How much did she earn?",
        "target": 60
    },
    {
        "input": "A classroom has 3 rows of desks with 5 desks in each row. If 4 desks are removed, how many desks are left?",
        "target": 11
    },
    {
        "input": "James decided to build a dog house. It took him 3 hours to cut the wood, 2 hours to assemble it, and 1 hour to paint it. If he did this over 2 days, spending equal time each day, how many hours did he work per day?",
        "target": 3
    },
    {
        "input": "Mary has 15 apples. She eats 3 of them and gives 4 to her brother. She then buys 6 more apples. How many apples does Mary have now?",
        "target": 14
    },
    {
        "input": "A toy store sells toy cars for $4 each and toy trucks for $6 each. If Tim buys 3 cars and 2 trucks, how much does he spend in total?",
        "target": 24
    }
]

# ==========================================
# 🔌 Ollama Client Implementation
# ==========================================

def query_ollama(model: str, prompt: str, temperature: float = 0.0) -> str:
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("response", "")
    except urllib.error.URLError as e:
        print(f"Error connecting to Ollama: {e}")
        return ""

# ==========================================
# ⚙️ Task Scoring & Verification
# ==========================================

def score_json(output: str, target: dict) -> float:
    # Try to extract JSON from markdown wrappers
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        return 0.0
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0
    
    # Check key completeness and types
    score = 0.0
    for key in ["name", "age", "city", "skills"]:
        if key not in data:
            return 0.0
            
    if data["name"] == target["name"]:
        score += 0.25
    if int(data["age"]) == int(target["age"]):
        score += 0.25
    if target["city"].lower() in data["city"].lower():
        score += 0.25
    
    # Check skills list semantic match (case insensitive)
    target_skills = set(s.lower() for s in target["skills"])
    output_skills = set(s.lower() for s in data["skills"]) if isinstance(data["skills"], list) else set()
    if target_skills == output_skills:
        score += 0.25
        
    return score

def score_math(output: str, target: int) -> float:
    # Look for standard #### <value> pattern
    match = re.search(r"####\s*(-?\d+)", output)
    if not match:
        # Fallback: look for the last number in the text
        numbers = re.findall(r"-?\d+", output)
        if not numbers:
            return 0.0
        val = int(numbers[-1])
    else:
        val = int(match.group(1))
        
    return 1.0 if val == target else 0.0

# ==========================================
# 🚀 Run Evaluation
# ==========================================

def run_evaluation(model: str, task: str):
    print(f"\n🚀 Evaluating model: '{model}' on task: '{task}'")
    
    if task == "json":
        dataset = JSON_DATASET
        prompt_template = (
            "You are a precise data extraction agent. Extract the details of the person mentioned in the text below "
            "into a JSON object matching the following structure:\n"
            "{{\n"
            "  \"name\": string,\n"
            "  \"age\": integer,\n"
            "  \"city\": string,\n"
            "  \"skills\": list of strings\n"
            "}}\n"
            "Do not output any introductory or explanatory text. Output ONLY valid raw JSON.\n\n"
            "Text: {input}\n\n"
            "JSON:"
        )
        score_fn = score_json
    elif task == "math":
        dataset = MATH_DATASET
        prompt_template = (
            "Solve the following math word problem step by step. Write your final numerical answer at the end "
            "preceded by '#### ' (e.g. '#### 42').\n\n"
            "Problem: {input}\n\n"
            "Solution:"
        )
        score_fn = score_math
    else:
        print(f"Unknown task: {task}")
        return

    total_score = 0.0
    for idx, item in enumerate(dataset):
        prompt = prompt_template.format(input=item["input"])
        output = query_ollama(model, prompt)
        score = score_fn(output, item["target"])
        total_score += score
        
        print(f"\n--- [Case {idx+1}] ---")
        print(f"Input: {item['input']}")
        print(f"Target: {item['target']}")
        print(f"Output: {output.strip()}")
        print(f"Score: {score}")

    accuracy = (total_score / len(dataset)) * 100
    print(f"\n📊 Accuracy: {accuracy:.2f}% ({total_score}/{len(dataset)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate quantized models on structured reasoning tasks.")
    parser.add_argument("--model", type=str, required=True, help="Ollama model name (e.g. gemma2:2b)")
    parser.add_argument("--task", type=str, choices=["json", "math"], required=True, help="Evaluation task")
    args = parser.parse_args()
    
    run_evaluation(args.model, args.task)
