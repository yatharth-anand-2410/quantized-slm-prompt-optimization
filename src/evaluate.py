import argparse
import json
import re
import urllib.request
import urllib.error
import ast
import os

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
            "temperature": temperature,
            "num_predict": 512
        }
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("response", "")
    except Exception as e:
        print(f"Error connecting to Ollama or timeout: {e}")
        return ""

# ==========================================
# ⚙️ Task Scoring & Verification
# ==========================================

def score_math(output: str, target: float) -> float:
    # Look for standard #### <value> pattern
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", output)
    if not match:
        # Fallback: look for the last number in the text
        numbers = re.findall(r"-?\d+(?:\.\d+)?", output)
        if not numbers:
            return 0.0
        val = float(numbers[-1])
    else:
        val = float(match.group(1))
        
    return 1.0 if abs(val - target) < 1e-5 else 0.0

def score_pii(output: str, ground_truth: dict) -> float:
    # Extract JSON object from output
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        return 0.0
    try:
        predicted = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0

    # Ensure predicted is a dict
    if not isinstance(predicted, dict):
        return 0.0

    tp = 0
    fp = 0
    fn = 0

    # Clean predicted values
    pred_clean = {}
    for k, v in predicted.items():
        if v is not None and str(v).strip() != "" and str(v).lower() != "none":
            pred_clean[str(k).upper()] = str(v).strip().lower()

    # Clean ground truth values
    gt_clean = {}
    for k, v in ground_truth.items():
        if v is not None and str(v).strip() != "" and str(v).lower() != "none":
            gt_clean[str(k).upper()] = str(v).strip().lower()

    # Compute TP, FP, FN
    for k, v in gt_clean.items():
        if k in pred_clean:
            if pred_clean[k] == v:
                tp += 1
            else:
                fp += 1
                fn += 1
        else:
            fn += 1

    for k in pred_clean:
        if k not in gt_clean:
            fp += 1

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def score_financial(output: str, ground_truth: dict) -> float:
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        return 0.0
    try:
        predicted = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0

    if not isinstance(predicted, dict):
        return 0.0

    tp = 0
    fp = 0
    fn = 0

    # Normalize keys and list values
    pred_clean = {}
    for k, v in predicted.items():
        k_norm = str(k).capitalize()
        if isinstance(v, list):
            pred_clean[k_norm] = [str(x).strip().lower() for x in v if x is not None]
        elif isinstance(v, str) and v.strip() != "" and v.lower() != "none":
            pred_clean[k_norm] = [v.strip().lower()]

    gt_clean = {}
    for k, v in ground_truth.items():
        k_norm = str(k).capitalize()
        if isinstance(v, list):
            gt_clean[k_norm] = [str(x).strip().lower() for x in v if x is not None]
        elif isinstance(v, str) and v.strip() != "" and v.lower() != "none":
            gt_clean[k_norm] = [v.strip().lower()]

    # Compare lists per category
    all_keys = set(pred_clean.keys()).union(set(gt_clean.keys()))
    for k in all_keys:
        pred_list = pred_clean.get(k, [])
        gt_list = gt_clean.get(k, [])

        pred_set = set(pred_list)
        gt_set = set(gt_list)

        matched = pred_set.intersection(gt_set)
        tp += len(matched)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# ==========================================
# 📂 Dataset Loader
# ==========================================

def load_dataset(task: str, limit: int):
    data_dir = "data"
    dataset = []

    if task == "gsm8k":
        path = os.path.join(data_dir, "gsm8k_test.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                # Parse correct answer from the end
                match = re.search(r"####\s*(-?\d+)", item["answer"])
                target = int(match.group(1)) if match else 0
                dataset.append({
                    "input": item["question"],
                    "target": target
                })
                if len(dataset) >= limit:
                    break

    elif task == "svamp":
        path = os.path.join(data_dir, "svamp.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items[:limit]:
                dataset.append({
                    "input": f"{item['Body']} {item['Question']}",
                    "target": float(item["Answer"])
                })

    elif task == "pii":
        path = os.path.join(data_dir, "pii_extraction_train.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                # Parse ground truth using literal_eval safely
                target_dict = ast.literal_eval(item["ground_truth"])
                dataset.append({
                    "input": item["text"],
                    "target": target_dict
                })
                if len(dataset) >= limit:
                    break

    elif task == "financial":
        path = os.path.join(data_dir, "financial_ner_train.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                # Parse ground truth using literal_eval safely
                target_dict = ast.literal_eval(item["ground_truth"])
                dataset.append({
                    "input": item["text"],
                    "target": target_dict
                })
                if len(dataset) >= limit:
                    break
    else:
        raise ValueError(f"Unknown task: {task}")

    return dataset

# ==========================================
# 🚀 Run Evaluation
# ==========================================

def run_evaluation(model: str, task: str, limit: int, output_path: str = None):
    print(f"\n🚀 Evaluating model: '{model}' on task: '{task}' (Limit: {limit})")
    
    try:
        dataset = load_dataset(task, limit)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Prompt Templates
    if task in ["gsm8k", "svamp"]:
        prompt_template = (
            "Solve the following math word problem step by step. Write your final numerical answer at the end "
            "preceded by '#### ' (e.g. '#### 42').\n\n"
            "Problem: {input}\n\n"
            "Solution:"
        )
        score_fn = lambda out, tgt: score_math(out, tgt)
    elif task == "pii":
        prompt_template = (
            "You are a precise data extraction agent. Extract any personally identifiable information (PII) from the text below into a JSON object.\n"
            "The keys of the JSON object must be the PII categories present in the text, and the values must be the extracted strings.\n"
            "Only extract categories that actually appear in the text. Do not output any key with a null or empty value.\n\n"
            "Supported categories include:\n"
            "ACCOUNTNAME, ACCOUNTNUMBER, AGE, AMOUNT, BIC, BITCOINADDRESS, BUILDINGNUMBER, CITY, COMPANYNAME, COUNTY, CREDITCARDCVV, CREDITCARDISSUER, CREDITCARDNUMBER, CURRENCY, CURRENCYCODE, CURRENCYNAME, CURRENCYSYMBOL, DATE, DOB, EMAIL, ETHEREUMADDRESS, EYECOLOR, FIRSTNAME, GENDER, HEIGHT, IBAN, IP, IPV4, IPV6, JOBAREA, JOBTITLE, JOBTYPE, LASTNAME, LITECOINADDRESS, MAC, MASKEDNUMBER, MIDDLENAME, NEARBYGPSCOORDINATE, ORDINALDIRECTION, PASSWORD, PHONEIMEI, PHONENUMBER, PIN, PREFIX, SECONDARYADDRESS, SEX, SSN, STATE, STREET, TIME, URL, USERAGENT, USERNAME, VEHICLEVIN, VEHICLEVRM, ZIPCODE\n\n"
            "Do not output any introductory or explanatory text. Output ONLY valid raw JSON.\n\n"
            "Text: {input}\n\n"
            "JSON:"
        )
        score_fn = lambda out, tgt: score_pii(out, tgt)
    elif task == "financial":
        prompt_template = (
            "You are a precise data extraction agent. Extract financial entities from the text below into a JSON object.\n"
            "The keys of the JSON object must be the entity categories present in the text, and the values must be lists of extracted strings.\n"
            "Only extract categories that actually appear in the text. Do not output any key with a null or empty list.\n\n"
            "Supported categories are:\n"
            "Company, Date, Location, Money, Person, Product, Quantity\n\n"
            "Do not output any introductory or explanatory text. Output ONLY valid raw JSON.\n\n"
            "Text: {input}\n\n"
            "JSON:"
        )
        score_fn = lambda out, tgt: score_financial(out, tgt)

    total_score = 0.0
    results_log = []

    for idx, item in enumerate(dataset):
        prompt = prompt_template.format(input=item["input"])
        output = query_ollama(model, prompt)
        score = score_fn(output, item["target"])
        total_score += score
        
        results_log.append({
            "index": idx + 1,
            "input": item["input"],
            "target": item["target"],
            "output": output.strip(),
            "score": score
        })
        
        print(f"\n--- [Case {idx+1}/{len(dataset)}] ---")
        print(f"Input: {item['input']}")
        print(f"Target: {item['target']}")
        print(f"Output: {output.strip()}")
        print(f"Score: {score:.4f}")

    accuracy = (total_score / len(dataset)) * 100
    metric_name = "Accuracy" if task in ["gsm8k", "svamp"] else "F1 Score"
    print(f"\n📊 Overall {metric_name}: {accuracy:.2f}% (Sum of scores: {total_score:.4f}/{len(dataset)})")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary = {
            "model": model,
            "task": task,
            "limit": limit,
            "metric": metric_name,
            "score_percentage": accuracy,
            "results": results_log
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate quantized models on structured reasoning tasks.")
    parser.add_argument("--model", type=str, required=True, help="Ollama model name (e.g. gemma2:2b)")
    parser.add_argument("--task", type=str, choices=["gsm8k", "svamp", "pii", "financial"], required=True, help="Evaluation task")
    parser.add_argument("--limit", type=int, default=50, help="Number of examples to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for results")
    args = parser.parse_args()
    
    run_evaluation(args.model, args.task, args.limit, args.output)
