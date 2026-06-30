import os
import json
import argparse
import ast
from evaluate import load_dataset, query_ollama, score_math, score_pii, score_financial

BASELINE_PROMPTS = {
    "gsm8k": (
        "Solve the following math word problem step by step. Write your final numerical answer at the end "
        "preceded by '#### ' (e.g. '#### 42').\n\n"
        "Problem: {input}\n\n"
        "Solution:"
    ),
    "svamp": (
        "Solve the following math word problem step by step. Write your final numerical answer at the end "
        "preceded by '#### ' (e.g. '#### 42').\n\n"
        "Problem: {input}\n\n"
        "Solution:"
    ),
    "pii": (
        "You are a precise data extraction agent. Extract any personally identifiable information (PII) from the text below into a JSON object.\n"
        "The keys of the JSON object must be the PII categories present in the text, and the values must be the extracted strings.\n"
        "Only extract categories that actually appear in the text. Do not output any key with a null or empty value.\n\n"
        "Supported categories include:\n"
        "ACCOUNTNAME, ACCOUNTNUMBER, AGE, AMOUNT, BIC, BITCOINADDRESS, BUILDINGNUMBER, CITY, COMPANYNAME, COUNTY, CREDITCARDCVV, CREDITCARDISSUER, CREDITCARDNUMBER, CURRENCY, CURRENCYCODE, CURRENCYNAME, CURRENCYSYMBOL, DATE, DOB, EMAIL, ETHEREUMADDRESS, EYECOLOR, FIRSTNAME, GENDER, HEIGHT, IBAN, IP, IPV4, IPV6, JOBAREA, JOBTITLE, JOBTYPE, LASTNAME, LITECOINADDRESS, MAC, MASKEDNUMBER, MIDDLENAME, NEARBYGPSCOORDINATE, ORDINALDIRECTION, PASSWORD, PHONEIMEI, PHONENUMBER, PIN, PREFIX, SECONDARYADDRESS, SEX, SSN, STATE, STREET, TIME, URL, USERAGENT, USERNAME, VEHICLEVIN, VEHICLEVRM, ZIPCODE\n\n"
        "Do not output any introductory or explanatory text. Output ONLY valid raw JSON.\n\n"
        "Text: {input}\n\n"
        "JSON:"
    ),
    "financial": (
        "You are a precise data extraction agent. Extract financial entities from the text below into a JSON object.\n"
        "The keys of the JSON object must be the entity categories present in the text, and the values must be lists of extracted strings.\n"
        "Only extract categories that actually appear in the text. Do not output any key with a null or empty list.\n\n"
        "Supported categories are:\n"
        "Company, Date, Location, Money, Person, Product, Quantity\n\n"
        "Do not output any introductory or explanatory text. Output ONLY valid raw JSON.\n\n"
        "Text: {input}\n\n"
        "JSON:"
    )
}

def load_evolved_prompt(task: str, precision: str) -> str:
    """Load evolved prompt for llama3.2 from results."""
    if precision == "q4":
        path = f"results/optimized_llama3.2_{task}.json"
    elif precision == "fp16":
        path = f"results/optimized_llama3.2-fp16_{task}.json"
    else:
        raise ValueError(f"Unknown precision: {precision}")
    
    if not os.path.exists(path):
        print(f"⚠️ Warning: Evolved prompt file {path} not found. Using baseline prompt instead.")
        return BASELINE_PROMPTS[task]
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("evolved_prompt", BASELINE_PROMPTS[task])

def evaluate_prompt(model: str, task: str, prompt_template: str, dataset: list) -> float:
    if task in ["gsm8k", "svamp"]:
        score_fn = score_math
    elif task == "pii":
        score_fn = score_pii
    elif task == "financial":
        score_fn = score_financial
    else:
        raise ValueError(f"Unknown task: {task}")
        
    total_score = 0.0
    for idx, item in enumerate(dataset):
        prompt = prompt_template.replace("{input}", item["input"])
        output = query_ollama(model, prompt)
        score = score_fn(output, item["target"])
        total_score += score
        
    return (total_score / len(dataset)) * 100

def unload_model(model: str):
    import urllib.request
    import json
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": 0
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            response.read()
    except Exception as e:
        print(f"Note: failed to unload model {model}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run cross-precision transfer evaluations on test set.")
    parser.add_argument("--output", type=str, default="results/cross_precision_transfer.json", help="Path to save evaluation results")
    args = parser.parse_args()
    
    tasks = ["gsm8k", "svamp", "pii", "financial"]
    models = {
        "q4": "llama3.2:latest",
        "fp16": "llama3.2:3b-instruct-fp16"
    }
    
    results = {}
    prompts_cache = {}
    
    # Load existing results to support resuming
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                results = existing_data.get("results", {})
                prompts_cache = existing_data.get("prompts", {})
                print(f"🔄 Loaded existing results from {args.output}. Resuming...")
        except Exception as e:
            print(f"⚠️ Could not load existing results: {e}. Starting fresh.")
    
    for task in tasks:
        if task in results and len(results[task]) == len(models):
            print(f"Skipping Task {task} (already fully evaluated).")
            continue
            
        print(f"\n==========================================")
        print(f"📊 Evaluating Task: {task}")
        print(f"==========================================")
        
        # Load test dataset split (index 20 to 50)
        full_dataset = load_dataset(task, 50)
        test_dataset = full_dataset[20:50]
        print(f"Loaded test dataset split: {len(test_dataset)} examples (indices 20 to 50)")
        
        baseline_p = BASELINE_PROMPTS[task]
        evolved_q4_p = load_evolved_prompt(task, "q4")
        evolved_fp16_p = load_evolved_prompt(task, "fp16")
        
        prompts_cache[task] = {
            "baseline": baseline_p,
            "evolved_q4": evolved_q4_p,
            "evolved_fp16": evolved_fp16_p
        }
        
        results[task] = {}
        
        for model_precision, model_name in models.items():
            results[task][model_name] = {}
            print(f"\n🤖 Running evaluations on target model: {model_name} ({model_precision.upper()})")
            
            # Evaluate baseline prompt
            print("Evaluating baseline prompt...")
            baseline_score = evaluate_prompt(model_name, task, baseline_p, test_dataset)
            results[task][model_name]["baseline_prompt"] = baseline_score
            print(f"Baseline Score: {baseline_score:.2f}%")
            
            # Evaluate evolved Q4 prompt
            print("Evaluating evolved Q4 prompt...")
            q4_score = evaluate_prompt(model_name, task, evolved_q4_p, test_dataset)
            results[task][model_name]["evolved_q4_prompt"] = q4_score
            print(f"Evolved Q4 Score: {q4_score:.2f}%")
            
            # Evaluate evolved FP16 prompt
            print("Evaluating evolved FP16 prompt...")
            fp16_score = evaluate_prompt(model_name, task, evolved_fp16_p, test_dataset)
            results[task][model_name]["evolved_fp16_prompt"] = fp16_score
            print(f"Evolved FP16 Score: {fp16_score:.2f}%")
            # Unload model from VRAM to prevent VRAM swapping overhead/hangs
            print(f"🧹 Unloading model {model_name} from VRAM...")
            unload_model(model_name)
            
        # Save output after each task
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        output_data = {
            "results": results,
            "prompts": prompts_cache
        }
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
            
        print(f"💾 Checkpoint saved for task {task} in {args.output}")

if __name__ == "__main__":
    main()
