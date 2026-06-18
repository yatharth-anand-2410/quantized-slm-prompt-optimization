import argparse
import json
import os
import re
import random
from evaluate import load_dataset, query_ollama, score_math, score_pii, score_financial

# ==========================================
# 🧬 Genetic Prompt Optimizer
# ==========================================

# Baseline prompts for tasks
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

# Mutation instructions for the meta-model
MUTATION_PROMPTS = [
    "Rephrase the instruction to make it clearer and more direct for a small, quantized model.",
    "Add explicit guidelines or step-by-step constraints to ensure accuracy and strict format adherence.",
    "Introduce few-shot examples or inline instructions to guide formatting (e.g., how to format JSON or mark math answers).",
    "Simplify the language, avoiding complex nested constraints that might confuse a quantized model.",
    "Emphasize negative guardrails (e.g. what NOT to output, avoiding extra chat or null fields)."
]

def mutate_prompt(meta_model: str, task: str, current_prompt: str, score: float, history: list) -> str:
    mutation_instruction = random.choice(MUTATION_PROMPTS)
    
    prompt = f"""You are an expert prompt engineer. Your goal is to improve a system prompt for a language model performing the task: '{task}'.
The target model is a small quantized model, which has limitations in instruction following.

Here is the current prompt template (with '{{input}}' as the placeholder for the input text):
---
{current_prompt}
---

The model's current score on this task using this prompt is {score:.2f}%.
Your modification goal is: {mutation_instruction}

Generate a new, improved prompt template.
Your response MUST contain ONLY the new prompt template text and nothing else. Do not wrap it in quotes, codeblocks, or explanatory text.
Ensure that '{{input}}' is included in your new prompt template as the placeholder for the input data.

New Prompt Template:"""

    response = query_ollama(meta_model, prompt, temperature=0.7)
    # Basic cleanup
    response_clean = response.strip()
    if response_clean.startswith("```"):
        # Strip markdown block if model ignored instructions
        response_clean = re.sub(r"^```[a-zA-Z]*\n", "", response_clean)
        response_clean = re.sub(r"\n```$", "", response_clean)
    
    # Ensure placeholder is preserved
    if "{input}" not in response_clean:
        print("⚠️ Warning: Evolved prompt was missing '{input}'. Reverting to parent prompt.")
        return current_prompt
        
    return response_clean.strip()

def evaluate_prompt(model: str, task: str, prompt_template: str, dataset: list) -> float:
    # Set scoring function
    if task in ["gsm8k", "svamp"]:
        score_fn = score_math
    elif task == "pii":
        score_fn = score_pii
    elif task == "financial":
        score_fn = score_financial
    else:
        return 0.0

    total_score = 0.0
    for item in dataset:
        prompt = prompt_template.format(input=item["input"])
        output = query_ollama(model, prompt)
        total_score += score_fn(output, item["target"])
        
    return (total_score / len(dataset)) * 100

def run_optimization(target_model: str, meta_model: str, task: str, limit: int, generations: int, pop_size: int, output_path: str = None):
    print(f"\n🧬 Starting Prompt Optimization Loop")
    print(f"Target Model: {target_model} | Meta-Model: {meta_model} | Task: {task}")
    print(f"Generations: {generations} | Population Size: {pop_size} | Evaluation Limit: {limit}")

    # 1. Load data and create train/validation split
    try:
        full_dataset = load_dataset(task, limit)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Split: 40% train/val (for optimization feedback), 60% test (for final test evaluation)
    split_idx = max(5, int(len(full_dataset) * 0.4))
    val_dataset = full_dataset[:split_idx]
    test_dataset = full_dataset[split_idx:]
    
    print(f"Splits created: Validation set size = {len(val_dataset)}, Test set size = {len(test_dataset)}")

    # 2. Evaluate Baseline Prompt
    baseline_prompt = BASELINE_PROMPTS[task]
    print("\n📈 Evaluating baseline prompt on validation set...")
    baseline_val_score = evaluate_prompt(target_model, task, baseline_prompt, val_dataset)
    print(f"Baseline Validation Score: {baseline_val_score:.2f}%")

    current_best_prompt = baseline_prompt
    current_best_score = baseline_val_score
    history = [{"generation": 0, "prompt": baseline_prompt, "score": baseline_val_score}]

    # 3. Evolution Loop
    for gen in range(1, generations + 1):
        print(f"\n--- 🧬 Generation {gen}/{generations} ---")
        candidates = []
        
        # Keep the best prompt from last generation (elitism)
        candidates.append((current_best_prompt, current_best_score))
        
        # Mutate to generate population
        for pop_idx in range(pop_size - 1):
            print(f"Mutating candidate {pop_idx+1}...")
            mutated = mutate_prompt(meta_model, task, current_best_prompt, current_best_score, history)
            print(f"Candidate {pop_idx+1} evolved. Evaluating...")
            score = evaluate_prompt(target_model, task, mutated, val_dataset)
            print(f"Score: {score:.2f}%")
            candidates.append((mutated, score))
            
        # Find the best candidate of this generation
        candidates.sort(key=lambda x: x[1], reverse=True)
        gen_best_prompt, gen_best_score = candidates[0]
        
        print(f"\nGeneration {gen} Best Score: {gen_best_score:.2f}% (Parent Best was: {current_best_score:.2f}%)")
        
        if gen_best_score > current_best_score:
            print(f"✨ Found new best prompt!")
            current_best_prompt = gen_best_prompt
            current_best_score = gen_best_score
            
        history.append({
            "generation": gen,
            "prompt": current_best_prompt,
            "score": current_best_score
        })

    # 4. Final Evaluation on Test Set
    print("\n🏁 Final Evaluation on Test Split...")
    print("Evaluating Baseline Prompt on Test Split...")
    baseline_test_score = evaluate_prompt(target_model, task, baseline_prompt, test_dataset)
    print("Evaluating Evolved Prompt on Test Split...")
    evolved_test_score = evaluate_prompt(target_model, task, current_best_prompt, test_dataset)

    print("\n📊 === Optimization Summary ===")
    print(f"Task: {task}")
    print(f"Baseline Test Score: {baseline_test_score:.2f}%")
    print(f"Evolved Test Score:  {evolved_test_score:.2f}%")
    print(f"Absolute Recovery:   {evolved_test_score - baseline_test_score:+.2f}%")
    print("Best Evolved Prompt:")
    print("----------------------------------------")
    print(current_best_prompt)
    print("----------------------------------------")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary = {
            "task": task,
            "target_model": target_model,
            "meta_model": meta_model,
            "baseline_prompt": baseline_prompt,
            "evolved_prompt": current_best_prompt,
            "baseline_test_score": baseline_test_score,
            "evolved_test_score": evolved_test_score,
            "improvement": evolved_test_score - baseline_test_score,
            "history": history
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Evolved prompt and run log saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize discrete prompt templates using genetic mutation.")
    parser.add_argument("--target-model", type=str, required=True, help="Ollama model to optimize prompts for")
    parser.add_argument("--meta-model", type=str, required=True, help="Ollama model to generate mutations")
    parser.add_argument("--task", type=str, choices=["gsm8k", "svamp", "pii", "financial"], required=True, help="Task to optimize")
    parser.add_argument("--limit", type=int, default=50, help="Total dataset subset size (to split into val/test)")
    parser.add_argument("--generations", type=int, default=3, help="Number of evolutionary generations")
    parser.add_argument("--pop-size", type=int, default=4, help="Population size per generation")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for results")
    args = parser.parse_args()
    
    run_optimization(
        args.target_model,
        args.meta_model,
        args.task,
        args.limit,
        args.generations,
        args.pop_size,
        args.output
    )
