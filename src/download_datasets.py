import os
import json
import urllib.request
from datasets import load_dataset

def main():
    os.makedirs("data", exist_ok=True)
    
    # 1. Download GSM8K
    print("Downloading GSM8K test set...")
    gsm8k_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    gsm8k_path = "data/gsm8k_test.jsonl"
    urllib.request.urlretrieve(gsm8k_url, gsm8k_path)
    print(f"Saved GSM8K to {gsm8k_path}")
    
    # 2. Download SVAMP
    print("Downloading SVAMP...")
    svamp_url = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"
    svamp_path = "data/svamp.json"
    urllib.request.urlretrieve(svamp_url, svamp_path)
    print(f"Saved SVAMP to {svamp_path}")
    
    # 3. Download Cleanlab pii-extraction
    print("Downloading Cleanlab/pii-extraction...")
    try:
        pii_dataset = load_dataset("Cleanlab/pii-extraction")
        print(f"pii-extraction splits: {list(pii_dataset.keys())}")
        for split in pii_dataset.keys():
            split_path = f"data/pii_extraction_{split}.json"
            pii_dataset[split].to_json(split_path, orient="records", force_ascii=False)
            print(f"Saved pii_extraction_{split} to {split_path}")
    except Exception as e:
        print(f"Error downloading Cleanlab/pii-extraction: {e}")
        
    # 4. Download Cleanlab fire-financial-ner-extraction
    print("Downloading Cleanlab/fire-financial-ner-extraction...")
    try:
        financial_dataset = load_dataset("Cleanlab/fire-financial-ner-extraction")
        print(f"fire-financial-ner-extraction splits: {list(financial_dataset.keys())}")
        for split in financial_dataset.keys():
            split_path = f"data/financial_ner_{split}.json"
            financial_dataset[split].to_json(split_path, orient="records", force_ascii=False)
            print(f"Saved financial_ner_{split} to {split_path}")
    except Exception as e:
        print(f"Error downloading Cleanlab/fire-financial-ner-extraction: {e}")

if __name__ == "__main__":
    main()
