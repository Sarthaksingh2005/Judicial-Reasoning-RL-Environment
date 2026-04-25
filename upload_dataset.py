"""
Upload JusticeEngine-01 legal cases dataset to HuggingFace Hub.
Run: python upload_dataset.py
Requires: pip install datasets huggingface_hub
"""
import json
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID  = "RishitaRamola42/indian-legal-cases"

def main():
    try:
        from datasets import Dataset
        from huggingface_hub import login
    except ImportError:
        print("Run: pip install datasets huggingface_hub")
        return

    if not HF_TOKEN:
        print("HF_TOKEN not found in .env")
        return

    login(token=HF_TOKEN)
    print(f"Logged in to HuggingFace Hub")

    with open("data/cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Flatten list fields for HF Dataset compatibility
    rows = []
    for c in cases:
        rows.append({
            "case_id":               c["case_id"],
            "domain":                c["domain"],
            "difficulty":            c["difficulty"],
            "fact_pattern":          c["fact_pattern"],
            "applicable_statutes":   "\n".join(c.get("applicable_statutes", [])),
            "evidence_flags":        ", ".join(c.get("evidence_flags", [])),
            "gold_label_verdict":    c.get("gold_label_verdict", ""),
            "gold_label_reasoning":  c.get("gold_label_reasoning", ""),
            "num_precedents":        len(c.get("precedents", [])),
        })

    ds = Dataset.from_list(rows)
    print(f"Uploading {len(ds)} cases to {REPO_ID}...")
    ds.push_to_hub(REPO_ID, token=HF_TOKEN, private=False)
    print(f"Done! View at: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()
