"""
TRL GRPO Training Script for JusticeEngine-01
Team ALACRITY | OpenEnv Hackathon | Scaler × Meta | April 2026

Run on Google Colab (T4 or A100):
  !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  !pip install --no-deps xformers trl peft accelerate bitsandbytes
  !pip install wandb datasets huggingface_hub

RLVR Reward Components:
  - format_reward:   +0.5 if all 5 XML tags present (<action>, <verdict>, <confidence_score>, <reasoning_chain>, <ratio_decidendi>, <obiter_dicta>)
  - accuracy_reward: +0.0–1.0 from JudicialEnv._accuracy_score() vs expert gold label
  - logic_reward:    +0.0–1.0 from BNS/Constitution keyword density and reasoning length
All scores clamped to (0.001, 0.999) per hackathon spec.
"""

import os
import json
import re

# ─── Graceful import for local testing ─────────────────
try:
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
    TRAINING_AVAILABLE = True
except ImportError:
    print("⚠️  GPU/TRL libraries not found. Run on Colab for full training.")
    TRAINING_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Run: pip install wandb")

# ─── Local imports ──────────────────────────────────────
from environment import JudicialEnv, JudicialAction
from graders.programmatic_grader import ProgrammaticGrader

# ==========================================
# 1. Configuration
# ==========================================
MODEL_NAME        = "unsloth/Meta-Llama-3-8B-Instruct"
MAX_SEQ_LENGTH    = 4096
LORA_RANK         = 16
MAX_STEPS         = 250
HF_REPO_ID        = "RishitaRamola42/justice-engine-01-lora"   # HF Hub destination
DATASET_REPO_ID   = "RishitaRamola42/indian-legal-cases"        # HF Dataset destination
WANDB_PROJECT     = "justice-engine-01"
WANDB_RUN_NAME    = "grpo-bns-v2"

SYSTEM_PROMPT = """You are JusticeEngine-01, an AI legal mediator for Indian courts.
You must strictly follow the Constitution of India and the Bharatiya Nyaya Sanhita (BNS) 2023.
Always follow court hierarchy: Supreme Court > High Court > Sessions Court > Magistrate.

Respond ONLY in valid XML format:
<action>
  <verdict>liable OR not_liable OR partial_liability OR forward_to_judge</verdict>
  <confidence_score>0.0 to 1.0</confidence_score>
  <reasoning_chain>Your step-by-step reasoning citing BNS sections and precedents</reasoning_chain>
  <ratio_decidendi>The single binding legal principle of this decision</ratio_decidendi>
  <obiter_dicta>Non-binding judicial observations made in passing</obiter_dicta>
</action>"""

# ==========================================
# 2. Reward Functions (RLVR)
# ==========================================

def extract_xml_action(completion: str) -> dict:
    """Helper to extract XML fields from LLM completion."""
    try:
        verdict    = re.search(r'<verdict>(.*?)</verdict>', completion, re.DOTALL)
        confidence = re.search(r'<confidence_score>(.*?)</confidence_score>', completion, re.DOTALL)
        reasoning  = re.search(r'<reasoning_chain>(.*?)</reasoning_chain>', completion, re.DOTALL)
        ratio      = re.search(r'<ratio_decidendi>(.*?)</ratio_decidendi>', completion, re.DOTALL)
        obiter     = re.search(r'<obiter_dicta>(.*?)</obiter_dicta>', completion, re.DOTALL)
        return {
            "verdict":          verdict.group(1).strip() if verdict else "invalid",
            "confidence_score": float(confidence.group(1).strip()) if confidence else 0.0,
            "reasoning_chain":  reasoning.group(1).strip() if reasoning else "",
            "ratio_decidendi":  ratio.group(1).strip() if ratio else "",
            "obiter_dicta":     obiter.group(1).strip() if obiter else "",
            "cited_precedents": []
        }
    except Exception:
        return None


def format_reward(prompts, completions, **kwargs):
    """
    Reward for adhering to the exact XML format with ALL 5 required tags.
    +0.5 if <action>, <verdict>, <confidence_score>, <reasoning_chain>, <ratio_decidendi> all present.
    +0.25 bonus if <obiter_dicta> also present.
    0.0 if missing required tags.
    """
    rewards = []
    required_tags = ["<action>", "</action>", "<verdict>", "<confidence_score>", "<reasoning_chain>", "<ratio_decidendi>"]
    for comp in completions:
        comp_str = comp[0]["content"] if isinstance(comp, list) else comp
        if all(tag in comp_str for tag in required_tags):
            score = 0.5
            if "<obiter_dicta>" in comp_str:
                score += 0.25
            rewards.append(min(score, 0.999))
        else:
            rewards.append(0.001)
    return rewards


def accuracy_reward(prompts, completions, **kwargs):
    """
    Reward for giving the correct legal verdict evaluated against expert gold labels.
    Uses JudicialEnv step() and returns the accuracy_score from the environment.
    Clamped to (0.001, 0.999) per hackathon spec.
    """
    rewards = []
    for prompt, comp in zip(prompts, completions):
        comp_str = comp[0]["content"] if isinstance(comp, list) else comp
        action_dict = extract_xml_action(comp_str)

        if not action_dict or action_dict["verdict"] == "invalid":
            rewards.append(0.001)
            continue

        try:
            action = JudicialAction(**action_dict)
            env = JudicialEnv(domain="contract", difficulty="easy")
            env.reset()
            obs, reward, done, trunc, info = env.step(action)
            score = float(info.get('accuracy_score', 0.0))
            rewards.append(max(0.001, min(score, 0.999)))
        except Exception:
            rewards.append(0.001)
    return rewards


def logic_reward(prompts, completions, **kwargs):
    """
    Reward for logical legal reasoning — BNS keywords, Constitution references,
    court hierarchy language, and minimum reasoning depth.
    """
    BNS_KEYWORDS = [
        "constitution", "bns", "sanhita", "bnss", "bharatiya",
        "section", "supreme court", "high court", "precedent",
        "liable", "burden of proof", "ratio", "obiter",
        "contract act", "cognizable", "fir", "forward_to_judge"
    ]
    rewards = []
    for comp in completions:
        comp_str = comp[0]["content"] if isinstance(comp, list) else comp
        action = extract_xml_action(comp_str)
        if not action:
            rewards.append(0.001)
            continue

        text = action["reasoning_chain"].lower()
        score = 0.0
        # Keyword density
        hits = sum(1 for kw in BNS_KEYWORDS if kw in text)
        score += min(hits / len(BNS_KEYWORDS), 1.0) * 0.5
        # Reasoning depth
        if len(text) > 100: score += 0.2
        if len(text) > 300: score += 0.2
        # ratio_decidendi present
        if action.get("ratio_decidendi"): score += 0.1
        rewards.append(max(0.001, min(score, 0.999)))
    return rewards


# ==========================================
# 3. Dataset Preparation + HF Upload
# ==========================================
def load_and_upload_dataset(push_to_hub: bool = True):
    data_path = os.path.join("data", "cases.json")
    with open(data_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    dataset_rows = []
    for c in cases:
        prompt_text = (
            f"FACT PATTERN:\n{c['fact_pattern']}\n\n"
            f"APPLICABLE STATUTES:\n{chr(10).join(c.get('applicable_statutes', []))}\n\n"
            f"PRECEDENTS:\n{json.dumps(c.get('precedents', []), indent=2)}\n\n"
            f"EVIDENCE FLAGS: {', '.join(c.get('evidence_flags', [])) or 'None'}"
        )
        dataset_rows.append({
            "case_id":    c["case_id"],
            "domain":     c["domain"],
            "difficulty": c["difficulty"],
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt_text}
            ],
            "gold_label_verdict":   c.get("gold_label_verdict", ""),
            "gold_label_reasoning": c.get("gold_label_reasoning", ""),
        })

    ds = Dataset.from_list(dataset_rows)

    if push_to_hub:
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            print(f"📤 Uploading dataset to HF Hub: {DATASET_REPO_ID}")
            ds.push_to_hub(DATASET_REPO_ID, token=hf_token, private=False)
            print("✅ Dataset uploaded successfully.")
        else:
            print("⚠️  HF_TOKEN not set. Skipping dataset upload.")
    return ds


# ==========================================
# 4. Training Loop
# ==========================================
def main():
    # ─── Init Wandb ─────────────────────────────────────
    if WANDB_AVAILABLE:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "model": MODEL_NAME,
                "lora_rank": LORA_RANK,
                "max_steps": MAX_STEPS,
                "reward_components": ["format", "accuracy", "logic"],
                "reward_clamping": "(0.001, 0.999)",
                "legal_framework": "BNS 2023 / BNSS 2023 / BSA 2023",
            }
        )
        print(f"📊 Wandb run started: {wandb.run.get_url()}")

    if not TRAINING_AVAILABLE:
        print("❌ Training libraries not found. Please run on a Colab GPU instance.")
        return

    print("⚖️  JusticeEngine-01 GRPO Training Starting...")

    # ─── Load Model ─────────────────────────────────────
    print("🔄 Loading base model via Unsloth (4-bit quantized)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name            = MODEL_NAME,
        max_seq_length        = MAX_SEQ_LENGTH,
        load_in_4bit          = True,
        fast_inference        = True,
        max_lora_rank         = LORA_RANK,
        gpu_memory_utilization = 0.6,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                       = LORA_RANK,
        target_modules          = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha              = LORA_RANK,
        use_gradient_checkpointing = "unsloth",
        random_state            = 3407,
    )

    # ─── Load Dataset ────────────────────────────────────
    print("📂 Loading and preparing legal cases dataset...")
    dataset = load_and_upload_dataset(push_to_hub=True)
    print(f"✅ Dataset ready: {len(dataset)} cases")

    # ─── GRPO Trainer ────────────────────────────────────
    print("⚙️  Configuring GRPO Trainer...")
    training_args = GRPOConfig(
        use_vllm                    = True,
        learning_rate               = 5e-6,
        adam_beta1                  = 0.9,
        adam_beta2                  = 0.99,
        weight_decay                = 0.1,
        warmup_ratio                = 0.1,
        lr_scheduler_type           = "cosine",
        logging_steps               = 1,
        bf16                        = True,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        num_generations             = 4,
        max_prompt_length           = 512,
        max_completion_length       = 512,
        max_steps                   = MAX_STEPS,
        save_steps                  = 50,
        output_dir                  = "outputs",
        report_to                   = "wandb" if WANDB_AVAILABLE else "none",
        run_name                    = WANDB_RUN_NAME,
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = [format_reward, accuracy_reward, logic_reward],
        args             = training_args,
        train_dataset    = dataset,
    )

    # ─── Train ──────────────────────────────────────────
    print("🚀 Starting RL Training...")
    trainer.train()

    # ─── Save & Push to HF Hub ──────────────────────────
    print("💾 Saving LoRA adapter...")
    model.save_lora("outputs/justice_engine_lora")

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        print(f"📤 Uploading LoRA adapter to HF Hub: {HF_REPO_ID}")
        model.push_to_hub_merged(
            HF_REPO_ID,
            tokenizer,
            save_method="lora",
            token=hf_token,
        )
        print(f"✅ Model uploaded: https://huggingface.co/{HF_REPO_ID}")
    else:
        print("⚠️  HF_TOKEN not set. Skipping HF Hub upload.")

    if WANDB_AVAILABLE:
        wandb.finish()

    print("🏆 Training Complete! JusticeEngine-01 LoRA adapter saved.")


if __name__ == "__main__":
    main()
