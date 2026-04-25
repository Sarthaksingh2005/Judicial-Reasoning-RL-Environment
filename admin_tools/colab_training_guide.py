"""
JusticeEngine-01 — Colab Training Notebook
============================================
Run this file cell-by-cell in Google Colab (T4 GPU).
Each # ── CELL N ── block is one Colab code cell.
"""

# ── CELL 1: Set Runtime ──────────────────────────────────────────
# Before running anything:
# Click Runtime → Change runtime type → Select "T4 GPU" → Save
# Then run this cell to confirm GPU is active:
"""
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
"""

# ── CELL 2: Install Unsloth (run this FIRST, takes ~2 min) ───────
"""
%%capture
# Correct installation order — avoids mergekit/llm-blender errors
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --upgrade trl transformers accelerate peft datasets bitsandbytes
"""

# ── CELL 3: Restart Runtime (REQUIRED after Cell 2) ──────────────
# After Cell 2 finishes, Colab will show a "Restart session" popup.
# Click "Restart session" to reload the updated packages.
# Then continue from Cell 4. Do NOT re-run Cells 1-2 again.

# ── CELL 4: Clone Your Project ───────────────────────────────────
"""
import os

# Clone repo (skip if already cloned)
if not os.path.exists("/content/judicial-reasoning-env"):
    !git clone https://github.com/rishitaramola/judicial-reasoning-env.git

# Move into project directory
%cd /content/judicial-reasoning-env
!ls   # You should see: server/, data/, environment.py, admin_tools/, etc.
"""

# ── CELL 5: Verify GPU + Imports ─────────────────────────────────
"""
import torch
import sys
sys.path.insert(0, "/content/judicial-reasoning-env")

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from environment import JudicialEnv, JudicialAction

# PatchFastRL intentionally removed — causes OSError on Colab (not needed)

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("✅ All imports successful — ready to train!")
"""

# ── CELL 6: Load Model + LoRA ─────────────────────────────────────
"""
MODEL_NAME     = "unsloth/llama-3-8b-bnb-4bit"   # Pre-quantized, works on T4
MAX_SEQ_LENGTH = 4096
LORA_RANK      = 16

print("Loading model... (takes 1-2 min on first run)")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name            = MODEL_NAME,
    max_seq_length        = MAX_SEQ_LENGTH,
    load_in_4bit          = True,
    fast_inference        = False,   # vLLM not needed for training
    max_lora_rank         = LORA_RANK,
    gpu_memory_utilization = 0.6,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                      = LORA_RANK,
    target_modules         = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
    lora_alpha             = LORA_RANK,
    use_gradient_checkpointing = "unsloth",
    random_state           = 3407,
)

print(f"✅ Model loaded: {MODEL_NAME}")
print(f"   LoRA rank: {LORA_RANK} | 4-bit quantized | Gradient checkpointing: ON")
"""

# ── CELL 7: Define Reward Functions ───────────────────────────────
"""
import re, json
from environment import JudicialEnv, JudicialAction

SYSTEM_PROMPT = \"""You are JusticeEngine-01, an AI legal mediator for Indian courts.
Strictly follow the Constitution of India and the Bharatiya Nyaya Sanhita (BNS).
NEVER invent monetary amounts. Use only figures from the case facts.
Respond ONLY in valid XML format:
<action>
  <verdict>liable OR not_liable OR guilty OR not_guilty OR forward_to_judge</verdict>
  <confidence_score>0.9</confidence_score>
  <reasoning_chain>Your step-by-step reasoning citing specific statutes</reasoning_chain>
</action>\"""

def extract_xml_action(completion):
    try:
        verdict    = re.search(r'<verdict>(.*?)</verdict>', completion, re.DOTALL)
        confidence = re.search(r'<confidence_score>(.*?)</confidence_score>', completion, re.DOTALL)
        reasoning  = re.search(r'<reasoning_chain>(.*?)</reasoning_chain>', completion, re.DOTALL)
        return {
            "verdict":           verdict.group(1).strip()           if verdict    else "invalid",
            "confidence_score":  float(confidence.group(1).strip()) if confidence else 0.0,
            "reasoning_chain":   reasoning.group(1).strip()         if reasoning  else "",
            "cited_precedents":  []
        }
    except Exception:
        return None

def format_reward(prompts, completions, **kwargs):
    rewards = []
    for comp in completions:
        s = comp[0]["content"] if isinstance(comp, list) else comp
        rewards.append(0.5 if "<action>" in s and "</action>" in s else 0.0)
    return rewards

def accuracy_reward(prompts, completions, **kwargs):
    rewards = []
    for comp in completions:
        s = comp[0]["content"] if isinstance(comp, list) else comp
        action_dict = extract_xml_action(s)
        if not action_dict or action_dict["verdict"] == "invalid":
            rewards.append(-1.0)
            continue
        try:
            action = JudicialAction(**action_dict)
            env = JudicialEnv(domain="contract", difficulty="easy")
            env.reset()
            _, reward, _, _, info = env.step(action)
            rewards.append(float(info.get("accuracy_score", 0.0)))
        except Exception:
            rewards.append(-0.5)
    return rewards

def logic_reward(prompts, completions, **kwargs):
    rewards = []
    for comp in completions:
        s = comp[0]["content"] if isinstance(comp, list) else comp
        action = extract_xml_action(s)
        if action and "reasoning_chain" in action:
            text  = action["reasoning_chain"].lower()
            score = 0.0
            if "constitution" in text:                  score += 0.2
            if "bns" in text or "sanhita" in text:     score += 0.3
            if "specific relief" in text:               score += 0.2
            if "limitation act" in text:                score += 0.1
            if len(text) > 100:                         score += 0.2
            rewards.append(min(score, 1.0))
        else:
            rewards.append(0.0)
    return rewards

print("✅ Reward functions defined (format + accuracy + logic)")
"""

# ── CELL 8: Load Dataset ──────────────────────────────────────────
"""
import json, os
from datasets import Dataset

with open("data/cases.json", "r") as f:
    cases = json.load(f)

dataset_list = []
for c in cases:
    prompt_text = (
        f"FACT PATTERN:\\n{c['fact_pattern']}\\n\\n"
        f"STATUTES:\\n{chr(10).join(c.get('applicable_statutes', []))}"
    )
    dataset_list.append({
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text}
        ]
    })

dataset = Dataset.from_list(dataset_list)
print(f"✅ Dataset loaded: {len(dataset)} cases")
"""

# ── CELL 9: Configure & Start Training ────────────────────────────
"""
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    use_vllm                    = False,  # vLLM not installed on free Colab
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
    max_steps                   = 250,
    save_steps                  = 250,
    output_dir                  = "outputs",
)

trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = [format_reward, accuracy_reward, logic_reward],
    args             = training_args,
    train_dataset    = dataset,
)

print("🚀 Starting RL Training (250 steps — approx 15-20 min on T4)...")
trainer.train()
print("✅ Training complete!")
"""

# ── CELL 10: Save LoRA Weights ────────────────────────────────────
"""
model.save_lora("outputs/justice_engine_lora")
print("✅ LoRA adapter saved to: outputs/justice_engine_lora")
print("   To use these weights, load with FastLanguageModel.from_pretrained")
print("   and pass adapter_name='outputs/justice_engine_lora'")
"""
