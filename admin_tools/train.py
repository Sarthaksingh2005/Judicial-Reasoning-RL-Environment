"""
TRL GRPO Training Script for JusticeEngine-01

Prerequisites (Run on a GPU instance like Google Colab):
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
"""
"""

import os
import json
import re

try:
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
except ImportError:
    print("Warning: GPU/TRL libraries not found. Run on Colab for full training.")

# Local imports
from environment import JudicialEnv, JudicialAction
from graders.programmatic_grader import ProgrammaticGrader


MODEL_NAME = "unsloth/Meta-Llama-3-8B-Instruct"
MAX_SEQ_LENGTH = 4096
LORA_RANK = 16

SYSTEM_PROMPT = """You are JusticeEngine-01, an AI legal mediator for Indian courts.
You must strictly follow the Constitution of India and the Bharatiya Nyaya Sanhita (BNS).
Respond ONLY in valid XML format:
<action>
  <verdict>liable OR not_liable OR guilty OR not_guilty OR forward_to_judge</verdict>
  <confidence_score>0.9</confidence_score>
  <reasoning_chain>Your step-by-step reasoning</reasoning_chain>
</action>"""


def extract_xml_action(completion: str) -> dict:
    """Helper to extract XML fields from LLM completion."""
    try:
        verdict = re.search(r'<verdict>(.*?)</verdict>', completion, re.DOTALL)
        confidence = re.search(r'<confidence_score>(.*?)</confidence_score>', completion, re.DOTALL)
        reasoning = re.search(r'<reasoning_chain>(.*?)</reasoning_chain>', completion, re.DOTALL)
        
        return {
            "verdict": verdict.group(1).strip() if verdict else "invalid",
            "confidence_score": float(confidence.group(1).strip()) if confidence else 0.0,
            "reasoning_chain": reasoning.group(1).strip() if reasoning else "",
            "cited_precedents": []
        }
    except Exception:
        return None

def format_reward(prompts, completions, **kwargs):
    """Reward for adhering to the exact XML format."""
    rewards = []
    for comp in completions:
        comp_str = comp[0]["content"] if isinstance(comp, list) else comp
        if "<action>" in comp_str and "</action>" in comp_str:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward(prompts, completions, **kwargs):
    """Reward for giving the correct legal verdict (via JudicialEnv)."""
    rewards = []
    
    # We need the case domain/difficulty. For simplicity in RL, 
    # we can mock or extract the case details from the prompt.
    for prompt, comp in zip(prompts, completions):
        comp_str = comp[0]["content"] if isinstance(comp, list) else comp
        action_dict = extract_xml_action(comp_str)
        
        if not action_dict or action_dict["verdict"] == "invalid":
            rewards.append(-1.0) # Penalize invalid formats
            continue
            
        try:
            action = JudicialAction(**action_dict)
            env = JudicialEnv(domain="contract", difficulty="easy")
            env.reset()
            obs, reward, done, trunc, info = env.step(action)
            
            rewards.append(float(info.get('accuracy_score', 0.0)))
        except Exception:
            rewards.append(-0.5)
            
    return rewards

def logic_reward(prompts, completions, **kwargs):
    """Reward for logical legal reasoning and citation of Indian law."""
    rewards = []
    for comp in completions:
        comp_str = comp[0]["content"] if isinstance(comp, list) else comp
        action = extract_xml_action(comp_str)
        
        if action and "reasoning_chain" in action:
            text = action["reasoning_chain"].lower()
            score = 0.0
            if "constitution" in text: score += 0.2
            if "bns" in text or "sanhita" in text: score += 0.3
            if len(text) > 100: score += 0.5
            rewards.append(score)
        else:
            rewards.append(0.0)
    return rewards


def load_dataset():
    data_path = os.path.join("data", "cases.json")
    with open(data_path, "r") as f:
        cases = json.load(f)
        
    dataset = []
    for c in cases:
        prompt_text = f"FACT PATTERN:\n{c['fact_pattern']}\n\nSTATUTES:\n{chr(10).join(c.get('applicable_statutes', []))}"
        dataset.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ]
        })
    return Dataset.from_list(dataset)


def main():
    print("Loading Model via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True, 
        fast_inference = True, 
        max_lora_rank = LORA_RANK,
        gpu_memory_utilization = 0.6,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = LORA_RANK,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    print("Loading Dataset...")
    dataset = load_dataset()

    print("Configuring GRPO Trainer...")
    training_args = GRPOConfig(
        use_vllm = True, 
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        num_generations = 4, 
        max_prompt_length = 512,
        max_completion_length = 512,
        max_steps = 250,
        save_steps = 250,
        output_dir = "outputs",
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            format_reward,
            accuracy_reward,
            logic_reward
        ],
        args = training_args,
        train_dataset = dataset,
    )

    print("Starting RL Training...")
    trainer.train()

    print("Saving Model...")
    model.save_lora("outputs/justice_engine_lora")
    print("Training Complete!")

if __name__ == "__main__":
    main()
