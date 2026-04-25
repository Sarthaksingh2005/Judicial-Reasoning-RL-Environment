---
title: Judicial Reasoning Env
emoji: ⚖️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# ⚖️ Judicial Reasoning RL Environment

> **OpenEnv Hackathon | Team ALACRITY | Scaler × Meta**

An RL environment where an LLM-based agent acts as a judge. Each episode, the agent receives a real Indian legal case file and must reason through it to deliver a structured verdict. Built for the OpenEnv Hackathon by Team ALACRITY.

---

## 🔗 Hackathon Submission Links

- **Hugging Face Space (Live Demo):** [RishitaRamola42/judicial-reasoning-env](https://huggingface.co/spaces/RishitaRamola42/judicial-reasoning-env)
- **Colab Training Notebook:** [JusticeEngine_01_GRPO_Training.ipynb](https://colab.research.google.com/github/Sarthaksingh2005/Judicial-Reasoning-RL-Environment/blob/master/training_notebook.ipynb)
- **Code Repository:** [Sarthaksingh2005/Judicial-Reasoning-RL-Environment](https://github.com/Sarthaksingh2005/Judicial-Reasoning-RL-Environment)
- **Hugging Face Blog Post:** [Read our Blog Post (hf_blog.md)](https://github.com/Sarthaksingh2005/Judicial-Reasoning-RL-Environment/blob/master/hf_blog.md)

---

## Why This Environment Exists

No public RL benchmark tests **legal reasoning**. This environment forces agents to demonstrate everything agentic AI needs:

- **Retrieval** — Apply the right statute from a curated list
- **Reasoning** — Build a logical chain from facts to verdict
- **Fairness** — Produce consistent verdicts across similar cases
- **Citation** — Reference only real precedents (anti-hallucination)
- **Justification** — Explain decisions in natural language

Legal AI directly impacts real-world fairness. Getting it wrong has consequences — making this the ideal domain for rigorous RL evaluation.

---

## Architecture

```
JudicialEnv (gymnasium.Env)
    │
    ├── reset()  →  JudicialObservation (Pydantic)
    ├── step()   →  (obs, reward, done, truncated, info)
    └── state()  →  dict
```

### Observation Space

| Field | Type | Description |
|---|---|---|
| `case_id` | string | Unique case identifier (e.g. `C001`, `T002`) |
| `fact_pattern` | string | Who did what, when, and to whom |
| `statutes` | list[string] | Applicable laws (1–3 per case) |
| `precedents` | list[dict] | Prior cases with verdicts and rationale |
| `evidence_flags` | list[string] | Disputed facts and reliability markers |
| `domain` | string | `contract` / `tort` / `property` |
| `difficulty` | string | `easy` / `medium` / `hard` |

### Action Space

```json
{
  "verdict": "liable | not_liable | guilty | not_guilty | partial_liability",
  "confidence_score": 0.0,
  "reasoning_chain": "step-by-step natural language justification",
  "cited_precedents": ["case_id_1", "case_id_2"]
}
```

---

## Reward Function

```
R = 0.3·logic + 0.4·accuracy + 0.2·fairness + 0.1·citation
  − 0.2·(per hallucinated precedent, capped at −0.4)
  + 0.1·(adversarial bonus for hard cases with rich reasoning)
```

| Component | Weight | Description |
|---|---|---|
| **Logic** | 0.3 | Confidence × quality of reasoning chain (length + legal keywords) |
| **Accuracy** | 0.4 | Exact match against gold-label expert verdict |
| **Fairness** | 0.2 | Consistency of verdicts across same-domain cases |
| **Citation** | 0.1 | Ratio of valid cited precedents to total cited |
| **Hallucination penalty** | −0.2 each | Citing precedents not in the provided case file |
| **Adversarial bonus** | +0.1 | Hard cases with ≥2 evidence disputes + rich reasoning |

---

## TRL GRPO Training (Hackathon Judges)

To fulfill the Meta OpenEnv requirement for a full Reinforcement Learning loop, we have provided the `train.py` script. Because RL on LLMs requires significant VRAM, this should be run on a GPU instance (e.g., Google Colab).

The script uses **Unsloth** for rapid loading and the **TRL `GRPOTrainer`** to optimize the model using our custom Verifiable Rewards (RLVR):
1. **Format Reward:** Ensures the LLM outputs strict XML (`<action>`, `<verdict>`, `<reasoning_chain>`).
2. **Logic & Citation Reward:** Scans the reasoning chain for logical depth and explicit citations of the Constitution and the BNS.
3. **Accuracy Reward:** Hooks directly into `JudicialEnv.step()` to programmatically evaluate if the LLM reached the correct legal conclusion.

**How to run it:**
1. Open a Google Colab notebook (T4 GPU is sufficient).
2. Clone this repository.
3. Run the following:
   ```bash
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   !pip install --no-deps xformers trl peft accelerate bitsandbytes datasets
   !python train.py
   ```
4. *Local Testing:* If you want to verify the reward logic without a GPU, simply run `python test_reward_loop.py` locally.

---

## Tasks

| Task | Domain | Difficulty | Baseline Score | Description |
|---|---|---|---|---|
| `task1_contract` | Contract Law | Easy | **0.9342** | Breach, delay, title defects under Indian Contract Act |
| `task2_tort` | Tort/Negligence | Medium | **0.9632** | Conflicting evidence, contributory negligence, medical malpractice |
| `task3_property` | Property/Inheritance | Hard | **1.0000** | Competing claims, adverse possession, contested wills |
| **Overall** | | | **0.9658** | |

---

## Dataset

- **14 curated cases** across 3 domains
- **5 contract cases** — breach, delay, title defects, substantial performance
- **4 tort cases** — medical negligence, road accidents, product liability, restaurant duty of care
- **5 property cases** — adverse possession, co-ownership, tenancy rights, building violations, contested wills

**Sources:**
- [IndianKanoon](https://indiankanoon.org/) — Primary source for Indian case law
- [Harvard Caselaw Access Project](https://case.law/) — Common law precedents (Hadley v Baxendale, Donoghue v Stevenson, etc.)
- Expert-curated gold labels and reasoning chains

---

## Setup

### Docker (Recommended)

```bash
docker build -t judicial-env .

docker run \
  -e GROQ_API_KEY=your_groq_key \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.3-70b-versatile \
  -e HF_TOKEN=your_hf_token \
  -p 7860:7860 \
  judicial-env
```

### Local Run

```bash
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=your_groq_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile

python inference.py
```

---

## API Endpoints (HF Space)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Environment info and current status |
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/tasks` | GET | List all 3 tasks with metadata |
| `/reset` | POST | Reset environment, returns initial observation |
| `/results` | GET | Baseline scores after inference completes |

**Live Space:** https://huggingface.co/spaces/RishitaRamola42/judicial-reasoning-env

---

## Python Usage

```python
from environment import JudicialEnv, JudicialAction

# Initialize
env = JudicialEnv(domain="contract", difficulty="easy")

# Reset
obs, info = env.reset()
print(obs.fact_pattern)

# Create an action
action = JudicialAction(
    verdict="liable",
    confidence_score=0.92,
    reasoning_chain="The defendant breached the contract by failing to deliver...",
    cited_precedents=["P001", "P002"]
)

# Step
obs, reward, done, truncated, info = env.step(action)
print(f"Reward: {reward:.4f}")
print(f"Breakdown: {info}")
```

---

## Competitive Differentiators

- **No existing benchmark** tests legal reasoning in RL — this fills a real gap
- **Anti-hallucination built-in** — citation scoring and hallucination penalty
- **3-difficulty progression** — supports curriculum learning
- **Indian jurisdiction focus** — directly relevant to the Indian hackathon context, using IndianKanoon precedents
- **Modular & scalable** — designed to scale to 500+ cases for Round 2

---

## Team

**ALACRITY**
- Rishita Ramola — Team Lead
- Sarthak Singh

*OpenEnv Hackathon | Scaler × Meta | April 2026*