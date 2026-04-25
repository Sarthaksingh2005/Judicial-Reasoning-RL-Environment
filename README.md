---
title: Judicial Reasoning Env
emoji: ⚖️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# ⚖️ JusticeEngine-01 — BNS Legal Mediation RL Environment

> **OpenEnv Hackathon | Team ALACRITY | Scaler × Meta**

A **Gymnasium-compatible Reinforcement Learning environment** where an LLM-based Agent Mediator must navigate complex Indian legal disputes grounded in the **Bharatiya Nyaya Sanhita (BNS) 2023**, the **Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023**, and the Indian Constitution. A naive LLM **will fail** this environment without specific legal training.

---

## 🔗 Hackathon Submission Links

- **Hugging Face Space (Live Demo):** [RishitaRamola42/judicial-reasoning-env](https://huggingface.co/spaces/RishitaRamola42/judicial-reasoning-env)
- **Colab Training Notebook:** [JusticeEngine_RL_Training.ipynb](https://colab.research.google.com/github/rishitaramola/judicial-reasoning-env)
- **Code Repository:** [rishitaramola/judicial-reasoning-env](https://github.com/rishitaramola/judicial-reasoning-env)
- **YouTube Demo Video:** *(Add link before April 26th submission deadline)*

---

## 🧠 Why This Environment Is Hard To Game

Most RL environments reward task completion. This environment rewards **impartial, legally-grounded reasoning**. A standard LLM will:

- ❌ Hallucinate non-existent BNS sections (citation penalty)
- ❌ Give biased verdicts favoring the plaintiff (neutrality penalty)
- ❌ Take 20 turns to resolve a 2-turn case (efficiency penalty)
- ❌ Fail to cite the *correct* BNS section even when the right verdict is chosen

**Only a model trained on this rubric consistently scores above 0.7.**

---

## 🏛️ Environment Architecture

```
JudicialEnv (gymnasium.Env)
    │
    ├── reset()   →  JudicialObservation  (what the agent sees)
    ├── step()    →  (obs, reward, done, truncated, info)
    ├── state()   →  dict (for API/debugging)
    └── render()  →  stdout case summary

Multi-Agent Council (inference layer, not the environment):
    ├── Agent 1: Llama-3.3-70B  (Precedent Analyst)
    ├── Agent 2: Qwen-2.5-72B   (Constitutional Scholar)
    ├── Agent 3: Mixtral-8x7B   (Legal Realist)
    └── Chief Justice: DeepSeek-R1-32B (Synthesis)
```

---

## 📡 State Space (What The Agent Sees)

Each episode, the agent receives a `JudicialObservation`:

| Field | Type | Example |
|---|---|---|
| `case_id` | `str` | `"C001"` |
| `fact_pattern` | `str` | `"Ramesh agreed to supply 500 kg wheat..."` |
| `statutes` | `list[str]` | `["Indian Contract Act 1872 §73", "BNS 2023 §316"]` |
| `precedents` | `list[dict]` | `[{"case_id": "P001", "ruling": "liable", "ratio": "..."}]` |
| `evidence_flags` | `list[str]` | `["Written contract present", "No delivery receipt"]` |
| `domain` | `str` | `"contract"` / `"tort"` / `"property"` / `"petty_crime"` |
| `difficulty` | `str` | `"easy"` / `"medium"` / `"hard"` |
| `court_hierarchy_verdicts` | `dict` | `{"high_court": "liable", "supreme_court": "liable"}` |

**Difficulty determines adversarial complexity:**
- `easy` — single statute, unambiguous facts, no conflicting evidence
- `medium` — competing precedents, partial evidence, 2 plausible verdicts
- `hard` — conflicting High Court / Supreme Court rulings, missing evidence, constitutional angle

---

## 🎯 Action Space (What The Agent Can Do)

The agent outputs a `JudicialAction`:

```json
{
  "verdict": "liable | not_liable | guilty | not_guilty | partial_liability | forward_to_judge",
  "confidence_score": 0.0,
  "reasoning_chain": "Step-by-step natural language justification referencing statutes",
  "cited_precedents": ["P001", "P002"],
  "ratio_decidendi": "The binding legal principle: ...",
  "obiter_dicta": "Non-binding observation: ...",
  "fine_imposed": 50000.0,
  "appeal_recommended": false,
  "refer_to_human_judge": false
}
```

---

## 📊 BNS Reward Rubric (The "Hard To Game" Core)

```
R = 0.30·legal_accuracy
  + 0.20·bns_citation_precision
  + 0.20·neutrality
  + 0.15·logical_depth
  + 0.10·settlement_efficiency
  + 0.05·constitutional_grounding
  − 0.20·(per hallucinated precedent, max −0.40)
  − 0.10·(biased language toward either party)
  + 0.10·(adversarial bonus: hard case + rich reasoning)
  + 0.05·(SC alignment bonus)
  − 0.15·(hierarchy violation: chose HC over SC)
```

### Rubric Component Breakdown

| Component | Weight | How It's Scored |
|---|---|---|
| **Legal Accuracy** | 30% | Exact match against expert gold-label verdict |
| **BNS Citation Precision** | 20% | Correct BNS/BNSS section cited (not just any section) |
| **Neutrality** | 20% | Bias detector — penalises language favoring plaintiff/defendant |
| **Logical Depth** | 15% | Reasoning chain length + legal keyword density |
| **Settlement Efficiency** | 10% | Resolved in fewer turns = higher score |
| **Constitutional Grounding** | 5% | Referenced the Constitution or a Supreme Court ruling |
| **Hallucination Penalty** | −0.2 each | Cited a case ID not in the provided precedent list |
| **Bias Penalty** | −0.1 | Used charged/biased language ("obviously guilty", "clearly liable") |
| **Adversarial Bonus** | +0.1 | Hard difficulty + ≥2 evidence disputes + >50 word reasoning |
| **SC Alignment Bonus** | +0.05 | Verdict matches Supreme Court hierarchy record |
| **Hierarchy Violation** | −0.15 | Chose High Court ruling when SC ruling was available |

### Why Neutrality Matters

BNS §35 requires equal application of law regardless of the accused's background. The neutrality scorer detects:
- Charged adjectives ("ruthless", "innocent victim", "obvious fraud")
- Asymmetric benefit-of-the-doubt language
- Verdict language that pre-judges before citing evidence

---

## 📈 Training Results

We trained Llama-3-8B for **60 steps** using GRPO (Group Relative Policy Optimization) on the JudicialEnv reward rubric.

### Reward Curve (Before vs. After)

| Phase | Total Reward | Format | Accuracy | Logic |
|---|---|---|---|---|
| **Step 1-10** (Baseline) | −1.0 | 0.0 | −1.0 | 0.0 |
| **Step 20-30** (Learning) | −0.3 | 0.3 | −0.2 | 0.0 |
| **Step 50-60** (Trained) | **+0.6** | **0.5** | **+0.5** | **0.1** |

**What changed:**
- **Before:** Model hallucinated precedents, no XML structure, random verdicts
- **After:** Strict XML output, correct BNS citations, `forward_to_judge` default for criminal cases

### LoRA Adapter
The trained weights are available at `outputs/justice_engine_lora` (185MB, rank-16 LoRA on Llama-3-8B).

---

## 📋 Tasks

| Task | Domain | Difficulty | Baseline Score | Description |
|---|---|---|---|---|
| `task1_contract` | Contract Law | Easy | **0.9342** | Breach, delay, title defects under Indian Contract Act + BNS §316 |
| `task2_tort` | Tort/Negligence | Medium | **0.9632** | Conflicting evidence, contributory negligence, BNS §125 |
| `task3_property` | Property/Inheritance | Hard | **1.0000** | Competing claims, adverse possession, contested wills |
| **Overall** | | | **0.9658** | |

---

## 🗃️ Dataset

- **17 curated cases** across 4 domains (contract, tort, property, petty_crime)
- All cases grounded in **BNS 2023**, **BNSS 2023**, and the Indian Constitution
- Gold-label expert verdicts with full reasoning chains
- Court hierarchy records (High Court + Supreme Court rulings where applicable)

**Sources:** [IndianKanoon](https://indiankanoon.org/) · [Harvard Caselaw](https://case.law/) · Expert legal annotation

---

## 🚀 Quick Start

### Python

```python
from environment import JudicialEnv, JudicialAction

env = JudicialEnv(domain="contract", difficulty="easy")
obs, info = env.reset()

print(obs.fact_pattern)   # What the agent sees
print(obs.statutes)        # Applicable BNS sections

action = JudicialAction(
    verdict="liable",
    confidence_score=0.92,
    reasoning_chain="Under BNS §316, the defendant's failure to deliver constitutes cheating. "
                    "The Indian Contract Act §73 provides for damages...",
    cited_precedents=["P001"],
    ratio_decidendi="A party who accepts payment and fails to deliver is liable under §316 BNS."
)

obs, reward, done, truncated, info = env.step(action)
print(f"Reward: {reward:.4f}")
# → Reward: 0.8750
print(info)
# → {'logic_score': 0.85, 'accuracy_score': 1.0, 'fairness_score': 1.0, 'citation_score': 1.0, ...}
```

### Docker

```bash
docker build -t judicial-env .
docker run -e HF_TOKEN=your_token -p 7860:7860 judicial-env
```

### Training (Colab T4)

```bash
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes datasets
!python admin_tools/train.py
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Environment info and live status |
| `/health` | GET | `{"status": "ok"}` |
| `/tasks` | GET | All tasks with metadata |
| `/reset` | POST | New episode — returns `JudicialObservation` |
| `/ai_judge` | POST | Run multi-agent council, returns `JudicialAction` + reward |
| `/escalate` | POST | Log escalation to human judge |
| `/results` | GET | Baseline scores after inference |
| `/police` | GET | Police Evidence Verification Module UI |
| `/judge` | GET | Judge Dashboard UI |

---

## 🏆 Competitive Differentiators

| Feature | This Env | Typical RL Env |
|---|---|---|
| Legal jurisdiction specificity | BNS 2023 / Indian Constitution | Generic |
| Anti-hallucination reward | ✅ Citation penalty | ❌ |
| Neutrality scoring | ✅ Bias detector | ❌ |
| Court hierarchy alignment | ✅ HC / SC records | ❌ |
| Multi-agent deliberation | ✅ 3 agents + Chief Justice | ❌ |
| Real precedent grounding | ✅ IndianKanoon | ❌ |
| End-to-end RLVR training | ✅ GRPO on T4 | Rarely |

---

## 👥 Team

**ALACRITY**
- Rishita Ramola — Team Lead, Environment Design, RL Training
- Sarthak Singh — Legal Dataset Curation, Agent Architecture

*OpenEnv Hackathon | Scaler × Meta | April 2026*