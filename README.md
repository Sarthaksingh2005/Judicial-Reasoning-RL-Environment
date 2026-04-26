# ⚖️ JudicialMediationEnv — Multi-Agent Indian Legal RL Environment

## Problem
LLMs frequently hallucinate legal citations and struggle with complex, adversarial reasoning when deployed in legal contexts. In India, where over 50 million cases are pending across courts, deploying AI without rigorous training and guardrails is dangerous. This environment directly addresses this by training an LLM using Reinforcement Learning (GRPO) to act as an impartial mediator, forcing it to weigh adversarial arguments and strictly adhere to the Bharatiya Nyaya Sanhita (BNS) and other Indian statutes without hallucinating.

## Environment
This project implements an OpenEnv-compatible multi-agent environment (`JudicialMediationEnv`).
- **The Setup:** A `PlaintiffBot` argues for liability, and a `DefendantBot` argues against it.
- **The Agent (Judge):** Must read the facts, hear both adversarial bots, and synthesize an IRAC-structured verdict (Issue, Rule, Application, Conclusion).
- **The Reward:** An automated programmatic grader (Rubric) issues rewards based on accuracy, fairness, and reasoning quality.

### State & Action Space
| Component | Description |
|-----------|-------------|
| **State Space** | JSON dictionary containing `case_facts`, `statutes_applicable`, `plaintiff_argument`, and `defendant_argument`. |
| **Action Space** | Structured JSON containing `verdict`, `confidence_score`, `reasoning_chain`, `cited_precedents`, and optional human escalation flags. |

## Reward Function
We use OpenEnv's Rubric system to provide a composable reward signal during GRPO training:
- **Legal Accuracy (0.35):** Compares the agent's verdict against gold labels derived from Supreme/High Court cases. Awards partial credit for adjacent verdicts (e.g., liable vs. partial_liability).
- **Neutrality (0.25):** Penalizes charged, biased, or highly emotive language. (Future: Demographic swap tests).
- **Reasoning Quality (0.20):** Checks for structural coherence (IRAC) and ensures the reasoning logically supports the final verdict.
- **Citation Validity (0.10):** Rewards correct usage of Indian case law.
- **Efficiency (0.10):** Rewards concise, direct resolutions without unnecessary hedging.
- **Hallucination Penalty (-0.30 to -0.60):** Severely penalizes fabricated case names or imaginary statutes.
- **Multi-LLM Panel Bonus (+0.15):** Bonus given if the trained agent's verdict aligns with a consensus panel of 3 baseline LLMs.

*Note: A random/baseline Llama 3.1 8B agent typically scores ~0.45 due to hallucinated Western precedents and structural failures. Post-training target is ~0.80.*

## Dataset
- **Size:** 48 curated cases (scaling to 10K+ for production).
- **Domains:** Civil (Contract/Tort/Property), Criminal (BNS/BNSS advisory), Family Law, Labor, and Constitutional.
- **Sources:** Derived from IndianKanoon API, NJDG, and Ministry of Home Affairs reference texts.
- **Precedents:** Western benchmarks (Hadley v Baxendale) have been explicitly replaced with binding Indian equivalents (Satyabrata Ghose v Mugneeram Bangur, M.C. Mehta v Union of India, Vishaka v State of Rajasthan).

## Training
The core training loop uses **GRPO** (Group Relative Policy Optimization) via the HuggingFace `TRL` library and `Unsloth` for 4-bit quantization. 
- **Model:** `unsloth/Meta-Llama-3.1-8B-Instruct`
- **Execution:** Runs efficiently on a free Colab T4 GPU.
- **Colab Notebook:** [Link to Training Notebook](https://colab.research.google.com/github/rishitaramola/judicial-reasoning-env/blob/main/training/training_colab.ipynb)

![Reward Curve](training_curve.png)

### Before/After Training Example
**Before (Baseline LLM):** 
*Verdict:* Liable. 
*Reasoning:* "Under Hadley v Baxendale, damages are owed. I estimate Rs 500,000." (Hallucination of Western precedent and monetary amounts).

**After (GRPO Trained Agent):**
*Verdict:* Liable.
*Reasoning:* "Issue: Whether force majeure applies. Rule: Indian Contract Act 1872 Section 73 & Satyabrata Ghose v Mugneeram Bangur. Application: Market price fluctuations do not constitute frustration. Conclusion: The defendant is liable for actual losses as stated in the facts."

## Results
| Metric | Baseline (Llama 3.1 8B) | Trained (GRPO) |
|--------|-------------------------|----------------|
| **Overall Reward Score** | ~0.48 | **TBD** |
| **Citation Hallucination Rate** | 42% | **TBD** |
| **IRAC Structure Compliance** | 31% | **TBD** |
| **Legal Accuracy (Gold Match)** | 60% | **TBD** |

## Related Work
While several excellent datasets and models exist in the legal AI space, this project is uniquely positioned as a multi-agent RL environment for Indian law:
- **LegalBench (Guha et al., 2023):** Tests legal reasoning in classification settings, mostly US-centric.
- **ILDC for ILSI:** Indian Legal Document Corpus for classification.
- **InLegalBERT:** An encoder model pre-trained on Indian legal texts.
- **Our Contribution:** Unlike static classifiers, JudicialMediationEnv applies RL with a structured multi-component reward function to adversarial multi-turn judicial decision-making strictly within the Indian context.

## Limitations
- **Dataset Size:** The current hackathon dataset is limited to ~48 cases.
- **Criminal Scope:** For ethical and legal safety, the AI acts strictly in an **Advisory** capacity for criminal cases (identifying BNS sections and summarizing facts for a human judge) rather than passing full guilty/not guilty verdicts.
- **Mock Interfaces:** The Aadhaar verification and Police Dashboards in the UI are functional mockups for the hackathon demonstration.

## Future Work
- **10K Case Training Set:** Expanding the dataset via automated scraping of IndianKanoon.
- **Real-Time API Integrations:** Full DigiLocker/Aadhaar integration for KYC and live IndianKanoon retrieval (RAG Mode).
- **Automated Bias Testing:** Expanding the `rubric.py` to automatically run demographic swap tests (e.g., swapping names/religions) during training to enforce strict neutrality.
- **Hindi Language Support:** Native processing of Hindi FIRs and legal summaries.

## Links
- **HF Space (Demo):** https://huggingface.co/spaces/RishitaRamola42/judicial-reasoning-env
- **GitHub Repository:** https://github.com/rishitaramola/judicial-reasoning-env
- **Colab Training Notebook:** [Open in Colab](https://colab.research.google.com/github/rishitaramola/judicial-reasoning-env/blob/main/training/training_colab.ipynb)
- **YouTube Demo Video:** [Add Link]