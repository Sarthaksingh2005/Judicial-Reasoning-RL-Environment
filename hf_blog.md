---
title: "JusticeEngine-01: Tackling the Indian Judicial Backlog with RLVR"
thumbnail: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlvr/thumbnail.png"
authors:
  - user: rishitaramola
  - user: Sarthaksingh2005
---

# ⚖️ JusticeEngine-01: Training Legal AI with Verifiable Rewards

*A Meta-PyTorch OpenEnv Hackathon Submission by Team ALACRITY*

## The Problem: A 50-Million Case Backlog
The Indian Judicial System is currently facing an unprecedented crisis. According to the **National Judicial Data Grid (NJDG)**, there are over 50 million pending cases. More than 3.48 lakh of these are uncontested or undated, clogging the system and delaying justice for critical issues.

Furthermore, India has just transitioned to a completely new legal framework in 2024: the **Bharatiya Nyaya Sanhita (BNS)** has replaced the colonial-era IPC. Legal professionals are overwhelmed, and citizens are left waiting.

## The Solution: JusticeEngine-01
We built **JusticeEngine-01**, an open-source Reinforcement Learning (RL) environment designed to train LLMs to act as highly strictly-regulated legal triage agents.

Using **Reinforcement Learning with Verifiable Rewards (RLVR)**, we train models not just to talk like lawyers, but to reason logically, cite existing precedents accurately, and—most importantly—know when to pass a verdict and when to escalate to a human judge.

### 🌟 Key Technical Differentiators

#### 1. Multi-Agent "Council of AIs"
Justice isn't a single opinion; it's a consensus. In our inference loop, we spin up three distinct AI personas:
- **The Strict Constitutionalist**
- **The Empathetic Mediator**
- **The Precedent Analyst**

These agents independently evaluate the case and vote. The environment enforces a majority consensus, significantly reducing edge-case hallucinations.

#### 2. Strict Civil vs. Criminal Domain Separation
Ethical legal AI must understand its boundaries. In our environment:
- **Civil Cases (Contract, Tort, Property):** The AI provides mediation and issues direct verdicts (`liable` / `not_liable`).
- **Criminal Cases (Petty Crime under BNS):** The AI is strictly instructed **never** to pass a verdict. It acts as a digital paralegal—bundling facts, verifying evidence through our simulated Police Module, identifying the exact BNS sections, and defaulting the outcome to `forward_to_judge`.

#### 3. Verifiable Reward Engineering
If you only have a single reward signal, LLMs will hack it. Our `JudicialEnv` implements layered, programmatic verification across four dimensions:
1. **Logic Score:** Checks reasoning length and legal keyword density.
2. **Accuracy Score:** A hard outcome check against expert gold labels.
3. **Fairness Score:** Evaluates consistency across similar cases in the same domain.
4. **Citation Anti-Hallucination:** Strictly penalizes any hallucinated case IDs.

## TRL GRPO Training Pipeline
To optimize the model, we use the **TRL `GRPOTrainer`** integrated with **Unsloth** for rapid loading. During training, the LLM generates reasoning paths in strict XML. The environment parses this XML, executes `env.step()`, and returns the programmatic rewards. Over time, the model learns to stop hallucinating precedents and starts citing the BNS accurately.

## Try It Yourself
JusticeEngine-01 isn't just an environment; it features a full frontend wizard for testing.
- **Hugging Face Space:** [Play with the interactive demo](https://huggingface.co/spaces/RishitaRamola42/judicial-reasoning-env)
- **Train it on Colab:** Check out our `training_notebook.ipynb` to run the RL loop yourself!
- **Codebase:** [View on GitHub](https://github.com/Sarthaksingh2005/Judicial-Reasoning-RL-Environment)

By automating the preliminary triage of uncontested cases with verifiable RL, we can help clear the NJDG backlog and ensure timely justice for all.
