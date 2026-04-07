"""
LLM Grader — Judicial Reasoning RL Environment
Team ALACRITY — OpenEnv Hackathon

LLM-based rubric grader for evaluating reasoning chain quality.
Used primarily for Task 2 (tort/negligence) where ambiguity requires
qualitative assessment beyond gold-label matching.

All scores are strictly in range [0.0, 1.0].
"""

import os
import json
import sys
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import JudicialAction, JudicialObservation

load_dotenv()


class LLMGrader:
    """
    LLM-based rubric grader for qualitative reasoning evaluation.

    Uses the same OpenAI-compatible client as inference.py.
    Evaluates reasoning chains on a 5-point rubric:
      1. Legal accuracy (correct statute/precedent application)
      2. Logical structure (premises → conclusion)
      3. Evidence handling (addresses disputed facts)
      4. Proportionality (verdict matches severity)
      5. Citation quality (references are apt and relevant)
    """

    RUBRIC = """You are a senior legal examiner grading a judge's reasoning chain.

Score the following reasoning chain on each criterion from 0 to 2:
1. Legal accuracy — Does the reasoning correctly apply the cited statutes and precedents?
2. Logical structure — Does the reasoning flow coherently from facts to verdict?
3. Evidence handling — Does the reasoning address disputed evidence and flag ambiguities?
4. Proportionality — Is the verdict proportionate to the severity of the facts?
5. Citation quality — Are the cited precedents directly relevant to the fact pattern?

FACT PATTERN:
{fact_pattern}

VERDICT: {verdict}
REASONING CHAIN:
{reasoning_chain}

CITED PRECEDENTS: {cited_precedents}

Respond ONLY with a valid JSON object:
{{
  "legal_accuracy": 0-2,
  "logical_structure": 0-2,
  "evidence_handling": 0-2,
  "proportionality": 0-2,
  "citation_quality": 0-2,
  "overall_comment": "one sentence summary"
}}"""

    def __init__(self):
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("GROQ_API_KEY", "")
        base_url = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
        self.model = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._available = bool(api_key)

    def grade_reasoning(
        self,
        obs: JudicialObservation,
        action: JudicialAction,
        fallback_score: float = 0.5
    ) -> dict:
        """
        Grade a reasoning chain using an LLM rubric.

        Args:
            obs: The observation the agent was responding to
            action: The agent's action containing the reasoning chain
            fallback_score: Score to return if LLM call fails

        Returns:
            dict with individual criterion scores and normalized total [0.0, 1.0]
        """
        if not self._available:
            return self._fallback_result(fallback_score, reason="No API key configured")

        prompt = self.RUBRIC.format(
            fact_pattern=obs.fact_pattern,
            verdict=action.verdict,
            reasoning_chain=action.reasoning_chain,
            cited_precedents=", ".join(action.cited_precedents) or "None"
        )

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0  # Deterministic grading
                )
                raw = response.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                data = json.loads(raw)

                # Normalize: max raw score = 5 criteria × 2 = 10 → divide by 10
                raw_total = (
                    data.get("legal_accuracy", 0)
                    + data.get("logical_structure", 0)
                    + data.get("evidence_handling", 0)
                    + data.get("proportionality", 0)
                    + data.get("citation_quality", 0)
                )
                normalized = max(0.0, min(1.0, raw_total / 10.0))

                return {
                    "legal_accuracy": data.get("legal_accuracy", 0) / 2,
                    "logical_structure": data.get("logical_structure", 0) / 2,
                    "evidence_handling": data.get("evidence_handling", 0) / 2,
                    "proportionality": data.get("proportionality", 0) / 2,
                    "citation_quality": data.get("citation_quality", 0) / 2,
                    "overall_comment": data.get("overall_comment", ""),
                    "normalized_score": round(normalized, 4)
                }

            except (json.JSONDecodeError, KeyError, Exception):
                if attempt == 2:
                    return self._fallback_result(fallback_score, reason="LLM parse failed after 3 attempts")
                continue

        return self._fallback_result(fallback_score)

    def _fallback_result(self, score: float, reason: str = "") -> dict:
        """Return a safe fallback result when LLM grading is unavailable."""
        return {
            "legal_accuracy": score / 2,
            "logical_structure": score / 2,
            "evidence_handling": score / 2,
            "proportionality": score / 2,
            "citation_quality": score / 2,
            "overall_comment": f"Fallback score used. {reason}",
            "normalized_score": round(max(0.0, min(1.0, score)), 4)
        }


if __name__ == "__main__":
    from environment import JudicialEnv

    print("Running LLMGrader smoke test (requires GROQ_API_KEY)...")
    env = JudicialEnv(domain="tort", difficulty="medium")
    obs, _ = env.reset()

    test_action = JudicialAction(
        verdict="liable",
        confidence_score=0.85,
        reasoning_chain=(
            "The hospital owed a duty of care to the patient as established in Jacob Mathew v State of Punjab. "
            "The surgical sponge found inside the patient post-operation invokes res ipsa loquitur — "
            "the thing speaks for itself. The hospital's claim that the infection was unrelated is insufficient "
            "to rebut this presumption. Therefore the hospital is liable for medical negligence."
        ),
        cited_precedents=["P004", "P005"]
    )

    grader = LLMGrader()
    result = grader.grade_reasoning(obs, test_action)
    print(json.dumps(result, indent=2))
