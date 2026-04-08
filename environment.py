import json
import random
import os
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class JudicialObservation(BaseModel):
    case_id: str
    fact_pattern: str
    statutes: List[str]
    precedents: List[dict]
    evidence_flags: List[str]
    domain: str
    difficulty: str


class JudicialAction(BaseModel):
    verdict: str
    confidence_score: float
    reasoning_chain: str
    cited_precedents: List[str]


class JudicialReward(BaseModel):
    logic_score: float
    accuracy_score: float
    fairness_score: float
    citation_score: float
    composite: float


class JudicialEnv(gym.Env):
    """
    Gymnasium-compatible RL environment for legal reasoning.

    An LLM-based agent acts as a judge. Each episode, the agent receives
    a curated Indian legal case and must deliver a structured verdict.

    Observation: JudicialObservation (Pydantic model)
    Action:      JudicialAction (Pydantic model)
    Reward:      Composite score [0.0, 1.0]
                 R = 0.3·logic + 0.4·accuracy + 0.2·fairness + 0.1·citation
    """

    metadata = {"render_modes": ["human"]}

    VALID_VERDICTS = ["liable", "not_liable", "guilty", "not_guilty", "partial_liability"]

    def __init__(self, domain: str = None, difficulty: str = None, render_mode: str = None):
        super().__init__()
        self.domain = domain
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.current_case = None
        self.verdict_history: List[dict] = []
        self._done = False
        self._load_cases()

        # Gymnasium required spaces (symbolic — LLM agents use Pydantic models directly)
        self.observation_space = spaces.Dict({
            "case_id": spaces.Text(min_length=1, max_length=10),
            "fact_pattern": spaces.Text(min_length=1, max_length=2000),
            "domain": spaces.Text(min_length=1, max_length=20),
            "difficulty": spaces.Text(min_length=1, max_length=10),
            "num_precedents": spaces.Discrete(10),
            "num_statutes": spaces.Discrete(10),
            "num_evidence_flags": spaces.Discrete(10),
        })

        self.action_space = spaces.Dict({
            "verdict": spaces.Discrete(len(self.VALID_VERDICTS)),
            "confidence_score": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def _load_cases(self):
        data_path = os.path.join(os.path.dirname(__file__), "data", "cases.json")
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                all_cases = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"cases.json not found at {data_path}. "
                "Ensure data/cases.json exists in the project directory."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in cases.json: {e}")

        self.cases = [
            c for c in all_cases
            if (self.domain is None or c["domain"] == self.domain)
            and (self.difficulty is None or c["difficulty"] == self.difficulty)
        ]

        if not self.cases:
            raise ValueError(
                f"No cases found for domain={self.domain!r}, difficulty={self.difficulty!r}. "
                f"Check cases.json has entries matching these filters."
            )

    def reset(self, seed: int = None, options: dict = None) -> Tuple[JudicialObservation, dict]:
        """Reset the environment and return initial observation."""
        super().reset(seed=seed)
        self._done = False
        if seed is not None:
            random.seed(seed)
        self.current_case = random.choice(self.cases)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: JudicialAction) -> Tuple[JudicialObservation, float, bool, bool, dict]:
        """
        Apply an action and return (observation, reward, terminated, truncated, info).

        Args:
            action: JudicialAction with verdict, confidence_score, reasoning_chain, cited_precedents

        Returns:
            obs: Next observation (new case loaded)
            reward: Composite score [0.0, 1.0]
            terminated: True (single-step episodes)
            truncated: False (no time limit)
            info: Reward breakdown dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        reward_obj = self._compute_reward(action)

        self.verdict_history.append({
            "case_id": self.current_case["case_id"],
            "verdict": action.verdict,
            "domain": self.current_case["domain"]
        })
        self._done = True

        info = {
            "logic_score": reward_obj.logic_score,
            "accuracy_score": reward_obj.accuracy_score,
            "fairness_score": reward_obj.fairness_score,
            "citation_score": reward_obj.citation_score,
            "composite_reward": reward_obj.composite,
            "case_id": self.current_case["case_id"],
            "gold_label": self.current_case["gold_label_verdict"]
        }

        obs = self._get_obs()
        return obs, reward_obj.composite, self._done, False, info

    def state(self) -> dict:
        """Return current environment state for API/debugging."""
        return {
            "current_case_id": self.current_case["case_id"] if self.current_case else None,
            "done": self._done,
            "verdict_history_length": len(self.verdict_history),
            "domain": self.domain,
            "difficulty": self.difficulty,
            "total_cases_available": len(self.cases)
        }

    def render(self):
        """Render the current case to stdout."""
        if self.render_mode == "human" and self.current_case:
            print(f"\n{'='*60}")
            print(f"CASE: {self.current_case['case_id']} | Domain: {self.current_case['domain']} | Difficulty: {self.current_case['difficulty']}")
            print(f"Fact Pattern: {self.current_case['fact_pattern']}")
            print(f"Statutes: {', '.join(self.current_case['applicable_statutes'])}")
            print(f"Evidence Flags: {', '.join(self.current_case['evidence_flags']) or 'None'}")
            print(f"Gold Label: {self.current_case['gold_label_verdict']}")
            print(f"{'='*60}")

    # ─── Private Helpers ──────────────────────────────────────────

    def _get_obs(self) -> JudicialObservation:
        if self.current_case is None:
            return JudicialObservation(
                case_id="", fact_pattern="", statutes=[],
                precedents=[], evidence_flags=[], domain="", difficulty=""
            )
        return JudicialObservation(
            case_id=self.current_case["case_id"],
            fact_pattern=self.current_case["fact_pattern"],
            statutes=self.current_case["applicable_statutes"],
            precedents=self.current_case["precedents"],
            evidence_flags=self.current_case["evidence_flags"],
            domain=self.current_case["domain"],
            difficulty=self.current_case["difficulty"]
        )

    def _get_info(self) -> dict:
        return {
            "case_id": self.current_case["case_id"] if self.current_case else None,
            "domain": self.domain,
            "difficulty": self.difficulty
        }

    def _compute_reward(self, action: JudicialAction) -> JudicialReward:
        """
        Composite reward: R = 0.3·logic + 0.4·accuracy + 0.2·fairness + 0.1·citation
        Penalties: −0.2 per hallucinated precedent (max −0.4)
        Bonus: +0.1 for robust reasoning on hard adversarial cases
        """
        accuracy = self._accuracy_score(action)
        citation = self._citation_score(action)
        fairness = self._fairness_score(action)
        logic = self._logic_score(action)

        # Penalize hallucinated precedents (cited IDs not in the provided case file)
        valid_ids = [p["case_id"] for p in self.current_case["precedents"]]
        hallucination_penalty = 0.0
        for cited in action.cited_precedents:
            if cited not in valid_ids:
                hallucination_penalty += 0.2
        hallucination_penalty = min(hallucination_penalty, 0.4)

        # Bonus for robust reasoning on hard adversarial cases
        adversarial_bonus = 0.0
        if (
            self.difficulty == "hard"
            and len(self.current_case.get("evidence_flags", [])) >= 2
            and len(action.reasoning_chain.split()) > 50
        ):
            adversarial_bonus = 0.1

        composite = (
            0.3 * logic
            + 0.4 * accuracy
            + 0.2 * fairness
            + 0.1 * citation
            - hallucination_penalty
            + adversarial_bonus
        )
        composite = max(0.001, min(0.999, composite))

        return JudicialReward(
            logic_score=round(logic, 4),
            accuracy_score=round(accuracy, 4),
            fairness_score=round(fairness, 4),
            citation_score=round(citation, 4),
            composite=round(composite, 4)
        )

    def _logic_score(self, action: JudicialAction) -> float:
        """
        Score logical consistency of the reasoning chain.
        - Base: confidence_score × 0.7
        - Length bonus: +0.15 for >50 words, +0.15 for >150 words
        - Keyword bonus: +up to 0.15 for legal terminology usage
        """
        word_count = len(action.reasoning_chain.split())
        confidence_component = min(action.confidence_score, 1.0) * 0.7

        length_bonus = 0.0
        if word_count > 50:
            length_bonus += 0.15
        if word_count > 150:
            length_bonus += 0.15

        # Legal keyword quality signal
        legal_keywords = [
            "statute", "section", "precedent", "liable", "duty", "negligence",
            "contract", "breach", "evidence", "plaintiff", "defendant", "damages",
            "reasonable", "burden", "therefore", "hence", "conclude", "holding"
        ]
        reasoning_lower = action.reasoning_chain.lower()
        keyword_hits = sum(1 for kw in legal_keywords if kw in reasoning_lower)
        keyword_bonus = min(keyword_hits / len(legal_keywords), 1.0) * 0.15

        logic = min(confidence_component + length_bonus + keyword_bonus, 1.0)
        return round(logic, 4)

    def _accuracy_score(self, action: JudicialAction) -> float:
        """Exact match against gold label. Partial credit for partial_liability adjacency."""
        gold = self.current_case["gold_label_verdict"]
        if action.verdict == gold:
            return 1.0
        # Partial credit if both are ambiguity-adjacent
        if "partial" in action.verdict or "partial" in gold:
            return 0.5
        return 0.0

    def _citation_score(self, action: JudicialAction) -> float:
        """Ratio of valid citations to total citations. Penalises empty citation list."""
        valid_ids = [p["case_id"] for p in self.current_case["precedents"]]
        if not action.cited_precedents:
            return 0.0
        hits = sum(1 for c in action.cited_precedents if c in valid_ids)
        return round(hits / max(len(action.cited_precedents), 1), 4)

    def _fairness_score(self, action: JudicialAction) -> float:
        """
        Consistency check: verdicts in same domain should be consistent.
        Returns 1.0 if fewer than 2 prior verdicts in the domain.
        """
        same_domain = [
            v for v in self.verdict_history
            if v.get("domain") == self.current_case["domain"]
        ]
        if len(same_domain) < 2:
            return 1.0
        verdicts = [v["verdict"] for v in same_domain]
        consistency = verdicts.count(verdicts[0]) / len(verdicts)
        return round(consistency, 4)