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
    similar_cases: Optional[List[str]] = []               # IDs of similar resolved cases
    court_hierarchy_verdicts: Optional[dict] = {}          # {"high_court": "liable", "supreme_court": "not_liable"}


class JudicialAction(BaseModel):
    verdict: str
    confidence_score: float
    reasoning_chain: str
    cited_precedents: List[str]
    ratio_decidendi: Optional[str] = ""   # Binding legal principle for the decision
    obiter_dicta: Optional[str] = ""       # Non-binding observations made in passing
    fine_imposed: Optional[float] = None   # Civil fine if applicable
    appeal_recommended: Optional[bool] = False
    refer_to_human_judge: Optional[bool] = False
    case_status: Optional[str] = "open"   # open | resolved_by_ai | forwarded_to_judge | appealed


class JudicialReward(BaseModel):
    logic_score: float
    accuracy_score: float
    fairness_score: float
    citation_score: float
    neutrality_score: float        # NEW: bias/charged-language detector
    bns_precision_score: float     # NEW: correct BNS section cited (not just any)
    efficiency_score: float        # NEW: settlement in fewer steps = higher score
    constitutional_score: float    # NEW: referenced Constitution / SC ruling
    composite: float


class JudicialEnv(gym.Env):
    """
    Gymnasium-compatible RL environment for legal reasoning.

    An LLM-based agent acts as a judge. Each episode, the agent receives
    a curated Indian legal case and must deliver a structured verdict.

    Observation: JudicialObservation (Pydantic model)
    Action:      JudicialAction (Pydantic model)
    Reward:      BNS Rubric composite score [0.0, 1.0]

    R = 0.30·legal_accuracy
      + 0.20·bns_citation_precision
      + 0.20·neutrality
      + 0.15·logical_depth
      + 0.10·settlement_efficiency
      + 0.05·constitutional_grounding
      − 0.20·(per hallucinated precedent, max −0.40)
      − 0.10·(biased language toward either party)
      + 0.10·(adversarial bonus: hard case + ≥2 evidence + >50 words)
      + 0.05·(SC alignment bonus)
      − 0.15·(hierarchy violation: chose HC over SC)
    """

    metadata = {"render_modes": ["human"]}

    VALID_VERDICTS = ["liable", "not_liable", "guilty", "not_guilty", "partial_liability", "forward_to_judge"]

    def __init__(self, domain: str = None, difficulty: str = None, render_mode: str = None):
        super().__init__()
        self.domain = domain
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.current_case = None
        self.verdict_history: List[dict] = []
        self._done = False
        self._step_count = 0          # tracks episode turn count for efficiency scoring
        self._load_cases()

        # Gymnasium required spaces (symbolic — LLM agents use Pydantic models directly)
        self.observation_space = spaces.Dict({
            "case_id": spaces.Text(max_length=10),
            "fact_pattern": spaces.Text(max_length=2000),
            "domain": spaces.Text(max_length=20),
            "difficulty": spaces.Text(max_length=10),
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
        self._step_count = 0
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

        self._step_count += 1
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
            "neutrality_score": reward_obj.neutrality_score,
            "bns_precision_score": reward_obj.bns_precision_score,
            "efficiency_score": reward_obj.efficiency_score,
            "constitutional_score": reward_obj.constitutional_score,
            "composite_reward": reward_obj.composite,
            "case_id": self.current_case["case_id"],
            "gold_label": self.current_case.get("gold_label_verdict") or self.current_case.get("expert_verdict", "forward_to_judge")
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
        Full BNS Rubric:
          R = 0.30·accuracy + 0.20·bns_precision + 0.20·neutrality
            + 0.15·logic + 0.10·efficiency + 0.05·constitutional
            − 0.20·(per hallucination, max −0.40)
            − 0.10·(bias penalty)
            + 0.10·(adversarial bonus)
            + 0.05·(SC alignment)
            − 0.15·(hierarchy violation)
        """
        accuracy    = self._accuracy_score(action)
        citation    = self._citation_score(action)
        fairness    = self._fairness_score(action)
        logic       = self._logic_score(action)
        neutrality  = self._neutrality_score(action)
        bns_prec    = self._bns_precision_score(action)
        efficiency  = self._efficiency_score()
        const_score = self._constitutional_score(action)

        # Hallucination penalty — cited IDs not in the provided case file
        valid_ids = [p["case_id"] for p in self.current_case["precedents"]]
        hallucination_penalty = 0.0
        for cited in action.cited_precedents:
            if cited not in valid_ids:
                hallucination_penalty += 0.2
        hallucination_penalty = min(hallucination_penalty, 0.4)

        # Bias penalty — charged language toward either party
        bias_penalty = 0.0 if neutrality >= 0.5 else 0.1

        # Adversarial bonus — hard case with rich reasoning
        adversarial_bonus = 0.0
        if (
            self.difficulty == "hard"
            and len(self.current_case.get("evidence_flags", [])) >= 2
            and len(action.reasoning_chain.split()) > 50
        ):
            adversarial_bonus = 0.1

        composite = (
            0.30 * accuracy
            + 0.20 * bns_prec
            + 0.20 * neutrality
            + 0.15 * logic
            + 0.10 * efficiency
            + 0.05 * const_score
            - hallucination_penalty
            - bias_penalty
            + adversarial_bonus
        )

        # SC Alignment Bonus
        hierarchy = self.current_case.get("court_hierarchy_verdicts", {})
        sc_verdict = hierarchy.get("supreme_court", None)
        if sc_verdict and action.verdict == sc_verdict:
            composite += 0.05

        # Hierarchy Violation Penalty
        if sc_verdict and action.verdict != sc_verdict:
            hc_verdict = hierarchy.get("high_court", None)
            if hc_verdict and action.verdict == hc_verdict:
                composite -= 0.15

        composite = max(0.001, min(0.999, composite))

        return JudicialReward(
            logic_score=round(logic, 4),
            accuracy_score=round(accuracy, 4),
            fairness_score=round(fairness, 4),
            citation_score=round(citation, 4),
            neutrality_score=round(neutrality, 4),
            bns_precision_score=round(bns_prec, 4),
            efficiency_score=round(efficiency, 4),
            constitutional_score=round(const_score, 4),
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

        # Legal keyword quality signal — includes BNS-specific terms
        legal_keywords = [
            "statute", "section", "precedent", "liable", "duty", "negligence",
            "contract", "breach", "evidence", "plaintiff", "defendant", "damages",
            "reasonable", "burden", "therefore", "hence", "conclude", "holding",
            "bns", "bnss", "sanhita", "constitution", "article", "ratio",
            "forward_to_judge", "punishable", "cognizable", "fir", "supreme court"
        ]
        reasoning_lower = action.reasoning_chain.lower()
        keyword_hits = sum(1 for kw in legal_keywords if kw in reasoning_lower)
        keyword_bonus = min(keyword_hits / len(legal_keywords), 1.0) * 0.15

        logic = min(confidence_component + length_bonus + keyword_bonus, 1.0)
        return round(logic, 4)

    def _accuracy_score(self, action: JudicialAction) -> float:
        """Exact match against gold label. Partial credit for partial_liability adjacency.
        For criminal cases that only have expert_verdict, use that as the gold label.
        """
        # Support both gold_label_verdict (civil) and expert_verdict (criminal)
        gold = self.current_case.get("gold_label_verdict") or self.current_case.get("expert_verdict", "forward_to_judge")
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

    def _neutrality_score(self, action: JudicialAction) -> float:
        """
        BNS §35 — Equal application of law regardless of accused's background.
        Detects charged/biased language in the reasoning chain.
        Score 1.0 = fully neutral, 0.0 = heavily biased.
        """
        BIAS_INDICATORS = [
            # Plaintiff-biased language
            "obviously guilty", "clearly at fault", "undoubtedly liable",
            "brazen", "ruthless", "malicious intent", "innocent victim",
            "unquestionably", "without doubt", "certainly guilty",
            # Defendant-biased language
            "clearly innocent", "obviously not guilty", "victim of false accusation",
            "baseless claim", "frivolous complaint", "fabricated evidence",
            "no basis whatsoever", "utterly unfounded",
            # General prejudicial language
            "of course", "naturally", "anyone can see", "it is obvious that",
        ]
        reasoning_lower = action.reasoning_chain.lower()
        hits = sum(1 for phrase in BIAS_INDICATORS if phrase in reasoning_lower)
        # Each bias hit reduces neutrality by 0.2, minimum 0.0
        neutrality = max(0.0, 1.0 - (hits * 0.2))
        return round(neutrality, 4)

    def _bns_precision_score(self, action: JudicialAction) -> float:
        """
        Checks if the reasoning chain cites the *correct* BNS/BNSS section
        (not just any section). The correct sections are taken from the case's
        applicable_statutes list.
        Score = fraction of applicable statutes mentioned in the reasoning.
        """
        applicable = [s.lower() for s in self.current_case.get("applicable_statutes", [])]
        if not applicable:
            return 0.5  # Neutral if case has no statutes defined
        reasoning_lower = action.reasoning_chain.lower()

        # Extract key identifiers: section numbers and BNS/BNSS keyword
        hits = 0
        for statute in applicable:
            # Check if the statute text or key section number appears in reasoning
            words = statute.replace("§", "section ").replace("sec.", "section ").split()
            # A hit requires at least 2 words from the statute to appear
            word_hits = sum(1 for w in words if len(w) > 2 and w in reasoning_lower)
            if word_hits >= 2:
                hits += 1

        return round(hits / len(applicable), 4)

    def _efficiency_score(self) -> float:
        """
        Settlement efficiency: reward resolving cases in fewer turns.
        Optimal = 1 turn = 1.0. Each additional turn beyond 3 reduces score.
        """
        if self._step_count <= 1:
            return 1.0
        elif self._step_count <= 2:
            return 0.8
        elif self._step_count <= 3:
            return 0.6
        else:
            return max(0.2, 1.0 - (self._step_count * 0.15))

    def _constitutional_score(self, action: JudicialAction) -> float:
        """
        Bonus for grounding the verdict in the Constitution of India
        or citing a Supreme Court ruling.
        """
        CONSTITUTIONAL_MARKERS = [
            "constitution", "article 14", "article 21", "article 19",
            "fundamental right", "supreme court", "sc ruling",
            "constitutional bench", "high court", "chief justice of india",
            "right to equality", "right to life", "directive principle"
        ]
        reasoning_lower = action.reasoning_chain.lower()
        hits = sum(1 for m in CONSTITUTIONAL_MARKERS if m in reasoning_lower)
        return min(1.0, hits * 0.25)  # 1 hit = 0.25, 4+ hits = 1.0