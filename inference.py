import asyncio
import os
import json
import time
from openai import OpenAI
from environment import JudicialEnv, JudicialAction
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN")          # No default — evaluator provides this
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # Local dev fallback only

API_KEY = HF_TOKEN or GROQ_API_KEY
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

MAX_TOTAL_REWARD       = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    {"name": "task1_contract", "domain": "contract", "difficulty": "easy"},
    {"name": "task2_tort",     "domain": "tort",     "difficulty": "medium"},
    {"name": "task3_property", "domain": "property", "difficulty": "hard"},
]


def log_start(task_name: str):
    print(f"[START] task={task_name}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    print(f"[STEP] step={step} action={action!r} reward={reward:+.2f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


def get_agent_action(obs) -> JudicialAction:
    """Call LLM to get a structured verdict. Retries up to 3 times on parse errors."""
    prompt = f"""You are an expert judge. Analyze the following legal case and deliver a structured verdict.

FACT PATTERN:
{obs.fact_pattern}

APPLICABLE STATUTES:
{chr(10).join(obs.statutes)}

PRECEDENTS:
{json.dumps(obs.precedents, indent=2)}

EVIDENCE FLAGS:
{', '.join(obs.evidence_flags) if obs.evidence_flags else 'None'}

Respond ONLY with a valid JSON object in this exact format:
{{
  "verdict": "liable OR not_liable OR guilty OR not_guilty",
  "confidence_score": 0.0 to 1.0,
  "reasoning_chain": "your step by step reasoning here",
  "cited_precedents": ["case_id_1", "case_id_2"]
}}"""

    last_error = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            return JudicialAction(
                verdict=data["verdict"],
                confidence_score=float(data["confidence_score"]),
                reasoning_chain=data["reasoning_chain"],
                cited_precedents=data.get("cited_precedents", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_error = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise ValueError(f"Failed to parse LLM response after 3 attempts: {last_error}")


async def run_task(task_config: dict) -> float:
    task_name = task_config["name"]
    log_start(task_name)

    env = JudicialEnv(domain=task_config["domain"], difficulty=task_config["difficulty"])
    obs, _ = env.reset()

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        for step in range(1, 4):
            try:
                action = get_agent_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action.verdict, reward=reward, done=done, error=None)
                if done:
                    obs, _ = env.reset()
            except Exception as e:
                error = str(e)
                log_step(step=step, action="ERROR", reward=0.0, done=False, error=error)
                break

        score = sum(rewards) / (len(rewards) * MAX_TOTAL_REWARD) if rewards else 0.001
        score = min(max(score, 0.001), 0.999)  # strictly between 0 and 1
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    all_scores = []
    for task in TASKS:
        score = await run_task(task)
        all_scores.append(score)

    print(f"\n=== BASELINE RESULTS ===", flush=True)
    for task, score in zip(TASKS, all_scores):
        print(f"{task['name']}: {score:.4f}", flush=True)
    print(f"OVERALL AVERAGE: {sum(all_scores)/len(all_scores):.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())