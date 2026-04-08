"""
FastAPI application for the Judicial Reasoning RL Environment.
Team ALACRITY — OpenEnv Hackathon

This module creates an HTTP server that exposes the JudicialEnv
over HTTP endpoints, compatible with the OpenEnv spec.

The server:
- Exposes /reset, /step, /state endpoints for the RL API
- Hosts the inference runner as a background task on startup
- Stays alive persistently on port 7860 for HF Spaces
"""

import asyncio
import json
import os
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from dotenv import load_dotenv
from openai import OpenAI

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import JudicialEnv, JudicialAction
from server.models import (
    ResetRequest, StepRequest,
    ResetResponse, StepResponse, StateResponse, HealthResponse
)

load_dotenv()

# ─── Configuration ────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY      = os.environ.get("GROQ_API_KEY") or os.environ.get("HF_TOKEN", "")

MAX_TOTAL_REWARD       = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    {"name": "task1_contract", "domain": "contract", "difficulty": "easy"},
    {"name": "task2_tort",     "domain": "tort",     "difficulty": "medium"},
    {"name": "task3_property", "domain": "property", "difficulty": "hard"},
]

# Global results store
RESULTS: dict = {"status": "starting", "scores": {}, "overall": 0.0}

# ─── FastAPI App ──────────────────────────────────────────────

app = FastAPI(
    title="Judicial Reasoning RL Environment",
    description="An RL environment where an LLM agent acts as a judge over Indian legal cases.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect to the auto-generated API documentation (Swagger UI)."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint — required by OpenEnv pre-submission checklist."""
    return HealthResponse(status="ok")


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = None):
    """
    Reset the environment and return the initial observation.
    Required by OpenEnv spec — must return HTTP 200.
    """
    domain     = request.domain     if request else "contract"
    difficulty = request.difficulty if request else "easy"
    env = JudicialEnv(domain=domain, difficulty=difficulty)
    obs, info = env.reset()
    return ResetResponse(observation=obs.model_dump(), info=info)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Execute one environment step with the provided action."""
    env = JudicialEnv(domain=request.domain, difficulty=request.difficulty)
    env.reset()
    action = JudicialAction(**request.action)
    obs, reward, done, truncated, info = env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        truncated=truncated,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
def get_state(domain: str = "contract", difficulty: str = "easy"):
    """Return current environment state."""
    env = JudicialEnv(domain=domain, difficulty=difficulty)
    env.reset()
    return StateResponse(state=env.state())


@app.get("/tasks")
def get_tasks():
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "id": "task1_contract",
                "difficulty": "easy",
                "domain": "contract",
                "description": "Contract breach and dispute resolution under Indian Contract Act",
                "expected_baseline": 0.9760,
            },
            {
                "id": "task2_tort",
                "difficulty": "medium",
                "domain": "tort",
                "description": "Tort and negligence cases with conflicting evidence",
                "expected_baseline": 0.6853,
            },
            {
                "id": "task3_property",
                "difficulty": "hard",
                "domain": "property",
                "description": "Property and inheritance disputes with adversarial ambiguous facts",
                "expected_baseline": 0.5520,
            },
        ]
    }


@app.get("/results")
def results():
    """Return inference results after baseline run completes."""
    return JSONResponse(content=RESULTS)


# ─── Inference Runner (background) ───────────────────────────

def log_start(task_name: str):
    print(f"[START] task={task_name}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    print(f"[STEP] step={step} action={action!r} reward={reward:+.2f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


def get_agent_action(obs, client: OpenAI) -> JudicialAction:
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


async def run_task(task_config: dict, client: OpenAI) -> float:
    task_name = task_config["name"]
    log_start(task_name)

    env = JudicialEnv(domain=task_config["domain"], difficulty=task_config["difficulty"])
    obs, _ = env.reset()

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        for step_num in range(1, 4):
            try:
                action = get_agent_action(obs, client)
                obs, reward, done, truncated, info = env.step(action)
                rewards.append(reward)
                steps_taken = step_num
                log_step(step=step_num, action=action.verdict, reward=reward, done=done, error=None)
                if done:
                    obs, _ = env.reset()
            except Exception as e:
                log_step(step=step_num, action="ERROR", reward=0.0, done=False, error=str(e))
                break

        score = sum(rewards) / (len(rewards) * MAX_TOTAL_REWARD) if rewards else 0.001
        score = min(max(score, 0.001), 0.999)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def run_all_tasks():
    global RESULTS
    RESULTS["status"] = "running"

    if not API_KEY:
        print("[WARN] No API key set (HF_TOKEN / GROQ_API_KEY). Skipping inference.", flush=True)
        RESULTS.update({"status": "skipped", "scores": {}, "overall": 0.0})
        return

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    all_scores = {}

    for task in TASKS:
        score = await run_task(task, client)
        all_scores[task["name"]] = round(score, 4)

    overall = sum(all_scores.values()) / len(all_scores)

    print(f"\n=== BASELINE RESULTS ===", flush=True)
    for name, s in all_scores.items():
        print(f"{name}: {s:.4f}", flush=True)
    print(f"OVERALL AVERAGE: {overall:.4f}", flush=True)

    results_data = {
        "status": "complete",
        "scores": all_scores,
        "overall": round(overall, 4),
    }

    try:
        with open("results.json", "w") as f:
            json.dump(results_data, f, indent=2)
    except Exception:
        pass

    RESULTS.update(results_data)


def run_inference_background():
    asyncio.run(run_all_tasks())


@app.on_event("startup")
async def startup_event():
    """Launch inference in background when server starts."""
    thread = threading.Thread(target=run_inference_background, daemon=True)
    thread.start()


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
