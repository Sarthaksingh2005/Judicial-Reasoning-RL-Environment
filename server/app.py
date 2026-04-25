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
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from dotenv import load_dotenv
from openai import OpenAI

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import JudicialEnv, JudicialAction
from server.models import (
    ResetRequest, StepRequest,
    ResetResponse, StepResponse, StateResponse, HealthResponse, AIJudgeResponse,
    EscalateRequest, ChatRequest, ChatResponse,
    SummonsRequest, CaseStatusRequest
)

load_dotenv()

# ─── Configuration ────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY      = os.environ.get("GROQ_API_KEY", "")

# Fallback to Hugging Face if Groq is blocked or missing, using DeepSeek!
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    API_BASE_URL = "https://api-inference.huggingface.co/v1/"
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    API_KEY = HF_TOKEN
    print(f"OK: Hugging Face DeepSeek Model active via HF_TOKEN")
elif not API_KEY:
    print("WARNING: No API keys set. AI Judge will use offline demo mode.")
else:
    print(f"OK: GROQ_API_KEY loaded ({API_KEY[:8]}...)")

MAX_TOTAL_REWARD       = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    {"name": "task1_contract", "domain": "contract", "difficulty": "easy"},
    {"name": "task2_tort",     "domain": "tort",     "difficulty": "medium"},
    {"name": "task3_property", "domain": "property", "difficulty": "hard"},
    {"name": "task4_petty_crime", "domain": "petty_crime", "difficulty": "hard"},
]

# Global results store
RESULTS: dict = {"status": "starting", "scores": {}, "overall": 0.0}

ESCALATED_CASES = []


# ─── FastAPI App ──────────────────────────────────────────────

app = FastAPI(
    title="Judicial Reasoning RL Environment",
    description="An RL environment where an LLM agent acts as a judge over Indian legal cases.",
    version="1.0.0",
)


ui_dir = os.path.join(os.path.dirname(__file__), "ui")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(ui_dir, "index.html"), headers={"Cache-Control": "no-cache, no-store"})

@app.get("/styles.css", include_in_schema=False)
def styles(v: str = None):
    return FileResponse(os.path.join(ui_dir, "styles.css"), media_type="text/css", headers={"Cache-Control": "no-cache, no-store"})

@app.get("/script.js", include_in_schema=False)
def script(v: str = None):
    return FileResponse(os.path.join(ui_dir, "script.js"), media_type="application/javascript", headers={"Cache-Control": "no-cache, no-store"})


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
    
    if request and request.custom_facts:
        obs.fact_pattern = request.custom_facts
        obs.case_id = "USR-CUSTOM"
        if request.custom_evidence:
            obs.evidence_flags = request.custom_evidence
            
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


@app.post("/ai_judge", response_model=AIJudgeResponse)
def ai_judge(request: ResetRequest):
    """Generate an AI judgment for the requested domain/difficulty and evaluate it."""
    env = JudicialEnv(domain=request.domain, difficulty=request.difficulty)
    obs, _ = env.reset()

    # Override with custom user facts
    if request.custom_facts:
        obs.fact_pattern = request.custom_facts
        obs.case_id = "USR-CUSTOM"
        if request.custom_evidence:
            obs.evidence_flags = request.custom_evidence
            
        # Mock env.current_case so env.step() evaluation doesn't crash
        env.current_case["case_id"] = "USR-CUSTOM"
        env.current_case["fact_pattern"] = request.custom_facts
        env.current_case["evidence"] = request.custom_evidence or []
        env.current_case["gold_label_verdict"] = "forward_to_judge" if request.domain == "petty_crime" else "liable"
        env.current_case["expert_verdict"] = env.current_case["gold_label_verdict"]
        env.current_case["precedents"] = []

    if not API_KEY:
        # ─── Offline Demo Mode ─────────────────────────────────────────────────
        # Returns a realistic mock judgment using the user's actual facts when offline.
        is_criminal = obs.domain == "petty_crime"
        
        # Use the actual custom facts the user provided so context is not lost!
        facts_preview = (obs.fact_pattern[:200] + '...') if len(obs.fact_pattern) > 200 else obs.fact_pattern
        
        mock_action = JudicialAction(
            verdict          = "forward_to_judge" if is_criminal else "liable",
            confidence_score = 0.91,
            reasoning_chain  = (
                "[COUNCIL OF AI MAJORITY VOTE: 3/3 AGREED — OFFLINE FALLBACK]\n\n"
                f"Agent 1 (Fact Analyst): The user's case states: \"{facts_preview}\". This establishes a prima facie grievance.\n\n"
                f"Agent 2 (Legal Expert): Evaluating the provided facts against statutory law. The evidence supports the complainant's claim.\n\n"
                f"Agent 3 (Chief Justice): I concur. The defendant has failed in their legal obligations as established by the facts presented."
            ),
            cited_precedents = ["P001", "P002"] if not is_criminal else ["P-BNS-101"],
            ratio_decidendi  = (
                "When a party refuses to return a security deposit or breaches an agreement as stated in the facts, they are liable under Section 73 of the Indian Contract Act."
                if not is_criminal else
                "Criminal matters established in the facts are forwarded to a human judge per constitutional design."
            ),
            obiter_dicta     = "Parties are advised to attempt mediation before further legal proceedings.",
            refer_to_human_judge = is_criminal,
            case_status      = "forwarded_to_judge" if is_criminal else "resolved_by_ai",
        )
        obs_next, reward, done, truncated, info = env.step(mock_action)
        return AIJudgeResponse(
            action=mock_action.model_dump(),
            evaluation=StepResponse(observation=obs_next.model_dump(), reward=reward, done=done, truncated=truncated, info=info)
        )

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    action = get_agent_action(obs, client)
    obs_next, reward, done, truncated, info = env.step(action)
    return AIJudgeResponse(
        action=action.model_dump(),
        evaluation=StepResponse(observation=obs_next.model_dump(), reward=reward, done=done, truncated=truncated, info=info)
    )


@app.post("/escalate")
def escalate_case(request: EscalateRequest):
    ESCALATED_CASES.append(request.model_dump())
    return {"status": "success", "appeal_type": request.appeal_type}

@app.get("/api/escalated-cases")
def get_escalated_cases():
    return {"cases": ESCALATED_CASES}

@app.post("/summons")
def generate_summons(request: SummonsRequest):
    """Generate a Summons Notice for the opposing party."""
    import datetime
    now = datetime.datetime.now()
    summons_id = f"SUM-{now.strftime('%Y%m%d%H%M%S')}"
    return {
        "status": "success",
        "summons_id": summons_id,
        "case_id": request.case_id,
        "issued_to": request.respondent_name,
        "issued_on": now.isoformat(),
        "message": f"Summons Notice {summons_id} generated for {request.respondent_name} in Case {request.case_id}."
    }

@app.post("/case_status")
def get_case_status(request: CaseStatusRequest):
    """Return the current status of a registered case."""
    # In production, this queries a DB. For demo: return mock status.
    return {
        "case_id": request.case_id,
        "status": "under_ai_analysis",
        "status_label": "Under AI Analysis — Council is deliberating",
        "last_updated": "2026-04-25T10:00:00",
        "cause_list": [
            {"step": "Case Registered", "done": True},
            {"step": "KYC Verified", "done": True},
            {"step": "Evidence Uploaded", "done": True},
            {"step": "AI Fact-Finding Complete", "done": True},
            {"step": "Council Judgment", "done": False},
            {"step": "AI Resolution Certificate Issued", "done": False},
        ]
    }

@app.get("/judge", include_in_schema=False)
def judge_dashboard():
    return FileResponse(os.path.join(ui_dir, "judge.html"))

@app.get("/judge.js", include_in_schema=False)
def judge_js():
    return FileResponse(os.path.join(ui_dir, "judge.js"))

@app.post("/chat", response_model=ChatResponse)
def fact_finding_chat(request: ChatRequest):
    """Real LLM-powered fact-finding chat using Groq/Llama-3.3."""
    if not API_KEY:
        return ChatResponse(response="API key not configured. Please add your GROQ_API_KEY to the .env file to enable AI fact-finding.")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    is_criminal = request.case_type == "criminal"

    if is_criminal:
        system_prompt = """You are JusticeEngine-01, an AI paralegal for Indian courts under the new Bharatiya Nyaya Sanhita (BNS).
Your task is to gather facts for a CRIMINAL case through targeted, specific questions.

Rules:
- Ask ONE short, specific question per response.
- Focus on: FIR number, BNS section that may apply, nature of offence, evidence available (CCTV, witnesses, medical reports, vehicle plate), date/time/location, whether police have been contacted.
- Ask if the accused is known or unknown.
- Once you have gathered enough facts (after 5-7 exchanges), say EXACTLY: "DOSSIER_COMPLETE: I have gathered sufficient information. You may now generate the AI Fact Bundle for the Judge."
- NEVER suggest guilt or innocence. You are gathering facts ONLY.
- Keep language simple and respectful."""
    else:
        system_prompt = """You are JusticeEngine-01, an AI legal analyst for Indian courts.
Your task is to gather facts about a CIVIL case through targeted, specific questions.

Rules:
- Ask ONE short, specific question per response.
- Questions should help clarify: written agreements/contracts, timeline of events, evidence available (receipts, messages, photos), witnesses, prior disputes, formal complaints filed, and whether an attempt at out-of-court settlement was made.
- Once you have gathered enough facts (after 4-6 exchanges), say EXACTLY: "DOSSIER_COMPLETE: I have gathered sufficient information. You may now generate the AI Judgment."
- Keep language simple, clear, and professional.
- Do NOT give legal opinions yet. Only gather facts."""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add case context as the first assistant message
    messages.append({
        "role": "assistant",
        "content": f"I am reviewing your case. Here are the facts on file:\n\n{request.fact_pattern}\n\nTo help me build your complete legal dossier, I need to ask you a few targeted questions."
    })
    
    # Add the conversation history
    for msg in request.chat_history:
        role = "assistant" if msg.get("role") == "ai" else "user"
        messages.append({"role": role, "content": msg.get("content", "")})
    
    # Add the latest user message
    if request.user_message:
        messages.append({"role": "user", "content": request.user_message})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=200,
        )
        reply = response.choices[0].message.content.strip()
        return ChatResponse(response=reply)
    except Exception as e:
        err = str(e)[:80]
        return ChatResponse(response=f"I encountered an issue connecting to the AI. Please try again. (Error: {err})")



@app.get("/state", response_model=StateResponse)
def get_state(domain: str = "contract", difficulty: str = "easy"):
    """Return current environment state."""
    env = JudicialEnv(domain=domain, difficulty=difficulty)
    env.reset()
    return StateResponse(state=env.state())


@app.get("/police")
async def serve_police_dashboard():
    return FileResponse(os.path.join(ui_dir, "police_dashboard.html"))


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
            {
                "id": "task4_petty_crime",
                "difficulty": "easy",
                "domain": "petty_crime",
                "description": "Petty crimes using BNS and Constitutional law emphasizing restorative justice",
                "expected_baseline": 0.8500,
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
    is_criminal = obs.domain == "petty_crime"
    
    # Define the 3 Personas for the Council of AIs
    personas = [
        "The Strict Constitutionalist: You adhere strictly to the letter of the law, constitutional provisions, and precedents.",
        "The Empathetic Mediator: You focus on restorative justice, equitable outcomes, and the context of the parties involved.",
        "The Precedent Analyst: You heavily weigh past case outcomes and aim for extreme consistency with previous rulings."
    ]
    
    base_prompt = f"""You are JusticeEngine-01, an AI legal mediator for Indian courts.
FACT PATTERN:
{obs.fact_pattern}

APPLICABLE STATUTES:
{chr(10).join(obs.statutes)}

PRECEDENTS:
{json.dumps(obs.precedents, indent=2)}

EVIDENCE FLAGS:
{', '.join(obs.evidence_flags) if obs.evidence_flags else 'None'}
"""
    
    if is_criminal:
        task_prompt = """
CRITICAL INSTRUCTION: This is a CRIMINAL case. You MUST NOT pass a final judgment or verdict.
Your task is to:
1. Bundle all the facts and evidence logically for a human judge.
2. Identify if this is a punishable offense strictly under the new Bharatiya Nyaya Sanhita (BNS) or BNSS.
3. State the potential punishment range (minimum and maximum).
4. Set the verdict field EXACTLY to "forward_to_judge".
5. State whether the offence is cognizable and bailable or non-bailable.

COURT HIERARCHY RULE: Always cite Supreme Court of India rulings first, then High Court.

Respond ONLY with a valid JSON object:
{
  "verdict": "forward_to_judge",
  "confidence_score": 0.0 to 1.0,
  "reasoning_chain": "Your bundled facts, BNS offense identification, and potential punishment.",
  "cited_precedents": ["case_id_1", "case_id_2"],
  "ratio_decidendi": "The binding legal principle: this act is [punishable/not] under BNS Section [X].",
  "obiter_dicta": "Non-binding observations about the case."
}"""
    else:
        task_prompt = """
CRITICAL INSTRUCTION: This is a CIVIL case. You must analyze the facts and provide a final verdict.
Base your reasoning strictly on the Constitution of India, the precedents provided, and follow court hierarchy:
Supreme Court of India > High Court > Sessions Court > Magistrate Court.

IMPORTANT: If the same fact pattern was decided differently at High Court and Supreme Court, ALWAYS follow the Supreme Court verdict.

You MUST provide:
- ratio_decidendi: The single binding legal principle forming the foundation of your decision.
- obiter_dicta: Non-binding observations made in passing.
- If no prior precedent: label as FRESH CASE.

Respond ONLY with a valid JSON object:
{
  "verdict": "liable OR not_liable OR partial_liability",
  "confidence_score": 0.0 to 1.0,
  "reasoning_chain": "Step by step reasoning referencing Constitution, BNS, and court hierarchy.",
  "cited_precedents": ["case_id_1", "case_id_2"],
  "ratio_decidendi": "The ratio of this case is: [single binding principle].",
  "obiter_dicta": "This court notes, obiter, that [non-binding observation]."
}"""

    votes = []
    
    # Get 3 votes from the 3 personas
    for persona in personas:
        full_prompt = f"{persona}\n\n{base_prompt}\n\n{task_prompt}"
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.3,
                )
                raw = response.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                data = json.loads(raw)
                votes.append(data)
                break
            except Exception as e:
                time.sleep(1)
                
    if not votes:
        raise ValueError("All 3 agents failed to generate a valid response.")

    # Majority Voting Logic
    verdict_counts = {}
    for v in votes:
        verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1

    majority_verdict = max(verdict_counts, key=verdict_counts.get)
    max_votes = verdict_counts[majority_verdict]

    # 3-WAY SPLIT: if all three disagree (each unique verdict), auto-escalate to human judge
    if len(verdict_counts) == len(votes) and max_votes == 1:
        final_reasoning = "[COUNCIL OF AI: 3-WAY SPLIT — No consensus reached. This case is automatically being escalated to a Human Judge as per constitutional design.]\n\nVotes:\n"
        for i, v in enumerate(votes):
            final_reasoning += f"  Agent {i+1} ({personas[i].split(':')[0]}): {v.get('verdict')} — {v.get('reasoning_chain', '')[:100]}...\n"
        return JudicialAction(
            verdict="forward_to_judge",
            confidence_score=0.0,
            reasoning_chain=final_reasoning,
            cited_precedents=[],
            ratio_decidendi="3-way split among AI Council — no binding ratio established.",
            obiter_dicta="Human judicial review is required.",
            refer_to_human_judge=True,
            case_status="forwarded_to_judge"
        )

    # Find the best reasoning chain (the one that matches the majority verdict)
    winning_vote = next((v for v in votes if v["verdict"] == majority_verdict), votes[0])

    # Modify reasoning to show it was a majority vote
    final_reasoning = f"[COUNCIL OF AI MAJORITY VOTE: {max_votes}/3 AGREED]\n\n" + winning_vote.get("reasoning_chain", "")

    return JudicialAction(
        verdict=majority_verdict,
        confidence_score=float(winning_vote.get("confidence_score", 0.8)),
        reasoning_chain=final_reasoning,
        cited_precedents=winning_vote.get("cited_precedents", []),
        ratio_decidendi=winning_vote.get("ratio_decidendi", ""),
        obiter_dicta=winning_vote.get("obiter_dicta", ""),
        case_status="resolved_by_ai" if majority_verdict != "forward_to_judge" else "forwarded_to_judge"
    )


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
