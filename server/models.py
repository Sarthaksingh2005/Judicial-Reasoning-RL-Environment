"""
Pydantic models for the Judicial Reasoning RL Environment HTTP API.
Team ALACRITY — OpenEnv Hackathon
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ResetRequest(BaseModel):
    domain: str = "contract"
    difficulty: str = "easy"


class StepRequest(BaseModel):
    domain: str = "contract"
    difficulty: str = "easy"
    action: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any] = {}


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    truncated: bool = False
    info: Dict[str, Any] = {}


class StateResponse(BaseModel):
    state: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "ok"
