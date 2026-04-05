"""
FastAPI server for the Customer Support OpenEnv environment.
Exposes /reset, /step, /state, /health endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from models import SupportAction, SupportObservation, SupportState
from environment import CustomerSupportEnvironment

app = FastAPI(
    title="Customer Support OpenEnv",
    description="An RL environment for training agents on customer support tasks.",
    version="1.0.0",
)

# One environment instance per session (simple single-session server)
# For multi-session, use a dict keyed by session_id
_envs: dict[str, CustomerSupportEnvironment] = {}


def _get_or_create(session_id: str, task_name: str = "easy") -> CustomerSupportEnvironment:
    if session_id not in _envs:
        _envs[session_id] = CustomerSupportEnvironment(task_name=task_name)
    return _envs[session_id]


@app.get("/health")
def health():
    return {"status": "healthy", "environment": "customer-support-env"}


@app.post("/reset")
def reset(task_name: str = "easy", session_id: str = "default"):
    """Start a new episode. task_name: easy | medium | hard"""
    try:
        env = CustomerSupportEnvironment(task_name=task_name)
        _envs[session_id] = env
        obs = env.reset()
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(action: SupportAction, session_id: str = "default"):
    """Take one action in the environment."""
    if session_id not in _envs:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    env = _envs[session_id]
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str = "default"):
    """Get current episode state."""
    if session_id not in _envs:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return _envs[session_id].state().model_dump()


@app.get("/tasks")
def list_tasks():
    """List available tasks."""
    return {
        "tasks": [
            {"name": "easy", "description": "Classify the ticket only.", "max_steps": 3},
            {"name": "medium", "description": "Classify + respond to the customer.", "max_steps": 5},
            {"name": "hard", "description": "Classify + respond + correct escalation decision.", "max_steps": 7},
        ]
    }
