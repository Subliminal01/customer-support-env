"""
FastAPI server for the Customer Support OpenEnv environment.
Exposes /reset, /step, /state, /health endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
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
        _envs[session_id
