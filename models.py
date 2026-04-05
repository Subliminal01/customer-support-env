from typing import Optional, Literal
from pydantic import BaseModel, Field


# ── Actions the agent can take ──────────────────────────────────────────────

class SupportAction(BaseModel):
    """
    The agent submits ONE action per step.

    action_type choices:
      - classify  : label the ticket with a category
      - respond   : write a reply to the customer
      - escalate  : escalate ticket to a human agent (with reason)
      - resolve   : mark ticket as resolved (with resolution summary)
    """
    action_type: Literal["classify", "respond", "escalate", "resolve"] = Field(
        ..., description="The type of action to take"
    )
    category: Optional[Literal["billing", "shipping", "technical", "refund", "other"]] = Field(
        None, description="Required when action_type is 'classify'"
    )
    message: Optional[str] = Field(
        None, description="Required when action_type is 'respond', 'escalate', or 'resolve'"
    )


# ── Observations the agent receives ─────────────────────────────────────────

class CustomerHistory(BaseModel):
    """Summary of the customer's past interactions."""
    total_orders: int
    previous_complaints: int
    account_age_days: int
    is_premium: bool


class SupportObservation(BaseModel):
    """Everything the agent sees at each step."""
    ticket_id: str
    ticket_text: str
    customer_name: str
    customer_history: CustomerHistory
    current_step: int
    max_steps: int
    actions_taken: list[str] = Field(default_factory=list)
    feedback: Optional[str] = Field(
        None, description="Feedback from the environment after each action"
    )
    task_name: str
    task_description: str
    done: bool = False


# ── State (episode metadata) ─────────────────────────────────────────────────

class SupportState(BaseModel):
    """Internal episode state returned by state() endpoint."""
    episode_id: str
    task_name: str
    step_count: int
    is_done: bool
    current_score: float
    classified_as: Optional[str] = None
    has_responded: bool = False
    resolution: Optional[str] = None  # "escalated" | "resolved" | None


# ── Reward ───────────────────────────────────────────────────────────────────

class SupportReward(BaseModel):
    """Breakdown of the reward signal."""
    total: float = Field(..., ge=0.0, le=1.0)
    classification_score: float = Field(0.0, ge=0.0, le=1.0)
    response_score: float = Field(0.0, ge=0.0, le=1.0)
    resolution_score: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = ""
