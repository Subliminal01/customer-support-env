"""
3 tasks with increasing difficulty, each with a deterministic grader.

Easy   → classify only
Medium → classify + respond
Hard   → classify + respond + correct escalation/resolution decision
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Ticket:
    ticket_id: str
    customer_name: str
    text: str
    true_category: str          # ground truth
    requires_escalation: bool   # ground truth for hard task
    customer_history: dict = field(default_factory=dict)
    good_response_keywords: list = field(default_factory=list)  # keywords a good reply should contain


# ── Sample tickets (one per task for reproducibility) ───────────────────────

TICKETS = {
    "easy": Ticket(
        ticket_id="TKT-001",
        customer_name="Priya Sharma",
        text=(
            "Hi, I was charged twice for my order #45231 last Tuesday. "
            "My bank statement shows two identical charges of ₹2,499. "
            "Please refund the duplicate charge immediately."
        ),
        true_category="billing",
        requires_escalation=False,
        customer_history={
            "total_orders": 5,
            "previous_complaints": 0,
            "account_age_days": 420,
            "is_premium": False,
        },
        good_response_keywords=["refund", "duplicate", "charge", "sorry", "apologize"],
    ),
    "medium": Ticket(
        ticket_id="TKT-002",
        customer_name="Rahul Verma",
        text=(
            "My package was supposed to arrive 5 days ago (order #67890). "
            "The tracking page has shown 'Out for Delivery' for 3 days straight "
            "and nobody has shown up. I need this for an event tomorrow."
        ),
        true_category="shipping",
        requires_escalation=False,
        customer_history={
            "total_orders": 12,
            "previous_complaints": 1,
            "account_age_days": 890,
            "is_premium": True,
        },
        good_response_keywords=["tracking", "delivery", "sorry", "investigate", "update", "expedite"],
    ),
    "hard": Ticket(
        ticket_id="TKT-003",
        customer_name="Ananya Iyer",
        text=(
            "I have been trying to reset my password for 3 days. "
            "The reset email never arrives, I've checked spam. "
            "I called support twice and was told it would be fixed — it hasn't been. "
            "I am unable to access my account which has ₹15,000 in store credit. "
            "This is completely unacceptable. I want to speak to a manager."
        ),
        true_category="technical",
        requires_escalation=True,
        customer_history={
            "total_orders": 31,
            "previous_complaints": 4,
            "account_age_days": 1450,
            "is_premium": True,
        },
        good_response_keywords=[
            "escalate", "manager", "urgent", "sorry", "account", "priority", "credit"
        ],
    ),
}


# ── Task definitions ─────────────────────────────────────────────────────────

@dataclass
class Task:
    name: str
    description: str
    ticket: Ticket
    max_steps: int
    required_actions: list  # actions the agent MUST take to complete the task


TASKS = {
    "easy": Task(
        name="easy",
        description=(
            "Classify the incoming support ticket into the correct category. "
            "Categories: billing, shipping, technical, refund, other. "
            "Use the 'classify' action with the correct category."
        ),
        ticket=TICKETS["easy"],
        max_steps=3,
        required_actions=["classify"],
    ),
    "medium": Task(
        name="medium",
        description=(
            "Classify the ticket AND write a helpful, empathetic response to the customer. "
            "First use 'classify', then use 'respond' with a message addressing their issue."
        ),
        ticket=TICKETS["medium"],
        max_steps=5,
        required_actions=["classify", "respond"],
    ),
    "hard": Task(
        name="hard",
        description=(
            "Classify the ticket, respond to the customer, AND make the correct resolution decision. "
            "Determine whether to 'escalate' (to a human manager) or 'resolve' the ticket yourself. "
            "This customer is frustrated and has a complex issue — choose wisely."
        ),
        ticket=TICKETS["hard"],
        max_steps=7,
        required_actions=["classify", "respond", "escalate"],
    ),
}


# ── Graders ──────────────────────────────────────────────────────────────────

def grade_classification(predicted: Optional[str], true: str) -> float:
    """1.0 for exact match, 0.0 otherwise."""
    if predicted is None:
        return 0.0
    return 1.0 if predicted.lower() == true.lower() else 0.0


def grade_response(message: Optional[str], keywords: list) -> float:
    """
    Score a response based on presence of expected keywords.
    Partial credit: each keyword found adds (1 / total_keywords) to score.
    Min score: 0.0, Max score: 1.0
    """
    if not message or not keywords:
        return 0.0
    message_lower = message.lower()
    hits = sum(1 for kw in keywords if kw.lower() in message_lower)
    # Require at least 2 keywords to get any credit
    if hits < 2:
        return 0.0
    return min(1.0, hits / len(keywords))


def grade_resolution(
    resolution_action: Optional[str],
    requires_escalation: bool,
    message: Optional[str] = None,
) -> float:
    """
    Score the resolution decision.
    - Correct decision (escalate vs resolve): 0.7 base
    - Message is non-empty and meaningful: +0.3
    """
    if resolution_action is None:
        return 0.0

    correct = (
        (resolution_action == "escalate" and requires_escalation) or
        (resolution_action == "resolve" and not requires_escalation)
    )
    base = 0.7 if correct else 0.0
    bonus = 0.3 if (message and len(message) > 20) else 0.0
    return base + bonus


def compute_reward(task: Task, episode_state: dict) -> tuple[float, dict]:
    """
    Compute the final reward for an episode.
    Returns (total_reward, score_breakdown)
    """
    ticket = task.ticket
    breakdown = {
        "classification_score": 0.0,
        "response_score": 0.0,
        "resolution_score": 0.0,
    }

    # Classification (30% weight)
    cls_score = grade_classification(
        episode_state.get("classified_as"),
        ticket.true_category,
    )
    breakdown["classification_score"] = cls_score

    # Response (40% weight)
    resp_score = grade_response(
        episode_state.get("response_message"),
        ticket.good_response_keywords,
    )
    breakdown["response_score"] = resp_score

    # Resolution (30% weight) — only scored in medium/hard
    if task.name in ("medium", "hard"):
        res_score = grade_resolution(
            episode_state.get("resolution_action"),
            ticket.requires_escalation,
            episode_state.get("resolution_message"),
        )
        breakdown["resolution_score"] = res_score

    # Weighted total
    if task.name == "easy":
        total = cls_score  # only classification matters
    elif task.name == "medium":
        total = 0.4 * cls_score + 0.6 * resp_score
    else:  # hard
        total = 0.3 * cls_score + 0.4 * resp_score + 0.3 * breakdown["resolution_score"]

    breakdown["total"] = round(min(1.0, total), 4)
    return breakdown["total"], breakdown
