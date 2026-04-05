"""
CustomerSupportEnvironment — core logic.
Implements reset() / step() / state() per OpenEnv spec.
"""

import uuid
from typing import Optional
from models import SupportAction, SupportObservation, SupportState, SupportReward, CustomerHistory
from tasks import TASKS, Task, compute_reward


class CustomerSupportEnvironment:

    def __init__(self, task_name: str = "easy"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS.keys())}")
        self.task: Task = TASKS[task_name]
        self._episode_id: str = ""
        self._step_count: int = 0
        self._done: bool = False

        # Track what the agent has done this episode
        self._classified_as: Optional[str] = None
        self._response_message: Optional[str] = None
        self._resolution_action: Optional[str] = None
        self._resolution_message: Optional[str] = None
        self._actions_taken: list[str] = []
        self._last_feedback: Optional[str] = None
        self._current_score: float = 0.0

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self) -> SupportObservation:
        """Start a fresh episode. Returns initial observation."""
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._classified_as = None
        self._response_message = None
        self._resolution_action = None
        self._resolution_message = None
        self._actions_taken = []
        self._last_feedback = None
        self._current_score = 0.0

        return self._make_observation(feedback="New episode started. Read the ticket and take your first action.")

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: SupportAction) -> tuple[SupportObservation, float, bool, dict]:
        """
        Process one agent action.
        Returns (observation, reward, done, info)
        """
        if self._done:
            return self._make_observation(feedback="Episode already done."), 0.0, True, {}

        self._step_count += 1
        feedback, step_reward = self._apply_action(action)
        self._last_feedback = feedback

        # Episode ends when: agent resolves/escalates OR max steps reached
        if action.action_type in ("resolve", "escalate"):
            self._done = True
        elif self._step_count >= self.task.max_steps:
            self._done = True
            feedback += " [Max steps reached — episode ending.]"

        # Compute full reward when done
        if self._done:
            episode_state = {
                "classified_as": self._classified_as,
                "response_message": self._response_message,
                "resolution_action": self._resolution_action,
                "resolution_message": self._resolution_message,
            }
            total_reward, breakdown = compute_reward(self.task, episode_state)
            self._current_score = total_reward
            reward = total_reward
            info = breakdown
        else:
            reward = step_reward
            info = {}

        obs = self._make_observation(feedback=feedback)
        return obs, reward, self._done, info

    # ── state ────────────────────────────────────────────────────────────────

    def state(self) -> SupportState:
        """Return current episode metadata."""
        return SupportState(
            episode_id=self._episode_id,
            task_name=self.task.name,
            step_count=self._step_count,
            is_done=self._done,
            current_score=self._current_score,
            classified_as=self._classified_as,
            has_responded=self._response_message is not None,
            resolution=self._resolution_action,
        )

    # ── internal helpers ─────────────────────────────────────────────────────

    def _apply_action(self, action: SupportAction) -> tuple[str, float]:
        """Apply the action and return (feedback_message, immediate_reward)."""

        if action.action_type == "classify":
            if self._classified_as is not None:
                return "Already classified. Move on to the next action.", 0.0
            if not action.category:
                return "classify requires a 'category' field.", 0.0
            self._classified_as = action.category
            self._actions_taken.append(f"classify:{action.category}")
            return f"Ticket classified as '{action.category}'.", 0.1

        elif action.action_type == "respond":
            if self._classified_as is None:
                return "Please classify the ticket before responding.", 0.0
            if not action.message:
                return "respond requires a 'message' field.", 0.0
            self._response_message = action.message
            self._actions_taken.append("respond")
            return "Response recorded. Now decide: escalate or resolve?", 0.1

        elif action.action_type == "escalate":
            if not action.message:
                return "escalate requires a 'message' explaining why.", 0.0
            self._resolution_action = "escalate"
            self._resolution_message = action.message
            self._actions_taken.append("escalate")
            return "Ticket escalated to human agent.", 0.0  # full reward computed at end

        elif action.action_type == "resolve":
            if not action.message:
                return "resolve requires a 'message' summarizing the resolution.", 0.0
            self._resolution_action = "resolve"
            self._resolution_message = action.message
            self._actions_taken.append("resolve")
            return "Ticket marked as resolved.", 0.0  # full reward computed at end

        return "Unknown action type.", 0.0

    def _make_observation(self, feedback: Optional[str] = None) -> SupportObservation:
        ticket = self.task.ticket
        history = ticket.customer_history
        return SupportObservation(
            ticket_id=ticket.ticket_id,
            ticket_text=ticket.text,
            customer_name=ticket.customer_name,
            customer_history=CustomerHistory(**history),
            current_step=self._step_count,
            max_steps=self.task.max_steps,
            actions_taken=list(self._actions_taken),
            feedback=feedback or self._last_feedback,
            task_name=self.task.name,
            task_description=self.task.description,
            done=self._done,
        )
