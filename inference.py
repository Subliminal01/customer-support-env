#!/usr/bin/env python3

import os
from openai import OpenAI
from environment import CustomerSupportEnvironment
from models import SupportAction


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["API_BASE_URL"],
    )


def llm_call(client: OpenAI, system_prompt: str, user_prompt: str, max_tokens: int = 80) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception:
        return ""


def classify_ticket(client: OpenAI, ticket_text: str) -> str:
    text = llm_call(
        client,
        system_prompt=(
            "Classify this support ticket into exactly one category from: "
            "billing, shipping, technical, refund, other. "
            "Return only the category word."
        ),
        user_prompt=ticket_text,
        max_tokens=10,
    ).lower()

    allowed = {"billing", "shipping", "technical", "refund", "other"}
    if text in allowed:
        return text

    lower = ticket_text.lower()
    if any(word in lower for word in ["charged twice", "duplicate charge", "bank statement", "refund", "charged"]):
        return "billing"
    if any(word in lower for word in ["delivery", "tracking", "package", "shipping"]):
        return "shipping"
    if any(word in lower for word in ["password", "login", "account access", "reset email", "technical"]):
        return "technical"
    return "other"


def main():
    task_name = "easy"
    print(f"[START] task={task_name}", flush=True)

    total_reward = 0.0
    step_num = 0

    try:
        client = get_client()
        env = CustomerSupportEnvironment(task_name=task_name)
        observation = env.reset()

        step_num = 1
        category = classify_ticket(client, observation.ticket_text)

        action = SupportAction(
            action_type="classify",
            category=category,
        )

        observation, reward, done, info = env.step(action)
        total_reward += float(reward)

        print(f"[STEP] step={step_num} reward={reward} done={done}", flush=True)

        score = max(0.0, min(1.0, total_reward))
        print(f"[END] task={task_name} score={score:.2f} steps={step_num}", flush=True)

    except Exception as e:
        print(f"[END] task={task_name} score=0.00 steps={step_num} error={type(e).__name__}", flush=True)


if __name__ == "__main__":
    main()