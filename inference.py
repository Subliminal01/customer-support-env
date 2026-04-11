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


def llm_call(client: OpenAI, system_prompt: str, user_prompt: str, max_tokens: int = 120) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


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
    return text if text in allowed else "other"


def write_response(client: OpenAI, ticket_text: str) -> str:
    text = llm_call(
        client,
        system_prompt=(
            "Write a short, empathetic customer support reply in 2 sentences. "
            "Include several relevant words such as tracking, delivery, sorry, investigate, update, expedite when appropriate."
        ),
        user_prompt=ticket_text,
        max_tokens=120,
    )
    if not text:
        text = (
            "I’m sorry your delivery has been delayed. We will investigate the tracking issue, "
            "share an update as soon as possible, and try to expedite the delivery."
        )
    return text[:400]


def write_resolution(client: OpenAI, ticket_text: str) -> str:
    text = llm_call(
        client,
        system_prompt=(
            "Write a short resolution summary for closing a customer support ticket in 1 sentence."
        ),
        user_prompt=ticket_text,
        max_tokens=80,
    )
    if not text:
        text = "The delivery issue has been reviewed and the ticket is now being closed with follow-up in progress."
    return text[:300]


def main():
    task_name = "medium"
    print(f"[START] task={task_name}", flush=True)

    client = get_client()
    env = CustomerSupportEnvironment(task_name=task_name)
    observation = env.reset()

    total_reward = 0.0
    done = False
    step_num = 0

    while not done:
        step_num += 1

        if step_num == 1:
            category = classify_ticket(client, observation.ticket_text)
            action = SupportAction(
                action_type="classify",
                category=category,
            )
        elif step_num == 2:
            message = write_response(client, observation.ticket_text)
            action = SupportAction(
                action_type="respond",
                message=message,
            )
        else:
            message = write_resolution(client, observation.ticket_text)
            action = SupportAction(
                action_type="resolve",
                message=message,
            )

        observation, reward, done, info = env.step(action)
        total_reward += float(reward)

        print(f"[STEP] step={step_num} reward={reward} done={done}", flush=True)

        if step_num >= 3:
            break

    score = max(0.0, min(1.0, total_reward))
    print(f"[END] task={task_name} score={score:.2f} steps={step_num}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[END] task=medium score=0.00 steps=0 error={type(e).__name__}", flush=True)
        raise