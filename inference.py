#!/usr/bin/env python3
"""
Inference script for Customer Support OpenEnv.
Prints structured stdout blocks required by the validator:
[START], [STEP], [END]
Makes LLM calls through the injected LiteLLM proxy.
"""

import os
from openai import OpenAI
from environment import CustomerSupportEnvironment
from models import SupportAction


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["API_BASE_URL"],
    )


def make_action_with_llm(client: OpenAI, observation_text: str, step_num: int) -> SupportAction:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a customer support agent in an RL environment. "
                    "Return one short helpful response for the customer issue."
                ),
            },
            {
                "role": "user",
                "content": f"Step {step_num}. Observation: {observation_text}",
            },
        ],
        temperature=0,
        max_tokens=60,
    )

    text = (response.choices[0].message.content or "").strip()

    if step_num == 1:
        return SupportAction(action_type="classify", content="general_support")

    return SupportAction(
        action_type="respond",
        content=text[:200] if text else "Thank you for contacting support. I will help with your request.",
    )


def main():
    task_name = "easy"
    print(f"[START] task={task_name}", flush=True)

    client = get_client()
    env = CustomerSupportEnvironment(task_name=task_name)
    observation = env.reset()

    total_reward = 0.0
    done = False
    step_num = 0
    max_steps = 5

    while not done and step_num < max_steps:
        step_num += 1
        observation_text = str(observation.model_dump())

        action = make_action_with_llm(client, observation_text, step_num)
        observation, reward, done, info = env.step(action)
        total_reward += float(reward)

        print(f"[STEP] step={step_num} reward={reward} done={done}", flush=True)

    score = max(0.0, min(1.0, total_reward))
    print(f"[END] task={task_name} score={score:.2f} steps={step_num}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[END] task=easy score=0.00 steps=0 error={type(e).__name__}", flush=True)
        raise