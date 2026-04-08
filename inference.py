#!/usr/bin/env python3
"""
Inference script for Customer Support OpenEnv.
This script prints structured output blocks required by validation:
[START], [STEP], [END]
"""

from environment import CustomerSupportEnvironment
from models import SupportAction


def choose_action(observation: dict, step_num: int) -> SupportAction:
    text = str(observation).lower()

    if "refund" in text:
        return SupportAction(action_type="respond", content="I understand your refund request and will help process it.")
    if "cancel" in text or "subscription" in text:
        return SupportAction(action_type="respond", content="I can help you with subscription cancellation.")
    if "order" in text or "status" in text:
        return SupportAction(action_type="respond", content="Let me check your order status.")
    if "technical" in text or "issue" in text or "error" in text:
        return SupportAction(action_type="respond", content="I understand the technical issue and will help troubleshoot it.")

    if step_num == 1:
        return SupportAction(action_type="classify", content="general_support")
    return SupportAction(action_type="respond", content="Thank you for contacting support. How can I help you further?")


def main():
    task_name = "easy"
    print(f"[START] task={task_name}", flush=True)

    env = CustomerSupportEnvironment(task_name=task_name)
    observation = env.reset()

    total_reward = 0.0
    done = False
    step_num = 0
    max_steps = 5

    while not done and step_num < max_steps:
        step_num += 1

        action = choose_action(observation.model_dump(), step_num)
        observation, reward, done, info = env.step(action)
        total_reward += float(reward)

        print(
            f"[STEP] step={step_num} reward={reward} done={done}",
            fstep_num = 0
     )

    score = max(0.0, min(1.0, total_reward))
    print(
        f"[END] task={task_name} score={score:.2f} steps={step_num}",
        flush=True
    )


if __name__ == "__main__":
    main()