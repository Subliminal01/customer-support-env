#!/usr/bin/env python3
"""
Inference script for Customer Support OpenEnv.
Prints structured stdout blocks required by the validator:
[START], [STEP], [END]
"""

from environment import CustomerSupportEnvironment
from models import SupportAction


def make_action(step_num: int) -> SupportAction:
    if step_num == 1:
        return SupportAction(action_type="classify", content="general_support")
    return SupportAction(
        action_type="respond",
        content="Thank you for contacting support. I will help with your request."
    )


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
        action = make_action(step_num)

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