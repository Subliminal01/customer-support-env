"""
inference.py — Baseline inference script for Customer Support OpenEnv.

Reads credentials from environment variables:
  API_BASE_URL  : LLM API endpoint
  MODEL_NAME    : model identifier
  HF_TOKEN      : API key

Runs the OpenAI client against all 3 tasks and emits structured logs.
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are a customer support agent AI. 
You will receive a customer support ticket and must handle it step by step.

Available actions (respond with a JSON object):
1. Classify: {"action_type": "classify", "category": "<billing|shipping|technical|refund|other>"}
2. Respond:  {"action_type": "respond", "message": "<your response to customer>"}
3. Escalate: {"action_type": "escalate", "message": "<reason for escalation>"}
4. Resolve:  {"action_type": "resolve", "message": "<resolution summary>"}

Rules:
- Always classify first
- For medium/hard tasks: classify, then respond, then escalate or resolve
- For hard tasks: if customer is very frustrated or issue is complex, escalate
- Respond with ONLY valid JSON, no extra text
"""


def call_env(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    url = f"{ENV_BASE_URL}{endpoint}"
    if method == "POST":
        r = requests.post(url, json=data, timeout=30)
    else:
        r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def get_agent_action(observation: dict, history: list) -> dict:
    """Ask the LLM what action to take given the current observation."""
    ticket_context = f"""
Ticket ID: {observation['ticket_id']}
Customer: {observation['customer_name']}
Message: {observation['ticket_text']}

Customer History:
- Total orders: {observation['customer_history']['total_orders']}
- Previous complaints: {observation['customer_history']['previous_complaints']}
- Premium customer: {observation['customer_history']['is_premium']}

Task: {observation['task_description']}
Step: {observation['current_step']}/{observation['max_steps']}
Actions taken so far: {observation['actions_taken']}
Last feedback: {observation.get('feedback', '')}

What is your next action? Respond with JSON only.
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": ticket_context},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=256,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()

    # Clean up markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


def run_task(task_name: str) -> dict:
    """Run one full episode for a given task. Returns result dict."""
    session_id = f"inference-{task_name}"

    # Reset
    obs = call_env(f"/reset?task_name={task_name}&session_id={session_id}", method="POST")

    print(json.dumps({
        "event": "START",
        "task": task_name,
        "ticket_id": obs["ticket_id"],
        "customer": obs["customer_name"],
    }))

    history = []
    total_reward = 0.0
    done = False
    step_num = 0

    while not done:
        step_num += 1
        try:
            action = get_agent_action(obs, history)
        except Exception as e:
            print(json.dumps({"event": "ERROR", "step": step_num, "error": str(e)}))
            break

        # Record in conversation history
        history.append({"role": "assistant", "content": json.dumps(action)})

        # Step environment
        result = call_env(f"/step?session_id={session_id}", method="POST", data=action)
        obs      = result["observation"]
        reward   = result["reward"]
        done     = result["done"]
        info     = result.get("info", {})
        total_reward = reward if done else total_reward

        print(json.dumps({
            "event": "STEP",
            "task": task_name,
            "step": step_num,
            "action_type": action.get("action_type"),
            "reward": reward,
            "done": done,
            "feedback": obs.get("feedback", ""),
        }))

        if done:
            break

    print(json.dumps({
        "event": "END",
        "task": task_name,
        "total_steps": step_num,
        "final_reward": total_reward,
        "score_breakdown": info,
    }))

    return {"task": task_name, "reward": total_reward, "info": info}


def main():
    print(json.dumps({"event": "START", "model": MODEL_NAME, "tasks": TASKS}))

    results = []
    for task_name in TASKS:
        try:
            result = run_task(task_name)
            results.append(result)
        except Exception as e:
            print(json.dumps({"event": "ERROR", "task": task_name, "error": str(e)}))
            results.append({"task": task_name, "reward": 0.0, "error": str(e)})
        time.sleep(1)

    # Summary
    avg = sum(r["reward"] for r in results) / len(results)
    print(json.dumps({
        "event": "END",
        "summary": results,
        "average_reward": round(avg, 4),
    }))


if __name__ == "__main__":
    main()
