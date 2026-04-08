---
title: Customer Support OpenEnv
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Customer Support OpenEnv 🎧

An RL environment for training agents on customer support tasks. Deployed as FastAPI server on Hugging Face Spaces.

## 🚀 Live Demo

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue?logo=huggingface)](https://huggingface.co/spaces/Subliminal01/customer-support-env)

## 📋 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/health` | Health check |
| `POST` | `/reset`  | Start new episode |
| `POST` | `/step`   | Take an action |
| `GET`  | `/state`  | Current state |
| `GET`  | `/tasks`  | List all tasks |

## 🛠️ Local Setup

```bash
git clone https://github.com/Subliminal01/customer-support-env
cd customer-support-env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
