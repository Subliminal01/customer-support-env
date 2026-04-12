from tasks import TASKS, compute_reward


def _score_task(task_name: str, episode_state: dict) -> float:
    task = TASKS[task_name]
    score, _ = compute_reward(task, episode_state)
    return float(score)


def grade_easy(episode_state: dict) -> float:
    return _score_task("easy", episode_state)


def grade_medium(episode_state: dict) -> float:
    return _score_task("medium", episode_state)


def grade_hard(episode_state: dict) -> float:
    return _score_task("hard", episode_state)