from typing import Any, List

def reached_goal(episode_history: List[Any], final_outcome: str) -> float:
    """
    Returns 1.0 if the agent reached the goal, 0.0 otherwise.
    """
    return 1.0 if final_outcome == "goal" else 0.0

def fell_in_hole(episode_history: List[Any], final_outcome: str) -> float:
    """
    Returns -1.0 if the agent fell in a hole, 0.0 otherwise.
    """
    return -1.0 if final_outcome == "hole" else 0.0
