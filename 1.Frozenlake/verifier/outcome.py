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

def hit_wall(episode_history: List[Any], final_outcome: str) -> float:
    """
    Penalizes hitting a wall.
    Iterates through history to count wall hits.
    Reward: -1.0 per wall hit.
    """
    penalty = 0.0
    for step in episode_history:
        # Check outcome_msg or state_msg for "hit a wall"
        # In train loop, step has 'outcome_msg'.
        msg = step.get('outcome_msg', '')
        if "hit a wall" in msg:
            penalty -= 1.0
            
    return penalty
