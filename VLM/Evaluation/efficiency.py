from typing import Any, List

def step_efficiency(episode_history: List[Any], final_outcome: str) -> float:
    """
    Rewards solving the task in fewer steps.
    Only applied if the goal is reached.
    Score = 1.0 / (number_of_steps + 1)
    """
    if final_outcome != "goal":
        return 0.0
    
    # We count the number of actions taken.
    # Assuming episode_history contains (action, observation) tuples or similar.
    # For this simplified wrapper, we might pass the raw step count if available,
    # or just the length of the history.
    steps = len(episode_history)
    return 1.0 / (steps + 1)
