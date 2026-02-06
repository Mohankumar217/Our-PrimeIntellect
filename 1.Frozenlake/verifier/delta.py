from typing import Any, List

def distance_delta_reward(episode_history: List[Any], final_outcome: str) -> float:
    """
    Rewards/Penalizes based on whether the agent moved closer to or further from the goal.
    
    Scheme:
    - Moved Closer (Delta > 0): +0.5
    - Moved Away (Delta < 0): -0.5
    - No Change (Delta = 0): 0.0
    
    We sum this up for the entire trajectory.
    """
    GOAL_POS = (3, 3)
    total_reward = 0.0
    
    # Need at least 2 steps to compare, or 1 step + start pos.
    # episode_history contains our step records.
    # Format: { "position": (r,c), ... }
    
    if not episode_history:
        return 0.0
        
    # We need to reconstruct the path including start (0,0)
    # If episode_history[0] is the result of first step from (0,0).
    
    previous_pos = (0, 0) # Start
    
    for step in episode_history:
        current_pos = step.get('position')
        
        if not current_pos:
            continue
            
        # Manhattan Distances
        prev_dist = abs(previous_pos[0] - GOAL_POS[0]) + abs(previous_pos[1] - GOAL_POS[1])
        curr_dist = abs(current_pos[0] - GOAL_POS[0]) + abs(current_pos[1] - GOAL_POS[1])
        
        delta = prev_dist - curr_dist # Positive if closer
        
        if delta > 0:
            total_reward += 0.5
        elif delta < 0:
            total_reward -= 0.5
        
        previous_pos = current_pos
        
    return total_reward
