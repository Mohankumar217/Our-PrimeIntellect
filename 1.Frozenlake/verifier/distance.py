from typing import Any, List
import math

def manhattan_distance_reward(episode_history: List[Any], final_outcome: str) -> float:
    """
    Rewards the agent based on how close it got to the goal (Manhattan Distance).
    
    Formula: 
    - Max distance on 4x4 map from (0,0) to (3,3) is 3+3 = 6.
    - We want a reward that increases as distance decreases.
    - If goal reached (dist=0), reward is maximized.
    
    However, this verifier is called at the END of the episode.
    If the agent fell in a hole, we calculate distance from that hole to goal.
    
    Reward = 1.0 - (final_distance / max_distance)
    Range: [0.0, 1.0]
    """
    
    # Grid properties
    GOAL_POS = (3, 3)
    MAX_DIST = 6.0 # abs(0-3) + abs(0-3)
    
    # 1. Get Final Position
    if not episode_history:
        return 0.0
        
    # Assuming episode_history[-1] contains the final state/observation
    # We need to extract position. 
    # The 'episode_history' format depends on how it's passed from rubric.calculate_score.
    # In train_loop_updated.py: actions_list is passed as episode_history.
    # Wait, rubric.calculate_score signature is (actions_list, outcome, text_list, parser).
    # This verifier signature is (episode_history, final_outcome).
    # 'actions_list' only has strings like "RIGHT". It does NOT have positions.
    
    # CRITICAL FIX: The Rubric interface in frozenlake.py passes 'episode_history' which is 'actions_list'.
    # This is insufficient to calculate distance. 
    # We generally need the TRAJECTORY (state/obs) to get position.
    
    # However, 'frozenlake_updated.py' handles the environment.
    # Let's look at 'train_loop_updated.py' again.
    # It passes: score = env.rubric.calculate_score(actions_list, ...)
    
    # To fix this properly without breaking interface:
    # We should allow 'episode_history' to be the full trajectory dicts if possible, 
    # OR we re-simulate (too hard),
    # OR we rely on 'text_history' if it contains coordinates (unreliable).
    
    # BEST APPROACH:
    # Update 'train_loop_updated.py' to pass the FULL trajectory to calculate_score.
    # The 'episode_data["trajectory"]' contains 'state_msg'.
    # But 'frozenlake.py' type hint says List[Any].
    
    # For now, let's assume we will update 'train_loop_updated.py' to pass the FULL trajectory objects
    # as the first argument, instead of just actions_list.
    # The full trajectory items have: {"state_msg": ..., "action": ..., "outcome_msg": ...}
    
    # But wait, we need coordinates.
    # 'state_msg' is a string. Parsing it is messy.
    # 'FrozenLakeWorld' observation has 'position': (r, c).
    # 'train_loop_updated.py' does NOT store the raw observation dict in trajectory currently!
    # It stores 'state_msg' which is obs['message'].
    
    # STEP: We must enable 'train_loop_updated.py' to store the full 'obs' or at least 'position'.
    
    # Let's handle the extraction gracefully here assuming 'episode_history' will contain dicts with 'position'.
    # If not present, return 0.
    
    final_step = episode_history[-1]
    
    # Check if it has 'position' directly (if we update loop to store valid obs)
    if isinstance(final_step, dict) and 'position' in final_step:
        pos = final_step['position']
    elif hasattr(final_step, 'position'):
        pos = final_step.position
    else:
        # Fallback: Can't calculate
        return 0.0
        
    dist = abs(pos[0] - GOAL_POS[0]) + abs(pos[1] - GOAL_POS[1])
    
    # Normalize
    reward = 1.0 - (dist / MAX_DIST)
    return max(0.0, reward)
