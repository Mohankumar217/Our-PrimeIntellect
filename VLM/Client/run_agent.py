import sys
import os
import random
import json
import time

# Ensure imports work from VLM root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from World.frozenlake_world import FrozenLakeWorld
from World.frozenlake_renderer import FrozenLakeRenderer
from Wrapper.vlm_wrapper import VLMWrapper
from Evaluation.outcome import reached_goal, fell_in_hole
from Evaluation.efficiency import step_efficiency

# Import the new Memory System
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../1.Frozenlake/agent/updated')))
try:
    from trajectory_memory_updated import TrajectoryMemory
except ImportError:
    # Fallback if path is tricky, or just assume it is there due to sys.path
    from trajectory_memory_updated import TrajectoryMemory

MEMORY_FILE = os.path.join(os.path.dirname(__file__), '../Memory/memory.json')

def mock_vlm_model(prompt):
    """
    SIMULATED VLM.
    In a real scenario, this would send 'prompt' (images+text) to an API.
    Here, we return a random valid action to prove the loop works.
    """
    # Simulate processing time
    # time.sleep(0.1) 
    
    # In a real VLM, the prompt contains the image.
    # We would analyze the image to decide.
    # Since we can't see, we guess.
    
    action = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
    
    # Return in XML format as expected
    return f"<thought>I see a grid. I will go {action}.</thought>\n<action>{action}</action>"

def run_agent(episodes=5):
    print("Initializing VLM-Style FrozenLake Agent (Q-Table Memory)...")
    
    # 1. Init Components
    world = FrozenLakeWorld()
    renderer = FrozenLakeRenderer()
    wrapper = VLMWrapper(renderer)
    
    # Initialize Q-Table Memory
    memory = TrajectoryMemory(filepath=MEMORY_FILE)
    
    print(f"Memory Loaded. Knowledge contains {len(memory.q_table)} states.")
    
    for ep in range(episodes):
        print(f"\n--- Episode {ep+1} ---")
        wrapper.reset_history() # Reset Proximity History
        obs_data = world.reset()
        current_frame = renderer.render(world)
        
        trajectory = []
        step_count = 0
        max_steps = 10
        
        last_distance = None
        current_feedback = ""

        # Initial distance
        start_pos = obs_data["position"]
        goal_pos = obs_data["goal_pos"]
        last_distance = wrapper.calculate_manhattan(start_pos, goal_pos)
        
        while not obs_data["terminated"] and step_count < max_steps:
            # 2. Get Q-Values for Current State
            current_pos = obs_data["position"]
            q_values = memory.get_q_values(current_pos)
            
            # 3. Build Prompt (Inject Q-Values)
            prompt = wrapper.build_prompt(obs_data, q_values, current_frame, current_feedback)
            
            # 4. Model Inference
            response = mock_vlm_model(prompt)
            
            # 5. Parse Action
            action = wrapper.parse_action(response)
            
            if not action:
                print("Invalid action format from model.")
                break
                
            # 6. Step
            print(f"Step {step_count}: Action {action}")
            prev_pos = current_pos # Store for update
            obs_data = world.step(action)
            current_frame = renderer.render(world)
            
            # 6a. Calculate Proximity Feedback
            new_pos = obs_data["position"]
            new_distance = wrapper.calculate_manhattan(new_pos, goal_pos)
            
            step_score = 0.0
            if new_distance < last_distance:
                current_feedback = "EVALUATION: Good move. (+0.1 Score) You moved CLOSER."
                step_score = 0.1
            elif new_distance > last_distance:
                current_feedback = "EVALUATION: Bad move. (-0.1 Score) You moved AWAY."
                step_score = -0.1
            else:
                current_feedback = "EVALUATION: Neutral move. (-0.1 Score) Hit wall or same."
                step_score = -0.1
                
            # ADJUST REWARD Logic for Q-Learning
            # Standard FrozenLake is Sparse (0,0,0,1).
            # But we want to reinforce efficiency too.
            q_reward = step_score 
            if obs_data["outcome"] == "goal":
                q_reward = 1.0
            elif obs_data["outcome"] == "hole":
                q_reward = -1.0
                
            # 7. UPDATE MEMORY (Q-Learning Step)
            memory.update_step(prev_pos, action, q_reward, new_pos, obs_data["terminated"])
                
            last_distance = new_distance
            
            # Record
            trajectory.append({
                "observation_msg": obs_data["message"],
                "action": action,
                "outcome_state": obs_data["outcome"],
                "feedback": current_feedback,
                "step_score": step_score
            })
            
            step_count += 1
            
            if obs_data["terminated"]:
                print(f"Terminated: {obs_data['outcome']}")
        
        # End of Episode
        final_outcome = obs_data["outcome"]
        
        # Simple scoring for display
        score = sum(t['step_score'] for t in trajectory)
        if final_outcome == "goal": score += 1.0
        elif final_outcome == "hole": score -= 1.0
            
        print(f"Episode Score: {score:.2f}")
            
    print("\nRun Complete.")

if __name__ == "__main__":
    run_agent()
