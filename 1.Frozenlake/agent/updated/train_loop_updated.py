import sys
import os
import argparse
import time

# Ensure path visibility (Add Root Project Dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from wrapper.frozenlake_updated import load_environment_updated
from agent.updated.trajectory_memory_updated import TrajectoryMemory
from agent.updated.qwen_agent_updated import QwenAgentUpdated

def format_trajectory_for_prompt(ep_data):
    """
    Formats a single episode trajectory for In-Context Learning.
    Contains: Step-by-step actions, outcomes, and final result.
    """
    lines = [f"--- Example Episode (Fitness: {ep_data['fitness']:.2f}) ---"]
    for step in ep_data['trajectory']:
        # step keys: state_msg, action, response, outcome_msg
        lines.append(f"Observation: {step['state_msg']}")
        lines.append(f"Action: {step['action']}")
        lines.append(f"Result: {step['outcome_msg']}")
    lines.append(f"Final Outcome: {ep_data['final_outcome']}")
    return "\n".join(lines)

def run_episode(env, agent, memory, verbose=False):
    """
    Runs a single training episode with full trajectory capture.
    """
    # 1. Reset
    obs = env.reset()
    
    # 2. Prepare Episode Container
    episode_data = {
        "trajectory": [],
        "steps": 0,
        "score": 0.0,
        "fitness": 0.0,
        "final_outcome": "ongoing"
    }

    step_count = 0
    max_steps = 5
    
    # 3. Construct In-Context Learning Prompt (ICL) from Memory
    top_k_episodes = memory.get_top_k()
    icl_prompt = ""
    if top_k_episodes:
        icl_prompt = "\n\n=== SUCCESSFULLY RECALLED MEMORIES ===\n"
        for ep in top_k_episodes:
            icl_prompt += format_trajectory_for_prompt(ep) + "\n"
        icl_prompt += "======================================\n"
    
    if verbose:
        print(f"\n--- Episode Start (Memory Size: {len(top_k_episodes)}) ---")

    while step_count < max_steps:
        # Current Observation (Prompt)
        # We append ICL only at the start or keep it in context? 
        # Typically, ICL is part of the 'conversation history' or 'system' context.
        # We'll put it in the User Prompt for visibility to the model this turn.
        
        current_msg = obs["message"]
        prompt = f"{icl_prompt}\nObservation: {current_msg}" if step_count == 0 else f"Observation: {current_msg}"
        
        # 4. Agent Generate
        # System Prompt comes from Env (Evolved)
        print(env.current_system_prompt)
        response = agent.generate(env.current_system_prompt, prompt)
        
        # 5. Parse
        action_text = env.parser.parse(response)
        
        step_record = {
            "state_msg": current_msg,
            "response": response,
            "action": action_text if action_text else "INVALID",
            "outcome_msg": "" # Populated after step
        }
        
        if not action_text:
            step_record["outcome_msg"] = "Invalid Action Format."
            # Penalty handled by Rubric usually, or we skip step
            # For this loop, we treat as no-op or penalty
            next_obs = env.world._get_observation("Invalid format. Use <action>...</action>.") # Internal hack to get msg
            # Logic: just feedback
            # Feedback
            feedback_msg = "Invalid format."
            step_record["outcome_msg"] = feedback_msg
            episode_data["trajectory"].append(step_record)
            if verbose: print(f"Agent Invalid: {response}")
        else:
            # 6. Step
            next_obs = env.step(action_text)
            
            # 7. Feedback (Causal)
            feedback_msg = env.feedback(next_obs) # This updates internal prev_pos too
            
            step_record["outcome_msg"] = feedback_msg
            episode_data["trajectory"].append(step_record)
            
            obs = next_obs
            
            if verbose:
                print(f"Step {step_count}: {action_text} -> {feedback_msg}")

            if obs["terminated"]:
                episode_data["final_outcome"] = obs["outcome"]
                break
        
        step_count += 1

    episode_data["steps"] = step_count
    
    # 8. Calculate Score
    # We reconstruct lists for the rubric
    # Rubric expects (episode_history, final_outcome, text_history, parser)
    # episode_history stored actions
    actions_list = [s["action"] for s in episode_data["trajectory"]]
    texts_list = [s["response"] for s in episode_data["trajectory"]]
    
    score = env.rubric.calculate_score(actions_list, episode_data["final_outcome"], texts_list, env.parser)
    episode_data["score"] = score
    
    if verbose:
        print(f"Episode End. Outcome: {episode_data['final_outcome']}, Score: {score:.2f}, Steps: {step_count}")

    return episode_data

def train_loop(episodes=20, verbose=True):
    print("Initializing Prime-Intellect Upgrade System...")
    
    # 1. Initialize Components
    memory = TrajectoryMemory(filepath="memory.json", k=5)
    agent = QwenAgentUpdated() # Model loading
    # Curriculum: We could randomize map here if needed, keeping default for now
    env = load_environment_updated() 
    
    print(f"Memory loaded with {len(memory.episodes)} episodes.")
    
    for i in range(episodes):
        print(f"\n>>> TRAINING EPISODE {i+1}/{episodes} <<<")
        
        # 2. Run Episode
        ep_data = run_episode(env, agent, memory, verbose=verbose)
        
        # 3. Memory & Selection
        # Only add valid runs (score > 0 implies some success or at least not total fail, 
        # but fitness logic handles sorting. We generally only want to store 'goals' or 'high progress'.)
        # If we store failures, they might push out empty slots?
        # Let's add all, Memory class filters Top-K.
        memory.add_episode(ep_data)
        
        # 4. Evolution (Every 3 episodes)
        if (i + 1) % 3 == 0:
            print("--- EVOLVING SYSTEM PROMPT ---")
            new_prompt = env.evolve_system_prompt(memory)
            # print(f"New Strategy Snippet: ...{new_prompt[-200:]}")

    print("\nTraining Complete.")
    print("Top Memories:")
    for i, ep in enumerate(memory.get_top_k()):
        print(f"{i+1}. Fitness: {ep['fitness']:.2f} | Score: {ep['score']} | Steps: {ep['steps']} | Outcome: {ep['final_outcome']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    
    train_loop(episodes=args.episodes)
