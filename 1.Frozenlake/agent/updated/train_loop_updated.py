import sys
import os
import argparse
import time

# Ensure path visibility (Add Root Project Dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from wrapper.frozenlake_updated import load_environment_updated
from agent.updated.trajectory_memory_updated import TrajectoryMemory
from agent.updated.qwen_agent_updated import QwenAgentUpdated
from verifier.outcome import reached_goal, fell_in_hole, hit_wall
from verifier.delta import distance_delta_reward

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
        # print(env.current_system_prompt) # REMOVED: Too verbose
        response = agent.generate(env.current_system_prompt, prompt)
        
        # 5. Parse
        action_text = env.parser.parse(response)
        
        step_record = {
            "state_msg": current_msg,
            "response": response,
            "action": action_text if action_text else "INVALID",
            "action": action_text if action_text else "INVALID",
            "outcome_msg": "", # Populated after step
            "position": obs.get("position") 
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
            # 5.5 Calculate Immediate Reward (Step-wise)
            # We reconstruct a mini-history [step_record] just for this step's delta, 
            # or pass the relevant info.
            # Verifiers expect a list. 
            step_history = [step_record] 
            
            # Note: Delta needs context of previous position.
            # Step record has 'position'. We need 'previous_pos' which is handled in env but not explicitly here.
            # Actually, 'step_record' has 'position' (current). 
            # 'env.previous_pos' tracks prior. 
            # We can re-use env.previous_pos logic but that's internal.
            # Better: In 'env.step', we get observation. 
            # Let's rely on the environment's feedback or just calculate cleanly here.
            
            # Simplified: Call verifiers directly with robust checking
            r_wall = hit_wall(step_history, obs['outcome'])
            r_hole = fell_in_hole(step_history, obs['outcome'])
            r_goal = reached_goal(step_history, obs['outcome'])
            
            # Delta is tricky without history. 
            # However, env.feedback() updates its internal state.
            # We can approximate delta reward by manually calculating dist change.
            # OR we can just add a 'reward' field to the observation in the wrapper? 
            # NO, wrapper shouldn't leak reward unless we standardized it.
            
            # Let's calculate delta simple here:
            prev_pos = env.previous_pos # Accessed via instance (it was updated in feedback call?)
            # Wait, env.feedback() updates previous_pos. It was called slightly below in original code.
            # Let's move feedback call UP or handle delta manually.
            
            # Let's calculate Delta manually for clarity:
            # We need PRE-move position.
            # 'obs' is POST-move.
            # The agent WAS at... we didn't store it explicitly in a var, but we can infer.
            # Actually, let's just use the `distance_delta_reward` which handles history.
            # If we pass `[prev_step, current_step]`, it works.
            
            # Hack: We stored `episode_data["trajectory"]` so far.
            # But the Current Step is not appended yet.
            
            # Let's append first?
            # step_record["outcome_msg"] = feedback_msg (not yet)
            
            # Fix: We'll calculate Reward AFTER feedback updates.
            pass
        else:
            # 6. Step
            next_obs = env.step(action_text)
            
            # 7. Feedback (Causal)
            feedback_msg = env.feedback(next_obs) # This updates internal prev_pos
            
            step_record["outcome_msg"] = feedback_msg
            if "position" not in step_record or step_record["position"] is None:
                 step_record["position"] = next_obs.get("position")

            # --- Calculate Immediate Reward ---
            # We construct a 2-step history for Delta: [Previous (if exists), Current]
            history_for_reward = []
            if len(episode_data["trajectory"]) > 0:
                history_for_reward.append(episode_data["trajectory"][-1])
            history_for_reward.append(step_record)
            
            # Calculate components
            # Note: hit_wall uses 'outcome_msg' which we just set.
            # fell_in_hole uses 'final_outcome' (obs['outcome'])
            
            r_wall = hit_wall([step_record], next_obs['outcome'])
            r_hole = fell_in_hole([step_record], next_obs['outcome'])
            r_goal = reached_goal([step_record], next_obs['outcome'])
            r_delta = distance_delta_reward(history_for_reward, next_obs['outcome'])
            
            step_reward = r_wall + r_hole + r_goal + r_delta
            step_record["reward"] = step_reward
            
            episode_data["trajectory"].append(step_record)
            
            if verbose:
                pos = obs.get("position", "Unknown")
                print(f"[Step {step_count} @ {pos}] Action: {action_text} >> {feedback_msg}")

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
    
    score = env.rubric.calculate_score(episode_data["trajectory"], episode_data["final_outcome"], texts_list, env.parser)
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
        # Always update Q-Table with experience
        memory.update_q_table(ep_data)

        # Only add valid runs (score > 0). Storing failures (score <= 0) would reward "Fast Suicide".
        if ep_data['score'] > 0:
            print(f"*** SUCCESS! Saving Episode (Score: {ep_data['score']}) ***")
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
