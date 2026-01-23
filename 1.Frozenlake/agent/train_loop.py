import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Ensure we can import the local packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wrapper.frozenlake import load_environment
from agent.mock_llm import MockLLMAgent
from agent.gemini_agent import GeminiAgent
from agent.qwen_agent import QwenAgent
# from agent.hf_agent import HuggingFaceAgent

def run_episode(env, agent, verbose=False):
    """
    Runs a single evaluation episode.
    """
    # 1. Reset Environment
    observation = env.reset()
    
    text_history = [] # For verifiers
    episode_history = [] # For verifiers: now stores ACTION STRINGS
    
    # Initial Prompt
    current_prompt = env.system_prompt + f"\n\nObservation: {observation['message']}"
    
    # TODO: Add prompt summarization here if context gets too long
    
    step_count = 0
    max_steps = 20
    
    final_outcome = "ongoing"
    
    # Data for agent update (few-shot memory only)
    episode_data = {
        'prompt': current_prompt,
        'response': "",
        'score': 0.0,
        'outcome': "ongoing"
    }
    
    while step_count < max_steps:
        if verbose:
            print(f"--- Step {step_count} ---")
        
        # 2. Agent Generates Action
        response = agent.generate(current_prompt)
        text_history.append(response)
        
        episode_data['response'] = response 
        
        if verbose:
            print(f"Agent:\n{response.strip()}")
            
        # 3. Parse Action
        action_text = env.parser.parse(response)
        
        if not action_text:
            if verbose:
                print("Invalid XML format!")
            
            feed_msg = "Invalid format. Use <action>...</action>."
            obs = {"message": feed_msg, "outcome": "ongoing", "terminated": False}
        else:
            # 4. Step Environment
            obs = env.step(action_text)
            
            # CRITICAL: Store action string, NOT raw observation
            episode_history.append(action_text)
            
            final_outcome = obs["outcome"]
            
        # 5. Update Prompt
        feedback = env.feedback(obs)
        current_prompt += f"\n\nAction: {response}\nObservation: {feedback}"
        
        if verbose:
            print(f"Env: {feedback}")

        if obs["terminated"]:
            break
            
        step_count += 1

    # 6. Evaluation (Rubric)
    score = env.rubric.calculate_score(episode_history, final_outcome, text_history, env.parser)
    
    episode_data['score'] = score
    episode_data['outcome'] = final_outcome
    
    if verbose:
        print(f"\nEpisode Finished. Outcome: {final_outcome}. Score: {score}")
        
    return score, episode_data

def run_evaluation(agent_type="mock", episodes=10):
    """
    Runs the LLM evaluation loop.
    No gradient updates or backprop are performed here.
    """
    print(f"Starting Evaluation for {episodes} episodes using {agent_type} agent...")
    
    # Load Real Environment
    env = load_environment()
    
    # Load Agent
    if agent_type == "gemini":
        try:
            agent = GeminiAgent()
        except ValueError as e:
            print(f"Error initializing GeminiAgent: {e}")
            print("Please set GOOGLE_API_KEY environment variable.")
            return
    elif agent_type == "qwen":
        try:
            agent = QwenAgent()
        except ImportError:
            print("Error: 'transformers' or 'torch' not found. Please install them:")
            print("pip install torch transformers accelerate")
            return
        except Exception as e:
            print(f"Error initializing QwenAgent: {e}")
            return
    elif agent_type == "hf":
        print("HuggingFaceAgent is currently disabled/missing.")
        return
        # try:
        #     agent = HuggingFaceAgent()
        # except ValueError as e:
        #      print(f"Error initializing HuggingFaceAgent: {e}")
        #      print("Please set HF_TOKEN environment variable.")
        #      return
    else:
        agent = MockLLMAgent(policy="random")
    
    total_score = 0
    wins = 0
    holes = 0
    
    for i in range(episodes):
        print(f"\n=== Episode {i+1} ===")
        score, episode_info = run_episode(env, agent, verbose=True)
        
        total_score += score
        outcome = episode_info['outcome']
        if outcome == "goal":
            wins += 1
        elif outcome == "hole":
            holes += 1
        
        # --- AGENT UPDATE (FEW-SHOT ONLY) ---
        # This does NOT update model weights. It only updates the agent's context/memory.
        if hasattr(agent, 'update'):
            agent.update([episode_info])
            
        # Add a small delay for API rate limits if real agent
        if agent_type == "gemini":
             import time
             time.sleep(10)
        
    avg_score = total_score / episodes
    win_rate = wins / episodes
    hole_rate = holes / episodes
    
    print(f"\nEvaluation Complete.")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Win Rate:      {win_rate:.2%}")
    print(f"Hole Rate:     {hole_rate:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="mock", choices=["mock", "gemini", "qwen", "hf"], help="Agent type to evaluate")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    args = parser.parse_args()
    
    run_evaluation(agent_type=args.agent, episodes=args.episodes)
