import argparse
import sys
import os

# Ensure we can import the frozenlake_pi package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from frozenlake_pi.wrapper.frozenlake import load_environment
from frozenlake_pi.agent.mock_llm import MockLLMAgent
from frozenlake_pi.agent.gemini_agent import GeminiAgent
from frozenlake_pi.agent.qwen_agent import QwenAgent
from frozenlake_pi.agent.hf_agent import HuggingFaceAgent

def run_training_episode(env, agent, verbose=False):
    """
    Runs a single episode of interaction between the Environment and the Agent.
    """
    # 1. Reset Environment
    observation = env.reset()
    
    text_history = [] # For verifiers
    episode_history = [] # For verifiers
    
    # Initial Prompt
    current_prompt = env.system_prompt + f"\n\nObservation: {observation['message']}"
    
    step_count = 0
    max_steps = 20
    
    final_outcome = "ongoing"
    
    # Data for training update
    episode_data = {
        'prompt': current_prompt,
        'response': "",
        'score': 0.0
    }
    
    while step_count < max_steps:
        if verbose:
            print(f"--- Step {step_count} ---")
        
        # 2. Agent Generates Action
        response = agent.generate(current_prompt)
        text_history.append(response)
        
        # Keep track of the full interaction for the "response" part if we were doing single-turn optimization,
        # but for multi-turn, we might want the whole conversation.
        # For the simple update() logic in GeminiAgent, let's just use the last response effectively.
        # But actually, GeminiAgent.update() expects a 'response'.
        episode_data['response'] = response # Just capturing the last one for now or we need a better structure suitable for the Agent's update method
        
        if verbose:
            print(f"Agent:\n{response.strip()}")
            
        # 3. Parse Action
        action_text = env.parser.parse(response)
        
        if not action_text:
            if verbose:
                print("Invalid XML format!")
            # Should punish agent, but here we just likely crash or skip
            # In real training, we'd give negative reward and maybe end episode
            feed_msg = "Invalid format. Use <action>...</action>."
            obs = {"message": feed_msg, "outcome": "ongoing", "terminated": False}
        else:
            # 4. Step Environment
            obs = env.step(action_text)
            episode_history.append(obs)
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
    
    if verbose:
        print(f"\nEpisode Finished. Outcome: {final_outcome}. Score: {score}")
        
    return score, episode_data

def train_loop(agent_type="mock", episodes=10):
    """
    Simulates a training loop.
    """
    print(f"Starting Training Simulation for {episodes} episodes using {agent_type} agent...")
    
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
        try:
            agent = HuggingFaceAgent()
        except ValueError as e:
             print(f"Error initializing HuggingFaceAgent: {e}")
             print("Please set HF_TOKEN environment variable.")
             return
    else:
        agent = MockLLMAgent(policy="random")
    
    total_score = 0
    batch_data = []
    
    for i in range(episodes):
        print(f"\n=== Episode {i+1} ===")
        score, episode_info = run_training_episode(env, agent, verbose=True)
        total_score += score
        batch_data.append(episode_info)
        
        # --- TRAINING UPDATE STEP ---
        # Update the agent (e.g. store successful examples for Few-Shot)
        if hasattr(agent, 'update'):
            agent.update([episode_info])
            
        # Add a small delay for API rate limits if real agent
        if agent_type == "gemini":
             import time
             time.sleep(10)
        
    avg_score = total_score / episodes
    print(f"\nTraining Complete.")
    print(f"Average Score: {avg_score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="mock", choices=["mock", "gemini", "qwen", "hf"], help="Agent type to train")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    args = parser.parse_args()
    
    train_loop(agent_type=args.agent, episodes=args.episodes)
