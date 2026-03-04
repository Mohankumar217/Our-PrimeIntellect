VLM TRAINING PLAN: Q-LEARNING INTEGRATION FOR VISION-LANGUAGE MODELS

Executive Summary

This document outlines a comprehensive plan for training Vision-Language Model (VLM) and Large Language Model (LLM) agents on the FrozenLake environment using Q-Learning integration. The approach combines the strengths of modern foundation models (reasoning, in-context learning) with classical reinforcement learning (value function approximation, reward-based optimization).

Key Philosophy: The agent remains frozen (no gradient updates to model weights). Learning happens via:
1. Q-Table Memory - Stores state-action values learned from experience
2. In-Context Learning (ICL) - Successful trajectories injected into prompts
3. Prompt Evolution - System prompts updated to reflect learned strategies

This is a Prime-Intellect style approach: Selection-based evolution + memory augmentation, NOT fine-tuning.

---

1. CURRENT STATE: EXISTING IMPLEMENTATIONS

1.1 LLM Training System (Baseline)

Location: 1.Frozenlake/agent/updated/train_loop_updated.py

Architecture:
- Agent: Qwen2.5 LLM (text-only)
- Memory: Hybrid system
    - Q-Table: {state_coords: {action: Q-value}}
    - Trajectory Memory: Top-K successful episodes for ICL
- Learning Loop:
    Episode → Experience → Q-Update → Trajectory Selection → Memory Storage

Key Features:
- Q-Learning updates with Bellman equation
- In-Context Learning from successful episodes
- Prompt evolution every N episodes
- Reward shaping (goal, hole, wall, distance delta)
- Selection-based memory (only store score > 0)

Limitations:
- Text-only modality (no visual perception)
- Requires symbolic state (coordinates exposed)

---

1.2 VLM System (Visual Learning)

Location: VLM/Client/run_agent.py

Architecture:
- Agent: Mock VLM (placeholder for Qwen-VL, GPT-4o, Gemini-Vision)
- Memory: Q-Table only {(row, col): {action: Q-value}}
- Input: RGB frames + text prompt with Q-values
- Learning Loop:
    Frame → VLM Reasoning → Action → Q-Update → Memory Storage

Key Features:
- Multimodal prompting (image + text)
- Q-value injection in prompt context
- Proximity-based reward shaping
- Visual state representation (no symbolic leak)

Limitations:
- No trajectory memory or ICL
- Mock model (random actions, not real VLM)
- Simple reward structure

---

1.3 VLM2 System (Pure Video Learning)

Location: VLM2/

Architecture:
- Agent: Heuristic (goal-seeking + danger avoidance)
- Memory: Trajectory-based experiences (no Q-table)
- Input: RGB frames only (strict video-only constraint)
- Learning: Selection-based (keep successes, novel failures)

Key Features:
- Pure computer vision perception
- No symbolic state access
- Experiential memory (situation → lesson)
- Modular architecture (7 components)

Limitations:
- No VLM integration yet
- No Q-learning or value function
- Heuristic agent (not learning-based)

---

2. PROPOSED ARCHITECTURE: UNIFIED VLM Q-LEARNING SYSTEM

2.1 System Design

Combine strengths of all three systems:

[VLM Q-LEARNING AGENT]

MULTIMODAL INPUT PREPARATION
• Current Frame (RGB Image)
• Observation (Text Description)
• Q-Values for Current State
• Top-K Successful Trajectories (ICL)
• Evolved System Prompt

VISION-LANGUAGE MODEL (VLM)
Input: Image + Text Prompt
Output: <thought>...</thought><action>UP/DOWN/LEFT/RIGHT</action>
Models: Qwen-VL, GPT-4o, Gemini-Vision

ACTION EXECUTION
• Parse action from XML
• Execute in environment
• Render new frame
• Calculate reward

DUAL MEMORY UPDATE
1. Q-TABLE UPDATE (Q-Learning)
     Q(s,a) ← Q(s,a) + α[r + γ max Q' - Q]
2. TRAJECTORY MEMORY (Selection)
     • If success → ALWAYS store
     • If score > threshold → Store
     • Keep top-K by fitness

PROMPT EVOLUTION (Periodic)
• Analyze memory patterns
• Generate new strategy hints
• Update system prompt

---

2.2 Core Components

A. Multimodal Prompt Builder

Function: Construct prompts that combine visual and textual information with learned knowledge.

Prompt Structure:

SYSTEM PROMPT (Evolved):
You are an agent navigating a FrozenLake grid.
[... evolved strategies from successful episodes ...]
Goal: Reach the green tile (G), avoid red holes (H).
Output format: <thought>reasoning</thought><action>DIRECTION</action>

=== IN-CONTEXT LEARNING (Top-3 Successes) ===
Episode 1 (Fitness: 0.95):
    Step 1: [Position (0,0)] Action: RIGHT → Moved closer
    Step 2: [Position (0,1)] Action: DOWN → Reached Goal!

[... more examples ...]
===============================================

CURRENT STATE:
[Attached Image: current_frame.png]

Observation: You are at position (1,1). The goal is visible to the south-east.

MEMORY Q-VALUES for current position:
    • UP: -0.3 (likely wall or hole)
    • DOWN: 0.8 (high value, leads toward goal)
    • LEFT: 0.1 (neutral)
    • RIGHT: 0.5 (moderate value)

Based on the image, memory, and Q-values, choose your next action.

Implementation (vlm_q_wrapper.py):

def build_multimodal_prompt(
        frame: Image,
        observation: str,
        q_values: Dict[str, float],
        icl_examples: List[Dict],
        system_prompt: str
) -> Dict:
        """Constructs multimodal VLM prompt."""
        
        # 1. System context
        prompt_text = system_prompt + "\n\n"
        
        # 2. ICL examples
        if icl_examples:
                prompt_text += "=== SUCCESSFUL MEMORIES ===\n"
                for ep in icl_examples:
                        prompt_text += format_trajectory(ep) + "\n"
                prompt_text += "=" * 30 + "\n\n"
        
        # 3. Current state
        prompt_text += f"CURRENT STATE:\n{observation}\n\n"
        
        # 4. Q-Values
        prompt_text += "MEMORY Q-VALUES:\n"
        for action, value in sorted(q_values.items(), key=lambda x: -x[1]):
                prompt_text += f"  • {action}: {value:.2f}\n"
        
        prompt_text += "\nChoose your action wisely."
        
        return {
                "image": frame,
                "text": prompt_text
        }

---

B. Hybrid Memory System

Function: Maintain both Q-table (value function) and trajectory library (successful strategies).

Q-Table Memory:
- Structure: {state_key: {action: Q-value}}
- State Key:
    - Option 1: Coordinates (row, col) - simpler but symbolic
    - Option 2: Visual hash hash(frame_pixels) - pure visual but more complex
    - Recommendation: Start with coordinates, migrate to visual hash later
- Update Rule: Standard Q-learning
    Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') - Q(s,a)]
- Hyperparameters:
    - Learning rate α = 0.1 (slow, stable updates)
    - Discount γ = 0.9 (high foresight)

Trajectory Memory:
- Structure: List of episode dictionaries
    {
        "trajectory": [
            {"state_msg": "...", "action": "RIGHT", "outcome_msg": "...", "reward": 0.1},
            ...
        ],
        "fitness": 0.87,
        "steps": 5,
        "final_outcome": "goal"
    }
- Selection Criteria:
    - Always store: Success episodes (reached goal)
    - Conditionally store: High-score episodes (score > threshold)
    - Discard: Failures, timeouts
- Fitness Function:
    fitness = (goal_bonus * 10 + efficiency_bonus - penalties) / max_possible_score
- Pruning: Keep top-K=20 by fitness

Implementation (vlm_q_memory.py):

class VLMQMemory:
        def __init__(self, q_table_path="q_table.json", trajectory_path="trajectories.json", k=20):
                self.q_table = {}  # {state: {action: value}}
                self.trajectories = []  # Top-K episodes
                self.k = k
                self._load()
        
        def get_q_values(self, state):
                """Retrieve Q-values for state."""
                return self.q_table.get(str(state), {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0})
        
        def update_q(self, state, action, reward, next_state, done):
                """Q-learning update step."""
                alpha, gamma = 0.1, 0.9
                current_q = self.get_q_values(state).get(action, 0.0)
                max_next_q = max(self.get_q_values(next_state).values()) if not done else 0.0
                new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
                
                state_key = str(state)
                if state_key not in self.q_table:
                        self.q_table[state_key] = {}
                self.q_table[state_key][action] = round(new_q, 4)
                self._save_q_table()
        
        def add_trajectory(self, episode_data):
                """Add successful trajectory, maintain top-K."""
                episode_data['fitness'] = calculate_fitness(episode_data)
                self.trajectories.append(episode_data)
                self.trajectories.sort(key=lambda x: x['fitness'], reverse=True)
                self.trajectories = self.trajectories[:self.k]
                self._save_trajectories()
        
        def get_top_k(self, k=5):
                """Retrieve top-K trajectories for ICL."""
                return self.trajectories[:k]

---

C. Reward Structure

Function: Provide meaningful learning signals that balance goal achievement with efficiency.

Reward Components:

Event: Goal Reached
Reward: +10.0
Rationale: Primary objective, large positive

Event: Fell in Hole
Reward: -10.0
Rationale: Terminal failure, large negative

Event: Hit Wall / Invalid
Reward: -0.1
Rationale: Wasted action, small penalty

Event: Moved Closer to Goal
Reward: +0.1
Rationale: Distance shaping, encourage progress

Event: Moved Away from Goal
Reward: -0.1
Rationale: Distance shaping, discourage regression

Distance Calculation:

def calculate_manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def distance_reward(prev_pos, new_pos, goal_pos):
        prev_dist = calculate_manhattan_distance(prev_pos, goal_pos)
        new_dist = calculate_manhattan_distance(new_pos, goal_pos)
        
        if new_dist < prev_dist:
                return 0.1  # Closer
        elif new_dist > prev_dist:
                return -0.1  # Farther
        else:
                return 0.0  # Same (likely wall)

Total Step Reward:

step_reward = goal_bonus + hole_penalty + wall_penalty + distance_reward

---

D. Prompt Evolution System

Function: Automatically improve system prompts based on memory patterns.

Evolution Triggers:
- Every N episodes (e.g., N=5)
- After significant memory growth (e.g., +10 new trajectories)

Evolution Process:
1. Analyze Memory:
     - Extract common successful patterns
     - Identify frequent failure modes
     - Calculate Q-value statistics

2. Generate Strategy Hints:

def extract_strategy_hints(memory):
        hints = []
        
        # Analyze successful trajectories
        for ep in memory.get_top_k(k=10):
                # Find common action patterns
                actions = [step['action'] for step in ep['trajectory']]
                if actions.count('RIGHT') > actions.count('LEFT'):
                        hints.append("Favor moving RIGHT when possible")
        
        # Analyze Q-values
        high_value_states = find_high_value_states(memory.q_table)
        hints.append(f"High-value states found at: {high_value_states}")
        
        return hints

3. Update System Prompt:

def evolve_system_prompt(base_prompt, hints):
        evolved = base_prompt + "\n\n=== LEARNED STRATEGIES ===\n"
        for i, hint in enumerate(hints, 1):
                evolved += f"{i}. {hint}\n"
        evolved += "=" * 30 + "\n"
        return evolved

Example Evolved Prompt:
You are an agent navigating FrozenLake. Goal: reach green tile, avoid holes.

=== LEARNED STRATEGIES ===
1. From memory: Moving RIGHT from start position has 85% success rate
2. Avoid DOWN action at position (1,1) - leads to hole
3. Best path observed: RIGHT → DOWN → RIGHT → DOWN (4 steps)
4. When stuck, try alternating RIGHT and DOWN movements
==============================

---

3. TRAINING ALGORITHM

3.1 Training Loop Pseudocode

def train_vlm_agent(vlm_model, env, episodes=100):
        """
        Main training loop for VLM agent with Q-learning.
        """
        # 1. Initialize
        memory = VLMQMemory()
        renderer = FrozenLakeRenderer()
        system_prompt = load_base_prompt()
        
        for episode_id in range(episodes):
                # 2. Reset episode
                obs = env.reset()
                frame = renderer.render(env)
                trajectory = []
                
                # 3. Episode loop
                while not obs['terminated'] and len(trajectory) < MAX_STEPS:
                        # 3a. Get current state
                        state = obs['position']  # or hash(frame) for visual
                        
                        # 3b. Retrieve memory
                        q_values = memory.get_q_values(state)
                        icl_examples = memory.get_top_k(k=3)
                        
                        # 3c. Build multimodal prompt
                        prompt = build_multimodal_prompt(
                                frame=frame,
                                observation=obs['message'],
                                q_values=q_values,
                                icl_examples=icl_examples,
                                system_prompt=system_prompt
                        )
                        
                        # 3d. VLM inference
                        response = vlm_model.generate(prompt)
                        action = parse_action(response)  # Extract from <action>...</action>
                        
                        if not action:
                                # Invalid format penalty
                                continue
                        
                        # 3e. Execute action
                        prev_state = state
                        obs = env.step(action)
                        new_frame = renderer.render(env)
                        
                        # 3f. Calculate reward
                        reward = calculate_reward(obs, prev_state, state)
                        
                        # 3g. Update Q-table
                        memory.update_q(
                                state=prev_state,
                                action=action,
                                reward=reward,
                                next_state=obs['position'],
                                done=obs['terminated']
                        )
                        
                        # 3h. Record step
                        trajectory.append({
                                'state_msg': obs['message'],
                                'action': action,
                                'response': response,
                                'reward': reward,
                                'outcome_msg': obs.get('feedback', '')
                        })
                        
                        # 3i. Update for next iteration
                        frame = new_frame
                
                # 4. Episode end - evaluate
                episode_data = {
                        'trajectory': trajectory,
                        'steps': len(trajectory),
                        'final_outcome': obs['outcome'],
                        'score': sum(step['reward'] for step in trajectory)
                }
                
                # 5. Memory selection
                if episode_data['final_outcome'] == 'goal' or episode_data['score'] > SCORE_THRESHOLD:
                        memory.add_trajectory(episode_data)
                        print(f"✓ Episode {episode_id}: SUCCESS stored (score={episode_data['score']:.2f})")
                else:
                        print(f"✗ Episode {episode_id}: Discarded (score={episode_data['score']:.2f})")
                
                # 6. Prompt evolution (every 5 episodes)
                if (episode_id + 1) % 5 == 0:
                        hints = extract_strategy_hints(memory)
                        system_prompt = evolve_system_prompt(BASE_PROMPT, hints)
                        print(f"🔄 Prompt evolved with {len(hints)} new strategies")
        
        # 7. Final evaluation
        print("\n=== Training Complete ===")
        print(f"Total successful episodes: {len(memory.trajectories)}")
        print(f"Q-table states covered: {len(memory.q_table)}")
        print_top_strategies(memory)

---

3.2 Hyperparameters & Configuration

Learning Parameters:
LEARNING_RATE = 0.1        # Q-learning α
DISCOUNT_FACTOR = 0.9      # Q-learning γ
EPSILON = 0.0              # No ε-greedy (VLM decides exploration)

Memory Parameters:
MAX_TRAJECTORIES = 20      # Top-K trajectory memory
SCORE_THRESHOLD = 0.5      # Minimum score to store trajectory

Training Configuration:
NUM_EPISODES = 100         # Total training episodes
MAX_STEPS_PER_EPISODE = 20 # Prevent infinite loops
EVOLUTION_INTERVAL = 5     # Episodes between prompt evolution
ICL_EXAMPLES = 3           # Number of examples in prompt

VLM Configuration:
VLM_MODEL = "Qwen/Qwen2-VL-7B-Instruct"  # Or GPT-4o, Gemini-Vision
MAX_TOKENS = 150           # Response length limit
TEMPERATURE = 0.7          # Creativity vs. determinism

---

4. IMPLEMENTATION ROADMAP

Phase 1: Foundation (Weeks 1-2)

Goal: Set up basic VLM Q-learning infrastructure

Tasks:
- 1.1: Create vlm_q_wrapper.py - multimodal prompt builder
- 1.2: Implement vlm_q_memory.py - hybrid memory system
- 1.3: Set up VLM API integration (Qwen-VL, Hugging Face)
- 1.4: Implement reward calculation module
- 1.5: Create basic training loop script

Deliverable: Working training loop with mock VLM (random actions)

---

Phase 2: VLM Integration (Weeks 3-4)

Goal: Replace mock with real VLM model

Tasks:
- 2.1: Configure Qwen-VL API or local inference
- 2.2: Implement image preprocessing pipeline
- 2.3: Test multimodal prompt formatting
- 2.4: Validate XML parsing robustness
- 2.5: Run 10 test episodes with real VLM

Deliverable: Real VLM making decisions based on frames + Q-values

---

Phase 3: Training & Optimization (Weeks 5-6)

Goal: Train agent to solve FrozenLake consistently

Tasks:
- 3.1: Run 100-episode training run
- 3.2: Implement prompt evolution system
- 3.3: Tune hyperparameters (α, γ, reward weights)
- 3.4: Add curriculum learning (easy → hard maps)
- 3.5: Generate training metrics and plots

Success Criteria:
- 70%+ success rate on default 4x4 map
- Q-table coverage > 80% of reachable states
- Average episode length < 10 steps

---

Phase 4: Evaluation & Scaling (Weeks 7-8)

Goal: Validate generalization and scale to harder environments

Tasks:
- 4.1: Evaluate on unseen 8x8 maps
- 4.2: Test zero-shot transfer to new map layouts
- 4.3: Compare vs. baseline (LLM-only, pure Q-learning)
- 4.4: Generate comprehensive evaluation report
- 4.5: Create demonstration videos

Deliverable: Full evaluation report with comparative analysis

---

5. EXPECTED CHALLENGES & SOLUTIONS

Challenge 1: VLM Response Inconsistency

Problem: VLMs may not consistently follow XML format <action>...</action>

Solutions:
- Strict output constraints in system prompt
- Few-shot examples with correct XML formatting
- Retry logic with format correction prompt
- Penalty for invalid formats (reward -= 0.5)
- Parsing fallback: Extract action from natural language if XML fails

---

Challenge 2: Visual Perception Errors

Problem: VLM may misinterpret frame (especially small grids at low resolution)

Solutions:
- High-resolution rendering (cell_size=100+ pixels)
- Color contrast enhancement (distinct agent/goal/hole colors)
- Grid overlay with thin borders for visual clarity
- Text annotation in observation (backup to vision)

---

Challenge 3: Q-Learning Convergence

Problem: Q-values may converge slowly or to suboptimal policies

Solutions:
- Reward shaping with distance heuristics
- Experience replay from trajectory memory
- Learning rate schedule (start α=0.2, decay to 0.05)
- Target network (optional, for stability)

---

Challenge 4: Computational Cost

Problem: VLM inference is expensive ($/call or compute time)

Solutions:
- Local inference with quantized models (Qwen-VL-2B)
- Batched training (accumulate frames, batch inference)
- Episode length limits (max_steps=20)
- Smart caching (reuse Q-values for seen states)

---

6. EVALUATION METRICS

Training Metrics

Per-Episode:
- Success rate (goal reached / total episodes)
- Average episode length (steps)
- Average cumulative reward
- Q-value convergence (mean absolute change)

Memory Metrics:
- Q-table coverage (states with non-zero values / total states)
- Trajectory library size
- Fitness distribution of stored episodes

Learning Curve:
- Success rate over time (rolling average 10 episodes)
- Q-value evolution for key states
- Prompt evolution impact (success rate before vs. after)

---

Final Evaluation

Performance:
- Success rate on default map (4x4)
- Success rate on larger map (8x8)
- Average solution length (steps to goal)
- Optimality gap (vs. optimal policy)

Comparison Baselines:
- Random agent: ~5% success rate
- Heuristic agent (goal-seeking): ~30% success rate
- Pure Q-learning (no VLM): 70-90% success rate
- LLM-only (no vision): 40-60% success rate

Target Performance:
- VLM + Q-learning: 80-95% success rate
- Generalization: 60%+ on unseen maps

---

7. CONCLUSION

This training plan provides a comprehensive blueprint for integrating Vision-Language Models with Q-Learning on the FrozenLake environment. The approach leverages:

1. Visual Grounding: VLMs perceive actual game frames
2. Value Functions: Q-table guides decision-making with learned values
3. In-Context Learning: Successful trajectories bootstrap new episodes
4. Prompt Evolution: System prompts adapt to discovered strategies
5. Selection-Based Memory: Only valuable experiences are retained

Key Innovation: Unlike pure RL (which requires millions of samples) or pure LLM prompting (which lacks grounding), this hybrid approach achieves:
- Sample Efficiency: Learn from dozens, not millions, of episodes
- Interpretability: Q-values and prompts are human-readable
- Adaptability: System prompt evolves to encode strategies
- Zero Fine-Tuning: VLM remains frozen, no gradient updates

Next Steps:
1. Implement Phase 1 components
2. Set up VLM API (Qwen-VL or GPT-4o)
3. Run initial 20-episode pilot
4. Iterate based on results

---

APPENDIX A: Code Templates

A.1 VLM API Client

class VLMClient:
        """Wrapper for Vision-Language Model inference."""
        
        def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
                from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name)
        
        def generate(self, image, text_prompt, max_tokens=150):
                """Generate response from image + text."""
                inputs = self.processor(
                        text=[text_prompt],
                        images=[image],
                        return_tensors="pt"
                )
                
                outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.7
                )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                return response

A.2 Training Script Entry Point

# train_vlm_q.py
import argparse
from vlm_q_trainer import VLMQTrainer

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--episodes', type=int, default=100)
        parser.add_argument('--model', type=str, default='Qwen/Qwen2-VL-7B-Instruct')
        parser.add_argument('--learning-rate', type=float, default=0.1)
        parser.add_argument('--discount', type=float, default=0.9)
        parser.add_argument('--output-dir', type=str, default='./outputs')
        
        args = parser.parse_args()
        
        trainer = VLMQTrainer(
                vlm_model_name=args.model,
                learning_rate=args.learning_rate,
                discount_factor=args.discount
        )
        
        trainer.train(
                num_episodes=args.episodes,
                output_dir=args.output_dir
        )

if __name__ == '__main__':
        main()

---

Document Version: 1.0
Last Updated: 2026-02-06
Status: Ready for Implementation
