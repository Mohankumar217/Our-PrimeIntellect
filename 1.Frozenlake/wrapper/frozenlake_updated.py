import sys
import os
import math

# Adjust path to import original modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from World.frozenlake_world import FrozenLakeWorld
from World.frozenlake_world import FrozenLakeWorld
from wrapper.frozenlake import XMLParser, Rubric, reached_goal, fell_in_hole, step_efficiency
from verifier.outcome import hit_wall
from verifier.delta import distance_delta_reward

# --- 1. Strategic/Causal Feedback ---

def get_manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def causal_feedback(observation, previous_pos, current_pos, goal_pos=(3,3), grid_map=None):
    """
    Generates feedback based on causal reasoning (Distance Delta + Risk).
    """
    if observation['outcome'] == 'hole':
        return f"{observation['message']} CRITICAL FAILURE: You stepped onto a generic Hole. This strategy is fatal."
    
    if observation['outcome'] == 'goal':
        return f"{observation['message']} SUCCESS: You reached the goal."

    if current_pos == previous_pos:
         # Hit a wall or stayed still
         return f"{observation['message']} CAUTION: Action had no effect (hit wall?). Wasted step."

    # Distance Delta
    prev_dist = get_manhattan_distance(previous_pos, goal_pos)
    curr_dist = get_manhattan_distance(current_pos, goal_pos)
    delta = prev_dist - curr_dist
    
    dist_msg = ""
    if delta > 0:
        dist_msg = "Good. You moved CLOSER to the goal."
    elif delta < 0:
        dist_msg = "Bad. You moved AWAY from the goal."
    else:
        dist_msg = "Neutral move."

    # Risk Assessment (Lookahead - simplified: just state current safety)
    # Ideally, we would look at adjacent tiles, but let's stick to the current state info.
    # The agent knows 'tile' type.
    
    return f"{observation['message']} Analysis: {dist_msg} Current tile is Safe."

# --- 2. Upgraded Environment Container ---

class FrozenLakeEnvironmentUpdated:
    def __init__(self, world, parser, rubric, system_prompt_template):
        self.world = world
        self.parser = parser
        self.rubric = rubric
        self.base_system_prompt = system_prompt_template
        self.current_system_prompt = system_prompt_template
        self.previous_pos = world.agent_pos

    def reset(self):
        self.previous_pos = self.world.start_pos # Reset tracker
        return self.world.reset()

    def step(self, action):
        observation = self.world.step(action)
        return observation

    def feedback(self, observation, current_pos=None):
        # We need current position to calculate delta.
        # The observation from world has 'position'
        curr = observation.get('position', self.world.agent_pos)
        
        msg = causal_feedback(
            observation, 
            self.previous_pos, 
            curr, 
            goal_pos=(self.world.rows-1, self.world.cols-1), # Assume bottom-right goal
            grid_map=self.world.grid_map
        )
        
        self.previous_pos = curr
        return msg

    def evolve_system_prompt(self, trajectory_memory):
        """
        Updates the system prompt with lessons from memory.
        """
        lessons = trajectory_memory.get_lessons()
        evolution_block = f"""
        
--- EVOLVED STRATEGY GUIDANCE ---
Based on previous successful survivors, adhere to these successful patterns:
{lessons}
---------------------------------
"""
        self.current_system_prompt = self.base_system_prompt + evolution_block
        return self.current_system_prompt

# --- 3. Loader ---

BASE_SYSTEM_PROMPT = """You are an RL agent playing FrozenLake 4x4.
Your goal is to reach the Goal (G) from the Start (S) without falling into Holes (H).

THE MAP LAYOUT (Use coordinates):
- Row 0: S (Start), F, F, F
- Row 1: F, H (Hole!), F, F
- Row 2: F, F, F, H (Hole!)
- Row 3: H (Hole!), F, F, G (Goal)

COORDINATE LOGIC:
- You start at (0, 0).
- Goal is at (3, 3).
- DANGER: Do NOT go to (1, 1). That is a HOLE.
- DANGER: Do NOT go to (2, 3). That is a HOLE.
- DANGER: Do NOT go to (3, 0). That is a HOLE.

Output instructions:
1. First, PLAN your move inside <thought> tags. Analyze your current position and adjacent tiles. Check if they are safe or holes.
   - CHECK DANGER: (1,1), (2,3), (3,0).
2. Return ONLY the action tag: <action>...</action>.
   - You have only four actions: LEFT, RIGHT, UP, DOWN.
3. Do NOT hallucinate an <observation> tag.
4. Do NOT output a list of actions.
5. Do NOT output the current state or coordinates as final text.

Example Valid Output:
<thought>
I am at (0,0).
Candidates:
- RIGHT -> (0,1): Safe.
- DOWN -> (1,0): Safe.
SAFETY CHECK:
- (0,1) is NOT a hole.
- (1,0) is NOT a hole.
I will move RIGHT.
</thought>
<action>RIGHT</action>
"""

import re

class RobustParser:
    """
    A loose parser that extracts action keywords from free text.
    Replaces the strict XML requirement for smaller models.
    """
    def __init__(self, fields=None):
        self.fields = fields

    def parse(self, text: str):
        # Look for actions (case-insensitive)
        match = re.search(r"\b(LEFT|RIGHT|UP|DOWN)\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    def format_reward(self, text: str) -> float:
        return 1.0 if self.parse(text) is not None else 0.0

def load_environment_updated(grid_map=None, **kwargs):
    # Curriculum: Support variable grid_map passed in
    world = FrozenLakeWorld(grid_map=grid_map) 
    
    # Switch to RobustParser
    parser = RobustParser()
    
    rubric = Rubric()
    rubric.add_verifier(reached_goal, weight=2.0) # Bonus for goal to distinguish from just moving close
    rubric.add_verifier(fell_in_hole, weight=1.0) # -1.0
    rubric.add_verifier(hit_wall, weight=1.0)      # -1.0
    rubric.add_verifier(distance_delta_reward, weight=1.0) # +/- 0.5
    
    return FrozenLakeEnvironmentUpdated(
        world=world,
        parser=parser,
        rubric=rubric,
        system_prompt_template=BASE_SYSTEM_PROMPT
    )
