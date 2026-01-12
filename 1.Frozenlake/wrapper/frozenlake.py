import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from ..env.frozenlake_world import FrozenLakeWorld
    from ..verifier import reached_goal, fell_in_hole, step_efficiency
except ImportError:
    # Fallback for testing/standalone execution
    import sys
    import os
# --- 1. System Prompt ---

DEFAULT_SYSTEM_PROMPT = """You are an RL agent playing FrozenLake 4x4.
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

Output your action in XML format: <action>LEFT</action>, <action>RIGHT</action>, <action>UP</action>, or <action>DOWN</action>.
Choose the safest path to (3,3).
"""


# --- 2. Parser (Verifiers Mock) ---

class XMLParser:
    """
    Parses XML-formatted actions from LLM output.
    Design strictly follows Prime Intellect verifiers.XMLParser pattern.
    """
    def __init__(self, fields: Dict[str, str]):
        """
        Args:
            fields: A dict mapping field names to their expected tag names.
                    e.g. {"answer": "action"}
        """
        self.fields = fields
        # In this specific case, we care about the 'answer' field which maps to <action>
        self.tag_name = fields.get("answer", "action")

    def parse(self, text: str) -> Optional[str]:
        """
        Extracts the content of the action tag.
        Returns None if not found or malformed.
        """
        pattern = f"<{self.tag_name}>(.*?)</{self.tag_name}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def format_reward(self, text: str) -> float:
        """
        Returns 1.0 if the format is correct (tag exists), 0.0 otherwise.
        """
        return 1.0 if self.parse(text) is not None else 0.0


# --- 3. Feedback Function ---

def feedback_function(observation: dict) -> str:
    """
    Converts the raw world observation into a human-readable string.
    
    Args:
        observation: The dict returned by FrozenLakeWorld.step()
                     keys: position, tile, terminated, outcome, message
    
    Returns:
        str: The message to be shown to the LLM.
    """
    # The world already provides a decent message, but we enforce the format here
    # to ensure no internal types are leaked.
    return str(observation.get("message", "Unknown state."))


# --- 4. Rubric ---

class Rubric:
    """
    Combines multiple verifiers into a final score.
    """
    def __init__(self):
        self.verifiers: List[Tuple[Callable, float]] = []

    def add_verifier(self, verifier_func: Callable, weight: float = 1.0):
        self.verifiers.append((verifier_func, weight))

    def calculate_score(self, episode_history: List[Any], final_outcome: str, text_history: List[str], parser: XMLParser) -> float:
        """
        Computes the weighted sum of verifier scores.
        
        Args:
            episode_history: List of steps taken.
            final_outcome: The 'outcome' string from the world.
            text_history: List of raw text outputs from the LLM.
            parser: The parser instance (for format rewards).
        """
        total_score = 0.0
        
        # 1. format_reward 
        # The prompt implied adding "format_reward" using parser's built-in.
        # We can account for it if we added a specific verifier for it,
        # or we could hardcode it here. For strict extensibility, we loop through added verifiers.
        
        for verifier, weight in self.verifiers:
            # We assume verifiers match the signature (episode_history, final_outcome).
            # If the verifier needs more context (like text_history), we might need a more complex signature check.
            # But the provided verifiers only use (episode_history, final_outcome).
            score = verifier(episode_history, final_outcome)
            total_score += score * weight

        return total_score

# --- 5. Environment Loader ---

class FrozenLakeEnvironment:
    """
    A container for the environment components.
    This mimics the structure returned by a PI-style loader.
    """
    def __init__(self, world, system_prompt, parser, feedback_fn, rubric):
        self.world = world
        self.system_prompt = system_prompt
        self.parser = parser
        self.feedback = feedback_fn
        self.rubric = rubric

    def reset(self):
        return self.world.reset()

    def step(self, action):
        return self.world.step(action)


def load_environment(grid_size: int = 4, seed: int = 0, **kwargs) -> FrozenLakeEnvironment:
    """
    Loads the FrozenLake environment with all associated components.
    
    Args:
        grid_size (int): Size of the grid (default 4).
        seed (int): Random seed (unused for deterministic world, but kept for API compatibility).
    
    Returns:
        FrozenLakeEnvironment: The configured environment object.
    """
    # 1. Instantiate World
    world = FrozenLakeWorld()  # Default 4x4
    
    # 2. Setup Parser
    parser = XMLParser(fields={"answer": "action"})
    
    # 3. Setup Rubric
    rubric = Rubric()
    rubric.add_verifier(reached_goal, weight=1.0)
    rubric.add_verifier(fell_in_hole, weight=1.0)
    rubric.add_verifier(step_efficiency, weight=1.0)
    
    # 4. Return Container
    return FrozenLakeEnvironment(
        world=world,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        parser=parser,
        feedback_fn=feedback_function,
        rubric=rubric
    )
