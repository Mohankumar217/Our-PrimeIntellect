import os 
import sys

# Adjust path to find sibling directories if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Wrapper.xml_parser import XMLParser

class VLMWrapper:
    """
    Multimodal wrapper for FrozenLake.
    Constructs prompts with images and text instructions.
    Manages the parsing of VLM outputs.
    """
    def __init__(self, renderer):
        self.renderer = renderer
        self.parser = XMLParser(fields={"answer": "action"})
        self.last_distance = None
        
        self.system_prompt = """You are an agent navigating a world. Reach the goal safely.
The world is a 4x4 grid. You can move LEFT, RIGHT, UP, or DOWN.
Dark tiles are HOLES - avoid them!
The Gold tile is the GOAL.
The Red circle is YOU.

OUTPUT FORMAT:
First, think about which action to take based on the image.
Then, output your action inside <action> tags.
Valid actions: UP, DOWN, LEFT, RIGHT.

Example:
<action>RIGHT</action>
"""

    def reset_history(self):
        self.last_distance = None

    def calculate_manhattan(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def build_prompt(self, observation, q_values, current_frame, current_feedback=""):
        """
        Constructs the multimodal prompt with Q-Value Context.
        """
        
        # 1. System Prompt
        prompt_content = [
            {"type": "text", "text": "SYSTEM:\n" + self.system_prompt}
        ]
        
        # 2. Add Q-Value Context (Implicit Memory)
        if q_values:
            memory_text = "\nMEMORY (Action History Scores for Current View):\n"
            # Sort helpfulness for display
            sorted_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
            
            for action, score in sorted_actions:
                annotation = ""
                if score > 0.5: annotation = " (Recommended)"
                elif score < -0.5: annotation = " (Avoid)"
                memory_text += f"- {action}: {score:.2f}{annotation}\n"
                     
            prompt_content.append({"type": "text", "text": memory_text})
            
        # 3. Add Current Observation (Image) + PROXIMITY FEEDBACK
        proximity_msg = ""
        if current_feedback:
            proximity_msg = "\n" + current_feedback

        obs_text = "\nOBSERVATION:\n(See attached image)\n" + proximity_msg
        prompt_content.append({"type": "text", "text": obs_text})
        prompt_content.append({"type": "image", "image": current_frame})
        
        # 4. Add Action Constraint Reminder
        prompt_content.append({"type": "text", "text": "\nOUTPUT FORMAT:\n<action>UP|DOWN|LEFT|RIGHT</action>"})
        
        return prompt_content

    def parse_action(self, model_output):
        """
        Extracts the action from the model's text response.
        """
        return self.parser.parse(model_output)
