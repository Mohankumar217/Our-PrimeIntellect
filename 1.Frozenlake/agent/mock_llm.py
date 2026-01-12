import random

class MockLLMAgent:
    """
    A Mock Agent that simulates an LLM's behavior.
    It takes a text prompt and returns an XML-formatted action.
    """
    def __init__(self, policy="random"):
        self.policy = policy
        self.actions = ["LEFT", "RIGHT", "UP", "DOWN"]

    def generate(self, prompt: str) -> str:
        """
        Simulates the LLM generation process.
        
        Args:
            prompt (str): The input prompt (system prompt + history).
            
        Returns:
            str: The generated text containing the XML action.
        """
        # In a real scenario, this would call OpenAI/Anthropic API or a local model.
        
        if self.policy == "random":
            action = random.choice(self.actions)
        else:
            action = "RIGHT" # Fixed policy for testing
            
        # Simulate some "thought process" chain of thought (optional)
        response = f"""
I need to reach the goal. 
I am at a safe tile.
I will move {action}.
<action>{action}</action>
"""
        return response
