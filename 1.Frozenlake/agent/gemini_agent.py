import os
import time
import google.generativeai as genai
from typing import List, Dict, Any

class GeminiAgent:
    """
    A real agent that uses Google's Gemini API to play FrozenLake.
    Includes simple In-Context Learning (ICL) capabilities for 'training'.
    """
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None):
        """
        Initialize the Gemini Agent.
        
        Args:
            model_name (str): The Gemini model to use (default: gemini-1.5-flash).
            api_key (str): Google API Key. If None, checks GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in environment or pass it to init.")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Buffer to store successful trajectories for Few-Shot Prompting
        # This is our simple form of "Training" / Optimization
        self.successful_examples: List[str] = []
        self.max_examples = 3

    def generate(self, prompt: str) -> str:
        """
        Generates an action using the Gemini API.
        Appends successful examples to the prompt if available (Few-Shot).
        """
        
        # 1. Augment Prompt with Examples (ICL)
        augmented_prompt = prompt
        if self.successful_examples:
            examples_str = "\n\n--- SUCCESSFUL EXAMPLES ---\n" + "\n".join(self.successful_examples) + "\n---------------------------\n"
            # Insert examples before the actual current situation if possible, 
            # generally appended to the system prompt or prepended to current history.
            # For simplicity, we prepend to the whole prompt here.
            augmented_prompt = examples_str + "\n" + prompt

        # 2. Call API
        try:
            # We set temperature low for deterministic actions
            response = self.model.generate_content(
                augmented_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    candidate_count=1
                )
            )
            return response.text
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Fallback to a safe action or retry logic
            # For now, return a safe default to avoid crashing the loop
            return "<action>RIGHT</action>"

    def update(self, batch: List[Dict[str, Any]]):
        """
        'Trains' the agent by storing successful episodes as few-shot examples.
        
        Args:
            batch (list): A list of episode dictionaries containing:
                          {'prompt': str, 'response': str, 'score': float, ...}
        """
        for episode in batch:
            if episode.get('score', 0) >= 1.0: # If successful (Won)
                # Format: "Context -> Action" representation
                # Using a simplified summary for the example
                example_text = f"User: ...\nAgent: {episode.get('response', '')}"
                
                if len(self.successful_examples) < self.max_examples:
                    self.successful_examples.append(example_text)
                else:
                    # Replace oldest or random? Let's replace oldest (FIFO)
                    self.successful_examples.pop(0)
                    self.successful_examples.append(example_text)
                    
        print(f"Agent Updated: Now holds {len(self.successful_examples)} successful examples.")

if __name__ == "__main__":
    # verification
    try:
        agent = GeminiAgent()
        print("GeminiAgent initialized successfully.")
    except Exception as e:
        print(f"Initialization skipped (expected if no key): {e}")
