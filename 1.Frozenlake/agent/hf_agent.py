import os
from huggingface_hub import InferenceClient
from typing import List, Dict, Any

class HuggingFaceAgent:
    """
    An agent that uses the Hugging Face Inference API (Serverless).
    Requires 'huggingface_hub' installed and HF_TOKEN env var.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", api_key: str = None):
        """
        Initialize the Hugging Face Agent.
        
        Args:
            model_name (str): HF Model ID (default: Qwen/Qwen2.5-7B-Instruct).
            api_key (str): HF API Token. If None, checks HF_TOKEN env var.
        """
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError("HF_TOKEN not found. Please set environment variable or pass to init.")
            
        self.model_name = model_name
        self.client = InferenceClient(api_key=self.api_key)
        
        self.successful_examples: List[str] = []
        self.max_examples = 3
        print(f"HuggingFaceAgent initialized with model: {self.model_name}")

    def generate(self, prompt: str) -> str:
        """
        Generates an action using the HF Inference API.
        """
        # 1. Augment Prompt with Examples
        augmented_prompt = prompt
        if self.successful_examples:
            examples_str = "\n\n--- SUCCESSFUL EXAMPLES ---\n" + "\n".join(self.successful_examples) + "\n---------------------------\n"
            augmented_prompt = examples_str + "\n" + prompt

        # 2. Call API
        try:
            # Instruct models on HF Inference API often require chat_completion structure
            messages = [{"role": "user", "content": augmented_prompt}]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=60,
                temperature=0.1, # Low temp for determinism
                stop=["Observation:", "\nObservation"]
            )
            
            # Extract content from ChatCompletionOutput
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error calling HF API: {e}")
            return "<action>RIGHT</action>"

    def update(self, batch: List[Dict[str, Any]]):
        """
        'Trains' the agent by storing successful episodes as few-shot examples.
        """
        for episode in batch:
            if episode.get('score', 0) >= 1.0:
                example_text = f"User: ...\nAgent: {episode.get('response', '')}"
                
                if len(self.successful_examples) < self.max_examples:
                    self.successful_examples.append(example_text)
                else:
                    self.successful_examples.pop(0)
                    self.successful_examples.append(example_text)
                    
        if len(batch) > 0 and batch[0].get('score', 0) >= 1.0:
             print(f"Agent Updated: Now holds {len(self.successful_examples)} successful examples.")

if __name__ == "__main__":
    print("Run train_loop.py --agent hf to use this.")
