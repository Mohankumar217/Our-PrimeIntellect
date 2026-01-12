import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

class QwenAgent:
    """
    A local/offline agent that uses the Qwen-7B model (via HuggingFace Transformers).
    Requires 'transformers', 'torch', 'accelerate' installed.
    """
    def __init__(self, model_path: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "auto"):
        """
        Initialize the Qwen Agent.
        
        Args:
            model_path (str): Path to local model or HuggingFace repo ID.
                              Use 'Qwen/Qwen2.5-0.5B-Instruct' or a local path.
            device (str): 'cuda', 'cpu', or 'auto'.
        """
        print(f"Loading Qwen model from {model_path} (Device: {device})...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).eval()
        
        # Buffer for In-Context Learning
        self.successful_examples: List[str] = []
        self.max_examples = 3
        print("Qwen Agent Ready.")

    def generate(self, prompt: str) -> str:
        """
        Generates an action using the local Qwen model.
        """
        
        # 1. Augment Prompt with Examples
        augmented_prompt = prompt
        if self.successful_examples:
            examples_str = "\n\n--- SUCCESSFUL EXAMPLES ---\n" + "\n".join(self.successful_examples) + "\n---------------------------\n"
            augmented_prompt = examples_str + "\n" + prompt

        # 2. Prepare inputs
        # Qwen-Chat models expect specific chat formatting usually, but for base interaction
        # or simple completions, raw prompting works if the system prompt is clear.
        # However, it's safer to use the 'chat' format if it is a Chat model.
        # We'll assume raw text generation for flexibility here.
        inputs = self.tokenizer(augmented_prompt, return_tensors="pt").to(self.model.device)
        
        # 3. Generate
        # Define stop token (we want to stop before "Observation:")
        # We also need to handle EOS token.
        # Simple stopping via string checking in loop is often easier for custom implementations,
        # but transformers generate supports 'stopping_criteria'.
        # For simplicity with this model, let's just generate and then truncate.
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60, 
                temperature=0.3, # Lower temperature for more stability
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # 4. Decode
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Post-processing truncation (Stop Sequence logic)
        stop_marker = "Observation:"
        if stop_marker in response:
            response = response.split(stop_marker)[0]
            
        return response.strip()

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
                    
        print(f"Qwen Agent Updated: Now holds {len(self.successful_examples)} successful examples.")

if __name__ == "__main__":
    # verification stub
    print("This file contains the QwenAgent class. Run train_loop.py to use it.")
