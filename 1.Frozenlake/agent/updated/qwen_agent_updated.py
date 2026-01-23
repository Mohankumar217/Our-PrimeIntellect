import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

class QwenAgentUpdated:
    """
    Stateless Agent using Qwen2.5-3B-Instruct.
    Includes a fallback Mock mode if torch/transformers fails to load
    (allowing logic verification of the PI system without GPU/Env).
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.mock_mode = False
        self.tokenizer = None
        self.model = None
        
        try:
            print(f"Loading {model_name} (Updated Agent)...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            print(f"CRITICAL WARNING: Model load failed ({e}). Switching to MOCK mode for system validation.")
            self.mock_mode = True

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a response. Uses Mock if model failed to load.
        """
        if self.mock_mode:
            # Deterministic mock policy to simulate 'learning' (or at least valid actions)
            # Just output valid XML so the loop continues.
            actions = ["LEFT", "RIGHT", "UP", "DOWN"]
            action = random.choice(actions)
            return f"<thought>Mock thought for {action}</thought>\n<action>{action}</action>"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.1,  # Low temp for deterministic logic
            top_p=0.9
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
