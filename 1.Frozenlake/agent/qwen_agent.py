import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

class QwenAgent:
    """
    Agent using Qwen2.5-1.5B-Instruct via Hugging Face Transformers.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.successful_examples: List[str] = []
        self.max_examples = 3

    def generate(self, prompt: str) -> str:
        # Augment with ICL
        augmented_prompt = prompt
        if self.successful_examples:
            examples_str = "\n\n--- SUCCESSFUL EXAMPLES ---\n" + "\n".join(self.successful_examples) + "\n---------------------------\n"
            augmented_prompt = examples_str + "\n" + prompt

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": augmented_prompt}
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

    def update(self, batch: List[Dict[str, Any]]):
        for episode in batch:
            if episode.get('score', 0) >= 1.0:
                example_text = f"User: ...\nAgent: {episode.get('response', '')}"
                if len(self.successful_examples) < self.max_examples:
                    self.successful_examples.append(example_text)
                else:
                    self.successful_examples.pop(0)
                    self.successful_examples.append(example_text)
