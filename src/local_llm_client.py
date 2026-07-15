import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.config import QWEN_MODEL_NAME

class LocalLLM:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        print(f"Loading local LLM ({QWEN_MODEL_NAME}) with 4-bit quantization. This may take a moment...")
        
        # Setup HuggingFace cache directory (useful for Colab GDrive caching)
        cache_dir = "/content/drive/MyDrive/huggingface_cache"
        if not os.path.exists("/content/drive"):
            cache_dir = None # Fallback to default if not in Colab or drive not mounted
            
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_NAME, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            return_full_text=False
        )
        print("LLM Loaded Successfully!")
        
    def generate_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Try applying chat template for instruct models
            prompt_formatted = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            # Fallback if model doesn't have a chat template
            prompt_formatted = prompt
            
        outputs = self.pipe(prompt_formatted)
        return outputs[0]["generated_text"]
