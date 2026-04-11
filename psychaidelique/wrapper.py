"""
Psyk-AI-deliK - The Universal Sovereign Wrapper (PyTorch)
Orchestrating RLEF-guided inference across all platforms.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .profiles import PsychedelicLibrary
from .attention import PsychedelicAttention

class PsychedelicWrapper:
    def __init__(self, config):
        self.config = config
        
        # Détection automatique du hardware (Souveraineté matérielle)
        if torch.backends.mps.is_available():
            self.device = "mps" # Apple Silicon
        elif torch.cuda.is_available():
            self.device = "cuda" # Nvidia
        else:
            self.device = "cpu" # Manjaro OneTwo / Standard Linux
            
        print(f"--- Psyk-AI-deliK initialized on: {self.device} ---")

        # Chargement universel via HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, 
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        ).to(self.device)
        
        self.library = PsychedelicLibrary()
        self.params = self.library.get_dose_response(config.profile, config.dose)
        
        self.psyk_attention = PsychedelicAttention(
            entropy_factor=self.params["current_entropy"],
            connectivity_boost=self.params["current_bridge"]
        )

    def generate(self, prompt: str, max_tokens: int = 200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # L'injection de l'entropie se fait ici via les paramètres de sampling
        # simulant la reconfiguration neurale.
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            temperature=self.params["current_entropy"],
            do_sample=True,
            top_p=0.95 # Relaxation des priors
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
