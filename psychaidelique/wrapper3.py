import torch
import ollama
from .attention import PsychedelicAttention
from .profiles import PsychedelicLibrary
from .reward_model import RLEFRewardModel
from .config import CONFIG

class PsychedelicWrapper:
    def __init__(self, model_name_or_path="mistral", device_override="mps"):
        self.device = device_override
        self.model_name = model_name_or_path
        print(f"--- Psyk-AI-deliK : Moteur Ollama branché sur {self.model_name} ({self.device}) ---")
        
        self.config = CONFIG
        self.library = PsychedelicLibrary()
        self.reward_model = RLEFRewardModel()
        
        self.current_state = {
            "substance": "sober",
            "dose": 0.0,
            "current_entropy": 1.0
        }

    def set_consciousness(self, substance, dose):
        response = self.library.get_dose_response(substance, dose)
        self.current_state.update({
            "substance": substance,
            "dose": dose,
            "current_entropy": response["current_entropy"]
        })
        return self.current_state

    def generate(self, prompt):
        # Inférence réelle via Ollama
        # On ajuste la température en fonction de l'entropie de la substance
        temp = self.config["temperature_base"] * self.current_state["current_entropy"]
        
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": temp,
                "num_predict": self.config["max_tokens"]
            }
        )
        return response['response']

    def evaluate_output(self, prompt, generation):
        # Calcul du score VES basé sur Girn & Bzdok (2026)
        return self.reward_model.calculate_reward(
            prompt, 
            generation, 
            self.current_state.get("current_entropy", 1.0)
        )
