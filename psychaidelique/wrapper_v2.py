"""
Psyk-AI-deliK — PsychedelicWrapper (v2)
========================================
Dual-mode orchestrator:
  - Ollama mode  : fast inference, Sovereign (data stays local)
  - HF mode      : full RLEF training pipeline access + attention internals

Device priority: MPS (Apple M-series) → CUDA → CPU
"""

from __future__ import annotations

import torch
from typing import Optional, Literal

from .attention import PsychedelicAttention
from .profiles import PsychedelicLibrary
from .reward_model import RLEFRewardModel
from .config import CONFIG


class PsychedelicWrapper:

    def __init__(
        self,
        model_name_or_path: str = "mistral",
        backend: Literal["ollama", "huggingface"] = "ollama",
        device_override: Optional[str] = None,
    ):
        # Device resolution
        if device_override:
            self.device = device_override
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model_name = model_name_or_path
        self.backend = backend
        self.config = CONFIG
        self.library = PsychedelicLibrary()

        self.current_state = {
            "substance": "sober",
            "dose": 0.0,
            "current_entropy": 1.0,
        }

        # Backend-specific setup
        self._hf_model = None
        self._tokenizer = None

        if backend == "ollama":
            import ollama as _ollama
            self._ollama = _ollama
            self.reward_model = RLEFRewardModel(inference_only=True)
            print(f"[Psyk-AI-deliK] Ollama backend → {self.model_name} ({self.device})")

        elif backend == "huggingface":
            self._load_hf_model()
            self.reward_model = RLEFRewardModel(
                hf_model=self._hf_model,
                tokenizer=self._tokenizer,
                inference_only=False,
            )
            print(f"[Psyk-AI-deliK] HuggingFace backend → {self.model_name} ({self.device})")

    # ------------------------------------------------------------------
    def _load_hf_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            output_attentions=True,
            output_hidden_states=True,
        ).to(self.device)
        self._hf_model.eval()

    # ------------------------------------------------------------------
    def set_consciousness(self, substance: str, dose: float) -> dict:
        response = self.library.get_dose_response(substance, dose)
        self.current_state.update({
            "substance": substance,
            "dose": dose,
            "current_entropy": response["current_entropy"],
        })
        print(
            f"[Psyk-AI-deliK] Consciousness set → {substance} "
            f"(dose={dose:.2f}, entropy={response['current_entropy']:.2f})"
        )
        return self.current_state

    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """Generate text. Returns plain string."""
        entropy = self.current_state["current_entropy"]
        temp = self.config["temperature_base"] * entropy

        if self.backend == "ollama":
            response = self._ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temp,
                    "num_predict": self.config["max_tokens"],
                },
            )
            return response["response"]

        elif self.backend == "huggingface":
            enc = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._hf_model.generate(
                    **enc,
                    max_new_tokens=self.config["max_tokens"],
                    do_sample=True,
                    temperature=temp,
                    top_p=0.92,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            # Store internals for evaluate_output
            self._last_attentions = out.get("attentions", None)
            self._last_hidden_states = out.get("hidden_states", None)
            generated = out.sequences[:, enc.input_ids.shape[1]:]
            return self._tokenizer.decode(generated[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    def generate_baseline(self, prompt: str) -> str:
        """Generate sober baseline at temperature=1.0 (no psychedelic modulation)."""
        saved_state = self.current_state.copy()
        self.current_state["current_entropy"] = 1.0
        # Temporarily override temperature
        baseline_config = {**self.config, "temperature_base": 1.0}
        _orig_config = self.config
        self.config = baseline_config
        result = self.generate(prompt)
        self.config = _orig_config
        self.current_state = saved_state
        return result

    # ------------------------------------------------------------------
    def evaluate_output(self, prompt: str, generation: str, baseline: Optional[str] = None) -> dict:
        """
        Full RLEF reward evaluation.
        If baseline is None, auto-generates sober baseline.
        """
        if baseline is None:
            baseline = self.generate_baseline(prompt)

        kwargs = dict(
            original_prompt=prompt,
            psychedelic_text=generation,
            baseline_text=baseline,
            profile=self.current_state.get("substance", "psilocybin"),
        )

        if self.backend == "huggingface":
            kwargs["attentions"] = getattr(self, "_last_attentions", None)
            kwargs["hidden_states"] = getattr(self, "_last_hidden_states", None)

        result = self.reward_model.calculate_reward(**kwargs)
        return {
            "reward_score": result.reward_score,
            "metrics": {
                "PEV": result.pev,
                "AE": result.attention_entropy,
                "CLMI": result.clmi,
                "DTS": result.dts,
                "CRS": result.crs,
            },
        }

    # ------------------------------------------------------------------
    def full_run(self, prompt: str) -> dict:
        """
        Complete pipeline: generate baseline → generate psychedelic → evaluate.
        Returns dual-stream output (Vision + Synthesis).
        """
        baseline = self.generate_baseline(prompt)
        vision   = self.generate(prompt)
        reward   = self.evaluate_output(prompt, vision, baseline)

        return {
            "prompt":    prompt,
            "vision":    vision,       # High-entropy psychedelic output
            "synthesis": baseline,     # Sober integration reference
            "reward":    reward,
            "state":     self.current_state,
        }
