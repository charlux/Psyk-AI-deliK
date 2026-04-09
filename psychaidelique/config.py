from dataclasses import dataclass

@dataclass
class RLEFConfig:
    model_name: str
    profile: str = "psilocybin"
    dose: float = 0.8
    rlef_guidance: bool = True
    cross_layer: bool = True
