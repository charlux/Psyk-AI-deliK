"""Time-course modulation for Psyk-AI-deliK.

This is a software control curve, not a pharmacokinetic model and not a claim
that an LLM experiences a drug-like state. All values are dimensionless and
must be calibrated against output-quality measurements before production use.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class KineticProfile:
    """Dimensionless time-course profile.

    onset_tau, plateau_tau and decay_tau are expressed in generation steps.
    intensity is bounded to [0, 1]. baseline is the residual modulation floor.
    """
    name: str
    onset_tau: float
    plateau_steps: float
    decay_tau: float
    intensity: float = 1.0
    baseline: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        for field_name in ("onset_tau", "plateau_steps", "decay_tau"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and > 0")
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("intensity must be within [0, 1]")
        if not 0.0 <= self.baseline <= 1.0:
            raise ValueError("baseline must be within [0, 1]")

    def value_at(self, step: float) -> float:
        """Return normalized modulation at a non-negative generation step."""
        if not math.isfinite(step) or step < 0:
            raise ValueError("step must be finite and >= 0")
        if step <= self.plateau_steps:
            # Exponential onset approaches, but never exceeds, intensity.
            fraction = 1.0 - math.exp(-step / self.onset_tau)
            return self.baseline + (self.intensity - self.baseline) * fraction
        at_plateau = self.baseline + (self.intensity - self.baseline) * (
            1.0 - math.exp(-self.plateau_steps / self.onset_tau)
        )
        elapsed = step - self.plateau_steps
        return self.baseline + (at_plateau - self.baseline) * math.exp(
            -elapsed / self.decay_tau
        )


def dose_adjusted_profile(profile: KineticProfile, dose: float) -> KineticProfile:
    """Scale modulation amplitude by a normalized software dose (0..1).

    Dose is an abstract UI/control parameter, not a chemical dose or a
    recommended drug dosage. Time constants are intentionally not scaled.
    """
    if not math.isfinite(dose) or not 0.0 <= dose <= 1.0:
        raise ValueError("dose must be finite and within [0, 1]")
    return KineticProfile(
        name=profile.name,
        onset_tau=profile.onset_tau,
        plateau_steps=profile.plateau_steps,
        decay_tau=profile.decay_tau,
        intensity=profile.intensity * dose,
        baseline=profile.baseline,
    )


DEFAULT_PROFILES: dict[str, KineticProfile] = {
    "psilocybin": KineticProfile("psilocybin", 5.0, 24.0, 12.0, 0.55),
    "lsd": KineticProfile("lsd", 8.0, 40.0, 20.0, 0.60),
    "dmt_inhaled": KineticProfile("dmt_inhaled", 1.0, 4.0, 3.0, 0.90),
    "mescaline": KineticProfile("mescaline", 10.0, 35.0, 22.0, 0.50),
    "ayahuasca_oral": KineticProfile("ayahuasca_oral", 12.0, 30.0, 18.0, 0.60),
}
