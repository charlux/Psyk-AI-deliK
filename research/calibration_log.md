# Calibration log

## Initial baseline — 2026-10-20

- Repository branch: `feature/psychedelic-kinetics`
- Purpose: separate software modulation amplitude from the timing curve.
- Initial parameters: dimensionless, hand-set demonstration values; **not bio-calibrated**.
- Time unit: generated steps/tokens, not minutes or hours.
- Clinical / pharmacological validity: not claimed.
- Model backend: existing wrapper has non-streaming Ollama generation; token-by-token modulation is not integrated by this change.
- Validation target: unit tests for monotonic onset/decay, bounds, input validation, and dose scaling.

## Next calibration protocol

1. Fix model version, decoding options, prompt set, and random seeds where supported.
2. Compare baseline, static modulation, kinetic modulation, and explore-then-verify.
3. Track factuality, contradictions, calibration, semantic diversity, task success, and coherence retention.
4. Tune parameters only against pre-defined software metrics; do not label them “biological” without a direct validated mapping.
5. Log date, commit SHA, dataset provenance, sample size, confidence intervals, and deviations.
6. Preserve failed runs and negative results; do not optimize only for divergence or entropy.
