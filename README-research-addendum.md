# Research addendum: kinetics and evidence discipline

This branch adds an experimental, dimensionless time-course module and a small evidence dossier.

## New files

- `psychaidelique/kinetics.py`: onset/plateau/decay control curve and normalized amplitude scaling.
- `tests/test_kinetics.py`: standard-library unit tests.
- `research/literature_review.md`: human PK/PD references and interpretation limits.
- `research/profile_evidence.csv`: machine-readable evidence summary.
- `research/equivalence_mapping.md`: proposed operational analogies and evaluation design.
- `research/calibration_log.md`: initial assumptions and a future calibration protocol.

## Run the tests

From the repository root:

```bash
python -m unittest tests.test_kinetics -v
```

## Important limits

The constants are illustrative software parameters. They are not calibrated to chemical doses, plasma concentration, subjective intensity, or clock-time duration. Erowid is treated as descriptive secondary evidence only. Do not scrape or analyze individual reports without checking terms and obtaining permission where required.

The current wrapper generates non-streaming output in Ollama mode. The kinetics module is standalone and does not yet modulate tokens in the live generation loop. Integrating it requires a backend-specific streaming/callback design and controlled evaluation; this addendum does not claim that integration is complete.
