# Literature review: psychedelic time courses and computational analogy

**Scope.** This is a compact evidence map for a software research prototype, not medical advice, dosing guidance, or evidence that a language model can reproduce a human psychedelic state. Human duration estimates vary with dose, route, study design, setting, and individual factors.

## Primary human evidence

- **Psilocybin / psilocin pharmacokinetics and pharmacodynamics.** Controlled human PK/PD study: https://pubmed.ncbi.nlm.nih.gov/36507738/ . Reported subjective effects persist for hours while plasma psilocin has a much shorter elimination half-life; plasma half-life must not be equated with subjective duration.
- **LSD.** Controlled dose-response PK/PD study: https://pmc.ncbi.nlm.nih.gov/articles/PMC5591798/ . The study reports dose-dependent duration and exposure. The software should represent duration and amplitude separately rather than use one scalar “intensity.”
- **Mescaline.** Controlled dose-response study: https://pmc.ncbi.nlm.nih.gov/articles/PMC11442856/ . Small sample; duration increased across studied doses. Do not extrapolate the estimates as individual predictions.
- **Ayahuasca.** Human pharmacology study: https://pubmed.ncbi.nlm.nih.gov/12660312/ . Oral ayahuasca is not interchangeable with inhaled DMT: beta-carbolines alter DMT metabolism and the experience has a different time course.
- **Classic psychedelics and REBUS.** Carhart-Harris & Friston, “REBUS and the Anarchic Brain” (2019): https://pharmrev.aspetjournals.org/content/71/3/316 . This is a theoretical account of relaxed beliefs under psychedelics, not a validated recipe for model decoding.

## Route-specific descriptive sources

- Erowid, mushrooms effects: https://www.erowid.org/plants/mushrooms/mushrooms_effects.shtml
- Erowid, LSD effects: https://www.erowid.org/chemicals/lsd/lsd_effects.shtml
- Erowid, DMT effects: https://www.erowid.org/chemicals/dmt/dmt_effects.shtml

Erowid is a curated secondary/experience-report resource. It can inform qualitative hypotheses but should not be treated as controlled clinical evidence. No scraping, report-level extraction, or automated text analysis should be undertaken without confirming the applicable terms and obtaining explicit permission where required.

## Evidence grading used in this repository

- **A — controlled human study:** directly reports measured human PK/PD or time course.
- **B — systematic review / broad scientific review:** useful synthesis, with heterogeneity noted.
- **C — curated descriptive source:** useful for generating hypotheses, not estimating causal effects.
- **D — computational hypothesis:** proposed mapping for evaluation; not a biological fact.

## Interpretation guardrails

1. Pharmacological duration is measured in clock time; software kinetics are measured in generated tokens/steps. These units are not convertible without an explicitly validated experimental mapping.
2. “Entropy” is not synonymous with psychedelic intensity, creativity, therapeutic value, or consciousness.
3. Do not infer a universal ranking of substances from the current profile constants.
4. Separate onset, plateau, decay, residual modulation, and response amplitude.
5. Record uncertainty, route, endpoint, study population, and evidence grade whenever adding a numerical estimate.
