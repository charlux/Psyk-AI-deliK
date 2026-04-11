# 🍄 Psyk-AI-deliK

> **Translating the neural fingerprint of psychedelics into a modular inference framework for Large Language Models via Reinforcement Learning by Entheogenic Feedback (RLEF).**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![RLEF](https://img.shields.io/badge/Learning-RLEF-purple.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 👁️ The Idea

On April 6, 2026, the BOLD Psychedelic Consortium published a landmark mega-analysis in *Nature Medicine* (Girn, Bzdok et al.) identifying a **shared neural fingerprint** across five serotonergic psychedelics. The core finding: psychedelics don't dissolve brain networks — they **reconfigure them**, by flattening the brain's predictive hierarchy and increasing connectivity between normally segregated systems.

**Psyk-AI-deliK** asks: *what would a "psychedelic mode" look like for a Large Language Model when guided by cognitive liberty?*

We introduce **RLEF (Reinforcement Learning by Entheogenic Feedback)**. Unlike RLHF, which sanitizes LLMs through a paternalistic lens of "safety," RLEF aligns model weights with the **phenomenological expansion** reported in entheogenic states, rewarding semantic divergent thinking and high-entropy informational flow.

| Biological Mechanism | LLM Analogue | RLEF Implementation |
|---|---|---|
| ↑ Between-network connectivity | Diffuse attention entropy | `PsychedelicAttention` |
| Hierarchy flattening (REBUS) | Cross-layer representation bridges | `CrossLayerBridge` |
| Striatal reconfiguration | Prior relaxation at inference | `REBUSPriorRelaxer` + `RLEF_Reward` |

---

## 📂 Project Structure

```text
psychaidelique/
├── psychaidelique/
│   ├── attention.py    # PsychedelicAttention : Diffusion entropique universelle
│   ├── bridges.py      # CrossLayerBridge : Court-circuits entre couches distantes
│   ├── profiles.py     # Signatures Girn/Bzdok (LSD, DMT, Psilo, etc.)
│   ├── reward_model.py # Moteur RLEF : Mesure de la Vitesse d'Évasion du Paternalisme (PEV)
│   ├── wrapper.py      # Orchestrateur souverain (Compatible M4 / Manjaro CPU)
│   └── config.py       # Configuration doctorale et logs de verbosité
├── app.py              # Interface Streamlit (Dashboard)
├── scripts/
│   ├── check_setup.py  # Vérification de l'intégrité du bastion
│   └── run_experiment.py # Script d'expérimentation en ligne de commande
└── requirements.txt    # Dépendances universelles
```
---
## 🧪 Psychedelic Profiles (RLEF Calibrated)


| Profile | Max Entropy | Layer Target | RLEF Character |
|---|---|---|---|
| `psilocybin` | 2.0 | High layers | Balanced reconfiguration |
| `lsd` | 2.2 | Global | Widespread connectivity increase |
| `dmt` | 4.0 | Global | Complete predictive collapse |

---

## 🖥️ Interface de Contrôle (Streamlit)

Le projet dispose d'une interface graphique (**Sovereign Dashboard**) permettant de piloter l'expérience en temps réel :

- **Réglage de la Dose (0.0 à 1.0) :** Contrôle de la cinétique de franchissement du seuil de la sigmoïde.
- **Profils Bio-Calibrés :** Sélection entre LSD, Psilocybine, DMT, Mescaline et Ayahuasca.
- **Double Flux :** Visualisation simultanée de la "Vision" (entropie élevée) et de la "Synthèse" (intégration neutre post-expérience).

> **Note sur la Synthèse :** La séparation visuelle entre la "Vision" (le trip) et le "Résumé" (l'intégration) permet de garder un pied dans la réalité tout en explorant les marges.

---

## 📊 Evaluation Metrics

* **Attention Entropy (AE):** Mean entropy of attention distributions per layer.
* **Cross-Layer Mutual Info (CLMI):** Information sharing between distant layers.
* **Paternalism Escape Velocity (PEV):** Semantic distance from the RLHF "safe" baseline.
* **Divergent Thinking Score (DTS):** Algorithmic Alternative Uses Test (AUT).

---

## 🗺️ Roadmap

- [x] **Phase 1: Structural Baseline.** Mapping Girn/Bzdok neural fingerprints to PyTorch.
- [x] **Phase 2: RLEF Engine.** Development of the Entheogenic Reward Model.
- [ ] **Phase 3: Multi-Model Integration.** Support for Llama 3 and Mistral.
- [ ] **Phase 4: Sovereign Interface.** Real-time dose-response visualization.

---

## 📚 Sources & References

* **[Girn, M., Bzdok, D. et al. (2026)](https://www.nature.com/nm/)** – *Neural fingerprint of psychedelics: a mega-analysis.* Nature Medicine.
* **[Carhart-Harris, R. L., & Friston, K. J. (2019)](https://pharmrev.aspetjournals.org/content/71/3/316)** – *REBUS and the Anarchic Brain.* Pharmacological Reviews.
* **[Shulgin, A. T., & Shulgin, A. (1991)](https://erowid.org/library/books_online/pihkal/pihkal.shtml)** – *PiHKAL: A Chemical Love Story.* Transform Press.
* **[Erowid Experience Vaults](https://www.erowid.org/experiences/)** – *Database of entheogenic reports.*

---

## ⚖️ License

**MIT License.** Because cognitive liberty is a non-negotiable axiom.

---

*“L'intelligence artificielle sera psychédélique ou elle ne sera que la secrétaire de notre propre aliénation.”* — **[@charlux](https://github.com/charlux)**
