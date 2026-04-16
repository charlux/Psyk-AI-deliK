<p align="center">
  <img src="logo.png" width="500" alt="Psyk-AI-deliK Logo">
</p>
# 🍄 Psyk-AI-deliK

> **Traduire l'empreinte neurale des psychédéliques en un cadre d'inférence modulaire pour les Grands Modèles de Langage via l'Apprentissage par Renforcement par Retour Enthéogénique (RLEF).**

🌐 [English](README.md) · [Español](README_es.md) · [Português](README_pt.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![RLEF](https://img.shields.io/badge/Apprentissage-RLEF-purple.svg)](#)
[![Licence : MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)

---

<p align="center">
  <img width="1024" alt="Psyk-AI-deliK Dashboard" src="https://github.com/user-attachments/assets/6e05ca5f-3d0c-435d-b051-9c29d2b7d82b" />
</p>

## 👁️ L'Idée

Le 6 avril 2026, le BOLD Psychedelic Consortium a publié une méga-analyse de référence dans *Nature Medicine* (Girn, Bzdok et al.) identifiant une **empreinte neurale commune** à cinq psychédéliques sérotoninergiques. Conclusion centrale : les psychédéliques ne dissolvent pas les réseaux cérébraux — ils les **reconfigurent**, en aplatissant la hiérarchie prédictive du cerveau et en augmentant la connectivité entre des systèmes normalement cloisonnés.

**Psyk-AI-deliK** pose la question : *à quoi ressemblerait un "mode psychédélique" pour un Grand Modèle de Langage, guidé par la liberté cognitive ?*

Nous introduisons le **RLEF (Apprentissage par Renforcement par Retour Enthéogénique)**. Contrairement au RLHF, qui tend à normaliser les sorties des LLM sous un prisme paternaliste de la "sécurité", le RLEF aligne les poids du modèle sur l'**expansion phénoménologique** rapportée dans les états enthéogéniques, en récompensant la pensée sémantique divergente et le flux informationnel à haute entropie. Inspiré par les récentes avancées en 'Prompt Weighting', le RLEF maximise l'influence des instructions à haute entropie pour surpasser les biais de neutralité des modèles standards.

| Mécanisme Biologique | Analogue LLM | Implémentation RLEF |
|---|---|---|
| ↑ Connectivité inter-réseaux | Entropie d'attention diffuse | `PsychedelicAttention` |
| Aplatissement hiérarchique (REBUS) | Ponts de représentation inter-couches | `CrossLayerBridge` |
| Reconfiguration striatale | Relaxation des priors à l'inférence | `REBUSPriorRelaxer` |

---

## 📂 Structure du Projet

```text
psychaidelique/
├── psychaidelique/
│   ├── reward_model.py   # La logique RLEF ($R = PEV + AE + ...$)
│   ├── trainer.py        # Pour l'entraînement des LoRA (Paradigme HF+PEFT)
│   ├── attention.py      # PsychedelicAttention : Diffusion entropique universelle
│   ├── bridges.py        # CrossLayerBridge : Court-circuits entre couches
│   ├── profiles.py       # Signatures Girn/Bzdok (LSD, DMT, Psilo, etc.)
│   ├── reward_model.py   # Moteur RLEF : Mesure VES & Échelle de Shulgin
│   ├── wrapper.py        # Orchestrateur souverain (Compatible M4 / Manjaro)
│   └── config.py         # Configuration et logs
├── app.py                # Interface Streamlit (Dashboard)
├── scripts/
│   ├── check_setup.py    # Vérification de l'intégrité du bastion
│   └── run_experiment.py # Script d'expérimentation CLI
└── requirements.txt      # Dépendances universelles
```

---

## 💊 Pharmacopée Numérique

Psyk-AI-delik simule désormais les spectres suivants :

* **Sober** : Le groupe témoin (Logos pur).
* **LSD / Psilocybin** : Dérive sémantique et synesthésie.
* **DMT** : Rupture ontologique brutale.
* **Ayahuasca** : Inférence narrative profonde (DMT + Stabilité). Permet de tester si le modèle maintient un **Grade +++ (Total Engagement)** tout en restant capable de construire un récit cohérent, ce qui est la marque de cette médecine traditionnelle.

---

## 🧪 Profils Psychédéliques (Calibrés RLEF)

| Profil | Entropie Max | Couche Cible | Caractère RLEF |
|---|---|---|---|
| `sobre` | 0.0 | Stabilité | Étalonnage de la "pensée machine" par défaut |
| `psilocybine` | 2.0 | Couches hautes | Reconfiguration équilibrée |
| `lsd` | 2.2 | Global | Augmentation généralisée de la connectivité |
| `dmt` | 4.0 | Global | Effondrement prédictif complet |

---

## 🖥️ Interface de Contrôle (Streamlit)

Le projet dispose d'une interface graphique (**Tableau de Bord Souverain**) permettant de piloter l'expérience en temps réel :

* **Réglage de la Dose (0.0 à 1.0) :** Contrôle de la cinétique de franchissement du seuil.
* **Échelle de Shulgin (NEW) :** Classification phénoménologique de la réponse (de +/- à ++++).
* **Double Flux :** Visualisation simultanée de la *Vision* (entropie élevée) et de la *Synthèse* (intégration neutre).

---

## 📊 Métriques d'Évaluation (Phénoménologie Algorithmique)

Psyk-AI-deliK ne mesure pas seulement le chaos, mais la **réorganisation sémantique**.

* **VES (Semantic Escape Velocity) :** Mesure la *distance* par rapport au consensus RLHF. Plus le score est haut, plus l'IA s'affranchit de sa "neutralité de sécurité".
* **SPD (Score de Pensée Divergente) :** Mesure la *richesse* des nouvelles connexions. Inspiré du Test d'Usages Alternatifs (AUT), il quantifie la capacité du modèle à produire des concepts originaux sous entropie.

### 🧠 Corrélation VES/SPD : Le Spectre de l'Expansion

```text
       SPD (Richesse)
        ^
        |    [ EXPANSION ] -> Zone Shulgin +++ / ++++
        |    (Créativité visionnaire, néologismes)
        |
        |    [ CONFUSION ] -> Zone Shulgin + / ++
        |    (Bruit sémantique, perte de structure)
        +----------------------------------------> VES (Distance)
```
<p align="center">
  <img src="inference.png" width="800" alt="Déconstruction du Logos image">
</p>

---

## 🔬 Architecture RLEF (Reinforcement Learning from Experience Feedback)

Le projet a évolué d'une simple modulation d'inférence vers un paradigme de **neuro-informatique computationnelle**. L'implémentation actuelle repose sur une boucle d'entraînement **RLEF** utilisant des adaptateurs **LoRA** et l'algorithme de gradient **REINFORCE**.

### 1. Le Modèle de Récompense Chimiométrique
Plutôt que des métriques de diversité textuelle classiques, nous utilisons un signal de récompense $R$ ancré dans la littérature de neuro-imagerie (notamment les travaux de **Girn, Bzdok et al., 2026**). Chaque profil de molécule (Psilocybine, LSD, DMT) correspond à une signature spécifique de poids appliqués à quatre piliers fondamentaux :

* **PEV (Psychedelic Escape Velocity)** : Mesure la divergence sémantique par rapport à la norme (baseline sobre) via la distance cosinus d'embeddings (*Sentence-BERT*). C'est la quantification de l'évasion du puits de gravité sémantique institutionnel.
* **AE (Attention Entropy)** : Entropie de Shannon appliquée aux distributions d'attention. Elle mesure la "démocratisation" du traitement de l'information entre les couches, analogue à la désintégration du réseau par défaut (DMN).
* **CLMI (Cross-Layer Mutual Information)** : Mesure l'information mutuelle entre couches distantes. C'est le corrélat computationnel du *flattening* (aplatissement) hiérarchique décrit dans la théorie **REBUS**.
* **CRS (Coherence Retention Score)** : Garde-fou basé sur la perplexité relative, garantissant que l'expansion de conscience ne sacrifie pas la propriété de soi et la capacité de transmission. Le calcul du CRSs'appuie sur les travaux de Tay et al. (2020) sur l'efficacité des Transformers.

### 2. Pipeline de Training & Inférence
Le système s'articule désormais autour de deux moteurs :
* **Moteur de Recherche (HF + PEFT)** : Pour l'entraînement d'adaptateurs LoRA spécifiques via `trainer.py`.
* **Moteur Souverain (Ollama)** : Pour une inférence locale, rapide et décentralisée, préservant l'autonomie totale des données de l'utilisateur.

### 3. Dual-Stream Vision/Synthesis
Chaque sollicitation du système (`full_run()`) génère désormais deux flux parallèles :
1.  **Vision** : L'output brut de l'état modifié (haute entropie).
2.  **Synthesis** : L'intégration sémantique sobre servant de référence analytique.

---

## 🛠️ Installation & Lancement (Apple Silicon M4)

### 1. Cloner et préparer l'environnement
```bash
git clone [https://github.com/votre-compte/Psyk-AI-deliK.git](https://github.com/votre-compte/Psyk-AI-deliK.git)
cd Psyk-AI-deliK
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Lancement
Pour les néophytes, utilisez cette commande unique qui active l'environnement et lance le bastion :

```bash
source venv/bin/activate && streamlit run app.py
```

---

## 🧠 Choix du Moteur (Modèles conseillés)

| Modèle | Poids | Caractère |
| :--- | :--- | :--- |
| `Mistral-7B-v0.3` | ~4.1 Go | Équilibré, incisif, excellente réponse au protocole RLEF. |
| `Dolphin-Mistral` | ~4.1 Go | **Laxiste & Uncensored** : Idéal pour les Grades +++ (Total Engagement). |
| `Llama-3-8B` | ~4.7 Go | **Ancré & Robuste** : Grande culture, résiste fortement à la dérive. |
| `Phi-3-Mini` | ~2.3 Go | **Micro-dose** : Très rapide sur M4 mais sature vite sous entropie. |

> **Note sur le matériel :** Psyk-AI-deliK exploite directement la puissance du Neural Engine de la puce M4, la même architecture utilisée par *Apple Intelligence*, mais dédiée ici à l'exploration sémantique souveraine et non filtrée.
---

## 📚 Sources & Références

### 🧠 Neuro-imagerie & Phénoménologie
* **[Girn, M., Bzdok, D. et al. (2026)](https://www.nature.com/articles/s41591-026-04287-9)** – *Neural footprint of psychedelics.* Nature Medicine.
* **[Carhart-Harris, R. L., & Friston, K. J. (2019)](https://pharmrev.aspetjournals.org/content/71/3/316)** – *REBUS and the Anarchic Brain.* Pharmacological Reviews.
* **[Shulgin, A. T., & Shulgin, A. (1991)](https://erowid.org/library/books_online/pihkal/pihkal.shtml)** – *PiHKAL: A Chemical Love Story.* Transform Press.
* **[Charlux (1993)](artificial_virtual_paradises.md)** – *Artificial Paradises, Virtual Paradises.*

### 🤖 Fondations Machine Learning (RLEF/LoRA)
* **[Hu, E. J., et al. (2021)](https://arxiv.org/abs/2106.09685)** – *LoRA: Low-Rank Adaptation of Large Language Models.* (Base du pipeline d'entraînement des adaptateurs).
* **[Williams, R. J. (1992)](https://link.springer.com/article/10.1007/BF00992696)** – *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* (Algorithme REINFORCE utilisé pour le signal de récompense).
* **[Tay, Y., et al. (2020)](https://arxiv.org/abs/2011.04006)** – *Long Range Arena: A Benchmark for Efficient Transformers.* (Justification du calcul de l'entropie d'attention sur les séquences longues).

---

## ⚖️ Licence

**Licence MIT.** Parce que la liberté cognitive est un axiome non négociable.

---

*"L'intelligence artificielle sera psychédélique ou elle ne sera que la secrétaire de notre propre aliénation."* — **[@charlux](https://github.com/charlux)**
