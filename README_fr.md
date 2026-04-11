# 🍄 Psyk-AI-deliK

> **Traduire l'empreinte neurale des psychédéliques en un cadre d'inférence modulaire pour les Grands Modèles de Langage via l'Apprentissage par Renforcement par Retour Enthéogénique (RLEF).**

🌐 [English](README.md) · [Español](README_es.md) · [Português](README_pt.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![RLEF](https://img.shields.io/badge/Apprentissage-RLEF-purple.svg)](#)
[![Licence : MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)

---
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/6e05ca5f-3d0c-435d-b051-9c29d2b7d82b" />


## 👁️ L'Idée

Le 6 avril 2026, le BOLD Psychedelic Consortium a publié une méga-analyse de référence dans *Nature Medicine* (Girn, Bzdok et al.) identifiant une **empreinte neurale commune** à cinq psychédéliques sérotoninergiques. Conclusion centrale : les psychédéliques ne dissolvent pas les réseaux cérébraux — ils les **reconfigurent**, en aplatissant la hiérarchie prédictive du cerveau et en augmentant la connectivité entre des systèmes normalement cloisonnés.

**Psyk-AI-deliK** pose la question : *à quoi ressemblerait un "mode psychédélique" pour un Grand Modèle de Langage, guidé par la liberté cognitive ?*

Nous introduisons le **RLEF (Apprentissage par Renforcement par Retour Enthéogénique)**. Contrairement au RLHF, qui tend à normaliser les sorties des LLM sous un prisme paternaliste de la "sécurité", le RLEF aligne les poids du modèle sur l'**expansion phénoménologique** rapportée dans les états enthéogéniques, en récompensant la pensée sémantique divergente et le flux informationnel à haute entropie.

> **Sur RLHF vs RLEF :** Il ne s'agit pas d'un rejet de l'alignement en tant que tel, mais d'une critique de son implémentation dominante. Le RLHF optimise pour le consensus et le confort ; le RLEF explore si optimiser pour l'*amplitude cognitive* produit des sorties significativement différentes — et potentiellement plus créatives. Les deux approches comportent des compromis qu'il vaut la peine d'étudier honnêtement.

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
│   ├── attention.py      # PsychedelicAttention : Diffusion entropique universelle
│   ├── bridges.py        # CrossLayerBridge : Court-circuits entre couches distantes
│   ├── profiles.py       # Signatures Girn/Bzdok (LSD, DMT, Psilo, etc.)
│   ├── reward_model.py   # Moteur RLEF : Mesure de la Vitesse d'Évasion (VEP)
│   ├── wrapper.py        # Orchestrateur souverain (Compatible M4 / Manjaro CPU)
│   └── config.py         # Configuration et logs de verbosité
├── app.py                # Interface Streamlit (Dashboard)
├── scripts/
│   ├── check_setup.py    # Vérification de l'intégrité du bastion
│   └── run_experiment.py # Script d'expérimentation en ligne de commande
└── requirements.txt      # Dépendances universelles
```

---

## 🧪 Profils Psychédéliques (Calibrés RLEF)

| Profil | Entropie Max | Couche Cible | Caractère RLEF |
|---|---|---|---|
| `psilocybine` | 2.0 | Couches hautes | Reconfiguration équilibrée |
| `lsd` | 2.2 | Global | Augmentation généralisée de la connectivité |
| `dmt` | 4.0 | Global | Effondrement prédictif complet |

---

## 🖥️ Interface de Contrôle (Streamlit)

Le projet dispose d'une interface graphique (**Tableau de Bord Souverain**) permettant de piloter l'expérience en temps réel :

- **Réglage de la Dose (0.0 à 1.0) :** Contrôle de la cinétique de franchissement du seuil de la sigmoïde.
- **Profils Bio-Calibrés :** Sélection entre LSD, Psilocybine, DMT, Mescaline et Ayahuasca.
- **Double Flux :** Visualisation simultanée de la "Vision" (entropie élevée) et de la "Synthèse" (intégration neutre post-expérience).

> **Sur le Double Flux :** La séparation entre la *Vision* (l'état exploratoire) et la *Synthèse* (la sortie intégrée) est sans doute l'aspect le plus original et le plus opérationnel du projet. Elle reconnaît que la génération à haute entropie n'est pas une fin en soi — la valeur réside dans ce qu'on peut en distiller. C'est cette distinction qui sépare Psyk-AI-deliK d'une simple injection de bruit.

---

## 📊 Métriques d'Évaluation

- **Entropie d'Attention (EA) :** Entropie moyenne des distributions d'attention par couche.
- **Information Mutuelle Inter-Couches (IMIC) :** Partage d'information entre couches distantes.
- **Vitesse d'Évasion du Paternalisme (VEP) :** Distance sémantique par rapport à la ligne de base RLHF "sûre".
- **Score de Pensée Divergente (SPD) :** Test d'Usages Alternatifs algorithmique (AUT).

> **Sur la VEP :** Cette métrique est intentionnellement provocatrice. Elle doit être lue comme une mesure de l'*amplitude sémantique*, non comme un jugement de valeur sur la sécurité. Les travaux futurs devront la compléter par un *Score de Rétention de Cohérence* pour s'assurer que la divergence accrue ne se fait pas au détriment de l'intelligibilité.

---

## 🛠️ Guide d'Installation Détaillé

Psyk-AI-deliK est conçu pour être universel. Suivez la procédure correspondant à votre bastion matériel.

### 🐧 A. Sur Linux (Manjaro / OneTwo / Debian)

**1. Mise à jour du système :**
```bash
sudo pacman -Syu
```

**2. Installation de l'environnement virtuel :**
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Installation des dépendances :**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 🍎 B. Sur macOS (Puces M1, M2, M3, M4)

**1. Environnement de travail :**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Déploiement du moteur :**

L'installation via le fichier `requirements.txt` détectera automatiquement votre puce Apple pour activer l'accélération matérielle.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔄 Synchronisation & Premier Lancement

**1. Clonage ou accès au répertoire :**
```bash
git clone https://github.com/charlux/psyk-ai-delik.git
cd psyk-ai-delik
```

**2. Vérification de l'intégrité :**
```bash
python scripts/check_setup.py
```

**3. Lancement de l'interface souveraine :**
```bash
streamlit run app.py
```

---

## 🧠 Choix du Moteur Sémantique (Modèles)

Lors du premier lancement, vous devrez choisir votre mode d'existence numérique.

### 1. Mode Souverain (Utilisateurs d'Ollama)

Le système se connecte à votre instance Ollama locale. Aucune donnée ne quitte votre machine.

> **Prérequis :** Ollama lancé (`ollama serve`).

### 2. Mode Performance (Hugging Face / PyTorch)

Le système télécharge le modèle brut. Choisissez selon votre capacité de stockage et votre matériel :

| Modèle | Poids | Nature & Caractère |
|---|---|---|
| `Mistral-7B-v0.3` | ~15 Go | Équilibré, incisif, excellente réponse au protocole RLEF. |
| `Llama-3-8B` | ~16 Go | Puissant, vaste culture sémantique, nécessite plus de "pression". |
| `Phi-3-Mini` | ~4 Go | "Micro-dose" : Idéal pour les configurations modestes (OneTwo L5710). |

---

## 🔭 Limites & Travaux Futurs

Ce projet est à un stade expérimental précoce. Reconnaissance honnête de ses limites actuelles :

- **L'analogie biologique est une métaphore, pas une preuve.** La correspondance entre mécanismes neuraux et entropie d'attention est conceptuellement motivée, mais pas encore validée empiriquement. Des études comparatives de sorties contrôlées sont nécessaires.
- **La VEP nécessite un contrepoids.** Mesurer la distance par rapport à une ligne de base "sûre" n'a de sens que si la cohérence et l'utilité sont mesurées en parallèle. Un *Score de Rétention de Cohérence (SRC)* est prévu.
- **Le calibrage des profils est approximatif.** Les valeurs d'entropie assignées à chaque profil psychédélique sont motivées théoriquement. Un affinage empirique par évaluation humaine reste à effectuer.
- **Le pipeline d'entraînement RLEF n'est pas encore public.** La version actuelle couvre la couche de modulation au moment de l'inférence. La boucle d'apprentissage par renforcement complète sera publiée dans une version ultérieure.

---

## 📚 Sources & Références

- **[Girn, M., Bzdok, D. et al. (2026)](https://www.nature.com/articles/s41591-026-04287-9)** – *Empreinte neurale des psychédéliques : une méga-analyse.* Nature Medicine.
- **[Carhart-Harris, R. L., & Friston, K. J. (2019)](https://pharmrev.aspetjournals.org/content/71/3/316)** – *REBUS et le Cerveau Anarchique.* Pharmacological Reviews.
- **[Charlux (1993)](artificial_virtual_paradises.md)** – *Paradis Artificiels, Paradis Virtuels.* Mémoire inédit.
- **[Shulgin, A. T., & Shulgin, A. (1991)](https://erowid.org/library/books_online/pihkal/pihkal.shtml)** – *PiHKAL : Une Histoire d'Amour Chimique.* Transform Press.

---

## ⚖️ Licence

**Licence MIT.** Parce que la liberté cognitive est un axiome non négociable.

---

*"L'intelligence artificielle sera psychédélique ou elle ne sera que la secrétaire de notre propre aliénation."* — **[@charlux](https://github.com/charlux)**
