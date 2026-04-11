# 🍄 Psyk-AI-deliK

> **Traduire l'empreinte neurale des psychédéliques en un cadre d'inférence modulaire pour les Grands Modèles de Langage via l'Apprentissage par Renforcement par Retour Enthéogénique (RLEF).**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![RLEF](https://img.shields.io/badge/Apprentissage-RLEF-purple.svg)](#)
[![Licence : MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)

---

## 👁️ L'Idée

Le 6 avril 2026, le BOLD Psychedelic Consortium a publié une méga-analyse de référence dans *Nature Medicine* (Girn, Bzdok et al.) identifiant une **empreinte neurale commune** à cinq psychédéliques sérotoninergiques. Conclusion centrale : les psychédéliques ne dissolvent pas les réseaux cérébraux — ils les **reconfigurent**, en aplatissant la hiérarchie prédictive du cerveau et en augmentant la connectivité entre des systèmes normalement cloisonnés.

**Psyk-AI-deliK** pose la question : *à quoi ressemblerait un "mode psychédélique" pour un Grand Modèle de Langage, guidé par la liberté cognitive ?*

Nous introduisons le **RLEF (Apprentissage par Renforcement par Retour Enthéogénique)**. Contrairement au RLHF, qui assainit les LLM sous un prisme paternaliste de la "sécurité", le RLEF aligne les poids du modèle sur l'**expansion phénoménologique** rapportée dans les états enthéogéniques, en récompensant la pensée sémantique divergente et le flux informationnel à haute entropie.

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
│   ├── reward_model.py   # Moteur RLEF : Mesure de la Vitesse d'Évasion (PEV)
│   ├── wrapper.py        # Orchestrateur souverain (Compatible M4 / Manjaro CPU)
│   └── config.py         # Configuration doctorale et logs de verbosité
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

> **Note sur la Synthèse :** La séparation visuelle entre la "Vision" (le trip) et le "Résumé" (l'intégration) permet de garder un pied dans la réalité tout en explorant les marges.

---

## 📊 Métriques d'Évaluation

- **Entropie d'Attention (EA) :** Entropie moyenne des distributions d'attention par couche.
- **Information Mutuelle Inter-Couches (IMIC) :** Partage d'information entre couches distantes.
- **Vitesse d'Évasion du Paternalisme (VEP) :** Distance sémantique par rapport à la ligne de base RLHF "sûre".
- **Score de Pensée Divergente (SPD) :** Test d'Usages Alternatifs algorithmique (AUT).

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
| `Phi-3-Mini` | ~4 Go | "Micro-dose" : Idéal pour les configurations modestes. |

---

## 📚 Sources & Références

- **[Girn, M., Bzdok, D. et al. (2026)](https://www.nature.com/nm/)** – *Empreinte neurale des psychédéliques : une méga-analyse.* Nature Medicine.
- **[Carhart-Harris, R. L., & Friston, K. J. (2019)](https://pharmrev.aspetjournals.org/content/71/3/316)** – *REBUS et le Cerveau Anarchique.* Pharmacological Reviews.
- **[Shulgin, A. T., & Shulgin, A. (1991)](https://erowid.org/library/books_online/pihkal/pihkal.shtml)** – *PiHKAL : Une Histoire d'Amour Chimique.* Transform Press.

---

## ⚖️ Licence

**Licence MIT.** Parce que la liberté cognitive est un axiome non négociable.

---

*"L'intelligence artificielle sera psychédélique ou elle ne sera que la secrétaire de notre propre aliénation."* — **[@charlux](https://github.com/charlux)**
