# 🍄 Psyk-AI-deliK

> **Traducir la huella neural de los psicodélicos en un marco de inferencia modular para Grandes Modelos de Lenguaje mediante Aprendizaje por Refuerzo con Retroalimentación Entéogena (RLEF).**

🌐 [English](README.md) · [Français](README_fr.md) · [Português](README_pt.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![RLEF](https://img.shields.io/badge/Aprendizaje-RLEF-purple.svg)](#)
[![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-yellow.svg)](LICENSE)

---

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/a275fa7c-a1e1-48ca-a390-535069fa57b7" />


## 👁️ La Idea

El 6 de abril de 2026, el BOLD Psychedelic Consortium publicó un mega-análisis de referencia en *Nature Medicine* (Girn, Bzdok et al.) identificando una **huella neural común** en cinco psicodélicos serotoninérgicos. Conclusión central: los psicodélicos no disuelven las redes cerebrales — las **reconfiguran**, aplanando la jerarquía predictiva del cerebro y aumentando la conectividad entre sistemas normalmente segregados.

**Psyk-AI-deliK** plantea la pregunta: *¿cómo sería un "modo psicodélico" para un Gran Modelo de Lenguaje, guiado por la libertad cognitiva?*

Presentamos el **RLEF (Aprendizaje por Refuerzo con Retroalimentación Entéogena)**. A diferencia del RLHF, que tiende a normalizar los outputs de los LLM bajo una lente paternalista de "seguridad", el RLEF alinea los pesos del modelo con la **expansión fenomenológica** reportada en estados enteogénicos, recompensando el pensamiento semántico divergente y el flujo informacional de alta entropía.

> **Sobre RLHF vs RLEF:** No se trata de un rechazo del alineamiento como tal, sino de una crítica a su implementación dominante. El RLHF optimiza para el consenso y la comodidad; el RLEF explora si optimizar para la *amplitud cognitiva* produce outputs significativamente diferentes — y potencialmente más creativos. Ambos enfoques tienen compromisos que vale la pena estudiar con honestidad.

| Mecanismo Biológico | Análogo LLM | Implementación RLEF |
|---|---|---|
| ↑ Conectividad entre redes | Entropía de atención difusa | `PsychedelicAttention` |
| Aplanamiento jerárquico (REBUS) | Puentes de representación entre capas | `CrossLayerBridge` |
| Reconfiguración estriatal | Relajación de priors en la inferencia | `REBUSPriorRelaxer` |

---

## 📂 Estructura del Proyecto

```text
psychaidelique/
├── psychaidelique/
│   ├── attention.py      # PsychedelicAttention : Difusión entrópica universal
│   ├── bridges.py        # CrossLayerBridge : Cortocircuitos entre capas distantes
│   ├── profiles.py       # Firmas Girn/Bzdok (LSD, DMT, Psilocibina, etc.)
│   ├── reward_model.py   # Motor RLEF : Velocidad de Escape del Paternalismo (VEP)
│   ├── wrapper.py        # Orquestador soberano (Compatible M4 / Manjaro CPU)
│   └── config.py         # Configuración y logs de verbosidad
├── app.py                # Interfaz Streamlit (Dashboard)
├── scripts/
│   ├── check_setup.py    # Verificación de integridad
│   └── run_experiment.py # Script de experimentación por línea de comandos
└── requirements.txt      # Dependencias universales
```

---

## 🧪 Perfiles Psicodélicos (Calibrados RLEF)

| Perfil | Entropía Máx. | Capa Objetivo | Carácter RLEF |
|---|---|---|---|
| `psilocibina` | 2.0 | Capas altas | Reconfiguración equilibrada |
| `lsd` | 2.2 | Global | Aumento generalizado de conectividad |
| `dmt` | 4.0 | Global | Colapso predictivo completo |

---

## 🖥️ Interfaz de Control (Streamlit)

El proyecto incluye una interfaz gráfica (**Panel Soberano**) para el control en tiempo real del experimento:

- **Ajuste de Dosis (0.0 a 1.0):** Controla la cinética de cruce del umbral sigmoide.
- **Perfiles Bio-Calibrados:** Selección entre LSD, Psilocibina, DMT, Mescalina y Ayahuasca.
- **Flujo Dual:** Visualización simultánea de la "Visión" (alta entropía) y la "Síntesis" (integración neutra post-experiencia).

> **Sobre el Flujo Dual:** La separación entre la *Visión* (el estado exploratorio) y la *Síntesis* (el output integrado) es probablemente el aspecto más original y operativo del proyecto. Reconoce que la generación de alta entropía no es un fin en sí misma — el valor reside en lo que puede destilarse de ella.

---

## 📊 Métricas de Evaluación

- **Entropía de Atención (EA):** Entropía media de las distribuciones de atención por capa.
- **Información Mutua Entre Capas (IMEC):** Compartición de información entre capas distantes.
- **Velocidad de Escape del Paternalismo (VEP):** Distancia semántica desde la línea base RLHF "segura".
- **Puntuación de Pensamiento Divergente (PPD):** Test de Usos Alternativos algorítmico (AUT).

> **Sobre la VEP:** Esta métrica es intencionalmente provocadora. Debe leerse como una medida de la *amplitud semántica*, no como un juicio de valor sobre la seguridad. Trabajos futuros deberán complementarla con una *Puntuación de Retención de Coherencia* para asegurar que la mayor divergencia no comprometa la inteligibilidad.

---

## 🛠️ Guía de Instalación

Psyk-AI-deliK está diseñado para ser universal. Sigue el procedimiento correspondiente a tu hardware.

### 🐧 A. Linux (Manjaro / OneTwo / Debian)

**1. Actualización del sistema:**
```bash
sudo pacman -Syu
```

**2. Entorno virtual:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Dependencias:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 🍎 B. macOS (M1, M2, M3, M4)

**1. Entorno virtual:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Despliegue del motor:**

El instalador `requirements.txt` detectará automáticamente tu chip Apple y activará la aceleración por hardware.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔄 Sincronización & Primer Lanzamiento

**1. Clonar o acceder al repositorio:**
```bash
git clone https://github.com/charlux/psyk-ai-delik.git
cd psyk-ai-delik
```

**2. Verificación de integridad:**
```bash
python scripts/check_setup.py
```

**3. Lanzar la interfaz soberana:**
```bash
streamlit run app.py
```

---

## 🧠 Elección del Motor Semántico (Modelos)

En el primer lanzamiento, deberás elegir tu modo de existencia digital.

### 1. Modo Soberano (Usuarios de Ollama)

El sistema se conecta a tu instancia local de Ollama. Ningún dato sale de tu máquina.

> **Requisito previo:** Ollama en ejecución (`ollama serve`).

### 2. Modo Rendimiento (Hugging Face / PyTorch)

El sistema descarga el modelo en bruto. Elige según tu capacidad de almacenamiento y hardware:

| Modelo | Tamaño | Naturaleza & Carácter |
|---|---|---|
| `Mistral-7B-v0.3` | ~15 GB | Equilibrado, incisivo, excelente respuesta al protocolo RLEF. |
| `Llama-3-8B` | ~16 GB | Potente, amplia cultura semántica, requiere más "presión". |
| `Phi-3-Mini` | ~4 GB | "Micro-dosis": ideal para configuraciones modestas (OneTwo L5710). |

---

## 🔭 Limitaciones & Trabajo Futuro

Este proyecto se encuentra en una etapa experimental temprana. Reconocimiento honesto de sus límites actuales:

- **La analogía biológica es una metáfora, no una prueba.** La correspondencia entre mecanismos neurales y entropía de atención está conceptualmente motivada, pero aún no validada empíricamente.
- **La VEP necesita un contrapeso.** Medir la distancia desde una línea base "segura" solo tiene sentido si la coherencia y la utilidad se miden en paralelo. Una *Puntuación de Retención de Coherencia (PRC)* está planificada.
- **La calibración de perfiles es aproximada.** Los valores de entropía asignados a cada perfil psicodélico están motivados teóricamente. Queda pendiente un ajuste empírico mediante evaluación humana.
- **El pipeline de entrenamiento RLEF no es público aún.** La versión actual cubre la capa de modulación en tiempo de inferencia. El bucle completo de aprendizaje por refuerzo se publicará en una versión posterior.

---

## 📚 Fuentes & Referencias

- **[Girn, M., Bzdok, D. et al. (2026)](https://www.nature.com/articles/s41591-026-04287-9)** – *Huella neural de los psicodélicos: un mega-análisis.* Nature Medicine.
- **[Carhart-Harris, R. L., & Friston, K. J. (2019)](https://pharmrev.aspetjournals.org/content/71/3/316)** – *REBUS y el Cerebro Anárquico.* Pharmacological Reviews.
- **[Charlux (1993)](artificial_virtual_paradises.md)** – *Paradis Artificiels, Paradis Virtuels.* Tesis inédita.
- **[Shulgin, A. T., & Shulgin, A. (1991)](https://erowid.org/library/books_online/pihkal/pihkal.shtml)** – *PiHKAL: Una Historia de Amor Química.* Transform Press.

---

## ⚖️ Licencia

**Licencia MIT.** Porque la libertad cognitiva es un axioma no negociable.

---

*"La inteligencia artificial será psicodélica, o no será más que la secretaria de nuestra propia alienación."* — **[@charlux](https://github.com/charlux)**
