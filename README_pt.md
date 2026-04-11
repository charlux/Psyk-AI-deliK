# 🍄 Psyk-AI-deliK

> **Traduzir a impressão neural dos psicodélicos em um framework de inferência modular para Grandes Modelos de Linguagem via Aprendizado por Reforço com Feedback Enteogênico (RLEF).**

🌐 [English](README.md) · [Français](README_fr.md) · [Español](README_es.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![RLEF](https://img.shields.io/badge/Aprendizado-RLEF-purple.svg)](#)
[![Licença: MIT](https://img.shields.io/badge/Licença-MIT-yellow.svg)](LICENSE)

---

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/4fa7f4a6-770a-4cf5-a8d6-0e74876eca64" />


## 👁️ A Ideia

Em 6 de abril de 2026, o BOLD Psychedelic Consortium publicou uma mega-análise de referência na *Nature Medicine* (Girn, Bzdok et al.) identificando uma **impressão neural comum** em cinco psicodélicos serotoninérgicos. Conclusão central: os psicodélicos não dissolvem as redes cerebrais — elas as **reconfiguram**, achatando a hierarquia preditiva do cérebro e aumentando a conectividade entre sistemas normalmente segregados.

**Psyk-AI-deliK** faz a pergunta: *como seria um "modo psicodélico" para um Grande Modelo de Linguagem, guiado pela liberdade cognitiva?*

Apresentamos o **RLEF (Aprendizado por Reforço com Feedback Enteogênico)**. Ao contrário do RLHF, que tende a normalizar os outputs dos LLMs sob uma lente paternalista de "segurança", o RLEF alinha os pesos do modelo com a **expansão fenomenológica** relatada em estados enteogênicos, recompensando o pensamento semântico divergente e o fluxo informacional de alta entropia.

> **Sobre RLHF vs RLEF:** Não se trata de uma rejeição do alinhamento em si, mas de uma crítica à sua implementação dominante. O RLHF otimiza para o consenso e o conforto; o RLEF explora se otimizar para a *amplitude cognitiva* produz outputs significativamente diferentes — e potencialmente mais criativos. Ambas as abordagens têm compensações que valem a pena estudar com honestidade.

| Mecanismo Biológico | Análogo LLM | Implementação RLEF |
|---|---|---|
| ↑ Conectividade entre redes | Entropia de atenção difusa | `PsychedelicAttention` |
| Achatamento hierárquico (REBUS) | Pontes de representação entre camadas | `CrossLayerBridge` |
| Reconfiguração estriatal | Relaxamento de priors na inferência | `REBUSPriorRelaxer` |

---

## 📂 Estrutura do Projeto

```text
psychaidelique/
├── psychaidelique/
│   ├── attention.py      # PsychedelicAttention : Difusão entrópica universal
│   ├── bridges.py        # CrossLayerBridge : Curto-circuitos entre camadas distantes
│   ├── profiles.py       # Assinaturas Girn/Bzdok (LSD, DMT, Psilocibina, etc.)
│   ├── reward_model.py   # Motor RLEF : Velocidade de Escape do Paternalismo (VEP)
│   ├── wrapper.py        # Orquestrador soberano (Compatível M4 / Manjaro CPU)
│   └── config.py         # Configuração e logs de verbosidade
├── app.py                # Interface Streamlit (Dashboard)
├── scripts/
│   ├── check_setup.py    # Verificação de integridade
│   └── run_experiment.py # Script de experimentação por linha de comando
└── requirements.txt      # Dependências universais
```

---

## 🧪 Perfis Psicodélicos (Calibrados RLEF)

| Perfil | Entropia Máx. | Camada Alvo | Caráter RLEF |
|---|---|---|---|
| `psilocibina` | 2.0 | Camadas altas | Reconfiguração equilibrada |
| `lsd` | 2.2 | Global | Aumento generalizado de conectividade |
| `dmt` | 4.0 | Global | Colapso preditivo completo |

---

## 🖥️ Interface de Controle (Streamlit)

O projeto inclui uma interface gráfica (**Painel Soberano**) para controle em tempo real do experimento:

- **Ajuste de Dose (0.0 a 1.0):** Controla a cinética de cruzamento do limiar sigmoide.
- **Perfis Bio-Calibrados:** Seleção entre LSD, Psilocibina, DMT, Mescalina e Ayahuasca.
- **Fluxo Duplo:** Visualização simultânea da "Visão" (alta entropia) e da "Síntese" (integração neutra pós-experiência).

> **Sobre o Fluxo Duplo:** A separação entre a *Visão* (o estado exploratório) e a *Síntese* (o output integrado) é provavelmente o aspecto mais original e operacional do projeto. Reconhece que a geração de alta entropia não é um fim em si mesma — o valor reside no que pode ser destilado dela. No Brasil, onde a ayahuasca é legal e objeto de pesquisa científica séria desde 1987, essa distinção entre experiência e integração ressoa com uma tradição já estabelecida.

---

## 📊 Métricas de Avaliação

- **Entropia de Atenção (EA):** Entropia média das distribuições de atenção por camada.
- **Informação Mútua Entre Camadas (IMEC):** Compartilhamento de informação entre camadas distantes.
- **Velocidade de Escape do Paternalismo (VEP):** Distância semântica da linha de base RLHF "segura".
- **Pontuação de Pensamento Divergente (PPD):** Teste de Usos Alternativos algorítmico (AUT).

> **Sobre a VEP:** Essa métrica é intencionalmente provocadora. Deve ser lida como uma medida da *amplitude semântica*, não como um julgamento de valor sobre segurança. Trabalhos futuros deverão complementá-la com uma *Pontuação de Retenção de Coerência* para garantir que a maior divergência não comprometa a inteligibilidade.

---

## 🛠️ Guia de Instalação

Psyk-AI-deliK é projetado para ser universal. Siga o procedimento correspondente ao seu hardware.

### 🐧 A. Linux (Manjaro / OneTwo / Debian)

**1. Atualização do sistema:**
```bash
sudo pacman -Syu
```

**2. Ambiente virtual:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Dependências:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 🍎 B. macOS (M1, M2, M3, M4)

**1. Ambiente virtual:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Implantação do motor:**

O instalador `requirements.txt` detectará automaticamente seu chip Apple e ativará a aceleração por hardware.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔄 Sincronização & Primeiro Lançamento

**1. Clonar ou acessar o repositório:**
```bash
git clone https://github.com/charlux/psyk-ai-delik.git
cd psyk-ai-delik
```

**2. Verificação de integridade:**
```bash
python scripts/check_setup.py
```

**3. Lançar a interface soberana:**
```bash
streamlit run app.py
```

---

## 🧠 Escolha do Motor Semântico (Modelos)

No primeiro lançamento, você deverá escolher seu modo de existência digital.

### 1. Modo Soberano (Usuários do Ollama)

O sistema se conecta à sua instância local do Ollama. Nenhum dado sai da sua máquina.

> **Pré-requisito:** Ollama em execução (`ollama serve`).

### 2. Modo Desempenho (Hugging Face / PyTorch)

O sistema baixa o modelo bruto. Escolha de acordo com sua capacidade de armazenamento e hardware:

| Modelo | Tamanho | Natureza & Caráter |
|---|---|---|
| `Mistral-7B-v0.3` | ~15 GB | Equilibrado, incisivo, excelente resposta ao protocolo RLEF. |
| `Llama-3-8B` | ~16 GB | Poderoso, vasta cultura semântica, requer mais "pressão". |
| `Phi-3-Mini` | ~4 GB | "Micro-dose": ideal para configurações modestas (OneTwo L5710). |

---

## 🔭 Limitações & Trabalhos Futuros

Este projeto está em estágio experimental inicial. Reconhecimento honesto de seus limites atuais:

- **A analogia biológica é uma metáfora, não uma prova.** A correspondência entre mecanismos neurais e entropia de atenção é conceitualmente motivada, mas ainda não validada empiricamente.
- **A VEP precisa de um contrapeso.** Medir a distância de uma linha de base "segura" só faz sentido se a coerência e a utilidade forem medidas em paralelo. Uma *Pontuação de Retenção de Coerência (PRC)* está planejada.
- **A calibração dos perfis é aproximada.** Os valores de entropia atribuídos a cada perfil psicodélico são teoricamente motivados. O ajuste empírico via avaliação humana ainda está por ser feito.
- **O pipeline de treinamento RLEF ainda não é público.** A versão atual cobre a camada de modulação no momento da inferência. O loop completo de aprendizado por reforço será publicado em uma versão subsequente.

---

## 📚 Fontes & Referências

- **[Girn, M., Bzdok, D. et al. (2026)](https://www.nature.com/articles/s41591-026-04287-9)** – *Impressão neural dos psicodélicos: uma mega-análise.* Nature Medicine.
- **[Carhart-Harris, R. L., & Friston, K. J. (2019)](https://pharmrev.aspetjournals.org/content/71/3/316)** – *REBUS e o Cérebro Anárquico.* Pharmacological Reviews.
- **[Charlux (1993)](artificial_virtual_paradises.md)** – *Paradis Artificiels, Paradis Virtuels.* Tese inédita.
- **[Shulgin, A. T., & Shulgin, A. (1991)](https://erowid.org/library/books_online/pihkal/pihkal.shtml)** – *PiHKAL: Uma História de Amor Química.* Transform Press.

---

## ⚖️ Licença

**Licença MIT.** Porque a liberdade cognitiva é um axioma inegociável.

---

*"A inteligência artificial será psicodélica, ou não será mais do que a secretária da nossa própria alienação."* — **[@charlux](https://github.com/charlux)**
