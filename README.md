# 🧠 Personalized Agent Evaluation and Question Generation Framework

This repository contains a modular workflow for **evaluating AI shopping assistants and personalized dialogue agents**.  
It integrates **Retrieval-Augmented Generation (RAG)**, **user profiling**, and **multilingual CEFR-based dialogue evaluation**.  
The pipeline supports both automated evaluation (LLM-driven) and dynamic question generation for personalization testing.

---

## 🚀 Overview

The system is designed to:
- **Generate user-specific profiles** and personalized shopping questions.
- **Run multi-turn dialogue evaluations** using LLMs (GPT, Claude, Qwen, etc.).
- **Assess model responses** across metrics such as *Global Personalization Consistency*, *Cumulative Information Synthesis*, *Interaction Trajectory*, and *Tone Consistency*.
- **Support multilingual learner evaluations** with CEFR-level alignment (A1–C2).

This repository combines both **agent evaluation pipelines** and **personalized dataset generation utilities**.

---

## 📂 Repository Structure

| Folder/File | Description |
|--------------|-------------|
| **AgentWorkflow/** | Core prompt workflows and pipeline definitions for evaluating agent behavior. |
| **questions_v3/** | Redundant and domain-specific datasets (e.g., Amazon Fashion). |
| **user_profiles/** | Example user profiles for different demographics and preference categories. |
| **Eval\_Agents\_Responses.py** | Baseline evaluation pipeline for multi-turn agent scoring. |
| **Eval\_Agents\_Response\_V2.py** | New version with integrated RAG (Retrieval-Augmented Generation) and CEFR-style evaluation. |
| **Eval\_Agents\_Response\_V2\_backup.py** | Backup of the updated pipeline (before refactor). |
| **regenerate\_personalized\_questions\_v3.py** | Experimental generation version (v3 questions). |
| **generate\_shopping\_questions.py** | Template-driven generation of shopping-related conversational prompts. |
| **analyze\_item\_review\_distribution.py** | Statistical visualization of item reviews for distribution and coverage analysis. |
| **vanilla\_rag.py** | Simple single-query Retrieval-Augmented Generation implementation for baseline comparison. |
| **requirements.txt** | Python dependencies (LLM, FAISS/Chroma, visualization, etc.). |
| **README.md** | Project documentation (this file). |

---

## 🧩 Core Components

### 1️⃣ User Profiling
Scripts under `generate_user_profiles.py` and `user_profiles/` build diverse persona sets capturing:
- Demographics (age, gender, region)
- Preferences (style, price sensitivity, tone preference)
- Communication behaviors (direct vs. exploratory)

### 2️⃣ Question Generation
The `questions_v*` modules generate personalized or domain-specific dialogue questions using profile-based templates.  
Each iteration refines diversity and relevance:
- **v1:** static baseline  
- **v2:** user profile-driven  
- **v3:** fashion-specific redundancy testing

### 3️⃣ Evaluation Pipeline
The `Eval_Agents_Responses*` scripts implement:
- LLM-based scoring of dialogue sessions  
- Aggregate evaluation metrics (GPC, CIS, IT, TC)  
- Optional CEFR-based multilingual learner assessment

The evaluation prompt is standardized for reproducibility and supports JSON-based structured output.

---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **GPC** | Global Personalization Consistency – retention of user-specific constraints. |
| **CIS** | Cumulative Information Synthesis – coherence and factual integration across turns. |
| **IT** | Interaction Trajectory – goal-oriented flow and conversational structure. |
| **TC** | Tone Consistency – stylistic and persona stability across dialogue. |

Each metric is rated on a **1–5 scale**, aligned with human-annotated qualitative rubrics.

---

## 🧠 CEFR Multilingual Extension
A parallel evaluation framework supports **multilingual dialogues** (e.g., English–Chinese).  
It assesses learner-level appropriateness using CEFR descriptors:
- **Range**, **Accuracy**, **Fluency**, **Interaction**, **Coherence**
- Supports role-based scoring (Tutor vs. Learner)
- Output in strict JSON format for downstream processing

Example:  
`Eval_Agents_Response_V2.py` integrates this logic to evaluate learner dialogues automatically.

---

## ⚙️ Setup and Usage

### Installation
```bash
git clone https://github.com/<...>.git
cd personalized-agent-eval
pip install -r requirements.txt
