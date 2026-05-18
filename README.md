# Multimodal RAG-Indexed Personalized Shopping Agents

This repository accompanies the paper *Multimodal RAG Indexing for Personalized
Shopping-Agent Evaluation*. It contains the full pipeline for:

1. building **per-user profiles** from review history,
2. generating **decision-point shopping dialogues** that probe personalization,
3. running **baseline / text-RAG / multimodal-RAG** shopping agents over those
   dialogues, and
4. **evaluating** the resulting transcripts with an LLM-as-judge protocol over
   four metrics (GPC, CIS, IT, TC), plus pairwise comparison.

See [`docs/METHOD.md`](docs/METHOD.md) for a short method summary and
[`docs/REPRODUCE.md`](docs/REPRODUCE.md) for end-to-end reproduction commands.

---

## Repository layout

```
.
├── src/
│   ├── data_prep/         # user-profile + base question generation
│   ├── question_gen/      # canonical v3 personalized-question regeneration
│   ├── rag/               # baseline single-query RAG
│   ├── agents/            # base / text-RAG / multimodal-RAG agents + ablation
│   ├── evaluation/        # LLM-as-judge (V2) and pairwise eval
│   └── analysis/          # result summarization, distribution analysis
├── data/
│   ├── questions_v3/      # 33 categories × personalized questions (canonical)
│   ├── user_profiles/     # 33 categories × generated user profiles
│   └── samples/RagData/   # tiny sample for quickstart / smoke tests
├── docs/                  # METHOD.md, REPRODUCE.md, flowexplanation.md
└── archive/               # superseded versions, preserved for provenance
    ├── question_gen_v1/, question_gen_v1_data/, question_gen_v2/
    ├── eval_v1/           # earlier eval scripts (V1 + V2 pre-refactor backup)
    ├── rag_legacy/        # pre-refactor RAG builders
    ├── debug/             # one-off rerun / analysis utilities
    └── sample_outputs/    # dated example agent outputs
```

The `archive/` tree is not part of the canonical pipeline; nothing under
`src/` imports from it.

---

## Quickstart

### 1. Environment

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# API keys (used by the agents and judges)
cp .env.example .env
# fill in OPENAI_API_KEY (and QWEN_API_KEY / OPENROUTER_API_KEY if used)
```

The processed Amazon review corpus is **not** in this repo. Point the pipeline
at it via an env var before generating profiles or building RAG indices:

```bash
export USER_REVIEWS_DIR=/path/to/AmazonReviews/Processed/user_center
```

### 2. Smoke test (single category, sample data)

```bash
# generate / refresh personalized questions for one category
python src/question_gen/regenerate_personalized_questions_v3.py \
    --input data/questions_v3/All_Beauty_questions_*.json

# run the baseline agent on a single category
python src/agents/baseAgent.py --category All_Beauty

# evaluate the resulting dialogues
python src/evaluation/Eval_Agents_Response_V2.py \
    --input src/agents/baseline_dialogues_output \
    --output-dir eval_results/baseline
```

---

## Pipeline

End-to-end execution (full benchmark, all 33 categories):

| Stage | Script | Inputs | Outputs |
|---|---|---|---|
| 1. User profiles | `src/data_prep/generate_user_profiles.py --all` | `$USER_REVIEWS_DIR`, `data/questions_v3/` | `data/user_profiles/` |
| 2. Personalized questions (v3) | `src/question_gen/regenerate_personalized_questions_v3.py --all` | `archive/question_gen_v1_data/questions/`, `data/user_profiles/` | `data/questions_v3/` |
| 3. RAG indices | `src/agents/build_rag.py` | `$USER_REVIEWS_DIR` | `src/agents/storage/vectorstore/` |
| 4a. Baseline agent | `src/agents/baseAgent.py --all` | `data/questions_v3/`, RAG store | `src/agents/baseline_dialogues_output/` |
| 4b. Text-RAG agent | `src/agents/textRagBaseAgent.py --all` | same | `src/agents/text_rag_dialogues_output/` |
| 4c. Multimodal-RAG agent | `src/agents/imageRagBaseAgent.py --all` | same + image captions | `src/agents/multimodal_dialogues_output/` |
| 4d. Ablation (no user reviews) | `src/agents/imageRagBaseAgent_ablation_no_user_review.py --all` | same | `src/agents/multimdoal_no_user_review_output/` |
| 5. Aggregate evaluation | `src/evaluation/Eval_Agents_Response_V2.py` | per-method dialogue dir | `eval_results/<method>/` |
| 5b. Pairwise comparison | `src/evaluation/Eval_Pairwise_Comparison.py` | two dialogue dirs | `pairwise_eval_results/` |
| 6. Summarize | `src/analysis/summarize_eval_results.py` | `eval_results/` | summary CSVs / tables |

Full commands with required arguments and expected runtimes are in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

---

## Evaluation metrics

| Metric | Name | What it measures |
|---|---|---|
| **GPC** | Global Personalization Consistency | Retention of user-specific constraints across the dialogue |
| **CIS** | Cumulative Information Synthesis | Coherence and factual integration across turns |
| **IT** | Interaction Trajectory | Goal-oriented flow and conversational structure |
| **TC** | Tone Consistency | Stylistic and persona stability across the dialogue |

Each metric is rated 1–5 by an LLM-as-judge under a fixed rubric (see
`src/evaluation/Eval_Agents_Response_V2.py` for the exact prompt). Pairwise
comparison provides a complementary direct-preference signal.

---

## Dependencies

Pinned in [`requirements.txt`](requirements.txt). Tested with Python 3.10.
The agent pipeline calls OpenAI / OpenRouter APIs — set the corresponding
environment variables in `.env` (template in `.env.example`).

---

## License

To be added.
