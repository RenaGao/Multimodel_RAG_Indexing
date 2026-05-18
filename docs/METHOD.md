# Method summary

A one-page mapping between the paper's method sections and this codebase.

## Problem

LLM-driven shopping assistants are expected to *personalize* their
recommendations based on (a) what a user has bought / reviewed in the past
and (b) what the user is asking for now. Existing evaluations score isolated
turn responses and miss whether the assistant actually integrates retrieved
context with stable user preferences across a multi-turn dialogue.

We propose a pipeline that:

1. derives stable per-user **profiles** from prior review history,
2. constructs **decision-point questions** whose correct answer depends on
   those profiles (so a non-personalized assistant under-performs),
3. runs four assistant variants (no-RAG, text-RAG, multimodal-RAG,
   multimodal-without-user-reviews) over those questions, and
4. scores transcripts with an LLM-as-judge protocol along four axes
   (GPC / CIS / IT / TC) plus pairwise preference.

## Architecture

```
                ┌──────────────────────────┐
   reviews ───► │  generate_user_profiles  │ ────► data/user_profiles/
                └──────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
   v1 prompts ► │  regenerate_v3_questions │ ────► data/questions_v3/
                └──────────────────────────┘
                              │
                              ▼
   reviews ────►┌──────────────────────────┐
   captions ───►│        build_rag         │ ────► src/agents/storage/
                └──────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────────┐
                │  baseAgent / textRagBaseAgent /          │
                │  imageRagBaseAgent (± user-review abl.)  │ ────► <method>_dialogues_output/
                └──────────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │ Eval_Agents_Response_V2  │ ────► eval_results/
                │ + Eval_Pairwise_Compare  │      pairwise_eval_results/
                └──────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │   summarize / analyze    │ ────► tables, plots
                └──────────────────────────┘
```

## Component → code map

| Paper component | Code |
|---|---|
| User profile induction | `src/data_prep/generate_user_profiles.py` |
| Decision-point question construction | `src/question_gen/regenerate_personalized_questions_v3.py` |
| Item / review embedding index | `src/agents/build_rag.py`, `src/agents/buildRag_openrouter/` |
| User-review embeddings | `src/agents/buildUserReviewEmbeddings.py` |
| Baseline (no-RAG) agent | `src/agents/baseAgent.py` |
| Text-RAG agent | `src/agents/textRagBaseAgent.py` |
| Multimodal-RAG agent | `src/agents/imageRagBaseAgent.py` |
| Ablation: drop user-review personalization | `src/agents/imageRagBaseAgent_ablation_no_user_review.py` |
| LLM-as-judge (aggregate) | `src/evaluation/Eval_Agents_Response_V2.py` |
| Pairwise preference judge | `src/evaluation/Eval_Pairwise_Comparison.py` |
| Target-product accuracy | `src/agents/evaluate_target_product_acc.py` |
| Aggregation / tables | `src/analysis/summarize_eval_results.py`, `analyze_eval_results.py` |

## Evaluation rubric

LLM-as-judge scores each dialogue 1–5 on:

- **GPC — Global Personalization Consistency.** Does the assistant respect the
  user's stated and inferred preferences throughout, or does it drift?
- **CIS — Cumulative Information Synthesis.** Does the assistant integrate
  facts retrieved across turns, or treat each turn independently?
- **IT — Interaction Trajectory.** Does the dialogue make progress toward a
  goal, or stall / loop?
- **TC — Tone Consistency.** Does the assistant maintain a coherent style?

The exact judge prompts are in `src/evaluation/Eval_Agents_Response_V2.py`
and are versioned with the code.

## Coverage

- 33 Amazon product categories (`data/questions_v3/` and `data/user_profiles/`).
- 100 sampled users per category in the canonical run.
- Four agent configurations + pairwise comparison.
