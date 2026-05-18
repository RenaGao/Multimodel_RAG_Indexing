# Reproducing paper results

This document gives the exact commands to reproduce the main result tables. All
commands assume the working directory is the repository root and that the
environment is set up per the **Quickstart** in [`../README.md`](../README.md).

> **Inputs you must provide.** The processed Amazon review corpus (per-category
> JSON files of the form `<Category>_top_100_user_data.json` and
> `<Category>_top_100_user_itemReview.json`) is not redistributed in this
> repository. Set `USER_REVIEWS_DIR` to the directory containing those files.
> A two-category sample is provided under `data/samples/RagData/` for smoke
> testing.

```bash
export USER_REVIEWS_DIR=/path/to/AmazonReviews/Processed/user_center
# Optional: place .env in repo root with OPENAI_API_KEY / OPENROUTER_API_KEY
```

---

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 is the tested version. The pinned dependency set is in
`requirements.txt`.

---

## 1. Generate user profiles (≈ 2 hr on full corpus, GPT-4o)

```bash
python src/data_prep/generate_user_profiles.py --all
```

- Reads category list from `data/questions_v3/` (set with `--questions-dir`).
- Reads review history from `$USER_REVIEWS_DIR` (set with `--user-data-dir`).
- Writes `data/user_profiles/<Category>_user_profiles.json` (one file per
  category).
- Single-category mode: `--questions <path> --user-data <path> --output <path>`.

Skip this step if you want to use the profiles already shipped in
`data/user_profiles/`.

---

## 2. Regenerate v3 personalized questions (≈ 1 hr, GPT-4o)

```bash
python src/question_gen/regenerate_personalized_questions_v3.py --all
```

- Uses base v1 questions from `archive/question_gen_v1_data/questions/`.
- Uses precomputed profiles from `data/user_profiles/`.
- Writes timestamped files to `data/questions_v3/`.
- Skips categories already present in `data/questions_v3/`.
- Single category: `--input <questions_file>`.

The v3 questions already in `data/questions_v3/` are the ones used in the
paper; rerunning will produce new timestamped files alongside them.

---

## 3. Build RAG indices (≈ 30 min/category on CPU)

```bash
# All canonical categories
python src/agents/build_rag.py --all

# A single category
python src/agents/build_rag.py --category Amazon_Fashion

# Caption (image) embeddings for the multimodal agent
python src/agents/build_rag.py --captions --category Amazon_Fashion
```

- Reads source data from `$USER_REVIEWS_DIR` (`DATA_DIR` in
  `src/agents/buildRag_openrouter/config.py`).
- Writes FAISS indices and embeddings under `src/agents/storage/vectorstore/`.
- Caption embeddings expect image captions at
  `image_captions/<Category>_products_meta.jsonl`.

---

## 4. Run shopping agents

Each agent writes one JSON per user per category into its own output directory.
All four can be run on a single category (`--category <name>`) or batch mode
(`--all`).

```bash
# 4a. Baseline (no RAG)
python src/agents/baseAgent.py --all

# 4b. Text-RAG
python src/agents/textRagBaseAgent.py --all

# 4c. Multimodal RAG (text + item reviews + image captions)
python src/agents/imageRagBaseAgent.py --all

# 4d. Ablation: multimodal without user-review personalization
python src/agents/imageRagBaseAgent_ablation_no_user_review.py --all
```

Output directories:

| Agent | Output dir |
|---|---|
| baseline | `src/agents/baseline_dialogues_output/<Category>/` |
| text-RAG | `src/agents/text_rag_dialogues_output/<Category>/` |
| multimodal | `src/agents/multimodal_dialogues_output/<Category>/` |
| ablation | `src/agents/multimdoal_no_user_review_output/<Category>/` |

Approximate cost: ≈ 5–10 minutes per category on a single GPU node for each
agent, plus API spend (depends on model choice — defaults in each script).

---

## 5. Evaluation

### 5a. Aggregate LLM-as-judge (GPC / CIS / IT / TC)

```bash
for method in baseline text_rag multimodal multimdoal_no_user_review; do
  python src/evaluation/Eval_Agents_Response_V2.py \
      --input "src/agents/${method}_dialogues_output" \
      --output-dir "eval_results/${method}"
done
```

(Adjust the directory names to match your actual agent output paths.)

Each run writes per-category JSON files into `eval_results/<method>/`.

### 5b. Pairwise comparison (e.g., multimodal vs ablation)

```bash
python src/evaluation/Eval_Pairwise_Comparison.py \
    --input-a src/agents/multimodal_dialogues_output \
    --input-b src/agents/multimdoal_no_user_review_output \
    --label-a multimodal \
    --label-b no_user_review \
    --output-dir pairwise_eval_results
```

### 5c. Target-product accuracy

```bash
python src/agents/evaluate_target_product_acc.py \
    --dialogue-dir src/agents/multimodal_dialogues_output \
    --output target_product_acc_results.csv
```

---

## 6. Aggregate and analyze

```bash
# Summary tables across methods
python src/analysis/summarize_eval_results.py --eval-dir eval_results

# A/B compare two methods
python src/analysis/analyze_eval_results.py \
    --dir-a eval_results/baseline \
    --dir-b eval_results/multimodal

# Item / review distribution charts (paper appendix)
python src/analysis/analyze_item_review_distribution.py
```

---

## Re-evaluating problematic dialogues

If a small set of dialogues failed to generate or evaluate cleanly, the list
lives in `archive/debug/problematic_dialogues.csv`. To re-run just those:

```bash
python src/evaluation/reeval_problematic_dialogues.py
```

The defaults already point at `archive/debug/problematic_dialogues.csv` and
the agent output / eval / profile directories described above.
