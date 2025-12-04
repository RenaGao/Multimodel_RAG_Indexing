import json
import re
import pickle
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from buildRag_old import build_user_index_for
import torch
import os
import threading
from openai import OpenAI  # For OpenRouterEmbedder (multimodal RAG)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global model cache with thread-safe access
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()

BASE_DIR   = Path("./storage")
VSTORE_DIR = BASE_DIR / "vectorstore"
CATALOG_DIR = VSTORE_DIR / "catalog"
USERS_DIR   = VSTORE_DIR / "users"

# Dataset category - configurable via set_dataset_category()
_DATASET_CATEGORY = "Clothing_Shoes_and_Jewelry"  # Default value

def set_dataset_category(category: str):
    """Set the current dataset category."""
    global _DATASET_CATEGORY
    _DATASET_CATEGORY = category
    print(f"[ragFunctions] Dataset category set to: {category}")

def get_dataset_category() -> str:
    """Get the current dataset category."""
    return _DATASET_CATEGORY

for p in [VSTORE_DIR, CATALOG_DIR, USERS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def clear_user_storage():
    """Clear entire user storage directory."""
    if USERS_DIR.exists():
        shutil.rmtree(USERS_DIR)
        USERS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[CLEAR] Cleared entire USERS_DIR")

def item_index_dir() -> Path:
    return CATALOG_DIR / get_dataset_category() / "item_index_qwen3"

def user_review_embeddings_dir() -> Path:
    """Get directory for user review embeddings."""
    return CATALOG_DIR / get_dataset_category() / "user_review_embeddings"

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _norm(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _safe_user_dir(user_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", user_name or "anonymous")

def _user_paths(user_name: str) -> Dict[str, Path]:
    udir = USERS_DIR / _safe_user_dir(user_name)
    udir.mkdir(parents=True, exist_ok=True)
    return {
        "root": udir,
        "user_index": udir / "user_index",
        "chat_history": udir / "chat_history.jsonl",
        "round_embeddings": udir / "round_embeddings.pkl",
        "round_metadata": udir / "round_metadata.pkl",
        "last_round": udir / "last_round.json",
        "sampled_reviews": udir / "sampled_reviews.json",
    }

def _iter_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            yield json.loads(line)

# def _sample_user_reviews(user_name: str, num_reviews: int = 5) -> Optional[Dict]:
#     """
#     Sample random reviews for a user from Amazon_Fashion_top_100_user_data.json.

#     Args:
#         user_name: User identifier
#         num_reviews: Number of reviews to sample (default: 5)

#     Returns:
#         Dict with sampled review data or None if user not found
#     """
#     # Path to user data file (in project root)
#     user_data_path = Path("/data/gpfs/projects/punim2620/Multimodel_RAG_Indexing/AmazonReviews/Processed/user_center/Amazon_Fashion_top_100_user_data.json")

#     if not user_data_path.exists():
#         print(f"[SAMPLE_REVIEWS] User data file not found: {user_data_path}")
#         return None

#     try:
#         # Load user data
#         with user_data_path.open("r", encoding="utf-8") as f:
#             user_data = json.load(f)

#         if user_name not in user_data:
#             print(f"[SAMPLE_REVIEWS] User {user_name} not found in user data")
#             return None

#         # Extract all reviews for this user
#         all_reviews = []
#         user_items = user_data[user_name]

#         for item_tuple in user_items:
#             if not isinstance(item_tuple, list) or len(item_tuple) < 2:
#                 continue

#             product_meta = item_tuple[0]
#             review_list = item_tuple[1]

#             # Add product context to each review
#             for review in review_list:
#                 enriched_review = {
#                     "review_text": review.get("text", ""),
#                     "rating": review.get("rating"),
#                     "title": review.get("title", ""),
#                     "product_title": product_meta.get("title", ""),
#                     "product_brand": product_meta.get("store", ""),
#                     "product_price": product_meta.get("price"),
#                     "timestamp": review.get("timestamp", ""),
#                 }
#                 all_reviews.append(enriched_review)

#         if not all_reviews:
#             print(f"[SAMPLE_REVIEWS] No reviews found for user {user_name}")
#             return None

#         # Sample reviews (with reproducible random seed)
#         import random
#         random.seed(42)
#         sampled = random.sample(all_reviews, min(num_reviews, len(all_reviews)))

#         result = {
#             "user_id": user_name,
#             "sampled_at": _now(),
#             "num_reviews": len(sampled),
#             "total_available_reviews": len(all_reviews),
#             "reviews": sampled
#         }

#         print(f"[SAMPLE_REVIEWS] Sampled {len(sampled)} reviews for user {user_name}")
#         return result

#     except Exception as e:
#         print(f"[SAMPLE_REVIEWS ERROR] {e}")
#         return None

def _load_sampled_reviews(user_name: str) -> Optional[Dict]:
    """
    Load previously sampled reviews for a user.

    Args:
        user_name: User identifier

    Returns:
        Dict with sampled review data or None if not found
    """
    up = _user_paths(user_name)

    if not up["sampled_reviews"].exists():
        return None

    try:
        with up["sampled_reviews"].open("r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[LOAD_REVIEWS] Loaded {data.get('num_reviews', 0)} sampled reviews for user {user_name}")
        return data
    except Exception as e:
        print(f"[LOAD_REVIEWS ERROR] {e}")
        return None

def _save_sampled_reviews(user_name: str, sampled_data: Dict):
    """
    Save sampled reviews for a user.

    Args:
        user_name: User identifier
        sampled_data: Dict containing sampled review data
    """
    up = _user_paths(user_name)

    try:
        with up["sampled_reviews"].open("w", encoding="utf-8") as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=2)
        print(f"[SAVE_REVIEWS] Saved {sampled_data.get('num_reviews', 0)} sampled reviews for user {user_name}")
    except Exception as e:
        print(f"[SAVE_REVIEWS ERROR] {e}")

def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = (a * a).sum(axis=1) ** 0.5
    nb = (b * b).sum(axis=1) ** 0.5
    sim = (a @ b.T) / (na[:, None] * nb[None, :] + 1e-12)
    return sim

class _Index:
    def __init__(self, index_dir: Path):
        self.dir = index_dir
        self.meta = index_dir / "meta.pkl"
        self.mat  = index_dir / "matrix.pkl"
        self.review_mat = index_dir / "review_matrix.pkl"
        self.docs: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        self.model: Optional[SentenceTransformer] = None
        self.matrix: Optional[np.ndarray] = None
        self.review_matrix: Optional[np.ndarray] = None

    def exists(self) -> bool:
        return self.meta.exists() and self.mat.exists()

    def load(self):
        if not self.exists():
            raise RuntimeError(f"Index not found under {self.dir}")

        d = pickle.load(self.meta.open("rb"))
        self.docs = d["docs"]
        self.texts = d["texts"]

        # Use cached model if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_key = f"Qwen/Qwen3-Embedding-8B:{device}"

        with _MODEL_CACHE_LOCK:
            if model_key not in _MODEL_CACHE:
                print(f"[LOAD] Loading new SentenceTransformer model: {model_key}")
                _MODEL_CACHE[model_key] = SentenceTransformer(
                    "Qwen/Qwen3-Embedding-8B",
                    device=device
                )
                print(f"[LOAD] Model loaded and cached successfully")
            else:
                print(f"[LOAD] Reusing cached SentenceTransformer model: {model_key}")

        self.model = _MODEL_CACHE[model_key]
        self.matrix = pickle.load(self.mat.open("rb"))

        # Load review matrix if available (v2 indices)
        if self.review_mat.exists():
            self.review_matrix = pickle.load(self.review_mat.open("rb"))
            print(f"[LOAD] Review matrix loaded: {self.review_matrix.shape}")
        else:
            print(f"[INFO] No review_matrix.pkl found - using metadata embeddings only")

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        if not query.strip() or self.matrix is None:
            return []
        qv = self.model.encode([query], normalize_embeddings=True)
        sims = _cosine(qv, self.matrix).ravel()
        order = sims.argsort()[::-1]
        return [(int(i), float(sims[i])) for i in order[:max(1, k)]]

    def doc(self, i: int) -> Dict[str, Any]:
        return self.docs[i]

def _generate_round_embedding(query: str, model: SentenceTransformer) -> np.ndarray:
    """Generate embedding for a user query."""
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        prompt_name="query",
        batch_size=1
    )
    return embedding[0]  # Return single vector

def _load_round_embeddings(user_name: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """Load round embeddings and metadata for a user."""
    up = _user_paths(user_name)

    if not up["round_embeddings"].exists() or not up["round_metadata"].exists():
        return None, None

    try:
        with up["round_embeddings"].open("rb") as f:
            embeddings = pickle.load(f)
        with up["round_metadata"].open("rb") as f:
            metadata = pickle.load(f)
        return embeddings, metadata
    except Exception as e:
        print(f"[ERROR] Loading round embeddings: {e}")
        return None, None

def _load_user_review_embeddings(user_name: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Load user review embeddings and metadata from catalog directory.

    Args:
        user_name: User identifier

    Returns:
        Tuple of (embeddings_matrix, metadata_dict) or (None, None) if not found
    """
    # Load from catalog directory
    review_embs_dir = user_review_embeddings_dir()
    emb_path = review_embs_dir / f"{user_name}_review_embs.pkl"
    meta_path = review_embs_dir / f"{user_name}_review_meta.pkl"

    if not emb_path.exists() or not meta_path.exists():
        return None, None

    try:
        with emb_path.open("rb") as f:
            embeddings = pickle.load(f)
        with meta_path.open("rb") as f:
            metadata = pickle.load(f)
        print(f"[LOAD_REVIEW_EMBS] Loaded {len(embeddings)} review embeddings for user {user_name}")
        return embeddings, metadata
    except Exception as e:
        print(f"[LOAD_REVIEW_EMBS ERROR] {e}")
        return None, None


def _select_relevant_reviews(user_query: str, user_name: str, top_k: int = 5) -> List[Dict]:
    """
    Select top-k most relevant reviews based on query similarity.

    Args:
        user_query: Current user query
        user_name: User identifier
        top_k: Number of reviews to select (default: 5)

    Returns:
        List of selected review dicts with similarity scores
    """
    # Load user review embeddings
    review_embs, metadata = _load_user_review_embeddings(user_name)

    # if review_embs is None:
    #     # Fallback to old fixed sampling method
    #     print(f"[SELECT_REVIEWS] No review embeddings found, using fallback sampling")
    #     sampled_data = _sample_user_reviews(user_name, num_reviews=top_k)
    #     if sampled_data:
    #         return sampled_data.get("reviews", [])
    #     return []

    # Load item index model to encode query
    iidx_dir = item_index_dir()
    iidx = _Index(iidx_dir)
    iidx.load()

    # Generate query embedding
    query_emb = iidx.model.encode([user_query], normalize_embeddings=True)

    # Calculate similarities
    sims = _cosine(query_emb, review_embs).ravel()

    # Get top-k indices
    top_indices = sims.argsort()[::-1][:min(top_k, len(sims))]

    # Extract top-k reviews
    selected_reviews = []
    for idx in top_indices:
        review_data = metadata["reviews"][idx].copy()
        review_data["similarity_score"] = float(sims[idx])
        selected_reviews.append(review_data)

    print(f"[SELECT_REVIEWS] Selected {len(selected_reviews)} reviews based on query similarity")
    if selected_reviews:
        print(f"[SELECT_REVIEWS] Top similarity score: {selected_reviews[0]['similarity_score']:.4f}")

    return selected_reviews


def _append_round_embedding(user_name: str, query: str, round_id: str, round_number: int):
    """Append a new query embedding to user's embedding matrix."""
    up = _user_paths(user_name)

    # Load item index model to generate embedding
    iidx_dir = item_index_dir()
    iidx = _Index(iidx_dir)
    iidx.load()

    # Generate embedding
    query_emb = _generate_round_embedding(query, iidx.model)

    # Load existing embeddings or create new
    existing_embs, metadata = _load_round_embeddings(user_name)

    if existing_embs is None:
        # First round
        new_embs = query_emb.reshape(1, -1)
        metadata = {
            "round_ids": [round_id],
            "round_number_to_idx": {round_number: 0},
            "embedding_dim": query_emb.shape[0],
            "model": "Qwen/Qwen3-Embedding-8B",
            "last_updated": _now()
        }
    else:
        # Append to existing
        new_embs = np.vstack([existing_embs, query_emb.reshape(1, -1)])
        metadata["round_ids"].append(round_id)
        metadata["round_number_to_idx"][round_number] = len(existing_embs)
        metadata["last_updated"] = _now()

    # Save
    with up["round_embeddings"].open("wb") as f:
        pickle.dump(new_embs, f)
    with up["round_metadata"].open("wb") as f:
        pickle.dump(metadata, f)

def save_chat_history(chat_history: str, user_name: str) -> bool:
    """
    Save chat history with round-based storage and query embeddings.
    Detects [USER_QUESTION_N] markers to identify new rounds.
    """
    try:
        up = _user_paths(user_name)

        # Check if this is a user question (new round start)
        import re
        user_question_match = re.match(r'\[USER_QUESTION_(\d+)\]\s+\[([^\]]+)\]\s+(.*)', chat_history)

        if user_question_match:
            # This is a new round starting
            round_number = int(user_question_match.group(1))
            agent_name = user_question_match.group(2)
            user_query = user_question_match.group(3)

            round_id = f"round_{round_number:03d}"
            timestamp = _now()

            # Initialize round data
            round_data = {
                "round_id": round_id,
                "round_number": round_number,
                "timestamp": timestamp,
                "user_query": user_query,
                "dataset_category": get_dataset_category(),
                "conversation": [
                    {
                        "agent": agent_name,
                        "role": "user",
                        "content": user_query,
                        "timestamp": timestamp
                    }
                ]
            }

            # Save round data
            with up["chat_history"].open("a", encoding="utf-8") as f:
                f.write(json.dumps(round_data, ensure_ascii=False) + "\n")

            # Generate and save query embedding
            _append_round_embedding(user_name, user_query, round_id, round_number)

            print(f"[SAVE] Round {round_number} started for user={user_name}")

        else:
            # This is a message within an existing round
            # Parse the message to extract agent and content
            role_match = re.match(r'\[(\w+)\]\s+(.*)', chat_history)

            if role_match:
                role = role_match.group(1).lower()
                content = role_match.group(2)
            else:
                role = "assistant"
                content = chat_history

            # Load existing rounds to append to the latest one
            if up["chat_history"].exists():
                all_rounds = list(_iter_jsonl(up["chat_history"]))
                if all_rounds:
                    # Update the last round
                    last_round = all_rounds[-1]
                    last_round["conversation"].append({
                        "role": role,
                        "content": content,
                        "timestamp": _now()
                    })
                    last_round["metadata"]["round_status"] = "in_progress"

                    # Rewrite the last round (read all, modify last, write all)
                    with up["chat_history"].open("w", encoding="utf-8") as f:
                        for i, r in enumerate(all_rounds):
                            if i == len(all_rounds) - 1:
                                # Write updated last round
                                f.write(json.dumps(last_round, ensure_ascii=False) + "\n")
                            else:
                                # Write unchanged rounds
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")

                    print(f"[SAVE] Message added to round {last_round['round_number']} for user={user_name}")

        return True

    except Exception as e:
        print(f"[SAVE ERROR] {e}")
        return False

def append_agent_message(content: str, role: str, agent_name: str, user_name: str) -> bool:
    """
    Append an agent message to the current round in chat history.

    Args:
        content: Message content
        role: Message role ("user" or "assistant")
        agent_name: Name of the agent that generated this message
        user_name: User identifier

    Returns:
        True if successful, False otherwise
    """
    try:
        up = _user_paths(user_name)

        if not up["chat_history"].exists():
            print(f"[APPEND_MSG ERROR] No chat history found for user {user_name}")
            return False

        # Load all rounds
        all_rounds = list(_iter_jsonl(up["chat_history"]))

        if not all_rounds:
            print(f"[APPEND_MSG ERROR] No rounds found in chat history")
            return False

        # Update the last round
        last_round = all_rounds[-1]
        last_round["conversation"].append({
            "agent": agent_name,
            "role": role,
            "content": content,
            "timestamp": _now()
        })

        # Rewrite all rounds
        with up["chat_history"].open("w", encoding="utf-8") as f:
            for r in all_rounds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[APPEND_MSG] Added {role} message to round {last_round['round_number']} for user={user_name}")
        return True

    except Exception as e:
        print(f"[APPEND_MSG ERROR] {e}")
        return False

def save_round_summary(user_name: str, round_number: int, summary: str) -> bool:
    """
    Save a summary for a specific round to chat_history.

    Args:
        user_name: User identifier
        round_number: Round number to save summary for
        summary: Summary text

    Returns:
        True if successful, False otherwise
    """
    try:
        up = _user_paths(user_name)

        if not up["chat_history"].exists():
            print(f"[SAVE_SUMMARY ERROR] No chat history found for user {user_name}")
            return False

        # Load all rounds
        all_rounds = list(_iter_jsonl(up["chat_history"]))

        # Find and update the specific round
        found = False
        for round_data in all_rounds:
            if round_data.get("round_number") == round_number:
                round_data["round_summary"] = summary
                found = True
                break

        if not found:
            print(f"[SAVE_SUMMARY ERROR] Round {round_number} not found")
            return False

        # Rewrite all rounds
        with up["chat_history"].open("w", encoding="utf-8") as f:
            for r in all_rounds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[SAVE_SUMMARY] Saved summary for round {round_number}, user={user_name}")
        return True

    except Exception as e:
        print(f"[SAVE_SUMMARY ERROR] {e}")
        return False

def get_current_round_number(user_name: str) -> int:
    """
    Determine the current round number by counting rounds in chat_history.

    Args:
        user_name: User identifier

    Returns:
        Current round number (1-indexed). Returns 1 if no history exists.
    """
    try:
        up = _user_paths(user_name)

        if not up["chat_history"].exists():
            return 1

        all_rounds = list(_iter_jsonl(up["chat_history"]))
        return len(all_rounds)

    except Exception as e:
        print(f"[GET_CURRENT_ROUND ERROR] {e}")
        return 1  # Default to round 1 on error

def load_previous_round_summaries(user_name: str, up_to_round: int) -> List[Dict]:
    """
    Load all round summaries up to (but not including) a specific round.

    Args:
        user_name: User identifier
        up_to_round: Load summaries before this round number

    Returns:
        List of {"round_number": int, "summary": str}
    """
    try:
        up = _user_paths(user_name)

        if not up["chat_history"].exists():
            return []

        all_rounds = list(_iter_jsonl(up["chat_history"]))
        summaries = []

        for round_data in all_rounds:
            round_num = round_data.get("round_number")
            if round_num >= up_to_round:
                break

            summary = round_data.get("round_summary")
            if summary:
                summaries.append({
                    "round_number": round_num,
                    "summary": summary
                })

        return summaries

    except Exception as e:
        print(f"[LOAD_SUMMARIES ERROR] {e}")
        return []

def _generate_summary_for_round(round_data: dict, llm_config: dict) -> Optional[str]:
    """
    Generate a summary for a single round using LLM.

    Args:
        round_data: The round data from chat_history (contains conversation, user_query, round_number)
        llm_config: LLM configuration for API call

    Returns:
        Summary string if successful, None otherwise
    """
    try:
        round_num = round_data.get("round_number")
        user_query = round_data.get("user_query", "")

        # Extract all agent messages from conversation
        agent_messages = []
        for msg in round_data.get("conversation", []):
            if msg.get("role") == "assistant":
                agent_name = msg.get("agent", "").split("_")[0]  # Extract agent type
                content = msg.get("content", "")
                if content:
                    agent_messages.append(f"{agent_name}: {content}")  # Full content, no truncation

        if not agent_messages:
            print(f"[SUMMARY_GEN] No agent messages found for Round {round_num}")
            return None

        # Build summary prompt
        summary_prompt = (
            f"Summarize Round {round_num} of the conversation in 2-3 concise sentences.\n\n"
            f"User Query: {user_query}\n\n"
            f"Agent Outputs:\n" + "\n\n".join(agent_messages) + "\n\n"
            f"Summary (focus on: what user wanted, what was recommended, key preferences):"
        )

        # Call LLM to generate summary
        import autogen
        client = autogen.OpenAIWrapper(**llm_config)
        response = client.create(messages=[{"role": "user", "content": summary_prompt}])
        summary = response.choices[0].message.content if response.choices else None

        if not summary:
            print(f"[SUMMARY_GEN] LLM returned no summary for Round {round_num}")
            return None

        return summary

    except Exception as e:
        print(f"[SUMMARY_GEN ERROR] Failed to generate summary: {e}")
        return None

# Dialogue History Agent
def rag_search_for_dialogue_history_agent(user_name: str, current_query: str = None, llm_config: dict = None) -> dict:
    """
    Search for relevant dialogue history rounds using semantic similarity.
    Also generates summaries for previous rounds that don't have them.

    Args:
        user_name: User identifier
        current_query: Current user query for similarity search (optional)
        llm_config: LLM configuration for summary generation (optional)

    Returns:
        dict with relative_rounds list and previous_summaries
    """
    # Debug: Print received current_query
    print(f"[DEBUG] rag_search_for_dialogue_history_agent called")
    print(f"[DEBUG]   user_name: {user_name}")
    print(f"[DEBUG]   current_query: {current_query}")

    up = _user_paths(user_name)

    # Load historical rounds
    all_rounds = list(_iter_jsonl(up["chat_history"]))
    print(f"[DEBUG] Found {len(all_rounds)} total rounds")

    # Determine current round number
    current_round_number = len(all_rounds)

    # Incrementally generate summaries for all previous rounds that don't have them
    if llm_config and len(all_rounds) > 1:
        for round_data in all_rounds[:-1]:  # Exclude current round
            round_num = round_data.get("round_number")
            if not round_data.get("round_summary"):
                print(f"[SUMMARY_GEN] Generating summary for Round {round_num}")
                summary = _generate_summary_for_round(round_data, llm_config)
                if summary:
                    save_round_summary(user_name, round_num, summary)
                    print(f"[SUMMARY_GEN] Successfully saved summary for Round {round_num}")

    # Load all previous summaries (including newly generated ones)
    previous_summaries = load_previous_round_summaries(user_name, current_round_number)
    print(f"[DEBUG] Loaded {len(previous_summaries)} previous round summaries")

    # Exclude the last round (current round being processed)
    if len(all_rounds) > 0:
        all_rounds = all_rounds[:-1]
        print(f"[DEBUG] Excluding current round, searching in {len(all_rounds)} historical rounds")

    if not all_rounds:
        # No history (or only current round)
        print(f"[DEBUG] No historical rounds to search, returning empty")
        return {
            "user_name": user_name,
            "relative_rounds": [],
            "previous_summaries": previous_summaries
        }

    # Load round embeddings
    round_embs, metadata = _load_round_embeddings(user_name)

    # Also exclude the last embedding (current round's embedding)
    if round_embs is not None and round_embs.shape[0] > 0:
        round_embs = round_embs[:-1]
        print(f"[DEBUG] Using {round_embs.shape[0]} historical embeddings for search")

    if round_embs is None or current_query is None:
        # No embeddings or no query: return empty
        print(f"[DEBUG] No embeddings or no query (round_embs={round_embs is not None}, current_query={current_query is not None}), returning empty")
        return {
            "user_name": user_name,
            "relative_rounds": [],
            "previous_summaries": previous_summaries
        }

    # Generate embedding for current query
    iidx_dir = item_index_dir()
    iidx = _Index(iidx_dir)
    iidx.load()

    current_emb = _generate_round_embedding(current_query, iidx.model)
    current_emb = current_emb.reshape(1, -1)

    # Calculate similarities
    sims = _cosine(current_emb, round_embs).ravel()

    # Get top 1 (can be extended to top-k in future)
    top_k = 1
    order = sims.argsort()[::-1]
    top_indices = order[:top_k]

    # Build result
    relative_rounds = []
    for idx in top_indices:
        round_data = all_rounds[idx]
        similarity_score = float(sims[idx])
        relative_rounds.append({
            "round_data": round_data,
            "similarity_score": similarity_score
        })
        print(f"[DEBUG] Top similar round: round_number={round_data.get('round_number')}, "
              f"query='{round_data.get('user_query', '')[:50]}...', similarity={similarity_score:.4f}")

    return {
        "user_name": user_name,
        "relative_rounds": relative_rounds,
        "previous_summaries": previous_summaries
    }


# Dialogue History Agent (Multimodal version - uses OpenRouter API)
def rag_search_for_dialogue_history_agent_multimodal(
    user_name: str,
    current_query: str = None,
    llm_config: dict = None
) -> dict:
    """
    Multimodal version: uses OpenRouter API instead of local SentenceTransformer.
    Search for relevant dialogue history rounds using semantic similarity.
    """
    print(f"[DEBUG] rag_search_for_dialogue_history_agent_multimodal called")
    print(f"[DEBUG]   user_name: {user_name}")
    print(f"[DEBUG]   current_query: {current_query}")

    up = _user_paths(user_name)

    # Load historical rounds
    all_rounds = list(_iter_jsonl(up["chat_history"]))
    print(f"[DEBUG] Found {len(all_rounds)} total rounds")

    # Determine current round number
    current_round_number = len(all_rounds)

    # Incrementally generate summaries for all previous rounds that don't have them
    if llm_config and len(all_rounds) > 1:
        for round_data in all_rounds[:-1]:
            round_num = round_data.get("round_number")
            if not round_data.get("round_summary"):
                print(f"[SUMMARY_GEN] Generating summary for Round {round_num}")
                summary = _generate_summary_for_round(round_data, llm_config)
                if summary:
                    save_round_summary(user_name, round_num, summary)
                    print(f"[SUMMARY_GEN] Successfully saved summary for Round {round_num}")

    # Load all previous summaries (including newly generated ones)
    previous_summaries = load_previous_round_summaries(user_name, current_round_number)
    print(f"[DEBUG] Loaded {len(previous_summaries)} previous round summaries")

    # Exclude the last round (current round being processed)
    if len(all_rounds) > 0:
        all_rounds = all_rounds[:-1]
        print(f"[DEBUG] Excluding current round, searching in {len(all_rounds)} historical rounds")

    if not all_rounds:
        print(f"[DEBUG] No historical rounds to search, returning empty")
        return {
            "user_name": user_name,
            "relative_rounds": [],
            "previous_summaries": previous_summaries
        }

    # Load round embeddings
    round_embs, metadata = _load_round_embeddings(user_name)

    # Also exclude the last embedding (current round's embedding)
    if round_embs is not None and round_embs.shape[0] > 0:
        round_embs = round_embs[:-1]
        print(f"[DEBUG] Using {round_embs.shape[0]} historical embeddings for search")

    if round_embs is None or current_query is None:
        print(f"[DEBUG] No embeddings or no query, returning empty")
        return {
            "user_name": user_name,
            "relative_rounds": [],
            "previous_summaries": previous_summaries
        }

    # Generate embedding for current query using OpenRouter API
    embedder = OpenRouterEmbedder()
    current_emb = embedder.encode([current_query], normalize_embeddings=True)

    # Calculate similarities
    sims = _cosine(current_emb, round_embs).ravel()

    # Get top 1
    top_k = 1
    order = sims.argsort()[::-1]
    top_indices = order[:top_k]

    # Build result
    relative_rounds = []
    for idx in top_indices:
        round_data = all_rounds[idx]
        similarity_score = float(sims[idx])
        relative_rounds.append({
            "round_data": round_data,
            "similarity_score": similarity_score
        })
        print(f"[DEBUG] Top similar round: round_number={round_data.get('round_number')}, "
              f"query='{round_data.get('user_query', '')[:50]}...', similarity={similarity_score:.4f}")

    return {
        "user_name": user_name,
        "relative_rounds": relative_rounds,
        "previous_summaries": previous_summaries
    }


def _load_user_profile_embedding(user_name: str) -> Optional[np.ndarray]:
    up = _user_paths(user_name)
    uidx_dir = up["user_index"]
    uidx_dir.mkdir(parents=True, exist_ok=True)
    uidx = _Index(uidx_dir)

    if not uidx.exists():
        build_user_index_for(user_name, get_dataset_category())

    uidx.load()

    if uidx.matrix is None or uidx.matrix.shape[0] == 0:
        return None

    user_emb = uidx.matrix.mean(axis=0, keepdims=True)  
    return user_emb


# Recommend Agent
def rag_search_for_recommend_agent(user_name: str, user_query: str, top_k: int = 5) -> dict:
    # 1) loading Item RAG
    iidx_dir = item_index_dir()
    iidx = _Index(iidx_dir)
    iidx.load()
    if iidx.matrix is None or iidx.model is None:
        raise RuntimeError("Item index is not properly loaded (matrix/model is None).")

    # 2) construct query embedding
    q_text = (user_query or '').strip()
    q_emb = iidx.model.encode([q_text], normalize_embeddings=True)

    # 3) compute similarities with metadata and review embeddings
    sim_metadata = _cosine(q_emb, iidx.matrix).ravel()

    if iidx.review_matrix is not None:
        # Dual fusion: 50% metadata + 50% review
        sim_review = _cosine(q_emb, iidx.review_matrix).ravel()
        score_final = 0.5 * sim_metadata + 0.5 * sim_review
        fusion_info = {
            "alpha_metadata": 0.5,
            "beta_review": 0.5,
            "review_embeddings_used": True,
        }
    else:
        # Fallback: metadata only
        score_final = sim_metadata
        fusion_info = {
            "alpha_metadata": 1.0,
            "review_embeddings_used": False,
        }

    order = score_final.argsort()[::-1]
    candidates: List[Dict[str, Any]] = []
    for idx in order:
        d = iidx.doc(int(idx))
        meta_text = (d.get("text") or "").strip()
        candidates.append({
            "item_id": d.get("item_id"),
            "title": d.get("title", ""),
            "price": d.get("price", ""),
            "brand": d.get("brand", ""),
            "review": meta_text,
            "score": round(float(score_final[idx]), 6),
        })

    seen, recs = set(), []
    for r in candidates:
        if r["item_id"] in seen:
            continue
        seen.add(r["item_id"])
        recs.append(r)
        if len(recs) >= top_k:
            break

    # 4) Dynamically select relevant reviews based on query similarity
    selected_reviews = _select_relevant_reviews(user_query, user_name, top_k=5)

    up = _user_paths(user_name)

    # Update the current round with recommendations
    if up["chat_history"].exists():
        all_rounds = list(_iter_jsonl(up["chat_history"]))
        if all_rounds:
            # Update the last round with recommendation data
            last_round = all_rounds[-1]
            last_round["recommendations"] = recs
            last_round["fusion_info"] = fusion_info

            # Rewrite all rounds with updated last round
            with up["chat_history"].open("w", encoding="utf-8") as f:
                for i, r in enumerate(all_rounds):
                    if i == len(all_rounds) - 1:
                        f.write(json.dumps(last_round, ensure_ascii=False) + "\n")
                    else:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "dataset_category": get_dataset_category(),
        "recommendations": recs,
        "relevant_user_reviews": selected_reviews,
        "review_selection_method": "dynamic_similarity",
        "ts": _now(),
    }


# =============================================================================
# Multimodal RAG Components (metadata + review + caption)
# =============================================================================

class OpenRouterEmbedder:
    """
    Lightweight embedder for runtime query encoding via OpenRouter API.
    Uses Qwen3-Embedding-8B model (4096 dimensions).
    Thread-safe and stateless - suitable for concurrent requests.
    """

    def __init__(self):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Please set it with your OpenRouter API key."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model_name = "qwen/qwen3-embedding-8b"
        self.embedding_dim = 4096
        print(f"[OpenRouterEmbedder] Initialized with model: {self.model_name}")

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings via OpenRouter API.
        Includes retry logic with exponential backoff for transient errors.

        Args:
            texts: List of texts to encode
            normalize_embeddings: Whether to L2-normalize the embeddings

        Returns:
            numpy array of embeddings, shape (len(texts), 4096)
        """
        import time

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    timeout=60.0
                )
                embeddings = np.array(
                    [item.embedding for item in response.data],
                    dtype=np.float32
                )

                if normalize_embeddings:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    embeddings = embeddings / (norms + 1e-8)

                return embeddings

            except Exception as e:
                error_str = str(e)
                # 检查是否为可重试的错误 (503, 502, 429)
                if any(code in error_str for code in ["503", "502", "429"]):
                    if attempt < max_retries - 1:
                        wait_time = 2 ** (attempt + 1)  # 指数退避: 2, 4, 8, 16, 32 秒
                        print(f"[RETRY] API error ({error_str[:80]}...), waiting {wait_time}s... ({attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                # 不可重试的错误或已耗尽重试次数
                print(f"[OpenRouterEmbedder ERROR] Failed to encode: {e}")
                raise


class _MultimodalIndex:
    """
    Enhanced index class with caption support and OpenRouter API encoding.
    Loads metadata, review, and caption embeddings for multimodal RAG.
    """

    def __init__(self, index_dir: Path):
        self.dir = index_dir
        self.meta = index_dir / "meta.pkl"
        self.mat = index_dir / "matrix.pkl"
        self.review_mat = index_dir / "review_matrix.pkl"
        self.caption_mat = index_dir / "caption_matrix.pkl"
        self.caption_texts_path = index_dir / "caption_texts.pkl"

        self.docs: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        self.caption_texts: List[str] = []
        self.embedder: Optional[OpenRouterEmbedder] = None

        self.matrix: Optional[np.ndarray] = None
        self.review_matrix: Optional[np.ndarray] = None
        self.caption_matrix: Optional[np.ndarray] = None

    def exists(self) -> bool:
        return self.meta.exists() and self.mat.exists()

    def load(self):
        if not self.exists():
            raise RuntimeError(f"Index not found under {self.dir}")

        # Load metadata (docs and texts)
        d = pickle.load(self.meta.open("rb"))
        self.docs = d["docs"]
        self.texts = d["texts"]

        # Initialize OpenRouter embedder (lightweight, no GPU loading)
        self.embedder = OpenRouterEmbedder()
        print(f"[_MultimodalIndex] OpenRouter embedder initialized")

        # Load metadata embedding matrix
        self.matrix = pickle.load(self.mat.open("rb"))
        print(f"[_MultimodalIndex] Metadata matrix loaded: {self.matrix.shape}")

        # Load review matrix if available
        if self.review_mat.exists():
            self.review_matrix = pickle.load(self.review_mat.open("rb"))
            print(f"[_MultimodalIndex] Review matrix loaded: {self.review_matrix.shape}")
        else:
            print(f"[_MultimodalIndex] No review_matrix.pkl found")

        # Load caption matrix and texts if available
        if self.caption_mat.exists():
            self.caption_matrix = pickle.load(self.caption_mat.open("rb"))
            print(f"[_MultimodalIndex] Caption matrix loaded: {self.caption_matrix.shape}")
        else:
            print(f"[_MultimodalIndex] No caption_matrix.pkl found")

        if self.caption_texts_path.exists():
            self.caption_texts = pickle.load(self.caption_texts_path.open("rb"))
            print(f"[_MultimodalIndex] Caption texts loaded: {len(self.caption_texts)} captions")
        else:
            print(f"[_MultimodalIndex] No caption_texts.pkl found")

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Basic search using metadata embeddings only."""
        if not query.strip() or self.matrix is None:
            return []
        qv = self.embedder.encode([query], normalize_embeddings=True)
        sims = _cosine(qv, self.matrix).ravel()
        order = sims.argsort()[::-1]
        return [(int(i), float(sims[i])) for i in order[:max(1, k)]]

    def doc(self, i: int) -> Dict[str, Any]:
        """Get document by index."""
        return self.docs[i]

    def get_caption(self, idx: int) -> str:
        """Get caption text for an item by index."""
        if self.caption_texts and 0 <= idx < len(self.caption_texts):
            return self.caption_texts[idx]
        return ""


def _select_relevant_reviews_multimodal(user_query: str, user_name: str, top_k: int = 5) -> List[Dict]:
    """
    Select top-k most relevant reviews based on query similarity using OpenRouter API.
    Multimodal version using _MultimodalIndex.
    """
    # Load user review embeddings
    review_embs, metadata = _load_user_review_embeddings(user_name)

    if review_embs is None:
        print(f"[SELECT_REVIEWS_MULTIMODAL] No review embeddings found for user {user_name}")
        return []

    # Load item index to use embedder
    iidx_dir = item_index_dir()
    iidx = _MultimodalIndex(iidx_dir)
    iidx.load()

    # Generate query embedding
    query_emb = iidx.embedder.encode([user_query], normalize_embeddings=True)

    # Calculate similarities
    sims = _cosine(query_emb, review_embs).ravel()

    # Get top-k indices
    top_indices = sims.argsort()[::-1][:min(top_k, len(sims))]

    # Extract top-k reviews
    selected_reviews = []
    for idx in top_indices:
        review_data = metadata["reviews"][idx].copy()
        review_data["similarity_score"] = float(sims[idx])
        selected_reviews.append(review_data)

    print(f"[SELECT_REVIEWS_MULTIMODAL] Selected {len(selected_reviews)} reviews based on query similarity")
    if selected_reviews:
        print(f"[SELECT_REVIEWS_MULTIMODAL] Top similarity score: {selected_reviews[0]['similarity_score']:.4f}")

    return selected_reviews


def rag_search_for_recommend_agent_multimodal(user_name: str, user_query: str, top_k: int = 5) -> dict:
    """
    Search for recommendations using triple-fusion scoring (metadata + review + caption).
    Returns recommendations with caption text included.

    Scoring formula: score = 0.33 * sim_metadata + 0.33 * sim_review + 0.33 * sim_caption
    Fallback to dual or single fusion if some embeddings are missing.

    Args:
        user_name: User identifier
        user_query: The user's search query
        top_k: Number of recommendations to return

    Returns:
        dict with recommendations (including caption), fusion_info, and user reviews
    """
    # 1) Load Multimodal Item RAG index
    iidx_dir = item_index_dir()
    iidx = _MultimodalIndex(iidx_dir)
    iidx.load()

    # Validate all required embeddings are present
    if iidx.matrix is None or iidx.embedder is None:
        raise RuntimeError("Item index is not properly loaded (matrix/embedder is None).")
    if iidx.review_matrix is None:
        raise RuntimeError("Review embeddings (review_matrix.pkl) are required for multimodal RAG.")
    if iidx.caption_matrix is None:
        raise RuntimeError("Caption embeddings (caption_matrix.pkl) are required for multimodal RAG.")

    # 2) Generate query embedding via OpenRouter API
    q_text = (user_query or '').strip()
    q_emb = iidx.embedder.encode([q_text], normalize_embeddings=True)

    # 3) Compute similarities with all three embedding types
    sim_metadata = _cosine(q_emb, iidx.matrix).ravel()
    sim_review = _cosine(q_emb, iidx.review_matrix).ravel()
    sim_caption = _cosine(q_emb, iidx.caption_matrix).ravel()

    # 4) Triple fusion: 33% metadata + 33% review + 33% caption
    score_final = 0.33 * sim_metadata + 0.33 * sim_review + 0.33 * sim_caption
    fusion_info = {
        "fusion_type": "triple",
        "alpha_metadata": 0.33,
        "beta_review": 0.33,
        "gamma_caption": 0.33,
    }
    print(f"[MULTIMODAL_RAG] Using triple fusion: 0.33*meta + 0.33*review + 0.33*caption")

    # 5) Rank and build recommendations with caption
    order = score_final.argsort()[::-1]
    candidates: List[Dict[str, Any]] = []

    for idx in order:
        d = iidx.doc(int(idx))
        meta_text = (d.get("text") or "").strip()
        caption = iidx.get_caption(int(idx))  # Get caption text

        candidates.append({
            "item_id": d.get("item_id"),
            "title": d.get("title", ""),
            "price": d.get("price", ""),
            "brand": d.get("brand", ""),
            "review": meta_text,
            "caption": caption,  # Include caption in results
            "score": round(float(score_final[idx]), 6),
        })

    # Deduplicate
    seen, recs = set(), []
    for r in candidates:
        if r["item_id"] in seen:
            continue
        seen.add(r["item_id"])
        recs.append(r)
        if len(recs) >= top_k:
            break

    # 6) Select relevant user reviews using multimodal index
    selected_reviews = _select_relevant_reviews_multimodal(user_query, user_name, top_k=5)

    # 7) Update chat history with recommendations
    up = _user_paths(user_name)
    if up["chat_history"].exists():
        all_rounds = list(_iter_jsonl(up["chat_history"]))
        if all_rounds:
            last_round = all_rounds[-1]
            last_round["recommendations"] = recs
            last_round["fusion_info"] = fusion_info

            with up["chat_history"].open("w", encoding="utf-8") as f:
                for i, r in enumerate(all_rounds):
                    if i == len(all_rounds) - 1:
                        f.write(json.dumps(last_round, ensure_ascii=False) + "\n")
                    else:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "dataset_category": get_dataset_category(),
        "recommendations": recs,
        "relevant_user_reviews": selected_reviews,
        "review_selection_method": "dynamic_similarity_multimodal",
        "fusion_info": fusion_info,
        "ts": _now(),
    }
