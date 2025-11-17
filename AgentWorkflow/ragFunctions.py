import json
import re
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from buildRag import build_user_index_for 
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR   = Path("./storage")
VSTORE_DIR = BASE_DIR / "vectorstore"
CATALOG_DIR = VSTORE_DIR / "catalog"
USERS_DIR   = VSTORE_DIR / "users"

DATASET_CATEGORY = "Amazon_Fashion"

for p in [VSTORE_DIR, CATALOG_DIR, USERS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def item_index_dir() -> Path:
    return CATALOG_DIR / DATASET_CATEGORY / "item_index_v2_qwen3"

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
    }

def _iter_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            yield json.loads(line)

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device=device)
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
                "dataset_category": DATASET_CATEGORY,
                "conversation": [
                    {
                        "agent": agent_name,
                        "role": "user",
                        "content": user_query,
                        "timestamp": timestamp
                    }
                ],
                "metadata": {
                    "query_length": len(user_query),
                    "round_status": "in_progress"
                }
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

# Dialogue History Agent
def rag_search_for_dialogue_history_agent(user_name: str, current_query: str = None) -> dict:
    """
    Search for relevant dialogue history rounds using semantic similarity.

    Args:
        user_name: User identifier
        current_query: Current user query for similarity search (optional)

    Returns:
        dict with relative_rounds list
    """
    up = _user_paths(user_name)

    # Load historical rounds
    all_rounds = list(_iter_jsonl(up["chat_history"]))

    if not all_rounds:
        # No history
        return {
            "user_name": user_name,
            "relative_rounds": []
        }

    # Load round embeddings
    round_embs, metadata = _load_round_embeddings(user_name)

    if round_embs is None or current_query is None:
        # No embeddings or no query: return empty
        return {
            "user_name": user_name,
            "relative_rounds": []
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
        relative_rounds.append({
            "round_data": all_rounds[idx],
            "similarity_score": float(sims[idx])
        })

    return {
        "user_name": user_name,
        "relative_rounds": relative_rounds
    }

def _load_user_profile_embedding(user_name: str) -> Optional[np.ndarray]:
    up = _user_paths(user_name)
    uidx_dir = up["user_index"]
    uidx_dir.mkdir(parents=True, exist_ok=True)
    uidx = _Index(uidx_dir)

    if not uidx.exists():
        build_user_index_for(user_name, DATASET_CATEGORY)

    uidx.load()

    if uidx.matrix is None or uidx.matrix.shape[0] == 0:
        return None

    user_emb = uidx.matrix.mean(axis=0, keepdims=True)  
    return user_emb


# Recommend Agent 
def rag_search_for_recommend_agent(user_name: str, user_query: str, user_labels: List[str], top_k: int) -> dict:
    # 1) loading Item RAG
    iidx_dir = item_index_dir()
    iidx = _Index(iidx_dir)
    iidx.load()
    if iidx.matrix is None or iidx.model is None:
        raise RuntimeError("Item index is not properly loaded (matrix/model is None).")

    # 2) construct query embedding
    q_text = f"{user_query or ''} {' '.join(user_labels or [])}".strip()
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

    up = _user_paths(user_name)

    # Update the current round with recommendations
    if up["chat_history"].exists():
        all_rounds = list(_iter_jsonl(up["chat_history"]))
        if all_rounds:
            # Update the last round with recommendation data
            last_round = all_rounds[-1]
            last_round["recommendations"] = recs
            last_round["user_labels"] = user_labels
            last_round["engine"] = "Sentence-Transformers (Qwen3-Embedding-8B)"
            last_round["fusion_info"] = fusion_info
            last_round["metadata"]["recommendations_count"] = len(recs)

            # Rewrite all rounds with updated last round
            with up["chat_history"].open("w", encoding="utf-8") as f:
                for i, r in enumerate(all_rounds):
                    if i == len(all_rounds) - 1:
                        f.write(json.dumps(last_round, ensure_ascii=False) + "\n")
                    else:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "dataset_category": DATASET_CATEGORY,
        "recommendations": recs,
        "ts": _now(),
    }


# Reason Agent 
# def rag_search_for_reason_agent(user_name: str) -> dict:
#     up = _user_paths(user_name)
#     if not up["last_round"].exists():
#         return {
#             "explanation": "No prior recommendation context found for this user.",
#             "user_query": "",
#             "user_labels": [],
#             "recommendations": [],
#         }

#     data = json.loads(up["last_round"].read_text(encoding="utf-8"))
#     user_query = data.get("user_query", "")
#     labels     = data.get("user_labels", [])
#     recs       = data.get("recommendations", [])
#     hist       = data.get("history_excerpt", [])
#     engine     = data.get("engine", "Sentence-Transformers")
#     dataset_category = data.get("dataset_category", DATASET_CATEGORY)
#     fusion     = data.get("fusion", {})

#     label_clause = ", ".join(labels) if labels else "general preferences inferred from history"
#     why_bits = []
#     for r in recs[:5]:
#         parts = [r.get("title", "")]
#         if r.get("brand"):
#             parts.append(f"by {r['brand']}")
#         why_bits.append(" ".join([p for p in parts if p]))

#     explanation = (
#         f"For catalog '{dataset_category}', based on the query \"{user_query}\" and labels ({label_clause}), "
#         f"I retrieved products using a RAG retriever ({engine}) over item reviews. "
#     )

#     review_bits = []
#     for r in recs[:3]:
#         snippet = r.get("review")
#         if snippet:
#             review_bits.append(f"• {snippet}")

#     if review_bits:
#         explanation += "Some supporting user reviews:\n" + "\n".join(review_bits) + " "


#     if fusion.get("user_profile_used"):
#         explanation += (
#             f"I combined similarity to the current query (weight {fusion.get('alpha_query', 0.7)}) "
#             f"with similarity to the user's past purchases and reviews "
#             f"(weight {fusion.get('beta_user_profile', 0.3)}). "
#         )
#     else:
#         explanation += "User profile information was not available, so only query similarity was used. "

#     if why_bits:
#         explanation += "Top matches: " + "; ".join(why_bits) + ". "
#     if hist:
#         explanation += "Recent dialogue snippets considered: " + " | ".join(hist[:2]) + "."

#     return {
#         "explanation": explanation,
#         "user_query": user_query,
#         "user_labels": labels,
#         "recommendations": recs,
#     }
