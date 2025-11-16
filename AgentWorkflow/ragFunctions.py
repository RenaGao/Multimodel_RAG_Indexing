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
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR   = Path("./storage")
VSTORE_DIR = BASE_DIR / "vectorstore"
CATALOG_DIR = VSTORE_DIR / "catalog"
USERS_DIR   = VSTORE_DIR / "users"

DATASET_CATEGORY = "All_Beauty"

for p in [VSTORE_DIR, CATALOG_DIR, USERS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def item_index_dir() -> Path:
    return CATALOG_DIR / DATASET_CATEGORY / "item_index"

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
        self.docs: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        self.model: Optional[SentenceTransformer] = None
        self.matrix: Optional[np.ndarray] = None

    def exists(self) -> bool:
        return self.meta.exists() and self.mat.exists()

    def load(self):
        if not self.exists():
            raise RuntimeError(f"Index not found under {self.dir}")

        d = pickle.load(self.meta.open("rb"))
        self.docs = d["docs"]
        self.texts = d["texts"]

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.matrix = pickle.load(self.mat.open("rb"))

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        if not query.strip() or self.matrix is None:
            return []
        qv = self.model.encode([query], normalize_embeddings=True)
        sims = _cosine(qv, self.matrix).ravel()
        order = sims.argsort()[::-1]
        return [(int(i), float(sims[i])) for i in order[:max(1, k)]]

    def doc(self, i: int) -> Dict[str, Any]:
        return self.docs[i]

def save_chat_history(chat_history: str, user_name: str) -> bool:
    try:
        up = _user_paths(user_name)
        rec = {"ts": _now(), "text": chat_history}
        with up["chat_history"].open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[SAVE] chat for user={user_name} -> {up['chat_history']}")
        return True
    except Exception as e:
        print("[SAVE ERROR]", e)
        return False

# Dialogue History Agent
def rag_search_for_dialogue_history_agent(user_name: str) -> dict:
    up = _user_paths(user_name)
    rows = list(_iter_jsonl(up["chat_history"]))
    max_rows = 50
    recent_rows = rows[-max_rows:] if rows else []

    texts = [r.get("text", r.get("raw", "")) for r in recent_rows]
    if texts:
        last_bits = texts[-3:]
        summary = " || ".join(last_bits)
        if len(summary) > 350:
            summary = summary[:347] + "..."
        round_hint = "subsequent_rounds"
    else:
        summary = "No prior dialogue history — returning a neutral profile."
        round_hint = "first_round"

    payload = {
        "user_name": user_name,
        "summary": summary,
        "chat_history": recent_rows,
        "ts": _now(),
        "round_hint": round_hint,
    }
    return payload

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
    sim_query = _cosine(q_emb, iidx.matrix).ravel()                

    user_emb = _load_user_profile_embedding(user_name)
    sim_user = _cosine(user_emb, iidx.matrix).ravel() 
    alpha = 0.7  
    score_final = alpha * sim_query + (1.0 - alpha) * sim_user
    fusion_info = {
        "alpha_query": alpha,
        "beta_user_profile": 1.0 - alpha,
        "user_profile_used": True,
    }

    order = score_final.argsort()[::-1]
    candidates: List[Dict[str, Any]] = []
    for idx in order:
        d = iidx.doc(int(idx))
        review_text = (d.get("text") or "").strip()
        candidates.append({
            "item_id": d.get("item_id"),
            "title": d.get("title", ""),
            "price": d.get("price", ""),
            "brand": d.get("brand", ""),
            "review": review_text,
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
    history_excerpt = []
    if up["chat_history"].exists():
        all_rows = list(_iter_jsonl(up["chat_history"]))
        for row in all_rows[-5:]:
            history_excerpt.append(_norm(row.get("text", row.get("raw", "")))[:400])

    round_payload = {
        "ts": _now(),
        "dataset_category": DATASET_CATEGORY,
        "user_query": user_query,
        "user_labels": user_labels,
        "recommendations": recs,
        "history_excerpt": history_excerpt,
        "engine": "Sentence-Transformers (all-MiniLM-L6-v2)",
        "fusion": fusion_info,
    }
    with up["last_round"].open("w", encoding="utf-8") as f:
        json.dump(round_payload, f, ensure_ascii=False, indent=2)

    return {
        "dataset_category": DATASET_CATEGORY,
        "recommendations": recs,
        "ts": _now(),
    }


# Reason Agent 
def rag_search_for_reason_agent(user_name: str) -> dict:
    up = _user_paths(user_name)
    if not up["last_round"].exists():
        return {
            "explanation": "No prior recommendation context found for this user.",
            "user_query": "",
            "user_labels": [],
            "recommendations": [],
        }

    data = json.loads(up["last_round"].read_text(encoding="utf-8"))
    user_query = data.get("user_query", "")
    labels     = data.get("user_labels", [])
    recs       = data.get("recommendations", [])
    hist       = data.get("history_excerpt", [])
    engine     = data.get("engine", "Sentence-Transformers")
    dataset_category = data.get("dataset_category", DATASET_CATEGORY)
    fusion     = data.get("fusion", {})

    label_clause = ", ".join(labels) if labels else "general preferences inferred from history"
    why_bits = []
    for r in recs[:5]:
        parts = [r.get("title", "")]
        if r.get("brand"):
            parts.append(f"by {r['brand']}")
        why_bits.append(" ".join([p for p in parts if p]))

    explanation = (
        f"For catalog '{dataset_category}', based on the query \"{user_query}\" and labels ({label_clause}), "
        f"I retrieved products using a RAG retriever ({engine}) over item reviews. "
    )

    review_bits = []
    for r in recs[:3]:
        snippet = r.get("review")
        if snippet:
            review_bits.append(f"• {snippet}")

    if review_bits:
        explanation += "Some supporting user reviews:\n" + "\n".join(review_bits) + " "


    if fusion.get("user_profile_used"):
        explanation += (
            f"I combined similarity to the current query (weight {fusion.get('alpha_query', 0.7)}) "
            f"with similarity to the user's past purchases and reviews "
            f"(weight {fusion.get('beta_user_profile', 0.3)}). "
        )
    else:
        explanation += "User profile information was not available, so only query similarity was used. "

    if why_bits:
        explanation += "Top matches: " + "; ".join(why_bits) + ". "
    if hist:
        explanation += "Recent dialogue snippets considered: " + " | ".join(hist[:2]) + "."

    return {
        "explanation": explanation,
        "user_query": user_query,
        "user_labels": labels,
        "recommendations": recs,
    }
