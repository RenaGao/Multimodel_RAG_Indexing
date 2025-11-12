import json, re, math, os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
from datetime import datetime, timezone
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import faiss

# TODO:
def rag_search_include_image_for_recommend_agent(
    user_query: str,
    dialogue_summary: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Perform RAG search and return products (with image URLs) for the recommender agent.

    Args:
        user_query (str): User's query (e.g., "moisturizer for dry skin").
        dialogue_summary (str): User preference summary from DialogueHistoryAgent.
        top_k (int): Number of retrieved items to return.

    Returns:
        dict: Structured data containing query, context, items (with images), and metadata.
    """

    # example output
    retrieved_items = [
        {
            "id": "cerave-123",
            "title": "CeraVe Moisturizing Cream 340g",
            "summary": (
                "Fragrance-free moisturizing cream with ceramides and hyaluronic acid. "
                "Ideal for very dry or eczema-prone skin."
            ),
            "price": {"value": 18.99, "currency": "AUD"},
            "images": [
                {"url": "https://cdn.example.com/img/cerave-1.jpg", "alt": "Front jar shot"},
                {"url": "https://cdn.example.com/img/cerave-2.jpg", "alt": "Texture close-up"}
            ],
            "tags": ["dry-skin", "hydration", "fragrance-free"]
        },
        {
            "id": "laroche-001",
            "title": "La Roche-Posay Lipikar Baume AP+M 400ml",
            "summary": (
                "Soothing repairing balm for atopic, itchy, and very dry skin. "
                "Fragrance-free and suitable for babies and adults."
            ),
            "price": {"value": 28.50, "currency": "AUD"},
            "images": [
                {"url": "https://cdn.example.com/img/laroche-1.jpg", "alt": "Bottle packaging"}
            ],
            "tags": ["dry-skin", "sensitive-skin", "repairing"]
        }
    ][:top_k]

    result = {
        "query": user_query,
        "dialogue_context": dialogue_summary,
        "items": retrieved_items,
    }

    return result

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#  Configuration settings 
class Config:
    USER_DATA_PATH = Path("./RagData/All_Beauty_top_100_user_data.json")
    ITEMREVIEW_PATH = Path("./RagData/All_Beauty_top_100_user_itemReview.json")
    INDEX_DIR = Path("./index")
    CTX_FAISS_PATH = INDEX_DIR / "context_product_index.faiss"
    CTX_IDS_PATH = INDEX_DIR / "context_product_index.ids.json"
    CHAT_HISTORY_PATH = Path("./vectorstore/chat_history.jsonl")
    
    ALPHA_META = 0.7
    BETA_USER = 0.15
    GAMMA_CTX = 0.15
    TOPK_CANDIDATES = 50
    MIN_REVIEW_TEXT_LEN = 80
    EMB_DIM_FALLBACK = 384

#  Global state 
class GlobalState:
    user_data: Dict[str, Any] | None = None
    itemreview: Dict[str, Any] | None = None
    st_model: SentenceTransformer | None = None
    
    meta_emb: np.ndarray | None = None
    meta_ids: List[str] = []
    meta_texts: List[str] = []
    
    rev_emb: np.ndarray | None = None
    rev_ids: List[str] = []
    rev_texts: List[str] = []
    
    ctx_emb: np.ndarray | None = None
    ctx_ids: List[str] = []
    ctx_texts: List[str] = []
    
    PID2META: Dict[str, Dict[str, Any]] = {}
    last_user_profile: Dict[str, Any] | None = None
    last_recommendation: Dict[str, Any] | None = None
    
    FAISS_AVAILABLE = True

#  Utility functions 
class Utilities:
    
    @staticmethod
    def set_data_paths(user_data_path: str | Path, itemreview_path: str | Path) -> None:
        Config.USER_DATA_PATH = Path(user_data_path)
        Config.ITEMREVIEW_PATH = Path(itemreview_path)
        GlobalState.user_data = None
        GlobalState.itemreview = None
        Utilities.reset_indices()
    
    @staticmethod
    def reset_indices():
        GlobalState.meta_emb = None
        GlobalState.meta_ids = []
        GlobalState.meta_texts = []
        GlobalState.rev_emb = None
        GlobalState.rev_ids = []
        GlobalState.rev_texts = []
        GlobalState.ctx_emb = None
        GlobalState.ctx_ids = []
        GlobalState.ctx_texts = []
    
    @staticmethod
    def ensure_loaded() -> None:
        if GlobalState.user_data is None:
            if not Config.USER_DATA_PATH.exists():
                raise FileNotFoundError(f"Missing user_data.json at: {Config.USER_DATA_PATH}")
            GlobalState.user_data = json.loads(Config.USER_DATA_PATH.read_text(encoding="utf-8"))
        if GlobalState.itemreview is None:
            if not Config.ITEMREVIEW_PATH.exists():
                raise FileNotFoundError(f"Missing user_itemreview.json at: {Config.ITEMREVIEW_PATH}")
            GlobalState.itemreview = json.loads(Config.ITEMREVIEW_PATH.read_text(encoding="utf-8"))
    
    @staticmethod
    def ensure_embedder() -> None:
        if GlobalState.st_model is None:
            os.environ["CLEAN_UP_TOKENIZATION_SPACES"] = "False"
            GlobalState.st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
            if hasattr(GlobalState.st_model, 'tokenizer'):
                GlobalState.st_model.tokenizer.clean_up_tokenization_spaces = False
    @staticmethod
    def norm_txt(t: str) -> str:
        return re.sub(r"\s+", " ", (t or "").strip())
    
    @staticmethod
    def safe_float(x, default=None):
        try:
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    return default
                s = re.sub(r"[^\d\.\-eE+]", "", s)
                return float(s) if s not in ("", ".", "-", "+") else default
            return default
        except:
            return default
    
    @staticmethod
    def unit_normalize(mat: np.ndarray) -> np.ndarray:
        if mat.ndim == 1:
            n = norm(mat)
            return mat / (n if n != 0 else 1.0)
        n = norm(mat, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return mat / n
    
    @staticmethod
    def stable_id_from_title(title: str) -> str:
        import hashlib
        t = (title or "").strip().lower()
        return "T_" + hashlib.md5(t.encode("utf-8")).hexdigest()[:16] if t else ""

#  Embedding management 
class EmbeddingManager:
    
    @staticmethod
    def embed_texts(texts: List[str]) -> np.ndarray:
        Utilities.ensure_embedder()
        if not texts:
            return np.zeros((0, Config.EMB_DIM_FALLBACK), dtype=np.float32)
        E = GlobalState.st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
        return Utilities.unit_normalize(E)
    
    @staticmethod
    def embed_query(query: str) -> np.ndarray:
        v = EmbeddingManager.embed_texts([Utilities.norm_txt(query)])
        return v[0] if v.shape[0] else np.zeros((Config.EMB_DIM_FALLBACK,), dtype=np.float32)
    
    @staticmethod
    def topk_dot(sim_vec: np.ndarray, mat: np.ndarray, ids: List[str], k: int) -> List[Tuple[str, float]]:
        if mat is None or mat.shape[0] == 0:
            return []
        sims = (mat @ sim_vec).astype(np.float32)
        idx = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in idx if sims[i] > 0]


    
@staticmethod
def save_chat_history(event: dict | str, path: Path = Config.CHAT_HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(event, dict):
        ev = dict(event)
    elif isinstance(event, str):
        s = event.strip()
        m = re.match(r"^\s*\[(?P<agent>[^\]]+)\]\s*(?P<content>.*)$", s, flags=re.S)
        if m:
            agent = m.group("agent").strip()
            content = m.group("content").strip()
            a_low = agent.lower()
            if a_low.startswith("user") or "proxy" in a_low: 
                role = "user"
            elif "tool" in a_low: 
                role = "tool"
            elif "system" in a_low: 
                role = "system"
            else: 
                role = "assistant"
            ev = {"type": "message", "role": role, "agent": agent, "content": content}
        else:
            ev = {"type": "message", "role": "assistant", "agent": "Unknown", "content": s}
    else:
        ev = {"type": "message", "role": "assistant", "agent": "Unknown", "content": str(event)}
    
    ev.setdefault("type", "message")
    ev.setdefault("role", "assistant")
    ev.setdefault("agent", "Unknown")
    ev.setdefault("content", "")
    ev.setdefault("payload", None)
    ev.setdefault("ts", datetime.now(timezone.utc).isoformat())
    
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")

@staticmethod
def load_chat_history(max_events: int = 600) -> List[Dict[str, Any]]:
    if not Config.CHAT_HISTORY_PATH.exists():
        return []
    lines = Config.CHAT_HISTORY_PATH.read_text(encoding="utf-8").splitlines()[-max_events:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

#  Product data management
class ProductManager:
    
    @staticmethod
    def extract_pid_from_product(product: Dict[str, Any], fallback_title_ok: bool = True) -> str:
        for k in ("asin", "parent_asin"):
            v = product.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    
    @staticmethod
    def iter_all_products_from_user_data():
        Utilities.ensure_loaded()
        
        has_user_buckets = any(isinstance(v, list) for v in GlobalState.user_data.values())
        if has_user_buckets:
            for _, pairs in GlobalState.user_data.items():
                if not isinstance(pairs, list): 
                    continue
                for pair in pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2: 
                        continue
                    product, _reviews = pair
                    if not isinstance(product, dict): 
                        continue
                    pid = ProductManager.extract_pid_from_product(product, fallback_title_ok=True)
                    if not pid: 
                        continue
                    yield pid, product
            return
        
        for item_id, obj in GlobalState.user_data.items():
            product = obj.get("meta") if isinstance(obj, dict) and "meta" in obj else obj
            if not isinstance(product, dict): 
                continue
            pid = ProductManager.extract_pid_from_product(product, fallback_title_ok=False) or str(item_id)
            if pid:
                yield pid, product
    
    @staticmethod
    def get_reviews_for_pid(pid: str, fallback_pair_reviews: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        Utilities.ensure_loaded()
        if isinstance(GlobalState.itemreview, dict):
            rv = GlobalState.itemreview.get(pid)
            if isinstance(rv, list):
                return rv
        if isinstance(fallback_pair_reviews, list) and fallback_pair_reviews:
            return fallback_pair_reviews
        return []
    
    @staticmethod
    def load_product_meta(pid: str) -> Dict[str, Any]:
        if pid in GlobalState.PID2META:
            return GlobalState.PID2META[pid]
        
        Utilities.ensure_loaded()
        
        for _, pairs in GlobalState.user_data.items():
            if isinstance(pairs, list):
                for pair in pairs or []:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    product, _ = pair
                    apid = str(product.get("asin") or "")
                    if apid == pid:
                        GlobalState.PID2META[pid] = product
                        return product
        
        for item_id, obj in GlobalState.user_data.items():
            if isinstance(obj, dict):
                product = obj.get("meta") if "meta" in obj else obj
                if isinstance(product, dict):
                    apid = str(product.get("asin") or item_id)
                    if apid == pid:
                        GlobalState.PID2META[pid] = product
                        return product
        
        return {}
    
    @staticmethod
    def extract_brand_from_product(product_meta: Dict[str, Any]) -> str:
        b = product_meta.get("brand")
        return str(b).strip() if b else ""

#  Text processing for products and reviews 
class TextProcessor:
    
    @staticmethod
    def mk_meta_text(product: Dict[str, Any]) -> str:
        parts = []
        for key in ("brand", "title", "store"):
            if product.get(key):
                parts.append(str(product[key]))
        for key in ("categories", "features", "description", "details"):
            val = product.get(key)
            if isinstance(val, list):
                parts.extend([str(x) for x in val if x])
            elif isinstance(val, dict):
                parts.extend([f"{k}: {v}" for k, v in val.items() if v])
            elif val:
                parts.append(str(val))
        return Utilities.norm_txt(" . ".join(parts))
    
    @staticmethod
    def mk_review_aggregate_text(reviews: List[Dict[str, Any]]) -> str:
        if not reviews:
            return ""
        
        def score(rv):
            hv = int(rv.get("helpful_vote", 0) or 0)
            rt = Utilities.safe_float(rv.get("rating"), 0.0) or 0.0
            return hv + 0.1 * rt
        
        reps = sorted(reviews, key=score, reverse=True)[:10]
        segs = []
        for r in reps:
            t = " ".join([str(r.get("title") or ""), str(r.get("text") or "")])
            t = Utilities.norm_txt(t)
            if len(t) >= 20:
                segs.append(t)
        joined = " . ".join(segs)
        return joined if len(joined) >= Config.MIN_REVIEW_TEXT_LEN else ""
    
    @staticmethod
    def build_product_context(
        product: Dict[str, Any],
        reviews: List[Dict[str, Any]],
        *,
        review_top_n: int = 8,
        max_chars: int = 1600,
    ) -> str:
        def j(v):
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        
        parts = []
        if product.get("brand"): parts.append(f"BRAND: {product['brand']}")
        if product.get("title"): parts.append(f"TITLE: {product['title']}")
        if product.get("store"): parts.append(f"STORE: {product['store']}")
        if product.get("categories"): parts.append(f"CATEGORIES: {j(product['categories'])}")
        if product.get("features"): parts.append(f"FEATURES: {j(product['features'])}")
        if product.get("description"): parts.append(f"DESCRIPTION: {product['description']}")
        if product.get("details"): parts.append(f"DETAILS: {j(product['details'])}")
        if product.get("price") is not None: parts.append(f"PRICE: {product['price']}")
        if product.get("average_rating") is not None: parts.append(f"AVG_RATING: {product['average_rating']}")
        if product.get("rating_number") is not None: parts.append(f"RATING_COUNT: {product['rating_number']}")
        
        meta_block = " \n".join(Utilities.norm_txt(p) for p in parts if p)
        ctx_parts = [meta_block]
        
        if reviews:
            def score(rv):
                hv = int(rv.get("helpful_vote", 0) or 0)
                rt = Utilities.safe_float(rv.get("rating"), 0.0) or 0.0
                return hv + 0.1 * rt
            
            reps = sorted(reviews, key=score, reverse=True)[:review_top_n]
            r_lines = []
            for r in reps:
                title = str(r.get("title") or "").strip()
                text = str(r.get("text") or "").strip()
                rating = r.get("rating")
                one = " ".join([title, text]).strip()
                if not one:
                    continue
                if rating is not None:
                    one = f"(rating={rating}) " + one
                r_lines.append(Utilities.norm_txt(one))
            if r_lines:
                ctx_parts.append("REVIEWS:\n- " + "\n- ".join(r_lines))
        
        context = "\n\n".join([x for x in ctx_parts if x]).strip()
        if len(context) > max_chars:
            context = context[:max_chars] + " ..."
        return context

#  Index management 
class IndexManager:
    
    @staticmethod
    def index(rebuild: bool = False) -> None:
        Utilities.ensure_loaded()
        Utilities.ensure_embedder()
        
        if (not rebuild) and GlobalState.meta_emb is not None and GlobalState.rev_emb is not None and GlobalState.ctx_emb is not None:
            if GlobalState.FAISS_AVAILABLE and Config.CTX_FAISS_PATH.exists() and Config.CTX_IDS_PATH.exists():
                return
        
        seen = set()
        meta_ids, meta_texts = [], []
        rev_ids, rev_texts = [], []
        ctx_ids, ctx_texts = [], []
        
        GlobalState.PID2META = {}
        
        Utilities.ensure_loaded()
        has_user_buckets = any(isinstance(v, list) for v in GlobalState.user_data.values())
        
        if has_user_buckets:
            for _, pairs in GlobalState.user_data.items():
                if not isinstance(pairs, list):
                    continue
                for pair in pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    product, pair_reviews = pair
                    if not isinstance(product, dict):
                        continue
                    pid = ProductManager.extract_pid_from_product(product, fallback_title_ok=True)
                    if not pid or pid in seen:
                        continue
                    seen.add(pid)
                    
                    GlobalState.PID2META[pid] = product
                    
                    meta_txt = TextProcessor.mk_meta_text(product)
                    if not meta_txt:
                        continue
                    
                    reviews = ProductManager.get_reviews_for_pid(pid, fallback_pair_reviews=pair_reviews)
                    rev_txt = TextProcessor.mk_review_aggregate_text(reviews)
                    ctx_txt = TextProcessor.build_product_context(product, reviews)
                    
                    meta_ids.append(pid); meta_texts.append(meta_txt)
                    if rev_txt:
                        rev_ids.append(pid); rev_texts.append(rev_txt)
                    ctx_ids.append(pid); ctx_texts.append(ctx_txt)
        else:
            for item_id, obj in GlobalState.user_data.items():
                product = obj.get("meta") if isinstance(obj, dict) and "meta" in obj else obj
                if not isinstance(product, dict):
                    continue
                pid = ProductManager.extract_pid_from_product(product, fallback_title_ok=False) or str(item_id)
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                
                GlobalState.PID2META[pid] = product
                
                meta_txt = TextProcessor.mk_meta_text(product)
                if not meta_txt:
                    continue
                
                pair_reviews = []
                if isinstance(obj, dict) and isinstance(obj.get("reviews"), list):
                    pair_reviews = obj["reviews"]
                
                reviews = ProductManager.get_reviews_for_pid(pid, fallback_pair_reviews=pair_reviews)
                rev_txt = TextProcessor.mk_review_aggregate_text(reviews)
                ctx_txt = TextProcessor.build_product_context(product, reviews)
                
                meta_ids.append(pid); meta_texts.append(meta_txt)
                if rev_txt:
                    rev_ids.append(pid); rev_texts.append(rev_txt)
                ctx_ids.append(pid); ctx_texts.append(ctx_txt)
        
        def encode(texts: List[str]) -> np.ndarray:
            if not texts:
                return np.zeros((0, Config.EMB_DIM_FALLBACK), dtype=np.float32)
            em = GlobalState.st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
            return Utilities.unit_normalize(em)
        
        GlobalState.meta_ids, GlobalState.meta_texts = meta_ids, meta_texts
        GlobalState.rev_ids, GlobalState.rev_texts = rev_ids, rev_texts
        GlobalState.ctx_ids, GlobalState.ctx_texts = ctx_ids, ctx_texts
        
        GlobalState.meta_emb = encode(meta_texts)
        GlobalState.rev_emb = encode(rev_texts)
        GlobalState.ctx_emb = encode(ctx_texts)
        
        if GlobalState.FAISS_AVAILABLE:
            Config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
            idx = faiss.IndexFlatIP(GlobalState.ctx_emb.shape[1])
            if GlobalState.ctx_emb.shape[0] > 0:
                idx.add(GlobalState.ctx_emb.astype("float32"))
            faiss.write_index(idx, str(Config.CTX_FAISS_PATH))
            Config.CTX_IDS_PATH.write_text(json.dumps(GlobalState.ctx_ids, ensure_ascii=False), encoding="utf-8")
    
    @staticmethod
    def ensure_meta_index() -> None:
        if GlobalState.meta_emb is None:
            IndexManager.index(rebuild=True)
    
    @staticmethod
    def ensure_review_index() -> None:
        if GlobalState.rev_emb is None:
            IndexManager.index(rebuild=True)
    
    @staticmethod
    def ensure_faiss_ready() -> bool:
        if not GlobalState.FAISS_AVAILABLE:
            return False
        try:
            need = True
            if Config.CTX_FAISS_PATH.exists() and Config.CTX_IDS_PATH.exists():
                ids = json.loads(Config.CTX_IDS_PATH.read_text(encoding="utf-8"))
                need = (not isinstance(ids, list)) or (len(ids) == 0)
            if need:
                IndexManager.index(rebuild=True)
            if not Config.CTX_FAISS_PATH.exists() or not Config.CTX_IDS_PATH.exists():
                return False
            ids = json.loads(Config.CTX_IDS_PATH.read_text(encoding="utf-8"))
            return isinstance(ids, list) and len(ids) > 0
        except Exception:
            return False
    
    @staticmethod
    def ensure_index_ready():
        try:
            need = True
            if Config.CTX_FAISS_PATH.exists() and Config.CTX_IDS_PATH.exists():
                ids = json.loads(Config.CTX_IDS_PATH.read_text(encoding="utf-8"))
                need = (not isinstance(ids, list)) or (len(ids) == 0)
            if need:
                IndexManager.index(rebuild=True)
        except Exception:
            pass

# Search functions
class SearchManager:
    
    @staticmethod
    def search(query: str, top_k: int = 5, index_path: str | Path = Config.CTX_FAISS_PATH) -> List[Tuple[str, float]]:
        if not IndexManager.ensure_faiss_ready():
            return []
        
        idx = faiss.read_index(str(index_path))
        id_list = json.loads(Config.CTX_IDS_PATH.read_text(encoding="utf-8"))
        
        Utilities.ensure_embedder()
        q_emb = GlobalState.st_model.encode([Utilities.norm_txt(query)], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = idx.search(q_emb, top_k)
        
        hits = []
        for dist, ix in zip(D[0], I[0]):
            if ix < 0:
                continue
            item_id = id_list[ix]
            hits.append((item_id, float(dist)))
        return hits
    
    @staticmethod
    def search_vec(q_vec: np.ndarray, top_k: int = 5, index_path: str | Path = Config.CTX_FAISS_PATH) -> List[Tuple[str, float]]:
        if not IndexManager.ensure_faiss_ready():
            return []
        idx = faiss.read_index(str(index_path))
        id_list = json.loads(Config.CTX_IDS_PATH.read_text(encoding="utf-8"))
        q = q_vec.astype("float32").reshape(1, -1)
        D, I = idx.search(q, top_k)
        return [(id_list[i], float(d)) for d, i in zip(D[0], I[0]) if i >= 0]
    
    @staticmethod
    def hybrid_search_with_user_pref_and_context(
        query_vec: np.ndarray,
        user_pref_vec: np.ndarray,
        topk: int,
        alpha_meta: float = Config.ALPHA_META,
        beta_user: float = Config.BETA_USER,
        gamma_ctx: float = Config.GAMMA_CTX,
    ) -> Dict[str, float]:
        IndexManager.ensure_meta_index()
        IndexManager.ensure_review_index()
        IndexManager.index(rebuild=False)
        
        scores = {}
        
        meta_hits = EmbeddingManager.topk_dot(query_vec, GlobalState.meta_emb, GlobalState.meta_ids, Config.TOPK_CANDIDATES)
        for pid, s in meta_hits:
            scores[pid] = scores.get(pid, 0.0) + alpha_meta * s
        
        rev_hits = EmbeddingManager.topk_dot(query_vec, GlobalState.rev_emb, GlobalState.rev_ids, Config.TOPK_CANDIDATES)
        for pid, s in rev_hits:
            scores[pid] = scores.get(pid, 0.0) + (1.0 - alpha_meta) * s
        
        if GlobalState.ctx_emb is not None and len(GlobalState.ctx_ids) > 0 and gamma_ctx > 0:
            ctx_hits = EmbeddingManager.topk_dot(query_vec, GlobalState.ctx_emb, GlobalState.ctx_ids, Config.TOPK_CANDIDATES)
            for pid, s in ctx_hits:
                scores[pid] = scores.get(pid, 0.0) + gamma_ctx * s
        
        cand_pids = set([pid for pid, _ in meta_hits] + [pid for pid, _ in rev_hits])
        if GlobalState.ctx_emb is not None:
            cand_pids |= set([pid for pid, _ in EmbeddingManager.topk_dot(query_vec, GlobalState.ctx_emb, GlobalState.ctx_ids, Config.TOPK_CANDIDATES)])
        
        if user_pref_vec is not None and user_pref_vec.shape[0] > 0 and len(cand_pids) > 0:
            pid2idx = {pid: i for i, pid in enumerate(GlobalState.meta_ids)}
            for pid in cand_pids:
                i = pid2idx.get(pid)
                if i is None:
                    continue
                up = float(GlobalState.meta_emb[i] @ user_pref_vec)
                scores[pid] = scores.get(pid, 0.0) + beta_user * up
        
        return dict(sorted(scores.items(), key=lambda x: -x[1])[:topk])

#  User preference analysis 
class UserPreferenceAnalyzer:
    
    @staticmethod
    def gather_user_corpus(user_name: str, max_hist: int = 80, include_chat: bool = True) -> List[str]:
        Utilities.ensure_loaded()
        corpus = []
        pairs = GlobalState.user_data.get(user_name, [])
        for pair in pairs[:max_hist]:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2: 
                continue
            _, reviews = pair
            for rv in (reviews or []):
                t = f"{rv.get('title') or ''} {rv.get('text') or ''}".strip()
                if t: corpus.append(t)
        if include_chat:
            events = load_chat_history()
            for ev in events[-max_hist:]:
                if ev.get("role") in ("user", "assistant", "tool"):
                    c = str(ev.get("content") or "").strip()
                    if c: corpus.append(c)
                    payload = ev.get("payload")
                    if isinstance(payload, dict):
                        try: 
                            corpus.append(json.dumps(payload, ensure_ascii=False))
                        except Exception: 
                            pass
        return corpus
    
    @staticmethod
    def price_bucket(price_vals: List[float]) -> str:
        vals = [p for p in price_vals if p is not None and not math.isnan(p)]
        if not vals: 
            return "unknown"
        q = np.quantile(vals, [0.33, 0.66])
        med = np.median(vals)
        if med <= q[0]: 
            return "budget"
        if med >= q[1]: 
            return "premium"
        return "mid"
    
    @staticmethod
    def analyze_user_preferences_semantic(user_name: str) -> Dict[str, Any]:
        price_vals, brand_cnt = [], Counter()
        samples = []
        pairs = GlobalState.user_data.get(user_name, [])
        for pair in pairs[:12]:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2: 
                continue
            product, _ = pair
            pid = str(product.get("asin") or "")
            title = product.get("title") or ""
            brand = product.get("brand") or ""
            price = Utilities.safe_float(product.get("price"), None)
            if price is not None: 
                price_vals.append(price)
            if brand: 
                brand_cnt[brand] += 1
            samples.append({"item_id": pid, "title": title, "brand": brand, "price": price})
        
        corpus = UserPreferenceAnalyzer.gather_user_corpus(user_name)
        V = EmbeddingManager.embed_texts(corpus)
        if V.shape[0] > 0:
            user_pref_vec = Utilities.unit_normalize(V.mean(axis=0))
        else:
            user_pref_vec = np.zeros((Config.EMB_DIM_FALLBACK,), dtype=np.float32)
        
        price_stats = {
            "mentions": len(price_vals),
            "bucket": UserPreferenceAnalyzer.price_bucket(price_vals),
            "min": min(price_vals) if price_vals else None,
            "max": max(price_vals) if price_vals else None,
            "median": float(np.median(price_vals)) if price_vals else None,
        }
        return {
            "user_name": user_name,
            "user_pref_vec": user_pref_vec.tolist(),
            "price_stats": price_stats,
            "top_brands": [b for b, _ in brand_cnt.most_common(5)],
            "history_samples": samples,
        }
    
    @staticmethod
    def build_enhanced_search_query_vec(user_query: str, user_name: str, lambda_query: float = 0.6) -> np.ndarray:
        qv = EmbeddingManager.embed_query(user_query)
        pref = UserPreferenceAnalyzer.analyze_user_preferences_semantic(user_name).get("user_pref_vec") or []
        pv = np.array(pref, dtype=np.float32) if pref else np.zeros((Config.EMB_DIM_FALLBACK,), dtype=np.float32)
        fused = Utilities.unit_normalize(lambda_query * qv + (1.0 - lambda_query) * pv)
        return fused

#  Result filtering & reranking 
class ResultProcessor:
    
    @staticmethod
    def filter_by_profile(pid_scores: Dict[str, float], user_pref: Dict[str, Any]) -> List[Tuple[str, float]]:
        price_bucket = (user_pref.get("price_stats") or {}).get("bucket", "unknown")
        median_price = (user_pref.get("price_stats") or {}).get("median", None)
        
        items = []
        for pid, s in pid_scores.items():
            meta = ProductManager.load_product_meta(pid)
            price = Utilities.safe_float(meta.get("price"), None)
            bonus = 0.0
            
            if price is not None and price_bucket != "unknown":
                if price_bucket == "budget" and (median_price is None or price <= median_price):
                    bonus += 0.02
                elif price_bucket == "premium" and (median_price is None or price >= median_price):
                    bonus += 0.02
            
            items.append((pid, s + bonus))
        
        return sorted(items, key=lambda x: -x[1])
    
    @staticmethod
    def generate_label_justification_from_pref(pref: Dict[str, Any]) -> str:
        price = pref.get("price_stats", {})
        brands = pref.get("top_brands", [])
        lines = []
        if price:
            lines.append(f"- Price: mentions={price.get('mentions')}, bucket={price.get('bucket')}, median={price.get('median')}")
        if brands:
            lines.append("- Top brands: " + ", ".join(brands[:5]))
        return "\n".join(lines) if lines else "- No strong structured preferences found."

#  Agent functions 
def load_complete_chat_history(user_name: str, max_messages: int = 100) -> List[Dict[str, Any]]:
    all_events = load_chat_history(max_events=200)
    
    user_events = []
    user_identifier = f"User_{user_name}_proxy_agent"
    
    for event in all_events:
        agent_name = event.get("agent", "")
        content = event.get("content", "")
        
        if (user_identifier in agent_name or 
            f"[USER_QUESTION" in content and user_name in content or
            f"Dialogue_History_{user_name}_agent" in agent_name or
            f"Recommender_{user_name}_agent" in agent_name or
            f"Reasoner_{user_name}_agent" in agent_name):
            
            user_events.append(event)
    
    weighted_events = apply_time_weighting(user_events)
    return weighted_events[:max_messages]

def apply_time_weighting(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []
    
    weighted_events = []
    total_events = len(events)
    
    for i, event in enumerate(events):
        weight = 1.0 - (i * 0.5 / total_events)
        weighted_event = {
            **event,
            "time_weight": round(weight, 2),
            "position": i
        }
        weighted_events.append(weighted_event)
    
    return weighted_events

def analyze_chat_history_for_preferences(chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chat_history:
        return {
            "summary": "No chat history available for this user.",
            "preferences": {},
            "recent_topics": [],
            "interaction_pattern": "new_user"
        }
    
    user_messages = []
    all_content = []
    
    for event in chat_history:
        content = str(event.get("content", ""))
        role = event.get("role", "")
        agent = event.get("agent", "")
        weight = event.get("time_weight", 1.0)
        
        all_content.append(content)
        
        if role == "user" or "User_" in agent:
            user_messages.append({
                "content": content,
                "weight": weight,
                "timestamp": event.get("ts", "")
            })
    
    preferences = extract_preferences_from_messages(user_messages)
    recent_topics = identify_recent_topics(chat_history[-10:])
    interaction_pattern = analyze_interaction_pattern(chat_history)
    
    summary = generate_chat_history_summary(preferences, recent_topics, interaction_pattern, len(chat_history))
    
    return {
        "summary": summary,
        "preferences": preferences,
        "recent_topics": recent_topics,
        "interaction_pattern": interaction_pattern
    }

def extract_preferences_from_messages(user_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    preferences = {
        "price_sensitivity": "unknown",
        "quality_focus": "unknown", 
        "brand_preference": "unknown",
        "mentioned_categories": [],
        "specific_requirements": []
    }
    
    all_user_text = " ".join([msg["content"] for msg in user_messages])
    
    if not all_user_text.strip():
        return preferences
    
    Utilities.ensure_embedder()
    user_embedding = GlobalState.st_model.encode([all_user_text], normalize_embeddings=True)[0]
    
    preference_dimensions = {
        "price_sensitivity": [
            "I care about price and want affordable options",
            "I'm looking for good value for money", 
            "Budget is important to me"
        ],
        "quality_focus": [
            "I want high quality products",
            "Quality is my top priority",
            "I'm looking for durable and reliable products"
        ],
        "brand_preference": [
            "I prefer specific brands", 
            "Brand reputation matters to me",
            "I'm loyal to certain brands"
        ]
    }
    
    for pref_key, pref_texts in preference_dimensions.items():
        pref_embeddings = GlobalState.st_model.encode(pref_texts, normalize_embeddings=True)
        similarities = pref_embeddings @ user_embedding
        avg_similarity = np.mean(similarities)
        
        if avg_similarity > 0.3:
            if pref_key == "price_sensitivity":
                preferences["price_sensitivity"] = "high"
            elif pref_key == "quality_focus":
                preferences["quality_focus"] = "high"
            elif pref_key == "brand_preference":
                preferences["brand_preference"] = "high"
    
    return preferences

def identify_recent_topics(recent_messages: List[Dict[str, Any]]) -> List[str]:
    if not recent_messages:
        return []
    
    recent_content = [str(msg.get("content", "")) for msg in recent_messages if msg.get("content")]
    
    if not recent_content:
        return []
    
    Utilities.ensure_embedder()
    embeddings = GlobalState.st_model.encode(recent_content, normalize_embeddings=True)
    
    if len(embeddings) > 1:
        centroid = np.mean(embeddings, axis=0)
        similarities = embeddings @ centroid
        top_indices = np.argsort(-similarities)[:3]
        
        topics = []
        for idx in top_indices:
            if similarities[idx] > 0.5:
                content = recent_content[idx]
                if len(content) > 20:
                    topics.append(content[:50] + "...")
        
        return topics
    
    return []

def analyze_interaction_pattern(chat_history: List[Dict[str, Any]]) -> str:
    user_count = sum(1 for msg in chat_history if msg.get("role") == "user")
    tool_count = sum(1 for msg in chat_history if "tool" in str(msg.get("content", "")).lower())
    
    if user_count == 0:
        return "new_user"
    elif tool_count / max(user_count, 1) > 0.7:
        return "tool_heavy"
    elif user_count > 10:
        return "frequent_user"
    elif user_count > 3:
        return "active_user"
    else:
        return "casual_user"

def generate_chat_history_summary(preferences: Dict[str, Any], recent_topics: List[str], 
                                    interaction_pattern: str, total_messages: int) -> str:
    summary_parts = []
    
    if interaction_pattern == "new_user":
        summary_parts.append("This is a new user with no previous conversation history.")
    elif interaction_pattern == "frequent_user":
        summary_parts.append("This user has extensive conversation history.")
    elif interaction_pattern == "active_user":
        summary_parts.append("This user has moderate conversation activity.")
    
    if preferences["price_sensitivity"] == "high":
        summary_parts.append("The user shows sensitivity to pricing.")
    if preferences["quality_focus"] == "high":
        summary_parts.append("The user emphasizes product quality.")
    if preferences["brand_preference"] == "high":
        summary_parts.append("The user has brand preferences.")
    
    if recent_topics:
        topics_str = ", ".join(recent_topics[:2])
        summary_parts.append(f"Recent discussions include: {topics_str}")
    
    return " ".join(summary_parts) if summary_parts else "Limited information available from chat history."


def generate_user_labels_from_analysis(analysis_result: Dict[str, Any]) -> List[str]:
    labels = []
    preferences = analysis_result["preferences"]
    interaction_pattern = analysis_result["interaction_pattern"]
    
    if preferences["price_sensitivity"] == "high":
        labels.append("price_sensitive")
    if preferences["quality_focus"] == "high":
        labels.append("quality_focused")
    if preferences["brand_preference"] == "high":
        labels.append("brand_conscious")
    
    if interaction_pattern == "frequent_user":
        labels.append("experienced_user")
    elif interaction_pattern == "active_user":
        labels.append("engaged_user")
    elif interaction_pattern == "new_user":
        labels.append("new_user")
    
    if not labels:
        labels.append("general_shopper")
    
    return labels

def enhance_query_with_user_context(user_query: str, user_labels: List[str]) -> str:
    if not user_labels:
        return user_query
    
    label_enhancements = {
        "price_sensitive": "affordable budget-friendly",
        "quality_focused": "high-quality premium", 
        "brand_conscious": "reputable brand",
        "experienced_user": "highly-rated popular"
    }
    
    enhancements = []
    for label in user_labels:
        if label in label_enhancements:
            enhancements.append(label_enhancements[label])
    
    if enhancements:
        return f"{user_query} {' '.join(enhancements)}"
    else:
        return user_query

def rerank_with_user_preferences(search_results: List[Tuple[str, float]], 
                                user_labels: List[str], 
                                user_query: str) -> List[Tuple[str, float]]:
    if not search_results:
        return []
    
    reranked = []
    
    for item_id, score in search_results:
        meta = ProductManager.load_product_meta(item_id)
        if not meta:
            continue
            
        adjusted_score = score
        
        for label in user_labels:
            if label == "price_sensitive":
                price = Utilities.safe_float(meta.get("price"), float('inf'))
                if price < 50:
                    adjusted_score += 0.1
                    
            elif label == "quality_focused":
                rating = Utilities.safe_float(meta.get("average_rating"), 0)
                if rating >= 4.0:
                    adjusted_score += 0.1
                    
            elif label == "brand_conscious":
                brand = meta.get("brand", "")
                if brand and len(brand.strip()) > 0:
                    adjusted_score += 0.05
        
        reranked.append((item_id, adjusted_score))
    
    return sorted(reranked, key=lambda x: x[1], reverse=True)

def format_recommendations(results: List[Tuple[str, float]], 
                            user_query: str, 
                            user_labels: List[str]) -> List[Dict[str, Any]]:
    recommendations = []
    
    for item_id, score in results:
        meta = ProductManager.load_product_meta(item_id)
        reviews = ProductManager.get_reviews_for_pid(item_id)
        
        recommendation = {
            "item_id": item_id,
            "title": meta.get("title", "Unknown"),
            "brand": meta.get("brand", ""),
            "price": Utilities.safe_float(meta.get("price"), None),
            "confidence_score": round(score, 4),
            "match_reasons": explain_recommendation_match(meta, user_query, user_labels),
            "review_count": len(reviews),
            "average_rating": Utilities.safe_float(meta.get("average_rating"), None)
        }
        
        recommendations.append(recommendation)
    
    return recommendations

def explain_recommendation_match(meta: Dict[str, Any], 
                                user_query: str, 
                                user_labels: List[str]) -> List[str]:
    reasons = ["matches your search query"]
    
    for label in user_labels:
        if label == "price_sensitive" and is_affordable(meta):
            reasons.append("fits your budget")
        elif label == "quality_focused" and is_high_quality(meta):
            reasons.append("meets your quality standards")
        elif label == "brand_conscious" and meta.get("brand"):
            reasons.append("from a reputable brand")
    
    return reasons

def is_affordable(meta: Dict[str, Any]) -> bool:
    price = Utilities.safe_float(meta.get("price"), float('inf'))
    return price < 100

def is_high_quality(meta: Dict[str, Any]) -> bool:
    rating = Utilities.safe_float(meta.get("average_rating"), 0)
    return rating >= 4.0

def get_current_round_context() -> Dict[str, Any]:
    recent_events = load_chat_history(max_events=50)
    
    current_round = {
        "user_question": {},
        "dialogue_analysis": {},
        "recommendations": {}
    }
    
    for event in reversed(recent_events):
        event_type = event.get("type")
        
        if not current_round["user_question"] and event.get("role") == "user":
            current_round["user_question"] = event
        
        elif not current_round["dialogue_analysis"] and event_type == "dialogue_analysis":
            current_round["dialogue_analysis"] = event.get("payload", {})
        
        elif not current_round["recommendations"] and event_type == "recommendation":
            current_round["recommendations"] = event.get("payload", {})
        
        if all(current_round.values()):
            break
    
    return current_round

def generate_comprehensive_reasoning(user_question: Dict[str, Any], 
                                    dialogue_analysis: Dict[str, Any], 
                                    recommendations: Dict[str, Any]) -> str:
    parts = []
    
    user_content = user_question.get("content", "this query")
    parts.append(f"Based on your question about '{user_content}', here's the reasoning behind my response:")
    
    user_labels = dialogue_analysis.get("user_labels", [])
    if user_labels:
        parts.append("")
        parts.append("**User Profile Analysis:**")
        parts.append(f"I identified you as: {', '.join(user_labels)} based on our conversation history.")
        
        label_explanations = {
            "price_sensitive": "This means you're conscious of getting good value for your money.",
            "quality_focused": "This indicates you prioritize product quality and reliability.", 
            "brand_conscious": "This shows you pay attention to brand reputation and trust.",
            "experienced_user": "Your extensive interaction history helps me better understand your preferences."
        }
        
        for label in user_labels:
            if label in label_explanations:
                parts.append(f"- {label_explanations[label]}")
    
    rec_list = recommendations.get("recommendations", [])
    if rec_list:
        parts.append("")
        parts.append("**Recommendation Rationale:**")
        parts.append("The products I recommended were selected because:")
        
        for i, rec in enumerate(rec_list[:3]):
            title = rec.get("title", "Unknown product")
            match_reasons = rec.get("match_reasons", [])
            
            if match_reasons:
                reason_text = " and ".join(match_reasons)
                parts.append(f"{i+1}. **{title}** - {reason_text}")
            else:
                parts.append(f"{i+1}. **{title}** - relevant to your search criteria")
    else:
        parts.append("")
        parts.append("No specific recommendations were generated for this query.")
    
    if user_labels and rec_list:
        parts.append("")
        parts.append("**Connection Between Profile and Recommendations:**")
        if "price_sensitive" in user_labels:
            parts.append("- Your price sensitivity guided me toward more affordable options.")
        if "quality_focused" in user_labels:
            parts.append("- Your focus on quality led me to prioritize highly-rated products.")
        if "brand_conscious" in user_labels:
            parts.append("- Your brand awareness influenced the selection of reputable brands.")
    
    return "\n".join(parts)

# Main Agent Functions  
def rag_search_for_dialogue_history_agent(user_name: str) -> str:
    # TODO: 1. get user name -> get conversation history -> 1st round can be empty str, let agent give general output
    # TODO: -> 2nd round start seeing impact
    IndexManager.ensure_index_ready()
    
    chat_history = load_complete_chat_history(user_name)
    analysis_result = analyze_chat_history_for_preferences(chat_history)
    user_labels = generate_user_labels_from_analysis(analysis_result)
    
    # build natural language response
    summary = analysis_result["summary"]
    preferences = analysis_result["preferences"]
    recent_topics = analysis_result["recent_topics"]
    
    natural_response = f"User Profile Analysis for {user_name}:\n\n"
    natural_response += f"Summary: {summary}\n\n"
    
    # add preference details
    if preferences:
        natural_response += "Key Preferences:\n"
        if preferences["price_sensitivity"] != "unknown":
            natural_response += f"- Price sensitivity: {preferences['price_sensitivity']}\n"
        if preferences["quality_focus"] != "unknown":
            natural_response += f"- Quality focus: {preferences['quality_focus']}\n"
        if preferences["brand_preference"] != "unknown":
            natural_response += f"- Brand preference: {preferences['brand_preference']}\n"
    
    # add recent topics if available
    if recent_topics:
        natural_response += f"\nRecent discussion topics: {', '.join(recent_topics)}\n"
    
    # add user labels
    if user_labels:
        natural_response += f"\nUser characteristics: {', '.join(user_labels)}\n"
    
    # add conversation statistics
    total_messages = len(chat_history)
    if total_messages > 0:
        natural_response += f"\nBased on analysis of {total_messages} previous messages."
    
    output = {
        "user_name": user_name,
        "chat_history_summary": analysis_result["summary"],
        "user_labels": user_labels,
        "preference_analysis": analysis_result["preferences"],
        "conversation_context": {
            "total_messages": len(chat_history),
            "recent_topics": analysis_result["recent_topics"],
            "interaction_pattern": analysis_result["interaction_pattern"]
        },
        "raw_chat_history_preview": chat_history[-5:]
    }
    
    GlobalState.last_user_profile = output
    
    save_chat_history({
        "type": "dialogue_analysis",
        "role": "tool", 
        "agent": "DialogueHistory",
        "content": natural_response,  
        "payload": output,  
    })
    
    return natural_response  

def rag_search_for_recommend_agent(user_query: str, user_labels: List[str], top_k: int = 5) -> str:
    # TODO: 1. make retrieve rule base function separately, plugin whenever we need to use
    Utilities.ensure_loaded()
    Utilities.ensure_embedder()
    IndexManager.ensure_index_ready()
    
    user_name = (GlobalState.last_user_profile or {}).get("user_name") or "Unknown"
    
    enhanced_query = enhance_query_with_user_context(user_query, user_labels)
    search_results = SearchManager.search(enhanced_query, top_k=top_k*2, index_path=Config.CTX_FAISS_PATH)
    reranked_results = rerank_with_user_preferences(search_results, user_labels, user_query)
    final_recommendations = format_recommendations(reranked_results[:top_k], user_query, user_labels)
    
    # Build natural language response instead of JSON
    if not final_recommendations:
        natural_response = f"I couldn't find any products matching '{user_query}'. Please try different search terms."
    else:
        natural_response = f"I found these {len(final_recommendations)} products for '{user_query}':\n\n"
        
        for i, rec in enumerate(final_recommendations, 1):
            title = rec.get("title", "Unknown Product")
            brand = rec.get("brand", "")
            price = rec.get("price")
            rating = rec.get("average_rating")
            review_count = rec.get("review_count", 0)
            confidence = rec.get("confidence_score", 0)
            
            # Format the product information
            product_line = f"{i}. {title}"
            if brand:
                product_line += f" by {brand}"
            if price and not math.isnan(price):
                product_line += f" - ${price:.2f}"
            if rating:
                product_line += f" ⭐ {rating}/5"
            if review_count > 0:
                product_line += f" ({review_count} reviews)"
            
            natural_response += product_line + "\n"
    
    # Still save structured data for internal use
    output = {
        "original_query": user_query,
        "enhanced_query": enhanced_query,
        "user_labels_used": user_labels,
        "recommendations": final_recommendations,
        "search_metadata": {
            "total_candidates": len(search_results),
            "final_count": len(final_recommendations),
            "reranking_applied": True
        }
    }
    
    GlobalState.last_recommendation = output
    
    save_chat_history({
        "type": "recommendation",
        "role": "tool",
        "agent": "Recommend",
        "content": natural_response, 
        "payload": output, 
    })
    
    return natural_response  

def rag_search_for_reason_agent() -> str:
    # get the current round context only
    current_round_context = get_current_round_context()
    
    # extract components for reasoning
    user_question = current_round_context.get("user_question", {})
    dialogue_analysis = current_round_context.get("dialogue_analysis", {})
    recommendations = current_round_context.get("recommendations", {})
    
    # generate comprehensive reasoning
    reasoning = generate_comprehensive_reasoning(user_question, dialogue_analysis, recommendations)
    
    # return natural language instead of JSON
    natural_response = reasoning 
    
    output = {
        "reasoning": reasoning,
        "context_used": {
            "user_question": user_question.get("content", "Unknown"),
            "user_labels": dialogue_analysis.get("user_labels", []),
            "recommendation_count": len(recommendations.get("recommendations", []))
        },
        "scope": "current_round_only"
    }
    
    save_chat_history({
        "type": "reasoning",
        "role": "tool",
        "agent": "Reason",
        "content": natural_response,  
        "payload": output,  
    })
    
    return natural_response  

#  Initialisation
IndexManager.index()

if __name__ == "__main__":
    print(rag_search_for_dialogue_history_agent("AG73BVBKUOH22USSFJA5ZWL7AKXA"))
    print(rag_search_for_recommend_agent(
        "Can you help me find a good moisturizer for dry skin?",
        [],
        5
    ))
    print(rag_search_for_reason_agent())