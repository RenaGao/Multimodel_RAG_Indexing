import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List
import re
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR   = Path("./storage")         
VSTORE_DIR = BASE_DIR / "vectorstore"
USERS_DIR  = VSTORE_DIR / "users"     
CATALOG_DIR = VSTORE_DIR / "catalog"
DATA_DIR   = Path("../data")

for p in [VSTORE_DIR, DATA_DIR, USERS_DIR, CATALOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def get_files_for_category(category: str):
    user_data_file   = DATA_DIR / f"{category}_top_100_user_data.json"
    item_review_file = DATA_DIR / f"{category}_top_100_user_itemReview.json"
    return user_data_file, item_review_file

def norm(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)

def load_json(path: Path):
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8")
    txt = txt.replace("NaN", "null")
    return json.loads(txt)

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

class Embedder:
    def __init__(self):
        print("[Embedder] Loading SentenceTransformer: all-MiniLM-L6-v2")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

def normalize_item_docs_from_itemReview(all_users: Any) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if all_users is None:
        return docs

    if isinstance(all_users, dict):
        user_items = all_users.items()
    else:
        user_items = enumerate(all_users)

    for user_id, blocks in user_items:
        if not isinstance(blocks, list):
            continue

        for block in blocks:
            if not isinstance(block, list) or len(block) == 0:
                continue

            meta = block[0] if isinstance(block[0], dict) else None
            if not meta:
                continue

            product_title = meta.get("title", "") or ""
            parent_asin = meta.get("parent_asin") or meta.get("asin")
            store = meta.get("store", "") or ""
            details = meta.get("details") or {}
            brand = details.get("Brand") or meta.get("brand") or store
            price = meta.get("price")  

            reviews_part = block[1:]
            if len(reviews_part) == 1 and isinstance(reviews_part[0], list):
                reviews = reviews_part[0]
            else:
                reviews = reviews_part

            if not isinstance(reviews, list):
                continue

            for rv in reviews:
                if not isinstance(rv, dict):
                    continue

                asin = rv.get("asin") or rv.get("parent_asin") or parent_asin
                review_title = rv.get("title") or ""
                review_text = rv.get("text") or ""
                rating = rv.get("rating")

                try:
                    rating = float(rating)
                except Exception:
                    rating = None

                full_text = " ".join([
                    norm(review_title),
                    norm(review_text),
                ]).strip()

                if not full_text:
                    continue

                item_id = asin or parent_asin or f"ITEM_{len(docs)+1}"
                doc_id = f"item::{item_id}::{_hash_bytes(full_text.encode('utf-8'))[:8]}"

                docs.append({
                    "doc_id": doc_id,
                    "item_id": item_id,
                    "title": norm(product_title),
                    "brand": norm(brand),
                    "price": norm(price),
                    "rating": rating,
                    "text": full_text,
                    "source": "user_data_review",
                })

    return docs

def build_item_index_for_category(category: str):
    item_dir = CATALOG_DIR / category / "item_index"
    item_dir.mkdir(parents=True, exist_ok=True)

    # use user_data to build item RAG
    user_data_file, _ = get_files_for_category(category)
    all_users = load_json(user_data_file)

    docs = normalize_item_docs_from_itemReview(all_users)
    if not docs:
        print(f"[WARN] No docs built for category={category}")
        return

    texts = [d["title"] + " " + d["text"] for d in docs]

    emb = Embedder().encode(texts)

    meta_path = item_dir / "meta.pkl"
    matrix_path = item_dir / "matrix.pkl"
    embedder_path = item_dir / "embedder.pkl"

    import pickle
    with meta_path.open("wb") as f:
        pickle.dump({"docs": docs, "texts": texts}, f)

    with matrix_path.open("wb") as f:
        pickle.dump(emb, f)

    with embedder_path.open("wb") as f:
        pickle.dump({"kind": "sbert", "model": "all-MiniLM-L6-v2"}, f)

    print(f"[SAVE] item index -> {item_dir}")


def normalize_user_profile_for_user(all_users: Any, only_user_id: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if all_users is None or only_user_id is None:
        return docs

    user_blocks = all_users.get(only_user_id)
    if not user_blocks:
        return docs

    titles = []
    details_texts = []
    review_texts = []

    for block in user_blocks:
        if not isinstance(block, list) or len(block) == 0:
            continue

        meta = block[0] if isinstance(block[0], dict) else None
        reviews = block[1:] if len(block) > 1 else []

        if meta:
            title = meta.get("title") or ""
            titles.append(str(title))
            details = meta.get("details") or {}
            desc    = meta.get("description") or []
            features = meta.get("features") or []
            details_texts.append(
                " ".join([
                    norm(title),
                    norm(details),
                    norm(desc),
                    norm(features),
                ])
            )

        for rv in reviews:
            if not isinstance(rv, dict):
                continue
            review_texts.append(norm(rv.get("title")) + " " + norm(rv.get("text")))

    if not (titles or details_texts or review_texts):
        return docs

    full_text = " ".join([
        " ".join(titles),
        " ".join(details_texts),
        " ".join(review_texts),
    ])

    docs.append({
        "doc_id": f"user::{only_user_id}",
        "user_id": only_user_id,
        "title": f"User profile {only_user_id}",
        "text": full_text,
        "source": "user_data",
    })
    return docs

def build_user_index_for(user_name: str, category: str):
    safe_user = re.sub(r"[^A-Za-z0-9_\-]+", "_", user_name or "anonymous")
    user_dir = USERS_DIR / safe_user / "user_index"
    user_dir.mkdir(parents=True, exist_ok=True)
    user_data_file, _ = get_files_for_category(category)
    all_users = load_json(DATA_DIR / user_data_file)
    docs = normalize_user_profile_for_user(all_users, only_user_id=user_name)

    texts = [d["title"] + " " + d["text"] for d in docs]
    emb = Embedder().encode(texts)

    meta_path = user_dir / "meta.pkl"
    matrix_path = user_dir / "matrix.pkl"
    embedder_path = user_dir / "embedder.pkl"

    with meta_path.open("wb") as f:
        import pickle
        pickle.dump({"docs": docs, "texts": texts}, f)
    with matrix_path.open("wb") as f:
        import pickle
        pickle.dump(emb, f)
    with embedder_path.open("wb") as f:
        import pickle
        pickle.dump({"kind": "sbert"}, f)

    print(f"[SAVE] user index -> {user_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", help="e.g. All_Beauty or Amazon_Fashion")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all:
        cat2file = {}
        for p in DATA_DIR.glob("*_top_100_user_itemReview.json"):
            stem = p.stem  
            suffix = "_top_100_user_itemReview"
            if not stem.endswith(suffix):
                continue
            cat = stem[: -len(suffix)]
            cat2file[cat] = p.name      

        else:
            for cat, fname in sorted(cat2file.items()):
                print(f"\n===== Building item index for category: {cat} (from {fname}) =====")
                build_item_index_for_category(cat)

    elif args.category:
        print(f"\n===== Building item index for category: {args.category} =====")
        build_item_index_for_category(args.category)
