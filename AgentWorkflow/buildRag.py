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
DATA_DIR   = Path("/data/gpfs/projects/punim2620/Multimodel_RAG_Indexing/AmazonReviews/Processed/user_center/")

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

def construct_item_search_text(
    product_meta: Any,
    all_item_reviews: List[dict],
    max_reviews: int = 20
) -> str:
    """
    Construct text for embedding from product metadata and reviews
    (Copied from fashion_rag_index.py)

    Only uses string concatenation of original fields, does not call LLM

    Args:
        product_meta: Product metadata dict (may be None)
        all_item_reviews: List of all reviews for this item
        max_reviews: Maximum number of reviews to use

    Returns:
        str: Concatenated search text
    """
    text_parts = []

    # 1. Add product metadata (if exists)
    if product_meta:
        # Title
        title = product_meta.get('title', '')
        if title:
            text_parts.append(f"Product: {title}")

        # Features list
        features = product_meta.get('features', [])
        if features and isinstance(features, list):
            features_text = ' '.join([str(f) for f in features if f])
            if features_text:
                text_parts.append(f"Features: {features_text}")

        # Description
        description = product_meta.get('description', [])
        if description and isinstance(description, list):
            desc_text = ' '.join([str(d) for d in description if d])
            if desc_text:
                text_parts.append(f"Description: {desc_text}")

        # Categories
        categories = product_meta.get('categories', [])
        if categories and isinstance(categories, list):
            cat_text = ' '.join([str(c) for c in categories if c])
            if cat_text:
                text_parts.append(f"Categories: {cat_text}")

    # 2. Add reviews (first N)
    reviews_to_use = all_item_reviews[:max_reviews]
    for idx, review in enumerate(reviews_to_use):
        review_parts = []

        # Review title
        review_title = review.get('title', '')
        if review_title:
            review_parts.append(review_title)

        # Review text
        review_text = review.get('text', '')
        if review_text:
            review_parts.append(review_text)

        if review_parts:
            review_combined = ' '.join(review_parts)
            text_parts.append(f"Review: {review_combined}")

    # Concatenate all parts
    final_text = '\n'.join(text_parts)
    return final_text if final_text else "No content available"

def load_json(path: Path):
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8")
    txt = txt.replace("NaN", "null")
    return json.loads(txt)

def calculate_item_avg_ratings(item_review_data: dict) -> Dict[str, float]:
    """
    Calculate average rating for each item from item_review_data

    Args:
        item_review_data: dict with structure {parent_asin: [review_list]}

    Returns:
        dict: {parent_asin: avg_rating}
    """
    item_avg_ratings = {}

    for parent_asin, reviews in item_review_data.items():
        if not isinstance(reviews, list):
            continue

        ratings = []
        for review in reviews:
            if isinstance(review, dict):
                rating = review.get("rating")
                if rating is not None:
                    try:
                        ratings.append(float(rating))
                    except (ValueError, TypeError):
                        pass

        if ratings:
            item_avg_ratings[parent_asin] = sum(ratings) / len(ratings)
        else:
            item_avg_ratings[parent_asin] = None

    return item_avg_ratings

def calculate_item_review_counts(item_review_data: dict) -> Dict[str, int]:
    """
    Calculate total review count for each item from item_review_data

    Args:
        item_review_data: dict with structure {parent_asin: [review_list]}

    Returns:
        dict: {parent_asin: review_count}
    """
    return {
        parent_asin: len(reviews)
        for parent_asin, reviews in item_review_data.items()
        if isinstance(reviews, list)
    }

def build_item_meta_map(user_data: dict) -> Dict[str, dict]:
    """
    Extract all item meta information from user_data

    Args:
        user_data: Data with structure {user_id: [[product_meta, reviews], ...]}

    Returns:
        dict: {item_id (parent_asin): product_meta}
    """
    item_meta_map = {}

    for user_id, purchases in user_data.items():
        if not isinstance(purchases, list):
            continue

        for block in purchases:
            if not isinstance(block, list) or len(block) == 0:
                continue

            product_meta = block[0] if isinstance(block[0], dict) else None
            if not product_meta:
                continue

            # Use parent_asin as item_id
            item_id = product_meta.get('parent_asin')
            if item_id and item_id not in item_meta_map:
                item_meta_map[item_id] = product_meta

    return item_meta_map

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

class Embedder:
    def __init__(self):
        print("[Embedder] Loading SentenceTransformer: all-MiniLM-L6-v2")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

def normalize_item_docs_from_itemReview_v1(
    all_users: Any,
    item_avg_ratings: Dict[str, float],
    item_review_counts: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    V1: One document per review, using item average rating and total review count from item_review_file
    """
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
            parent_asin = meta.get("parent_asin")
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

                review_parent_asin = rv.get("parent_asin") or parent_asin
                review_title = rv.get("title") or ""
                review_text = rv.get("text") or ""

                full_text = " ".join([
                    norm(review_title),
                    norm(review_text),
                ]).strip()

                if not full_text:
                    continue

                # Use parent_asin as item_id
                item_id = parent_asin or f"ITEM_{len(docs)+1}"
                doc_id = f"item::{item_id}::{_hash_bytes(full_text.encode('utf-8'))[:8]}"

                # Use item average rating and review count from item_review_file (keyed by parent_asin)
                avg_rating = item_avg_ratings.get(item_id)
                num_reviews = item_review_counts.get(item_id, 0)

                docs.append({
                    "doc_id": doc_id,
                    "item_id": item_id,
                    "title": norm(product_title),
                    "brand": norm(brand),
                    "price": norm(price),
                    "rating": avg_rating,
                    "num_reviews": num_reviews,  # Total reviews from item_review_file
                    "text": full_text,
                    "source": "user_data_review",
                })

    return docs

def normalize_item_docs_from_itemReview_v2(
    item_review_data: dict,
    item_meta_map: Dict[str, dict],
    item_avg_ratings: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    V2: Create one document per item using only product metadata (no reviews)
    Uses item_review_data as primary source and item_meta_map for metadata

    Args:
        item_review_data: dict with structure {parent_asin: [review_list]}
        item_meta_map: dict with structure {parent_asin: product_meta}
        item_avg_ratings: dict with structure {parent_asin: avg_rating}

    Returns:
        List of document dicts
    """
    docs: List[Dict[str, Any]] = []

    for parent_asin, reviews in item_review_data.items():
        if not isinstance(reviews, list) or len(reviews) == 0:
            continue

        # Get product metadata (may be None if not in user_data)
        product_meta = item_meta_map.get(parent_asin)

        # Use construct_item_search_text to create structured text (max_reviews=0 means no reviews)
        full_text = construct_item_search_text(product_meta, reviews, max_reviews=0)

        # Extract metadata
        product_title = ""
        brand = ""
        price = ""

        if product_meta:
            product_title = product_meta.get("title", "") or ""
            store = product_meta.get("store", "") or ""
            details = product_meta.get("details") or {}
            brand = details.get("Brand") or product_meta.get("brand") or store
            price = product_meta.get("price")

        # Get pre-calculated average rating (keyed by parent_asin)
        avg_rating = item_avg_ratings.get(parent_asin)

        doc_id = f"item::{parent_asin}"

        docs.append({
            "doc_id": doc_id,
            "item_id": parent_asin,
            "title": norm(product_title),
            "brand": norm(brand),
            "price": norm(price),
            "rating": avg_rating,
            "num_reviews": len(reviews),
            "text": full_text,
            "source": "item_review_metadata_only",
        })

    return docs

def build_item_index_for_category(category: str, version: str = "v1"):
    """
    Build item index for a category

    Args:
        category: Category name (e.g., "Amazon_Fashion")
        version: Version of text normalization to use ("v1" or "v2")
                 v1: One document per review (includes review text)
                 v2: One document per item (only product metadata, no reviews)
    """
    # Create version-specific directory
    item_dir = CATALOG_DIR / category / f"item_index_{version}"
    item_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Building index with version: {version}")
    print(f"[INFO] Output directory: {item_dir}")

    # Load data files
    user_data_file, item_review_file = get_files_for_category(category)

    print(f"[INFO] Loading user data from {user_data_file}")
    all_users = load_json(user_data_file)

    print(f"[INFO] Loading item review data from {item_review_file}")
    item_review_data = load_json(item_review_file)
    # Note: item_review_data keys are already parent_asin

    # Calculate item statistics from item_review_file
    print(f"[INFO] Calculating average ratings for each item...")
    item_avg_ratings = calculate_item_avg_ratings(item_review_data)
    print(f"[INFO] Calculated ratings for {len(item_avg_ratings)} items")

    print(f"[INFO] Calculating review counts for each item...")
    item_review_counts = calculate_item_review_counts(item_review_data)
    print(f"[INFO] Calculated review counts for {len(item_review_counts)} items")

    # Build item metadata map from user_data
    print(f"[INFO] Building item metadata map...")
    item_meta_map = build_item_meta_map(all_users)
    print(f"[INFO] Extracted metadata for {len(item_meta_map)} items")

    # Choose normalization function based on version
    if version == "v1":
        print(f"[INFO] Running v1: One doc per review with item avg ratings and review counts")
        docs = normalize_item_docs_from_itemReview_v1(all_users, item_avg_ratings, item_review_counts)
    elif version == "v2":
        print(f"[INFO] Running v2: One doc per item (metadata only) with avg ratings")
        docs = normalize_item_docs_from_itemReview_v2(item_review_data, item_meta_map, item_avg_ratings)
    else:
        raise ValueError(f"Unknown version: {version}. Use 'v1' or 'v2'")

    if not docs:
        print(f"[WARN] No docs built for category={category}")
        return

    print(f"[INFO] Generated {len(docs)} documents")

    # Save normalized docs as JSON
    normalized_docs_path = item_dir / "normalized_docs.json"
    with normalized_docs_path.open("w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] normalized docs -> {normalized_docs_path}")

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
    parser = argparse.ArgumentParser(description="Build RAG index for Amazon product data")
    parser.add_argument("--category", help="e.g. All_Beauty or Amazon_Fashion")
    parser.add_argument("--all", action="store_true", help="Build index for all available categories")
    parser.add_argument("--version", default="v1", choices=["v1", "v2"],
                        help="Version of text normalization: v1 (one doc per review), v2 (one doc per item, metadata only)")

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
                print(f"\n===== Building item index for category: {cat} (from {fname}) [Version: {args.version}] =====")
                build_item_index_for_category(cat, version=args.version)

    elif args.category:
        print(f"\n===== Building item index for category: {args.category} [Version: {args.version}] =====")
        build_item_index_for_category(args.category, version=args.version)
