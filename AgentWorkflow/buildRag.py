import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List
import re
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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

        # Details
        details = product_meta.get('details', {})
        if details and isinstance(details, dict):
            details_text = ' '.join([f"{k}: {v}" for k, v in details.items() if v])
            if details_text:
                text_parts.append(f"Details: {details_text}")

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
    def __init__(self, device: str = None):
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedder] Loading SentenceTransformer: Qwen/Qwen3-Embedding-8B on device: {device}")
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device=device)
        print(f"[Embedder] Model loaded successfully on {self.model.device}")

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        print(f"[Embedder] Encoding {len(texts)} texts with batch_size={batch_size}")
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            prompt_name="document",
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=False
        )

def load_user_target_products(questions_file: Path) -> Dict[str, str]:
    """
    Load questions file and create mapping of user_id -> target_product parent_asin

    Args:
        questions_file: Path to questions JSON file

    Returns:
        Dict mapping user_id to their target product's parent_asin
    """
    if not questions_file.exists():
        print(f"[WARN] Questions file not found: {questions_file}")
        return {}

    questions_data = load_json(questions_file)
    if not questions_data:
        return {}

    user_target_map = {}
    for entry in questions_data:
        user_id = entry.get('user_id')
        target_product = entry.get('target_product', {})
        product_meta = target_product.get('product_meta', {})
        parent_asin = product_meta.get('parent_asin')

        if user_id and parent_asin:
            user_target_map[user_id] = parent_asin

    print(f"[INFO] Loaded {len(user_target_map)} user-target product mappings")
    return user_target_map

def build_review_embeddings(
    item_review_data: dict,
    doc_order: List[str],
    embedder,
    user_target_map: Dict[str, str] = None
) -> np.ndarray:
    """
    Generate average review embeddings for each item

    Args:
        item_review_data: dict with structure {parent_asin: [review_list]}
        doc_order: List of parent_asins in the same order as docs
        embedder: Embedder instance
        user_target_map: Optional dict mapping user_id -> target_product_asin
                        If provided, excludes reviews from users for their target products

    Returns:
        numpy array: Each row is the average embedding of all reviews for that item
    """
    review_embeddings = []
    total_excluded = 0

    for parent_asin in tqdm(doc_order, desc="Building review embeddings"):
        reviews = item_review_data.get(parent_asin, [])

        # Collect all review texts (with optional filtering)
        review_texts = []
        excluded_count = 0

        for rv in reviews:
            if not isinstance(rv, dict):
                continue

            # Filter out target product reviews
            if user_target_map:
                review_user_id = rv.get('user_id')
                target_asin = user_target_map.get(review_user_id)

                # Skip if this review is from a user reviewing their target product
                if review_user_id and target_asin == parent_asin:
                    excluded_count += 1
                    total_excluded += 1
                    continue

            title = rv.get('title', '')
            text = rv.get('text', '')
            combined = f"{title} {text}".strip()
            if combined:
                review_texts.append(combined)

        if review_texts:
            # Generate embeddings for all reviews and take average
            embs = embedder.encode(review_texts, batch_size=8)
            avg_emb = np.mean(embs, axis=0)
            review_embeddings.append(avg_emb)
        else:
            # No reviews: use zero vector
            # Get dimension from embedder
            dim = embedder.model.get_sentence_embedding_dimension()
            review_embeddings.append(np.zeros(dim))

    if total_excluded > 0:
        print(f"[INFO] Excluded {total_excluded} target product reviews from embeddings")

    return np.array(review_embeddings, dtype=np.float32)

def normalize_item_docs_from_itemReview(
    item_review_data: dict,
    item_meta_map: Dict[str, dict]
) -> List[Dict[str, Any]]:
    """
    Create one document per item using only product metadata (no reviews)
    Uses item_review_data as primary source and item_meta_map for metadata

    Args:
        item_review_data: dict with structure {parent_asin: [review_list]}
        item_meta_map: dict with structure {parent_asin: product_meta}

    Returns:
        List of document dicts
    """
    docs: List[Dict[str, Any]] = []

    for parent_asin, reviews in tqdm(item_review_data.items(), desc="Processing items"):
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
        avg_rating = None

        if product_meta:
            product_title = product_meta.get("title", "") or ""
            store = product_meta.get("store", "") or ""
            details = product_meta.get("details") or {}
            brand = details.get("Brand") or product_meta.get("brand") or store
            price = product_meta.get("price")
            # Get average rating from product metadata
            avg_rating = product_meta.get("average_rating")

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
        })

    return docs

def build_item_index_for_category(category: str, questions_file: Path = None):
    """
    Build item index for a category
    Creates one document per item using product metadata only

    Args:
        category: Category name (e.g., "Amazon_Fashion")
        questions_file: Optional path to questions JSON file for excluding target product reviews
    """
    # Create model-specific directory
    model_suffix = "qwen3"
    item_dir = CATALOG_DIR / category / f"item_index_{model_suffix}"
    item_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Building item index with model: {model_suffix}")
    print(f"[INFO] Output directory: {item_dir}")

    # Load data files
    user_data_file, item_review_file = get_files_for_category(category)

    print(f"[INFO] Loading user data from {user_data_file}")
    all_users = load_json(user_data_file)

    print(f"[INFO] Loading item review data from {item_review_file}")
    item_review_data = load_json(item_review_file)
    # Note: item_review_data keys are already parent_asin

    # Load user-target product mapping if questions file provided
    user_target_map = None
    if questions_file and questions_file.exists():
        print(f"[INFO] Loading user-target product mappings from {questions_file}")
        user_target_map = load_user_target_products(questions_file)

    # Build item metadata map from user_data
    print(f"[INFO] Building item metadata map...")
    item_meta_map = build_item_meta_map(all_users)
    print(f"[INFO] Extracted metadata for {len(item_meta_map)} items")

    # Build one document per item (metadata only)
    print(f"[INFO] Building item documents (one per item, metadata only)")
    docs = normalize_item_docs_from_itemReview(item_review_data, item_meta_map)

    if not docs:
        print(f"[WARN] No docs built for category={category}")
        return

    print(f"[INFO] Generated {len(docs)} documents")

    # Save normalized docs as JSON
    normalized_docs_path = item_dir / "normalized_docs.json"
    with normalized_docs_path.open("w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] normalized docs -> {normalized_docs_path}")

    # Encode all fields as text
    texts = [
        " ".join(filter(None, [
            d["title"],
            d["brand"],
            d["price"],
            f"Rating: {d['rating']}" if d.get("rating") is not None else "",
            f"Reviews: {d['num_reviews']}",
            d["text"]
        ]))
        for d in docs
    ]

    embedder_instance = Embedder()
    emb = embedder_instance.encode(texts, batch_size=16)

    # Generate review embeddings
    print(f"[INFO] Generating review embeddings...")
    doc_order = [d["item_id"] for d in docs]
    review_emb = build_review_embeddings(item_review_data, doc_order, embedder_instance, user_target_map)
    print(f"[INFO] Generated review embeddings: {review_emb.shape}")

    meta_path = item_dir / "meta.pkl"
    matrix_path = item_dir / "matrix.pkl"
    embedder_path = item_dir / "embedder.pkl"

    import pickle
    with meta_path.open("wb") as f:
        pickle.dump({"docs": docs, "texts": texts}, f)

    with matrix_path.open("wb") as f:
        pickle.dump(emb, f)

    # Save review embeddings
    review_matrix_path = item_dir / "review_matrix.pkl"
    with review_matrix_path.open("wb") as f:
        pickle.dump(review_emb, f)
    print(f"[SAVE] review embeddings -> {review_matrix_path}")

    with embedder_path.open("wb") as f:
        pickle.dump({"kind": "sbert", "model": "Qwen/Qwen3-Embedding-8B", "dimension": 4096}, f)

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
    emb = Embedder().encode(texts, batch_size=16)

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
    parser.add_argument(
        "--questions-file",
        default="../questions/Amazon_Fashion_questions_20251110_214618.json",
        help="Path to questions JSON file for excluding target product reviews (default: ../questions/Amazon_Fashion_questions_20251110_214618.json)"
    )

    args = parser.parse_args()

    # Parse questions file path if provided
    questions_path = Path(args.questions_file) if args.questions_file else None

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
                build_item_index_for_category(cat, questions_path)

    elif args.category:
        print(f"\n===== Building item index for category: {args.category} =====")
        build_item_index_for_category(args.category, questions_path)
