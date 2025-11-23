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

def load_questions_data(questions_file: Path) -> Dict[str, dict]:
    """
    Load questions data and return dict keyed by user_id.

    Args:
        questions_file: Path to questions JSON file

    Returns:
        Dict mapping user_id to question entry (contains target_product, etc.)
    """
    if not questions_file.exists():
        print(f"[WARN] Questions file not found: {questions_file}")
        return {}

    data = load_json(questions_file)
    if not data:
        return {}

    questions_dict = {}
    for entry in data:
        user_id = entry.get("user_id")
        if user_id:
            questions_dict[user_id] = entry

    print(f"[INFO] Loaded questions for {len(questions_dict)} users")
    return questions_dict

def find_questions_file_for_category(category: str) -> Path:
    """
    Automatically find the latest questions file for a given category.
    Searches in ../questions directory for {category}_questions_*.json

    Args:
        category: Category name (e.g., "Amazon_Fashion")

    Returns:
        Path to questions file, or None if not found
    """
    questions_dir = Path("../questions")

    if not questions_dir.exists():
        print(f"[WARN] Questions directory not found: {questions_dir}")
        return None

    # Find matching files: {category}_questions_*.json
    pattern = f"{category}_questions_*.json"
    matching_files = list(questions_dir.glob(pattern))

    if matching_files:
        # If multiple files, choose the latest (by filename, timestamp in name)
        latest_file = sorted(matching_files)[-1]
        print(f"[INFO] Auto-found questions file: {latest_file}")
        return latest_file

    print(f"[WARN] No questions file found for category: {category}")
    return None

def check_category_complete(category: str, model_suffix: str = "qwen3") -> Dict:
    """
    Check if a category's index is completely built.
    Checks three parts:
    1. Item metadata (meta.pkl, normalized_docs.json, embedder.pkl, matrix.pkl)
    2. Product review embeddings (review_matrix.pkl)
    3. User review embeddings (user_review_embeddings/ directory with paired files)

    Args:
        category: Category name (e.g., "Amazon_Fashion")
        model_suffix: Model suffix for item index directory

    Returns:
        Dict with status information
    """
    item_dir = CATALOG_DIR / category / f"item_index_{model_suffix}"
    user_review_dir = CATALOG_DIR / category / "user_review_embeddings"

    # 1. Check item metadata files
    meta_files = ["meta.pkl", "normalized_docs.json", "embedder.pkl", "matrix.pkl"]
    missing_meta = [f for f in meta_files if not (item_dir / f).exists()]
    meta_complete = len(missing_meta) == 0

    # 2. Check product review embedding
    review_emb_complete = (item_dir / "review_matrix.pkl").exists()

    # Get expected counts from source files
    expected_user_count = 0
    expected_item_count = 0
    expected_user_ids = set()

    # Get expected user count from questions file
    questions_file = find_questions_file_for_category(category)
    if questions_file and questions_file.exists():
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
                expected_user_count = len(questions_data)
                expected_user_ids = {q["user_id"] for q in questions_data}
        except Exception as e:
            print(f"Warning: Could not load questions file for {category}: {e}")

    # Get expected item count from normalized_docs.json
    normalized_docs_file = item_dir / "normalized_docs.json"
    if normalized_docs_file.exists():
        try:
            with open(normalized_docs_file, 'r', encoding='utf-8') as f:
                docs = json.load(f)
                expected_item_count = len(docs)
        except Exception as e:
            print(f"Warning: Could not load normalized_docs.json for {category}: {e}")

    # 3. Check user review embeddings with exact count validation
    user_review_complete = False
    actual_user_count = 0
    actual_user_ids = set()
    missing_users = []

    if user_review_dir.exists():
        emb_files = list(user_review_dir.glob("*_review_embs.pkl"))
        meta_files_list = list(user_review_dir.glob("*_review_meta.pkl"))

        # Extract user IDs from embedding files
        actual_user_ids = {f.stem.replace("_review_embs", "") for f in emb_files}
        actual_user_count = len(actual_user_ids)

        # Check if counts match and files are paired
        if expected_user_count > 0:
            # Validate exact count match
            if (len(emb_files) == len(meta_files_list) and
                actual_user_count == expected_user_count):
                user_review_complete = True
            else:
                # Find missing users
                missing_users = sorted(expected_user_ids - actual_user_ids)
        else:
            # Fallback: if no questions file, use old logic (>= 5)
            if len(emb_files) == len(meta_files_list) and len(emb_files) >= 5:
                actual_user_count = len(emb_files)
                user_review_complete = True
            else:
                actual_user_count = len(emb_files)

    # 4. Validate item embedding counts match expected
    item_count_valid = True
    item_count_details = ""

    if expected_item_count > 0 and meta_complete:
        try:
            import pickle
            # Check matrix.pkl shape
            matrix_file = item_dir / "matrix.pkl"
            if matrix_file.exists():
                with open(matrix_file, 'rb') as f:
                    matrix = pickle.load(f)
                    if matrix.shape[0] != expected_item_count:
                        item_count_valid = False
                        item_count_details = f"matrix shape mismatch: {matrix.shape[0]} != {expected_item_count}"

            # Check review_matrix.pkl shape
            if review_emb_complete and item_count_valid:
                review_matrix_file = item_dir / "review_matrix.pkl"
                with open(review_matrix_file, 'rb') as f:
                    review_matrix = pickle.load(f)
                    if review_matrix.shape[0] != expected_item_count:
                        item_count_valid = False
                        item_count_details = f"review_matrix shape mismatch: {review_matrix.shape[0]} != {expected_item_count}"
        except Exception as e:
            item_count_valid = False
            item_count_details = f"Error validating item counts: {e}"

    # All parts must be complete
    complete = meta_complete and review_emb_complete and user_review_complete and item_count_valid

    # Generate detailed status
    status_parts = []
    if not meta_complete:
        status_parts.append(f"Missing item meta: {', '.join(missing_meta)}")
    if not item_count_valid and item_count_details:
        status_parts.append(item_count_details)
    if not review_emb_complete:
        status_parts.append("Missing product review embedding")
    if not user_review_complete:
        if actual_user_count == 0:
            status_parts.append("No user review embeddings")
        elif expected_user_count > 0:
            status_parts.append(f"User embeddings: {actual_user_count}/{expected_user_count}")
            if missing_users:
                # Show first 3 missing users
                missing_sample = missing_users[:3]
                missing_display = ", ".join(missing_sample)
                if len(missing_users) > 3:
                    missing_display += f" (+{len(missing_users)-3} more)"
                status_parts.append(f"Missing users: {missing_display}")
        else:
            status_parts.append(f"User review embeddings insufficient: {actual_user_count} users (need >= 5)")

    if status_parts:
        details = " | ".join(status_parts)
    else:
        details = f"Complete ({actual_user_count} users, {expected_item_count} items)"

    return {
        "complete": complete,
        "meta_complete": meta_complete,
        "product_review_complete": review_emb_complete,
        "user_review_complete": user_review_complete,
        "item_count_valid": item_count_valid,
        "user_count": actual_user_count,
        "expected_user_count": expected_user_count,
        "expected_item_count": expected_item_count,
        "missing_users": missing_users,
        "details": details
    }

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

def build_user_review_embeddings_for_single_user(
    user_id: str,
    user_purchases: List,
    target_product_asin: str,
    embedder: Embedder,
    output_dir: Path
) -> bool:
    """
    Build review embeddings for a single user (excluding target_product).

    Args:
        user_id: User identifier
        user_purchases: List of [(product_meta, [reviews])] for this user
        target_product_asin: parent_asin of target product to exclude
        embedder: Embedder instance for generating embeddings
        output_dir: Directory to save embeddings

    Returns:
        bool: True if successful, False otherwise
    """
    # Collect all reviews except target_product
    all_reviews = []
    review_texts = []

    for block in user_purchases:
        if not isinstance(block, list) or len(block) < 2:
            continue

        product_meta = block[0] if isinstance(block[0], dict) else None
        reviews = block[1] if len(block) > 1 and isinstance(block[1], list) else []

        if not product_meta:
            continue

        current_product_asin = product_meta.get("parent_asin")

        # Skip target product
        if current_product_asin == target_product_asin:
            continue

        # Process each review
        for review in reviews:
            if not isinstance(review, dict):
                continue

            review_text = review.get("text", "").strip()
            if not review_text:
                continue

            # Enrich review with product context
            enriched_review = {
                "review_text": review_text,
                "rating": review.get("rating"),
                "title": review.get("title", ""),
                "product_title": product_meta.get("title", ""),
                "product_brand": product_meta.get("store", ""),
                "product_price": product_meta.get("price"),
                "product_id": current_product_asin,
                "timestamp": review.get("timestamp", ""),
            }

            all_reviews.append(enriched_review)
            review_texts.append(review_text)

    if not all_reviews:
        return False

    # Generate embeddings
    embeddings = embedder.encode(review_texts, batch_size=16)

    # Create metadata
    from datetime import datetime
    metadata = {
        "user_id": user_id,
        "reviews": all_reviews,
        "embedding_dim": embeddings.shape[1],
        "num_reviews": len(all_reviews),
        "model": "Qwen/Qwen3-Embedding-8B",
        "target_product_excluded": target_product_asin,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"
    }

    # Save embeddings and metadata
    emb_path = output_dir / f"{user_id}_review_embs.pkl"
    meta_path = output_dir / f"{user_id}_review_meta.pkl"

    import pickle
    with emb_path.open("wb") as f:
        pickle.dump(embeddings, f)

    with meta_path.open("wb") as f:
        pickle.dump(metadata, f)

    return True

def build_all_user_review_embeddings(
    category: str,
    user_data: dict,
    questions_data: dict,
    embedder: Embedder
) -> Dict[str, int]:
    """
    Build review embeddings for all users in the dataset.

    Args:
        category: Category name (e.g., "Amazon_Fashion")
        user_data: Dict mapping user_id to purchase data
        questions_data: Dict mapping user_id to question entry (contains target_product)
        embedder: Embedder instance for generating embeddings

    Returns:
        Dict with statistics (success_count, skip_count)
    """
    # Create output directory
    output_dir = CATALOG_DIR / category / "user_review_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] User review embeddings output: {output_dir}")

    # Find users that exist in both datasets
    common_users = set(user_data.keys()) & set(questions_data.keys())
    print(f"[INFO] Found {len(common_users)} users in both datasets")

    success_count = 0
    skip_count = 0

    for user_id in tqdm(sorted(common_users), desc="Building user review embeddings"):
        try:
            # Get user's purchase data
            user_purchases = user_data[user_id]

            # Get target product from questions data
            target_product = questions_data[user_id].get("target_product")

            if not target_product:
                skip_count += 1
                continue

            target_product_meta = target_product.get("product_meta", {})
            target_product_asin = target_product_meta.get("parent_asin")

            if not target_product_asin:
                skip_count += 1
                continue

            # Build embeddings for this user
            success = build_user_review_embeddings_for_single_user(
                user_id=user_id,
                user_purchases=user_purchases,
                target_product_asin=target_product_asin,
                embedder=embedder,
                output_dir=output_dir
            )

            if success:
                success_count += 1
            else:
                skip_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to process user {user_id}: {e}")
            skip_count += 1
            continue

    return {
        "success": success_count,
        "skipped": skip_count,
        "total": len(common_users)
    }

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

    # Build user review embeddings (if questions file provided)
    if questions_file and questions_file.exists():
        print(f"\n[INFO] ====== Building User Review Embeddings ======")
        questions_data = load_questions_data(questions_file)

        if questions_data:
            stats = build_all_user_review_embeddings(
                category=category,
                user_data=all_users,
                questions_data=questions_data,
                embedder=embedder_instance  # Reuse the existing embedder
            )

            print(f"\n[DONE] User review embeddings complete:")
            print(f"       Successfully processed: {stats['success']} users")
            print(f"       Skipped: {stats['skipped']} users")
            print(f"       Total: {stats['total']} users")
        else:
            print(f"[WARN] No questions data loaded, skipping user review embeddings")


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

        print(f"\n{'='*80}")
        print(f"Checking completeness for {len(cat2file)} categories...")
        print(f"Check items: 1. Item Meta  2. Product Review Embedding  3. User Review Embedding")
        print(f"{'='*80}\n")

        # Check completion status for all categories
        to_process = []
        already_complete = []

        for cat in sorted(cat2file.keys()):
            status = check_category_complete(cat)

            if status["complete"]:
                already_complete.append(cat)
                print(f"✅ {cat:30s} - {status['details']}")
            else:
                to_process.append(cat)
                print(f"⚠️  {cat:30s} - {status['details']}")

        # Display statistics
        print(f"\n{'='*80}")
        print(f"Complete:   {len(already_complete)}/{len(cat2file)}")
        print(f"To process: {len(to_process)}/{len(cat2file)}")
        print(f"{'='*80}\n")

        if not to_process:
            print("[INFO] All categories are complete!")
        else:
            print(f"[INFO] Processing {len(to_process)} incomplete categories...\n")

            # Process incomplete categories
            for cat in to_process:
                fname = cat2file[cat]
                print(f"\n===== Building: {cat} (from {fname}) =====")

                # Auto-find questions file for this category
                questions_file = find_questions_file_for_category(cat)

                build_item_index_for_category(cat, questions_file)

    elif args.category:
        print(f"\n===== Building item index for category: {args.category} =====")

        # Determine questions file
        if args.questions_file:
            # Use user-specified questions file
            questions_path = Path(args.questions_file)
        else:
            # Auto-find questions file
            questions_path = find_questions_file_for_category(args.category)

        build_item_index_for_category(args.category, questions_path)
