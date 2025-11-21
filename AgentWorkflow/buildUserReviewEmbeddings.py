"""
Build user review embeddings for dynamic review selection.

This script generates embeddings for each user's reviews (excluding target_product)
to enable dynamic selection based on query similarity.

Usage:
    python buildUserReviewEmbeddings.py
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths
BASE_DIR = Path("./storage")
VSTORE_DIR = BASE_DIR / "vectorstore"
CATALOG_DIR = VSTORE_DIR / "catalog"

DATASET_CATEGORY = "Amazon_Fashion"
REVIEW_EMBEDDINGS_DIR = CATALOG_DIR / DATASET_CATEGORY / "user_review_embeddings"

USER_DATA_PATH = Path("/data/gpfs/projects/punim2620/Multimodel_RAG_Indexing/AmazonReviews/Processed/user_center/Amazon_Fashion_top_100_user_data.json")
QUESTIONS_PATH = Path("../questions/Amazon_Fashion_questions_20251110_214618.json")

# Ensure directories exist
for p in [VSTORE_DIR, CATALOG_DIR, REVIEW_EMBEDDINGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_user_data(path: Path) -> Dict:
    """
    Load user purchase data from JSON file.

    Args:
        path: Path to user data JSON file

    Returns:
        Dict mapping user_id to list of [(product_meta, [reviews])]
    """
    print(f"Loading user data from: {path}")

    if not path.exists():
        raise FileNotFoundError(f"User data file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded data for {len(data)} users")
    return data


def load_questions_data(path: Path) -> Dict:
    """
    Load questions data to get user-target_product mapping.

    Args:
        path: Path to questions JSON file

    Returns:
        Dict mapping user_id to question data (contains target_product)
    """
    print(f"Loading questions data from: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert list to dict keyed by user_id
    questions_dict = {}
    for entry in data:
        user_id = entry.get("user_id")
        if user_id:
            questions_dict[user_id] = entry

    print(f"Loaded questions for {len(questions_dict)} users")
    return questions_dict


def build_user_review_embeddings(
    user_id: str,
    user_data: List,
    target_product: Dict,
    model: SentenceTransformer
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Build embeddings for user's reviews excluding target_product.

    Args:
        user_id: User identifier
        user_data: List of [(product_meta, [reviews])] for this user
        target_product: Target product dict from questions data
        model: SentenceTransformer model for embedding generation

    Returns:
        Tuple of (embeddings_matrix, metadata_dict) or (None, None) if no reviews
    """
    # Extract target product ID
    target_product_meta = target_product.get("product_meta", {})
    target_product_id = target_product_meta.get("parent_asin")

    # Collect all reviews except target_product
    all_reviews = []
    review_texts = []

    for product_meta, reviews in user_data:
        current_product_id = product_meta.get("parent_asin")

        # Skip target product
        if current_product_id == target_product_id:
            continue

        # Process each review
        for review in reviews:
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
                "product_id": current_product_id,
                "timestamp": review.get("timestamp", ""),
            }

            all_reviews.append(enriched_review)
            review_texts.append(review_text)

    if not all_reviews:
        print(f"[WARNING] No reviews found for user {user_id} (after excluding target)")
        return None, None

    # Generate embeddings
    print(f"  Generating embeddings for {len(review_texts)} reviews...")
    embeddings = model.encode(
        review_texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    )

    # Create metadata
    metadata = {
        "user_id": user_id,
        "reviews": all_reviews,
        "embedding_dim": embeddings.shape[1],
        "num_reviews": len(all_reviews),
        "model": "Qwen/Qwen3-Embedding-8B",
        "target_product_excluded": target_product_id,
        "created_at": _now()
    }

    return embeddings, metadata


def save_user_review_embeddings(user_id: str, embeddings: np.ndarray, metadata: Dict):
    """
    Save user review embeddings and metadata to catalog directory.

    Args:
        user_id: User identifier
        embeddings: Numpy array of embeddings (N_reviews, embedding_dim)
        metadata: Metadata dict
    """
    # Use catalog directory with user_id as filename
    emb_path = REVIEW_EMBEDDINGS_DIR / f"{user_id}_review_embs.pkl"
    meta_path = REVIEW_EMBEDDINGS_DIR / f"{user_id}_review_meta.pkl"

    # Save embeddings
    with emb_path.open("wb") as f:
        pickle.dump(embeddings, f)

    # Save metadata
    with meta_path.open("wb") as f:
        pickle.dump(metadata, f)

    print(f"  Saved {len(embeddings)} review embeddings to {REVIEW_EMBEDDINGS_DIR}")


def main():
    """Main execution function."""
    print("="*80)
    print("Building User Review Embeddings")
    print("="*80)
    print()

    # Load data
    user_data = load_user_data(USER_DATA_PATH)
    questions_data = load_questions_data(QUESTIONS_PATH)
    print()

    # Find users that exist in both datasets
    common_users = set(user_data.keys()) & set(questions_data.keys())
    print(f"Found {len(common_users)} users in both datasets")
    print()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer model (device: {device})...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device=device)
    print("Model loaded successfully")
    print()

    # Process each user
    print(f"Processing {len(common_users)} users...")
    print()

    success_count = 0
    skip_count = 0

    for user_id in tqdm(sorted(common_users), desc="Building embeddings"):
        try:
            # Get user's purchase data
            user_purchases = user_data[user_id]

            # Get target product from questions data
            target_product = questions_data[user_id].get("target_product")

            if not target_product:
                print(f"[SKIP] User {user_id}: No target_product in questions data")
                skip_count += 1
                continue

            # Build embeddings
            embeddings, metadata = build_user_review_embeddings(
                user_id,
                user_purchases,
                target_product,
                model
            )

            if embeddings is None:
                skip_count += 1
                continue

            # Save embeddings
            save_user_review_embeddings(user_id, embeddings, metadata)
            success_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to process user {user_id}: {e}")
            skip_count += 1
            continue

    print()
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"Total users processed: {len(common_users)}")
    print(f"Successfully built embeddings: {success_count}")
    print(f"Skipped (no reviews or errors): {skip_count}")
    print()
    print(f"Embeddings saved to: {REVIEW_EMBEDDINGS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
