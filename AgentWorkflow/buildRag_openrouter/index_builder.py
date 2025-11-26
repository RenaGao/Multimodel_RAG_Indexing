"""
Index builder module for constructing RAG indexes.

This module contains the core logic for building:
- Review embeddings for items
- User review embeddings
- Item indexes for categories
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import numpy as np
from tqdm import tqdm

from .config import CATALOG_DIR, BATCH_SIZE, get_files_for_category
from .data_loader import (
    load_json,
    load_user_target_products,
    build_item_meta_map,
    load_questions_data
)
from .embedder import Embedder
from .validator import check_category_complete
from .document_builder import normalize_item_docs_from_itemReview


def build_review_embeddings(
    item_review_data: dict,
    doc_order: List[str],
    embedder,
    user_target_map: Dict[str, str] = None,
    checkpoint_dir: Path = None
) -> np.ndarray:
    """
    Generate average review embeddings for each item (OPTIMIZED VERSION with CHECKPOINT)

    Performance: ~16x faster than original by using global batch processing
    instead of per-item API calls.

    Checkpoint support: Checkpoint logic is now handled inside Embedder.encode().
    Saves progress every 50 batches. If interrupted, automatically resumes from
    last checkpoint on next run.

    Args:
        item_review_data: dict with structure {parent_asin: [review_list]}
        doc_order: List of parent_asins in the same order as docs
        embedder: Embedder instance
        user_target_map: Optional dict mapping user_id -> target_product_asin
                        If provided, excludes reviews from users for their target products
        checkpoint_dir: Directory to save/load checkpoint files (optional, enables checkpoint if provided)

    Returns:
        numpy array: Each row is the average embedding of all reviews for that item
                    Shape: (len(doc_order), embedding_dim)
                    Order: review_embeddings[i] corresponds to doc_order[i]
    """
    print(f"\n[OPTIMIZED] Building review embeddings for {len(doc_order)} items")
    print("[OPTIMIZED] Using global batch processing for 16x speedup")

    # ========================================================================
    # PHASE 1: Collect all review texts and build index mappings
    # ========================================================================
    print("\n[Phase 1/3] Collecting all review texts...")

    all_review_texts = []          # All review texts collected
    item_review_counts = []        # Number of reviews per item (in doc_order)
    empty_text_indices = set()     # Global indices of reviews with empty text (to be zeroed)
    total_excluded = 0
    global_review_idx = 0          # Track global index for empty text detection

    for item_idx, parent_asin in enumerate(tqdm(doc_order, desc="Collecting reviews")):
        reviews = item_review_data.get(parent_asin, [])
        item_review_count = 0

        for rv in reviews:
            if not isinstance(rv, dict):
                continue

            # Filter out target product reviews
            if user_target_map:
                review_user_id = rv.get('user_id')
                target_asin = user_target_map.get(review_user_id)

                # Skip if this review is from a user reviewing their target product
                if review_user_id and target_asin == parent_asin:
                    total_excluded += 1
                    continue

            # Extract review text
            title = rv.get('title', '')
            text = rv.get('text', '')
            combined = f"{title} {text}".strip()

            if combined:
                # Track empty text reviews (text is empty but title exists)
                if not text.strip():
                    empty_text_indices.add(global_review_idx)
                all_review_texts.append(combined)
                item_review_count += 1
                global_review_idx += 1

        # Record how many reviews this item has (in doc_order)
        item_review_counts.append(item_review_count)

    total_reviews = len(all_review_texts)
    print(f"[Phase 1/3] [OK] Collected {total_reviews:,} reviews from {len(doc_order):,} items")
    if total_excluded > 0:
        print(f"[Phase 1/3]   Excluded {total_excluded:,} target product reviews")
    if empty_text_indices:
        print(f"[Phase 1/3]   Found {len(empty_text_indices):,} reviews with empty text (will use zero embedding)")

    # ========================================================================
    # PHASE 2: Generate embeddings for all reviews (with checkpoint support)
    # ========================================================================
    print(f"\n[Phase 2/3] Generating embeddings for {total_reviews:,} reviews...")

    if total_reviews == 0:
        # No reviews at all, return zero vectors for all items
        print("[Phase 2/3] [WARN] No reviews found, returning zero vectors")
        dim = embedder.get_sentence_embedding_dimension()
        return np.zeros((len(doc_order), dim), dtype=np.float32)

    # Prepare checkpoint directory if provided
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Call encode() with checkpoint support and memmap for memory efficiency
    all_embeddings = embedder.encode(
        all_review_texts,
        batch_size=BATCH_SIZE,
        checkpoint_enabled=checkpoint_dir is not None,
        checkpoint_dir=checkpoint_dir,
        checkpoint_id="review_embeddings",
        validation_metadata={
            'task_type': 'review_embeddings',
            'item_review_counts': item_review_counts,
            'total_reviews': total_reviews,
            'num_items': len(doc_order)
        },
        use_memmap=True,        # Enable memory-mapped file for low memory usage
        return_memmap=True      # Return memmap reference for streaming processing
    )

    print(f"[Phase 2/3] [OK] Generated {len(all_embeddings):,} embeddings")
    print(f"[Phase 2/3]   Embedding dimension: {all_embeddings.shape[1]}")

    # Replace empty text embeddings with zero vectors
    if empty_text_indices:
        print(f"[Phase 2/3] Replacing {len(empty_text_indices):,} empty text embeddings with zeros...")
        zero_vector = np.zeros(all_embeddings.shape[1], dtype=np.float32)
        for idx in empty_text_indices:
            all_embeddings[idx] = zero_vector
        if hasattr(all_embeddings, 'flush'):
            all_embeddings.flush()
        print(f"[Phase 2/3] [OK] Replaced {len(empty_text_indices):,} embeddings with zeros")

    # ========================================================================
    # PHASE 3: Group embeddings by item and compute averages
    # ========================================================================
    print(f"\n[Phase 3/3] Computing average embeddings per item...")

    review_embeddings = []
    dim = all_embeddings.shape[1]
    current_review_idx = 0

    for item_idx, review_count in enumerate(tqdm(item_review_counts, desc="Averaging embeddings")):
        if review_count > 0:
            # Extract embeddings for this item (memmap slicing is lazy - no copy)
            item_embeddings = all_embeddings[current_review_idx:current_review_idx + review_count]
            # np.mean reads from disk in chunks, computes average incrementally
            avg_emb = np.mean(item_embeddings, axis=0)

            # Normalize the averaged embedding
            norm = np.linalg.norm(avg_emb)
            if norm > 1e-8:
                avg_emb = avg_emb / norm

            review_embeddings.append(avg_emb)
            current_review_idx += review_count
        else:
            # No reviews for this item, use zero vector
            review_embeddings.append(np.zeros(dim))

    result = np.array(review_embeddings, dtype=np.float32)
    print(f"[Phase 3/3] [OK] Final result shape: {result.shape}")

    # ========================================================================
    # VERIFICATION: Ensure correctness
    # ========================================================================
    print(f"\n[Verification] Validating embedding order...")

    # Verify 1: Check array length matches doc_order
    assert len(result) == len(doc_order), \
        f"Length mismatch: {len(result)} != {len(doc_order)}"
    print(f"[Verification] [OK] Length check passed: {len(result)} items")

    # Verify 2: Check total reviews processed
    assert current_review_idx == total_reviews, \
        f"Review count mismatch: processed {current_review_idx} != collected {total_reviews}"
    print(f"[Verification] [OK] Review count check passed: {total_reviews:,} reviews")

    # Verify 3: Sample validation (check random items)
    import random
    sample_size = min(5, len(doc_order))
    sample_indices = random.sample(range(len(doc_order)), sample_size)

    for idx in sample_indices:
        parent_asin = doc_order[idx]
        expected_count = item_review_counts[idx]

        # Count actual reviews for this item (must match Phase 1 logic: title + text)
        actual_reviews = item_review_data.get(parent_asin, [])
        actual_count = sum(1 for r in actual_reviews
                          if isinstance(r, dict) and f"{r.get('title', '')} {r.get('text', '')}".strip())

        # Account for excluded reviews (must match Phase 1 logic)
        if user_target_map:
            excluded = sum(1 for r in actual_reviews
                          if isinstance(r, dict) and f"{r.get('title', '')} {r.get('text', '')}".strip()
                          and user_target_map.get(r.get('user_id')) == parent_asin)
            actual_count -= excluded

        assert expected_count == actual_count, \
            f"Sample check failed for item {idx} ({parent_asin}): {expected_count} != {actual_count}"

    print(f"[Verification] [OK] Sample check passed: {sample_size} random items validated")
    print(f"[Verification] [SUCCESS] All verification checks passed!")

    print(f"\n[OPTIMIZED] [SUCCESS] Review embeddings complete!")
    print(f"[OPTIMIZED]    Items: {len(doc_order):,}")
    print(f"[OPTIMIZED]    Reviews: {total_reviews:,}")
    print(f"[OPTIMIZED]    Embedding dim: {dim}")

    # Clean up memmap file
    if isinstance(all_embeddings, np.memmap):
        memmap_filename = all_embeddings.filename
        del all_embeddings  # Close memmap
        if memmap_filename and Path(memmap_filename).exists():
            Path(memmap_filename).unlink()
            print(f"[Memmap] Temporary file deleted: {Path(memmap_filename).name}")

    return result


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
    embeddings = embedder.encode(review_texts, batch_size=BATCH_SIZE)

    # Create metadata
    metadata = {
        "user_id": user_id,
        "reviews": all_reviews,
        "embedding_dim": embeddings.shape[1],
        "num_reviews": len(all_reviews),
        "model": "qwen/qwen3-embedding-8b",
        "model_type": "openrouter_api",
        "target_product_excluded": target_product_asin,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"
    }

    # Save embeddings and metadata
    emb_path = output_dir / f"{user_id}_review_embs.pkl"
    meta_path = output_dir / f"{user_id}_review_meta.pkl"

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
            # Check if embeddings already exist for this user
            emb_path = output_dir / f"{user_id}_review_embs.pkl"
            meta_path = output_dir / f"{user_id}_review_meta.pkl"

            if emb_path.exists() and meta_path.exists():
                success_count += 1
                continue

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

    # Check what's already complete
    status = check_category_complete(category, model_suffix)

    skip_item_meta = status["meta_complete"] and status.get("item_count_valid", True)
    skip_product_review = status["product_review_complete"]
    skip_user_embeddings = status["user_review_complete"]

    if skip_item_meta and skip_product_review and skip_user_embeddings:
        print(f"[INFO] Category {category} is already complete, nothing to rebuild")
        return

    # Show what needs to be rebuilt
    rebuild_parts = []
    if not skip_item_meta:
        rebuild_parts.append("item metadata")
    if not skip_product_review:
        rebuild_parts.append("product reviews")
    if not skip_user_embeddings:
        rebuild_parts.append("user embeddings")

    print(f"[INFO] Will rebuild: {', '.join(rebuild_parts)}")

    # If we need to rebuild item meta, we must also rebuild product reviews (due to doc_order dependency)
    if not skip_item_meta:
        skip_product_review = False
        print(f"[INFO] Item metadata rebuild required, will also rebuild product reviews (dependency)")

    # Load data files
    user_data_file, item_review_file = get_files_for_category(category)

    print(f"[INFO] Loading user data from {user_data_file}")
    all_users = load_json(user_data_file)

    # Load user-target product mapping if questions file provided
    user_target_map = None
    if questions_file and questions_file.exists():
        print(f"[INFO] Loading user-target product mappings from {questions_file}")
        user_target_map = load_user_target_products(questions_file)

    # Variables that might be needed later
    embedder_instance = None
    docs = None
    doc_order = None

    # ===== ITEM METADATA BUILDING =====
    if not skip_item_meta:
        print(f"\n[INFO] ===== Building Item Metadata =====")

        print(f"[INFO] Loading item review data from {item_review_file}")
        item_review_data = load_json(item_review_file)

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
        emb = embedder_instance.encode(texts, batch_size=BATCH_SIZE)

        meta_path = item_dir / "meta.pkl"
        matrix_path = item_dir / "matrix.pkl"
        embedder_path = item_dir / "embedder.pkl"

        with meta_path.open("wb") as f:
            pickle.dump({"docs": docs, "texts": texts}, f)

        with matrix_path.open("wb") as f:
            pickle.dump(emb, f)

        with embedder_path.open("wb") as f:
            pickle.dump({"kind": "openrouter_api", "model": "qwen/qwen3-embedding-8b", "dimension": 4096}, f)

        print(f"[SAVE] item metadata -> {item_dir}")
        doc_order = [d["item_id"] for d in docs]

    else:
        print(f"\n[INFO] Item metadata already complete, skipping...")

    # ===== PRODUCT REVIEW EMBEDDINGS =====
    if not skip_product_review:
        print(f"\n[INFO] ===== Building Product Review Embeddings =====")

        # Load item_review_data if not already loaded
        if 'item_review_data' not in locals():
            print(f"[INFO] Loading item review data from {item_review_file}")
            item_review_data = load_json(item_review_file)

        # Load docs and create embedder if we skipped item meta building
        if docs is None:
            normalized_docs_path = item_dir / "normalized_docs.json"
            if normalized_docs_path.exists():
                with normalized_docs_path.open("r", encoding="utf-8") as f:
                    docs = json.load(f)
                doc_order = [d["item_id"] for d in docs]
            else:
                print(f"[ERROR] Cannot build product reviews without docs")
                return

        # Create embedder if not already created
        if embedder_instance is None:
            embedder_instance = Embedder()

        # Generate review embeddings (with checkpoint support)
        print(f"[INFO] Generating review embeddings...")
        review_emb = build_review_embeddings(
            item_review_data,
            doc_order,
            embedder_instance,
            user_target_map,
            checkpoint_dir=item_dir  # Pass checkpoint directory
        )
        print(f"[INFO] Generated review embeddings: {review_emb.shape}")

        # Save review embeddings
        review_matrix_path = item_dir / "review_matrix.pkl"
        with review_matrix_path.open("wb") as f:
            pickle.dump(review_emb, f)
        print(f"[SAVE] review embeddings -> {review_matrix_path}")

    else:
        print(f"\n[INFO] Product review embeddings already complete, skipping...")

    # ===== USER REVIEW EMBEDDINGS =====
    if not skip_user_embeddings:
        # Build user review embeddings (if questions file provided)
        if questions_file and questions_file.exists():
            print(f"\n[INFO] ===== Building User Review Embeddings =====")
            questions_data = load_questions_data(questions_file)

            if questions_data:
                # Create embedder if not already created
                if embedder_instance is None:
                    embedder_instance = Embedder()

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
        else:
            print(f"[WARN] No questions file provided, cannot build user review embeddings")
    else:
        print(f"\n[INFO] User review embeddings already complete, skipping...")

    print(f"\n[DONE] Build complete for category: {category}")


__all__ = [
    'build_review_embeddings',
    'build_user_review_embeddings_for_single_user',
    'build_all_user_review_embeddings',
    'build_item_index_for_category'
]
