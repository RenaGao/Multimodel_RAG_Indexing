#!/usr/bin/env python
"""
Fix script for empty text review embeddings.

This script locates reviews where `text` is empty/whitespace but `title` exists,
and replaces their embeddings with zero vectors in the memmap file.

Usage:
    python fix_empty_text_embeddings.py --category Beauty_and_Personal_Care
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Import config
from buildRag_openrouter.config import CATALOG_DIR, DATA_DIR, get_files_for_category
from buildRag_openrouter.data_loader import load_json, load_user_target_products, find_questions_file_for_category


def fix_empty_text_embeddings(category: str, dry_run: bool = False):
    """
    Fix embeddings for reviews with empty text.

    Args:
        category: Category name (e.g., "Beauty_and_Personal_Care")
        dry_run: If True, only report what would be fixed without modifying
    """
    model_suffix = "qwen3"
    item_dir = CATALOG_DIR / category / f"item_index_{model_suffix}"

    # Check required files exist
    checkpoint_file = item_dir / "review_embeddings_checkpoint.pkl"
    memmap_file = item_dir / "review_embeddings_embeddings.dat"
    normalized_docs_file = item_dir / "normalized_docs.json"

    print(f"[INFO] Category: {category}")
    print(f"[INFO] Item dir: {item_dir}")

    if not checkpoint_file.exists():
        print(f"[ERROR] Checkpoint file not found: {checkpoint_file}")
        return False

    if not memmap_file.exists():
        print(f"[ERROR] Memmap file not found: {memmap_file}")
        return False

    if not normalized_docs_file.exists():
        print(f"[ERROR] Normalized docs file not found: {normalized_docs_file}")
        return False

    # Load checkpoint to get metadata
    print(f"\n[Step 1/5] Loading checkpoint...")
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)

    validation_metadata = checkpoint_data.get('validation_metadata', {})
    item_review_counts = validation_metadata.get('item_review_counts', [])
    total_texts = checkpoint_data.get('total_texts')
    embedding_dim = checkpoint_data.get('embedding_dim', 4096)

    print(f"  Total texts: {total_texts:,}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Items with review counts: {len(item_review_counts)}")

    # Load normalized docs to get doc_order
    print(f"\n[Step 2/5] Loading doc order...")
    with open(normalized_docs_file, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    doc_order = [d["item_id"] for d in docs]
    print(f"  Doc order length: {len(doc_order)}")

    # Load item review data
    print(f"\n[Step 3/5] Loading item review data...")
    _, item_review_file = get_files_for_category(category)
    item_review_data = load_json(item_review_file)
    print(f"  Loaded {len(item_review_data)} items")

    # Load user target map for excluding target product reviews
    questions_file = find_questions_file_for_category(category)
    user_target_map = None
    if questions_file and questions_file.exists():
        user_target_map = load_user_target_products(questions_file)
        print(f"  Loaded user target map: {len(user_target_map)} users")

    # Find reviews with empty text
    print(f"\n[Step 4/5] Scanning for empty text reviews...")

    empty_text_indices = []
    current_global_idx = 0

    for item_idx, parent_asin in enumerate(tqdm(doc_order, desc="Scanning items")):
        reviews = item_review_data.get(parent_asin, [])

        for rv in reviews:
            if not isinstance(rv, dict):
                continue

            # Skip target product reviews (same logic as Phase 1)
            if user_target_map:
                review_user_id = rv.get('user_id')
                target_asin = user_target_map.get(review_user_id)
                if review_user_id and target_asin == parent_asin:
                    continue

            # Check if this review was included in embedding generation
            title = rv.get('title', '')
            text = rv.get('text', '')
            combined = f"{title} {text}".strip()

            if combined:  # This review was included
                # Check if text is empty/whitespace
                if not text.strip():
                    empty_text_indices.append({
                        'global_idx': current_global_idx,
                        'item_idx': item_idx,
                        'parent_asin': parent_asin,
                        'title': title[:50] + '...' if len(title) > 50 else title,
                        'text': repr(text)
                    })
                current_global_idx += 1

    print(f"\n  Found {len(empty_text_indices)} reviews with empty text")

    if empty_text_indices:
        print(f"\n  Sample of empty text reviews:")
        for info in empty_text_indices[:5]:
            print(f"    - Index {info['global_idx']}: item={info['parent_asin']}, "
                  f"title='{info['title']}', text={info['text']}")
        if len(empty_text_indices) > 5:
            print(f"    ... and {len(empty_text_indices) - 5} more")

    if dry_run:
        print(f"\n[DRY RUN] Would fix {len(empty_text_indices)} embeddings")
        return True

    if not empty_text_indices:
        print(f"\n[INFO] No empty text reviews found, nothing to fix")
        return True

    # Fix embeddings in memmap
    print(f"\n[Step 5/5] Fixing embeddings in memmap...")

    # Open memmap in read-write mode
    embeddings_memmap = np.memmap(
        memmap_file,
        dtype='float32',
        mode='r+',
        shape=(total_texts, embedding_dim)
    )

    # Replace with zero vectors
    zero_vector = np.zeros(embedding_dim, dtype=np.float32)

    for info in tqdm(empty_text_indices, desc="Fixing embeddings"):
        idx = info['global_idx']
        embeddings_memmap[idx] = zero_vector

    # Flush changes to disk
    embeddings_memmap.flush()
    del embeddings_memmap

    print(f"\n[SUCCESS] Fixed {len(empty_text_indices)} embeddings")
    print(f"[INFO] Memmap file updated: {memmap_file}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix embeddings for reviews with empty text"
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Category name (e.g., Beauty_and_Personal_Care)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be fixed, don't modify files"
    )

    args = parser.parse_args()

    success = fix_empty_text_embeddings(args.category, args.dry_run)

    if success:
        print(f"\n[DONE] Fix completed successfully")
        print(f"[NEXT] Run: python build_rag.py --all")
    else:
        print(f"\n[FAILED] Fix failed")
        exit(1)


if __name__ == "__main__":
    main()
