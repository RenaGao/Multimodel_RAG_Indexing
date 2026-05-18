"""
Validator module for checking RAG index completeness.

This module provides functions to validate:
- Item metadata completeness
- Product review embeddings
- User review embeddings
- Data consistency checks
"""

import json
import pickle
from pathlib import Path
from typing import Dict

from .config import CATALOG_DIR
from .data_loader import find_questions_file_for_category


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
