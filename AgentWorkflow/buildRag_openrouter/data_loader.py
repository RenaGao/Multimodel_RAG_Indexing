"""
Data loading module for RAG index building.

This module provides utilities for:
- JSON file loading with NaN handling
- Data normalization and extraction
- Questions file loading and processing
- Item metadata extraction from user data
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List


def norm(x: Any) -> str:
    """
    Normalize any value to a string representation.

    Args:
        x: Value to normalize (can be None, dict, list, or any type)

    Returns:
        String representation of the value
    """
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)


def load_json(path: Path):
    """
    Load JSON file with NaN value replacement.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data, or None if file doesn't exist
    """
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8")
    txt = txt.replace("NaN", "null")
    return json.loads(txt)


def _hash_bytes(b: bytes) -> str:
    """
    Generate SHA1 hash of bytes.

    Args:
        b: Bytes to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha1(b).hexdigest()


def calculate_item_review_counts(item_review_data: dict) -> Dict[str, int]:
    """
    Calculate total review count for each item from item_review_data.

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
    Extract all item meta information from user_data.

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


def load_user_target_products(questions_file: Path) -> Dict[str, str]:
    """
    Load questions file and create mapping of user_id -> target_product parent_asin.

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
