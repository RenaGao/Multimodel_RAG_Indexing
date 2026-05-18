"""
Configuration module for RAG index building.

This module contains all configuration constants, directory paths,
and configuration-related functions.
"""

import os
from pathlib import Path

# ============================================================================
# Batch Size Configuration
# ============================================================================
# Batch size: number of texts processed per batch
# - Controls API call size (texts per request)
# - Controls checkpoint frequency (via CHECKPOINT_INTERVAL)
BATCH_SIZE = 1000

# Checkpoint interval: save checkpoint every N batches
# (saves every CHECKPOINT_INTERVAL * BATCH_SIZE texts)
CHECKPOINT_INTERVAL = 50
# ============================================================================

# ============================================================================
# Directory Paths
# ============================================================================
BASE_DIR = Path("./storage")
VSTORE_DIR = BASE_DIR / "vectorstore"
USERS_DIR = VSTORE_DIR / "users"
CATALOG_DIR = VSTORE_DIR / "catalog"
# Directory containing per-category processed Amazon review JSONs. Override via
# USER_REVIEWS_DIR env var so the path stays out of source control.
DATA_DIR = Path(os.environ.get("USER_REVIEWS_DIR", "./data/amazon_reviews"))

# Create directories if they don't exist (skip DATA_DIR — it's an input, not an output)
for p in [VSTORE_DIR, USERS_DIR, CATALOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)
# ============================================================================


def get_files_for_category(category: str):
    """
    Get data file paths for a specific category.

    Args:
        category: Amazon category name (e.g., 'Amazon_Fashion')

    Returns:
        Tuple of (user_data_file, item_review_file) paths
    """
    user_data_file = DATA_DIR / f"{category}_top_100_user_data.json"
    item_review_file = DATA_DIR / f"{category}_top_100_user_itemReview.json"
    return user_data_file, item_review_file
