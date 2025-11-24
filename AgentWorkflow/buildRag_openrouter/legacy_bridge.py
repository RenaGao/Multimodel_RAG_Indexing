"""
Legacy bridge module - imports remaining functions from original buildRag_openrouter.py

This module serves as a temporary bridge to import functions that haven't been
modularized yet. This allows the package to be functional immediately while
providing a clear path for future complete modularization.

TODO: Move these functions to their respective modules:
- validator.py: check_category_complete
- document_builder.py: normalize_item_docs_from_itemReview, normalize_user_profile_for_user
- index_builder.py: build_review_embeddings, build_item_index_for_category, etc.
"""

import sys
from pathlib import Path

# Add parent directory to path to import original file
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from original file
# Note: This imports from the original buildRag_openrouter.py in the parent directory
try:
    from buildRag_openrouter import (
        check_category_complete,
        build_review_embeddings,
        normalize_item_docs_from_itemReview,
        build_user_review_embeddings_for_single_user,
        build_all_user_review_embeddings,
        build_item_index_for_category,
        normalize_user_profile_for_user,
        build_user_index_for
    )

    __all__ = [
        'check_category_complete',
        'build_review_embeddings',
        'normalize_item_docs_from_itemReview',
        'build_user_review_embeddings_for_single_user',
        'build_all_user_review_embeddings',
        'build_item_index_for_category',
        'normalize_user_profile_for_user',
        'build_user_index_for'
    ]
except ImportError as e:
    print(f"[WARN] Could not import from original buildRag_openrouter.py: {e}")
    print("[WARN] Legacy bridge functions not available")
