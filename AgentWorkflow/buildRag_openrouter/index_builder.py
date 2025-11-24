"""
Index builder module for constructing RAG indexes.

This module contains the core logic for building:
- Review embeddings for items
- User review embeddings
- Item indexes for categories
- User indexes

Note: This module imports from the original file for now.
Functions will be gradually migrated here for complete independence.
"""

# Import all build functions from the original file
# This provides immediate functionality while we complete the modularization
import sys
from pathlib import Path

# Temporarily import from original file
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import the build functions from the renamed original file
# These are complex functions that will be migrated in Phase 2
from buildRag_openrouter_original import (
    build_review_embeddings,
    build_user_review_embeddings_for_single_user,
    build_all_user_review_embeddings,
    build_item_index_for_category,
    build_user_index_for
)

__all__ = [
    'build_review_embeddings',
    'build_user_review_embeddings_for_single_user',
    'build_all_user_review_embeddings',
    'build_item_index_for_category',
    'build_user_index_for'
]

# TODO Phase 2: Migrate these functions into this file
# The functions are currently imported from the original file to maintain functionality
# Future work: Copy function implementations here and update imports to use local modules
