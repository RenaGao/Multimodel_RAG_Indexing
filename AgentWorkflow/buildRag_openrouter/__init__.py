"""
buildRag_openrouter package

A modular package for building RAG indexes using OpenRouter API embeddings.

Modular Structure:
- config: Configuration constants and paths
- embedder: Embedder class for generating embeddings
- data_loader: Data loading and JSON utilities
- text_processor: Text construction and processing
- validator: Validation and completeness checking
- document_builder: Document normalization
- index_builder: Index building functions
"""

# Configuration
from .config import (
    BATCH_SIZE,
    CHECKPOINT_INTERVAL,
    BASE_DIR,
    VSTORE_DIR,
    USERS_DIR,
    CATALOG_DIR,
    DATA_DIR,
    get_files_for_category
)

# Embedder
from .embedder import Embedder

# Data loading
from .data_loader import (
    norm,
    load_json,
    calculate_item_review_counts,
    build_item_meta_map,
    load_user_target_products,
    load_questions_data,
    find_questions_file_for_category
)

# Text processing
from .text_processor import construct_item_search_text

# Validation
from .validator import check_category_complete

# Document building
from .document_builder import (
    normalize_item_docs_from_itemReview
)

# Index building
from .index_builder import (
    build_review_embeddings,
    build_user_review_embeddings_for_single_user,
    build_all_user_review_embeddings,
    build_item_index_for_category,
    build_caption_embeddings_for_category
)

__version__ = '1.0.0'

__all__ = [
    # Config
    'BATCH_SIZE',
    'CHECKPOINT_INTERVAL',
    'BASE_DIR',
    'VSTORE_DIR',
    'USERS_DIR',
    'CATALOG_DIR',
    'DATA_DIR',
    'get_files_for_category',
    # Embedder
    'Embedder',
    # Data loader
    'norm',
    'load_json',
    'calculate_item_review_counts',
    'build_item_meta_map',
    'load_user_target_products',
    'load_questions_data',
    'find_questions_file_for_category',
    # Text processor
    'construct_item_search_text',
    # Validator
    'check_category_complete',
    # Document builder
    'normalize_item_docs_from_itemReview',
    # Index builder
    'build_review_embeddings',
    'build_user_review_embeddings_for_single_user',
    'build_all_user_review_embeddings',
    'build_item_index_for_category',
    'build_caption_embeddings_for_category'
]
