"""
Text processing module for RAG index building.

This module provides text manipulation and construction utilities for:
- Constructing searchable text from product metadata and reviews
- Future text processing functions
"""

from typing import Any, List


def construct_item_search_text(
    product_meta: Any,
    all_item_reviews: List[dict],
    max_reviews: int = 20
) -> str:
    """
    Construct text for embedding from product metadata and reviews.

    Only uses string concatenation of original fields, does not call LLM.

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
