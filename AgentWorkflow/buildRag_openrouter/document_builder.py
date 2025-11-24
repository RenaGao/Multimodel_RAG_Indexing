"""
Document builder module for normalizing raw data into document structures.

This module transforms raw data into normalized document formats for:
- Item documents (from product metadata)
- User profile documents (from purchase history and reviews)
"""

from typing import Any, Dict, List
from tqdm import tqdm

from .data_loader import norm
from .text_processor import construct_item_search_text


def normalize_item_docs_from_itemReview(
    item_review_data: dict,
    item_meta_map: Dict[str, dict]
) -> List[Dict[str, Any]]:
    """
    Create one document per item using only product metadata (no reviews).
    Uses item_review_data as primary source and item_meta_map for metadata.

    Args:
        item_review_data: dict with structure {parent_asin: [review_list]}
        item_meta_map: dict with structure {parent_asin: product_meta}

    Returns:
        List of document dicts
    """
    docs: List[Dict[str, Any]] = []

    for parent_asin, reviews in tqdm(item_review_data.items(), desc="Processing items"):
        if not isinstance(reviews, list) or len(reviews) == 0:
            continue

        # Get product metadata (may be None if not in user_data)
        product_meta = item_meta_map.get(parent_asin)

        # Use construct_item_search_text to create structured text (max_reviews=0 means no reviews)
        full_text = construct_item_search_text(product_meta, reviews, max_reviews=0)

        # Extract metadata
        product_title = ""
        brand = ""
        price = ""
        avg_rating = None

        if product_meta:
            product_title = product_meta.get("title", "") or ""
            store = product_meta.get("store", "") or ""
            details = product_meta.get("details") or {}
            brand = details.get("Brand") or product_meta.get("brand") or store
            price = product_meta.get("price")
            # Get average rating from product metadata
            avg_rating = product_meta.get("average_rating")

        doc_id = f"item::{parent_asin}"

        docs.append({
            "doc_id": doc_id,
            "item_id": parent_asin,
            "title": norm(product_title),
            "brand": norm(brand),
            "price": norm(price),
            "rating": avg_rating,
            "num_reviews": len(reviews),
            "text": full_text,
        })

    return docs
