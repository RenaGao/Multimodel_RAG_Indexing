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


def normalize_user_profile_for_user(all_users: Any, only_user_id: str) -> List[Dict[str, Any]]:
    """
    Create a user profile document from purchase history and reviews.

    Args:
        all_users: User data with structure {user_id: [[product_meta, reviews], ...]}
        only_user_id: User ID to create profile for

    Returns:
        List containing single user profile document (or empty list if no data)
    """
    docs: List[Dict[str, Any]] = []
    if all_users is None or only_user_id is None:
        return docs

    user_blocks = all_users.get(only_user_id)
    if not user_blocks:
        return docs

    titles = []
    details_texts = []
    review_texts = []

    for block in user_blocks:
        if not isinstance(block, list) or len(block) == 0:
            continue

        meta = block[0] if isinstance(block[0], dict) else None
        reviews = block[1:] if len(block) > 1 else []

        if meta:
            title = meta.get("title") or ""
            titles.append(str(title))
            details = meta.get("details") or {}
            desc = meta.get("description") or []
            features = meta.get("features") or []
            details_texts.append(
                " ".join([
                    norm(title),
                    norm(details),
                    norm(desc),
                    norm(features),
                ])
            )

        for rv in reviews:
            if not isinstance(rv, dict):
                continue
            review_texts.append(norm(rv.get("title")) + " " + norm(rv.get("text")))

    if not (titles or details_texts or review_texts):
        return docs

    full_text = " ".join([
        " ".join(titles),
        " ".join(details_texts),
        " ".join(review_texts),
    ])

    docs.append({
        "doc_id": f"user::{only_user_id}",
        "user_id": only_user_id,
        "title": f"User profile {only_user_id}",
        "text": full_text,
        "source": "user_data",
    })
    return docs
