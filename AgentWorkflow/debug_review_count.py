"""
Debug script to investigate review count issue for item B075FPPBH9
"""
import json
from pathlib import Path

# Test item
TEST_ASIN = "B075FPPBH9"

# File paths
item_review_file = Path("/data/gpfs/projects/punim2620/Multimodel_RAG_Indexing/AmazonReviews/Processed/user_center/Amazon_Fashion_top_100_user_itemReview.json")
user_data_file = Path("/data/gpfs/projects/punim2620/Multimodel_RAG_Indexing/AmazonReviews/Processed/user_center/Amazon_Fashion_top_100_user_data.json")

print(f"="*60)
print(f"Investigating review count for item: {TEST_ASIN}")
print(f"="*60)

# Load item_review_file
print(f"\n1. Checking item_review_file...")
with open(item_review_file, 'r', encoding='utf-8') as f:
    item_review_data = json.load(f)

if TEST_ASIN in item_review_data:
    reviews = item_review_data[TEST_ASIN]
    print(f"   [OK] Found {TEST_ASIN} in item_review_file")
    print(f"   Type: {type(reviews)}")
    print(f"   Length: {len(reviews)}")
    print(f"   Number of reviews: {len(reviews)}")

    for i, review in enumerate(reviews, 1):
        print(f"\n   Review {i}:")
        print(f"     - Title: {review.get('title', 'N/A')}")
        print(f"     - Rating: {review.get('rating', 'N/A')}")
        print(f"     - ASIN: {review.get('asin', 'N/A')}")
        print(f"     - Parent ASIN: {review.get('parent_asin', 'N/A')}")
else:
    print(f"   [ERROR] {TEST_ASIN} NOT found in item_review_file")

# Load user_data_file
print(f"\n2. Checking user_data_file...")
with open(user_data_file, 'r', encoding='utf-8') as f:
    user_data = json.load(f)

count_in_user_data = 0
for user_id, purchases in user_data.items():
    for block in purchases:
        if isinstance(block, list) and len(block) > 0:
            meta = block[0]
            if isinstance(meta, dict):
                parent_asin = meta.get('parent_asin') or meta.get('asin')
                if parent_asin == TEST_ASIN:
                    reviews_part = block[1:]
                    if len(reviews_part) == 1 and isinstance(reviews_part[0], list):
                        reviews = reviews_part[0]
                    else:
                        reviews = reviews_part

                    count_in_user_data += len(reviews)
                    print(f"   Found in user {user_id}: {len(reviews)} reviews")

print(f"\n   Total reviews in user_data for {TEST_ASIN}: {count_in_user_data}")

# Summary
print(f"\n" + "="*60)
print(f"SUMMARY")
print(f"="*60)
print(f"Reviews in item_review_file: {len(item_review_data.get(TEST_ASIN, []))}")
print(f"Reviews in user_data_file: {count_in_user_data}")
print(f"="*60)
