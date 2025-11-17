import json

def analyze_reviews(json_file_path):
    """
    Analyze Amazon Fashion reviews and calculate average reviews per item.

    Args:
        json_file_path: Path to the JSON file containing item reviews
    """
    print(f"Loading data from {json_file_path}...")

    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Calculate statistics
    num_items = len(data)
    total_reviews = sum(len(reviews) for reviews in data.values())

    # Calculate average
    if num_items > 0:
        avg_reviews_per_item = total_reviews / num_items
    else:
        avg_reviews_per_item = 0

    # Display results
    print("\n" + "="*50)
    print("Amazon Fashion Review Analysis")
    print("="*50)
    print(f"Total number of items: {num_items:,}")
    print(f"Total number of reviews: {total_reviews:,}")
    print(f"Average reviews per item: {avg_reviews_per_item:.2f}")
    print("="*50)

    # Additional statistics
    review_counts = [len(reviews) for reviews in data.values()]
    max_reviews = max(review_counts) if review_counts else 0
    min_reviews = min(review_counts) if review_counts else 0

    print(f"\nAdditional Statistics:")
    print(f"  Maximum reviews for a single item: {max_reviews:,}")
    print(f"  Minimum reviews for a single item: {min_reviews:,}")

    # Find item with most reviews
    if review_counts:
        max_item_id = max(data.items(), key=lambda x: len(x[1]))[0]
        print(f"  Item with most reviews: {max_item_id} ({max_reviews:,} reviews)")

    return {
        'num_items': num_items,
        'total_reviews': total_reviews,
        'avg_reviews_per_item': avg_reviews_per_item,
        'max_reviews': max_reviews,
        'min_reviews': min_reviews
    }

if __name__ == "__main__":
    json_file = "Amazon_Fashion_top_100_user_itemReview.json"
    results = analyze_reviews(json_file)
