"""
generate_user_profiles.py
-----------------------------------
Generates user profiles from review history using LLM.
Extracts all reviews (excluding target_product) and generates structured user profiles.
"""

import json
import os
import argparse
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# -----------------------------------
# Configuration
# -----------------------------------
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0
MAX_WORKERS = 5
load_dotenv()

# -----------------------------------
# Initialize OpenAI client
# -----------------------------------
api_key = os.getenv("OPENAI_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_KEY in environment variables. Please add it to your .env file.")

client = OpenAI(api_key=api_key)

# -----------------------------------
# Profile Generation Prompt
# -----------------------------------
PROFILE_GENERATION_PROMPT = """
You are an expert at analyzing shopping behavior and extracting user preferences from product reviews.

I will provide you with ALL reviews written by a specific Amazon Fashion customer (excluding one target product).
**IMPORTANT: Each review includes a RATING (1-5 stars). Pay close attention to ratings when analyzing preferences.**

Your task is to analyze these reviews with a focus on RATING-BASED PATTERNS:

### Rating Analysis Instructions:
1. **High Ratings (4-5 stars)**: Extract what the user LIKES and VALUES
   - Positive product features, styles, brands, materials
   - What made them satisfied?

2. **Low Ratings (1-3 stars)**: Extract what the user DISLIKES and AVOIDS
   - Negative experiences, quality issues, sizing problems
   - What disappointed them?

3. **Rating Patterns**: Notice trends
   - Do certain brands/styles consistently get high or low ratings?
   - Are there recurring complaints (e.g., "always too small", "fabric quality")?

### Profile Elements to Extract:
1. **Preferences**: What does this user actively LIKE? (from 4-5 star reviews)
2. **Style**: Fashion style preferences (Casual, Formal, Athletic, etc.)
3. **Budget**: Price sensitivity based on review patterns and ratings
4. **Size**: Common sizes mentioned, fit preferences
5. **Quality Priorities**: What quality aspects matter most?
6. **Dislikes**: What does this user actively AVOID or dislike? (from 1-3 star reviews)
7. **Pain Points**: Recurring issues that cause dissatisfaction

### User's Review History:
{review_corpus}

---

### Output Format (JSON ONLY):
{{
  "Preferences": "What user likes based on high-rated (4-5 star) reviews. Include specific features, styles, materials.",
  "Style": "Fashion style preferences (e.g., Casual, Athletic, Minimalist)",
  "Budget": "Budget-friendly | Mid-range | Premium | Mixed (infer from products and satisfaction levels)",
  "Size": "Size information and fit preferences",
  "Quality_Priorities": "Key quality aspects that matter (e.g., Fabric comfort, Durability, Accurate sizing)",
  "Dislikes": "What user dislikes based on low-rated (1-3 star) reviews. Include materials, styles, brands to avoid.",
  "Pain_Points": "Recurring issues causing dissatisfaction (e.g., sizing inconsistency, fabric pilling, poor stitching)",
  "Additional_Notes": "Other notable patterns or context"
}}

**Instructions:**
- Be SPECIFIC and EVIDENCE-BASED - cite rating levels when relevant
- Distinguish between high-rated and low-rated patterns
- If information is not available, use "Not specified" or "No clear pattern"
- Focus on PATTERNS across multiple reviews, not isolated mentions
- Keep each field concise (1-3 sentences max)
- Use phrases like "consistently rated 5 stars when..." or "gave 1-2 stars due to..."
"""

# -----------------------------------
# Helper Functions
# -----------------------------------

def load_json(file_path):
    """Load JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_reviews_for_user(user_id, target_asin, user_data):
    """
    Extract all reviews for a user from user_data.json,
    excluding reviews for the target_product.

    Args:
        user_id: The user's ID
        target_asin: The parent_asin of the target product to exclude
        user_data: The loaded user_data.json dict

    Returns:
        List of review texts
    """
    if user_id not in user_data:
        return []

    reviews = []
    user_products = user_data[user_id]

    for product_entry in user_products:
        # Each entry is [product_meta, [reviews_array]]
        product_meta = product_entry[0]
        product_reviews = product_entry[1] if len(product_entry) > 1 else []

        # Get the parent_asin of this product
        product_asin = product_meta.get("parent_asin", "")

        # Skip if this is the target product
        if product_asin == target_asin:
            continue

        # Extract all review texts for this product
        for review in product_reviews:
            if isinstance(review, dict):
                review_text = review.get("text", "").strip()
                review_title = review.get("title", "").strip()
                rating = review.get("rating", "N/A")

                if review_text or review_title:
                    # Format: Rating | Title | Review Text
                    formatted_review = f"Rating: {rating}/5"
                    if review_title:
                        formatted_review += f" | Title: {review_title}"
                    if review_text:
                        formatted_review += f" | Review: {review_text}"
                    reviews.append(formatted_review)

    return reviews

def format_review_corpus(reviews):
    """Format reviews into a readable corpus for the LLM."""
    if not reviews:
        return "No reviews available for this user."

    corpus = "\n\n".join([f"Review {i+1}:\n{review}" for i, review in enumerate(reviews)])
    return corpus

def generate_user_profile(user_id, reviews, verbose=False):
    """
    Generate user profile using GPT-4o.

    Args:
        user_id: User identifier
        reviews: List of review texts
        verbose: Print detailed info

    Returns:
        Dict with user profile or error
    """
    if not reviews:
        return {
            "user_id": user_id,
            "profile": {
                "Preferences": "Not specified",
                "Style": "Not specified",
                "Budget": "Not specified",
                "Size": "Not specified",
                "Quality_Priorities": "Not specified",
                "Dislikes": "Not specified",
                "Pain_Points": "Not specified",
                "Additional_Notes": "No review history available"
            },
            "review_count": 0
        }

    review_corpus = format_review_corpus(reviews)

    # Limit corpus size to avoid token limits (keep first ~30 reviews)
    max_reviews = 30
    if len(reviews) > max_reviews:
        truncated_reviews = reviews[:max_reviews]
        review_corpus = format_review_corpus(truncated_reviews)
        review_corpus += f"\n\n[Note: Showing first {max_reviews} of {len(reviews)} total reviews]"

    full_prompt = PROFILE_GENERATION_PROMPT.format(review_corpus=review_corpus)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert shopping behavior analyst."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )

        profile_json = json.loads(response.choices[0].message.content)

        if verbose:
            print(f"\n✓ Generated profile for {user_id}")
            print(f"  Reviews analyzed: {len(reviews)}")
            print(f"  Profile: {json.dumps(profile_json, indent=2)}")

        return {
            "user_id": user_id,
            "profile": profile_json,
            "review_count": len(reviews)
        }

    except Exception as e:
        print(f"\n✗ Error generating profile for {user_id}: {str(e)}")
        return {
            "user_id": user_id,
            "error": str(e),
            "review_count": len(reviews)
        }

def process_single_user(user_entry, user_data, verbose=False):
    """Process a single user from the questions file."""
    user_id = user_entry.get("user_id")
    target_product = user_entry.get("target_product", {})
    target_asin = target_product.get("product_meta", {}).get("parent_asin", "")

    # Extract reviews (excluding target product)
    reviews = extract_reviews_for_user(user_id, target_asin, user_data)

    # Generate profile
    return generate_user_profile(user_id, reviews, verbose=verbose)

def batch_generate_profiles(questions_path, user_data_path, output_path, verbose=False):
    """
    Main function to generate profiles for all users.

    Args:
        questions_path: Path to questions JSON file
        user_data_path: Path to user data JSON file
        output_path: Output path for user profiles
        verbose: Print detailed progress
    """
    print("🔄 Loading data files...")
    questions_data = load_json(questions_path)
    user_data = load_json(user_data_path)

    print(f"✓ Loaded {len(questions_data)} users from questions file")
    print(f"✓ Loaded data for {len(user_data)} users from user data file")

    print(f"\n🚀 Generating user profiles using {MODEL_NAME}...")
    print(f"   Concurrency: {MAX_WORKERS} workers")

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_single_user, user_entry, user_data, verbose)
            for user_entry in questions_data
        ]

        # Process results with progress bar
        with tqdm(total=len(futures), desc="Generating profiles") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    # Convert to user_id -> profile mapping
    user_profiles = {}
    error_count = 0

    for result in results:
        user_id = result.get("user_id")
        if "error" in result:
            error_count += 1
            print(f"⚠️  Skipped {user_id} due to error")
        else:
            user_profiles[user_id] = result.get("profile", {})

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(user_profiles, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Profile generation complete!")
    print(f"   Successfully generated: {len(user_profiles)} profiles")
    print(f"   Errors: {error_count}")
    print(f"   Output saved to: {output_path}")

    # Print sample profile
    if user_profiles:
        sample_user = list(user_profiles.keys())[0]
        print(f"\n📋 Sample profile ({sample_user}):")
        print(json.dumps(user_profiles[sample_user], indent=2))

# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate user profiles from review history using LLM"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="/data/gpfs/projects/punim2620/Multimodel_RAG_Indexing/AmazonReviews/Multimodel_RAG_Indexing/questions_v2/Clothing_Shoes_and_Jewelry_questions_20251127_235753.json",
        help="Path to questions JSON file (e.g., Amazon_Fashion_questions_20251121_222830.json)"
    )
    parser.add_argument(
        "--user-data",
        type=str,
        default="/data/gpfs/projects/punim2620/Multimodel_RAG_Indexing/AmazonReviews/Processed/user_center/Clothing_Shoes_and_Jewelry_top_100_user_data.json",
        help="Path to user data JSON file (e.g., Amazon_Fashion_top_100_user_data.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Clothing_Shoes_and_Jewelry_user_profiles.json",
        help="Output path for generated profiles (default: user_profiles.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress for each user"
    )

    args = parser.parse_args()

    batch_generate_profiles(
        questions_path=args.questions,
        user_data_path=args.user_data,
        output_path=args.output,
        verbose=args.verbose
    )
