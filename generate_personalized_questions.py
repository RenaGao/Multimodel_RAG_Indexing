from openai import OpenAI
import json
from tqdm import tqdm
import os
import random
from datetime import datetime
import glob

# -------------------------------------
# 0. Initialize OpenAI client
# -------------------------------------
# TODO: Replace with your actual OpenAI API key
OPENAI_API_KEY = ""

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------
# 1. Load user data
# -------------------------------------
def load_user_data(file_path="All_Beauty_top_100_user_data.json"):
    """
    Load user purchase data from JSON file.
    Data structure: {user_id: [(product_meta, [reviews]), ...]}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# -------------------------------------
# 2. Sampling functions
# -------------------------------------
def select_user_history(user_purchases, num=5):
    """
    Randomly select products from user's purchase history.
    Ensures a mix of high and low ratings.

    Args:
        user_purchases: List of [product_meta, [reviews]] for a user
        num: Number of products to select (default 5)

    Returns:
        List of dicts with product_meta, rating, and review_text
    """
    if len(user_purchases) < num:
        num = len(user_purchases)

    # Separate into high and low rating products
    high_rated = []  # rating >= 4
    low_rated = []   # rating < 4

    for product_meta, reviews in user_purchases:
        if not reviews:
            continue
        # Get the first review's rating
        rating = reviews[0].get("rating", 0)
        review_text = reviews[0].get("text", "")

        item = {
            "product_meta": product_meta,
            "rating": rating,
            "review_text": review_text,
            "review_title": reviews[0].get("title", "")
        }

        if rating >= 4:
            high_rated.append(item)
        else:
            low_rated.append(item)

    # Select a mix: try to get at least 1 low rating if available
    selected = []

    # Determine how many from each category
    if len(low_rated) > 0 and num > 1:
        num_low = min(max(1, num // 3), len(low_rated))  # At least 1 low, roughly 1/3
        num_high = min(num - num_low, len(high_rated))

        selected.extend(random.sample(low_rated, num_low))
        selected.extend(random.sample(high_rated, num_high))
    else:
        # All from available products
        all_products = high_rated + low_rated
        selected = random.sample(all_products, min(num, len(all_products)))

    # Shuffle to mix high and low ratings
    random.shuffle(selected)

    return selected


def select_target_product(user_purchases):
    """
    Select a high-rated product as target for question generation.

    Args:
        user_purchases: List of [product_meta, [reviews]] for a user

    Returns:
        Dict with product_meta, rating, and review_text
    """
    high_rated = []

    for product_meta, reviews in user_purchases:
        if not reviews:
            continue
        rating = reviews[0].get("rating", 0)
        if rating >= 4:
            high_rated.append({
                "product_meta": product_meta,
                "rating": rating,
                "review_text": reviews[0].get("text", ""),
                "review_title": reviews[0].get("title", "")
            })

    if not high_rated:
        # Fallback: return any product
        if user_purchases and user_purchases[0][1]:
            return {
                "product_meta": user_purchases[0][0],
                "rating": user_purchases[0][1][0].get("rating", 0),
                "review_text": user_purchases[0][1][0].get("text", ""),
                "review_title": user_purchases[0][1][0].get("title", "")
            }
        return None

    return random.choice(high_rated)


# -------------------------------------
# 3. Prompt template
# -------------------------------------
def build_prompt(user_history, target_product):
    """
    Build the prompt for generating personalized sequential questions.

    Args:
        user_history: List of 5 products with ratings and reviews
        target_product: The high-rated product to guide question generation

    Returns:
        Formatted prompt string
    """
    # Format user history
    history_text = ""
    for i, item in enumerate(user_history, 1):
        meta = item["product_meta"]
        history_text += f"\n{i}. Product: {meta.get('title', 'Unknown')}\n"
        history_text += f"   Rating: {item['rating']}/5\n"
        history_text += f"   Review: \"{item['review_text']}\"\n"

    # Extract target product category and type
    target_meta = target_product["product_meta"]
    target_title = target_meta.get("title", "Unknown")
    target_category = target_meta.get("main_category", "All Beauty")
    target_details = target_meta.get("details", {})

    prompt = f"""
[System Prompt]
You are a conversational AI that helps generate realistic shopping questions based on a user's purchase history and preferences.

[User Purchase History]
The user has previously purchased and reviewed the following products in the {target_category} domain:
{history_text}

[Task]
Based on this user's purchase history and preferences (inferred from their ratings and reviews), generate a sequence of natural, personalized shopping questions that this user might ask when looking for a new product similar to: "{target_title}".

**Important Guidelines:**
1. **Analyze the user's preferences**: Look at what they liked/disliked in their reviews, what features they mentioned (quality, price, effectiveness, packaging, etc.)
2. **Generate sequential questions**: Create a series of related questions that naturally progress from general to specific, reflecting a real shopping conversation
3. **Personalization**: The questions should reflect the user's past concerns and priorities
4. **Natural flow**: Questions should be conversational and build upon each other (e.g., start with general needs, then narrow down to specific features, price, delivery, etc.)
5. **No fixed count**: Generate as many questions as feel natural for this shopping scenario (typically 3-8 questions)
6. **Don't mention the target product directly**: The user is searching for something similar but doesn't know about this specific product yet

[Output Format]
Return ONLY a JSON array of question strings, like this:
["question 1", "question 2", "question 3", ...]

Example:
["I'm looking for a good moisturizer for dry skin. What do you recommend?", "Does it absorb quickly or leave a greasy feeling?", "What's the price range?", "Are there any options with natural ingredients?"]
"""

    return prompt


# -------------------------------------
# 4. Question generation function
# -------------------------------------
def generate_personalized_questions(user_id, user_history, target_product, model="gpt-4o-mini"):
    """
    Generate personalized sequential questions based on user's purchase history.

    Args:
        user_id: User identifier
        user_history: List of 5 products with ratings and reviews
        target_product: High-rated product to guide generation
        model: OpenAI model to use

    Returns:
        List of generated questions
    """
    prompt = build_prompt(user_history, target_product)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates realistic shopping questions based on user preferences."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=1000,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON array
        if content.startswith("[") and content.endswith("]"):
            questions = json.loads(content)
            return questions
        else:
            # Fallback: try to extract questions from text
            print(f"[Warning] Unexpected format for user {user_id}, attempting fallback parsing")
            # Try to parse as lines
            questions = [line.strip().strip('"\'') for line in content.split("\n") if line.strip()]
            return questions

    except Exception as e:
        print(f"[Error] Failed to generate questions for user '{user_id}': {e}")
        return []


# -------------------------------------
# 5. Helper function to clean product metadata
# -------------------------------------
def clean_product_meta(product_item):
    """
    Remove unnecessary fields (images, videos) from product metadata.

    Args:
        product_item: Dict containing product_meta, rating, review_text

    Returns:
        Cleaned product item
    """
    cleaned_item = product_item.copy()
    if "product_meta" in cleaned_item:
        cleaned_meta = cleaned_item["product_meta"].copy()
        # Remove images and videos
        cleaned_meta.pop("images", None)
        cleaned_meta.pop("videos", None)
        cleaned_item["product_meta"] = cleaned_meta
    return cleaned_item


# -------------------------------------
# 6. Main workflow
# -------------------------------------
def main(data_file="All_Beauty_top_100_user_data.json", max_users=None):
    """
    Main workflow to generate personalized questions for all users.

    Args:
        data_file: Path to user data JSON file
        max_users: Maximum number of users to process (None for all)
    """
    print("Loading user data...")
    user_data = load_user_data(data_file)

    print(f"Loaded {len(user_data)} users")

    # Process users
    all_results = []
    user_ids = list(user_data.keys())

    if max_users:
        user_ids = user_ids[:max_users]

    print(f"\nGenerating personalized questions for {len(user_ids)} users...\n")

    for user_id in tqdm(user_ids, desc="Processing users"):
        user_purchases = user_data[user_id]

        # Skip users with insufficient data
        if len(user_purchases) < 2:
            print(f"[Skip] User {user_id} has insufficient purchase history")
            continue

        try:
            # Sample user history and target product
            user_history = select_user_history(user_purchases, num=5)
            target_product = select_target_product(user_purchases)

            if not target_product:
                print(f"[Skip] User {user_id} has no suitable target product")
                continue

            # Generate questions
            questions = generate_personalized_questions(user_id, user_history, target_product)

            if not questions:
                print(f"[Skip] Failed to generate questions for user {user_id}")
                continue

            # Store result (clean product metadata before saving)
            result = {
                "user_id": user_id,
                "user_history": [clean_product_meta(item) for item in user_history],
                "target_product": clean_product_meta(target_product),
                "generated_questions": questions
            }

            all_results.append(result)

        except Exception as e:
            print(f"[Error] Failed to process user {user_id}: {e}")
            continue

    # -------------------------------------
    # 7. Save results
    # -------------------------------------
    # Extract category from data file name (e.g., "All_Beauty_top_100_user_data.json" -> "All_Beauty")
    category = os.path.basename(data_file).split("_")[0:2]  # Get first two parts
    category = "_".join(category) if len(category) > 1 else category[0]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("questions", exist_ok=True)
    output_path = os.path.join("questions", f"{category}_questions_{ts}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Generation completed! Processed {len(all_results)} users")
    print(f"  Results saved to: {output_path}")


# -------------------------------------
# Entry point
# -------------------------------------
if __name__ == "__main__":
    # Find all data files matching pattern: *_top_100_user_data.json
    data_files = glob.glob("*_top_100_user_data.json")

    if not data_files:
        print("No data files found matching pattern: *_top_100_user_data.json")
        exit(1)

    print(f"Found {len(data_files)} data file(s) to process:")
    for file in data_files:
        print(f"  - {file}")
    print()

    # Process each data file
    for data_file in data_files:
        category = os.path.basename(data_file).split("_")[0:2]
        category = "_".join(category) if len(category) > 1 else category[0]

        print(f"{'='*60}")
        print(f"Processing category: {category}")
        print(f"{'='*60}\n")

        # For testing, process first 10 users per file
        # Remove max_users parameter to process all users
        main(data_file=data_file, max_users=10)
        print()

    print(f"{'='*60}")
    print("All categories processed successfully!")
    print(f"{'='*60}")
