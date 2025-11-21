from openai import OpenAI
import json
from tqdm import tqdm
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------------------
# 0. Initialize OpenAI client
# -------------------------------------
api_key = os.getenv("OPENAI_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_KEY in environment variables. Please add it to your .env file.")

client = OpenAI(api_key=api_key)


# -------------------------------------
# 1. Load existing questions file
# -------------------------------------
def load_existing_questions(file_path):
    """
    Load the existing questions JSON file.

    Args:
        file_path: Path to the existing questions file

    Returns:
        List of user data with user_history, target_product, and old questions
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# -------------------------------------
# 2. Build new prompt (v2)
# -------------------------------------
def build_v2_prompt(user_history, target_product):
    """
    Build the v2 prompt for generating questions that reference user's own reviews.

    Args:
        user_history: List of 5 products with ratings and reviews
        target_product: The target product (for context, not directly mentioned)

    Returns:
        Formatted prompt string
    """
    # Format user history with detailed review information
    history_text = ""
    for i, item in enumerate(user_history, 1):
        meta = item["product_meta"]
        history_text += f"\n{i}. Product: {meta.get('title', 'Unknown')}\n"
        history_text += f"   Rating: {item['rating']}/5\n"
        history_text += f"   Review: \"{item['review_text']}\"\n"
        if item.get('review_title'):
            history_text += f"   Review Title: \"{item['review_title']}\"\n"

    # Extract target category for context
    target_meta = target_product["product_meta"]
    target_category = target_meta.get("main_category", "shopping")

    prompt = f"""
[System Prompt]
You are generating highly personalized shopping questions where the user EXPLICITLY REFERENCES their own purchase history and reviews.

[User's Previous Reviews]
The user has the following purchase and review history:
{history_text}

[Task]
Generate a sequence of natural, conversational questions where the user asks for product recommendations in the {target_category} category, BUT the questions MUST explicitly reference their own purchase history and preferences.

**Critical Requirements:**

1. **Reference Past Reviews**: Questions should mention specific things the user liked/disliked in their reviews
   - Example: "Based on my previous reviews where I mentioned..."
   - Example: "Considering that I complained about X in my last purchase..."

2. **Extract Preferences**: Analyze the reviews above and have the user reference:
   - Specific features they valued (quality, price, effectiveness, packaging, etc.)
   - Problems they experienced (sizing issues, material quality, delivery, etc.)
   - Brands or styles they preferred

3. **Require RAG**: Questions CANNOT be answered just from product search alone
   - Must retrieve and analyze user's review history to understand preferences
   - Agent needs to combine: user's query + user's historical reviews + product database

4. **Natural Progression**: Generate 5-8 questions that build on each other:
   - Start general: "Based on my purchase history, recommend something..."
   - Get specific: "Considering I mentioned X in my reviews..."
   - Follow-up: "Given my preference for Y, what about..."

5. **Conversational Style**: Make it sound natural, like a real shopping conversation
   - Use "I", "my", "I've mentioned", "I tend to prefer"
   - Reference specific details from the reviews above

**What NOT to do:**
- Don't generate generic questions like "I'm looking for a dress"
- Don't ignore the user's review history
- Don't make questions answerable without checking user reviews

[Output Format]
Return ONLY a JSON array of question strings:
["question 1", "question 2", "question 3", ...]

Example format:
["Based on my previous purchases where I mentioned preferring natural ingredients, what beauty products would you recommend?", "Looking at my review history, I tend to value quick absorption and non-greasy formulas. Any suggestions?", "Considering the sizing issues I mentioned in my past orders, what should I look for?"]
"""

    return prompt


# -------------------------------------
# 3. Generate v2 questions
# -------------------------------------
def generate_v2_questions(user_id, user_history, target_product, model="gpt-4o-mini"):
    """
    Generate v2 personalized questions that reference user's own reviews.

    Args:
        user_id: User identifier
        user_history: List of 5 products with ratings and reviews
        target_product: Target product for context
        model: OpenAI model to use

    Returns:
        List of generated questions
    """
    prompt = build_v2_prompt(user_history, target_product)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates highly personalized shopping questions based on user's review history."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=1200,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON array
        if content.startswith("[") and content.endswith("]"):
            questions = json.loads(content)
            return questions
        else:
            # Fallback: try to extract questions from text
            print(f"[Warning] Unexpected format for user {user_id}, attempting fallback parsing")
            questions = [line.strip().strip('"\'') for line in content.split("\n") if line.strip()]
            return questions

    except Exception as e:
        print(f"[Error] Failed to generate v2 questions for user '{user_id}': {e}")
        return []


# -------------------------------------
# 4. Main workflow
# -------------------------------------
def main(input_file="questions/Amazon_Fashion_questions_20251110_214618.json"):
    """
    Main workflow to regenerate v2 questions from existing questions file.

    Args:
        input_file: Path to existing questions JSON file
    """
    print(f"Loading existing questions from: {input_file}")
    existing_data = load_existing_questions(input_file)

    print(f"Loaded {len(existing_data)} users")

    # Process users
    all_results = []

    print(f"\nRegenerating v2 questions for {len(existing_data)} users...\n")

    for user_data in tqdm(existing_data, desc="Processing users"):
        user_id = user_data.get("user_id")
        user_history = user_data.get("user_history", [])
        target_product = user_data.get("target_product")
        old_questions = user_data.get("generated_questions", [])

        if not user_history or not target_product:
            print(f"[Skip] User {user_id} missing required data")
            continue

        try:
            # Generate v2 questions
            v2_questions = generate_v2_questions(user_id, user_history, target_product)

            if not v2_questions:
                print(f"[Skip] Failed to generate v2 questions for user {user_id}")
                continue

            # Store result with both old and new questions for comparison
            result = {
                "user_id": user_id,
                "user_history": user_history,
                "target_product": target_product,
                "generated_questions_v1": old_questions,  # Keep original for reference
                "generated_questions": v2_questions        # New v2 questions
            }

            all_results.append(result)

        except Exception as e:
            print(f"[Error] Failed to process user {user_id}: {e}")
            continue

    # -------------------------------------
    # 5. Save results
    # -------------------------------------
    # Extract category from input file name
    filename = os.path.basename(input_file)
    category = filename.split("_questions_")[0]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("questions", exist_ok=True)
    output_path = os.path.join("questions", f"{category}_questions_v2_{ts}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ V2 generation completed! Processed {len(all_results)} users")
    print(f"  Results saved to: {output_path}")

    # Print sample questions for comparison
    if all_results:
        sample_user = all_results[0]
        print(f"\n{'='*80}")
        print(f"Sample User: {sample_user['user_id']}")
        print(f"{'='*80}")
        print("\nV1 Questions (original):")
        for i, q in enumerate(sample_user.get('generated_questions_v1', [])[:3], 1):
            print(f"  {i}. {q}")
        print("\nV2 Questions (new - references user history):")
        for i, q in enumerate(sample_user.get('generated_questions', [])[:3], 1):
            print(f"  {i}. {q}")
        print(f"{'='*80}\n")


# -------------------------------------
# Entry point
# -------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate v2 personalized questions from existing questions file")
    parser.add_argument(
        "--input",
        type=str,
        default="questions/Amazon_Fashion_questions_20251110_214618.json",
        help="Path to existing questions JSON file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)

    print(f"{'='*80}")
    print(f"Regenerating V2 Questions (with user review references)")
    print(f"{'='*80}\n")

    main(input_file=args.input)

    print(f"\n{'='*80}")
    print("V2 generation completed successfully!")
    print("New questions explicitly reference user's review history for RAG testing")
    print(f"{'='*80}")
