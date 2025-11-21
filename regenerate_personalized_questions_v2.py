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
[CRITICAL INSTRUCTION]
Generate questions where the user ONLY MENTIONS they have purchase/review history, but DOES NOT explicitly state their preferences in the question itself.

The agent MUST use RAG (Retrieval-Augmented Generation) to retrieve the user's reviews and infer their preferences. If preferences are stated in the question, RAG becomes unnecessary!

[User's Previous Reviews - FOR YOUR ANALYSIS ONLY]
The user has the following purchase and review history (DO NOT include these details in the questions):
{history_text}

[WRONG Examples - DO NOT GENERATE THESE]
❌ "Based on my previous purchases where I prefer soft fabrics and comfortable fits, what would you recommend?"
   → BAD: Preferences (soft, comfortable) are stated in the question
❌ "I've noticed I really enjoy X and Y. Can you recommend?"
   → BAD: User explicitly says what they enjoy
❌ "Looking at my reviews, I tend to value A and B. Any suggestions?"
   → BAD: Specific values (A, B) are mentioned
❌ "Considering the sizing issues I mentioned, what should I look for?"
   → BAD: Explicitly states the problem (sizing)

[CORRECT Examples - GENERATE THESE TYPES]
✅ "Based on my purchase history, what would you recommend for me?"
   → GOOD: References history but doesn't state preferences
✅ "Looking at my previous orders, what's suitable for me?"
   → GOOD: Agent must RAG to discover what's suitable
✅ "Can you suggest something that matches my preferences?"
   → GOOD: Says "preferences" exist but doesn't reveal them
✅ "What should I consider when choosing a new product, given my past purchases?"
   → GOOD: Asks for guidance based on history without specifics
✅ "Are there options that fit my style?"
   → GOOD: References "style" but doesn't describe it
✅ "What would work well for me based on what I've bought before?"
   → GOOD: Requires RAG to understand "what works"

[Task]
Generate 5-8 questions in the {target_category} category where:

1. **Reference History**: User mentions "my purchase history", "my reviews", "my past orders", "what I've bought before"

2. **NO Explicit Preferences**: User does NOT state what they liked/disliked
   - Don't mention: soft fabrics, comfortable, sizing issues, specific features, brands, etc.
   - Agent MUST retrieve reviews to discover these

3. **Test RAG Requirement**: Questions should be:
   - WITHOUT RAG: Agent can only give generic recommendations
   - WITH RAG: Agent retrieves reviews, understands preferences, gives personalized recommendations

4. **Natural Progression**: Build questions sequentially:
   - Round 1: "Based on my history, what do you recommend?"
   - Round 2: "Are there options that match my preferences?" (still vague)
   - Round 3: "What about sizing/quality/price? What suits me?"
   - Round 4: "Any alternatives that would work for me?"

5. **Conversational**: Sound natural, use "I", "my", "me"

[Output Format]
Return ONLY a JSON array of question strings:
["question 1", "question 2", ...]

Remember: Questions should reference history but NOT reveal specific preferences. The agent must use RAG to figure out what the user likes!
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

        # Remove markdown code blocks if present
        if content.startswith("```"):
            # Remove opening ```json or ```
            content = content.lstrip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()
            # Remove closing ```
            if content.endswith("```"):
                content = content[:-3].strip()

        # Parse JSON array
        if content.startswith("[") and content.endswith("]"):
            questions = json.loads(content)
            return questions
        else:
            # Fallback: try to extract questions from text
            print(f"[Warning] Unexpected format for user {user_id}, attempting fallback parsing")
            # Try to find JSON array in text
            start_idx = content.find("[")
            end_idx = content.rfind("]")
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                try:
                    json_str = content[start_idx:end_idx+1]
                    questions = json.loads(json_str)
                    return questions
                except:
                    pass
            # Final fallback: split by lines
            questions = [line.strip().strip('"\'') for line in content.split("\n") if line.strip() and not line.startswith("[") and not line.startswith("]")]
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
