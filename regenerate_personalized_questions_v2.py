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
# 2. Build new prompt (v2) - Decision Point Approach
# -------------------------------------
def build_v2_prompt(user_history, target_product):
    """
    Build the v2 prompt using decision-point approach.
    Questions look generic but create scenarios where RAG adds value.

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

    # Extract target product information
    target_meta = target_product["product_meta"]
    target_title = target_meta.get("title", "product")
    target_category = target_meta.get("main_category", "shopping")

    # Extract generic product type (e.g., "tank top" not full brand/model name)
    target_words = target_title.lower().split()
    product_keywords = {
        "top": "top", "tank": "tank top", "dress": "dress", "shirt": "shirt",
        "hoodie": "hoodie", "sweater": "sweater", "pants": "pants", "jeans": "jeans",
        "shorts": "shorts", "skirt": "skirt", "jacket": "jacket", "coat": "coat",
        "shoes": "shoes", "boots": "boots", "sandals": "sandals", "sneakers": "sneakers",
        "bag": "bag", "purse": "purse", "wallet": "wallet", "belt": "belt",
        "hat": "hat", "scarf": "scarf", "gloves": "gloves", "socks": "socks"
    }

    # Find product type from title
    product_type = "product"  # default
    for word in target_words:
        if word in product_keywords:
            product_type = product_keywords[word]
            break

    # Special case: if "tank" and "top" both present, use "tank top"
    if "tank" in target_words and "top" in target_words:
        product_type = "tank top"

    # Extract product metadata for decision points
    target_details = target_meta.get("details", {})
    available_colors = target_meta.get("colors", [])
    available_sizes = target_meta.get("sizes", [])

    # Format metadata text
    metadata_text = f"Title: {target_title}\n"
    metadata_text += f"Category: {target_category}\n"
    if available_colors:
        metadata_text += f"Available Colors: {', '.join(available_colors)}\n"
    if available_sizes:
        metadata_text += f"Available Sizes: {', '.join(available_sizes)}\n"
    if target_details:
        metadata_text += "Details:\n"
        for key, value in target_details.items():
            metadata_text += f"  - {key}: {value}\n"

    prompt = f"""You are a question generator for a personalized shopping assistant.

INPUTS:
- Target product metadata (category, material, fit, colors, etc.).
- 3–5 of the user's past product reviews (with metadata of those items).

TARGET PRODUCT METADATA:
Product Type: {product_type}
Category: {target_category}
{metadata_text}

USER'S PAST REVIEWS:
{history_text}

SCENARIO:
This is a product RECOMMENDATION conversation. The user is shopping and wants the agent to recommend products.
- User does NOT know about the target product yet - they're asking for recommendations
- Agent should use RAG to retrieve user's review history and make personalized recommendations
- Questions should be a mix of recommendation requests and follow-up detail questions

GOAL:
Generate user-side shopping questions that:
- Look natural and relatively generic on the surface
- Do NOT explicitly state the user's detailed preferences
- Create decision points where RAG of user reviews enables personalized answers
- Frame as recommendation requests, not product review questions

CRITICAL CONSTRAINTS:
- DO NOT mention the specific product name/title (e.g., "Woolly Clothing Women's Merino...")
- Use generic product type instead: "{product_type}"
- Frame questions as recommendation requests ("what would you recommend?", "which one should I get?")
- Mix recommendation questions (60-70%) with follow-up detail questions (30-40%)

RAG VALUE TEST (CRITICAL):
For EACH question you generate, ask yourself:
- WITHOUT RAG: What would agent answer? (generic catalog response)
- WITH RAG: What would agent answer? (personalized based on user's reviews)
- If the answers would be the SAME → BAD question, revise it!

Examples of BAD vs GOOD questions:

BAD: "Are there soft and breathable {product_type}s?"
   - Without RAG: "Yes, we have these soft breathable options..." (lists products)
   - With RAG: "Yes, we have these soft breathable options..." (SAME answer)
   - RAG adds NO value

GOOD: "Which material would work best for me?"
   - Without RAG: "We have cotton, merino wool, polyester..." (generic list)
   - With RAG: "Based on your reviews, you're sensitive to itchy fabrics and prefer soft cotton. I recommend cotton or merino."
   - RAG enables PERSONALIZATION

BAD: "What colors are available?"
   - Without RAG: "Black, white, red, blue..." (catalog listing)
   - With RAG: "Black, white, red, blue..." (SAME answer)
   - RAG adds NO value

GOOD: "Which color should I choose?"
   - Without RAG: "We have black, white, red, blue. What's your preference?"
   - With RAG: "You typically choose neutral colors and avoid bright prints. I recommend black or white."
   - RAG enables PERSONALIZATION

BAD: "Can you suggest a {product_type} for layering?"
   - Without RAG: "These {product_type}s work well for layering..." (generic)
   - With RAG: "These {product_type}s work well for layering..." (SAME answer)
   - User already stated use case, RAG not needed

GOOD: "Is this suitable for how I typically use {product_type}s?"
   - Without RAG: "It works for various uses..." (generic)
   - With RAG: "Based on your reviews, you use {product_type}s for both workouts and casual wear. This one is perfect for both."
   - RAG discovers user's use case

QUESTION TYPES (generate a balanced mix - ALL must pass RAG VALUE TEST):

A. RECOMMENDATION Questions (60-70%) - MUST create RAG decision points:
   GOOD: "I'm looking for a {product_type}. What would you recommend?"
      → RAG: Recommends based on user's material/fit/color/use preferences from reviews

   GOOD: "Which {product_type} would work best for me?"
      → RAG: Analyzes user's typical use cases and preferences to personalize

   GOOD: "What style of {product_type} suits me?"
      → RAG: Checks user's past style preferences (relaxed vs fitted, casual vs sporty)

   DON'T: "Which {product_type} is best for summer?" (generic, no RAG needed)
   DON'T: "Are there breathable {product_type}s?" (user states need, just filter catalog)

B. FOLLOW-UP Questions (30-40%) - MUST require RAG to personalize:
   GOOD: "Which color should I choose?"
      → RAG: Knows user prefers neutral vs bright, solid vs prints

   GOOD: "What size would work for me?"
      → RAG: Knows user's sizing history and fit preferences (tight vs loose)

   GOOD: "Which material would be best?"
      → RAG: Knows user's material sensitivities (itchy wool, soft cotton, etc.)

   GOOD: "Is this suitable for how I typically use {product_type}s?"
      → RAG: Knows user's use cases from reviews (workout, casual, layering, etc.)

   DON'T: "What colors are available?" (catalog info, no personalization)
   DON'T: "How should I wash it?" (generic care info, no RAG needed)

WRONG Examples (DO NOT GENERATE):
   BAD: "Which size works for the Woolly Clothing Women's Merino..." (mentions product name)
   BAD: "I prefer soft fabrics. What do you recommend?" (states preference explicitly)
   BAD: "Are there {product_type}s that are soft and breathable?" (states needs, no RAG value)
   BAD: "What colors are available?" (generic catalog, RAG doesn't help)

PROCESS (INTERNALLY, DO NOT OUTPUT THESE STEPS):
1. From the user's past reviews, infer their preferences and sensitivities
   (e.g., typical colors, fit, comfort, durability, washing issues, use cases).
2. Identify 5–10 "decision points" where these preferences matter:
   - e.g., color choice, size choice, fit (tight vs relaxed), warmth vs breathability,
     durability vs softness, work vs casual use, etc.
3. For each decision point, design questions that:
   - Start with recommendation requests (60-70%)
   - Follow with detail questions about recommendations (30-40%)
   - Do NOT mention specific product names
   - Do NOT state user's preferences explicitly

QUESTION QUALITY CHECKLIST:
Before finalizing each question, verify it passes ALL these tests:

1. Does this question state what the user wants? (soft, breathable, for layering, etc.)
   → If YES, REVISE to be open-ended ("Which material?" not "Are there soft ones?")

2. Can agent answer this with just product catalog, without user history?
   → If YES, REVISE to require user preference knowledge

3. Would the answer be DIFFERENT if agent knew user's review history?
   → If NO, REVISE to create a real decision point

4. Does question ask about user's needs/preferences? (for me, suits me, typically use)
   → If NO, ADD reference to make it personalized

Good patterns:
   "Which X?" (forces choice based on preferences)
   "What X suits me?" (requires knowing user)
   "Is this suitable for how I use X?" (requires knowing use cases)

Bad patterns:
   "Are there X that are Y?" (just filters catalog)
   "What X options exist?" (just lists catalog)
   "I need X for Y, what do you recommend?" (states need)

COHERENCE REQUIREMENT:
- ALL questions must be about the SAME product type: "{product_type}"
- This is ONE shopping conversation for finding a {product_type}
- Questions should flow naturally: recommendation requests → follow-up details

OUTPUT FORMAT (JSON):
{{
  "latent_preferences": [
    "prefers neutral colors (black, gray, white), avoids bright prints",
    "sensitive to itchy fabrics, prefers soft cotton or merino wool",
    "likes relaxed fit, not tight or clingy",
    "uses {product_type}s for both workouts and casual wear",
    "concerned about durability and easy washing"
  ],
  "generated_questions": [
    "I'm looking for a {product_type}. What would you recommend?",
    "Which material would work best for me?",
    "What style would suit me - fitted or relaxed?",
    "Which color should I choose?",
    "What size would work for me?",
    "Is this suitable for how I typically use {product_type}s?",
    "How does the fit run - should I size up or down for my preference?"
  ]
}}

EXPLANATION - How each question demonstrates RAG value:
- Q1: Needs ALL user preferences to make personalized recommendation
- Q2: RAG discovers user is sensitive to itchy fabrics, prefers soft materials
- Q3: RAG discovers user prefers relaxed over fitted styles
- Q4: RAG discovers user prefers neutral colors, avoids bright prints
- Q5: RAG discovers user's sizing history and preferences
- Q6: RAG discovers user's use cases (workout vs casual vs both)
- Q7: RAG discovers user's fit preferences (tight vs loose)

IMPORTANT REMINDERS:
- Output ONLY valid JSON - NO comments like // or /* */ in the JSON output
- The `latent_preferences` are only for debugging and evaluation - they must NOT appear in questions
- 60-70% of questions should be RECOMMENDATION requests
- 30-40% of questions should be FOLLOW-UP details
- DO NOT mention specific product names - use generic "{product_type}" only
- Questions should flow naturally from asking for recommendations to asking about details

Generate 5-8 questions that cover different decision points (use case, color, size, fit, material, care, etc.).
"""

    return prompt


# -------------------------------------
# 3. Generate v2 questions
# -------------------------------------
def generate_v2_questions(user_id, user_history, target_product, model="gpt-4o-mini"):
    """
    Generate v2 questions using decision-point approach.
    Returns both latent preferences and generated questions.

    Args:
        user_id: User identifier
        user_history: List of 5 products with ratings and reviews
        target_product: Target product for context
        model: OpenAI model to use

    Returns:
        Tuple of (latent_preferences, questions) or ([], []) on failure
    """
    prompt = build_v2_prompt(user_history, target_product)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates decision-point based shopping questions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=1500,
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

        # Parse JSON object with latent_preferences and generated_questions
        if content.startswith("{") and content.endswith("}"):
            data = json.loads(content)
            latent_prefs = data.get("latent_preferences", [])
            questions = data.get("generated_questions", [])
            return latent_prefs, questions
        else:
            # Fallback: try to find JSON object in text
            print(f"[Warning] Unexpected format for user {user_id}, attempting fallback parsing")
            # Try to find JSON object in text
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                try:
                    json_str = content[start_idx:end_idx+1]
                    data = json.loads(json_str)
                    latent_prefs = data.get("latent_preferences", [])
                    questions = data.get("generated_questions", [])
                    return latent_prefs, questions
                except Exception as fallback_error:
                    print(f"[Warning] Fallback parsing failed: {fallback_error}")

            # Final fallback: return empty
            print(f"[Error] Could not parse response for user {user_id}")
            return [], []

    except Exception as e:
        print(f"[Error] Failed to generate v2 questions for user '{user_id}': {e}")
        return [], []


# -------------------------------------
# 4. Helper functions for batch processing
# -------------------------------------
def get_completed_categories(questions_v2_dir="questions_v2"):
    """Get list of already completed categories from questions_v2 folder."""
    completed = set()
    if os.path.exists(questions_v2_dir):
        for f in os.listdir(questions_v2_dir):
            if f.endswith(".json"):
                # Extract category from filename, e.g., "Amazon_Fashion_questions_xxx.json" -> "Amazon_Fashion"
                category = f.split("_questions_")[0]
                completed.add(category)
    return completed


def get_all_input_files(questions_dir="questions"):
    """Get all input files from questions folder."""
    files = []
    if os.path.exists(questions_dir):
        for f in os.listdir(questions_dir):
            if f.endswith(".json") and "_questions_" in f:
                files.append(os.path.join(questions_dir, f))
    return sorted(files)


# -------------------------------------
# 5. Main workflow
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
            # Generate v2 questions with decision-point approach
            latent_prefs, v2_questions = generate_v2_questions(user_id, user_history, target_product)

            if not v2_questions:
                print(f"[Skip] Failed to generate v2 questions for user {user_id}")
                continue

            # Store result with latent preferences and questions
            result = {
                "user_id": user_id,
                "user_history": user_history,
                "target_product": target_product,
                "latent_preferences": latent_prefs,          # Inferred preferences (for debugging)
                "generated_questions": v2_questions          # New v2 decision-point questions
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
    os.makedirs("questions_v2", exist_ok=True)
    output_path = os.path.join("questions_v2", f"{category}_questions_{ts}.json")

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

        print("\n" + "-"*80)
        print("V2 Approach: Decision-Point Questions")
        print("-"*80)

        # Show latent preferences (inferred from reviews)
        latent_prefs = sample_user.get('latent_preferences', [])
        if latent_prefs:
            print("\nLatent Preferences (inferred from user's reviews):")
            for i, pref in enumerate(latent_prefs, 1):
                print(f"  {i}. {pref}")

        # Show generated questions
        print("\nGenerated Questions (decision points - generic on surface):")
        for i, q in enumerate(sample_user.get('generated_questions', [])[:5], 1):
            print(f"  {i}. {q}")

        print("\n[Note] Questions look generic but create decision points where")
        print("       RAG retrieval of user reviews enables personalized answers.")
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
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all categories in questions/ folder, skip already completed ones in questions_v2/"
    )

    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"Regenerating V2 Questions (Decision-Point Approach)")
    print(f"{'='*80}\n")

    if args.all:
        # Batch processing mode
        completed = get_completed_categories()
        all_files = get_all_input_files()

        print(f"Found {len(all_files)} categories in questions/")
        print(f"Already completed: {len(completed)} categories in questions_v2/")
        if completed:
            print(f"  Completed: {', '.join(sorted(completed))}")
        print()

        to_process = []
        for input_file in all_files:
            category = os.path.basename(input_file).split("_questions_")[0]
            if category in completed:
                print(f"[SKIP] {category} - already completed")
            else:
                to_process.append((category, input_file))

        print(f"\nWill process {len(to_process)} categories\n")

        for i, (category, input_file) in enumerate(to_process, 1):
            print(f"\n{'='*80}")
            print(f"[{i}/{len(to_process)}] Processing: {category}")
            print(f"{'='*80}")
            main(input_file=input_file)

    else:
        # Single file processing mode
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            exit(1)

        main(input_file=args.input)

    print(f"\n{'='*80}")
    print("V2 generation completed successfully!")
    print("New questions use decision-point approach:")
    print("- Questions look generic on surface")
    print("- Create decision points where RAG adds personalization value")
    print("- Include latent preferences for evaluation/debugging")
    print(f"{'='*80}")
