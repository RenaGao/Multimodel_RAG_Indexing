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

QUESTION TYPES (generate a balanced mix):

A. RECOMMENDATION Questions (60-70%):
   ✅ "I'm looking for a {product_type}. What would you recommend?"
   ✅ "Which {product_type} would work best for summer/active wear/casual use?"
   ✅ "What {product_type} options do you have that might suit me?"
   ✅ "Can you suggest a good {product_type} for [scenario]?"

B. FOLLOW-UP Questions (30-40%):
   ✅ "What size should I get?" (about recommended product, not specific named product)
   ✅ "How should I care for it?"
   ✅ "What colors are available?"
   ✅ "Is it suitable for both casual and active wear?"

WRONG Examples (DO NOT GENERATE):
   ❌ "Which size works for the Woolly Clothing Women's Merino..." (mentions product name)
   ❌ "I prefer soft fabrics. What do you recommend?" (states preference)
   ❌ "Is this tank top breathable?" (assumes specific product, not asking for recommendation)

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

COHERENCE REQUIREMENT:
- ALL questions must be about the SAME product type: "{product_type}"
- This is ONE shopping conversation for finding a {product_type}
- Questions should flow naturally: recommendation requests → follow-up details

OUTPUT FORMAT (JSON):
{{
  "latent_preferences": [
    "prefers neutral colors, avoids bright prints",
    "sensitive to itchy fabrics",
    "likes slightly relaxed fit, not very tight",
    "values durability and easy care",
    ...
  ],
  "generated_questions": [
    "I'm looking for a {product_type} for summer. What would you recommend?",        // Recommendation
    "Which {product_type} would work best for active wear?",                        // Recommendation + constraint
    "What colors are available for the ones you'd suggest?",                        // Follow-up
    "What size should I get?",                                                      // Follow-up
    "How should I care for it?",                                                    // Follow-up
    "Are there options that work for both casual and active use?",                 // Recommendation + constraint
    ...
  ]
}}

IMPORTANT REMINDERS:
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
                "generated_questions_v1": old_questions,     # Keep original for reference
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

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)

    print(f"{'='*80}")
    print(f"Regenerating V2 Questions (Decision-Point Approach)")
    print(f"{'='*80}\n")

    main(input_file=args.input)

    print(f"\n{'='*80}")
    print("V2 generation completed successfully!")
    print("New questions use decision-point approach:")
    print("- Questions look generic on surface")
    print("- Create decision points where RAG adds personalization value")
    print("- Include latent preferences for evaluation/debugging")
    print(f"{'='*80}")
