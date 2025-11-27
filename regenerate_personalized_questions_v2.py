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

    # Extract product metadata
    target_details = target_meta.get("details", {})
    target_features = target_meta.get("features", [])
    target_description = target_meta.get("description", [])

    # Format metadata text - include ALL available metadata
    metadata_text = f"Title: {target_title}\n"
    metadata_text += f"Category: {target_category}\n"

    if target_features:
        metadata_text += "Features:\n"
        for feat in target_features[:10]:  # Limit to first 10 features
            if feat:
                metadata_text += f"  - {feat}\n"

    if target_description:
        desc_text = " ".join([str(d) for d in target_description if d])[:500]  # Limit length
        if desc_text:
            metadata_text += f"Description: {desc_text}\n"

    if target_details:
        metadata_text += "Details:\n"
        for key, value in target_details.items():
            metadata_text += f"  - {key}: {value}\n"

    prompt = f"""You are a question generator for a personalized shopping assistant.

INPUTS:
- Target product metadata (all available attributes).
- 3–5 of the user's past product reviews (with metadata of those items).

TARGET PRODUCT METADATA:
{metadata_text}

USER'S PAST REVIEWS:
{history_text}

CATEGORY & ATTRIBUTE AWARENESS (IMPORTANT):

You MUST determine:
- what kind of product this is,
- which attributes matter for evaluating or choosing it,
ONLY from the target product's metadata provided above.

Do NOT use any assumptions about product types such as clothing, books, electronics, or shoes.
Do NOT rely on any templates, stereotypes, or memorized category schemas.
Everything must be derived from THIS target product's metadata.

ATTRIBUTE CONSTRAINTS:
You are ONLY allowed to generate questions about attributes that are EXPLICITLY present,
or strongly implied, in the target product metadata.

Examples of allowed attribute sources:
- title
- features
- material
- format
- dimensions
- usage statements (e.g., "exercise", "home", "casual")
- included accessories
- technical specs
- pattern, print, color, if shown in metadata
- category name fields

You MUST NOT generate questions about attributes that do NOT exist in the metadata.
For example:
- Do NOT ask about size, fit, color, or patterns unless metadata mentions them.
- Do NOT ask about battery life unless metadata mentions electronics features.
- Do NOT ask about genre unless metadata mentions books or literary attributes.

PREFERENCE EXTRACTION RULE:
You MUST infer the user's preferences ONLY from the user reviews provided.
Extract the themes that repeatedly appear across their reviews.
You may NOT invent any preference that does not appear in the reviews.

QUESTION DESIGN RULE (VERY IMPORTANT):

Write questions as if the USER is asking a shopping assistant for help
choosing or evaluating THIS specific product or similar options.

Your questions MUST:
- NOT ask the user about their preferences (no “do you prefer…”, “how important is it to you…”, “would you prioritize…”, etc.).
- NOT mention any preference words (“I like…”, “I prefer…”, “I care a lot about…”).
- Sound like the user is asking the assistant to decide or suggest something FOR THEM:
  - e.g., “Which option would you recommend for me?”, “What should I pay attention to?”, “Do you think this would be suitable for me?”.
- Create decision points where a generic assistant and a preference-aware assistant would give DIFFERENT answers.
  - The question itself stays neutral.
  - The personalization happens only in the ANSWER, not in the wording of the question.
- Be natural for THIS product type.
- Use ONLY attributes present in THIS product's metadata.

RAG VALUE TEST (CRITICAL):

For EACH question you generate, simulate:

- WITHOUT RAG (no user reviews):
  - The assistant can only see the product metadata and the text of the question.
  - It can give only generic, one-size-fits-all answers.

- WITH RAG (has user reviews):
  - The assistant sees the same metadata and question,
  - PLUS the user's past reviews, and can use them to tailor the answer.

If a question would lead to almost the SAME answer in both cases,
it is a BAD question and you must revise it.

IMPORTANT:
- You must NOT fix this by asking directly about preferences.
  - Do NOT change the question into “How important is X to you?” or “Do you prefer A or B?”.
- Instead, change the question to ask the assistant to choose or evaluate something FOR THE USER:
  - e.g., “Which option would suit me better?” instead of “Do you prefer A or B?”.

STRICT PROHIBITIONS:
- Do NOT copy any example preferences from this instruction.
- Do NOT reuse generic templates about clothing, books, shoes, electronics, or any category.
- Do NOT ask about attributes that are irrelevant or not mentioned in metadata.
- Do NOT make up any new product attributes.
- Do NOT mention the specific product name/title in questions.
- Do NOT state user's preferences explicitly in questions.

OUTPUT FORMAT (JSON):

Return ONLY a JSON object with:

1. "latent_preferences":
   - 2–6 short strings,
   - each describing one preference/sensitivity derived ONLY from THIS user's reviews,
   - no invented details,
   - no copying from examples.

2. "generated_questions":
   - 5–8 natural questions,
   - each based ONLY on attributes present in THIS product's metadata,
   - each designed so that answers would differ depending on the user's preferences,
   - questions MUST NOT explicitly state any user preference.

Example STRUCTURE (values are placeholders and MUST NOT be copied):

{{
  "latent_preferences": [
    "PREFERENCE_1",
    "PREFERENCE_2"
  ],
  "generated_questions": [
    "QUESTION_1",
    "QUESTION_2"
  ]
}}

IMPORTANT REMINDERS:
- Output ONLY valid JSON - NO comments like // or /* */ in the JSON output
- The `latent_preferences` must be derived from THIS user's actual reviews
- Questions must only use attributes from THIS product's metadata
- Do NOT copy any example text - generate everything fresh based on the actual data provided
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
