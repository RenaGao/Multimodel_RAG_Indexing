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
# 2. Build new prompt (v3) - Decision Point Approach with User Profile
# -------------------------------------
def build_v3_prompt(user_profile, target_product):
    """
    Build the v3 prompt using decision-point approach.

    - Use a precomputed user_profile (summarized from ALL reviews) as latent preferences.
    - Do NOT use sampled reviews here.
    - Generate user-side shopping questions that look generic but create RAG-sensitive decision points.
    """

    # -------------------------
    # 1. Format user profile text
    # -------------------------
    profile_text = ""
    if isinstance(user_profile, dict):
        for key, value in user_profile.items():
            if isinstance(value, str) and value.strip() and value.strip().lower() != "not specified":
                profile_text += f"- {key}: {value.strip()}\n"
    elif isinstance(user_profile, str):
        profile_text = user_profile.strip()
    else:
        profile_text = "(no explicit profile provided)"

    # -------------------------
    # 2. Format target product metadata
    # -------------------------
    target_meta = target_product["product_meta"]
    target_title = target_meta.get("title", "product")
    target_category = target_meta.get("main_category", "shopping")
    target_details = target_meta.get("details", {})
    target_features = target_meta.get("features", [])
    target_description = target_meta.get("description", [])

    metadata_text = f"Title: {target_title}\nCategory: {target_category}\n"

    if target_features:
        metadata_text += "Features:\n"
        for feat in target_features[:10]:
            if feat:
                metadata_text += f"  - {feat}\n"

    if target_description:
        desc_text = " ".join([str(d) for d in target_description if d])[:500]
        if desc_text:
            metadata_text += f"Description: {desc_text}\n"

    if target_details:
        metadata_text += "Details:\n"
        for key, value in target_details.items():
            metadata_text += f"  - {key}: {value}\n"

    # -------------------------
    # 3. Infer product_type from title
    # -------------------------
    title_words = target_title.lower().split()
    product_keywords = {
        "tank": "tank top",
        "top": "top",
        "dress": "dress",
        "shirt": "shirt",
        "hoodie": "hoodie",
        "sweater": "sweater",
        "pants": "pants",
        "jeans": "jeans",
        "shorts": "shorts",
        "skirt": "skirt",
        "jacket": "jacket",
        "coat": "coat",
        "shoes": "shoes",
        "boots": "boots",
        "sandals": "sandals",
        "sneakers": "sneakers",
        "bag": "bag",
        "purse": "purse",
        "wallet": "wallet",
        "belt": "belt",
        "hat": "hat",
        "cap": "hat",
        "scarf": "scarf",
        "gloves": "gloves",
        "socks": "socks",
        "necklace": "necklace",
        "earrings": "earrings",
        "bracelet": "bracelet",
        "ring": "ring",
        "extensions": "hair extensions",
        "weave": "hair extensions",
        "wig": "wig",
        "book": "book"
    }

    product_type = "product"
    for w in title_words:
        if w in product_keywords:
            product_type = product_keywords[w]
            break
    if "tank" in title_words and "top" in title_words:
        product_type = "tank top"

    # -------------------------
    # 4. Prompt
    # -------------------------
    prompt = f"""You are a question generator for a personalized shopping assistant.

SCENARIO:
- This is a RECOMMENDATION conversation.
- The user has NOT seen any specific item yet.
- They are generally shopping for a {product_type} (or similar items) and want the assistant to recommend options.
- The target product metadata below represents one candidate item the assistant might recommend, but the user does NOT refer to it directly.

INPUTS:
- Target product metadata (all available attributes).
- A precomputed USER PROFILE summarizing their long-term preferences.

TARGET PRODUCT METADATA
(for your understanding only, NOT mentioned by name in questions):
{metadata_text}

USER PROFILE (INTERNAL LATENT PREFERENCES - DO NOT REPEAT LITERALLY):
{profile_text}

GOAL:
Generate user-side shopping questions that:
- Look natural and fairly generic on the surface.
- Are phrased as the user asking for RECOMMENDATIONS for a {product_type} or similar items.
- Do NOT explicitly state the user's preferences.
- Create decision points where a generic assistant and a preference-aware assistant would give different answers.

CATEGORY & ATTRIBUTE AWARENESS (IMPORTANT):

You MUST determine:
- what kind of product this is,
- which attributes matter for evaluating or choosing it,
ONLY from the target product's metadata provided above.

Do NOT use any assumptions about product types such as clothing, books, electronics, or shoes.
Do NOT rely on any templates, stereotypes, or memorized category schemas.
Everything must be derived from THIS target product's metadata.

ATTRIBUTE CONSTRAINTS (EXTREMELY IMPORTANT):

You are ONLY allowed to generate questions about attributes that are EXPLICITLY present,
or strongly implied, in the target product metadata.

Allowed attribute sources include:
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
- details fields

You MUST NOT generate questions about attributes that do NOT exist in the metadata.
For example:
- Do NOT ask about size, fit, color, or patterns unless the metadata clearly suggests they exist.
- Do NOT ask about battery life unless the metadata suggests an electronic device.
- Do NOT ask about genre unless the metadata suggests a book or similar media.
- Do NOT assume the item is clothing, electronics, books, shoes, or any specific category.
You MUST infer the type ONLY from the metadata itself.

USER PROFILE USAGE RULE (CRITICAL):

- The USER PROFILE above already summarizes this user's long-term preferences, style, budget,
  quality priorities, dislikes, and pain points.
- You MUST treat this profile as the PRIMARY source of latent preferences.
- You MUST NOT invent new preferences that contradict this profile.
- You MUST NOT copy sentences or long phrases from the profile into the questions.
- Instead, you should:
  - Use the profile to decide which decision dimensions are important for this user
    (e.g., comfort, natural look, ease of use, durability, value for money, etc.),
  - Design questions that create decision points ALONG those dimensions,
    so that an assistant WITH access to this profile can answer differently
    from an assistant WITHOUT it.
- The questions must sound like things the user might naturally ask,
  not like a restatement of the profile.

GOOD QUESTION STYLE EXAMPLE (DO NOT COPY TEXT, ONLY STYLE):

For a case where the user is looking for a tank top, good questions look like:

- "I'm looking for a tank top. What would you recommend?"
- "What style of tank top suits me best - fitted or relaxed?"
- "Which material would be most comfortable for me?"
- "Which color options align with my usual preferences?"
- "What size should I consider based on my past purchases?"
- "Is this tank top suitable for both workouts and casual outings?"
- "How does the fit of this kind of tank top compare to what I usually wear?"
- "What care instructions should I consider for maintaining quality?"

These questions are GOOD because:
- The user is asking for recommendations, not describing a specific known item.
- The assistant has to decide FOR THE USER.
- They hint at decisions that depend on the user's history ("my usual preferences",
  "what I usually wear", "based on my past purchases").
- They do NOT include specific hidden preferences like "for warm weather",
  "for sensitive skin", "because I sweat a lot".
- They refer to high-level attributes (style, material, color, size, use case, care)
  that are typical for that product type.

QUESTION DESIGN RULE (VERY IMPORTANT):

Write questions as if the USER is asking a shopping assistant for help
choosing or evaluating a {product_type} or similar items, BEFORE seeing any specific product page.

Your questions MUST satisfy ALL of the following:

1) RECOMMENDATION, NOT EXPLANATION
- The tone must be "help me choose / what would you suggest for me",
  NOT "explain this product to me" or "what are the advantages of X in general".
- BAD: "What material would offer the most durability for my needs?"
  -> almost any user gets the same answer.
- GOOD: "When choosing between softer and more durable materials in {product_type}s,
  which would you recommend for me?"

2) AROUND A PREFERENCE DIMENSION OR TRADE-OFF
- EACH question must revolve around at least one decision dimension that can depend on preferences:
  - comfort / softness
  - durability / build quality
  - appearance / style / how it looks on me
  - usage scenario (everyday vs special occasions, casual vs active, etc.)
  - ease of care / cleaning / maintenance
  - value for money, etc.
- Prefer questions that implicitly ask for a choice or trade-off:
  - BAD: "What features should I look for in shorts?"
  - GOOD: "Given what I usually go for, which features should matter most for me when choosing shorts?"

3) EXPLICITLY ABOUT "ME" AND OFTEN ABOUT PAST BEHAVIOR
- Every question must clearly indicate that the user is asking FOR THEMSELVES,
  using phrases like:
  - "for me", "for someone like me", "for my needs",
  - "based on what I usually wear/use",
  - "compared to what I typically go for",
  - "given what I've liked before".
- At least HALF of the questions must explicitly mention past or typical behavior, e.g.:
  - "based on what I usually wear",
  - "given the kinds of things I've liked before",
  - "compared to what I normally choose".
- Example:
  - GOOD: "Based on the kinds of {product_type}s I usually use, which style would you recommend for me?"

4) DO NOT BAKE HIDDEN PREFERENCES INTO THE QUESTION
- You MUST NOT write concrete personal conditions that reveal hidden preferences, such as:
  - "for warm weather", "for my sensitive skin", "because my feet are wide", etc.
- Those details should be discovered by the assistant from the USER PROFILE,
  NOT baked into the question itself.
- You MAY use neutral quality words like "comfortable", "durable", "stylish",
  but without overly specific situational constraints.
  - OK: "Which material would be more comfortable for me?"
  - NOT OK: "Which material would keep me comfortable in very hot weather?"

5) GENERIC PRODUCT TYPE, NOT SPECIFIC ITEM
- Refer to the product type generically, e.g.:
  - "a {product_type}", "this kind of {product_type}", "these {product_type}s".
- MUST NOT refer to "this product", "this item", or a specific product page.

RAG VALUE TEST (CRITICAL):

For EACH question you generate, simulate two scenarios:

1) WITHOUT PROFILE (no user preferences):
   - The assistant sees only the product metadata and the text of the question.
   - It can give only generic, one-size-fits-all advice.

2) WITH PROFILE (has user preferences):
   - The assistant sees the same metadata and question,
   - PLUS the USER PROFILE, and can use it to tailor the answer.

If a question would lead to almost the SAME answer in both cases,
it is a BAD question and you must revise it.

You must NOT fix this by asking directly about preferences.
Instead, change the question to ask the assistant to choose or evaluate something FOR THE USER.

STRICT PROHIBITIONS:

- Do NOT copy any sentences or long phrases from the USER PROFILE.
- Do NOT copy any example questions from this instruction.
- Do NOT reuse generic templates about clothing, books, shoes, electronics, or any category.
- Do NOT ask about attributes that are irrelevant or not mentioned in metadata.
- Do NOT make up any new product attributes.
- Do NOT mention the specific product name/title in questions.
- Do NOT refer to "this product", "this item", "this {product_type}".
- Do NOT state the user's preferences explicitly in questions.
- Do NOT include specific personal conditions like "for warm weather", "for my sensitive skin", etc.

ABSOLUTELY FORBIDDEN question patterns:
- Any question starting with or containing:
  - "Do you prefer"
  - "Would you prefer"
  - "How important is it to you"
  - "Would you prioritize"
  - "Do you have a preference for"
  - "Is it essential to you that"
- Generic product-explanation questions such as:
  - "What are the advantages of ..."
  - "What are the key benefits of ..."
  - "Can you tell me about ..."
  - "How does this product perform in general?"

OUTPUT FORMAT (JSON):

Return ONLY a JSON object with:

"generated_questions":
   - 5–8 natural questions,
   - each based ONLY on attributes present in THIS product's metadata,
   - each phrased as the user asking the assistant to recommend or judge {product_type}s FOR THEM,
   - each designed so that answers would differ depending on the user's preferences,
   - questions MUST NOT explicitly state any user preference.

Example STRUCTURE (values are placeholders and MUST NOT be copied):

{{
  "generated_questions": [
    "QUESTION_1",
    "QUESTION_2"
  ]
}}

IMPORTANT REMINDERS:
- Output ONLY valid JSON - NO comments like // or /* */ in the JSON output.
- "generated_questions" must use ONLY attributes from THIS product's metadata.
- Do NOT copy any example text - generate everything fresh based on the actual data provided.
"""

    return prompt


# -------------------------------------
# 3. Generate v3 questions
# -------------------------------------
def generate_v3_questions(user_id, user_profile, target_product, model="gpt-4o-mini"):
    """
    Generate v3 questions using decision-point approach.
    Uses precomputed user_profile instead of sampled reviews.

    Args:
        user_id: User identifier
        user_profile: Precomputed user profile dict
        target_product: Target product for context
        model: OpenAI model to use

    Returns:
        List of questions or [] on failure
    """
    prompt = build_v3_prompt(user_profile, target_product)

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
            content = content.lstrip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()

        # Parse JSON object
        if content.startswith("{") and content.endswith("}"):
            data = json.loads(content)
            questions = data.get("generated_questions", [])
            return questions
        else:
            # Fallback: try to find JSON object in text
            print(f"[Warning] Unexpected format for user {user_id}, attempting fallback parsing")
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                try:
                    json_str = content[start_idx:end_idx+1]
                    data = json.loads(json_str)
                    questions = data.get("generated_questions", [])
                    return questions
                except Exception as fallback_error:
                    print(f"[Warning] Fallback parsing failed: {fallback_error}")

            print(f"[Error] Could not parse response for user {user_id}")
            return []

    except Exception as e:
        print(f"[Error] Failed to generate v3 questions for user '{user_id}': {e}")
        return []


# -------------------------------------
# 4. Helper functions for batch processing
# -------------------------------------
def load_user_profiles(category, profiles_dir="user_profiles"):
    """Load precomputed user profiles for a category."""
    profile_path = os.path.join(profiles_dir, f"{category}_user_profiles.json")
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_completed_categories(questions_v3_dir="questions_v3"):
    """Get list of already completed categories from questions_v3 folder."""
    completed = set()
    if os.path.exists(questions_v3_dir):
        for f in os.listdir(questions_v3_dir):
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
def main(input_file="questions/Amazon_Fashion_questions_20251110_214618.json", profiles_dir="user_profiles"):
    """
    Main workflow to regenerate v3 questions from existing questions file.

    Args:
        input_file: Path to existing questions JSON file
        profiles_dir: Directory containing user profile files
    """
    print(f"Loading existing questions from: {input_file}")
    existing_data = load_existing_questions(input_file)

    # Extract category and load user profiles
    filename = os.path.basename(input_file)
    category = filename.split("_questions_")[0]

    print(f"Loading user profiles for category: {category}")
    user_profiles = load_user_profiles(category, profiles_dir)
    print(f"Loaded profiles for {len(user_profiles)} users")

    if not user_profiles:
        print(f"[Error] No user profiles found for {category} in {profiles_dir}")
        return

    print(f"Loaded {len(existing_data)} users from questions file")

    # Process users
    all_results = []

    print(f"\nRegenerating v3 questions for {len(existing_data)} users...\n")

    for user_data in tqdm(existing_data, desc="Processing users"):
        user_id = user_data.get("user_id")
        target_product = user_data.get("target_product")

        if not target_product:
            print(f"[Skip] User {user_id} missing target_product")
            continue

        # Get user profile
        user_profile = user_profiles.get(user_id, {})
        if not user_profile:
            print(f"[Skip] User {user_id} has no profile")
            continue

        try:
            # Generate v3 questions using user profile
            v3_questions = generate_v3_questions(user_id, user_profile, target_product)

            if not v3_questions:
                print(f"[Skip] Failed to generate v3 questions for user {user_id}")
                continue

            # Store result with user_profile and questions (no user_history, no latent_preferences)
            result = {
                "user_id": user_id,
                "target_product": target_product,
                "user_profile": user_profile,
                "generated_questions": v3_questions
            }

            all_results.append(result)

        except Exception as e:
            print(f"[Error] Failed to process user {user_id}: {e}")
            continue

    # -------------------------------------
    # Save results
    # -------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("questions_v3", exist_ok=True)
    output_path = os.path.join("questions_v3", f"{category}_questions_{ts}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nV2 generation completed! Processed {len(all_results)} users")
    print(f"  Results saved to: {output_path}")

    # Print sample output
    if all_results:
        sample_user = all_results[0]
        print(f"\n{'='*80}")
        print(f"Sample User: {sample_user['user_id']}")
        print(f"{'='*80}")

        # Show user profile
        print("\nUser Profile:")
        for key, value in sample_user.get('user_profile', {}).items():
            if value and str(value).strip().lower() != "not specified":
                print(f"  - {key}: {value}")

        # Show generated questions
        print("\nGenerated Questions:")
        for i, q in enumerate(sample_user.get('generated_questions', [])[:5], 1):
            print(f"  {i}. {q}")

        print(f"\n{'='*80}\n")


# -------------------------------------
# Entry point
# -------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate v3 personalized questions using precomputed user profiles")
    parser.add_argument(
        "--input",
        type=str,
        default="questions/Amazon_Fashion_questions_20251110_214618.json",
        help="Path to existing questions JSON file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all categories in questions/ folder, skip already completed ones in questions_v3/"
    )
    parser.add_argument(
        "--profiles-dir",
        type=str,
        default="./user_profiles",
        help="Directory containing user profile files (default: user_profiles)"
    )

    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"Regenerating V2 Questions (Using Precomputed User Profiles)")
    print(f"{'='*80}\n")
    print(f"Profiles directory: {args.profiles_dir}\n")

    if args.all:
        # Batch processing mode
        completed = get_completed_categories()
        all_files = get_all_input_files()

        print(f"Found {len(all_files)} categories in questions/")
        print(f"Already completed: {len(completed)} categories in questions_v3/")
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
            main(input_file=input_file, profiles_dir=args.profiles_dir)

    else:
        # Single file processing mode
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            exit(1)

        main(input_file=args.input, profiles_dir=args.profiles_dir)

    print(f"\n{'='*80}")
    print("V2 generation completed successfully!")
    print("New questions use decision-point approach with precomputed user profiles:")
    print("- Questions look generic on surface")
    print("- Create decision points where user profile enables personalized answers")
    print("- Output includes: user_id, target_product, user_profile, generated_questions")
    print(f"{'='*80}")
