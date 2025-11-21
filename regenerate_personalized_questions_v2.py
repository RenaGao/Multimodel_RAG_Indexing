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

    # Extract target category and product type for context
    target_meta = target_product["product_meta"]
    target_category = target_meta.get("main_category", "shopping")
    target_title = target_meta.get("title", "product")

    # Extract product type (e.g., "tank top", "dress", "hoodie")
    # Simple heuristic: take first few words of title
    target_type = " ".join(target_title.lower().split()[:3])

    prompt = f"""
[CRITICAL INSTRUCTION]
Generate questions where the user ONLY MENTIONS they have purchase/review history, but DOES NOT explicitly state their preferences in the question itself.

The agent MUST use RAG (Retrieval-Augmented Generation) to retrieve the user's reviews and infer their preferences. If preferences are stated in the question, RAG becomes unnecessary!

[CRITICAL: Product Focus]
ALL questions in this conversation must be about: "{target_type}"

This is ONE shopping session where the user is looking for ONE type of product.
Do NOT generate questions about different product types (e.g., don't mix "top", "hoodie", "dress").
All 5-8 questions should be a coherent conversation about finding the SAME product type.

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

[CORRECT Examples - NATURAL CONVERSATION STYLE]
Example: If target product is "cotton tank top", ALL questions should be about tank tops:

✅ "Hey, I'm looking for a new tank top. Based on what I usually buy, what would you suggest?"
   → GOOD: Casual opening, references history naturally, mentions target product
✅ "I need something similar to the tank tops I've ordered before. Any ideas?"
   → GOOD: Natural phrasing, agent must RAG to understand "similar", stays focused
✅ "What about the fit? What style of tank top would work for me?"
   → GOOD: Follow-up question, agent needs RAG to know user's fit preferences
✅ "I'm shopping for summer tank tops again. What would work for me?"
   → GOOD: Seasonal context, references pattern ("again") without specifics
✅ "Can you help me find a tank top? I want to stick with my usual style."
   → GOOD: Casual request, mentions "style" but doesn't describe it
✅ "Got any other tank top recommendations based on what I typically order?"
   → GOOD: Very casual, natural phrasing, asks for alternatives
✅ "What do you think would suit me for a tank top? I trust your picks based on my history."
   → GOOD: Friend-like tone, conversational, maintains product focus

[Task]
Generate 5-8 questions about "{target_type}" in the {target_category} category where:

1. **COHERENCE (CRITICAL)**: ALL questions must be about "{target_type}"
   - This is ONE shopping conversation for ONE product type
   - Questions should naturally progress through a coherent shopping journey
   - Example progression: Introduction → Style/Fit questions → Feature questions → Alternatives → Specifics
   - Do NOT ask about different products (e.g., don't mix tank tops with hoodies)

2. **Natural Conversation**: Sound like a REAL person talking to a shopping assistant
   - Use casual openings: "Hey", "I'm looking for", "I need", "Can you help me find"
   - Natural phrasing: "What do you think?", "Any ideas?", "Got suggestions?", "Thoughts?"
   - Add shopping context: "My old {target_type} wore out", "Shopping for {target_type} again"

3. **Reference History (but casually)**: Naturally mention purchase patterns
   - "based on what I usually buy", "similar to what I've ordered", "my usual style", "what I typically get"
   - NOT formal phrases like "Based on my purchase history" (too robotic)

4. **NO Explicit Preferences**: User does NOT state what they liked/disliked
   - Don't mention: soft fabrics, comfortable, sizing issues, specific features, brands, etc.
   - Agent MUST retrieve reviews to discover these

5. **Test RAG Requirement**: Questions should be:
   - WITHOUT RAG: Agent can only give generic recommendations
   - WITH RAG: Agent retrieves reviews, understands preferences, gives personalized recommendations

6. **Natural Progression**: Build conversation naturally about "{target_type}":
   - Start: "Hey, I'm looking for a new {target_type}. What would work for me?"
   - Follow-up: "I need {target_type} similar to what I usually get. Ideas?"
   - Clarify: "What about the fit/style of {target_type}? What suits me?"
   - Alternatives: "Got any other {target_type} options that match my taste?"

7. **Variety**: Mix different conversation styles (but all about "{target_type}"):
   - Direct requests: "I need a new {target_type}"
   - Questions: "What {target_type} do you think would work?"
   - Explanations: "My old {target_type}'s worn out"
   - Trust-building: "You know my style, what {target_type} is good?"

[Output Format]
Return ONLY a JSON array of question strings:
["question 1", "question 2", ...]

Remember:
- **ALL questions must be about "{target_type}"** - this is ONE coherent shopping conversation
- Sound like a REAL person having a natural conversation
- Reference purchase history CASUALLY (not formally)
- Don't state specific preferences
- Agent must use RAG to understand user's taste
- Questions should flow naturally as a progressive conversation about the SAME product
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
