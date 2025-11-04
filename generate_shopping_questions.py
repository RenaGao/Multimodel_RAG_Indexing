from openai import OpenAI
import json
from tqdm import tqdm
import os
import ast
from datetime import datetime

# -------------------------------------
# 0. Initialize OpenAI client
# -------------------------------------
# Make sure you have set your API key:
# export OPENAI_API_KEY="your_api_key"
client = OpenAI()

# -------------------------------------
# 1. Define shopping domain list
# -------------------------------------
shopping_domains = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown",
]

print("Shopping domains:", shopping_domains)

# -------------------------------------
# 2. Prompt template definition
# -------------------------------------
prompt_template = """

[System Prompt]
personalized conversational agent for {domain} shopping.
Generate 3 natural, diverse, realistic customer questions about the "{domain}" domain.
Each question should sound like it comes from a real shopper and should cover different intents
(e.g., product search, availability, delivery, return policy, price comparison, recommendation).
Your goal is not only to answer but also to continue the conversation naturally — 
ask appropriate follow-up questions when useful, showing genuine curiosity and understanding.

[User Context]
Assume a general online shopper with varied interests, neutral tone, and no prior purchase history.
You don’t have access to personal data — focus on creating realistic and relatable shopping questions.

[Session Context]
No previous messages or conversation history are provided.
You are starting a new session about {domain} products.

[Product Information]
No specific products or brands are given.
Base your questions on common types of items, features, and concerns relevant to {domain} shopping.

[Instruction]
Generate a response that:
1. Provides a relevant and empathetic reply to the user’s current message.
2. Reflects understanding of user’s intent and emotions.
3. Naturally introduces a follow-up question **IF**:
   - The user’s intent or preferences are unclear;
   - There is potential to refine recommendation;
   - The user shows hesitation or curiosity.
4. Follow-up questions should be short, context-aware, and non-intrusive.
5. Keep tone aligned with the user’s communication style.

Output in the format as follow:
{{
  "response": "...",  
  "reasoning": "why this response fits user context",
  "follow_up": "..."  # optional, leave blank if not needed
}}
"""

# -------------------------------------
# 3. Question generation function
# -------------------------------------
def generate_questions(domain: str, model: str = "gpt-4o-mini"):
    """
    Generate a list of realistic shopping-related customer questions for a given domain.
    """
    prompt = prompt_template.format(domain=domain)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful data generation assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=800,
        )

        content = response.choices[0].message.content.strip()

        # Try parsing as a Python list
        if content.startswith("[") and content.endswith("]"):
            try:
                data = ast.literal_eval(content)
            except Exception:
                # Fallback: parse line by line
                data = [line.strip("-• \n") for line in content.split("\n") if line.strip()]
        else:
            # Fallback: parse line by line
            data = [line.strip("-• \n") for line in content.split("\n") if line.strip()]

        return data

    except Exception as e:
        print(f"[Error] Failed to generate questions for '{domain}': {e}")
        return []

# -------------------------------------
# 4. Main logic
# -------------------------------------
if __name__ == "__main__":
    print("Generating shopping domain user questions...\n")
    all_data = {}

    for domain in tqdm(shopping_domains, desc="Generating domains"):
        questions = generate_questions(domain)
        all_data[domain] = questions

    # -------------------------------------
    # 5. Save results as a JSON file
    # -------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "shopping_user_questions_{ts}.json".format(ts=ts))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n Generation completed! File saved to: {output_path}")
