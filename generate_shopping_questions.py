
from openai import OpenAI
import json
from tqdm import tqdm
import os

# -------------------------------
# 0. 初始化 OpenAI 客户端
# -------------------------------
# export OPENAI_API_KEY="aaa"
client = OpenAI()

# -------------------------------
# 1. 定义 shopping domain 列表
# -------------------------------
# shopping_domains = [
# "All_Beauty", 
# "Amazon_Fashion", 
# "Appliances",
# "Arts_Crafts_and_Sewing",
# "Automotive", 
# ]


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


print("shopping domains: ", shopping_domains)


# -------------------------------
# 2. Prompt 模板定义
# -------------------------------
# prompt_template = """
# You are an e-commerce user interacting with an AI shopping assistant.
# Generate 15 natural, diverse, realistic customer questions about the "{domain}" domain.
# Each question should sound like it comes from a real shopper and should cover different intents
# (e.g., product search, availability, delivery, return policy, price comparison, recommendation).
# Output only a valid Python list of strings (no explanations, no extra text).
# """


prompt_template = """
You are an e-commerce user interacting with an AI shopping assistant.
Generate 15 natural, diverse, realistic customer questions about the "{domain}" domain.
Each question should sound like it comes from a real shopper and should cover different intents
(e.g., product search, availability, delivery, return policy, price comparison, recommendation).
Output only a valid list of strings (no explanations, no extra text).
"""

# -------------------------------
# 3. 生成函数
# -------------------------------
def generate_questions(domain: str, model="gpt-4o-mini"):
    """
    Generate a list of shopping-related customer questions for a given domain.
    """
    prompt = prompt_template.format(domain=domain)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful data generation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,  
            max_tokens=800
        )

        content = response.choices[0].message.content.strip()

        # 是 Python list 格式
        if content.startswith("[") and content.endswith("]"):
            data = eval(content)
        else:
            # 尝试简单解析
            data = [line.strip("-• \n") for line in content.split("\n") if line.strip()]

        return data

    except Exception as e:
        print(f"[Error] Failed to generate for {domain}: {e}")
        return []

# -------------------------------
# 4. 主逻辑
# -------------------------------
if __name__ == "__main__":
    print("Generating shopping domain user questions...\n")
    all_data = {}

    for domain in tqdm(shopping_domains, desc="Generating domains"):
        questions = generate_questions(domain)
        all_data[domain] = questions

    # -------------------------------
    # 5. 保存为 JSON 文件
    # -------------------------------
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "shopping_user_questions.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Generation completed! File saved to: {output_path}")
