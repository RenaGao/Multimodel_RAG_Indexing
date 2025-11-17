"""
evaluate_user_shopping_dialogues.py
-----------------------------------
LLM-as-a-Judge scoring for personalized shopping dialogues (1–5 scale).
Evaluates assistant responses from a user satisfaction perspective.
"""

import json
import os
import argparse
from tqdm import tqdm
from openai import OpenAI
import time

# -----------------------------------
# Configuration
# -----------------------------------
MODEL_NAME = "gpt-4o-mini"  # can change to gpt-5
TEMPERATURE = 0.2
MAX_TOKENS = 1400
SLEEP_TIME = 1.5

# -----------------------------------
# Initialize OpenAI client
# -----------------------------------
client = OpenAI()

# -----------------------------------
# LLM evaluation prompt template
# -----------------------------------
PROMPT_TEMPLATE = """
You are an expert evaluator of shopping-assistant dialogues, focusing on user experience and satisfaction.

Evaluate the following conversation between a *shopper (user)* and a *shopping assistant (agent)*.
Rate each criterion from **1 (worst)** to **5 (best)** according to user satisfaction and dialogue fluency.

---
[User Goal]
{user_goal}

[Conversation]
{dialogue_text}
---

### Evaluation Criteria
1. **User Preference Alignment (UPA)** – How well did the assistant respect and personalize responses to user needs (budget, brand, tone, style, delivery)?
2. **Dialogue History Trend Consistency (DHTC)** – Did the assistant maintain coherent understanding and context through multiple turns?
3. **Dialogue Interactive Ability (DIA)** – How engaging, natural, and balanced was the interaction?
4. **Macro-Level Features**
   - Topic Management (flow of topics)
   - Tone Appropriateness (style, politeness)
   - Conversation Opening (warm and contextual)
   - Conversation Closing (smooth wrap-up)
5. **Overall Score (1–5)** – Overall communication fluency and goal achievement from the *user’s perspective*.

### Scoring Scale
5 = Excellent / natural / fully satisfying user experience
4 = Good / minor gaps but goal achieved
3 = Moderate / partially successful or mechanical
2 = Poor / confusing or partly irrelevant
1 = Very Poor / broken or failed communication

### Output Format (strict JSON)
{{
  "UserPreferenceAlignment": {{"score": int, "rationale": "text"}},
  "HistoryTrendConsistency": {{"score": int, "rationale": "text"}},
  "InteractiveAbility": {{"score": int, "rationale": "text"}},
  "MacroFeatures": {{
      "TopicManagement": int,
      "ToneAppropriateness": int,
      "Opening": int,
      "Closing": int
  }},
  "OverallScore": int,
  "OverallRationale": "text"
}}

Example entry:
"UserPreferenceAlignment": {{"score": 5, "rationale": "Agent remembered the user's price limit and preferred color, providing precise matches."}},
"""

# -----------------------------------
# Helper Functions
# -----------------------------------

def format_dialogue(dialogue_history):
    """Convert dialogue history list into readable text block."""
    return "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in dialogue_history])


def evaluate_dialogue(entry):
    """Send dialogue entry to LLM and retrieve evaluation JSON."""
    user_goal = entry.get("user_goal", "")
    dialogue_text = format_dialogue(entry["dialogue_history"])

    prompt = PROMPT_TEMPLATE.format(user_goal=user_goal, dialogue_text=dialogue_text)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict yet fair evaluator of dialogue quality in shopping assistance."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw_output = response.choices[0].message.content.strip()

        # Attempt to parse JSON safely
        try:
            eval_data = json.loads(raw_output)
        except json.JSONDecodeError:
            print(f"[Warning] JSON parsing failed, returning raw text for this dialogue.")
            eval_data = {"raw_output": raw_output}

        return eval_data

    except Exception as e:
        print(f"[Error] Evaluation failed for dialogue: {entry.get('dialogue_id', 'unknown')} → {e}")
        return {"error": str(e)}


def batch_evaluate(input_path, output_path, sleep_time=SLEEP_TIME):
    """Run batch evaluation for all dialogues."""
    with open(input_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    results = []
    for entry in tqdm(dialogues, desc="Evaluating shopping dialogues"):
        eval_result = evaluate_dialogue(entry)
        eval_result["dialogue_id"] = entry.get("dialogue_id", None)
        results.append(eval_result)
        time.sleep(sleep_time)  # prevent rate-limit issues

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Evaluation completed. Results saved to: {output_path}")


# -----------------------------------
# Main Execution
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run user-focused shopping dialogue evaluation (1–5 scale).")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file containing dialogues.")
    parser.add_argument("--output", type=str, default="user_shopping_evaluations.json", help="Path to save evaluation results.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    batch_evaluate(args.input, args.output)
