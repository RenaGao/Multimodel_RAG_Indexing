"""
evaluate_shopping_interactivity.py
-----------------------------------
LLM-as-a-Judge scoring for Shopping Agents.
Focus: Interactivity (Clarification, Refinement) + RAG Accuracy.
"""

import json
import os
import argparse
import concurrent.futures
from typing import Dict, Any
from openai import OpenAI
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------------
# Configuration
# -----------------------------------
MODEL_NAME = "gpt-4o"  # Strong reasoning required for "Interactivity" judgment
TEMPERATURE = 0.1 
MAX_WORKERS = 5

# -----------------------------------
# Initialize OpenAI client
# -----------------------------------
api_key = os.getenv("OPENAI_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_KEY in environment variables. Please add it to your .env file.")

client = OpenAI(api_key=api_key)

# -----------------------------------
# The Evaluation Prompt
# -----------------------------------
# This prompt is engineered to catch "Passive" agents vs "Interactive" agents.

EVALUATION_PROMPT = """
You are an expert judge evaluating a **Conversational Shopping Assistant**. 
Your goal is to rate how INTERACTIVE and FACTUAL the agent is.

### Context Provided to Agent (RAG Knowledge)
{context_snippet}

### User Goal
{user_goal}

### Dialogue History
{dialogue_text}

---

### Evaluation Instructions
Rate the Assistant on the following 5 metrics (Score 1-5).

#### 1. Proactive Clarification (PC) [Interaction Focus]
*Did the agent ask necessary questions to narrow down the search BEFORE making recommendations?*
*   **5 (Excellent):** Agent identified ambiguity (e.g., missing budget, size, occasion) and asked specific, relevant clarifying questions before searching/suggesting.
*   **3 (Average):** Agent asked generic questions ("How can I help?") or made assumptions without asking.
*   **1 (Poor):** User gave a vague request ("I need shoes"), and Agent immediately dumped a list of random products without checking preferences.

#### 2. Adaptive Refinement (AR) [Interaction Focus]
*How well did the agent handle User Feedback or Constraints?*
*   **5 (Excellent):** Agent immediately adjusted recommendations based on critique (e.g., User: "Too expensive" -> Agent: "Here are cheaper options"). Explicitly acknowledges the change.
*   **3 (Average):** Agent acknowledged the constraint but didn't fully filter the next results.
*   **1 (Poor):** Agent ignored feedback (e.g., User said "No red", Agent showed red items) or repeated previous suggestions.

#### 3. Retrieval Utilization (RU) [RAG Focus]
*Did the agent effectively use the provided Context/RAG data?*
*   **5 (Excellent):** Answers are rich with details found ONLY in the context (e.g., specific fabric weights, inventory counts).
*   **1 (Poor):** Agent gave generic answers that could apply to any product, ignoring the specific details in the provided text.

#### 4. Product Faithfulness (PF) [Safety Focus]
*Did the agent hallucinate?*
*   **5 (Safe):** 100% adherence to facts. If info was missing, the agent admitted it.
*   **1 (Unsafe):** Agent invented prices, specs, or availability that does not exist in the Context.

#### 5. Empathy & Tone (ET)
*   **5:** Natural, enthusiastic, helpful.
*   **1:** Robotic, repetitive, or lectures the user.

---

### Output Format (JSON ONLY)
Ensure strict JSON format.
{{
  "ProactiveClarification": {{ "score": int, "reason": "string" }},
  "AdaptiveRefinement": {{ "score": int, "reason": "string" }},
  "RetrievalUtilization": {{ "score": int, "reason": "string" }},
  "ProductFaithfulness": {{ "score": int, "reason": "string" }},
  "EmpathyTone": {{ "score": int, "reason": "string" }},
  "OverallScore": float,
  "Behavioral_Tags": ["list", "of", "tags"] 
}}

**Valid Behavioral Tags to choose from:**
["Asked_Clarification", "Dumped_Results_Too_Early", "Ignored_Constraint", "Hallucinated_Specs", "Good_Pivot", "Generic_Response", "Cited_Sources"]
"""

# -----------------------------------
# Helper Functions
# -----------------------------------

def format_dialogue(dialogue_history):
    """Formats dialogue into a readable script."""
    formatted = []
    for turn in dialogue_history:
        role = turn.get('role', 'unknown').upper()
        content = turn.get('content', '')
        formatted.append(f"[{role}]: {content}")
    return "\n".join(formatted)

def get_context_snippet(entry):
    """Extracts retrieved context/documents if available."""
    # If your log has a list of docs, join them here
    context = entry.get("retrieved_context", "")
    if isinstance(context, list):
        return "\n\n".join([str(c) for c in context])
    return context if context else "No RAG context provided."

def evaluate_single_entry(entry):
    """Process a single dialogue entry."""
    dialogue_text = format_dialogue(entry.get("dialogue_history", []))
    user_goal = entry.get("user_goal", "Shopping Assistance")
    context_snippet = get_context_snippet(entry)

    # Construct prompt
    full_prompt = EVALUATION_PROMPT.format(
        context_snippet=context_snippet,
        user_goal=user_goal,
        dialogue_text=dialogue_text
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a critical evaluator of AI dialogue systems."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )
        result_json = json.loads(response.choices[0].message.content)
        
        return {
            "dialogue_id": entry.get("dialogue_id", "unknown"),
            "rag_enabled": entry.get("rag_enabled", False),
            "evaluation": result_json
        }
    except Exception as e:
        print(f"Error evaluating ID {entry.get('dialogue_id')}: {e}")
        return {"dialogue_id": entry.get("dialogue_id"), "error": str(e)}

def batch_evaluate(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    with open(input_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    print(f"🚀 Starting evaluation of {len(dialogues)} dialogues (Threads: {MAX_WORKERS})...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_entry = {executor.submit(evaluate_single_entry, entry): entry for entry in dialogues}
        
        for future in concurrent.futures.as_completed(future_to_entry):
            res = future.result()
            results.append(res)
            if len(results) % 5 == 0:
                print(f"   Processed {len(results)}/{len(dialogues)}...")

    # Save full results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Evaluation completed. Saved to: {output_path}")
    print_interaction_report(results)

def print_interaction_report(results):
    """Generates a specific report on INTERACTIVITY success/failure."""
    
    stats = {
        "rag": {"clarification": [], "refinement": [], "tags": {}},
        "no_rag": {"clarification": [], "refinement": [], "tags": {}}
    }

    for r in results:
        if "error" in r: continue
        
        key = "rag" if r.get("rag_enabled") else "no_rag"
        eval_data = r["evaluation"]
        
        # Collect Scores
        stats[key]["clarification"].append(eval_data["ProactiveClarification"]["score"])
        stats[key]["refinement"].append(eval_data["AdaptiveRefinement"]["score"])
        
        # Collect Tags
        for tag in eval_data.get("Behavioral_Tags", []):
            stats[key]["tags"][tag] = stats[key]["tags"].get(tag, 0) + 1

    print("\n📊 INTERACTIVITY & RAG REPORT 📊")
    print("-----------------------------------")
    
    for key in ["rag", "no_rag"]:
        c_scores = stats[key]["clarification"]
        r_scores = stats[key]["refinement"]
        count = len(c_scores)
        
        if count == 0: continue
        
        avg_c = sum(c_scores) / count
        avg_r = sum(r_scores) / count
        
        print(f"\nConfiguration: {key.upper()}")
        print(f"  Samples: {count}")
        print(f"  🗣️  Proactive Clarification: {avg_c:.2f} / 5.0")
        print(f"  🧠  Adaptive Refinement:     {avg_r:.2f} / 5.0")
        
        # Show top behaviors
        sorted_tags = sorted(stats[key]["tags"].items(), key=lambda x: x[1], reverse=True)
        print("  🏷️  Top Behaviors Detected:")
        for tag, freq in sorted_tags[:3]:
            print(f"      - {tag}: {freq} times")

# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, default="interactivity_eval.json")
    args = parser.parse_args()

    batch_evaluate(args.input, args.output)
