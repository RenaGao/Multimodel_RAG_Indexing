"""
Eval_Agents_Response_V2.py
-----------------------------------
LLM-as-a-Judge scoring for Shopping Agents.
Focus: WHOLE DIALOGUE AGGREGATE ASSESSMENT.

Evaluates dialogues from a single input directory (with category subdirs)
and outputs per-category eval results.

Usage:
    python Eval_Agents_Response_V2.py --input <dialogues_dir> --output-dir <eval_output_dir>
"""

import json
import os
import argparse
import glob
import concurrent.futures
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROFILES_DIR = str(_REPO_ROOT / "data" / "user_profiles")

# -----------------------------------
# Configuration
# -----------------------------------
MODEL_NAME = "openai/gpt-4o"
TEMPERATURE = 0.0
MAX_WORKERS = 10

load_dotenv()

# -----------------------------------
# Initialize OpenRouter client
# -----------------------------------
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Missing OPENROUTER_API_KEY in environment variables. Please add it to your .env file.")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# -----------------------------------
# The Aggregate Evaluation Prompt
# -----------------------------------
EVALUATION_PROMPT = """
You are a Lead Auditor evaluating the performance of an AI Shopping Assistant.
Your task is to score the **Agent's Cumulative Performance** across the entire session.
Do NOT evaluate the User. Use the User's turns only to determine if the Agent's responses were relevant.

### 1. User Profile & Goal (The Target)
Profile: {user_profile}
Goal: {user_goal}

### 3. The Interaction Log (Context)
(Use this to check if the Agent answered questions correctly)
{full_dialogue_text}

### 4. The Agent Response Stream (Target of Evaluation)
(Focus your scoring strictly on this text. Look for consistency, tone, and repeated errors)
{agent_only_text}

---

### Aggregate Scoring Criteria (Score 1-5)

#### 1. Global Personalization Consistency (GPC)
*Did the agent maintain the user's persona constraints throughout the ENTIRE dialogue?*
*   **1 (Inconsistent/Amnesic):** Agent started well but forgot constraints (e.g., budget, size) in later turns, or hallucinated user details.
*   **3 (Fragmented):** Agent remembered explicit constraints (Size M) but forgot implicit ones (Style preferences) as the chat went on.
*   **5 (Rock Solid):** Every single response, from start to finish, aligned with the User Profile. The agent never "reset" or forgot context.

#### 2. Cumulative Information Synthesis (CIS)
*Looking at all agent responses combined, did it build a complete, factual picture?*
*   **1 (Fragmented/Repetitive):** Agent repeated the same facts multiple times or gave contradictory information across different turns.
*   **3 (Basic Retrieval):** Agent provided facts when asked, but didn't connect them.
*   **5 (Coherent Narrative):** The agent's responses build on each other. Later responses reference earlier facts (e.g., "As I mentioned earlier regarding the fabric...").

#### 3. Interaction Trajectory (IT)
*Did the Agent's responses purposefully drive the conversation to a conclusion?*
*   **1 (Circular):** The agent went in circles, repeating suggestions or asking the same clarifying questions.
*   **3 (Passive):** The agent answered questions accurately but waited for the user to lead.
*   **5 (Strategic):** The agent's responses followed a logical arc: Clarify -> Recommend -> Refine -> Close.

#### 4. Tone Consistency (TC)
*Was the agent's voice stable?*
*   **1 (Erratic):** Switched between robotic formal ("As an AI...") and casual ("Hey there!") randomly.
*   **5 (Unified Voice):** Maintained a consistent, professional, and empathetic shopping persona from Hello to Goodbye.

---

### Output Format (JSON ONLY)
{{
  "GlobalPersonalizationConsistency": {{ "score": int, "reason": "string" }},
  "CumulativeInformationSynthesis": {{ "score": int, "reason": "string" }},
  "InteractionTrajectory": {{ "score": int, "reason": "string" }},
  "ToneConsistency": {{ "score": int, "reason": "string" }},
  "OverallScore": float,
  "Major_Flaws_Detected": ["list", "of", "flaws"]
}}

**Valid Flaws:**
["Forgot_Constraint_Mid_Chat", "Repetitive_Loops", "Tone_Switch", "Passive_Behavior", "None"]
"""

# -----------------------------------
# Helper Functions
# -----------------------------------

def format_full_dialogue(dialogue_history):
    """Formats the full back-and-forth for context."""
    formatted = []
    for turn in dialogue_history:
        role = turn.get('role', 'unknown').upper()
        content = turn.get('content', '')
        formatted.append(f"[{role}]: {content}")
    return "\n".join(formatted)


def extract_agent_stream(dialogue_history):
    """
    Extracts ONLY the agent's text.
    This allows the judge to analyze the agent's 'voice' in isolation.
    """
    agent_turns = []
    for i, turn in enumerate(dialogue_history):
        if turn.get('role') == 'assistant':
            agent_turns.append(f"[Response {i}]: {turn.get('content', '')}")

    if not agent_turns:
        return "No Agent Responses Found."

    return "\n\n".join(agent_turns)


def get_user_profile(entry, loaded_profiles=None):
    """
    Get user profile from loaded profiles file or fallback to entry's user_profile field.
    """
    if loaded_profiles:
        user_id = entry.get("user_id")
        if not user_id and "dialogue_id" in entry:
            dialogue_id = entry.get("dialogue_id", "")
            user_id = dialogue_id.split("_")[0] if "_" in dialogue_id else dialogue_id

        if user_id and user_id in loaded_profiles:
            profile = loaded_profiles[user_id]
            if profile:
                return ", ".join([f"{k}: {v}" for k, v in profile.items()])

    profile = entry.get("user_profile", {})
    if not profile:
        return "No Profile Available."
    return ", ".join([f"{k}: {v}" for k, v in profile.items()])


def load_dialogues_from_category(category_dir):
    """
    Load all dialogues from a category directory.
    Only loads .json files (ignores .jsonl backup files).

    Returns:
        List of dialogue entries
    """
    dialogues = []

    # Find all JSON files (ignore .jsonl backup files)
    json_files = glob.glob(os.path.join(category_dir, "*.json"))

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            dialogues.extend(data)
        except Exception as e:
            print(f"✗ Error loading {os.path.basename(file_path)}: {str(e)}")

    return dialogues


def evaluate_single_entry(entry, loaded_profiles=None):
    """Evaluate a single dialogue entry using LLM-as-a-Judge."""
    full_dialogue = format_full_dialogue(entry.get("dialogue_history", []))
    agent_stream = extract_agent_stream(entry.get("dialogue_history", []))

    user_goal = entry.get("user_goal", "Shopping")
    user_profile = get_user_profile(entry, loaded_profiles)

    full_prompt = EVALUATION_PROMPT.format(
        user_profile=user_profile,
        user_goal=user_goal,
        full_dialogue_text=full_dialogue,
        agent_only_text=agent_stream
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict auditor of AI Agent behavior."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )
        result_json = json.loads(response.choices[0].message.content)

        return {
            "dialogue_id": entry.get("dialogue_id", "unknown"),
            "evaluation": result_json
        }
    except Exception as e:
        return {"dialogue_id": entry.get("dialogue_id"), "error": str(e)}


def evaluate_category(category, category_dir, output_dir, loaded_profiles=None):
    """
    Evaluate all dialogues for a single category.

    Args:
        category: Category name
        category_dir: Path to category directory
        output_dir: Directory to save results
        loaded_profiles: Optional dict of user profiles

    Returns:
        Dict with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"📂 Processing category: {category}")
    print(f"{'='*60}")

    dialogues = load_dialogues_from_category(category_dir)

    if not dialogues:
        print(f"⚠️  No dialogues found for {category}")
        return None

    print(f"✓ Loaded {len(dialogues)} dialogues")

    # Evaluate dialogues with thread pool
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(evaluate_single_entry, d, loaded_profiles) for d in dialogues]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            results.append(future.result())
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(dialogues)} dialogues...")

    # Save category results
    output_path = os.path.join(output_dir, f"{category}_eval.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved results to: {output_path}")

    return {
        "category": category,
        "total_dialogues": len(dialogues),
        "results": results
    }


def batch_evaluate(input_dir, output_dir, profiles_dir=None):
    """
    Evaluate all categories in input directory.

    Args:
        input_dir: Directory containing category subdirectories
        output_dir: Directory to save evaluation results
        profiles_dir: Optional directory containing user profile files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Discover all categories (subdirectories)
    categories = [d for d in os.listdir(input_dir)
                  if os.path.isdir(os.path.join(input_dir, d))]
    categories.sort()

    if not categories:
        print(f"❌ No category subdirectories found in: {input_dir}")
        return

    print(f"\n{'='*60}")
    print(f"🚀 BATCH EVALUATION")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(categories)} categories")
    print(f"Max workers: {MAX_WORKERS}")

    category_summaries = []

    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] ", end="")

        category_dir = os.path.join(input_dir, category)

        # Load profiles for this category if profiles_dir provided
        loaded_profiles = None
        if profiles_dir:
            profiles_path = os.path.join(profiles_dir, f"{category}_user_profiles.json")
            if os.path.exists(profiles_path):
                try:
                    with open(profiles_path, "r", encoding="utf-8") as f:
                        loaded_profiles = json.load(f)
                    print(f"✓ Loaded profiles from {profiles_path}")
                except Exception as e:
                    print(f"⚠️  Error loading profiles: {str(e)}")

        result = evaluate_category(category, category_dir, output_dir, loaded_profiles)
        if result:
            category_summaries.append({
                "category": result["category"],
                "total_dialogues": result["total_dialogues"]
            })

    # Save summary
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(category_summaries, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"📊 EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total categories: {len(category_summaries)}")
    print(f"Total dialogues: {sum(s['total_dialogues'] for s in category_summaries)}")
    print(f"💾 Summary saved to: {summary_path}")


# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AI Shopping Assistant dialogues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all categories in a directory
  python %(prog)s --input dialogues_output --output-dir eval_results

  # With user profiles
  python %(prog)s --input dialogues_output --output-dir eval_results --profiles-dir user_profiles
"""
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing category subdirectories with dialogue files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--profiles-dir",
        type=str,
        default=_DEFAULT_PROFILES_DIR,
        help=f"Directory containing user profile JSON files (default: {_DEFAULT_PROFILES_DIR})"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise ValueError(f"Input directory not found: {args.input}")

    batch_evaluate(args.input, args.output_dir, args.profiles_dir)
