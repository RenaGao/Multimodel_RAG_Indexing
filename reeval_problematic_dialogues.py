"""
reeval_problematic_dialogues.py
-----------------------------------
Re-evaluate dialogues that were previously problematic and have been re-run.
Reads from problematic_dialogues.csv, finds updated dialogues, evaluates them,
and updates the corresponding eval_results JSON files.

Usage:
    python reeval_problematic_dialogues.py
    python reeval_problematic_dialogues.py --dry-run  # Preview without making changes
"""

import json
import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures

from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------------
# Configuration
# -----------------------------------
MODEL_NAME = "openai/gpt-4o"
TEMPERATURE = 0.0
MAX_WORKERS = 5

# Path mappings
# method in CSV -> (dialogue_outputs subdir, eval_results subdir)
METHOD_MAPPING = {
    "multimdoal_no_user_review_output": ("multimdoal_no_user_review_output", "multimodal_no_user_review_eval"),
    "multimodal_dialogues_output": ("multimodal_dialogues_output", "multimodalrag_eval"),
}

load_dotenv()

# -----------------------------------
# Initialize OpenRouter client
# -----------------------------------
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Missing OPENROUTER_API_KEY in environment variables.")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# -----------------------------------
# The Aggregate Evaluation Prompt (from Eval_Agents_Response_V2.py)
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
# Helper Functions (from Eval_Agents_Response_V2.py)
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
    """Extracts ONLY the agent's text."""
    agent_turns = []
    for i, turn in enumerate(dialogue_history):
        if turn.get('role') == 'assistant':
            agent_turns.append(f"[Response {i}]: {turn.get('content', '')}")
    if not agent_turns:
        return "No Agent Responses Found."
    return "\n\n".join(agent_turns)


def get_user_profile(entry, loaded_profiles=None):
    """Get user profile from loaded profiles or entry."""
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


# -----------------------------------
# Main Logic
# -----------------------------------

def load_problematic_dialogues(csv_path: str) -> List[Dict]:
    """Load problematic dialogues from CSV."""
    entries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "method": row["method"],
                "category": row["category"],
                "dialogue_id": row["dialogue_id"],
            })
    return entries


def find_dialogue_in_category(dialogue_dir: Path, category: str, dialogue_id: str) -> Optional[Dict]:
    """Find a specific dialogue by ID in a category directory."""
    category_dir = dialogue_dir / category
    if not category_dir.exists():
        return None

    for json_file in category_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for entry in data:
                if entry.get("dialogue_id") == dialogue_id:
                    return entry
        except Exception:
            continue
    return None


def update_eval_results(eval_path: Path, dialogue_id: str, new_eval: Dict) -> bool:
    """Update a specific dialogue's evaluation in the eval results file."""
    if not eval_path.exists():
        print(f"  ERROR: Eval file not found: {eval_path}")
        return False

    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)

    # Find and update the entry
    updated = False
    for i, entry in enumerate(eval_results):
        if entry.get("dialogue_id") == dialogue_id:
            eval_results[i] = new_eval
            updated = True
            break

    if not updated:
        # Entry not found, append it
        eval_results.append(new_eval)
        print(f"  NOTE: dialogue_id {dialogue_id} not found in eval, appending")

    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    return True


def load_user_profiles(profiles_dir: Path, category: str) -> Optional[Dict]:
    """Load user profiles for a category."""
    profiles_path = profiles_dir / f"{category}_user_profiles.json"
    if profiles_path.exists():
        try:
            with open(profiles_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate problematic dialogues")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--csv", type=str,
                        default="./AgentWorkflow/problematic_dialogues.csv",
                        help="Path to problematic dialogues CSV")
    parser.add_argument("--dialogue-dir", type=str,
                        default="./AgentWorkflow/dialogue_outputs",
                        help="Base directory for dialogue outputs")
    parser.add_argument("--eval-dir", type=str,
                        default="./eval_results",
                        help="Base directory for eval results")
    parser.add_argument("--profiles-dir", type=str,
                        default="./user_profiles",
                        help="Directory for user profiles")
    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent
    csv_path = base_dir / args.csv if not os.path.isabs(args.csv) else Path(args.csv)
    dialogue_base = base_dir / args.dialogue_dir if not os.path.isabs(args.dialogue_dir) else Path(args.dialogue_dir)
    eval_base = base_dir / args.eval_dir if not os.path.isabs(args.eval_dir) else Path(args.eval_dir)
    profiles_dir = base_dir / args.profiles_dir if not os.path.isabs(args.profiles_dir) else Path(args.profiles_dir)

    print("="*60)
    print("RE-EVALUATE PROBLEMATIC DIALOGUES")
    print("="*60)
    print(f"CSV file: {csv_path}")
    print(f"Dialogue base: {dialogue_base}")
    print(f"Eval base: {eval_base}")
    print(f"Profiles dir: {profiles_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Load problematic entries
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    entries = load_problematic_dialogues(str(csv_path))
    print(f"Loaded {len(entries)} problematic dialogues to re-evaluate")
    print()

    # Process each entry
    success_count = 0
    error_count = 0

    for i, entry in enumerate(entries, 1):
        method = entry["method"]
        category = entry["category"]
        dialogue_id = entry["dialogue_id"]

        print(f"[{i}/{len(entries)}] {method}/{category}/{dialogue_id}")

        # Get directory mappings
        if method not in METHOD_MAPPING:
            print(f"  SKIP: Unknown method '{method}'")
            error_count += 1
            continue

        dialogue_subdir, eval_subdir = METHOD_MAPPING[method]
        dialogue_dir = dialogue_base / dialogue_subdir
        eval_path = eval_base / eval_subdir / f"{category}_eval.json"

        # Find the updated dialogue
        dialogue_entry = find_dialogue_in_category(dialogue_dir, category, dialogue_id)
        if not dialogue_entry:
            print(f"  ERROR: Dialogue not found in {dialogue_dir / category}")
            error_count += 1
            continue

        print(f"  Found dialogue with {len(dialogue_entry.get('dialogue_history', []))} turns")

        if args.dry_run:
            print(f"  DRY RUN: Would evaluate and update {eval_path}")
            success_count += 1
            continue

        # Load user profiles
        loaded_profiles = load_user_profiles(profiles_dir, category)

        # Evaluate the dialogue
        print(f"  Evaluating...")
        new_eval = evaluate_single_entry(dialogue_entry, loaded_profiles)

        if "error" in new_eval:
            print(f"  ERROR: Evaluation failed: {new_eval['error']}")
            error_count += 1
            continue

        # Update the eval results
        if update_eval_results(eval_path, dialogue_id, new_eval):
            overall_score = new_eval.get("evaluation", {}).get("OverallScore", "N/A")
            print(f"  SUCCESS: Updated eval (OverallScore: {overall_score})")
            success_count += 1
        else:
            error_count += 1

    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total entries: {len(entries)}")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
