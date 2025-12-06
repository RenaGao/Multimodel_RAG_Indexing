"""
Eval_Pairwise_Comparison.py
-----------------------------------
Pairwise LLM-as-a-Judge evaluation for Shopping Agents.
Compares two dialogues (A vs B) with the same user_id and category,
letting LLM directly judge which is better.

Usage:
    python Eval_Pairwise_Comparison.py \
        --input-a multimodal_dialogues_output \
        --input-b multimdoal_no_user_review_output \
        --label-a multimodal \
        --label-b no_user_review \
        --output-dir pairwise_eval_results
"""

import json
import os
import argparse
import glob
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv

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
# Pairwise Evaluation Prompt
# -----------------------------------
PAIRWISE_PROMPT = """
You are a Lead Auditor evaluating TWO AI Shopping Assistants (A and B) on the SAME user query.
Compare them side-by-side and determine which performed better.
Do NOT evaluate the User. Use the User's turns only to determine if the Agent's responses were relevant.

### User Profile & Goal
Profile: {user_profile}
Goal: {user_goal}

---

### Dialogue A ({label_a})
{dialogue_a_text}

---

### Dialogue B ({label_b})
{dialogue_b_text}

---

### Comparison Criteria (Pick Winner: A, B, or Tie for each)

#### 1. Global Personalization Consistency (GPC)
*Which agent better maintained the user's persona constraints throughout the ENTIRE dialogue?*
- Consider: Did the agent remember budget, size, style preferences consistently?
- Consider: Did the agent hallucinate user details or "reset" context?

#### 2. Cumulative Information Synthesis (CIS)
*Which agent built a more complete, factual, and coherent picture across all responses?*
- Consider: Did responses build on each other or repeat/contradict?
- Consider: Did the agent connect facts meaningfully?

#### 3. Interaction Trajectory (IT)
*Which agent's responses more purposefully drove the conversation to a conclusion?*
- Consider: Logical arc (Clarify -> Recommend -> Refine -> Close) vs circular/passive behavior

#### 4. Tone Consistency (TC)
*Which agent maintained a more stable, professional voice throughout?*
- Consider: Consistent persona vs erratic switches between formal/casual

---

### Output Format (JSON ONLY)
{{
  "GlobalPersonalizationConsistency": {{ "winner": "A" or "B" or "Tie", "reason": "string" }},
  "CumulativeInformationSynthesis": {{ "winner": "A" or "B" or "Tie", "reason": "string" }},
  "InteractionTrajectory": {{ "winner": "A" or "B" or "Tie", "reason": "string" }},
  "ToneConsistency": {{ "winner": "A" or "B" or "Tie", "reason": "string" }},
  "OverallWinner": "A" or "B" or "Tie",
  "OverallReason": "string"
}}

Important: Be objective and fair. If both are truly equal on a criterion, say "Tie".
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
        Dict of dialogue_id -> dialogue entry
    """
    dialogues = {}

    # Find all JSON files (ignore .jsonl backup files)
    json_files = glob.glob(os.path.join(category_dir, "*.json"))

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for entry in data:
                dialogue_id = entry.get("dialogue_id")
                if dialogue_id:
                    dialogues[dialogue_id] = entry
        except Exception as e:
            print(f"✗ Error loading {os.path.basename(file_path)}: {str(e)}")

    return dialogues


def match_dialogues_by_id(dialogues_a, dialogues_b):
    """
    Match dialogues from A and B by dialogue_id.

    Returns:
        List of (dialogue_id, entry_a, entry_b) tuples
    """
    common_ids = set(dialogues_a.keys()) & set(dialogues_b.keys())
    pairs = []
    for dialogue_id in sorted(common_ids):
        pairs.append((dialogue_id, dialogues_a[dialogue_id], dialogues_b[dialogue_id]))
    return pairs


def evaluate_pair(dialogue_id, entry_a, entry_b, label_a, label_b, loaded_profiles=None):
    """Evaluate a single pair of dialogues using LLM-as-a-Judge."""
    dialogue_a_text = format_full_dialogue(entry_a.get("dialogue_history", []))
    dialogue_b_text = format_full_dialogue(entry_b.get("dialogue_history", []))

    # Use profile/goal from entry_a (they should be the same for both)
    user_goal = entry_a.get("user_goal", "Shopping")
    user_profile = get_user_profile(entry_a, loaded_profiles)

    full_prompt = PAIRWISE_PROMPT.format(
        user_profile=user_profile,
        user_goal=user_goal,
        label_a=label_a,
        label_b=label_b,
        dialogue_a_text=dialogue_a_text,
        dialogue_b_text=dialogue_b_text
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict auditor comparing two AI Agents."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )
        result_json = json.loads(response.choices[0].message.content)

        return {
            "dialogue_id": dialogue_id,
            "winner": result_json.get("OverallWinner", "Tie"),
            "comparison": {
                "GlobalPersonalizationConsistency": result_json.get("GlobalPersonalizationConsistency", {}),
                "CumulativeInformationSynthesis": result_json.get("CumulativeInformationSynthesis", {}),
                "InteractionTrajectory": result_json.get("InteractionTrajectory", {}),
                "ToneConsistency": result_json.get("ToneConsistency", {})
            },
            "overall_reason": result_json.get("OverallReason", "")
        }
    except Exception as e:
        return {"dialogue_id": dialogue_id, "error": str(e)}


def compute_category_stats(results, label_a, label_b):
    """Compute statistics for a category's results."""
    valid_results = [r for r in results if "error" not in r]
    n = len(valid_results)

    if n == 0:
        return None

    criteria = ["GlobalPersonalizationConsistency", "CumulativeInformationSynthesis",
                "InteractionTrajectory", "ToneConsistency"]

    # Overall counts
    a_wins = sum(1 for r in valid_results if r["winner"] == "A")
    b_wins = sum(1 for r in valid_results if r["winner"] == "B")
    ties = sum(1 for r in valid_results if r["winner"] == "Tie")

    # Per-criterion counts
    per_criterion = {}
    for criterion in criteria:
        c_a_wins = sum(1 for r in valid_results
                       if r["comparison"].get(criterion, {}).get("winner") == "A")
        c_b_wins = sum(1 for r in valid_results
                       if r["comparison"].get(criterion, {}).get("winner") == "B")
        c_ties = sum(1 for r in valid_results
                     if r["comparison"].get(criterion, {}).get("winner") == "Tie")
        per_criterion[criterion] = {"A_wins": c_a_wins, "B_wins": c_b_wins, "Tie": c_ties}

    return {
        "total_pairs": n,
        "A_wins": a_wins,
        "B_wins": b_wins,
        "Tie": ties,
        "per_criterion": per_criterion
    }


def evaluate_category_pairwise(category, dir_a, dir_b, output_dir, label_a, label_b, loaded_profiles=None):
    """
    Evaluate all dialogue pairs for a single category.
    """
    print(f"\n{'='*60}")
    print(f"📂 Processing category: {category}")
    print(f"{'='*60}")

    category_dir_a = os.path.join(dir_a, category)
    category_dir_b = os.path.join(dir_b, category)

    if not os.path.isdir(category_dir_a):
        print(f"⚠️  Category directory not found in A: {category_dir_a}")
        return None
    if not os.path.isdir(category_dir_b):
        print(f"⚠️  Category directory not found in B: {category_dir_b}")
        return None

    # Load dialogues
    dialogues_a = load_dialogues_from_category(category_dir_a)
    dialogues_b = load_dialogues_from_category(category_dir_b)

    print(f"  Loaded {len(dialogues_a)} dialogues from A ({label_a})")
    print(f"  Loaded {len(dialogues_b)} dialogues from B ({label_b})")

    # Match by dialogue_id
    pairs = match_dialogues_by_id(dialogues_a, dialogues_b)
    print(f"  Found {len(pairs)} matching dialogue pairs")

    if not pairs:
        print(f"⚠️  No matching dialogues found for {category}")
        return None

    # Evaluate pairs with thread pool
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(evaluate_pair, did, ea, eb, label_a, label_b, loaded_profiles)
            for did, ea, eb in pairs
        ]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            results.append(future.result())
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(pairs)} pairs...")

    # Save category results
    output_path = os.path.join(output_dir, f"{category}_pairwise_eval.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved results to: {output_path}")

    # Compute stats
    stats = compute_category_stats(results, label_a, label_b)
    if stats:
        stats["category"] = category

    return stats


def batch_pairwise_evaluate(input_a, input_b, label_a, label_b, output_dir, profiles_dir=None):
    """
    Evaluate all categories pairwise.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Discover all categories (subdirectories in both inputs)
    categories_a = set(d for d in os.listdir(input_a) if os.path.isdir(os.path.join(input_a, d)))
    categories_b = set(d for d in os.listdir(input_b) if os.path.isdir(os.path.join(input_b, d)))
    common_categories = sorted(categories_a & categories_b)

    if not common_categories:
        print(f"❌ No common category subdirectories found")
        print(f"  Categories in A: {sorted(categories_a)}")
        print(f"  Categories in B: {sorted(categories_b)}")
        return

    print(f"\n{'='*60}")
    print(f"🚀 PAIRWISE EVALUATION")
    print(f"{'='*60}")
    print(f"Input A ({label_a}): {input_a}")
    print(f"Input B ({label_b}): {input_b}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(common_categories)} common categories")
    print(f"Max workers: {MAX_WORKERS}")

    all_category_stats = []
    total_a_wins = 0
    total_b_wins = 0
    total_ties = 0
    total_pairs = 0

    # Initialize per-criterion totals
    criteria = ["GlobalPersonalizationConsistency", "CumulativeInformationSynthesis",
                "InteractionTrajectory", "ToneConsistency"]
    total_per_criterion = {c: {"A_wins": 0, "B_wins": 0, "Tie": 0} for c in criteria}

    for i, category in enumerate(common_categories, 1):
        print(f"\n[{i}/{len(common_categories)}] ", end="")

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

        stats = evaluate_category_pairwise(
            category, input_a, input_b, output_dir, label_a, label_b, loaded_profiles
        )

        if stats:
            all_category_stats.append(stats)
            total_a_wins += stats["A_wins"]
            total_b_wins += stats["B_wins"]
            total_ties += stats["Tie"]
            total_pairs += stats["total_pairs"]

            # Accumulate per-criterion stats
            for c in criteria:
                total_per_criterion[c]["A_wins"] += stats["per_criterion"][c]["A_wins"]
                total_per_criterion[c]["B_wins"] += stats["per_criterion"][c]["B_wins"]
                total_per_criterion[c]["Tie"] += stats["per_criterion"][c]["Tie"]

            # Print category summary
            print(f"\n  📊 {category} Summary:")
            print(f"     A ({label_a}) wins: {stats['A_wins']} ({stats['A_wins']/stats['total_pairs']*100:.1f}%)")
            print(f"     B ({label_b}) wins: {stats['B_wins']} ({stats['B_wins']/stats['total_pairs']*100:.1f}%)")
            print(f"     Tie: {stats['Tie']} ({stats['Tie']/stats['total_pairs']*100:.1f}%)")

    # Generate summary
    if total_pairs > 0:
        summary = {
            "total_categories": len(all_category_stats),
            "total_pairs": total_pairs,
            "label_a": label_a,
            "label_b": label_b,
            "A_wins": total_a_wins,
            "B_wins": total_b_wins,
            "Tie": total_ties,
            "A_win_rate": total_a_wins / total_pairs,
            "B_win_rate": total_b_wins / total_pairs,
            "Tie_rate": total_ties / total_pairs,
            "per_criterion": total_per_criterion,
            "per_category": all_category_stats
        }

        # Save summary
        summary_path = os.path.join(output_dir, "pairwise_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Print overall summary
        print(f"\n{'='*60}")
        print(f"📊 OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Total categories: {len(all_category_stats)}")
        print(f"Total pairs evaluated: {total_pairs}")
        print(f"\n🏆 Overall Results:")
        print(f"   A ({label_a}) wins: {total_a_wins:4d} ({total_a_wins/total_pairs*100:.1f}%)")
        print(f"   B ({label_b}) wins: {total_b_wins:4d} ({total_b_wins/total_pairs*100:.1f}%)")
        print(f"   Tie:                {total_ties:4d} ({total_ties/total_pairs*100:.1f}%)")

        print(f"\n📈 Per-Criterion Breakdown:")
        for c in criteria:
            pc = total_per_criterion[c]
            print(f"   {c}:")
            print(f"      A wins: {pc['A_wins']:3d}  B wins: {pc['B_wins']:3d}  Tie: {pc['Tie']:3d}")

        print(f"\n💾 Summary saved to: {summary_path}")


# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pairwise LLM-as-a-Judge evaluation for Shopping Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare multimodal vs no_user_review
  python %(prog)s \\
      --input-a multimodal_dialogues_output \\
      --input-b multimdoal_no_user_review_output \\
      --label-a multimodal \\
      --label-b no_user_review \\
      --output-dir pairwise_eval_results

  # With user profiles
  python %(prog)s \\
      --input-a dir_a \\
      --input-b dir_b \\
      --label-a variant_a \\
      --label-b variant_b \\
      --output-dir results \\
      --profiles-dir user_profiles
"""
    )
    parser.add_argument(
        "--input-a",
        type=str,
        required=True,
        help="Input directory A containing category subdirectories with dialogue files"
    )
    parser.add_argument(
        "--input-b",
        type=str,
        required=True,
        help="Input directory B containing category subdirectories with dialogue files"
    )
    parser.add_argument(
        "--label-a",
        type=str,
        required=True,
        help="Label for variant A (e.g., 'multimodal')"
    )
    parser.add_argument(
        "--label-b",
        type=str,
        required=True,
        help="Label for variant B (e.g., 'no_user_review')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for pairwise evaluation results"
    )
    parser.add_argument(
        "--profiles-dir",
        type=str,
        default='./user_profiles',
        help="Directory containing user profile JSON files (optional)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_a):
        raise ValueError(f"Input directory A not found: {args.input_a}")
    if not os.path.isdir(args.input_b):
        raise ValueError(f"Input directory B not found: {args.input_b}")

    batch_pairwise_evaluate(
        args.input_a, args.input_b,
        args.label_a, args.label_b,
        args.output_dir, args.profiles_dir
    )
