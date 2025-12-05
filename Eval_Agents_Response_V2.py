"""
evaluate_agent_aggregate_performance.py
-----------------------------------
LLM-as-a-Judge scoring for Shopping Agents.
Focus: WHOLE DIALOGUE AGGREGATE ASSESSMENT.
Evaluates the Agent's cumulative responses, ignoring User quality.

Supports flexible A vs B comparison (e.g., multimodal vs text-only, RAG vs baseline, etc.)
"""

import json
import os
import argparse
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------------
# Configuration
# -----------------------------------
MODEL_NAME = "openai/gpt-4o"
TEMPERATURE = 0.0
MAX_WORKERS = 10

# Base directory paths for all-categories mode (relative to script location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(BASE_DIR, "AgentWorkflow/baseline_dialogues_output")
RAG_DIR = os.path.join(BASE_DIR, "AgentWorkflow/dialogues_output")
PROFILES_DIR = os.path.join(BASE_DIR, "user_profiles")

# Default labels for comparison (can be overridden via CLI)
DEFAULT_LABEL_A = "baseline"
DEFAULT_LABEL_B = "variant"

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

def load_dialogues_from_directory(dir_path):
    """
    Scan directory for dialogue JSON files and automatically identify baseline vs RAG.

    Args:
        dir_path: Directory containing dialogue JSON files

    Returns:
        List of dialogue entries with rag_enabled field added
    """
    import glob

    if not os.path.isdir(dir_path):
        raise ValueError(f"Input path is not a directory: {dir_path}")

    # Find all JSON files in directory
    json_files = glob.glob(os.path.join(dir_path, "*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory: {dir_path}")

    all_dialogues = []
    baseline_count = 0
    rag_count = 0

    for file_path in json_files:
        filename = os.path.basename(file_path).lower()

        # Determine if this is baseline or RAG based on filename
        is_baseline = "baseline" in filename
        is_rag = ("eval_dialogues" in filename or "text_rag_dialogues" in filename) and not is_baseline

        # Skip files that don't match expected patterns
        if not (is_baseline or is_rag):
            print(f"⚠️  Skipping unrecognized file: {os.path.basename(file_path)}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                dialogues = json.load(f)

            # Handle both list and single dialogue formats
            if isinstance(dialogues, dict):
                dialogues = [dialogues]

            # Add rag_enabled field to each dialogue
            for dialogue in dialogues:
                dialogue["rag_enabled"] = is_rag
                dialogue["variant"] = "rag" if is_rag else "baseline"  # Generic variant field
                all_dialogues.append(dialogue)

            if is_baseline:
                baseline_count += len(dialogues)
            else:
                rag_count += len(dialogues)

            print(f"✓ Loaded {len(dialogues)} dialogues from {os.path.basename(file_path)} ({'RAG' if is_rag else 'Baseline'})")

        except Exception as e:
            print(f"✗ Error loading {os.path.basename(file_path)}: {str(e)}")
            continue

    print(f"\n📊 Total loaded: {len(all_dialogues)} dialogues")
    print(f"   - Baseline (no RAG): {baseline_count}")
    print(f"   - RAG enabled: {rag_count}")

    return all_dialogues


def load_dialogues_from_dual_dirs(dir_a, dir_b, label_a, label_b):
    """
    Load dialogues from two separate directories with custom labels.

    Args:
        dir_a: First directory path
        dir_b: Second directory path
        label_a: Label for first directory (e.g., "text_only", "baseline")
        label_b: Label for second directory (e.g., "multimodal", "rag")

    Returns:
        Tuple of (all_dialogues, count_a, count_b)
    """
    import glob

    all_dialogues = []
    count_a = 0
    count_b = 0

    for dir_path, label in [(dir_a, label_a), (dir_b, label_b)]:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Directory not found: {dir_path}")

        json_files = glob.glob(os.path.join(dir_path, "*.json"))

        if not json_files:
            print(f"⚠️  No JSON files found in: {dir_path}")
            continue

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    dialogues = json.load(f)

                if isinstance(dialogues, dict):
                    dialogues = [dialogues]

                for dialogue in dialogues:
                    dialogue["variant"] = label
                    # Keep rag_enabled for backward compatibility
                    dialogue["rag_enabled"] = (label == label_b)
                    all_dialogues.append(dialogue)

                if label == label_a:
                    count_a += len(dialogues)
                else:
                    count_b += len(dialogues)

                print(f"✓ Loaded {len(dialogues)} dialogues from {os.path.basename(file_path)} [{label}]")

            except Exception as e:
                print(f"✗ Error loading {os.path.basename(file_path)}: {str(e)}")
                continue

    print(f"\n📊 Total loaded: {len(all_dialogues)} dialogues")
    print(f"   - {label_a}: {count_a}")
    print(f"   - {label_b}: {count_b}")

    return all_dialogues, count_a, count_b

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
            # We add an index to help the judge pinpoint where an error occurred
            agent_turns.append(f"[Response {i}]: {turn.get('content', '')}")
            
    if not agent_turns:
        return "No Agent Responses Found."
    
    return "\n\n".join(agent_turns)

def get_user_profile(entry, loaded_profiles=None):
    """
    Get user profile from loaded profiles file or fallback to entry's user_profile field.

    Args:
        entry: The dialogue entry
        loaded_profiles: Dict of user_id -> profile (from user_profiles.json)

    Returns:
        Formatted profile string
    """
    # Try to get from loaded profiles first
    if loaded_profiles:
        # Try to get user_id from entry (could be "user_id" field or extracted from "dialogue_id")
        user_id = entry.get("user_id")
        if not user_id and "dialogue_id" in entry:
            # Try extracting from dialogue_id (e.g., "USER123_session1" -> "USER123")
            dialogue_id = entry.get("dialogue_id", "")
            user_id = dialogue_id.split("_")[0] if "_" in dialogue_id else dialogue_id

        if user_id and user_id in loaded_profiles:
            profile = loaded_profiles[user_id]
            if profile:
                return ", ".join([f"{k}: {v}" for k, v in profile.items()])

    # Fallback to entry's user_profile field
    profile = entry.get("user_profile", {})
    if not profile:
        return "No Profile Available."
    return ", ".join([f"{k}: {v}" for k, v in profile.items()])


def load_category_dialogues(category):
    """
    Load dialogues for a specific category from both baseline and RAG directories.

    Args:
        category: Category name (e.g., "Electronics", "Books")

    Returns:
        Tuple of (dialogues list, loaded_profiles dict)
    """
    import glob

    dialogues = []
    baseline_count = 0
    rag_count = 0

    # Load baseline dialogues
    baseline_category_dir = os.path.join(BASELINE_DIR, category)
    if os.path.isdir(baseline_category_dir):
        baseline_files = glob.glob(os.path.join(baseline_category_dir, "baseline_dialogues_*.json"))
        for file_path in baseline_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                for dialogue in data:
                    dialogue["rag_enabled"] = False
                    dialogue["category"] = category
                    dialogues.append(dialogue)
                baseline_count += len(data)
            except Exception as e:
                print(f"✗ Error loading {os.path.basename(file_path)}: {str(e)}")

    # Load RAG dialogues
    rag_category_dir = os.path.join(RAG_DIR, category)
    if os.path.isdir(rag_category_dir):
        rag_files = glob.glob(os.path.join(rag_category_dir, "text_rag_dialogues_*.json"))
        for file_path in rag_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                for dialogue in data:
                    dialogue["rag_enabled"] = True
                    dialogue["category"] = category
                    dialogues.append(dialogue)
                rag_count += len(data)
            except Exception as e:
                print(f"✗ Error loading {os.path.basename(file_path)}: {str(e)}")

    # Load user profiles for this category
    profiles_path = os.path.join(PROFILES_DIR, f"{category}_user_profiles.json")
    loaded_profiles = None
    if os.path.exists(profiles_path):
        try:
            with open(profiles_path, "r", encoding="utf-8") as f:
                loaded_profiles = json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading profiles for {category}: {str(e)}")

    return dialogues, loaded_profiles, baseline_count, rag_count


def batch_evaluate_category(category, output_dir):
    """
    Evaluate all dialogues for a single category.

    Args:
        category: Category name
        output_dir: Directory to save results

    Returns:
        Dict with evaluation results and statistics
    """
    print(f"\n{'='*60}")
    print(f"📂 Processing category: {category}")
    print(f"{'='*60}")

    dialogues, loaded_profiles, baseline_count, rag_count = load_category_dialogues(category)

    if not dialogues:
        print(f"⚠️  No dialogues found for {category}")
        return None

    print(f"✓ Loaded {len(dialogues)} dialogues (Baseline: {baseline_count}, RAG: {rag_count})")
    if loaded_profiles:
        print(f"✓ Loaded {len(loaded_profiles)} user profiles")

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

    # Generate paired comparison
    paired_stats = generate_paired_comparison(results)
    if "error" not in paired_stats:
        paired_output_path = os.path.join(output_dir, f"{category}_paired.json")
        with open(paired_output_path, "w", encoding="utf-8") as f:
            json.dump(paired_stats, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved paired comparison to: {paired_output_path}")

    # Print category summary
    print_aggregate_report(results)

    return {
        "category": category,
        "total_dialogues": len(dialogues),
        "baseline_count": baseline_count,
        "rag_count": rag_count,
        "results": results,
        "paired_stats": paired_stats
    }


def batch_evaluate_all_categories(output_dir="eval_results"):
    """
    Evaluate all categories and save per-category results.

    Args:
        output_dir: Directory to save all results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Discover all categories from baseline directory
    categories = []
    if os.path.isdir(BASELINE_DIR):
        categories = [d for d in os.listdir(BASELINE_DIR)
                      if os.path.isdir(os.path.join(BASELINE_DIR, d))]
    categories.sort()

    print(f"\n{'='*60}")
    print(f"🚀 BATCH EVALUATION - ALL CATEGORIES")
    print(f"{'='*60}")
    print(f"Found {len(categories)} categories to process")
    print(f"Output directory: {output_dir}")
    print(f"Max workers: {MAX_WORKERS}")

    all_results = []
    category_summaries = []

    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] ", end="")
        result = batch_evaluate_category(category, output_dir)
        if result:
            all_results.extend(result["results"])
            ps = result["paired_stats"]
            if "error" not in ps:
                category_summaries.append({
                    "category": result["category"],
                    "total_dialogues": result["total_dialogues"],
                    "baseline_count": result["baseline_count"],
                    "rag_count": result["rag_count"],
                    "paired_stats_summary": {
                        "total_pairs": ps.get("total_pairs", 0),
                        "rag": ps.get("rag", {}),
                        "baseline": ps.get("baseline", {}),
                        "rag_better_count": ps.get("rag_better_count", 0),
                        "baseline_better_count": ps.get("baseline_better_count", 0),
                    }
                })
            else:
                category_summaries.append({
                    "category": result["category"],
                    "total_dialogues": result["total_dialogues"],
                    "baseline_count": result["baseline_count"],
                    "rag_count": result["rag_count"],
                    "paired_stats_summary": {"error": ps["error"]}
                })

    # Save overall summary
    summary_path = os.path.join(output_dir, "all_categories_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(category_summaries, f, ensure_ascii=False, indent=2)

    # Print final summary
    print(f"\n{'='*60}")
    print(f"📊 FINAL SUMMARY - ALL CATEGORIES")
    print(f"{'='*60}")
    print(f"Total categories processed: {len(category_summaries)}")
    print(f"Total dialogues evaluated: {len(all_results)}")

    total_rag_better = sum(s["paired_stats_summary"].get("rag_better_count", 0) for s in category_summaries)
    total_baseline_better = sum(s["paired_stats_summary"].get("baseline_better_count", 0) for s in category_summaries)
    total_pairs = sum(s["paired_stats_summary"].get("total_pairs", 0) for s in category_summaries)

    if total_pairs > 0:
        print(f"\n🏆 Overall Win/Loss (across all categories):")
        print(f"   RAG Better:      {total_rag_better:3d} ({total_rag_better/total_pairs*100:.1f}%)")
        print(f"   Baseline Better: {total_baseline_better:3d} ({total_baseline_better/total_pairs*100:.1f}%)")

    print(f"\n💾 Summary saved to: {summary_path}")


def evaluate_single_entry(entry, loaded_profiles=None):
    # 1. Prepare Context (User + Agent)
    full_dialogue = format_full_dialogue(entry.get("dialogue_history", []))

    # 2. Prepare Target (Agent Only)
    agent_stream = extract_agent_stream(entry.get("dialogue_history", []))

    user_goal = entry.get("user_goal", "Shopping")
    context_snippet = str(entry.get("retrieved_context", "No Context"))
    user_profile = get_user_profile(entry, loaded_profiles)

    full_prompt = EVALUATION_PROMPT.format(
        user_profile=user_profile,
        context_snippet=context_snippet,
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
            "variant": entry.get("variant", "unknown"),
            "rag_enabled": entry.get("rag_enabled", False),  # Backward compatibility
            "evaluation": result_json
        }
    except Exception as e:
        return {"dialogue_id": entry.get("dialogue_id"), "variant": entry.get("variant"), "error": str(e)}

def batch_evaluate_dual(dir_a, dir_b, label_a, label_b, output_dir, profiles_path=None):
    """
    Evaluate dialogues from two directories with custom labels.

    Args:
        dir_a: First directory path
        dir_b: Second directory path
        label_a: Label for first directory
        label_b: Label for second directory
        output_dir: Output directory for results
        profiles_path: Optional path to user profiles JSON
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 DUAL DIRECTORY EVALUATION")
    print(f"{'='*60}")
    print(f"Directory A ({label_a}): {dir_a}")
    print(f"Directory B ({label_b}): {dir_b}")
    print(f"Output: {output_dir}")

    # Load dialogues from both directories
    dialogues, count_a, count_b = load_dialogues_from_dual_dirs(dir_a, dir_b, label_a, label_b)

    if not dialogues:
        print("❌ No dialogues loaded!")
        return

    # Load user profiles if provided
    loaded_profiles = None
    if profiles_path and os.path.exists(profiles_path):
        with open(profiles_path, "r", encoding="utf-8") as f:
            loaded_profiles = json.load(f)
        print(f"✓ Loaded {len(loaded_profiles)} user profiles")

    print(f"\n🚀 Evaluating {len(dialogues)} dialogues...")

    # Evaluate with thread pool
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(evaluate_single_entry, d, loaded_profiles) for d in dialogues]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            results.append(future.result())
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(dialogues)} dialogues...")

    # Save all results
    all_results_path = os.path.join(output_dir, "all_eval_results.json")
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 All results saved to: {all_results_path}")

    # Print aggregate report
    print_aggregate_report_dual(results, label_a, label_b)

    # Generate and print paired comparison
    paired_stats = generate_paired_comparison(results, label_a, label_b)
    paired_output_path = os.path.join(output_dir, "paired_comparison.json")
    print_paired_comparison_report(paired_stats, paired_output_path)

    return results


def print_aggregate_report_dual(results, label_a, label_b):
    """Print aggregate report with custom labels."""
    stats = {
        label_a: {"consistency": [], "flaws": {}},
        label_b: {"consistency": [], "flaws": {}}
    }

    for r in results:
        if "error" in r:
            continue
        variant = r.get("variant", label_a)
        if variant not in stats:
            continue
        eval_data = r["evaluation"]

        stats[variant]["consistency"].append(eval_data["GlobalPersonalizationConsistency"]["score"])

        for flaw in eval_data.get("Major_Flaws_Detected", []):
            if flaw != "None":
                stats[variant]["flaws"][flaw] = stats[variant]["flaws"].get(flaw, 0) + 1

    print("\n📈 AGGREGATE PERFORMANCE REPORT 📈")
    print("-----------------------------------")
    for key in [label_a, label_b]:
        cons = stats[key]["consistency"]
        count = len(cons)
        if count == 0:
            continue

        print(f"\nConfiguration: {key.upper()}")
        print(f"  Samples: {count}")
        print(f"  🧠 Global Consistency: {sum(cons)/count:.2f} / 5.0")

        print("  🚩 Common Flaws (Whole Dialogue):")
        sorted_flaws = sorted(stats[key]["flaws"].items(), key=lambda x: x[1], reverse=True)
        if not sorted_flaws:
            print("      - None detected!")
        for f, c in sorted_flaws[:3]:
            print(f"      - {f}: {c} dialogues")


def batch_evaluate(input_path, output_path, profiles_path=None, paired_output_path=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Check if input is directory or file
    if os.path.isdir(input_path):
        print("📂 Scanning directory for dialogue files...")
        dialogues = load_dialogues_from_directory(input_path)
    else:
        # Fallback to single file mode
        print("📄 Loading single dialogue file...")
        with open(input_path, "r", encoding="utf-8") as f:
            dialogues = json.load(f)
        # Ensure rag_enabled field exists
        if isinstance(dialogues, list):
            for d in dialogues:
                if "rag_enabled" not in d:
                    d["rag_enabled"] = False  # Default to baseline
        print(f"✓ Loaded {len(dialogues)} dialogues")

    # Load user profiles if provided
    loaded_profiles = None
    if profiles_path:
        if not os.path.exists(profiles_path):
            print(f"⚠️  Warning: Profiles file not found: {profiles_path}")
            print("   Falling back to user_profile field in dialogues")
        else:
            with open(profiles_path, "r", encoding="utf-8") as f:
                loaded_profiles = json.load(f)
            print(f"✓ Loaded {len(loaded_profiles)} user profiles from {profiles_path}")

    print(f"\n🚀 Evaluating AGGREGATE performance for {len(dialogues)} dialogues...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(evaluate_single_entry, d, loaded_profiles) for d in dialogues]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Save evaluation results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Evaluation results saved to: {output_path}")

    # Print aggregate report
    print_aggregate_report(results)

    # Generate and print paired comparison if requested
    if paired_output_path:
        paired_stats = generate_paired_comparison(results)
        print_paired_comparison_report(paired_stats, paired_output_path)
    else:
        # Still generate paired comparison but don't save
        paired_stats = generate_paired_comparison(results)
        if "error" not in paired_stats:
            print_paired_comparison_report(paired_stats)

def print_aggregate_report(results):
    """Compares RAG vs No-RAG on long-context consistency."""
    stats = {
        "rag": {"consistency": [], "flaws": {}},
        "no_rag": {"consistency": [], "flaws": {}}
    }

    for r in results:
        if "error" in r: continue
        key = "rag" if r.get("rag_enabled") else "no_rag"
        eval_data = r["evaluation"]

        stats[key]["consistency"].append(eval_data["GlobalPersonalizationConsistency"]["score"])

        for flaw in eval_data.get("Major_Flaws_Detected", []):
            if flaw != "None":
                stats[key]["flaws"][flaw] = stats[key]["flaws"].get(flaw, 0) + 1

    print("\n📈 AGGREGATE PERFORMANCE REPORT 📈")
    print("-----------------------------------")
    for key in ["rag", "no_rag"]:
        cons = stats[key]["consistency"]
        count = len(cons)
        if count == 0: continue

        print(f"\nConfiguration: {key.upper()}")
        print(f"  Samples: {count}")
        print(f"  🧠 Global Consistency: {sum(cons)/count:.2f} / 5.0")

        print("  🚩 Common Flaws (Whole Dialogue):")
        sorted_flaws = sorted(stats[key]["flaws"].items(), key=lambda x: x[1], reverse=True)
        if not sorted_flaws:
            print("      - None detected!")
        for f, c in sorted_flaws[:3]:
            print(f"      - {f}: {c} dialogues")

def generate_paired_comparison(results, label_a="baseline", label_b="rag"):
    """
    Generate paired comparison between two variants for same dialogue_id.

    Args:
        results: List of evaluation results
        label_a: Label for first variant (default: "baseline")
        label_b: Label for second variant (default: "rag")

    Returns:
        Dict with paired comparison statistics
    """
    # Group by dialogue_id
    grouped = {}
    for r in results:
        if "error" in r:
            continue
        dialogue_id = r.get("dialogue_id")
        variant = r.get("variant", "rag" if r.get("rag_enabled", False) else "baseline")

        if dialogue_id not in grouped:
            grouped[dialogue_id] = {}

        grouped[dialogue_id][variant] = r["evaluation"]

    # Calculate paired differences
    pairs = []
    for dialogue_id, data in grouped.items():
        if label_a in data and label_b in data:
            eval_a = data[label_a]
            eval_b = data[label_b]

            pair_diff = {
                "dialogue_id": dialogue_id,
                "consistency_diff": eval_b["GlobalPersonalizationConsistency"]["score"] - eval_a["GlobalPersonalizationConsistency"]["score"],
                "synthesis_diff": eval_b["CumulativeInformationSynthesis"]["score"] - eval_a["CumulativeInformationSynthesis"]["score"],
                "trajectory_diff": eval_b["InteractionTrajectory"]["score"] - eval_a["InteractionTrajectory"]["score"],
                "tone_diff": eval_b["ToneConsistency"]["score"] - eval_a["ToneConsistency"]["score"],
                "overall_diff": eval_b.get("OverallScore", 0) - eval_a.get("OverallScore", 0),
                f"{label_b}_scores": eval_b,
                f"{label_a}_scores": eval_a
            }
            pairs.append(pair_diff)

    if not pairs:
        return {"error": "No paired dialogues found", "label_a": label_a, "label_b": label_b}

    # Calculate separate averages for both variants
    n = len(pairs)
    stats = {
        "total_pairs": n,
        "label_a": label_a,
        "label_b": label_b,
        label_b: {
            "avg_consistency": sum(p[f"{label_b}_scores"]["GlobalPersonalizationConsistency"]["score"] for p in pairs) / n,
            "avg_synthesis": sum(p[f"{label_b}_scores"]["CumulativeInformationSynthesis"]["score"] for p in pairs) / n,
            "avg_trajectory": sum(p[f"{label_b}_scores"]["InteractionTrajectory"]["score"] for p in pairs) / n,
            "avg_tone": sum(p[f"{label_b}_scores"]["ToneConsistency"]["score"] for p in pairs) / n,
            "avg_overall": sum(p[f"{label_b}_scores"].get("OverallScore", 0) for p in pairs) / n
        },
        label_a: {
            "avg_consistency": sum(p[f"{label_a}_scores"]["GlobalPersonalizationConsistency"]["score"] for p in pairs) / n,
            "avg_synthesis": sum(p[f"{label_a}_scores"]["CumulativeInformationSynthesis"]["score"] for p in pairs) / n,
            "avg_trajectory": sum(p[f"{label_a}_scores"]["InteractionTrajectory"]["score"] for p in pairs) / n,
            "avg_tone": sum(p[f"{label_a}_scores"]["ToneConsistency"]["score"] for p in pairs) / n,
            "avg_overall": sum(p[f"{label_a}_scores"].get("OverallScore", 0) for p in pairs) / n
        },
        f"{label_b}_better_count": sum(1 for p in pairs if p["overall_diff"] > 0.1),
        f"{label_a}_better_count": sum(1 for p in pairs if p["overall_diff"] < -0.1),
        "tie_count": sum(1 for p in pairs if abs(p["overall_diff"]) <= 0.1),
        "pairs": pairs
    }

    # Keep backward compatible keys if using default labels
    if label_a == "baseline" and label_b == "rag":
        stats["rag_better_count"] = stats[f"{label_b}_better_count"]
        stats["baseline_better_count"] = stats[f"{label_a}_better_count"]
        stats["rag"] = stats[label_b]
        stats["baseline"] = stats[label_a]

    return stats

def print_paired_comparison_report(paired_stats, output_path=None):
    """
    Print and optionally save paired comparison report.

    Args:
        paired_stats: Output from generate_paired_comparison
        output_path: Optional path to save detailed paired results
    """
    if "error" in paired_stats:
        print(f"\n⚠️  {paired_stats['error']}")
        return

    # Get labels (with backward compatibility)
    label_a = paired_stats.get("label_a", "baseline")
    label_b = paired_stats.get("label_b", "rag")

    print("\n" + "="*60)
    print(f"📊 PAIRED COMPARISON REPORT ({label_b.upper()} vs {label_a.upper()})")
    print("="*60)

    total = paired_stats["total_pairs"]
    print(f"\n🔗 Total Paired Dialogues: {total}")

    variant_b = paired_stats.get(label_b, paired_stats.get("rag", {}))
    variant_a = paired_stats.get(label_a, paired_stats.get("baseline", {}))

    print(f"\n📈 {label_b.upper()} Average Scores:")
    print(f"   Global Consistency:     {variant_b['avg_consistency']:.2f}")
    print(f"   Information Synthesis:  {variant_b['avg_synthesis']:.2f}")
    print(f"   Interaction Trajectory: {variant_b['avg_trajectory']:.2f}")
    print(f"   Tone Consistency:       {variant_b['avg_tone']:.2f}")
    print(f"   Overall Score:          {variant_b['avg_overall']:.2f}")

    print(f"\n📉 {label_a.upper()} Average Scores:")
    print(f"   Global Consistency:     {variant_a['avg_consistency']:.2f}")
    print(f"   Information Synthesis:  {variant_a['avg_synthesis']:.2f}")
    print(f"   Interaction Trajectory: {variant_a['avg_trajectory']:.2f}")
    print(f"   Tone Consistency:       {variant_a['avg_tone']:.2f}")
    print(f"   Overall Score:          {variant_a['avg_overall']:.2f}")

    b_better = paired_stats.get(f"{label_b}_better_count", paired_stats.get("rag_better_count", 0))
    a_better = paired_stats.get(f"{label_a}_better_count", paired_stats.get("baseline_better_count", 0))
    tie = paired_stats["tie_count"]

    print(f"\n🏆 Win/Loss Breakdown:")
    print(f"   {label_b.upper()} Better:      {b_better:3d} ({b_better/total*100:.1f}%)")
    print(f"   {label_a.upper()} Better: {a_better:3d} ({a_better/total*100:.1f}%)")
    print(f"   Tie (±0.1):      {tie:3d} ({tie/total*100:.1f}%)")

    # Show top improvements and regressions
    pairs_sorted = sorted(paired_stats["pairs"], key=lambda x: x["overall_diff"], reverse=True)

    print(f"\n⬆️  Top 5 {label_b.upper()} Improvements:")
    for i, pair in enumerate(pairs_sorted[:5], 1):
        if pair["overall_diff"] > 0:
            print(f"   {i}. {pair['dialogue_id'][:20]}... : +{pair['overall_diff']:.2f}")

    print(f"\n⬇️  Top 5 {label_b.upper()} Regressions:")
    for i, pair in enumerate(reversed(pairs_sorted[-5:]), 1):
        if pair["overall_diff"] < 0:
            print(f"   {i}. {pair['dialogue_id'][:20]}... : {pair['overall_diff']:.2f}")

    # Save detailed paired results if requested
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(paired_stats, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Detailed paired comparison saved to: {output_path}")

# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AI Shopping Assistant dialogues with flexible A vs B comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two directories with custom labels (multimodal vs text-only)
  python %(prog)s --dir-a /path/to/text_only --dir-b /path/to/multimodal \\
                  --label-a text_only --label-b multimodal --output-dir results

  # Compare RAG vs baseline (using default labels)
  python %(prog)s --dir-a /path/to/baseline --dir-b /path/to/rag --output-dir results

  # Legacy: evaluate all categories from standard directory structure
  python %(prog)s --all --output-dir eval_results

  # Legacy: single input directory
  python %(prog)s --input /path/to/dialogues --output results.json
"""
    )

    # New dual-directory mode
    parser.add_argument(
        "--dir-a",
        type=str,
        default=None,
        help="First directory containing dialogue JSON files"
    )
    parser.add_argument(
        "--dir-b",
        type=str,
        default=None,
        help="Second directory containing dialogue JSON files"
    )
    parser.add_argument(
        "--label-a",
        type=str,
        default=DEFAULT_LABEL_A,
        help=f"Label for first directory (default: {DEFAULT_LABEL_A})"
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default=DEFAULT_LABEL_B,
        help=f"Label for second directory (default: {DEFAULT_LABEL_B})"
    )

    # Legacy modes
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all categories using standard directory structure"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for results (default: eval_results)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="[Legacy] Directory containing dialogue JSON files (auto-detects baseline/RAG) or single JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aggregate_eval.json",
        help="[Legacy] Output file for evaluation results (default: aggregate_eval.json)"
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=None,
        help="User profiles JSON file (optional)"
    )
    parser.add_argument(
        "--paired-output",
        type=str,
        default=None,
        help="[Legacy] Output file for paired comparison results"
    )
    args = parser.parse_args()

    # Determine which mode to use
    if args.dir_a and args.dir_b:
        # New dual-directory mode
        batch_evaluate_dual(
            args.dir_a,
            args.dir_b,
            args.label_a,
            args.label_b,
            args.output_dir,
            args.profiles
        )
    elif args.all:
        # All-categories mode
        batch_evaluate_all_categories(args.output_dir)
    elif args.input:
        # Legacy single input mode
        batch_evaluate(args.input, args.output, args.profiles, args.paired_output)
    else:
        parser.error("Must specify either --dir-a and --dir-b, --all, or --input")