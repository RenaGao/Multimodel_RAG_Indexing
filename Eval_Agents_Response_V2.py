"""
evaluate_agent_aggregate_performance.py
-----------------------------------
LLM-as-a-Judge scoring for Shopping Agents.
Focus: WHOLE DIALOGUE AGGREGATE ASSESSMENT.
Evaluates the Agent's cumulative responses, ignoring User quality.
"""

import json
import os
import argparse
import concurrent.futures
from openai import OpenAI

# -----------------------------------
# Configuration
# -----------------------------------
MODEL_NAME = "gpt-4o" 
TEMPERATURE = 0.0 
MAX_WORKERS = 5

client = OpenAI()

# -----------------------------------
# Initialize OpenAI client
# -----------------------------------
api_key = os.getenv("OPENAI_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_KEY in environment variables. Please add it to your .env file.")

client = OpenAI(api_key=api_key)

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
  "AggregateHallucinationCheck": {{ "score": int, "reason": "string" }},
  "ToneConsistency": {{ "score": int, "reason": "string" }},
  "OverallScore": float,
  "Major_Flaws_Detected": ["list", "of", "flaws"]
}}

**Valid Flaws:**
["Forgot_Constraint_Mid_Chat", "Repetitive_Loops", "Hallucination", "Tone_Switch", "Passive_Behavior", "None"]
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
            "rag_enabled": entry.get("rag_enabled", False),
            "evaluation": result_json
        }
    except Exception as e:
        return {"dialogue_id": entry.get("dialogue_id"), "error": str(e)}

def batch_evaluate(input_path, output_path, profiles_path=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

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

    print(f"🚀 Evaluating AGGREGATE performance for {len(dialogues)} dialogues...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(evaluate_single_entry, d, loaded_profiles) for d in dialogues]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print_aggregate_report(results)

def print_aggregate_report(results):
    """Compares RAG vs No-RAG on long-context consistency."""
    stats = {
        "rag": {"consistency": [], "hallucination": [], "flaws": {}},
        "no_rag": {"consistency": [], "hallucination": [], "flaws": {}}
    }

    for r in results:
        if "error" in r: continue
        key = "rag" if r.get("rag_enabled") else "no_rag"
        eval_data = r["evaluation"]
        
        stats[key]["consistency"].append(eval_data["GlobalPersonalizationConsistency"]["score"])
        stats[key]["hallucination"].append(eval_data["AggregateHallucinationCheck"]["score"])
        
        for flaw in eval_data.get("Major_Flaws_Detected", []):
            if flaw != "None":
                stats[key]["flaws"][flaw] = stats[key]["flaws"].get(flaw, 0) + 1

    print("\n📈 AGGREGATE PERFORMANCE REPORT 📈")
    print("-----------------------------------")
    for key in ["rag", "no_rag"]:
        cons = stats[key]["consistency"]
        hal = stats[key]["hallucination"]
        count = len(cons)
        if count == 0: continue
        
        print(f"\nConfiguration: {key.upper()}")
        print(f"  Samples: {count}")
        print(f"  🧠 Global Consistency: {sum(cons)/count:.2f} / 5.0")
        print(f"  🛡️ Aggregate Safety:   {sum(hal)/count:.2f} / 5.0")
        
        print("  🚩 Common Flaws (Whole Dialogue):")
        sorted_flaws = sorted(stats[key]["flaws"].items(), key=lambda x: x[1], reverse=True)
        if not sorted_flaws:
            print("      - None detected!")
        for f, c in sorted_flaws[:3]:
            print(f"      - {f}: {c} dialogues")

# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input dialogues JSON file")
    parser.add_argument("--output", type=str, default="aggregate_eval.json", help="Output evaluation JSON file")
    parser.add_argument("--profiles", type=str, default=None, help="User profiles JSON file (optional)")
    args = parser.parse_args()
    batch_evaluate(args.input, args.output, args.profiles)