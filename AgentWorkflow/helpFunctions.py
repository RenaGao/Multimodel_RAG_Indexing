from pathlib import Path
import json
from typing import Dict, List
from datetime import datetime

def load_all_questions(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_questions: Dict[str, List[str]] = {}
    for rec in data:
        uid = rec.get("user_id")
        if not uid:
            continue
        qs = rec.get("generated_questions") or []
        qs = [str(q) for q in qs if q]
        all_questions[uid] = qs

    return all_questions


def load_all_users(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    users: List[str] = []
    for rec in data:
        uid = rec.get("user_id")
        if uid:
            users.append(uid)
    return users

def load_all_users_and_questions(json_path: str) -> tuple[dict, list]:
    """Return result dict and users list"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    users = []

    for user in data:
        user_id = user.get("user_id")
        questions = user.get("generated_questions", [])
        result[user_id] = questions
        users.append(user_id)

    return result, users

def export_dialogues_for_eval(user_name: str, output_path: str = None) -> str:
    """
    Export user-reply dialogues for evaluation.

    Reads chat_history.jsonl and extracts each round's user query and reply agent response
    in the format expected by Eval_Agents_Responses.py.

    Args:
        user_name: User identifier
        output_path: Optional custom output path. If None, auto-generates filename.

    Returns:
        Path to the exported JSON file
    """
    from ragFunctions import _user_paths, _iter_jsonl

    try:
        up = _user_paths(user_name)

        if not up["chat_history"].exists():
            print(f"[EXPORT] No chat history found for user {user_name}")
            return None

        # Read all rounds from chat_history
        all_rounds = list(_iter_jsonl(up["chat_history"]))

        if not all_rounds:
            print(f"[EXPORT] No rounds found in chat history for user {user_name}")
            return None

        # Build dialogue entries for evaluation
        dialogues = []

        for round_data in all_rounds:
            round_num = round_data.get("round_number")
            user_query = round_data.get("user_query", "")

            # Find reply agent's response
            reply_content = None
            for msg in round_data.get("conversation", []):
                agent_name = msg.get("agent", "")
                if "Reply" in agent_name and msg.get("role") == "assistant":
                    reply_content = msg.get("content", "")
                    break

            if not reply_content:
                print(f"[EXPORT] Warning: No reply found for Round {round_num}, skipping")
                continue

            # Format as dialogue entry
            dialogue_entry = {
                "dialogue_id": f"{user_name}_round_{round_num}",
                "user_goal": user_query,
                "dialogue_history": [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": reply_content}
                ]
            }

            dialogues.append(dialogue_entry)

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"eval_dialogues_{user_name}_{timestamp}.json"

        # Save to JSON file
        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(dialogues, f, ensure_ascii=False, indent=2)

        print(f"[EXPORT] Successfully exported {len(dialogues)} dialogues to: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"[EXPORT ERROR] Failed to export dialogues: {e}")
        return None