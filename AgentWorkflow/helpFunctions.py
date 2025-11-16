from pathlib import Path
import json
from typing import Dict, List

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