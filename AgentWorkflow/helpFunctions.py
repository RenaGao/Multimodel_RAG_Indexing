import json

def load_all_questions(json_path: str) -> dict:
    """Get all question from json file"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def load_all_users(json_path: str) -> list:
    """Get all users from json file"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    users = list(data.keys())

    assert users
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