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

def generate_and_save_round_summary(
    user_name: str,
    round_number: int,
    user_query: str,
    round_messages: list,
    llm_config: dict
) -> bool:
    """
    Generate a summary for a round using LLM and save it.

    Args:
        user_name: User identifier
        round_number: Round number to summarize
        user_query: The user's query for this round
        round_messages: All messages from this round
        llm_config: LLM configuration for generating summary

    Returns:
        True if successful
    """
    try:
        # Extract agent outputs for summarization
        agent_contents = []
        for msg in round_messages:
            msg_name = msg.get("name", "")
            content = msg.get("content", "")

            # Filter by agent name instead of role (AutoGen may use role='user' for agent responses)
            if msg_name and content and any(agent_type in msg_name for agent_type in ["Dialogue_History", "Recommender", "Reasoner", "Reply"]):
                agent_name = msg_name.split("_")[0]  # Extract agent type
                agent_contents.append(f"{agent_name}: {content[:300]}")

        if not agent_contents:
            print(f"[SUMMARY] No agent outputs found for Round {round_number}")
            return False

        # Generate summary using LLM
        summary_prompt = (
            f"Summarize Round {round_number} of the conversation in 2-3 concise sentences.\n\n"
            f"User Query: {user_query}\n\n"
            f"Agent Outputs:\n" + "\n\n".join(agent_contents) + "\n\n"
            f"Summary (focus on: what user wanted, what was recommended, key preferences):"
        )

        # Use autogen's LLM client directly
        import autogen
        client = autogen.OpenAIWrapper(**llm_config)

        response = client.create(
            messages=[{"role": "user", "content": summary_prompt}]
        )

        # Extract summary from response
        summary = response.choices[0].message.content if response.choices else None

        if not summary:
            print(f"[SUMMARY ERROR] No summary generated for Round {round_number}")
            return False

        # Save summary
        from ragFunctions import save_round_summary
        success = save_round_summary(user_name, round_number, summary)

        if success:
            print(f"[SUMMARY] Generated and saved summary for Round {round_number}")

        return success

    except Exception as e:
        print(f"[SUMMARY ERROR] Failed to generate summary for Round {round_number}: {e}")
        return False