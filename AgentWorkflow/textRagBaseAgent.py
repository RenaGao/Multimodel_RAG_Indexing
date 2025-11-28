from dotenv import load_dotenv
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
from helpFunctions import load_all_questions, load_all_users, export_dialogues_for_eval
from datetime import datetime
from pathlib import Path
import json
import sys
import argparse
import re
from ragFunctions import save_chat_history, rag_search_for_recommend_agent, rag_search_for_dialogue_history_agent, append_agent_message, save_round_summary, load_previous_round_summaries, get_current_round_number, set_dataset_category, clear_user_storage
from agent_prompts import get_dialogue_history_prompt, get_recommender_prompt, get_reasoner_prompt, get_reply_prompt

# Custom logger class to capture all print outputs
class TeeLogger:
    """Captures print output to both console and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def get_all_question_files(questions_dir="../questions_v3"):
    """Get all question files from questions_v3 folder."""
    files = []
    if os.path.exists(questions_dir):
        for f in os.listdir(questions_dir):
            if f.endswith(".json") and "_questions_" in f:
                files.append(os.path.join(questions_dir, f))
    return sorted(files)

def extract_category(filename):
    """Extract category from filename."""
    basename = os.path.basename(filename)
    return basename.split("_questions_")[0]

def get_completed_users(category):
    """Get set of user IDs that have completed output files."""
    output_dir = Path(f"./dialogues_output/{category}")
    if not output_dir.exists():
        return set()

    completed = set()
    # Files are named: text_rag_dialogues_{user_id}_{version}_{date}_{time}.json
    for f in output_dir.glob("text_rag_dialogues_*.json"):
        stem = f.stem  # Remove .json
        parts = stem.split("_")

        # Find index of 'dialogues' and version (v1, v2, v3, ...)
        dialogues_idx = parts.index("dialogues") if "dialogues" in parts else -1

        version_idx = -1
        for i, p in enumerate(parts):
            if re.match(r'^v\d+$', p):  # Match v1, v2, v3, etc.
                version_idx = i
                break

        # Extract user_id between 'dialogues' and version
        if dialogues_idx >= 0 and version_idx > dialogues_idx + 1:
            user_id = "_".join(parts[dialogues_idx + 1 : version_idx])
            completed.add(user_id)

    return completed

def is_category_completed(questions_path, category):
    """Check if all users in questions file have output files."""
    users = set(load_all_users(questions_path))
    completed = get_completed_users(category)
    return users.issubset(completed)

def make_simple_flow(user_agent, dh_agent, rec_agent, rsn_agent, reply_agent):
    """
    Flow:
      User -> DialogueHistory (handles its tool call cycles)
           -> Recommender     (handles its tool call cycles)
           -> Reasoner        (handles its tool call cycles)
           -> Reply           (generates final user response)
           -> End
    """
    def _last(gc):
        return (gc.messages or [None])[-1] or {}

    def _has_tool_calls(msg: dict) -> bool:
        return isinstance(msg, dict) and bool(msg.get("tool_calls"))

    def _is_tool_return(msg: dict) -> bool:
        return isinstance(msg, dict) and msg.get("role") == "tool"

    def select_next(last_speaker, groupchat):
        last = _last(groupchat)

        if last_speaker is None and not groupchat.messages:
            return user_agent

        if _has_tool_calls(last) or _is_tool_return(last):
            return last_speaker

        if last_speaker is user_agent:
            return dh_agent

        if last_speaker is dh_agent:
            return rec_agent

        if last_speaker is rec_agent:
            return rsn_agent

        if last_speaker is rsn_agent:
            return reply_agent

        if last_speaker is reply_agent:
            return None

        return user_agent

    return select_next

def run_single_category(json_path, reply_prompt_version, category):
    """Run agent workflow for a single category."""
    # Set the dataset category for ragFunctions
    set_dataset_category(category)

    all_questions_dict = load_all_questions(json_path)
    all_users_list = load_all_users(json_path)

    # Filter out completed users for resume capability
    completed_users = get_completed_users(category)
    selected_users = [u for u in all_users_list if u not in completed_users]

    # how many questions can run for one category
    per_user_limit = None  # run all questions in the category

    # store output for json format
    results = {}

    print(f"Configuration:")
    print(f"  Category: {category}")
    print(f"  Reply Prompt Version: {reply_prompt_version}")
    print(f"  Total users: {len(all_users_list)}")
    print(f"  Completed users: {len(completed_users)}")
    print(f"  Remaining users: {len(selected_users)}")
    print(f"  Per user limit: {per_user_limit or 'All questions'}")
    print(f"  Question file: {json_path}")
    print()

    # Qwen3 VL 8B Instruct
    llm_config = {
        "config_list": [{
            "model": "qwen/qwen3-8b",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            "price": [0.001, 0.002],
            "timeout": 120,
        }],
        "timeout": 120,
        "cache_seed": None,
    }

    for user in selected_users:
        questions = all_questions_dict[user]
        if per_user_limit:
            questions = questions[:per_user_limit]

        dialogue_history_agent = AssistantAgent(
            name=f"Dialogue_History_{user}_agent",
            llm_config=llm_config,
            system_message=get_dialogue_history_prompt(user),
        )

        recommend_agent = AssistantAgent(
            name=f"Recommender_{user}_agent",
            llm_config=llm_config,
            system_message=get_recommender_prompt(user),
        )

        reason_agent = AssistantAgent(
            name=f"Reasoner_{user}_agent",
            llm_config=llm_config,
            system_message=get_reasoner_prompt(user),
        )

        reply_agent = AssistantAgent(
            name=f"Reply_{user}_agent",
            llm_config=llm_config,
            system_message=get_reply_prompt(version=reply_prompt_version),
        )

        # Create wrapper that captures user_id for reply_agent
        def make_reply_tool_wrapper(user_id: str):
            def wrapper():
                current_round = get_current_round_number(user_id)
                summaries = load_previous_round_summaries(user_id, current_round)
                if not summaries:
                    return {
                        "user_name": user_id,
                        "current_round": current_round,
                        "previous_summaries": [],
                        "message": "This is the first round. No previous conversation history exists."
                    }
                return {
                    "user_name": user_id,
                    "current_round": current_round,
                    "previous_summaries": summaries,
                    "message": f"Found {len(summaries)} previous round(s) of conversation."
                }
            return wrapper

        reply_wrapper = make_reply_tool_wrapper(user)

        reply_agent.register_for_llm(
            name="get_previous_round_summaries",
            description=(
                "Retrieve summaries of all previous rounds in this conversation session. "
                "Use this to determine if this is a continuation and to understand the conversation context. "
                "No parameters required (user_name and round_number are automatically provided)."
            ),
        )(reply_wrapper)

        reply_agent.register_for_execution(
            name="get_previous_round_summaries"
        )(reply_wrapper)

        user_proxy_agent = UserProxyAgent(
            name=f"User_{user}_proxy_agent",
            human_input_mode="NEVER",
            code_execution_config=False,
        )

        # Create wrapper for dialogue_history_agent
        def make_dh_tool_wrapper(user_id: str, llm_cfg: dict):
            def wrapper(current_query: str):
                return rag_search_for_dialogue_history_agent(
                    user_name=user_id,
                    current_query=current_query,
                    llm_config=llm_cfg
                )
            return wrapper

        dh_wrapper = make_dh_tool_wrapper(user, llm_config)

        dialogue_history_agent.register_for_llm(
            name="rag_search_for_dialogue_history_agent",
            description=(
                "Retrieve relevant user dialogue history and preference records from the vector database. "
                "Use it to infer user interests, habits, or preference. "
                "Only requires 'current_query' parameter (user_name is automatically provided)."
            ),
        )(dh_wrapper)

        dialogue_history_agent.register_for_execution(
            name="rag_search_for_dialogue_history_agent"
        )(dh_wrapper)

        # Create wrapper for recommend_agent
        def make_rec_tool_wrapper(user_id: str):
            def wrapper(user_query: str):
                return rag_search_for_recommend_agent(
                    user_name=user_id,
                    user_query=user_query
                )
            return wrapper

        rec_wrapper = make_rec_tool_wrapper(user)

        recommend_agent.register_for_llm(
            name="rag_search_for_recommend_agent",
            description=(
                "Search the vector database for product, item, or knowledge information relevant to the user's query. "
                "Use it to generate personalized shopping recommendations. "
                "Only requires 'user_query' parameter (user_name is automatically provided)."
            ),
        )(rec_wrapper)

        recommend_agent.register_for_execution(
            name="rag_search_for_recommend_agent"
        )(rec_wrapper)

        results[user] = []
        user_history = []

        print(f"\n{'='*80}")
        print(f"Processing User: {user}")
        print(f"Total Questions: {len(questions)}")
        print(f"{'='*80}\n")

        for i, q in enumerate(questions, start=1):
            print(f"\n{'-'*80}")
            print(f"[ROUND {i}/{len(questions)}] Question: {q}")
            print(f"{'-'*80}\n")

            groupchat = GroupChat(
                agents=[user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent, reply_agent],
                messages=[],
                max_round=10,
                speaker_selection_method=make_simple_flow(user_proxy_agent, dialogue_history_agent, recommend_agent, reason_agent, reply_agent)
            )

            manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

            save_chat_history(f"[USER_QUESTION_{i}] [{user_proxy_agent.name}] {q}", user)
            user_proxy_agent.initiate_chat(manager, message=q)

            new_messages = groupchat.messages

            for idx, msg in enumerate(new_messages):
                if not isinstance(msg, dict):
                    continue
                msg_name = msg.get("name", "")
                msg_role = msg.get("role", "")
                msg_content = msg.get("content", "")

                if msg_role in ("tool", "function"):
                    continue
                if not msg_content or str(msg_content).strip() in ("", "None"):
                    continue
                if msg_name == user_proxy_agent.name:
                    continue

                if msg_name in [dialogue_history_agent.name, recommend_agent.name, reason_agent.name, reply_agent.name]:
                    append_agent_message(msg_content, "assistant", msg_name, user)

            user_history.extend(new_messages)
            print(f"\n[ROUND {i} COMPLETED]\n")

        results[user] = user_history

        print(f"\n{'='*80}")
        print(f"Completed User: {user}")
        print(f"Total Rounds: {len(questions)}")
        print(f"{'='*80}\n")

        # Export dialogues for evaluation
        eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = Path(f"./dialogues_output/{category}")
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_output_path = eval_dir / f"text_rag_dialogues_{user}_{reply_prompt_version}_{eval_timestamp}.json"

        # Check if export succeeded
        export_result = export_dialogues_for_eval(user, output_path=str(eval_output_path))
        if export_result is None:
            print(f"[WARNING] Failed to export dialogues for user {user}")
        else:
            print(f"[SUCCESS] Exported dialogues to: {export_result}")

        # Clear user storage after each user completes
        clear_user_storage()

    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Text RAG Base Agent for Shopping Recommendations")
    parser.add_argument(
        "--reply-prompt-version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Version of reply agent prompt to use (default: v1, v2: adaptive/flexible)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all categories from questions_v3/ folder"
    )
    parser.add_argument(
        "--questions-dir",
        type=str,
        default="../questions_v3",
        help="Directory containing question files (default: ../questions_v3)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a specific questions JSON file (for single category mode)"
    )
    args = parser.parse_args()
    reply_prompt_version = args.reply_prompt_version

    load_dotenv()
    assert os.getenv("OPENROUTER_API_KEY"), "Missing QWEN_API_KEY in .env"

    # Clear user storage at script start to avoid cross-category conflicts
    clear_user_storage()

    # Setup logging
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"agent_run_{ts}.log"

    # Redirect stdout to both console and log file
    logger = TeeLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger  # Also capture error messages

    print(f"{'='*80}")
    print(f"Agent Workflow Execution Started")
    print(f"Timestamp: {ts}")
    print(f"Log file: {log_file}")
    print(f"{'='*80}\n")

    # Test API connection
    print("\n[TEST] Testing OpenRouter API connection...")
    try:
        import requests
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen/qwen3-8b",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            },
            timeout=30
        )
        print(f"[TEST] API responded with status: {response.status_code}")
        if response.status_code != 200:
            print(f"[TEST] API Error: {response.text}")
    except Exception as e:
        print(f"[TEST] API Test Failed: {e}")
    print()

    # Run based on mode
    if args.all:
        # Batch mode: process all categories from questions_v3/
        question_files = get_all_question_files(args.questions_dir)
        if not question_files:
            print(f"No question files found in {args.questions_dir}")
            sys.exit(1)

        print(f"Found {len(question_files)} categories to process:")
        for qf in question_files:
            print(f"  - {extract_category(qf)}: {qf}")
        print()

        for qf in question_files:
            category = extract_category(qf)

            # Check if category is already completed
            if is_category_completed(qf, category):
                print(f"Skipping {category} (already completed)")
                continue

            print(f"\n{'#'*80}")
            print(f"# Processing Category: {category}")
            print(f"# Question file: {qf}")
            print(f"{'#'*80}\n")

            try:
                run_single_category(qf, reply_prompt_version, category)
            except Exception as e:
                print(f"Error processing category {category}: {e}")
                continue

        print(f"\n{'='*80}")
        print(f"All Categories Completed")
        print(f"Output saved to: ./dialogues_output/{{category}}/")
        print(f"Log saved to: {log_file}")
        print(f"{'='*80}\n")

    else:
        # Single category mode
        if args.input:
            json_path = args.input
        else:
            # Default path for backward compatibility
            json_path = "../questions_v3/Clothing_Shoes_and_Jewelry_questions_20251127_235753.json"
            if not os.path.exists(json_path):
                # Try to find any question file in questions_v3
                question_files = get_all_question_files(args.questions_dir)
                if question_files:
                    json_path = question_files[0]
                else:
                    print(f"No question files found. Please specify --input or use --all mode.")
                    sys.exit(1)

        category = extract_category(json_path)
        print(f"Single category mode: {category}")
        print(f"Question file: {json_path}")
        print()

        run_single_category(json_path, reply_prompt_version, category)

        print(f"\n{'='*80}")
        print(f"Agent Workflow Execution Completed")
        print(f"Output saved to: ./dialogues_output/{category}/")
        print(f"Log saved to: {log_file}")
        print(f"{'='*80}\n")

    # Restore stdout and close log file
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    logger.close()

    print(f"Execution complete. Check {log_file} for detailed logs.")

if __name__ == "__main__":
    main()